"""
Feature engineering and the XGBoost forecasting model for StromCast.

Design notes
------------
* All lag and rolling features are computed on the *shifted* demand series so
  they only ever reference the past. That keeps training leakage-free and lets
  the multi-step forecaster build a feature row for time ``t`` using only
  values known at ``t-1``.
* Forecasting is recursive: predicted demand is fed back in as the lag input
  for the following step. Future exogenous drivers (temperature, wind, solar,
  price) must be supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from . import config


# ──────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``df`` (DatetimeIndex + at least the exog columns and,
    for training, the target) enriched with calendar, lag, rolling, and
    weather-interaction features. No rows are dropped here.
    """
    out = df.copy()
    idx = out.index

    # ── Calendar / time features ───────────────────────────────────────────
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    if "is_holiday" not in out.columns:
        out["is_holiday"] = 0

    # Cyclical encodings so the tree sees smooth wrap-around boundaries.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)

    # ── Weather interactions ───────────────────────────────────────────────
    out["heating_degree"] = np.maximum(0.0, config.TEMP_COMFORT_C - out["temp_c"])
    out["cooling_degree"] = np.maximum(0.0, out["temp_c"] - config.TEMP_COMFORT_C)
    out["renewable_index"] = 0.6 * out["wind_cf"] + 0.4 * out["solar_cf"]

    # ── Lag & rolling features on the target ───────────────────────────────
    if config.TARGET in out.columns:
        target = out[config.TARGET]
        for lag in config.LAG_HOURS:
            out[f"lag_{lag}h"] = target.shift(lag)
        shifted = target.shift(1)  # exclude the current step → no leakage
        for win in config.ROLLING_WINDOWS:
            out[f"roll_mean_{win}h"] = shifted.rolling(win).mean()
            out[f"roll_std_{win}h"] = shifted.rolling(win).std()

    return out


def feature_columns() -> list[str]:
    """The ordered list of model input columns produced by ``add_features``."""
    cols = [
        "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "temp_c", "wind_cf", "solar_cf", "price_eur",
        "heating_degree", "cooling_degree", "renewable_index",
    ]
    cols += [f"lag_{lag}h" for lag in config.LAG_HOURS]
    for win in config.ROLLING_WINDOWS:
        cols += [f"roll_mean_{win}h", f"roll_std_{win}h"]
    return cols


# ──────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────
class StromCastModel:
    """Thin, persistence-aware wrapper around an XGBoost regressor."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(config.XGB_PARAMS)
        self.model: XGBRegressor | None = None
        self.features: list[str] = feature_columns()
        self.residual_std: float = 0.0  # for forecast confidence bands

    # ── Training ───────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train on a history DataFrame. The most recent ``VALIDATION_FRACTION``
        of rows are held out for honest out-of-sample metrics.

        Returns a metrics dict (rmse, mae, mape, r2, n_train, n_val).
        """
        feat = add_features(df).dropna(subset=self.features + [config.TARGET])
        X = feat[self.features]
        y = feat[config.TARGET]

        n_val = max(1, int(len(feat) * config.VALIDATION_FRACTION))
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = self.model.predict(X_val)
        residuals = y_val.to_numpy() - val_pred
        self.residual_std = float(np.std(residuals))

        metrics = regression_metrics(y_val.to_numpy(), val_pred)
        metrics.update({"n_train": int(len(X_train)), "n_val": int(len(X_val))})
        return metrics

    # ── In-sample (matched) predictions ──────────────────────────────────
    def predict_matched(self, df: pd.DataFrame) -> pd.Series:
        """One-step predictions aligned to the historical index (for charts)."""
        self._require_model()
        feat = add_features(df).dropna(subset=self.features)
        preds = self.model.predict(feat[self.features])
        return pd.Series(preds, index=feat.index, name="prediction")

    # ── Recursive multi-step forecast ─────────────────────────────────────
    def forecast(self, history: pd.DataFrame, future_exog: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a recursive forecast for the timestamps in ``future_exog``.

        Parameters
        ----------
        history : DataFrame
            Recent observed history including ``demand_mw`` and exog columns.
            Only the tail is needed but more context is harmless.
        future_exog : DataFrame
            Indexed by future timestamps; must contain temp_c, wind_cf,
            solar_cf, price_eur and (optionally) is_holiday.

        Returns
        -------
        DataFrame indexed by the future timestamps with columns
        ``forecast_mw``, ``lower_mw``, ``upper_mw``.
        """
        self._require_model()
        max_lookback = max(config.LAG_HOURS + config.ROLLING_WINDOWS) + 1

        # Working frame: tail of history followed by the future rows.
        hist = history.tail(max_lookback + 24).copy()
        future = future_exog.copy()
        if "is_holiday" not in future.columns:
            future["is_holiday"] = 0
        future[config.TARGET] = np.nan

        work = pd.concat([hist, future.reindex(columns=hist.columns)])

        future_index = future.index
        predictions = []

        for ts in future_index:
            # Build features over a tail window ending at the current step.
            window = work.loc[:ts].tail(max_lookback + 1)
            feat_row = add_features(window).loc[[ts], self.features]
            yhat = float(self.model.predict(feat_row)[0])
            work.loc[ts, config.TARGET] = yhat
            predictions.append(yhat)

        preds = np.array(predictions)
        # Confidence band widens with the square root of the horizon as errors
        # compound through the recursion.
        steps = np.arange(1, len(preds) + 1)
        band = 1.96 * self.residual_std * np.sqrt(np.minimum(steps, 24) / 1.0) ** 0.5
        return pd.DataFrame(
            {
                "forecast_mw": preds,
                "lower_mw": preds - band,
                "upper_mw": preds + band,
            },
            index=future_index,
        )

    # ── Introspection ─────────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        self._require_model()
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.features, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path=config.MODEL_PATH) -> None:
        self._require_model()
        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "residual_std": self.residual_std,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path=config.MODEL_PATH) -> "StromCastModel":
        blob = joblib.load(path)
        obj = cls(params=blob.get("params"))
        obj.model = blob["model"]
        obj.features = blob["features"]
        obj.residual_std = blob.get("residual_std", 0.0)
        return obj

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded yet.")


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / np.where(y_true == 0, np.nan, y_true))) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
