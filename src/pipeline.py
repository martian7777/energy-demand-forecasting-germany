"""
StromCast orchestration pipeline.

Ties together data generation, feature engineering, model training, metric
tracking, and forecasting. The API layer calls into these functions; nothing
here imports FastAPI or Streamlit so the pipeline stays headless-testable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config
from . import data_generator as dg
from .model import StromCastModel


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def train(force_data: bool = False) -> dict:
    """
    Run the full training pipeline: load/generate data, fit the model,
    compute KPIs, persist the model and a metrics/metadata sidecar.

    Returns the metadata dict that is also written to ``METRICS_PATH``.
    """
    df = dg.load_or_generate(force=force_data)

    model = StromCastModel()
    metrics = model.fit(df)
    model.save()

    top_importance = (
        model.feature_importance()
        .head(15)
        .to_dict(orient="records")
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "history_start": df.index.min().isoformat(),
        "history_end": df.index.max().isoformat(),
        "residual_std": model.residual_std,
        "metrics": metrics,
        "feature_importance": top_importance,
        "xgb_params": model.params,
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


# ──────────────────────────────────────────────────────────────────────────
# Lazy singletons (so the API serves quickly after first load)
# ──────────────────────────────────────────────────────────────────────────
_MODEL: StromCastModel | None = None
_DATA: pd.DataFrame | None = None


def ensure_ready() -> None:
    """Make sure data exists and a trained model is loaded; train if needed."""
    global _MODEL, _DATA
    if _DATA is None:
        _DATA = dg.load_or_generate()
    if _MODEL is None:
        if not config.MODEL_PATH.exists():
            train()
        _MODEL = StromCastModel.load()


def get_model() -> StromCastModel:
    ensure_ready()
    assert _MODEL is not None
    return _MODEL


def get_data() -> pd.DataFrame:
    ensure_ready()
    assert _DATA is not None
    return _DATA


def reset_cache() -> None:
    """Force a reload of the model and data on the next access (post-retrain)."""
    global _MODEL, _DATA
    _MODEL = None
    _DATA = None


# ──────────────────────────────────────────────────────────────────────────
# Read APIs
# ──────────────────────────────────────────────────────────────────────────
def get_actuals(hours: int = 168) -> pd.DataFrame:
    """Most recent ``hours`` of observed history (demand + exog)."""
    data = get_data()
    return data.tail(max(1, hours)).copy()


def get_metrics() -> dict:
    """Load the persisted metrics/metadata sidecar, or train if absent."""
    if not config.METRICS_PATH.exists():
        return train()
    with open(config.METRICS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────
# Forecasting
# ──────────────────────────────────────────────────────────────────────────
def run_forecast(
    horizon: int = config.DEFAULT_HORIZON,
    sim: dict | None = None,
    include_matched: bool = True,
    matched_hours: int = 168,
) -> dict:
    """
    Generate a recursive out-of-sample forecast with optional what-if
    simulation overrides.

    Returns a dict with:
      * ``forecast``    : list of {timestamp, forecast_mw, lower_mw, upper_mw,
                          temp_c, wind_cf, solar_cf, price_eur}
      * ``matched``     : recent actuals with one-step predictions (for charts)
      * ``horizon``     : echoed horizon
      * ``generated_at``: ISO timestamp
    """
    horizon = int(max(1, min(horizon, config.MAX_HORIZON)))
    model = get_model()
    data = get_data()

    future_exog = dg.make_future_exog(data, horizon, sim=sim)
    fc = model.forecast(data, future_exog)

    fc_out = fc.join(future_exog[["temp_c", "wind_cf", "solar_cf", "price_eur"]])
    fc_records = [
        {
            "timestamp": ts.isoformat(),
            "forecast_mw": float(row.forecast_mw),
            "lower_mw": float(row.lower_mw),
            "upper_mw": float(row.upper_mw),
            "temp_c": float(row.temp_c),
            "wind_cf": float(row.wind_cf),
            "solar_cf": float(row.solar_cf),
            "price_eur": float(row.price_eur),
        }
        for ts, row in fc_out.iterrows()
    ]

    matched_records = []
    if include_matched:
        recent = data.tail(matched_hours + max(config.LAG_HOURS) + max(config.ROLLING_WINDOWS))
        preds = model.predict_matched(recent)
        joined = data.tail(matched_hours).join(preds.rename("prediction"))
        for ts, row in joined.iterrows():
            matched_records.append(
                {
                    "timestamp": ts.isoformat(),
                    "actual_mw": float(row["demand_mw"]),
                    "prediction_mw": (
                        float(row["prediction"]) if pd.notna(row.get("prediction")) else None
                    ),
                    "temp_c": float(row["temp_c"]),
                    "price_eur": float(row["price_eur"]),
                }
            )

    return {
        "horizon": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulation": sim or {},
        "forecast": fc_records,
        "matched": matched_records,
    }


if __name__ == "__main__":
    meta = train(force_data=True)
    print(json.dumps(meta["metrics"], indent=2))
    out = run_forecast(horizon=24)
    print("Forecast points:", len(out["forecast"]))
    print("First:", out["forecast"][0])
