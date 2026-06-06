"""Pydantic request/response schemas for the StromCast API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "stromcast-api"
    version: str
    model_trained: bool


# ──────────────────────────────────────────────────────────────────────────
# Actuals
# ──────────────────────────────────────────────────────────────────────────
class ActualPoint(BaseModel):
    timestamp: str
    demand_mw: float
    temp_c: float
    wind_cf: float
    solar_cf: float
    price_eur: float
    is_holiday: int


class ActualsResponse(BaseModel):
    count: int
    points: list[ActualPoint]


# ──────────────────────────────────────────────────────────────────────────
# Forecast
# ──────────────────────────────────────────────────────────────────────────
class SimulationParams(BaseModel):
    """What-if overrides applied to the future exogenous drivers."""
    temp_delta_c: float = Field(0.0, ge=-30, le=30,
                                description="°C added to every future hour")
    solar_factor: float = Field(1.0, ge=0.0, le=3.0,
                                description="Solar capacity-factor multiplier (0 = full cloud)")
    wind_factor: float = Field(1.0, ge=0.0, le=3.0,
                               description="Wind capacity-factor multiplier")
    price_factor: float = Field(1.0, ge=0.0, le=5.0,
                                description="EPEX price multiplier")


class ForecastPoint(BaseModel):
    timestamp: str
    forecast_mw: float
    lower_mw: float
    upper_mw: float
    temp_c: float
    wind_cf: float
    solar_cf: float
    price_eur: float


class MatchedPoint(BaseModel):
    timestamp: str
    actual_mw: float
    prediction_mw: Optional[float] = None
    temp_c: float
    price_eur: float


class ForecastResponse(BaseModel):
    horizon: int
    generated_at: str
    simulation: dict
    forecast: list[ForecastPoint]
    matched: list[MatchedPoint]


# ──────────────────────────────────────────────────────────────────────────
# Metrics / Training
# ──────────────────────────────────────────────────────────────────────────
class Metrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    n_train: Optional[int] = None
    n_val: Optional[int] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class MetricsResponse(BaseModel):
    trained_at: str
    n_rows: int
    history_start: str
    history_end: str
    residual_std: float
    metrics: Metrics
    feature_importance: list[FeatureImportance]


class TrainResponse(BaseModel):
    status: str = "trained"
    metadata: MetricsResponse
