"""
StromCast FastAPI application.

Exposes health, actuals, forecast, training, and metrics endpoints under
``/api/v1``. Run standalone with::

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__, config
from .. import pipeline
from . import schemas

app = FastAPI(
    title="StromCast API",
    description="Industrial German grid electricity-demand forecasting service.",
    version=__version__,
)

# Permissive CORS so the Streamlit dashboard (different port) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_up() -> None:
    # Generate data and train/load the model so the first request is fast.
    pipeline.ensure_ready()


# ──────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────
@app.get(f"{config.API_PREFIX}/health", response_model=schemas.HealthResponse, tags=["meta"])
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(
        version=__version__,
        model_trained=config.MODEL_PATH.exists(),
    )


# ──────────────────────────────────────────────────────────────────────────
# Actuals
# ──────────────────────────────────────────────────────────────────────────
@app.get(f"{config.API_PREFIX}/actuals", response_model=schemas.ActualsResponse, tags=["data"])
def actuals(
    hours: int = Query(168, ge=1, le=24 * 90, description="Hours of recent history"),
) -> schemas.ActualsResponse:
    df = pipeline.get_actuals(hours)
    points = [
        schemas.ActualPoint(
            timestamp=ts.isoformat(),
            demand_mw=float(row["demand_mw"]),
            temp_c=float(row["temp_c"]),
            wind_cf=float(row["wind_cf"]),
            solar_cf=float(row["solar_cf"]),
            price_eur=float(row["price_eur"]),
            is_holiday=int(row["is_holiday"]),
        )
        for ts, row in df.iterrows()
    ]
    return schemas.ActualsResponse(count=len(points), points=points)


# ──────────────────────────────────────────────────────────────────────────
# Forecast
# ──────────────────────────────────────────────────────────────────────────
@app.get(f"{config.API_PREFIX}/forecast", response_model=schemas.ForecastResponse, tags=["forecast"])
def forecast(
    horizon: int = Query(config.DEFAULT_HORIZON, ge=1, le=config.MAX_HORIZON),
    temp_delta_c: float = Query(0.0, ge=-30, le=30),
    solar_factor: float = Query(1.0, ge=0.0, le=3.0),
    wind_factor: float = Query(1.0, ge=0.0, le=3.0),
    price_factor: float = Query(1.0, ge=0.0, le=5.0),
    matched_hours: int = Query(168, ge=0, le=24 * 30),
) -> schemas.ForecastResponse:
    sim = {
        "temp_delta_c": temp_delta_c,
        "solar_factor": solar_factor,
        "wind_factor": wind_factor,
        "price_factor": price_factor,
    }
    result = pipeline.run_forecast(
        horizon=horizon,
        sim=sim,
        include_matched=matched_hours > 0,
        matched_hours=matched_hours,
    )
    return schemas.ForecastResponse(**result)


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
@app.get(f"{config.API_PREFIX}/metrics", response_model=schemas.MetricsResponse, tags=["model"])
def metrics() -> schemas.MetricsResponse:
    return schemas.MetricsResponse(**pipeline.get_metrics())


# ──────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────
@app.post(f"{config.API_PREFIX}/train", response_model=schemas.TrainResponse, tags=["model"])
def train(force_data: bool = Query(False, description="Regenerate the synthetic history")):
    metadata = pipeline.train(force_data=force_data)
    pipeline.reset_cache()
    pipeline.ensure_ready()
    return schemas.TrainResponse(metadata=schemas.MetricsResponse(**metadata))


@app.get("/", include_in_schema=False)
def root():
    return {"service": "stromcast-api", "docs": "/docs", "health": f"{config.API_PREFIX}/health"}
