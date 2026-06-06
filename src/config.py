"""
StromCast central configuration.

Holds all directory paths, simulation constants, feature-engineering
parameters, and XGBoost hyperparameters in a single place so the data
generator, model, pipeline, API, and dashboard stay in lockstep.
"""
from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for _d in (DATA_DIR, MODEL_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "grid_history.parquet"
MODEL_PATH = MODEL_DIR / "xgb_demand.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# ──────────────────────────────────────────────────────────────────────────
# Simulation window
# ──────────────────────────────────────────────────────────────────────────
# Generate a rolling window ending "today" so the dashboard always has a
# recent context. SIM_YEARS of hourly history (>= 2 years requested).
SIM_YEARS = 3
FREQ = "h"  # hourly resolution
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────
# German grid demand characteristics (megawatts, MW)
# Rough order of magnitude of the German national load (~40–80 GW).
# ──────────────────────────────────────────────────────────────────────────
BASE_LOAD_MW = 56_000.0        # mean national load
DAILY_AMPLITUDE_MW = 11_000.0  # morning/evening peak swing
WEEKEND_REDUCTION_MW = 7_000.0 # weekend demand drop
SEASONAL_AMPLITUDE_MW = 6_500.0  # winter heating vs summer
HOLIDAY_REDUCTION_MW = 9_000.0   # public-holiday demand drop
NOISE_STD_MW = 900.0           # stochastic measurement / behavior noise

# ──────────────────────────────────────────────────────────────────────────
# Weather characteristics (°C)
# ──────────────────────────────────────────────────────────────────────────
TEMP_ANNUAL_MEAN_C = 9.5       # German annual mean temperature
TEMP_SEASONAL_AMPLITUDE_C = 10.0
TEMP_DAILY_AMPLITUDE_C = 4.5
TEMP_NOISE_STD_C = 1.6
# Comfortable temperature where heating/cooling demand is minimal.
TEMP_COMFORT_C = 16.0
HEATING_COEF_MW_PER_C = 520.0  # extra MW per °C below comfort
COOLING_COEF_MW_PER_C = 380.0  # extra MW per °C above comfort

# ──────────────────────────────────────────────────────────────────────────
# Renewable indices (capacity factors, 0..1)
# ──────────────────────────────────────────────────────────────────────────
WIND_MEAN_CF = 0.24
SOLAR_PEAK_CF = 0.78           # midday clear-sky summer peak

# ──────────────────────────────────────────────────────────────────────────
# EPEX Spot day-ahead price (EUR/MWh)
# ──────────────────────────────────────────────────────────────────────────
PRICE_BASE_EUR = 85.0
PRICE_LOAD_COEF = 0.0016       # EUR per MW of residual load
PRICE_RENEWABLE_COEF = -70.0   # merit-order effect of renewables
PRICE_NOISE_STD = 9.0
PRICE_FLOOR_EUR = -50.0        # negative prices are allowed on EPEX
PRICE_CAP_EUR = 600.0

# ──────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────
TARGET = "demand_mw"
LAG_HOURS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [3, 24, 168]
EXOG_FEATURES = ["temp_c", "wind_cf", "solar_cf", "price_eur"]

# ──────────────────────────────────────────────────────────────────────────
# Forecast horizons
# ──────────────────────────────────────────────────────────────────────────
HORIZON_24H = 24
HORIZON_7D = 24 * 7
DEFAULT_HORIZON = HORIZON_24H
MAX_HORIZON = HORIZON_7D

# ──────────────────────────────────────────────────────────────────────────
# XGBoost hyperparameters
# ──────────────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "reg_lambda": 1.5,
    "reg_alpha": 0.2,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# Fraction of the most recent data held out for validation metrics.
VALIDATION_FRACTION = 0.1

# ──────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("STROMCAST_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("STROMCAST_API_PORT", "8000"))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
API_PREFIX = "/api/v1"

# Dashboard
DASHBOARD_PORT = int(os.getenv("STROMCAST_DASHBOARD_PORT", "8501"))
