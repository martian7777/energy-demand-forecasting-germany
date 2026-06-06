# ⚡ StromCast — German Grid Demand Forecasting Platform

An end-to-end, fully-local electricity-demand forecasting and analytics
platform tailored for the German *Energiewende* market. It simulates 3 years
of realistic hourly German grid load, weather, renewable capacity factors,
and EPEX Spot day-ahead prices; trains an **XGBoost** time-series model;
serves forecasts over a **FastAPI** REST API; and ships a premium
**Streamlit + Plotly** dashboard with real-time what-if simulation.

No paid API keys required — everything runs on synthetic-but-realistic data.

## Architecture

```
data_generator ─▶ pipeline ─▶ model (XGBoost) ─▶ FastAPI ─▶ Streamlit dashboard
   (sim data)      (ETL +        (features,        (REST)      (KPIs, charts,
                    metrics)      recursive fc)                 what-if sliders)
```

## Quick start

```powershell
# 1. Install dependencies (ideally in a virtualenv)
pip install -r requirements.txt

# 2. Launch everything (trains on first run, ~30–90s)
python run.py

# Dashboard → http://localhost:8501
# API docs  → http://localhost:8000/docs
```

Useful flags: `python run.py --train` (force retrain), `--api-only`,
`--dashboard-only`.

## REST API (`/api/v1`)

| Method | Path        | Description |
|--------|-------------|-------------|
| GET    | `/health`   | Service + model status |
| GET    | `/actuals`  | Recent observed demand & exogenous variables |
| GET    | `/forecast` | 24h / 7d recursive forecast with what-if params |
| GET    | `/metrics`  | Model KPIs (RMSE/MAE/MAPE/R²) + feature importance |
| POST   | `/train`    | Retrain the model and refresh metrics |

What-if query params on `/forecast`: `temp_delta_c`, `solar_factor`,
`wind_factor`, `price_factor`, plus `horizon` and `matched_hours`.

## Layout

```
requirements.txt   run.py (launcher)   README.md
src/
  config.py            simulation/model/API constants
  data_generator.py    synthetic load, weather, renewables, EPEX price
  model.py             feature engineering + XGBoost wrapper (recursive fc)
  pipeline.py          training loop, metrics, forecast orchestration
  api/main.py          FastAPI endpoints
  api/schemas.py       Pydantic schemas
dashboard/app.py       Streamlit UI
```

Generated artifacts live in `data/`, `models/`, and `artifacts/`.
