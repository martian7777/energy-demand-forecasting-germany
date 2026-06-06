# ⚡ StromCast — German Grid Demand Forecasting Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg?style=flat-square)](https://xgboost.readthedocs.io/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

StromCast is an industrial-grade, end-to-end electricity demand forecasting and market analytics platform tailored for the German energy transition (*Energiewende*). 

The platform simulates three years of hourly grid data (load, temperature, renewables, and EPEX Spot prices), trains a high-performance **XGBoost** recursive time-series forecasting model, exposes predictions via a **FastAPI** REST API, and visualizes them inside a premium **Streamlit + Plotly** dashboard featuring real-time what-if scenario testing.

---

## 🎨 Dashboard Preview

Below is a showcase of the StromCast dashboard in action:

````carousel
![24-Hour Grid Demand Forecast](docs/images/forecast_24h.png)
<!-- slide -->
![7-Day Grid Demand Forecast](docs/images/forecast_7d.png)
<!-- slide -->
![Price vs Demand Dashboard](docs/images/price_vs_demand.png)
<!-- slide -->
![Model Health & Feature Importance](docs/images/model_health.png)
````

---

## ✨ Core Features

* **🔌 High-Fidelity Local Simulator:** Deterministic, seed-based generator modeling German national grid patterns: morning/evening peaks, weekend drops, holiday troughs, weather-dependent heating/cooling degree days, and wind/solar capacity factors.
* **🧠 Recursive XGBoost Forecaster:** A time-series model engineered with historical lags, rolling averages, and cyclical sine/cosine calendar encodings, capable of multi-step recursive forecasting with compounding confidence intervals.
* **🌦️ What-If Scenarios:** Sliders to simulate cold snaps, heatwaves, or renewable surges on the fly, instantly recalculating demand curves and day-ahead wholesale price drops (the *merit-order effect*).
* **⚡ FastAPI REST Backend:** High-performance endpoints to fetch recent actuals, calculate dynamic forecasts, expose validation metrics, and trigger automated model retraining.
* **📊 Premium Analytical UI:** Dark-themed dashboard with custom CSS, containing heroic KPI cards, price-demand correlation scatter plots, and model health residual distributions.

---

## 🏗️ System Architecture

StromCast uses a decoupled four-tier architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  DATA GENERATOR │ ──▶ │    PIPELINE     │ ──▶ │   FASTAPI API   │ ──▶ │    STREAMLIT    │
│  (Physics Sim)  │     │  (ETL/XGBoost)  │     │   (Service)     │     │   (Dashboard)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       ▲                       ▲
         ▼                       ▼                       │                       │
 [grid_history.pq]        [xgb_demand.job] ──────────────┴───────────────────────┘
 (Raw Parquet Data)      [metrics.json] (Cached Model & Performance KPIs)
```

1. **Simulation Layer:** `src/data_generator.py` generates the historical base load and exogenous weather inputs.
2. **Modeling Layer:** `src/model.py` runs feature engineering, evaluates residuals, and handles multi-step recursive forecasting.
3. **Serving Layer:** `src/api/main.py` exposes the ML model and data over high-speed REST endpoints.
4. **Presentation Layer:** `dashboard/app.py` renders charts and handles what-if inputs, communicating with the API.

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
First, clone the repository and install the required libraries (virtual environment recommended):
```powershell
pip install -r requirements.txt
```

### 2. Launch the Application
Run the master runner script. This will compile the dataset, train the XGBoost model on the first run (~30–90 seconds), and start both the backend and frontend services:
```powershell
python run.py
```

* **Dashboard URL:** [http://localhost:8501](http://localhost:8501)
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Useful Flags
* `python run.py --train` : Forces regeneration of the dataset and retrains the model before starting.
* `python run.py --api-only` : Launches only the FastAPI backend server.
* `python run.py --dashboard-only` : Launches only the Streamlit dashboard app (requires a running API).

---

## 🔌 API Endpoints Reference

FastAPI exposes the following REST API endpoints:

| Method | Route | Parameter | Description |
| :---: | :--- | :--- | :--- |
| **GET** | `/api/v1/health` | None | Returns backend status, parquet status, and model training state. |
| **GET** | `/api/v1/actuals` | `hours` (default: 168) | Returns historical observed demand, temperatures, pricing, and renewables. |
| **GET** | `/api/v1/forecast` | `horizon` (24/168), what-if factors | Returns out-of-sample recursive forecasts with what-if inputs. |
| **GET** | `/api/v1/metrics` | None | Returns validation metrics (R², RMSE, MAE, MAPE) and feature importances. |
| **POST** | `/api/v1/train` | None | Triggers pipeline retraining and refreshes the cached model singleton. |

---

## 📂 Codebase Layout

```
├── requirements.txt         # Project dependencies
├── run.py                   # Master concurrent process launcher
├── README.md                # System overview and quickstart
├── wiki.md                  # Deep-dive developer wiki & formulas
├── findings.md              # Analytical findings and scenario report
├── artifacts/               # Persistent metrics and generated screenshots
│   ├── metrics.json         # Out-of-sample model metrics and feature importance
│   └── *.png                # Dashboard interface screenshots
├── dashboard/
│   └── app.py               # Streamlit dashboard interface & Plotly configs
└── src/
    ├── __init__.py          # Source package init
    ├── config.py            # Central constants, pathways, & hyperparameters
    ├── data_generator.py    # Synthetic weather, load, & EPEX spot price simulator
    ├── model.py             # Feature engineering, model wrapper, & recursive forecaster
    ├── pipeline.py          # Orchestration pipeline (ETL, fitting, and serialization)
    └── api/
        ├── main.py          # FastAPI application routing
        └── schemas.py       # Pydantic schemas for API validation
```

---

## 📚 Further Reading

For deeper documentation, please refer to the following local resources:
* **[Developer Wiki](wiki.md):** Deep-dive on system architecture, mathematical formulas (physics engine), and recursive loop mechanics.
* **[Analytical Findings & Grid Report](findings.md):** Model evaluations, feature importances, day-ahead price correlations, and detailed what-if scenario impacts.
