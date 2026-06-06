"""
StromCast dashboard — a premium Streamlit UI for German grid demand
forecasting. Talks to the FastAPI backend over REST.

Aesthetic: anthracite-gray base, neon-green demand/forecast, amber price.

Run standalone (after starting the API)::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Make ``src`` importable for config defaults even when launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

API = os.getenv("STROMCAST_API_URL", config.API_BASE_URL) + config.API_PREFIX

# ──────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────
BG = "#15171c"
PANEL = "#1d2026"
GRID = "#2c3038"
TEXT = "#e6e8eb"
MUTED = "#8b919c"
GREEN = "#39ff88"       # demand / forecast
GREEN_SOFT = "rgba(57,255,136,0.15)"
AMBER = "#ffb547"       # price
BLUE = "#4aa8ff"        # actuals
RED = "#ff5c7a"         # residuals / errors

st.set_page_config(
    page_title="StromCast · German Grid Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
    h1, h2, h3, h4 {{ color: {TEXT}; font-family: 'Segoe UI', sans-serif; }}
    .kpi-card {{
        background: linear-gradient(160deg, {PANEL} 0%, #23272f 100%);
        border: 1px solid {GRID};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }}
    .kpi-label {{ color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.08em;
                  text-transform: uppercase; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 700; margin-top: 4px; }}
    .kpi-sub   {{ color: {MUTED}; font-size: 0.8rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {PANEL}; border-radius: 8px 8px 0 0; padding: 8px 18px;
    }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:20px;
              font-size:0.72rem; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path: str, params: dict | None = None) -> dict:
    r = requests.post(f"{API}{path}", params=params, timeout=600)
    r.raise_for_status()
    return r.json()


def api_online() -> bool:
    try:
        api_get("/health")
        return True
    except Exception:
        return False


def base_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Segoe UI"),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kpi(col, label: str, value: str, sub: str = "", color: str = TEXT):
    col.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='margin-bottom:0'>⚡ Strom<span style='color:{GREEN}'>Cast</span></h1>"
    f"<div style='color:{MUTED}; margin-bottom:18px'>German Grid Electricity-Demand "
    f"Forecasting &amp; Analytics · <span class='badge' style='background:{GREEN_SOFT};"
    f"color:{GREEN}'>Energiewende</span></div>",
    unsafe_allow_html=True,
)

if not api_online():
    st.error(
        f"Cannot reach the StromCast API at **{API}**.\n\n"
        "Start the backend first — e.g. `python run.py`, or "
        "`uvicorn src.api.main:app --port 8000`.",
        icon="🔌",
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — What-If simulation & controls
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Forecast Controls")
    horizon_label = st.radio(
        "Forecast horizon",
        ["Next 24 hours", "Next 7 days"],
        index=0,
    )
    horizon = config.HORIZON_24H if horizon_label.startswith("Next 24") else config.HORIZON_7D

    st.markdown("---")
    st.markdown("### 🌦️ What-If Simulation")
    st.caption("Inject synthetic weather & market events into the forecast inputs.")

    temp_delta = st.slider("Temperature shift (°C)", -15.0, 15.0, 0.0, 0.5,
                           help="A cold snap (negative) lifts heating demand; a "
                                "heatwave (positive) lifts cooling demand.")
    solar_factor = st.slider("Solar availability ×", 0.0, 2.0, 1.0, 0.05,
                             help="0 = total cloud cover blocking solar PV.")
    wind_factor = st.slider("Wind availability ×", 0.0, 2.0, 1.0, 0.05)
    price_factor = st.slider("EPEX price ×", 0.0, 3.0, 1.0, 0.05)

    sim_active = not (temp_delta == 0.0 and solar_factor == 1.0
                      and wind_factor == 1.0 and price_factor == 1.0)
    if sim_active:
        st.markdown(f"<span class='badge' style='background:{AMBER};color:#000'>"
                    f"Simulation active</span>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔁 Retrain Model", use_container_width=True):
        with st.spinner("Retraining XGBoost on the full history…"):
            try:
                api_post("/train")
                api_get.clear()  # bust cache
                st.success("Model retrained.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Retrain failed: {exc}")

    if st.button("🔄 Refresh data", use_container_width=True):
        api_get.clear()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Fetch data
# ──────────────────────────────────────────────────────────────────────────
params = {
    "horizon": horizon,
    "temp_delta_c": temp_delta,
    "solar_factor": solar_factor,
    "wind_factor": wind_factor,
    "price_factor": price_factor,
    "matched_hours": 168,
}
try:
    fc = api_get("/forecast", params)
    metrics = api_get("/metrics")
    actuals = api_get("/actuals", {"hours": 336})
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch forecast: {exc}")
    st.stop()

df_fc = pd.DataFrame(fc["forecast"])
df_fc["timestamp"] = pd.to_datetime(df_fc["timestamp"])
df_matched = pd.DataFrame(fc["matched"])
if not df_matched.empty:
    df_matched["timestamp"] = pd.to_datetime(df_matched["timestamp"])
df_act = pd.DataFrame(actuals["points"])
df_act["timestamp"] = pd.to_datetime(df_act["timestamp"])

m = metrics["metrics"]

# ──────────────────────────────────────────────────────────────────────────
# Hero KPI cards
# ──────────────────────────────────────────────────────────────────────────
current_demand = df_act["demand_mw"].iloc[-1]
peak_fc = df_fc["forecast_mw"].max()
peak_time = df_fc.loc[df_fc["forecast_mw"].idxmax(), "timestamp"]
current_price = df_act["price_eur"].iloc[-1]
mape = m["mape"]

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Current Demand", f"{current_demand/1000:,.1f} GW",
    f"{current_demand:,.0f} MW · live", GREEN)
kpi(c2, "Peak Predicted Demand", f"{peak_fc/1000:,.1f} GW",
    f"at {peak_time:%a %H:%M}", BLUE)
kpi(c3, "Model MAPE", f"{mape:,.2f} %",
    f"RMSE {m['rmse']:,.0f} MW", GREEN if mape < 5 else AMBER)
kpi(c4, "EPEX Spot Price", f"{current_price:,.1f} €",
    "per MWh · day-ahead", AMBER)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────
tab_fc, tab_price, tab_health = st.tabs(
    ["📈 Forecast", "💶 Price vs Demand", "🩺 Model Health"]
)

# ── Forecast tab ───────────────────────────────────────────────────────────
with tab_fc:
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([df_fc["timestamp"], df_fc["timestamp"][::-1]]),
        y=pd.concat([df_fc["upper_mw"], df_fc["lower_mw"][::-1]]),
        fill="toself", fillcolor=GREEN_SOFT, line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="95% interval",
    ))
    # Actuals
    fig.add_trace(go.Scatter(
        x=df_act["timestamp"], y=df_act["demand_mw"], name="Actual demand",
        line=dict(color=BLUE, width=2),
    ))
    # Matched one-step predictions
    if not df_matched.empty:
        mp = df_matched.dropna(subset=["prediction_mw"])
        fig.add_trace(go.Scatter(
            x=mp["timestamp"], y=mp["prediction_mw"], name="Model (in-sample)",
            line=dict(color=GREEN, width=1.4, dash="dot"),
        ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=df_fc["timestamp"], y=df_fc["forecast_mw"], name="Forecast",
        line=dict(color=GREEN, width=3),
    ))
    # Now marker
    now_ts = df_act["timestamp"].iloc[-1]
    fig.add_vline(x=now_ts, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x=now_ts, y=1, yref="paper", text="now",
                       showarrow=False, font=dict(color=MUTED), yanchor="bottom")

    fig.update_yaxes(title="Demand (MW)")
    st.plotly_chart(base_layout(fig, 470), use_container_width=True)

    if sim_active:
        st.info(
            f"What-if applied → temp {temp_delta:+.1f} °C · solar ×{solar_factor:.2f} · "
            f"wind ×{wind_factor:.2f} · price ×{price_factor:.2f}", icon="🌦️")

    # Forecast driver detail
    cda, cdb = st.columns(2)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=df_fc["timestamp"], y=df_fc["temp_c"],
                            name="Temp °C", line=dict(color=AMBER)))
    f1.update_yaxes(title="Temperature (°C)")
    cda.plotly_chart(base_layout(f1, 300), use_container_width=True)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df_fc["timestamp"], y=df_fc["wind_cf"],
                            name="Wind CF", line=dict(color=BLUE)))
    f2.add_trace(go.Scatter(x=df_fc["timestamp"], y=df_fc["solar_cf"],
                            name="Solar CF", line=dict(color=GREEN)))
    f2.update_yaxes(title="Capacity factor")
    cdb.plotly_chart(base_layout(f2, 300), use_container_width=True)

# ── Price vs Demand tab ────────────────────────────────────────────────────
with tab_price:
    st.markdown("#### EPEX Spot price tracks residual grid load")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_act["timestamp"], y=df_act["demand_mw"],
                             name="Demand (MW)", line=dict(color=GREEN, width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_act["timestamp"], y=df_act["price_eur"],
                             name="EPEX price (€/MWh)", line=dict(color=AMBER, width=2)),
                  secondary_y=True)
    fig.update_yaxes(title_text="Demand (MW)", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Price (€/MWh)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(base_layout(fig, 420), use_container_width=True)

    cca, ccb = st.columns([2, 1])
    sc = go.Figure()
    sc.add_trace(go.Scatter(
        x=df_act["demand_mw"], y=df_act["price_eur"], mode="markers",
        marker=dict(color=df_act["solar_cf"] + df_act["wind_cf"],
                    colorscale="Viridis", size=7, opacity=0.75,
                    colorbar=dict(title="Renew.")),
        name="hours",
    ))
    sc.update_xaxes(title="Demand (MW)")
    sc.update_yaxes(title="Price (€/MWh)")
    cca.plotly_chart(base_layout(sc, 380), use_container_width=True)

    corr = df_act["demand_mw"].corr(df_act["price_eur"])
    ccb.markdown("<br>", unsafe_allow_html=True)
    kpi(ccb, "Demand ↔ Price corr.", f"{corr:+.2f}",
        "Pearson, last 14 days", AMBER)
    ccb.markdown("<br>", unsafe_allow_html=True)
    kpi(ccb, "Avg. price", f"{df_act['price_eur'].mean():,.1f} €",
        f"min {df_act['price_eur'].min():,.0f} · max {df_act['price_eur'].max():,.0f}",
        MUTED)

# ── Model Health tab ───────────────────────────────────────────────────────
with tab_health:
    h1, h2, h3, h4 = st.columns(4)
    kpi(h1, "RMSE", f"{m['rmse']:,.0f} MW", "validation", GREEN)
    kpi(h2, "MAE", f"{m['mae']:,.0f} MW", "validation", BLUE)
    kpi(h3, "MAPE", f"{m['mape']:,.2f} %", "validation", AMBER)
    kpi(h4, "R²", f"{m['r2']:,.3f}", "validation", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Trained {metrics['trained_at']} · {metrics['n_rows']:,} hourly rows · "
        f"{metrics['history_start'][:10]} → {metrics['history_end'][:10]}"
    )

    ca, cb = st.columns(2)

    # Feature importance
    fi = pd.DataFrame(metrics["feature_importance"]).sort_values("importance")
    fig_fi = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance"], colorscale=[[0, "#1d6b3f"], [1, GREEN]]),
    ))
    fig_fi.update_layout(title="Feature importance")
    fig_fi.update_xaxes(title="Gain importance")
    ca.plotly_chart(base_layout(fig_fi, 460), use_container_width=True)

    # Residual distribution from matched predictions
    if not df_matched.empty:
        res = (df_matched["actual_mw"] - df_matched["prediction_mw"]).dropna()
        fig_res = go.Figure(go.Histogram(
            x=res, nbinsx=40, marker=dict(color=RED, opacity=0.8)))
        fig_res.update_layout(title="Residuals (actual − predicted), recent week")
        fig_res.add_vline(x=0, line=dict(color=TEXT, width=1, dash="dash"))
        fig_res.update_xaxes(title="Residual (MW)")
        fig_res.update_yaxes(title="Count")
        cb.plotly_chart(base_layout(fig_res, 460), use_container_width=True)
        cb.caption(f"Residual mean {res.mean():,.0f} MW · std {res.std():,.0f} MW")

st.markdown(
    f"<div style='color:{MUTED}; text-align:center; margin-top:24px; font-size:0.78rem'>"
    f"StromCast · synthetic EPEX/Entso-E-style data · XGBoost time-series model</div>",
    unsafe_allow_html=True,
)
