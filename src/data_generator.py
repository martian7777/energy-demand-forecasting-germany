"""
High-fidelity simulator of German grid-level electricity demand and the
exogenous variables an energy analyst cares about: temperature, wind/solar
capacity factors, and EPEX Spot day-ahead prices.

The generator is fully deterministic given ``config.RANDOM_SEED`` so training
runs are reproducible, yet it reproduces the statistical signatures of real
load: diurnal twin peaks, the weekday/weekend split, the winter heating
ramp, public-holiday troughs, weather-driven heating/cooling, and the
merit-order price effect of renewables.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config


# ──────────────────────────────────────────────────────────────────────────
# German public holidays (nationwide)
# ──────────────────────────────────────────────────────────────────────────
def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def german_holidays(year: int) -> dict[date, str]:
    """Return the nationwide German public holidays for a given year."""
    easter = _easter_sunday(year)
    holidays = {
        date(year, 1, 1): "Neujahr",
        easter - timedelta(days=2): "Karfreitag",
        easter + timedelta(days=1): "Ostermontag",
        date(year, 5, 1): "Tag der Arbeit",
        easter + timedelta(days=39): "Christi Himmelfahrt",
        easter + timedelta(days=50): "Pfingstmontag",
        date(year, 10, 3): "Tag der Deutschen Einheit",
        date(year, 12, 25): "1. Weihnachtstag",
        date(year, 12, 26): "2. Weihnachtstag",
    }
    return holidays


def holiday_lookup(start: pd.Timestamp, end: pd.Timestamp) -> set[date]:
    """Set of holiday dates spanning the [start, end] range (inclusive years)."""
    dates: set[date] = set()
    for yr in range(start.year, end.year + 1):
        dates.update(german_holidays(yr).keys())
    return dates


# ──────────────────────────────────────────────────────────────────────────
# Weather generation
# ──────────────────────────────────────────────────────────────────────────
def _generate_temperature(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Seasonal + diurnal temperature with autocorrelated weather regimes."""
    n = len(idx)
    day_of_year = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()

    # Seasonal cycle: coldest ~mid-January (day 20), warmest ~late July.
    seasonal = config.TEMP_SEASONAL_AMPLITUDE_C * -np.cos(
        2 * np.pi * (day_of_year - 20) / 365.25
    )
    # Diurnal cycle: minimum ~05:00, maximum ~15:00.
    diurnal = config.TEMP_DAILY_AMPLITUDE_C * -np.cos(
        2 * np.pi * (hour - 15) / 24
    )

    # Autocorrelated weather regimes (multi-day heatwaves / cold snaps) via an
    # AR(1) process running on a daily timescale, then upsampled to hourly.
    n_days = int(np.ceil(n / 24)) + 1
    daily_regime = np.zeros(n_days)
    phi = 0.92  # daily persistence
    innov = rng.normal(0, 2.6, n_days)
    for t in range(1, n_days):
        daily_regime[t] = phi * daily_regime[t - 1] + innov[t]
    regime_hourly = np.repeat(daily_regime, 24)[:n]

    noise = rng.normal(0, config.TEMP_NOISE_STD_C, n)
    temp = config.TEMP_ANNUAL_MEAN_C + seasonal + diurnal + regime_hourly + noise
    return temp


def _generate_wind(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Wind capacity factor (0..1): windier in winter, strongly autocorrelated."""
    n = len(idx)
    day_of_year = idx.dayofyear.to_numpy()
    # Winter is windier in Germany.
    seasonal = 0.06 * -np.cos(2 * np.pi * (day_of_year - 10) / 365.25)

    # AR(1) weather-front persistence on an hourly scale.
    regime = np.zeros(n)
    phi = 0.96
    innov = rng.normal(0, 0.05, n)
    for t in range(1, n):
        regime[t] = phi * regime[t - 1] + innov[t]

    cf = config.WIND_MEAN_CF + seasonal + regime
    return np.clip(cf, 0.0, 1.0)


def _generate_solar(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Solar capacity factor (0..1): zero at night, seasonal+cloud modulation."""
    n = len(idx)
    day_of_year = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0

    # Daylight bell centered on solar noon (~13:00 local), wider in summer.
    summer = 0.5 * (1 - np.cos(2 * np.pi * (day_of_year - 172) / 365.25))  # 0..1
    half_day = 5.0 + 2.5 * summer  # daylight half-width in hours
    daylight = np.maximum(0.0, np.cos((hour - 13.0) / half_day * (np.pi / 2)))
    daylight = np.where(np.abs(hour - 13.0) <= half_day, daylight, 0.0)

    seasonal_peak = 0.35 + 0.65 * summer  # winter sun is weak
    # Cloud cover: autocorrelated daily multiplier.
    n_days = int(np.ceil(n / 24)) + 1
    cloud = np.clip(rng.normal(0.72, 0.22, n_days), 0.1, 1.0)
    cloud_hourly = np.repeat(cloud, 24)[:n]

    cf = config.SOLAR_PEAK_CF * daylight * seasonal_peak * cloud_hourly
    return np.clip(cf, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────
# Demand generation
# ──────────────────────────────────────────────────────────────────────────
def _daily_shape(hour: np.ndarray) -> np.ndarray:
    """
    Normalized daily load shape with the characteristic German twin peaks:
    a morning ramp (~08:00) and a higher evening peak (~19:00), a midday
    plateau, and a deep overnight trough. Returns values roughly in [-1, 1].
    """
    morning = np.exp(-((hour - 8.0) ** 2) / (2 * 2.2 ** 2))
    evening = np.exp(-((hour - 19.0) ** 2) / (2 * 2.6 ** 2))
    night = -np.exp(-((hour - 4.0) ** 2) / (2 * 3.0 ** 2))
    shape = 0.85 * morning + 1.0 * evening + 0.9 * night
    return shape


def _deterministic_demand(
    idx: pd.DatetimeIndex, temp: np.ndarray, is_holiday: np.ndarray
) -> np.ndarray:
    """The noise-free demand expectation from calendar + weather drivers (MW)."""
    hour = idx.hour.to_numpy().astype(float)
    dow = idx.dayofweek.to_numpy()
    day_of_year = idx.dayofyear.to_numpy()

    daily = _daily_shape(hour) * config.DAILY_AMPLITUDE_MW
    weekend = np.where(dow == 5, 0.55, np.where(dow == 6, 1.0, 0.0))
    weekend_effect = -weekend * config.WEEKEND_REDUCTION_MW
    seasonal = config.SEASONAL_AMPLITUDE_MW * -np.cos(
        2 * np.pi * (day_of_year - 15) / 365.25
    )
    heating = np.maximum(0.0, config.TEMP_COMFORT_C - temp) * config.HEATING_COEF_MW_PER_C
    cooling = np.maximum(0.0, temp - config.TEMP_COMFORT_C) * config.COOLING_COEF_MW_PER_C
    holiday_effect = -is_holiday * config.HOLIDAY_REDUCTION_MW

    demand = (
        config.BASE_LOAD_MW + daily + weekend_effect + seasonal
        + heating + cooling + holiday_effect
    )
    return demand


def _price_from(
    demand: np.ndarray, wind: np.ndarray, solar: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """EPEX Spot price from the merit-order relationship (EUR/MWh)."""
    residual_load = demand * (1.0 - 0.55 * wind - 0.45 * solar)
    price = (
        config.PRICE_BASE_EUR
        + config.PRICE_LOAD_COEF * (residual_load - config.BASE_LOAD_MW)
        + config.PRICE_RENEWABLE_COEF * (0.6 * wind + 0.4 * solar)
        + noise
    )
    return np.clip(price, config.PRICE_FLOOR_EUR, config.PRICE_CAP_EUR)


def generate(
    years: float = config.SIM_YEARS,
    end: pd.Timestamp | None = None,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate an hourly DataFrame of simulated German grid data.

    Columns: demand_mw, temp_c, wind_cf, solar_cf, price_eur, is_holiday,
    plus the DatetimeIndex named ``timestamp``.
    """
    rng = np.random.default_rng(seed)

    end = pd.Timestamp.now().floor("h") if end is None else pd.Timestamp(end).floor("h")
    periods = int(round(years * 365.25 * 24))
    idx = pd.date_range(end=end, periods=periods, freq=config.FREQ, name="timestamp")

    # ── Weather & renewables ───────────────────────────────────────────────
    temp = _generate_temperature(idx, rng)
    wind = _generate_wind(idx, rng)
    solar = _generate_solar(idx, rng)

    # ── Holiday mask ─────────────────────────────────────────────────────
    hol_dates = holiday_lookup(idx.min(), idx.max())
    py_dates = idx.date  # array of datetime.date
    is_holiday = np.array([d in hol_dates for d in py_dates], dtype=int)

    # ── Demand ─────────────────────────────────────────────────────────────
    noise = rng.normal(0, config.NOISE_STD_MW, periods)
    demand = _deterministic_demand(idx, temp, is_holiday) + noise
    demand = np.clip(demand, 25_000.0, None)

    # ── EPEX Spot price (merit-order: residual load up, renewables down) ────
    price = _price_from(demand, wind, solar, rng.normal(0, config.PRICE_NOISE_STD, periods))

    df = pd.DataFrame(
        {
            "demand_mw": demand,
            "temp_c": temp,
            "wind_cf": wind,
            "solar_cf": solar,
            "price_eur": price,
            "is_holiday": is_holiday,
        },
        index=idx,
    )
    return df


def make_future_exog(
    history: pd.DataFrame,
    horizon: int,
    sim: "dict | None" = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Build plausible *future* exogenous inputs (temp, wind, solar, price) for
    the ``horizon`` hours following the last timestamp in ``history``.

    Weather continues the seasonal/diurnal climatology with fresh stochastic
    regimes, smoothly blended onto the last observed values for the first
    half-day so the forecast inputs don't jump. Optional ``sim`` what-if
    overrides let the dashboard inject synthetic weather/price events:

        sim = {
            "temp_delta_c": float,      # add to every future temperature
            "solar_factor": float,      # scale solar (e.g. 0 = total cloud)
            "wind_factor": float,       # scale wind capacity factor
            "price_factor": float,      # scale the resulting EPEX price
        }
    """
    sim = sim or {}
    rng = np.random.default_rng(seed)
    last_ts = history.index.max()
    future_idx = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq=config.FREQ,
        name="timestamp",
    )

    temp = _generate_temperature(future_idx, rng)
    wind = _generate_wind(future_idx, rng)
    solar = _generate_solar(future_idx, rng)

    # Smoothly blend the first 12h onto the last observed weather so the
    # forecast inputs are continuous with reality.
    blend_n = min(12, horizon)
    if blend_n > 0 and len(history) > 0:
        w = np.linspace(1.0, 0.0, blend_n)
        last = history.iloc[-1]
        temp[:blend_n] = w * float(last["temp_c"]) + (1 - w) * temp[:blend_n]
        wind[:blend_n] = w * float(last["wind_cf"]) + (1 - w) * wind[:blend_n]
        solar[:blend_n] = w * float(last["solar_cf"]) + (1 - w) * solar[:blend_n]

    # ── Apply what-if simulation overrides ─────────────────────────────────
    temp = temp + float(sim.get("temp_delta_c", 0.0))
    wind = np.clip(wind * float(sim.get("wind_factor", 1.0)), 0.0, 1.0)
    solar = np.clip(solar * float(sim.get("solar_factor", 1.0)), 0.0, 1.0)

    # Holiday mask for the future window.
    hol_dates = holiday_lookup(future_idx.min(), future_idx.max())
    is_holiday = np.array([d in hol_dates for d in future_idx.date], dtype=int)

    # Price from the expected (noise-free) demand under these conditions.
    exp_demand = _deterministic_demand(future_idx, temp, is_holiday)
    price = _price_from(exp_demand, wind, solar, np.zeros(horizon))
    price = price * float(sim.get("price_factor", 1.0))

    return pd.DataFrame(
        {
            "temp_c": temp,
            "wind_cf": wind,
            "solar_cf": solar,
            "price_eur": price,
            "is_holiday": is_holiday,
        },
        index=future_idx,
    )


def load_or_generate(force: bool = False) -> pd.DataFrame:
    """
    Return the cached history from disk, regenerating it if missing, stale
    (does not reach within ~2 hours of now), or ``force`` is set.
    """
    path = config.DATASET_PATH
    if not force and path.exists():
        try:
            df = pd.read_parquet(path)
            fresh = (pd.Timestamp.now() - df.index.max()) < pd.Timedelta(hours=2)
            if fresh:
                return df
        except Exception:
            pass  # fall through to regeneration

    df = generate()
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    data = generate(years=0.05)
    print(data.tail())
    print("\nShape:", data.shape)
    print("Demand MW  -> mean {:.0f}  min {:.0f}  max {:.0f}".format(
        data.demand_mw.mean(), data.demand_mw.min(), data.demand_mw.max()))
    print("Price EUR  -> mean {:.1f}  min {:.1f}  max {:.1f}".format(
        data.price_eur.mean(), data.price_eur.min(), data.price_eur.max()))
