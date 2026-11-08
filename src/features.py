#!/usr/bin/env python3
"""
Spatiotemporal, Meteorological & Chemical Feature Engineering (`src/features.py`)
for the Google Mumbai Air Quality Forecasting Challenge.

Given a forecast cutoff timestamp `T0`, constructs strictly leak-free features
for any target row `(location_id, hour_start)`:
1. Multi-Scale Empirical Bayes Diurnal Priors anchored at `T0`:
   - Station-hour mean & median over the last 7d, 14d, 28d before `T0`
     (exponentially decayed with half-life 5 days).
   - Station overall level over the last 3d, 7d, 14d before `T0`.
   - James-Stein / Empirical Bayes shrunken prior:
       y_hat_diurnal(s, h) = alpha * y_bar_{s,h}^{(10d)} + (1 - alpha) * y_bar_s^{(7d)}
     plus fast-adapting diurnal anomaly prior `y_hat_diurnal_adaptive`.
2. Autoregressive & Cross-Station Lags from `T0`:
   - Lead time `horizon_hours = (hour_start - T0)` in hours.
   - Exact lag at `T0`, `T0-1h`, `T0-2h`, `T0-3h`, `T0-6h`, `T0-12h`, `T0-24h`,
     and same-hour-of-day prior reading at `T0` (`last_known_same_hour_pm25`, `prev_day_same_hour_pm25`).
   - Cross-station spatial context at `T0` (mean, min, max, std across the 4 target stations +
     14 auxiliary Mumbai stations from `Other_Mumbai_Stations_2025_2026.csv.gz`, plus
     station-to-city ratio at `T0`).
3. Physics & Meteorological Features from Open-Meteo (`data/external/open_meteo_weather.parquet`) & VABB METAR:
   - Hourly `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`,
     `cloud_cover`, `shortwave_radiation`, `precipitation`, `wind_speed_10m`,
     `wind_direction_10m`, `wind_gusts_10m`, `boundary_layer_height` (PBLH).
   - Ventilation Coefficient: `vc = boundary_layer_height * wind_speed_10m` and `log1p_vc = log(1 + vc)`.
   - Coastal Land-Sea Breeze Vectors: `u10`, `v10`, onshore component normal to Mumbai's 15-deg
     coastline (`onshore_breeze`), and Deonar-to-Shivaji-Nagar / Thane-Creek wind alignment cosine.
   - Thermal Inversion & Stagnation Proxies: Dewpoint depression `(T - Td)`, 3h/6h temperature
     change `delta_t_3h`, `delta_t_6h`, 6h rolling minimum PBLH, consecutive stagnation hours.
   - Weather Anomaly Relative to `T0`: Difference between target-hour forecast weather
     (`PBLH`, `temp`, `RH`, `wind`, `vc`) and the recent 7-day average weather at `T0` for that
     same hour of day!
4. Open-Meteo CAMS Chemical Forecast Features (`data/external/open_meteo_aq.parquet`):
   - CAMS gridded `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide`,
     `ozone`, `aerosol_optical_depth`, `dust`.
   - Station-Calibrated CAMS Ratio/Offset anchored at `T0`:
     `cams_pm25_calibrated`, `cams_pm25_offset_calibrated`, and `cams_pm25_rel_change`.
   - Physics-calibrated composite prior `prior_physics_calibrated`.
5. Historical Multi-Year Seasonal Climatology (2018-2025):
   - Smoothed day-of-year seasonal curve and 7-day slope from `History_Mumbai_2018_2020`,
     `History_Mumbai_2024`, and `Train.csv` 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import IST_TZ, as_ist, clean_pm25, load_and_align_core_data

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATION_ORDER: list[int] = [11606, 11611, 3409510, 3409511]
STATION_INDEX_MAP: dict[int, int] = {loc: idx for idx, loc in enumerate(STATION_ORDER)}

# Coordinates of the 4 Mumbai target stations in `STATION_ORDER` = [11606, 11611, 3409510, 3409511]
STATION_COORDS: dict[int, tuple[float, float]] = {
    11606: (19.23241, 72.86895),      # Borivali East
    11611: (19.19709, 72.82204),      # Malad West
    3409510: (18.97670, 72.83800),    # Byculla
    3409511: (19.060498, 72.923356),  # Shivaji Nagar
}


def _build_airphynet_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise physical distance d_ij (km) and unit direction vectors e_{j -> i} = (e_x, e_y)
    from station j to station i across the 4 target stations.
    """
    n = len(STATION_ORDER)
    dist_km = np.zeros((n, n), dtype=np.float64)
    dir_east = np.zeros((n, n), dtype=np.float64)
    dir_north = np.zeros((n, n), dtype=np.float64)
    for i, loc_i in enumerate(STATION_ORDER):
        lat_i, lon_i = STATION_COORDS[loc_i]
        for j, loc_j in enumerate(STATION_ORDER):
            if i == j:
                continue
            lat_j, lon_j = STATION_COORDS[loc_j]
            dy_km = (lat_i - lat_j) * 111.0
            dx_km = (lon_i - lon_j) * 111.0 * np.cos(np.deg2rad(0.5 * (lat_i + lat_j)))
            d = float(np.hypot(dx_km, dy_km))
            dist_km[i, j] = max(d, 0.5)
            dir_east[i, j] = dx_km / max(d, 1e-3)
            dir_north[i, j] = dy_km / max(d, 1e-3)
    return dist_km, dir_east, dir_north


_AIRPHYNET_DIST_KM, _AIRPHYNET_DIR_EAST, _AIRPHYNET_DIR_NORTH = _build_airphynet_geometry()


@dataclass
class ExternalDataStore:
    """In-memory pre-indexed cache of external weather, CAMS AQ, METAR, aux stations, and climatology."""

    weather_df: pd.DataFrame
    aq_df: pd.DataFrame
    metar_df: pd.DataFrame
    aux_hourly_df: pd.DataFrame
    climatology_doy: pd.DataFrame
    stations_df: pd.DataFrame


_GLOBAL_STORE: ExternalDataStore | None = None


def _compute_weather_physics(wx: pd.DataFrame) -> pd.DataFrame:
    """Enrich Open-Meteo weather table with physical dispersion, breeze, and stagnation features."""
    df = wx.sort_values(["location_id", "target_start_ist"]).reset_index(drop=True).copy()

    # Use NWP-consistent variables (`*_nwp`) so historical and forecast weather share identical physics
    for base_col in [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "surface_pressure",
        "cloud_cover",
        "shortwave_radiation",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "boundary_layer_height",
    ]:
        nwp_col = f"{base_col}_nwp"
        if nwp_col in df.columns:
            df[base_col] = df[nwp_col].combine_first(df[base_col])

    # 1. Ventilation Coefficient: VC = PBLH * wind_speed_10m and log(1 + VC)
    pblh = np.maximum(df["boundary_layer_height"].values, 15.0)
    wspd = np.maximum(df["wind_speed_10m"].values, 0.1)
    df["vc"] = pblh * wspd
    df["log1p_vc"] = np.log1p(df["vc"])
    df["log_pblh"] = np.log(pblh)

    # 2. Coastal Land-Sea Breeze Vectors
    # Meteorological wind_direction_10m is the direction wind is blowing FROM (degrees clockwise from North).
    # Standard meteorological vector components (direction wind is blowing TOWARDS):
    wdir_rad = np.deg2rad(df["wind_direction_10m"].values)
    df["u10"] = -wspd * np.sin(wdir_rad)
    df["v10"] = -wspd * np.cos(wdir_rad)
    df["wind_dir_sin"] = np.sin(wdir_rad)
    df["wind_dir_cos"] = np.cos(wdir_rad)

    # Mumbai coastline runs ~15 degrees NNE-SSW; normal pointing onshore (roughly WNW -> ESE, FROM 285 deg)
    # Onshore component > 0 when sea breeze blows from Arabian Sea (~285 deg) onto land
    coast_normal_from_rad = np.deg2rad(285.0)
    df["onshore_breeze"] = wspd * np.cos(wdir_rad - coast_normal_from_rad)

    # Deonar landfill / Thane Creek industrial alignment:
    # Deonar dumping ground lies immediately East/NE (~75 deg) of Shivaji Nagar, while Thane Creek/Mahul
    # industrial belt lies E/SE (~110 deg). Alignment cosine of wind coming FROM 75 deg (ENE) and 110 deg (ESE):
    df["deonar_wind_cos"] = np.cos(wdir_rad - np.deg2rad(75.0))
    df["thane_creek_wind_cos"] = np.cos(wdir_rad - np.deg2rad(110.0))
    df["deonar_transport_proxy"] = wspd * np.maximum(0.0, df["deonar_wind_cos"])

    # 3. Thermal Inversion & Stagnation Proxies
    df["dewpoint_depression"] = df["temperature_2m"] - df["dew_point_2m"]

    # Group by station to compute temporal changes and rolling stagnation without cross-station bleeding
    g = df.groupby("location_id", sort=False)
    df["delta_t_3h"] = g["temperature_2m"].diff(3).fillna(0.0)
    df["delta_t_6h"] = g["temperature_2m"].diff(6).fillna(0.0)
    df["delta_rh_3h"] = g["relative_humidity_2m"].diff(3).fillna(0.0)
    df["delta_pressure_6h"] = g["surface_pressure"].diff(6).fillna(0.0)
    df["pblh_min_6h"] = g["boundary_layer_height"].transform(
        lambda s: s.rolling(6, min_periods=1).min()
    )
    df["pblh_mean_6h"] = g["boundary_layer_height"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    df["vc_mean_6h"] = g["vc"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["wind_mean_6h"] = g["wind_speed_10m"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )

    # Consecutive stagnation hours (PBLH < 350m and wind_speed_10m < 8.5 km/h)
    is_stag = ((df["boundary_layer_height"] < 350.0) & (df["wind_speed_10m"] < 8.5)).astype(int)
    df["is_stagnant"] = is_stag

    stag_streaks: list[pd.Series] = []
    for _, grp in df.groupby("location_id", sort=False):
        st_flag = grp["is_stagnant"]
        # Cumulative count resetting on 0
        blocks = (st_flag == 0).cumsum()
        streak = st_flag.groupby(blocks).cumsum()
        stag_streaks.append(streak)
    df["stagnation_hours"] = pd.concat(stag_streaks).sort_index()

    return df


def _compute_aq_features(aq: pd.DataFrame) -> pd.DataFrame:
    """Enrich Open-Meteo CAMS Air Quality forecast table with temporal gradients and ratios."""
    df = aq.sort_values(["location_id", "target_start_ist"]).reset_index(drop=True).copy()
    g = df.groupby("location_id", sort=False)

    df["cams_pm25_roll6h"] = g["pm2_5"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["cams_pm25_roll24h"] = g["pm2_5"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["cams_pm25_diff3h"] = g["pm2_5"].diff(3).fillna(0.0)
    df["cams_pm25_diff24h"] = g["pm2_5"].diff(24).fillna(0.0)
    df["cams_pm25_to_pm10"] = df["pm2_5"] / np.maximum(df["pm10"], 1.0)
    df["cams_log_pm25"] = np.log1p(np.maximum(df["pm2_5"], 0.0))
    df["cams_log_pm10"] = np.log1p(np.maximum(df["pm10"], 0.0))
    df["cams_log_no2"] = np.log1p(np.maximum(df["nitrogen_dioxide"], 0.0))
    df["cams_log_so2"] = np.log1p(np.maximum(df["sulphur_dioxide"], 0.0))
    df["cams_log_co"] = np.log1p(np.maximum(df["carbon_monoxide"], 0.0))
    return df


def _build_climatology_doy(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Build multi-year smoothed day-of-year (1..366) seasonal climatology curve for Mumbai
    combining:
      - `History_Mumbai_2018_2020.csv.gz`
      - `History_Mumbai_2024.csv.gz`
      - `Train.csv` (2025-2026)
    Captures the post-monsoon October-November upward seasonal climb slope.
    """
    add_dir = raw_dir / "additional_history"
    doy_frames: list[pd.DataFrame] = []

    p18 = add_dir / "History_Mumbai_2018_2020.csv.gz"
    if p18.exists():
        h18 = pd.read_csv(p18, usecols=["Datetime", "PM2.5"], low_memory=False)
        h18["pm25"] = pd.to_numeric(h18["PM2.5"], errors="coerce")
        h18 = h18.loc[(h18["pm25"] > 0) & (h18["pm25"] <= 450.0)].copy()
        dt18 = pd.to_datetime(h18["Datetime"], errors="coerce")
        doy_frames.append(pd.DataFrame({"doy": dt18.dt.day_of_year, "pm25": h18["pm25"]}))

    p24 = add_dir / "History_Mumbai_2024.csv.gz"
    if p24.exists():
        h24 = pd.read_csv(p24, usecols=["timestamp_ist", "value"], low_memory=False)
        h24["pm25"] = pd.to_numeric(h24["value"], errors="coerce")
        h24 = h24.loc[(h24["pm25"] > 0) & (h24["pm25"] <= 450.0)].copy()
        dt24 = pd.to_datetime(h24["timestamp_ist"], utc=True, errors="coerce").dt.tz_convert(IST_TZ)
        doy_frames.append(pd.DataFrame({"doy": dt24.dt.day_of_year, "pm25": h24["pm25"]}))

    p_tr = raw_dir / "Train.csv"
    if p_tr.exists():
        tr = pd.read_csv(p_tr, usecols=["timestamp_ist", "pm25"])
        tr["pm25"] = pd.to_numeric(tr["pm25"], errors="coerce")
        tr = tr.loc[(tr["pm25"] > 0) & (tr["pm25"] <= 450.0)].copy()
        dt_tr = pd.to_datetime(tr["timestamp_ist"], utc=True, errors="coerce").dt.tz_convert(IST_TZ)
        doy_frames.append(pd.DataFrame({"doy": dt_tr.dt.day_of_year, "pm25": tr["pm25"]}))

    all_doy = pd.concat(doy_frames, ignore_index=True).dropna()
    daily_raw = all_doy.groupby("doy")["pm25"].agg(["mean", "median"]).reindex(range(1, 367))
    daily_raw["mean"] = daily_raw["mean"].interpolate(limit_direction="both")
    daily_raw["median"] = daily_raw["median"].interpolate(limit_direction="both")

    # Circular 15-day smoothing across year boundaries
    tripled_mean = pd.concat([daily_raw["mean"], daily_raw["mean"], daily_raw["mean"]], ignore_index=True)
    tripled_med = pd.concat([daily_raw["median"], daily_raw["median"], daily_raw["median"]], ignore_index=True)
    smooth_mean = tripled_mean.rolling(15, center=True, min_periods=1).mean().iloc[366 : 2 * 366].values
    smooth_med = tripled_med.rolling(15, center=True, min_periods=1).mean().iloc[366 : 2 * 366].values

    clim = pd.DataFrame(
        {
            "doy": np.arange(1, 367, dtype=int),
            "clim_doy_mean": smooth_mean,
            "clim_doy_median": smooth_med,
        }
    )
    # 7-day seasonal slope (captures rate of post-monsoon build-up per week)
    clim["clim_doy_slope_7d"] = pd.Series(clim["clim_doy_mean"]).diff(7).bfill().values
    clim["clim_doy_ratio_7d"] = clim["clim_doy_mean"] / np.maximum(
        pd.Series(clim["clim_doy_mean"]).shift(7).bfill().values, 5.0
    )
    return clim


def _load_auxiliary_stations_hourly(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load `Other_Mumbai_Stations_2025_2026.csv.gz` aggregated by `hour_end` across auxiliary stations."""
    aux_path = raw_dir / "additional_history" / "Other_Mumbai_Stations_2025_2026.csv.gz"
    if not aux_path.exists():
        return pd.DataFrame(columns=["hour_end", "aux_city_mean", "aux_city_median", "aux_city_max", "aux_station_count"])

    aux = pd.read_csv(aux_path, low_memory=False)
    aux["hour_end"] = as_ist(aux["timestamp_ist"])
    aux["mean"] = pd.to_numeric(aux["mean"], errors="coerce")
    aux = aux.loc[(aux["mean"] > 0.0) & (aux["mean"] <= 450.0)].copy()

    agg = (
        aux.groupby("hour_end", as_index=False)
        .agg(
            aux_city_mean=("mean", "mean"),
            aux_city_median=("mean", "median"),
            aux_city_max=("mean", "max"),
            aux_station_count=("mean", "count"),
        )
        .sort_values("hour_end")
        .reset_index(drop=True)
    )
    return agg


def get_external_data_store(
    raw_dir: Path = RAW_DIR,
    external_dir: Path = EXTERNAL_DIR,
) -> ExternalDataStore:
    """Load and cache all external & historical datasets once in memory."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE

    wx_path = external_dir / "open_meteo_weather.parquet"
    if not wx_path.exists():
        wx_path = external_dir / "open_meteo_weather.csv"
        wx = pd.read_csv(wx_path)
        wx["target_start_ist"] = as_ist(wx["target_start_ist"])
    else:
        wx = pd.read_parquet(wx_path)
        wx["target_start_ist"] = as_ist(wx["target_start_ist"])
    wx["location_id"] = wx["location_id"].astype(int)
    wx = _compute_weather_physics(wx)

    aq_path = external_dir / "open_meteo_aq.parquet"
    if not aq_path.exists():
        aq_path = external_dir / "open_meteo_air_quality.parquet"
    if not aq_path.exists():
        aq_path = external_dir / "open_meteo_air_quality.csv"
        aq = pd.read_csv(aq_path)
    else:
        aq = pd.read_parquet(aq_path)
    aq["target_start_ist"] = as_ist(aq["target_start_ist"])
    aq["location_id"] = aq["location_id"].astype(int)
    aq = _compute_aq_features(aq)

    metar_path = external_dir / "vabb_metar_hourly.parquet"
    if metar_path.exists():
        metar = pd.read_parquet(metar_path)
        metar["target_start_ist"] = as_ist(metar["target_start_ist"])
    else:
        metar = pd.DataFrame(columns=["target_start_ist"])

    aux_hourly = _load_auxiliary_stations_hourly(raw_dir)
    clim_doy = _build_climatology_doy(raw_dir)
    stations = pd.read_csv(raw_dir / "Stations.csv")
    stations["location_id"] = stations["location_id"].astype(int)

    _GLOBAL_STORE = ExternalDataStore(
        weather_df=wx,
        aq_df=aq,
        metar_df=metar,
        aux_hourly_df=aux_hourly,
        climatology_doy=clim_doy,
        stations_df=stations,
    )
    return _GLOBAL_STORE


def build_features_for_cutoff(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cutoff_t0: pd.Timestamp,
    store: ExternalDataStore | None = None,
) -> pd.DataFrame:
    """
    Given a forecast cutoff timestamp `T0` (`cutoff_t0`), construct strictly leak-free
    features for every row in `target_df` `(location_id, hour_start)`.

    Strict Anti-Leakage Contract:
      - Only observations in `history_df` with `hour_end <= cutoff_t0` are accessed!
      - VABB METAR ground observations are only accessed at `target_start_ist + 1h <= cutoff_t0`
        (to compute historical airport conditions at `T0`).
      - Open-Meteo NWP weather and CAMS AQ forecasts are merged at `target_start_ist`
        and compared against their own trailing 7-day baseline at `T0`.
    """
    if store is None:
        store = get_external_data_store()

    # Enforce strict causal filter on history_df
    hist = history_df.loc[history_df["hour_end"] <= cutoff_t0].copy()
    if "pm25_clean" not in hist.columns:
        hist = clean_pm25(hist)
    hist["hour"] = hist["hour_start"].dt.hour

    df = target_df.copy().reset_index(drop=True)
    df["location_id"] = df["location_id"].astype(int)
    if "hour_start" not in df.columns:
        df["hour_start"] = as_ist(df["target_start_ist"])
    df["target_start_ist"] = df["hour_start"]
    df["hour"] = df["hour_start"].dt.hour
    df["day_of_week"] = df["hour_start"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["doy"] = df["hour_start"].dt.day_of_year

    # Lead time horizon in hours relative to T0
    df["horizon_hours"] = (df["hour_start"] - cutoff_t0).dt.total_seconds() / 3600.0
    df["horizon_days"] = df["horizon_hours"] / 24.0
    df["log1p_horizon"] = np.log1p(np.maximum(df["horizon_hours"], 0.0))

    # Cyclical Fourier harmonics for hour of day & day of year
    for k in [1, 2, 3]:
        df[f"hour_sin_{k}"] = np.sin(2.0 * np.pi * k * df["hour"] / 24.0)
        df[f"hour_cos_{k}"] = np.cos(2.0 * np.pi * k * df["hour"] / 24.0)
    df["doy_sin_1"] = np.sin(2.0 * np.pi * df["doy"] / 365.25)
    df["doy_cos_1"] = np.cos(2.0 * np.pi * df["doy"] / 365.25)
    df["station_idx"] = df["location_id"].map(STATION_INDEX_MAP).fillna(0).astype(int)

    # -------------------------------------------------------------------------
    # 1. Multi-Scale Empirical Bayes Diurnal Priors anchored at T0
    # -------------------------------------------------------------------------
    age_h = (cutoff_t0 - hist["hour_end"]).dt.total_seconds() / 3600.0
    hist["age_hours"] = age_h
    # Exponential decay with half-life = 5 days (120h) as specified, plus fast 36h decay
    hist["w_hl5d"] = np.exp(-np.log(2.0) * age_h / (5.0 * 24.0))
    hist["w_hl36h"] = np.exp(-np.log(2.0) * age_h / 36.0)
    hist["w_hl18h"] = np.exp(-np.log(2.0) * age_h / 18.0)

    global_fallback = float(
        hist.loc[hist["age_hours"] <= 14 * 24, "pm25_clean"].mean()
        if (hist["age_hours"] <= 14 * 24).any()
        else hist["pm25_clean"].mean()
    )

    # Station overall levels over last 1d (24h), 3d, 7d, 14d before T0
    for d_win in [1, 3, 7, 14]:
        sub = hist.loc[hist["age_hours"] <= d_win * 24].copy()
        sub["wy"] = sub["w_hl5d"] * sub["pm25_clean"]
        s_sum = sub.groupby("location_id")[["wy", "w_hl5d"]].sum()
        s_ewm = (s_sum["wy"] / np.maximum(s_sum["w_hl5d"], 1e-6)).to_dict()
        s_med = sub.groupby("location_id")["pm25_clean"].median().to_dict()
        s_std = sub.groupby("location_id")["pm25_clean"].std().fillna(5.0).to_dict()
        df[f"station_level_{d_win}d_mean"] = df["location_id"].map(s_ewm).fillna(global_fallback)
        df[f"station_level_{d_win}d_median"] = df["location_id"].map(s_med).fillna(global_fallback)
        if d_win in (3, 7):
            df[f"station_level_{d_win}d_std"] = df["location_id"].map(s_std).fillna(8.0)

    # Fast-adapting station level at T0 (half-life 18h over last 3d)
    sub_3d = hist.loc[hist["age_hours"] <= 3 * 24].copy()
    sub_3d["wy18"] = sub_3d["w_hl18h"] * sub_3d["pm25_clean"]
    s_fast_sum = sub_3d.groupby("location_id")[["wy18", "w_hl18h"]].sum()
    s_fast_map = (s_fast_sum["wy18"] / np.maximum(s_fast_sum["w_hl18h"], 1e-6)).to_dict()
    df["station_level_fast_t0"] = df["location_id"].map(s_fast_map).fillna(df["station_level_7d_mean"])

    # Station-hour mean & median over last 7d, 10d, 14d, 28d before T0 (half-life 5 days)
    for d_win in [7, 10, 14, 28]:
        sub = hist.loc[hist["age_hours"] <= d_win * 24].copy()
        sub["wy"] = sub["w_hl5d"] * sub["pm25_clean"]
        sh_sum = sub.groupby(["location_id", "hour"])[["wy", "w_hl5d"]].sum()
        sh_mean = (sh_sum["wy"] / np.maximum(sh_sum["w_hl5d"], 1e-6)).rename(f"sh_mean_{d_win}d")
        sh_med = sub.groupby(["location_id", "hour"])["pm25_clean"].median().rename(f"sh_median_{d_win}d")

        df = df.merge(sh_mean.reset_index(), on=["location_id", "hour"], how="left")
        df = df.merge(sh_med.reset_index(), on=["location_id", "hour"], how="left")
        fallback_col = "station_level_7d_mean" if d_win <= 10 else "station_level_14d_mean"
        df[f"sh_mean_{d_win}d"] = df[f"sh_mean_{d_win}d"].fillna(df[fallback_col])
        df[f"sh_median_{d_win}d"] = df[f"sh_median_{d_win}d"].fillna(df[fallback_col])

    # James-Stein / Empirical Bayes shrunken prior:
    # y_hat_diurnal(s, h) = alpha * y_bar_{s,h}^{(10d)} + (1 - alpha) * y_bar_s^{(7d)}
    alpha_eb = 0.65
    df["prior_eb_diurnal"] = (
        alpha_eb * df["sh_mean_10d"] + (1.0 - alpha_eb) * df["station_level_7d_mean"]
    )

    # Adaptive level-shifted diurnal prior (combines fast station level at T0 with 7d diurnal shape)
    df["diurnal_shape_7d"] = df["sh_mean_7d"] - df["station_level_7d_mean"]
    df["diurnal_ratio_7d"] = np.clip(
        df["sh_mean_7d"] / np.maximum(df["station_level_7d_mean"], 5.0), 0.45, 2.1
    )
    level_anchor = 0.60 * df["station_level_fast_t0"] + 0.40 * df["station_level_7d_mean"]
    df["prior_diurnal_adaptive"] = np.clip(
        level_anchor + 0.65 * df["diurnal_shape_7d"], 2.0, 450.0
    )

    # Recent trend at T0 (3d level minus 14d level)
    df["station_trend_3d_vs_14d"] = df["station_level_3d_mean"] - df["station_level_14d_mean"]
    df["station_trend_1d_vs_7d"] = df["station_level_1d_mean"] - df["station_level_7d_mean"]

    # -------------------------------------------------------------------------
    # 2. Autoregressive & Cross-Station Lags from T0
    # -------------------------------------------------------------------------
    # Create dense hourly lookup per station up to T0 (forward-filling short missing gaps up to 12h)
    t0_grid = pd.date_range(
        start=cutoff_t0 - pd.Timedelta(days=10),
        end=cutoff_t0,
        freq="1h",
        tz=IST_TZ,
    )
    piv = (
        hist.loc[hist["hour_end"] >= cutoff_t0 - pd.Timedelta(days=10)]
        .pivot_table(index="hour_end", columns="location_id", values="pm25_clean", aggfunc="mean")
        .reindex(t0_grid)
        .ffill()
        .bfill()
    )
    for loc in STATION_ORDER:
        if loc not in piv.columns:
            piv[loc] = global_fallback

    # Exact lags at T0, T0-1h, T0-2h, T0-3h, T0-6h, T0-12h, T0-24h
    for lag_h in [0, 1, 2, 3, 6, 12, 24]:
        ts_lag = cutoff_t0 - pd.Timedelta(hours=lag_h)
        if ts_lag in piv.index:
            lag_map = piv.loc[ts_lag].to_dict()
        else:
            lag_map = piv.iloc[-1].to_dict()
        df[f"pm25_t0_lag_{lag_h}h"] = df["location_id"].map(lag_map).fillna(df["station_level_3d_mean"])

    df["pm25_t0_diff_3h"] = df["pm25_t0_lag_0h"] - df["pm25_t0_lag_3h"]
    df["pm25_t0_diff_24h"] = df["pm25_t0_lag_0h"] - df["pm25_t0_lag_24h"]

    # Same-hour-of-day prior reading at T0 (`last_known_same_hour_pm25` and `prev_day_same_hour_pm25`)
    last_24h_piv = piv.iloc[-24:].copy()
    last_24h_piv["hour"] = (last_24h_piv.index - pd.Timedelta(hours=1)).hour
    prev_24h_piv = piv.iloc[-48:-24].copy()
    prev_24h_piv["hour"] = (prev_24h_piv.index - pd.Timedelta(hours=1)).hour

    same_hr_last = (
        last_24h_piv.melt(id_vars="hour", var_name="location_id", value_name="last_known_same_hour_pm25")
        .drop_duplicates(["location_id", "hour"])
    )
    same_hr_prev = (
        prev_24h_piv.melt(id_vars="hour", var_name="location_id", value_name="prev_day_same_hour_pm25")
        .drop_duplicates(["location_id", "hour"])
    )
    df = df.merge(same_hr_last, on=["location_id", "hour"], how="left")
    df = df.merge(same_hr_prev, on=["location_id", "hour"], how="left")
    df["last_known_same_hour_pm25"] = df["last_known_same_hour_pm25"].fillna(df["sh_mean_7d"])
    df["prev_day_same_hour_pm25"] = df["prev_day_same_hour_pm25"].fillna(df["sh_mean_7d"])

    # Cross-station spatial context at T0 (24h trailing mean across the 4 target stations + aux stations)
    city_t0_24h = piv.iloc[-24:].mean(axis=0)
    city_mean_t0 = float(city_t0_24h.mean())
    city_min_t0 = float(city_t0_24h.min())
    city_max_t0 = float(city_t0_24h.max())
    city_std_t0 = float(city_t0_24h.std()) if len(city_t0_24h) > 1 else 0.0
    station_t0_24h_map = city_t0_24h.to_dict()

    df["city_pm25_mean_t0"] = city_mean_t0
    df["city_pm25_min_t0"] = city_min_t0
    df["city_pm25_max_t0"] = city_max_t0
    df["city_pm25_std_t0"] = city_std_t0
    df["station_to_city_ratio_t0"] = np.clip(
        df["location_id"].map(station_t0_24h_map).fillna(city_mean_t0) / max(city_mean_t0, 5.0),
        0.4,
        2.2,
    )

    # Auxiliary Mumbai stations context at T0 (from `Other_Mumbai_Stations_2025_2026.csv.gz`)
    aux_sub = store.aux_hourly_df.loc[
        (store.aux_hourly_df["hour_end"] <= cutoff_t0)
        & (store.aux_hourly_df["hour_end"] > cutoff_t0 - pd.Timedelta(days=3))
    ]
    if not aux_sub.empty:
        df["aux_mumbai_mean_t0"] = float(aux_sub["aux_city_mean"].mean())
        df["aux_mumbai_max_t0"] = float(aux_sub["aux_city_max"].mean())
    else:
        df["aux_mumbai_mean_t0"] = city_mean_t0
        df["aux_mumbai_max_t0"] = city_max_t0

    # -------------------------------------------------------------------------
    # 3. Physics & Meteorological Features + Weather Anomaly Relative to T0
    # -------------------------------------------------------------------------
    wx_cols = [
        "location_id",
        "target_start_ist",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "cloud_cover",
        "shortwave_radiation",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "boundary_layer_height",
        "vc",
        "log1p_vc",
        "log_pblh",
        "u10",
        "v10",
        "wind_dir_sin",
        "wind_dir_cos",
        "onshore_breeze",
        "deonar_wind_cos",
        "thane_creek_wind_cos",
        "deonar_transport_proxy",
        "dewpoint_depression",
        "delta_t_3h",
        "delta_t_6h",
        "delta_rh_3h",
        "delta_pressure_6h",
        "pblh_min_6h",
        "pblh_mean_6h",
        "vc_mean_6h",
        "wind_mean_6h",
        "is_stagnant",
        "stagnation_hours",
    ]
    df = df.merge(store.weather_df[wx_cols], on=["location_id", "target_start_ist"], how="left")

    # Compute 7-day average weather at T0 for each (location_id, hour) to calculate Weather Anomaly Relative to T0!
    wx_hist_7d = store.weather_df.loc[
        (store.weather_df["target_start_ist"] + pd.Timedelta(hours=1) <= cutoff_t0)
        & (store.weather_df["target_start_ist"] + pd.Timedelta(hours=1) > cutoff_t0 - pd.Timedelta(days=7))
    ].copy()
    wx_hist_7d["hour"] = wx_hist_7d["target_start_ist"].dt.hour
    wx_base_7d = (
        wx_hist_7d.groupby(["location_id", "hour"], as_index=False)
        .agg(
            t0_7d_pblh=("boundary_layer_height", "mean"),
            t0_7d_log1p_vc=("log1p_vc", "mean"),
            t0_7d_temp=("temperature_2m", "mean"),
            t0_7d_rh=("relative_humidity_2m", "mean"),
            t0_7d_wind=("wind_speed_10m", "mean"),
            t0_7d_onshore=("onshore_breeze", "mean"),
            t0_7d_dewdep=("dewpoint_depression", "mean"),
        )
    )
    df = df.merge(wx_base_7d, on=["location_id", "hour"], how="left")
    df = df.copy()

    # Weather Anomalies Relative to T0 (Target-hour forecast weather minus recent 7d baseline weather at T0)
    df["wx_anom_pblh"] = df["boundary_layer_height"] - df["t0_7d_pblh"].fillna(df["boundary_layer_height"])
    df["wx_ratio_pblh"] = np.clip(
        df["boundary_layer_height"] / np.maximum(df["t0_7d_pblh"].fillna(df["boundary_layer_height"]), 30.0),
        0.3,
        3.0,
    )
    df["wx_anom_log1p_vc"] = df["log1p_vc"] - df["t0_7d_log1p_vc"].fillna(df["log1p_vc"])
    df["wx_anom_temp"] = df["temperature_2m"] - df["t0_7d_temp"].fillna(df["temperature_2m"])
    df["wx_anom_rh"] = df["relative_humidity_2m"] - df["t0_7d_rh"].fillna(df["relative_humidity_2m"])
    df["wx_anom_wind"] = df["wind_speed_10m"] - df["t0_7d_wind"].fillna(df["wind_speed_10m"])
    df["wx_anom_onshore"] = df["onshore_breeze"] - df["t0_7d_onshore"].fillna(df["onshore_breeze"])
    df["wx_anom_dewdep"] = df["dewpoint_depression"] - df["t0_7d_dewdep"].fillna(df["dewpoint_depression"])

    # VABB Airport METAR baseline at T0 (strictly causal: only observations <= T0)
    if not store.metar_df.empty:
        metar_t0 = store.metar_df.loc[
            (store.metar_df["target_start_ist"] + pd.Timedelta(hours=1) <= cutoff_t0)
            & (store.metar_df["target_start_ist"] + pd.Timedelta(hours=1) > cutoff_t0 - pd.Timedelta(hours=24))
        ]
        if not metar_t0.empty:
            df["vabb_t0_24h_tmpc"] = float(metar_t0["vabb_tmpc"].mean())
            df["vabb_t0_24h_relh"] = float(metar_t0["vabb_relh"].mean())
            df["vabb_t0_24h_wind_ms"] = float(metar_t0["vabb_wind_speed_ms"].mean())
            df["vabb_t0_24h_dewdep"] = float(metar_t0["vabb_dewpoint_depression_c"].mean())
        else:
            df["vabb_t0_24h_tmpc"] = 29.0
            df["vabb_t0_24h_relh"] = 65.0
            df["vabb_t0_24h_wind_ms"] = 2.5
            df["vabb_t0_24h_dewdep"] = 6.0
    else:
        df["vabb_t0_24h_tmpc"] = 29.0
        df["vabb_t0_24h_relh"] = 65.0
        df["vabb_t0_24h_wind_ms"] = 2.5
        df["vabb_t0_24h_dewdep"] = 6.0

    # -------------------------------------------------------------------------
    # 4. Open-Meteo CAMS Chemical Forecast Features + Station-Calibrated CAMS
    # -------------------------------------------------------------------------
    aq_cols = [
        "location_id",
        "target_start_ist",
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "carbon_monoxide",
        "ozone",
        "aerosol_optical_depth",
        "dust",
        "uv_index",
        "cams_pm25_roll6h",
        "cams_pm25_roll24h",
        "cams_pm25_diff3h",
        "cams_pm25_diff24h",
        "cams_pm25_to_pm10",
        "cams_log_pm25",
        "cams_log_pm10",
        "cams_log_no2",
        "cams_log_so2",
        "cams_log_co",
    ]
    df = df.merge(store.aq_df[aq_cols], on=["location_id", "target_start_ist"], how="left")

    # Compute CAMS 7-day baseline at T0 per (location_id, hour) and per location_id
    aq_hist_7d = store.aq_df.loc[
        (store.aq_df["target_start_ist"] + pd.Timedelta(hours=1) <= cutoff_t0)
        & (store.aq_df["target_start_ist"] + pd.Timedelta(hours=1) > cutoff_t0 - pd.Timedelta(days=7))
    ].copy()
    aq_hist_7d["hour"] = aq_hist_7d["target_start_ist"].dt.hour
    aq_age_h = (cutoff_t0 - (aq_hist_7d["target_start_ist"] + pd.Timedelta(hours=1))).dt.total_seconds() / 3600.0
    aq_hist_7d["w_cams"] = np.exp(-np.log(2.0) * aq_age_h / (3.5 * 24.0))
    aq_hist_7d["wy_pm25"] = aq_hist_7d["w_cams"] * aq_hist_7d["pm2_5"]
    aq_hist_7d["wy_pm10"] = aq_hist_7d["w_cams"] * aq_hist_7d["pm10"]

    sh_cams_sum = aq_hist_7d.groupby(["location_id", "hour"])[["wy_pm25", "wy_pm10", "w_cams"]].sum()
    sh_cams_pm25 = (sh_cams_sum["wy_pm25"] / np.maximum(sh_cams_sum["w_cams"], 1e-6)).rename("cams_t0_7d_sh_pm25")
    sh_cams_pm10 = (sh_cams_sum["wy_pm10"] / np.maximum(sh_cams_sum["w_cams"], 1e-6)).rename("cams_t0_7d_sh_pm10")

    s_cams_sum = aq_hist_7d.groupby("location_id")[["wy_pm25", "w_cams"]].sum()
    s_cams_pm25 = (s_cams_sum["wy_pm25"] / np.maximum(s_cams_sum["w_cams"], 1e-6)).rename("cams_t0_7d_s_pm25")

    df = df.merge(sh_cams_pm25.reset_index(), on=["location_id", "hour"], how="left")
    df = df.merge(sh_cams_pm10.reset_index(), on=["location_id", "hour"], how="left")
    df = df.merge(s_cams_pm25.reset_index(), on="location_id", how="left")
    df = df.copy()

    df["cams_t0_7d_s_pm25"] = df["cams_t0_7d_s_pm25"].fillna(df["pm2_5"])
    df["cams_t0_7d_sh_pm25"] = df["cams_t0_7d_sh_pm25"].fillna(df["cams_t0_7d_s_pm25"])
    df["cams_t0_7d_sh_pm10"] = df["cams_t0_7d_sh_pm10"].fillna(df["pm10"])

    # Station-Calibrated CAMS Ratio & Offset at T0:
    # Ratio of Observed PM2.5 / CAMS PM2.5 over [T0-7d, T0]
    df["station_obs_to_cams_ratio_t0"] = np.clip(
        df["station_level_7d_mean"] / np.maximum(df["cams_t0_7d_s_pm25"], 8.0),
        0.35,
        2.5,
    )
    df["sh_obs_to_cams_ratio_t0"] = np.clip(
        df["sh_mean_7d"] / np.maximum(df["cams_t0_7d_sh_pm25"], 8.0),
        0.35,
        2.5,
    )
    df["sh_obs_minus_cams_offset_t0"] = df["sh_mean_7d"] - df["cams_t0_7d_sh_pm25"]

    # -------------------------------------------------------------------------
    # 4b. TimeMixer-Inspired Synoptic-Diurnal Decomposition + tanh-Saturating CAMS MOS
    # -------------------------------------------------------------------------
    # CAMS 7-day baseline at T0 (blending station-hour and station-level CAMS baselines)
    cams_base_blend = 0.65 * df["cams_t0_7d_sh_pm25"] + 0.35 * df["cams_t0_7d_s_pm25"]
    df["cams_log_ratio_h"] = np.log(
        np.maximum(df["pm2_5"], 5.0) / np.maximum(cams_base_blend, 8.0)
    )
    df["cams_log_ratio_24h"] = np.log(
        np.maximum(df["cams_pm25_roll24h"], 5.0) / np.maximum(df["cams_t0_7d_s_pm25"], 8.0)
    )

    # Horizon-adaptive synoptic weighting: near horizons (<=48h) rely more on 24h smoothed CAMS
    # and tighter tanh saturation to suppress transient regional dust spikes (e.g. Fold 4 Oct 18-19)
    h_long = np.clip((np.maximum(df["horizon_hours"], 0.0) - 48.0) / 48.0, 0.0, 1.0)
    w_24 = 0.28 * (1.0 - 0.65 * h_long)
    r_blend = (1.0 - w_24) * df["cams_log_ratio_h"] + w_24 * df["cams_log_ratio_24h"]

    # 1. Saturating Synoptic Level Multiplier (bounded tanh on positive log CAMS excursions)
    tau_cams = 0.60 + 0.32 * h_long
    r_sat = np.where(r_blend > 0.0, tau_cams * np.tanh(r_blend / tau_cams), r_blend)

    # 2. Hygroscopic, Boundary-Layer & Column-AOD MOS Correction:
    # Downweight CAMS upward spikes when RH > 68-75%, PBLH is elevated, or column AOD dust is aloft
    rh_exc = np.clip((df["relative_humidity_2m"] - 68.0) / 22.0, 0.0, 1.0)
    aod_exc = np.clip((df["aerosol_optical_depth"] - 0.85) / 0.35, 0.0, 1.0)
    pblh_exc = np.clip((df["boundary_layer_height"] - 480.0) / 500.0, 0.0, 1.0)
    mos_strength = 1.0 - 0.35 * h_long
    df["cams_mos_damp_factor"] = np.clip(
        1.0 - mos_strength * (0.22 * rh_exc + 0.26 * aod_exc + 0.12 * pblh_exc),
        0.45,
        1.0,
    )
    df["cams_log_ratio_mos"] = np.where(
        r_sat > 0.0, r_sat * df["cams_mos_damp_factor"], r_sat
    )

    df["cams_rel_factor_raw"] = np.clip(
        df["pm2_5"] / np.maximum(cams_base_blend, 10.0),
        0.45,
        2.2,
    )
    df["cams_rel_factor"] = np.clip(np.exp(df["cams_log_ratio_mos"]), 0.45, 2.2)
    df["cams_pm25_mos"] = np.clip(cams_base_blend * df["cams_rel_factor"], 2.0, 450.0)

    # Apply station calibration to MOS-stabilized CAMS forecast so extreme hallucinations never explode linearly
    df["cams_pm25_calibrated_station"] = df["cams_pm25_mos"] * df["station_obs_to_cams_ratio_t0"]
    df["cams_pm25_calibrated_hourly"] = df["cams_pm25_mos"] * df["sh_obs_to_cams_ratio_t0"]
    df["cams_pm25_calibrated"] = (
        0.55 * df["cams_pm25_calibrated_hourly"] + 0.45 * df["cams_pm25_calibrated_station"]
    )
    df["cams_pm25_offset_calibrated"] = np.maximum(
        2.0, df["cams_pm25_mos"] + df["sh_obs_minus_cams_offset_t0"]
    )
    df["cams_pm25_rel_change"] = df["prior_eb_diurnal"] * df["cams_rel_factor"]

    # Physics-Calibrated Composite Prior (Model 1 foundation):
    # Scales the adaptive Empirical Bayes diurnal profile by tanh-saturated CAMS MOS factor
    # and ventilation anomaly exp(-beta_vc * Delta log(1 + VC))
    vent_factor = np.clip(np.exp(-0.075 * df["wx_anom_log1p_vc"]), 0.84, 1.20)
    base_diurnal_blend = 0.55 * df["prior_eb_diurnal"] + 0.45 * df["prior_diurnal_adaptive"]
    c_exp_h = 0.42 + 0.08 * h_long
    scaled_synoptic_prior = base_diurnal_blend * (df["cams_rel_factor"] ** c_exp_h) * vent_factor
    cams_direct_mos = 0.65 * df["cams_pm25_calibrated"] + 0.35 * df["cams_pm25_offset_calibrated"]
    c_blend_h = 0.22 + 0.05 * h_long
    df["prior_physics_calibrated"] = np.clip(
        (1.0 - c_blend_h) * scaled_synoptic_prior + c_blend_h * cams_direct_mos,
        2.0,
        450.0,
    )

    # -------------------------------------------------------------------------
    # 4c. AirPhyNet Upwind Advection-Diffusion PDE Flux Features across the 4 Stations
    # -------------------------------------------------------------------------
    # Group by target_start_ist across the 4 stations to compute spatial PDE transport fluxes:
    # v_{j -> i}(t) = max(0, u_{10,j}(t) * e_{j->i, x} + v_{10,j}(t) * e_{j->i, y})
    # Phi_{adv, i}(t) = sum_{j != i} (v_{j -> i}(t) / d_ij) * (y_prior_j(t) - y_prior_i(t))
    ts_idx, ts_uniques = pd.factorize(df["target_start_ist"])
    n_ts = len(ts_uniques)
    n_st = len(STATION_ORDER)

    prior_mat = np.zeros((n_ts, n_st), dtype=np.float64)
    cams_mat = np.zeros((n_ts, n_st), dtype=np.float64)
    u_mat = np.zeros((n_ts, n_st), dtype=np.float64)
    v_mat = np.zeros((n_ts, n_st), dtype=np.float64)
    present_mat = np.zeros((n_ts, n_st), dtype=bool)

    st_indices = df["station_idx"].values
    prior_mat[ts_idx, st_indices] = df["prior_physics_calibrated"].values
    cams_mat[ts_idx, st_indices] = df["cams_pm25_calibrated"].values
    u_mat[ts_idx, st_indices] = df["u10"].values
    v_mat[ts_idx, st_indices] = df["v10"].values
    present_mat[ts_idx, st_indices] = True

    # Fill any missing station in a timestamp slice with row mean of present stations
    for t_i in range(n_ts):
        if not np.all(present_mat[t_i]) and np.any(present_mat[t_i]):
            prior_mat[t_i, ~present_mat[t_i]] = prior_mat[t_i, present_mat[t_i]].mean()
            cams_mat[t_i, ~present_mat[t_i]] = cams_mat[t_i, present_mat[t_i]].mean()
            u_mat[t_i, ~present_mat[t_i]] = u_mat[t_i, present_mat[t_i]].mean()
            v_mat[t_i, ~present_mat[t_i]] = v_mat[t_i, present_mat[t_i]].mean()

    # Wind projection v_{j -> i}(t): shape (n_ts, i, j)
    u_src = u_mat[:, None, :]  # (n_ts, 1, j)
    v_src = v_mat[:, None, :]  # (n_ts, 1, j)
    v_proj = np.maximum(
        0.0,
        u_src * _AIRPHYNET_DIR_EAST[None, :, :] + v_src * _AIRPHYNET_DIR_NORTH[None, :, :],
    )
    inv_dist = np.where(_AIRPHYNET_DIST_KM > 0.0, 1.0 / np.maximum(_AIRPHYNET_DIST_KM, 0.5), 0.0)
    adv_rate = v_proj * inv_dist[None, :, :]  # (n_ts, i, j)

    # Pairwise concentration differences: (y_j - y_i), shape (n_ts, i, j)
    diff_prior = prior_mat[:, None, :] - prior_mat[:, :, None]
    diff_cams = cams_mat[:, None, :] - cams_mat[:, :, None]

    phi_adv_prior = np.sum(adv_rate * diff_prior, axis=-1)  # (n_ts, i)
    phi_adv_cams = np.sum(adv_rate * diff_cams, axis=-1)    # (n_ts, i)

    # Symmetric Graph Laplacian diffusion: sum_{j != i} exp(-d_ij / 12.0) / d_ij * (y_j - y_i)
    diff_kernel = np.where(
        _AIRPHYNET_DIST_KM > 0.0,
        np.exp(-_AIRPHYNET_DIST_KM / 12.0) * inv_dist,
        0.0,
    )
    laplacian_diff = np.sum(diff_kernel[None, :, :] * diff_prior, axis=-1)  # (n_ts, i)
    upwind_velocity = np.sum(v_proj, axis=-1)  # (n_ts, i)

    df["airphynet_advection_flux"] = phi_adv_prior[ts_idx, st_indices]
    df["airphynet_cams_advection_flux"] = phi_adv_cams[ts_idx, st_indices]
    df["airphynet_diffusion_laplacian"] = laplacian_diff[ts_idx, st_indices]
    df["airphynet_upwind_velocity"] = upwind_velocity[ts_idx, st_indices]
    df["airphynet_pde_total_flux"] = (
        df["airphynet_advection_flux"] + 0.35 * df["airphynet_diffusion_laplacian"]
    )

    # -------------------------------------------------------------------------
    # 5. Historical Multi-Year Seasonal Climatology (2018-2025)
    # -------------------------------------------------------------------------
    df = df.merge(store.climatology_doy, on="doy", how="left")
    t0_doy = int(cutoff_t0.day_of_year)
    t0_clim_row = store.climatology_doy.loc[store.climatology_doy["doy"] == t0_doy]
    t0_clim_mean = float(t0_clim_row["clim_doy_mean"].iloc[0]) if not t0_clim_row.empty else 50.0
    df["clim_ratio_vs_t0"] = np.clip(df["clim_doy_mean"] / max(t0_clim_mean, 5.0), 0.7, 1.5)
    df["clim_scaled_prior"] = df["prior_eb_diurnal"] * (df["clim_ratio_vs_t0"] ** 0.5)

    # Clean any residual NaNs in numeric feature columns
    feature_cols = get_feature_columns(df)
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return df


NON_FEATURE_COLS: set[str] = {
    "ID",
    "timestamp_ist",
    "target_start_ist",
    "target_end_ist",
    "hour_start",
    "hour_end",
    "available_at",
    "station_name",
    "series_id",
    "scoring_block",
    "unit",
    "pm25",
    "pm25_raw",
    "pm25_clean",
    "mean",
    "count",
    "flag_high_1000",
    "is_glitch_stage1",
    "is_spike_stage2",
    "is_outlier",
    "sensor_id",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered list of numeric modeling feature columns."""
    cols: list[str] = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_multi_cutoff_training_dataset(
    train_df: pd.DataFrame,
    max_cutoff_t0: pd.Timestamp,
    n_cutoffs: int = 28,
    step_hours: int = 24,
    min_gap_hours: int = 12,
    max_horizon_hours: int = 96,
    include_oct_nov_2025: bool = True,
    store: ExternalDataStore | None = None,
) -> pd.DataFrame:
    """
    Construct a multi-cutoff historical training dataset where each historical cutoff `T_k <= max_cutoff_t0`
    predicts target windows `[T_k + min_gap_hours, T_k + max_horizon_hours]`, strictly using only
    observations up to `T_k`.

    Also includes post-monsoon Oct-Nov 2025 cutoffs so tree and neural models learn the exact
    late-October / early-November Mumbai boundary-layer and CAMS residual dynamics.
    """
    if store is None:
        store = get_external_data_store()

    cutoffs: list[pd.Timestamp] = []
    # Recent cutoffs leading up to max_cutoff_t0
    earliest_target_end = max_cutoff_t0
    latest_issue = earliest_target_end - pd.Timedelta(hours=48)
    for i in range(n_cutoffs):
        c_ts = latest_issue - pd.Timedelta(hours=i * step_hours)
        if c_ts > train_df["hour_end"].min() + pd.Timedelta(days=30):
            cutoffs.append(c_ts)

    # Add post-monsoon Oct 15 - Nov 18, 2025 cutoffs (every 48 hours) if available prior to max_cutoff_t0
    if include_oct_nov_2025:
        nov25_candidates = pd.date_range(
            start="2025-10-15 00:00:00",
            end="2025-11-16 00:00:00",
            freq="48h",
            tz=IST_TZ,
        )
        for c_ts in nov25_candidates:
            if c_ts + pd.Timedelta(hours=max_horizon_hours) <= max_cutoff_t0:
                cutoffs.append(c_ts)

    cutoffs = sorted(set(cutoffs))
    chunks: list[pd.DataFrame] = []

    for c_ts in cutoffs:
        t_start = c_ts + pd.Timedelta(hours=min_gap_hours)
        t_end = min(c_ts + pd.Timedelta(hours=max_horizon_hours), max_cutoff_t0 - pd.Timedelta(hours=1))
        if t_end <= t_start:
            continue

        target_slice = train_df.loc[
            (train_df["hour_start"] >= t_start) & (train_df["hour_start"] <= t_end)
        ].copy()
        if len(target_slice) < 24:
            continue

        hist_slice = train_df.loc[train_df["hour_end"] <= c_ts]
        feat_slice = build_features_for_cutoff(
            history_df=hist_slice,
            target_df=target_slice,
            cutoff_t0=c_ts,
            store=store,
        )
        feat_slice["cutoff_t0"] = c_ts
        # Sample weight: upweight recent October-November post-monsoon windows
        days_before_max = (max_cutoff_t0 - c_ts).total_seconds() / (3600.0 * 24.0)
        is_post_monsoon = feat_slice["hour_start"].dt.month.isin([10, 11]).astype(float)
        feat_slice["sample_weight"] = (
            np.exp(-days_before_max / 35.0) * (1.0 + 0.5 * is_post_monsoon) + 0.25 * is_post_monsoon
        )
        chunks.append(feat_slice)

    full_df = pd.concat(chunks, ignore_index=True)
    return full_df


def main() -> None:
    train, test, _, _ = load_and_align_core_data()
    store = get_external_data_store()
    cutoff_t0 = train["hour_end"].max()
    test_feats = build_features_for_cutoff(train, test, cutoff_t0=cutoff_t0, store=store)
    feat_cols = get_feature_columns(test_feats)
    print(f"Built {len(feat_cols)} leak-free features for Test ({len(test_feats):,} rows) at cutoff T0={cutoff_t0}")
    print("Sample feature columns:", feat_cols[:20])
    print("Null count across all features:", int(test_feats[feat_cols].isna().sum().sum()))


if __name__ == "__main__":
    main()
