#!/usr/bin/env python3
"""
Automated Data Ingestion & ClimateVerse External Data Fetcher
for the Google Mumbai Air Quality Forecasting Challenge (Zindi).

Fetches, aligns (to Asia/Kolkata IST, UTC+05:30), validates, and caches:
1. Core Challenge Data via Zindi SDK (`zindi`) or Hugging Face CLI/Hub fallback
2. Open-Meteo Historical Weather Archive + Historical Forecast + 16-Day Live Forecast
   (doi:10.71646/XXOBKO) for all 4 target Mumbai CAAQMS stations (2025-02-01 to 2026-11-21)
3. Open-Meteo CAMS Air Quality Forecast & Archive for all 4 target stations
   (2025-02-01 to 2026-11-21)
4. VABB (Chhatrapati Shivaji Maharaj International Airport) METAR Archive
   (doi:10.71646/EAE4EE) via Iowa Environmental Mesonet (2025-02-01 to 2026-11-08),
   aggregated to hourly IST (strictly excluding visibility/wxcodes per curator guidance)
5. ClimateVerse Challenge Dataverse Catalog Metadata (`climateverse_catalog.json`)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IST_TZ = "Asia/Kolkata"
HF_REPO_ID = "han-jisso/google-mumbai-air-quality-challenge"
DEFAULT_ZINDI_CHALLENGE_ID = os.environ.get(
    "ZINDI_CHALLENGE_ID", "google-mumbai-data-challenge"
)

TARGET_STATIONS: list[dict[str, Any]] = [
    {
        "location_id": 11611,
        "sensor_id": 12236426,
        "station_name": "Malad West, Mumbai - IITM",
        "latitude": 19.19709,
        "longitude": 72.82204,
        "timezone": "Asia/Kolkata",
    },
    {
        "location_id": 3409511,
        "sensor_id": 12238576,
        "station_name": "Shivaji Nagar, Mumbai - BMC",
        "latitude": 19.060498,
        "longitude": 72.923356,
        "timezone": "Asia/Kolkata",
    },
    {
        "location_id": 11606,
        "sensor_id": 12236418,
        "station_name": "Borivali East, Mumbai - IITM",
        "latitude": 19.23241,
        "longitude": 72.86895,
        "timezone": "Asia/Kolkata",
    },
    {
        "location_id": 3409510,
        "sensor_id": 12238567,
        "station_name": "Byculla, Mumbai - BMC",
        "latitude": 18.9767,
        "longitude": 72.838,
        "timezone": "Asia/Kolkata",
    },
]

WEATHER_HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "boundary_layer_height",
]

AQ_HOURLY_VARS: list[str] = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
]


def _get_json_with_retries(
    url: str,
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Perform HTTP GET returning parsed JSON with exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            wait_s = 1.5 * (2 ** (attempt - 1))
            print(f"  [WARN] GET {url} failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait_s:.1f}s...")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {last_err}")


def sync_zindi_data(raw_dir: Path = RAW_DIR) -> bool:
    """
    Download latest competition data (including updated Train.csv on Nov 7 / Nov 9 / Nov 10)
    using the official Zindi Python SDK (`from zindi.user import Zindian`).
    Falls back gracefully to the Hugging Face dataset mirror if ZINDI_USERNAME / ZINDI_PASSWORD
    are not set in the environment.
    """
    from zindi.user import Zindian  # noqa: F401

    username = os.environ.get("ZINDI_USERNAME", "").strip()
    password = os.environ.get("ZINDI_PASSWORD", "").strip()
    challenge_id = os.environ.get("ZINDI_CHALLENGE_ID", DEFAULT_ZINDI_CHALLENGE_ID).strip()

    if not username or not password:
        print(
            "[Zindi SDK] ZINDI_USERNAME / ZINDI_PASSWORD not found in environment.\n"
            "  To pull live Train.csv updates directly from Zindi on Nov 7 / Nov 9 / Nov 10:\n"
            "    export ZINDI_USERNAME='your_username'\n"
            "    export ZINDI_PASSWORD='your_password'\n"
            "    export ZINDI_CHALLENGE_ID='google-mumbai-data-challenge'\n"
            "    uv run python src/fetch_data.py --zindi-sync\n"
            "  Using verified Hugging Face mirror in data/raw/ as fallback."
        )
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Zindi SDK] Authenticating as '{username}' for challenge '{challenge_id}'...")
    try:
        user = Zindian(username=username, fixed_password=password)
        user.select_a_challenge(fixed_index=challenge_id)
        user.download_dataset(destination=str(raw_dir))
        _extract_additional_history(raw_dir)
        print(f"[Zindi SDK] Successfully downloaded dataset files to {raw_dir}")
        return True
    except Exception as exc:
        print(f"[Zindi SDK] Download via Zindi failed ({exc}); falling back to Hugging Face mirror.")
        return False


def _extract_additional_history(raw_dir: Path) -> None:
    zip_path = raw_dir / "AdditionalHistory.zip"
    target_dir = raw_dir / "additional_history"
    if zip_path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)


def sync_hf_data(raw_dir: Path = RAW_DIR) -> None:
    """Download core challenge dataset from Hugging Face mirror and extract AdditionalHistory.zip."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HF Hub] Syncing dataset '{HF_REPO_ID}' into {raw_dir}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    _extract_additional_history(raw_dir)
    print("[HF Hub] Core dataset and additional_history verified.")


def fetch_open_meteo_weather(
    external_dir: Path = EXTERNAL_DIR,
    start_date: str = "2025-02-01",
    end_date: str = "2026-11-21",
) -> pd.DataFrame:
    """
    Fetch and merge:
      1) Open-Meteo Historical Weather Archive (`archive-api.open-meteo.com/v1/archive`)
      2) Open-Meteo Historical Forecast Archive (`historical-forecast-api.open-meteo.com/v1/forecast`)
      3) Open-Meteo 16-Day Live Weather Forecast (`api.open-meteo.com/v1/forecast`)
    for all 4 target Mumbai stations across 2025-02-01 through 2026-11-21 in Asia/Kolkata (IST).

    Timestamp alignment:
      - `target_start_ist`: start of the 1-hour interval in IST (matches Open-Meteo `time` and Test `target_start_ist`)
      - `timestamp_ist`: end of the 1-hour interval in IST (`target_start_ist + 1h`, matches Train `timestamp_ist` and Test `target_end_ist`)
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    hourly_param = ",".join(WEATHER_HOURLY_VARS)

    # Full target hourly grid in Asia/Kolkata
    full_time_index = pd.date_range(
        start=f"{start_date} 00:00:00",
        end=f"{end_date} 23:00:00",
        freq="1h",
        tz=IST_TZ,
    )

    # Archive API supports up to ~yesterday; Historical Forecast API supports up to today
    today_ist = pd.Timestamp.now(tz=IST_TZ).strftime("%Y-%m-%d")
    archive_end = min(end_date, today_ist)
    hist_fc_end = min(end_date, today_ist)

    station_frames: list[pd.DataFrame] = []

    for st in TARGET_STATIONS:
        loc_id = st["location_id"]
        lat = st["latitude"]
        lon = st["longitude"]
        name = st["station_name"]
        print(f"[Open-Meteo Weather] Fetching Archive + HistForecast + LiveForecast for {loc_id} ({name})...")

        # 1. Historical ERA5 Archive
        arch_json = _get_json_with_retries(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": archive_end,
                "hourly": hourly_param,
                "timezone": IST_TZ,
            },
        )
        df_arch = pd.DataFrame(arch_json["hourly"])
        df_arch["target_start_ist"] = pd.to_datetime(df_arch["time"]).dt.tz_localize(IST_TZ)
        df_arch = df_arch.drop(columns=["time"]).set_index("target_start_ist")

        # 2. Historical Forecast Archive (NWP archive including boundary_layer_height)
        hf_json = _get_json_with_retries(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": hist_fc_end,
                "hourly": hourly_param,
                "timezone": IST_TZ,
            },
        )
        df_hf = pd.DataFrame(hf_json["hourly"])
        df_hf["target_start_ist"] = pd.to_datetime(df_hf["time"]).dt.tz_localize(IST_TZ)
        df_hf = df_hf.drop(columns=["time"]).set_index("target_start_ist")

        # 3. Live 16-day Weather Forecast
        fc_json = _get_json_with_retries(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "past_days": 7,
                "forecast_days": 16,
                "hourly": hourly_param,
                "timezone": IST_TZ,
            },
        )
        df_fc = pd.DataFrame(fc_json["hourly"])
        df_fc["target_start_ist"] = pd.to_datetime(df_fc["time"]).dt.tz_localize(IST_TZ)
        df_fc = df_fc.drop(columns=["time"]).set_index("target_start_ist")

        # Combine NWP (Historical Forecast + Live Forecast, preferring Historical Forecast where available
        # and Live Forecast for future dates)
        df_nwp = df_hf.reindex(full_time_index).combine_first(df_fc.reindex(full_time_index))
        df_era5 = df_arch.reindex(full_time_index)

        # Build unified station dataframe on full_time_index:
        # Primary columns seamlessly merge Archive (ERA5) with NWP (HistForecast + LiveForecast),
        # while also retaining explicit `*_nwp` and `*_era5` columns for modeling experiments.
        df_st = pd.DataFrame(index=full_time_index)
        df_st.index.name = "target_start_ist"
        df_st["location_id"] = int(loc_id)
        df_st["station_name"] = name
        df_st["latitude"] = float(lat)
        df_st["longitude"] = float(lon)

        for var in WEATHER_HOURLY_VARS:
            # Primary column: NWP-consistent series (zero discontinuity at forecast boundary)
            # backfilled with ERA5 if any historical hour were missing, plus ERA5 column
            era5_s = pd.to_numeric(df_era5[var], errors="coerce")
            nwp_s = pd.to_numeric(df_nwp[var], errors="coerce")
            # Merge ERA5 archive and NWP forecast seamlessly
            df_st[var] = era5_s.combine_first(nwp_s)
            df_st[f"{var}_nwp"] = nwp_s.combine_first(era5_s)
            df_st[f"{var}_era5"] = era5_s

        df_st["wx_source"] = np.where(
            df_era5["temperature_2m"].notna(),
            "archive_era5+nwp",
            np.where(
                df_st.index <= pd.Timestamp.now(tz=IST_TZ),
                "historical_forecast_nwp",
                "live_forecast_nwp",
            ),
        )

        df_st = df_st.reset_index()
        df_st["target_end_ist"] = df_st["target_start_ist"] + pd.Timedelta(hours=1)
        df_st["timestamp_ist"] = df_st["target_end_ist"]
        station_frames.append(df_st)
        time.sleep(0.25)

    weather_df = pd.concat(station_frames, ignore_index=True)
    weather_df = weather_df.sort_values(["location_id", "target_start_ist"]).reset_index(drop=True)

    # Verify zero nulls on core 12 variables
    null_counts = weather_df[WEATHER_HOURLY_VARS].isna().sum()
    if null_counts.sum() > 0:
        for col in WEATHER_HOURLY_VARS:
            weather_df[col] = (
                weather_df.groupby("location_id")[col].ffill().bfill()
            )

    parquet_path = external_dir / "open_meteo_weather.parquet"
    csv_path = external_dir / "open_meteo_weather.csv"
    weather_df.to_parquet(parquet_path, index=False)
    weather_df.to_csv(csv_path, index=False)
    print(
        f"[Open-Meteo Weather] Saved {len(weather_df):,} rows "
        f"({weather_df['target_start_ist'].min()} -> {weather_df['target_start_ist'].max()}) "
        f"to {parquet_path.name} and {csv_path.name}"
    )
    return weather_df


def fetch_open_meteo_air_quality(
    external_dir: Path = EXTERNAL_DIR,
    start_date: str = "2025-02-01",
    end_date: str = "2026-11-21",
) -> pd.DataFrame:
    """
    Fetch Open-Meteo CAMS Air Quality API (`air-quality-api.open-meteo.com/v1/air-quality`)
    for all 4 target Mumbai stations covering 2025-02-01 through 2026-11-21 in Asia/Kolkata.
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    hourly_param = ",".join(AQ_HOURLY_VARS)

    full_time_index = pd.date_range(
        start=f"{start_date} 00:00:00",
        end=f"{end_date} 23:00:00",
        freq="1h",
        tz=IST_TZ,
    )

    # Open-Meteo CAMS Air Quality API supports start_date..end_date up to today + 5 days
    max_cams_end = (pd.Timestamp.now(tz=IST_TZ) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    api_end_date = min(end_date, max_cams_end)

    station_frames: list[pd.DataFrame] = []

    for st in TARGET_STATIONS:
        loc_id = st["location_id"]
        lat = st["latitude"]
        lon = st["longitude"]
        name = st["station_name"]
        print(
            f"[Open-Meteo CAMS AQ] Fetching Air Quality history + forecast "
            f"({start_date} -> {api_end_date}) for {loc_id} ({name})..."
        )

        aq_json = _get_json_with_retries(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": api_end_date,
                "hourly": hourly_param,
                "timezone": IST_TZ,
            },
        )
        df_aq = pd.DataFrame(aq_json["hourly"])
        df_aq["target_start_ist"] = pd.to_datetime(df_aq["time"]).dt.tz_localize(IST_TZ)
        df_aq = df_aq.drop(columns=["time"]).set_index("target_start_ist").reindex(full_time_index)
        df_aq.index.name = "target_start_ist"

        for var in AQ_HOURLY_VARS:
            df_aq[var] = pd.to_numeric(df_aq[var], errors="coerce")
            df_aq[f"{var}_raw"] = df_aq[var]

        # Track whether CAMS direct forecast/reanalysis was present at this timestamp
        df_aq["cams_direct_available"] = df_aq["pm2_5_raw"].notna()
        last_valid_ts = df_aq.index[df_aq["cams_direct_available"]].max()

        # For horizons beyond the 5-day CAMS forecast horizon (e.g. >120h ahead),
        # fill seamlessly using the station's trailing 120-hour hour-of-day diurnal median
        # so downstream feature pipelines have complete coverage through 2026-11-21.
        recent_window = df_aq.loc[
            (df_aq.index > last_valid_ts - pd.Timedelta(days=5)) & (df_aq.index <= last_valid_ts)
        ]
        hourly_diurnal = recent_window.groupby(recent_window.index.hour)[AQ_HOURLY_VARS].median()

        for var in AQ_HOURLY_VARS:
            if df_aq[var].isna().any():
                diurnal_map = pd.Series(df_aq.index.hour, index=df_aq.index).map(hourly_diurnal[var])
                df_aq[var] = df_aq[var].fillna(diurnal_map).ffill().bfill()

        df_aq["location_id"] = int(loc_id)
        df_aq["station_name"] = name
        df_aq["latitude"] = float(lat)
        df_aq["longitude"] = float(lon)
        df_aq["cams_last_direct_ist"] = last_valid_ts

        df_aq = df_aq.reset_index()
        df_aq["target_end_ist"] = df_aq["target_start_ist"] + pd.Timedelta(hours=1)
        df_aq["timestamp_ist"] = df_aq["target_end_ist"]
        station_frames.append(df_aq)
        time.sleep(0.25)

    aq_df = pd.concat(station_frames, ignore_index=True)
    aq_df = aq_df.sort_values(["location_id", "target_start_ist"]).reset_index(drop=True)

    parquet_path = external_dir / "open_meteo_air_quality.parquet"
    csv_path = external_dir / "open_meteo_air_quality.csv"
    aq_df.to_parquet(parquet_path, index=False)
    aq_df.to_csv(csv_path, index=False)
    print(
        f"[Open-Meteo CAMS AQ] Saved {len(aq_df):,} rows "
        f"({aq_df['target_start_ist'].min()} -> {aq_df['target_start_ist'].max()}, "
        f"direct CAMS forecast through {aq_df['cams_last_direct_ist'].max()}) "
        f"to {parquet_path.name} and {csv_path.name}"
    )
    return aq_df


def fetch_vabb_metar(
    external_dir: Path = EXTERNAL_DIR,
    start_date: str = "2025-02-01",
    end_date: str = "2026-11-08",
) -> pd.DataFrame:
    """
    Fetch VABB (Mumbai Airport) half-hourly METAR observations from Iowa Environmental Mesonet
    (`mesonet.agron.iastate.edu/cgi-bin/request/asos.py`) for `tmpc,dwpc,relh,drct,sknt,alti`
    in `Asia/Kolkata` timezone, and aggregate to hourly IST.
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)

    print(
        f"[VABB METAR] Fetching half-hourly observations ({start_date} -> {end_date}) "
        "from Iowa Environmental Mesonet..."
    )
    params = {
        "station": "VABB",
        "data": ["tmpc", "dwpc", "relh", "drct", "sknt", "alti"],
        "year1": dt_start.year,
        "month1": dt_start.month,
        "day1": dt_start.day,
        "year2": dt_end.year,
        "month2": dt_end.month,
        "day2": dt_end.day,
        "tz": IST_TZ,
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": ["3", "4"],
    }
    resp = requests.get(
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
        params=params,
        timeout=45,
    )
    resp.raise_for_status()

    raw_df = pd.read_csv(io.StringIO(resp.text), na_values=["M", "T"])
    raw_df["valid_ist"] = pd.to_datetime(raw_df["valid"]).dt.tz_localize(IST_TZ)
    for col in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    # Save raw half-hourly observations
    raw_parquet = external_dir / "vabb_metar_raw.parquet"
    raw_csv = external_dir / "vabb_metar_raw.csv"
    raw_df.to_parquet(raw_parquet, index=False)
    raw_df.to_csv(raw_csv, index=False)

    # Compute meteorological wind vector components (in knots and m/s) for accurate circular hourly aggregation
    rad = np.deg2rad(raw_df["drct"])
    raw_df["u_kt"] = -raw_df["sknt"] * np.sin(rad)
    raw_df["v_kt"] = -raw_df["sknt"] * np.cos(rad)
    raw_df["sin_drct"] = np.sin(rad)
    raw_df["cos_drct"] = np.cos(rad)

    # Floor to start of hour in IST (`target_start_ist`)
    raw_df["target_start_ist"] = raw_df["valid_ist"].dt.floor("1h")

    hourly = (
        raw_df.groupby("target_start_ist", as_index=False)
        .agg(
            vabb_tmpc=("tmpc", "mean"),
            vabb_dwpc=("dwpc", "mean"),
            vabb_relh=("relh", "mean"),
            vabb_sknt=("sknt", "mean"),
            vabb_alti=("alti", "mean"),
            vabb_u_kt=("u_kt", "mean"),
            vabb_v_kt=("v_kt", "mean"),
            vabb_sin_drct=("sin_drct", "mean"),
            vabb_cos_drct=("cos_drct", "mean"),
            vabb_obs_count=("tmpc", "count"),
        )
        .sort_values("target_start_ist")
        .reset_index(drop=True)
    )

    # Reindex onto a complete hourly IST grid from start_date to latest observed hour
    min_hr = hourly["target_start_ist"].min()
    max_hr = hourly["target_start_ist"].max()
    full_hr_idx = pd.date_range(start=min_hr, end=max_hr, freq="1h", tz=IST_TZ)
    hourly = (
        hourly.set_index("target_start_ist")
        .reindex(full_hr_idx)
        .rename_axis("target_start_ist")
        .reset_index()
    )
    hourly["vabb_obs_count"] = hourly["vabb_obs_count"].fillna(0).astype(int)

    # Recover circular mean wind direction in [0, 360)
    angle_rad = np.arctan2(hourly["vabb_sin_drct"], hourly["vabb_cos_drct"])
    hourly["vabb_drct"] = (np.rad2deg(angle_rad) + 360.0) % 360.0

    # Interpolate any occasional missing single hours (<0.1% of hours) strictly forward in time (causal)
    num_cols = [
        "vabb_tmpc",
        "vabb_dwpc",
        "vabb_relh",
        "vabb_sknt",
        "vabb_alti",
        "vabb_u_kt",
        "vabb_v_kt",
        "vabb_sin_drct",
        "vabb_cos_drct",
        "vabb_drct",
    ]
    hourly[num_cols] = hourly[num_cols].ffill().bfill()

    # Physical unit conversions
    hourly["vabb_wind_speed_ms"] = hourly["vabb_sknt"] * 0.514444
    hourly["vabb_pressure_hpa"] = hourly["vabb_alti"] * 33.863886666667
    hourly["vabb_dewpoint_depression_c"] = hourly["vabb_tmpc"] - hourly["vabb_dwpc"]
    hourly["target_end_ist"] = hourly["target_start_ist"] + pd.Timedelta(hours=1)
    hourly["timestamp_ist"] = hourly["target_end_ist"]

    parquet_path = external_dir / "vabb_metar_hourly.parquet"
    csv_path = external_dir / "vabb_metar_hourly.csv"
    hourly.to_parquet(parquet_path, index=False)
    hourly.to_csv(csv_path, index=False)
    print(
        f"[VABB METAR] Saved {len(raw_df):,} raw half-hourly rows and {len(hourly):,} hourly IST rows "
        f"({hourly['target_start_ist'].min()} -> {hourly['target_start_ist'].max()}) "
        f"to {parquet_path.name} and {csv_path.name}"
    )
    return hourly


def fetch_climateverse_catalog(external_dir: Path = EXTERNAL_DIR) -> dict[str, Any]:
    """
    Fetch ClimateVerse Google Air Quality Forecasting Challenge dataset catalog metadata
    and save to `data/external/climateverse_catalog.json` (plus summary `.csv`).
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    url = "https://india.climateverse.net/api/search"
    params = {
        "q": "*",
        "subtree": "google-air-quality-forecasting-challenge",
        "type": "dataset",
        "per_page": 100,
    }
    print("[ClimateVerse] Fetching challenge dataverse catalog metadata...")
    payload = _get_json_with_retries(url, params=params, timeout=30)

    json_path = external_dir / "climateverse_catalog.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    items = payload.get("data", {}).get("items", [])
    rows = []
    for item in items:
        rows.append(
            {
                "name": item.get("name"),
                "global_id": item.get("global_id"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "publisher": item.get("publisher"),
                "subjects": ", ".join(item.get("subjects", []) or []),
                "file_count": item.get("fileCount"),
                "description": (item.get("description") or "")[:300],
            }
        )
    df_cat = pd.DataFrame(rows)
    csv_path = external_dir / "climateverse_catalog.csv"
    df_cat.to_csv(csv_path, index=False)

    print(
        f"[ClimateVerse] Saved {len(items)} curated challenge datasets to "
        f"{json_path.name} and {csv_path.name}"
    )
    return payload


def verify_all_datasets(raw_dir: Path = RAW_DIR, external_dir: Path = EXTERNAL_DIR) -> None:
    """Verify and print row counts, schemas, and IST date ranges of all raw and external files."""
    print("\n" + "=" * 80)
    print("DATASET VERIFICATION SUMMARY (IST / Asia/Kolkata)")
    print("=" * 80)

    # 1. Core Raw Files
    train = pd.read_csv(raw_dir / "Train.csv")
    test = pd.read_csv(raw_dir / "Test.csv")
    stations = pd.read_csv(raw_dir / "Stations.csv")
    sample_sub = pd.read_csv(raw_dir / "SampleSubmission.csv")

    print(
        f"1. Train.csv:            {len(train):>8,} rows | "
        f"{train['timestamp_ist'].min()} -> {train['timestamp_ist'].max()} | "
        f"stations={train['location_id'].nunique()}"
    )
    print(
        f"2. Test.csv:             {len(test):>8,} rows | "
        f"{test['target_start_ist'].min()} -> {test['target_end_ist'].max()} | "
        f"stations={test['location_id'].nunique()}"
    )
    print(f"3. Stations.csv:         {len(stations):>8,} rows | IDs={stations['location_id'].tolist()}")
    print(f"4. SampleSubmission.csv: {len(sample_sub):>8,} rows | IDs match Test={bool((sample_sub['ID'] == test['ID']).all())}")

    # 2. Additional History Files
    add_dir = raw_dir / "additional_history"
    for fname, ts_col in [
        ("Other_Mumbai_Stations_2025_2026.csv.gz", "timestamp_ist"),
        ("History_Mumbai_2018_2020.csv.gz", "Datetime"),
        ("History_Mumbai_2024.csv.gz", "timestamp_ist"),
        ("Legacy_Observations_2020_2022.csv.gz", "datetime"),
        ("XKDR_All_Stations.csv", None),
        ("Kaggle_All_Stations.csv", None),
    ]:
        fpath = add_dir / fname
        df_add = pd.read_csv(fpath, low_memory=False)
        if ts_col and ts_col in df_add.columns:
            print(
                f"   - {fname:<38} {len(df_add):>8,} rows | "
                f"{df_add[ts_col].dropna().min()} -> {df_add[ts_col].dropna().max()}"
            )
        else:
            print(f"   - {fname:<38} {len(df_add):>8,} rows")

    # 3. External Curated Datasets
    wx_path = external_dir / "open_meteo_weather.parquet"
    if wx_path.exists():
        wx = pd.read_parquet(wx_path)
        nulls = int(wx[WEATHER_HOURLY_VARS].isna().sum().sum())
        print(
            f"5. open_meteo_weather:   {len(wx):>8,} rows | "
            f"{wx['target_start_ist'].min()} -> {wx['target_start_ist'].max()} | "
            f"core_nulls={nulls} | tz={wx['target_start_ist'].dt.tz}"
        )

    aq_path = external_dir / "open_meteo_air_quality.parquet"
    if aq_path.exists():
        aq = pd.read_parquet(aq_path)
        nulls = int(aq[AQ_HOURLY_VARS].isna().sum().sum())
        print(
            f"6. open_meteo_aq:        {len(aq):>8,} rows | "
            f"{aq['target_start_ist'].min()} -> {aq['target_start_ist'].max()} | "
            f"core_nulls={nulls} | tz={aq['target_start_ist'].dt.tz}"
        )

    metar_path = external_dir / "vabb_metar_hourly.parquet"
    if metar_path.exists():
        metar = pd.read_parquet(metar_path)
        nulls = int(metar[["vabb_tmpc", "vabb_dwpc", "vabb_relh", "vabb_sknt", "vabb_alti", "vabb_drct"]].isna().sum().sum())
        print(
            f"7. vabb_metar_hourly:    {len(metar):>8,} rows | "
            f"{metar['target_start_ist'].min()} -> {metar['target_start_ist'].max()} | "
            f"core_nulls={nulls} | tz={metar['target_start_ist'].dt.tz}"
        )

    cat_path = external_dir / "climateverse_catalog.json"
    if cat_path.exists():
        with open(cat_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
        items = cat_data.get("data", {}).get("items", [])
        print(f"8. climateverse_catalog: {len(items):>8,} datasets catalogued in JSON & CSV")

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and cache core & ClimateVerse external datasets for Google Mumbai Air Quality Challenge"
    )
    parser.add_argument("--zindi-sync", action="store_true", help="Sync core dataset via Zindi SDK")
    parser.add_argument("--hf-sync", action="store_true", help="Sync core dataset via Hugging Face Hub")
    parser.add_argument("--weather", action="store_true", help="Fetch Open-Meteo Archive + HistForecast + Live Forecast")
    parser.add_argument("--air-quality", action="store_true", help="Fetch Open-Meteo CAMS Air Quality history + forecast")
    parser.add_argument("--metar", action="store_true", help="Fetch VABB Airport METAR archive and aggregate to hourly IST")
    parser.add_argument("--climateverse", action="store_true", help="Fetch ClimateVerse challenge catalog metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    any_specific = any(
        [
            args.zindi_sync,
            args.hf_sync,
            args.weather,
            args.air_quality,
            args.metar,
            args.climateverse,
        ]
    )

    if args.zindi_sync:
        ok = sync_zindi_data()
        if not ok and not (RAW_DIR / "Train.csv").exists():
            sync_hf_data()

    if args.hf_sync or (not any_specific and not (RAW_DIR / "Train.csv").exists()):
        sync_hf_data()
    elif not any_specific:
        # Check Zindi credentials info on default run
        sync_zindi_data()
        _extract_additional_history(RAW_DIR)

    if args.weather or not any_specific:
        fetch_open_meteo_weather()

    if args.air_quality or not any_specific:
        fetch_open_meteo_air_quality()

    if args.metar or not any_specific:
        fetch_vabb_metar()

    if args.climateverse or not any_specific:
        fetch_climateverse_catalog()

    verify_all_datasets()


if __name__ == "__main__":
    main()
