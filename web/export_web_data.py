"""Export real project data into compact JSON payloads for the VayuDrishti 3D web twin.

Reads (never writes) `data/raw`, `data/external`, `data/processed`, `submissions/`
and the curated catalogs in `app/spatial_catalog.py`, then writes:

    web/public/data/stations.json
    web/public/data/forecasts_72h.json
    web/public/data/wind_and_plumes.json
    web/public/data/wards_and_civic.json
    web/public/data/climateverse_catalog.json

Run:
    .venv/bin/python web/export_web_data.py

Conventions follow AGENTS.md: Train/aux `timestamp_ist` is hour_end, everything is
re-indexed on hour_start in Asia/Kolkata, and analog/ratio calibration only uses
observations with hour_end <= forecast cutoff (2026-10-22 00:00 IST).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
OUT = Path(__file__).resolve().parent / "public" / "data"
TZ = "Asia/Kolkata"

sys.path.insert(0, str(ROOT))
from app.spatial_catalog import (  # noqa: E402
    AUXILIARY_STATIONS,
    BMC_24_WARDS,
    CLIMATEVERSE_DOI_CATALOG,
    EMISSION_HOTSPOTS,
    TARGET_STATIONS,
    VULNERABLE_FACILITIES,
)

TARGET_IDS = [11611, 3409511, 11606, 3409510]
CUTOFF = pd.Timestamp("2026-10-22 00:00:00", tz=TZ)
TIMELINE_START = pd.Timestamp("2026-10-12 00:00:00", tz=TZ)
TIMELINE_END = pd.Timestamp("2026-11-20 23:00:00", tz=TZ)
NAAQS_24H = 60.0
WHO_24H = 15.0

MODEL_COLUMNS = {
    "m1": ("m1_calibrated_cams_prior", "Calibrated CAMS MOS Prior"),
    "m2": ("m2_huber_ridge_gam", "Huber-Ridge Physics GAM"),
    "m3": ("m3_lightgbm_residual", "LightGBM Residual"),
    "m4": ("m4_catboost_residual", "CatBoost Residual"),
    "m5": ("m5_xgboost_residual", "XGBoost Residual"),
    "m6": ("m6_wind_st_gat", "AirPhyNet Wind-ST-GAT"),
    "m7": ("m7_timesfm3_compact", "TimesFM 3.0 Compact Multi-View"),
    "m8": ("m8_chronos2_covariate", "Chronos-2 Multivariate Covariates"),
}
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES_21 = [0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]

# Approximate BMC municipal limits (lon, lat), clockwise from Colaba. Used only to clip
# Voronoi ward cells into a recognisable peninsula footprint.
MUMBAI_OUTLINE = [
    (72.8150, 18.8930), (72.8350, 18.9150), (72.8450, 18.9400), (72.8500, 18.9600),
    (72.8580, 18.9850), (72.8620, 19.0050), (72.8750, 19.0200), (72.8880, 19.0060),
    (72.9050, 18.9950), (72.9350, 18.9950), (72.9550, 19.0200), (72.9450, 19.0500),
    (72.9450, 19.0750), (72.9560, 19.1000), (72.9640, 19.1300), (72.9760, 19.1600),
    (72.9850, 19.1850), (72.9550, 19.1950), (72.9300, 19.2000), (72.8950, 19.2250),
    (72.8850, 19.2700), (72.8450, 19.2750), (72.8200, 19.2600), (72.8050, 19.2300),
    (72.8100, 19.2000), (72.8150, 19.1700), (72.8200, 19.1400), (72.8250, 19.1000),
    (72.8260, 19.0700), (72.8200, 19.0450), (72.8100, 19.0200), (72.8150, 19.0000),
    (72.8100, 18.9750), (72.7950, 18.9550), (72.8120, 18.9480), (72.8180, 18.9300),
    (72.8200, 18.9100),
]

MAJOR_BMC_HOSPITALS = [
    ("KEM Hospital, Parel", 19.0029, 72.8424, "F/S"),
    ("LTMG Sion Hospital", 19.0395, 72.8610, "F/N"),
    ("BYL Nair Hospital", 18.9726, 72.8197, "E"),
    ("HBT Cooper Hospital, Juhu", 19.1076, 72.8366, "K/W"),
    ("Rajawadi Hospital, Ghatkopar", 19.0790, 72.9020, "N"),
    ("Bhabha Hospital, Bandra", 19.0590, 72.8380, "H/W"),
    ("Shatabdi Hospital, Kandivali", 19.2065, 72.8530, "R/S"),
    ("MT Agarwal Hospital, Mulund", 19.1740, 72.9550, "T"),
    ("Siddharth Hospital, Goregaon", 19.1640, 72.8490, "P/S"),
    ("Trauma Care Hospital, Jogeshwari", 19.1380, 72.8570, "K/E"),
    ("Kasturba Hospital, Chinchpokli", 18.9840, 72.8290, "E"),
    ("VN Desai Hospital, Santacruz", 19.0820, 72.8460, "H/E"),
]

HIGHWAYS = [
    {
        "name": "Western Express Highway",
        "doi": "doi:10.71646/EV3RBJ",
        "strength": 0.65,
        "path": [[72.842, 19.055], [72.852, 19.100], [72.855, 19.120], [72.858, 19.165],
                 [72.860, 19.185], [72.864, 19.230], [72.866, 19.255]],
    },
    {
        "name": "Eastern Express Highway",
        "doi": "doi:10.71646/EV3RBJ",
        "strength": 0.62,
        "path": [[72.865, 19.040], [72.890, 19.065], [72.915, 19.085], [72.930, 19.110],
                 [72.940, 19.145], [72.955, 19.175]],
    },
    {
        "name": "Mumbai Port / JNPT Freight Corridor",
        "doi": "doi:10.71646/EREDWQ",
        "strength": 0.72,
        "path": [[72.845, 18.950], [72.855, 18.985], [72.860, 19.005], [72.900, 18.990],
                 [72.950, 18.960]],
    },
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def r1(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else round(f, 1)


def r2(x: Any) -> float | None:
    v = r1(x)
    return None if v is None else round(float(x), 2)


def series_list(s: pd.Series, fn=r1) -> list[float | None]:
    return [fn(v) for v in s.to_numpy()]


def iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S+05:30")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def to_ist(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)


def write_json(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


def clean_pm25_local(df: pd.DataFrame) -> pd.DataFrame:
    """Same 2-stage rule as src/validation.clean_pm25 (cap 450, 7h Hampel @ 4.5*1.4826*MAD)."""
    try:
        from src.validation import clean_pm25

        return clean_pm25(df)
    except Exception as exc:  # pragma: no cover - fallback keeps the exporter standalone
        print(f"  [warn] src.validation.clean_pm25 unavailable ({exc}); using local copy")
    out = df.sort_values(["location_id", "hour_start"]).reset_index(drop=True).copy()
    raw = pd.to_numeric(out["pm25"], errors="coerce").astype(float)
    out["pm25_raw"] = raw
    s1 = out["flag_high_1000"].fillna(False).astype(bool) | (raw > 450) | (raw < 0)
    out["is_glitch_stage1"] = s1
    clean = raw.where(~s1)
    out["pm25_clean"] = np.nan
    out["is_spike_stage2"] = False
    for _, idx in out.groupby("location_id").groups.items():
        s = clean.loc[idx]
        med = s.rolling(7, center=True, min_periods=3).median()
        mad = (s - med).abs().rolling(7, center=True, min_periods=3).median()
        spike = (s - med).abs() > 4.5 * 1.4826 * mad.clip(lower=1.0)
        s2 = s.where(~spike, med).interpolate(limit_direction="both").clip(upper=450)
        out.loc[idx, "pm25_clean"] = s2
        out.loc[idx, "is_spike_stage2"] = spike.fillna(False) & ~s1.loc[idx]
    out["is_outlier"] = out["is_glitch_stage1"] | out["is_spike_stage2"]
    return out


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_train() -> pd.DataFrame:
    tr = pd.read_csv(RAW / "Train.csv")
    tr["hour_end"] = to_ist(tr["timestamp_ist"])
    tr["hour_start"] = tr["hour_end"] - pd.Timedelta(hours=1)
    return clean_pm25_local(tr)


def load_aux() -> pd.DataFrame:
    ax = pd.read_csv(RAW / "additional_history" / "Other_Mumbai_Stations_2025_2026.csv.gz")
    ax["hour_end"] = to_ist(ax["timestamp_ist"])
    ax["hour_start"] = ax["hour_end"] - pd.Timedelta(hours=1)
    ax["pm25"] = pd.to_numeric(ax["mean"], errors="coerce")
    ax = ax[(ax["pm25"] >= 0) & (ax["pm25"] <= 450) & ~ax["flag_high_1000"].fillna(False).astype(bool)]
    return ax


def load_weather() -> pd.DataFrame:
    w = pd.read_parquet(EXT / "open_meteo_weather.parquet")
    w["hour_start"] = to_ist(w["target_start_ist"])
    w["wind_ms"] = w["wind_speed_10m"].astype(float) / 3.6  # Open-Meteo default unit is km/h
    rad = np.deg2rad(w["wind_direction_10m"].astype(float))
    w["u10"] = -w["wind_ms"] * np.sin(rad)
    w["v10"] = -w["wind_ms"] * np.cos(rad)
    return w


def load_cams() -> pd.DataFrame:
    a = pd.read_parquet(EXT / "open_meteo_air_quality.parquet")
    a["hour_start"] = to_ist(a["target_start_ist"])
    return a[["hour_start", "location_id", "pm2_5", "pm10", "nitrogen_dioxide", "aerosol_optical_depth"]]


def load_vabb() -> pd.DataFrame:
    v = pd.read_parquet(EXT / "vabb_metar_hourly.parquet")
    v["hour_start"] = to_ist(v["target_start_ist"])
    kt = 0.514444
    v["u_ms"] = v["vabb_u_kt"] * kt
    v["v_ms"] = v["vabb_v_kt"] * kt
    # Only tmpc/dwpc/relh/drct/sknt/alti-derived columns are allowed (no visibility / wxcodes).
    return v[["hour_start", "vabb_tmpc", "vabb_relh", "vabb_drct", "vabb_wind_speed_ms", "u_ms", "v_ms"]]


def load_test_predictions() -> pd.DataFrame:
    t = pd.read_csv(SUB / "test_predictions_detailed.csv")
    t["hour_start"] = to_ist(t["hour_start"])
    return t


def load_oof() -> pd.DataFrame:
    o = pd.read_csv(PROC / "oof_predictions_gap24h.csv")
    o["hour_start"] = to_ist(o["hour_start"])
    return o


# ---------------------------------------------------------------------------
# quantile calibration (split-conformal on OOF log-residuals)
# ---------------------------------------------------------------------------
def residual_quantiles(oof_path: Path, levels: list[float]) -> dict[int, np.ndarray]:
    o = pd.read_csv(oof_path)
    out: dict[int, np.ndarray] = {}
    pooled = np.log1p(o["pm25"].clip(lower=0)) - np.log1p(o["m7_stacked_ensemble"].clip(lower=0))
    for lid, g in o.groupby("location_id"):
        r = np.log1p(g["pm25"].clip(lower=0)) - np.log1p(g["m7_stacked_ensemble"].clip(lower=0))
        # Shrink each station's quantiles 30% toward the pooled quantiles (n≈220/station).
        q_st = np.quantile(r.dropna(), levels)
        q_pool = np.quantile(pooled.dropna(), levels)
        out[int(lid)] = 0.7 * q_st + 0.3 * q_pool
    return out


def trapezoid_mean(levels: list[float], values: np.ndarray) -> np.ndarray:
    """E[Y] from quantiles: integrate Q(p) dp with flat tails beyond the extreme levels."""
    lv = np.asarray(levels)
    vals = np.asarray(values)  # (..., L)
    mid = 0.5 * (vals[..., 1:] + vals[..., :-1]) * np.diff(lv)
    return mid.sum(-1) + vals[..., 0] * lv[0] + vals[..., -1] * (1 - lv[-1])


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def build() -> None:
    print("Loading datasets …")
    train = load_train()
    aux = load_aux()
    weather = load_weather()
    cams = load_cams()
    vabb = load_vabb()
    test = load_test_predictions()
    oof = load_oof()
    weights24 = json.loads((PROC / "ensemble_weights_gap24h.json").read_text())
    weights72 = json.loads((PROC / "ensemble_weights_gap72h.json").read_text())
    lb24 = pd.read_csv(PROC / "validation_leaderboard_gap24h.csv")
    lb72 = pd.read_csv(PROC / "validation_leaderboard_gap72h.csv")
    test_meta = pd.read_csv(RAW / "Test.csv")
    test_meta["hour_start"] = to_ist(test_meta["target_start_ist"])

    hours = pd.date_range(TIMELINE_START, TIMELINE_END, freq="h")
    H = len(hours)
    hour_index = pd.DatetimeIndex(hours)

    # scoring block per hour (validation replay < cutoff; test blocks after)
    blk = test_meta.groupby("hour_start")["scoring_block"].first()
    blocks = []
    for h in hours:
        if h < CUTOFF:
            blocks.append("validation")
        else:
            blocks.append(str(blk.get(h, "gap")))
    horizon = [max(0, int((h - CUTOFF) / pd.Timedelta(hours=1)) + 1) if h >= CUTOFF else 0 for h in hours]

    q9_24 = residual_quantiles(PROC / "oof_predictions_gap24h.csv", DECILES)
    q9_72 = residual_quantiles(PROC / "oof_predictions_gap72h.csv", DECILES)
    q21_24 = residual_quantiles(PROC / "oof_predictions_gap24h.csv", QUANTILES_21)
    q21_72 = residual_quantiles(PROC / "oof_predictions_gap72h.csv", QUANTILES_21)

    def horizon_blend(q24: np.ndarray, q72: np.ndarray, h: int) -> np.ndarray:
        if h <= 24:
            return q24
        t = min(1.0, (h - 24) / 48.0)
        base = q24 + (q72 - q24) * t
        return base * min(1.6, math.sqrt(max(1.0, h / 72.0)))

    station_payload: dict[str, Any] = {}
    glitch_payload: dict[str, Any] = {}
    stations_meta = {s["location_id"]: s for s in TARGET_STATIONS}

    for lid in TARGET_IDS:
        tr_s = train[train["location_id"] == lid].set_index("hour_start").sort_index()
        tr_s = tr_s[~tr_s.index.duplicated()]
        obs_raw = tr_s["pm25_raw"].reindex(hour_index)
        obs_clean = tr_s["pm25_clean"].reindex(hour_index)

        te = test[test["location_id"] == lid].set_index("hour_start").sort_index()
        te = te[~te.index.duplicated()]
        oo = oof[oof["location_id"] == lid].set_index("hour_start").sort_index()
        oo = oo[~oo.index.duplicated()]

        ens = te["Target"].reindex(hour_index).combine_first(oo["m7_stacked_ensemble"].reindex(hour_index))
        models = {"ensemble": series_list(ens)}
        for key, (col, _) in MODEL_COLUMNS.items():
            s = te[col].reindex(hour_index).combine_first(oo[col].reindex(hour_index))
            models[key] = series_list(s)

        # 9-decile & 21-quantile bands
        dec_rows, q21_rows, mean9, mean21, med = [], [], [], [], []
        for i, h in enumerate(hours):
            y = ens.iloc[i]
            if not np.isfinite(y):
                dec_rows.append(None)
                q21_rows.append(None)
                mean9.append(None)
                mean21.append(None)
                med.append(None)
                continue
            hz = horizon[i] if horizon[i] > 0 else 24
            d = np.expm1(np.log1p(max(y, 0)) + horizon_blend(q9_24[lid], q9_72[lid], hz)).clip(min=0)
            q = np.expm1(np.log1p(max(y, 0)) + horizon_blend(q21_24[lid], q21_72[lid], hz)).clip(min=0)
            d.sort()
            q.sort()
            dec_rows.append([r1(v) for v in d])
            q21_rows.append([r1(v) for v in q])
            mean9.append(r1(trapezoid_mean(DECILES, d)))
            mean21.append(r1(trapezoid_mean(QUANTILES_21, q)))
            med.append(r1(d[4]))

        models["timesfm_mean"] = mean9
        models["chronos_mean"] = mean21

        # TimeMixer-style two-scale view of the forecast trajectory
        ens_f = ens.interpolate(limit_area="inside")
        level = ens_f.rolling(24, center=True, min_periods=12).mean()
        diurnal = ens_f - level

        # CAMS raw
        cm = cams[cams["location_id"] == lid].set_index("hour_start")
        cm = cm[~cm.index.duplicated()]

        station_payload[str(lid)] = {
            "obs": series_list(obs_raw),
            "obs_clean": series_list(obs_clean),
            "models": models,
            "deciles": dec_rows,
            "q21": q21_rows,
            "median": med,
            "level": series_list(level),
            "diurnal": series_list(diurnal),
            "cams": series_list(cm["pm2_5"].reindex(hour_index)),
        }

        # Hampel audit
        events = tr_s[tr_s["is_outlier"]].copy()
        events["stage"] = np.where(events["is_glitch_stage1"], 1, 2)
        events["delta"] = (events["pm25_raw"] - events["pm25_clean"]).abs()
        top = events.sort_values("delta", ascending=False).head(40)
        ev_list = []
        for ts, row in top.iterrows():
            win = tr_s.loc[ts - pd.Timedelta(hours=18): ts + pd.Timedelta(hours=18)]
            ev_list.append({
                "hour_start": iso(ts),
                "raw": r1(row["pm25_raw"]),
                "clean": r1(row["pm25_clean"]),
                "stage": int(row["stage"]),
                "flag_high_1000": bool(row.get("flag_high_1000", False)),
                "context": {
                    "hours": [iso(t) for t in win.index],
                    "raw": series_list(win["pm25_raw"]),
                    "clean": series_list(win["pm25_clean"]),
                },
            })
        glitch_payload[str(lid)] = {
            "n_rows": int(len(tr_s)),
            "n_stage1": int(tr_s["is_glitch_stage1"].sum()),
            "n_stage2": int(tr_s["is_spike_stage2"].sum()),
            "max_raw": r1(tr_s["pm25_raw"].max()),
            "events": ev_list,
        }

    # ---------------- AnEn analogs ----------------
    print("Computing historical meteorological analogs …")
    analogs = compute_analogs(train, weather, hours)

    model_names = {"ensemble": "8-Model Stacked Ensemble", **{k: v[1] for k, v in MODEL_COLUMNS.items()},
                   "timesfm_mean": "TimesFM 3.0 · 9-decile trapezoidal mean",
                   "chronos_mean": "Chronos-2 · 21-quantile L2 mean"}
    lb_rows = []
    for _, row in lb24.iterrows():
        r72 = lb72[lb72["model_key"] == row["model_key"]]
        lb_rows.append({
            "key": row["model_key"],
            "name": row["Model"].split(". ", 1)[-1],
            "rmse24": r2(row["OOF_RMSE"]),
            "rmse72": r2(r72["OOF_RMSE"].iloc[0]) if len(r72) else None,
            "folds24": [r2(row[c]) for c in lb24.columns if c.startswith("Fold")],
        })

    windows = []
    for name in ["validation", "public", "unscored", "private"]:
        idx = [i for i, b in enumerate(blocks) if b == name]
        if idx:
            windows.append({"id": name, "start": idx[0], "end": idx[-1], "n": len(idx)})

    forecasts = {
        "meta": {
            "generated_at": iso(pd.Timestamp.now(tz=TZ).floor("s")),
            "cutoff_t0": iso(CUTOFF),
            "timezone": TZ,
            "hour_convention": "hour_start (Test.target_start_ist); Train hour_end shifted -1h",
            "deciles": DECILES,
            "quantiles21": QUANTILES_21,
            "model_names": model_names,
            "ensemble_weights": {"gap24h": weights24, "gap72h": weights72},
            "leaderboard": lb_rows,
            "provenance": {
                "ensemble_and_m1_m6": "submissions/test_predictions_detailed.csv (test) + data/processed/oof_predictions_gap24h.csv (Oct 12–21 validation replay)",
                "bands": "Split-conformal: station-shrunk empirical quantiles of OOF log-residuals (gap24h/gap72h), horizon-blended. Slots for TimesFM 3.0 (9 deciles) and Chronos-2 (21 quantiles); mean = trapezoidal ∫Q(p)dp.",
                "timemixer": "Two-scale view of the ensemble trajectory: 24h centred mean L_s(t) + residual D_s(t) (display decomposition, not a training feature).",
                "hampel": "src/validation.clean_pm25 (Stage 1: flag_high_1000 or >450 µg/m³; Stage 2: 7h Hampel 4.5·1.4826·MAD).",
                "analogs": "Daily Open-Meteo meteorology signature vs. post-monsoon/winter days with hour_end <= cutoff.",
            },
        },
        "hours": [iso(h) for h in hours],
        "blocks": blocks,
        "horizon": horizon,
        "windows": windows,
        "stations": station_payload,
        "hampel": glitch_payload,
        "analogs": analogs,
    }

    # ---------------- wind / PBLH / plumes ----------------
    print("Building wind field & plume payload …")
    wind_payload, mean_wind = build_wind(weather, vabb, hour_index)

    # ---------------- stations (18) ----------------
    print("Building stations payload …")
    stations_json = build_stations(train, aux, station_payload, hour_index, weather)

    # AirPhyNet graph topology
    all_st = stations_json["stations"]
    edges = []
    for a in all_st:
        for b in all_st:
            if a["id"] == b["id"]:
                continue
            d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if d > 14.0:
                continue
            brg = bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
            # time-mean advective alignment: wind blowing TO bearing a->b
            align = max(0.0, math.cos(math.radians(brg - mean_wind["to_deg"])))
            edges.append({"src": a["id"], "dst": b["id"], "km": round(d, 2), "bearing": round(brg, 1),
                          "mean_alignment": round(align, 3)})
    wind_payload["airphynet"] = {
        "edges": edges,
        "formula": "flux_ij(t) = C_i(t) · max(0, cos(θ_wind_to(t) − bearing_ij)) · |U(t)| / d_ij  −  κ·(C_i − C_j)/d_ij²",
        "kappa": 0.15,
    }
    wind_payload["plume_sources"] = [
        {"name": e["name"], "category": e["category"], "doi": e["doi"], "lat": e["lat"], "lon": e["lon"],
         "ward": e["ward_code"], "strength": e["emission_strength"], "description": e["description"]}
        for e in EMISSION_HOTSPOTS
    ]
    wind_payload["line_sources"] = HIGHWAYS

    # ---------------- wards ----------------
    print("Building wards & civic payload …")
    st_full = [
        {**s, "series_full": station_payload[str(s["id"])]["models"]["ensemble"] if s["role"] == "target" else s["series"]}
        for s in all_st
    ]
    wards_json = build_wards(st_full, hour_index, blocks, wind_payload)

    # ---------------- catalog ----------------
    catalog = build_catalog()

    print("Writing JSON …")
    write_json("forecasts_72h.json", forecasts)
    write_json("stations.json", stations_json)
    write_json("wind_and_plumes.json", wind_payload)
    write_json("wards_and_civic.json", wards_json)
    write_json("climateverse_catalog.json", catalog)

    # sanity summary
    priv = [i for i, b in enumerate(blocks) if b == "private"]
    for lid in TARGET_IDS:
        e = [v for i, v in enumerate(station_payload[str(lid)]["models"]["ensemble"]) if i in priv and v is not None]
        print(f"  {lid}: private-window ensemble hours={len(e)} mean={np.mean(e):.1f}")
    print("Done.")


def compute_analogs(train: pd.DataFrame, weather: pd.DataFrame, hours: pd.DatetimeIndex) -> dict[str, Any]:
    feats = ["wind_ms", "boundary_layer_height", "relative_humidity_2m", "temperature_2m", "u10", "v10",
             "shortwave_radiation"]
    w = weather.copy()
    w["day"] = w["hour_start"].dt.normalize()
    daily = w.groupby(["location_id", "day"])[feats].mean()
    daily["pblh_min"] = w.groupby(["location_id", "day"])["boundary_layer_height"].min()
    cols = feats + ["pblh_min"]

    tr = train[train["hour_end"] <= CUTOFF].copy()
    tr["day"] = tr["hour_start"].dt.normalize()
    tr["hod"] = tr["hour_start"].dt.hour
    obs_day = tr.groupby(["location_id", "day"])["pm25_clean"].agg(["mean", "count"])

    target_days = sorted({h.normalize() for h in hours if h >= CUTOFF})
    result: dict[str, Any] = {}
    for lid in TARGET_IDS:
        d = daily.loc[lid]
        mu, sd = d[cols].mean(), d[cols].std().replace(0, 1)
        z = (d[cols] - mu) / sd
        od = obs_day.loc[lid] if lid in obs_day.index.get_level_values(0) else pd.DataFrame()
        cand_days = [
            day for day in z.index
            if day + pd.Timedelta(days=1) <= CUTOFF
            and day.month in (10, 11, 12, 1)
            and day in od.index and od.loc[day, "count"] >= 20
        ]
        if not cand_days:
            continue
        zc = z.loc[cand_days]
        per_day = {}
        for tday in target_days:
            if tday not in z.index:
                continue
            dist = np.sqrt(((zc - z.loc[tday]) ** 2).mean(axis=1))
            best = dist.nsmallest(3)
            items = []
            for aday, dv in best.items():
                curve = tr[(tr["location_id"] == lid) & (tr["day"] == aday)].set_index("hod")["pm25_clean"]
                curve = curve[~curve.index.duplicated()].reindex(range(24)).interpolate(limit_direction="both")
                met = daily.loc[(lid, aday)]
                items.append({
                    "date": aday.strftime("%Y-%m-%d"),
                    "similarity": round(float(math.exp(-dv)), 3),
                    "distance": round(float(dv), 3),
                    "pm25_mean": r1(od.loc[aday, "mean"]),
                    "curve": series_list(curve),
                    "wind_ms": r1(met["wind_ms"]),
                    "pblh_m": r1(met["boundary_layer_height"]),
                    "rh": r1(met["relative_humidity_2m"]),
                })
            wsum = sum(i["similarity"] for i in items)
            anen = [
                r1(sum(i["similarity"] * (i["curve"][hh] or 0) for i in items) / max(wsum, 1e-9))
                for hh in range(24)
            ]
            tm = daily.loc[(lid, tday)]
            per_day[tday.strftime("%Y-%m-%d")] = {
                "target_met": {"wind_ms": r1(tm["wind_ms"]), "pblh_m": r1(tm["boundary_layer_height"]),
                               "rh": r1(tm["relative_humidity_2m"]), "temp_c": r1(tm["temperature_2m"])},
                "analogs": items,
                "anen_curve": anen,
            }
        result[str(lid)] = per_day
    return result


def build_wind(weather: pd.DataFrame, vabb: pd.DataFrame, hour_index: pd.DatetimeIndex):
    points: dict[str, Any] = {}
    pblh_stack, ws_stack, u_stack, v_stack, sw_stack = [], [], [], [], []
    for lid in TARGET_IDS:
        g = weather[weather["location_id"] == lid].set_index("hour_start").sort_index()
        g = g[~g.index.duplicated()].reindex(hour_index)
        meta = next(s for s in TARGET_STATIONS if s["location_id"] == lid)
        points[str(lid)] = {
            "lat": meta["latitude"], "lon": meta["longitude"],
            "u": series_list(g["u10"], r2), "v": series_list(g["v10"], r2),
            "pblh": series_list(g["boundary_layer_height"]),
            "temp": series_list(g["temperature_2m"]),
            "rh": series_list(g["relative_humidity_2m"]),
            "sw": series_list(g["shortwave_radiation"]),
        }
        pblh_stack.append(g["boundary_layer_height"].to_numpy(float))
        ws_stack.append(g["wind_ms"].to_numpy(float))
        u_stack.append(g["u10"].to_numpy(float))
        v_stack.append(g["v10"].to_numpy(float))
        sw_stack.append(g["shortwave_radiation"].to_numpy(float))

    vb = vabb.set_index("hour_start").sort_index()
    vb = vb[~vb.index.duplicated()].reindex(hour_index)
    points["VABB"] = {"lat": 19.0887, "lon": 72.8679, "u": series_list(vb["u_ms"], r2),
                      "v": series_list(vb["v_ms"], r2), "observed": True}

    pblh = np.nanmean(np.vstack(pblh_stack), axis=0)
    ws = np.nanmean(np.vstack(ws_stack), axis=0)
    u = np.nanmean(np.vstack(u_stack), axis=0)
    v = np.nanmean(np.vstack(v_stack), axis=0)
    sw = np.nanmean(np.vstack(sw_stack), axis=0)
    dir_from = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    onshore = ws * np.cos(np.deg2rad(dir_from - 285.0))
    deonar = ws * np.maximum(0, np.cos(np.deg2rad(dir_from - 75.0)))
    vc = pblh * ws

    def regime(d, spd):
        if not np.isfinite(d):
            return "unknown"
        if spd < 1.0:
            return "stagnant"
        if 230 <= d <= 330:
            return "sea_breeze"
        if 30 <= d <= 145:
            return "land_breeze"
        return "transitional"

    mean_to = (math.degrees(math.atan2(np.nanmean(u), np.nanmean(v))) + 360) % 360
    payload = {
        "meta": {
            "units": {"u": "m/s (toward east)", "v": "m/s (toward north)", "pblh": "m", "vc": "m²/s", "sw": "W/m²"},
            "source": "Open-Meteo NWP/ERA5 (doi:10.71646/XXOBKO) at 4 target stations + VABB METAR (doi:10.71646/EAE4EE) u/v only",
            "coast_normal_from_deg": 285, "deonar_axis_from_deg": 75, "thane_creek_axis_from_deg": 110,
        },
        "points": points,
        "city": {
            "pblh": series_list(pd.Series(pblh)),
            "wind_ms": series_list(pd.Series(ws), r2),
            "dir_from": series_list(pd.Series(dir_from)),
            "u": series_list(pd.Series(u), r2),
            "v": series_list(pd.Series(v), r2),
            "vc": [None if not np.isfinite(x) else int(round(x)) for x in vc],
            "onshore": series_list(pd.Series(onshore), r2),
            "deonar_transport": series_list(pd.Series(deonar), r2),
            "sw": series_list(pd.Series(sw)),
            "regime": [regime(d, s) for d, s in zip(dir_from, ws)],
        },
    }
    return payload, {"to_deg": mean_to}


def idw(lat: float, lon: float, pts: list[tuple[float, float, float]], power: float = 2.0) -> float:
    num = den = 0.0
    for plat, plon, val in pts:
        if val is None or not np.isfinite(val):
            continue
        d = max(0.3, haversine_km(lat, lon, plat, plon))
        w = 1.0 / d ** power
        num += w * val
        den += w
    return num / den if den else float("nan")


def build_stations(train, aux, station_payload, hour_index, weather):
    tgt_meta = {s["location_id"]: s for s in TARGET_STATIONS}
    ward_lookup = {w["ward_code"]: w for w in BMC_24_WARDS}
    tr_hist = train[train["hour_end"] <= CUTOFF]

    def diurnal_profile(df: pd.DataFrame, col: str) -> list[float | None]:
        recent = df[df["hour_start"] >= CUTOFF - pd.Timedelta(days=120)]
        src = recent if len(recent) > 200 else df
        return series_list(src.groupby(src["hour_start"].dt.hour)[col].median().reindex(range(24)))

    stations = []
    tgt_obs_wide = (tr_hist.pivot_table(index="hour_start", columns="location_id", values="pm25_clean"))

    for lid in TARGET_IDS:
        m = tgt_meta[lid]
        g = tr_hist[tr_hist["location_id"] == lid]
        nov25 = g[(g["hour_start"] >= pd.Timestamp("2025-10-15", tz=TZ)) & (g["hour_start"] < pd.Timestamp("2025-12-01", tz=TZ))]
        stations.append({
            "id": lid, "name": m["station_name"], "short": m["short_name"], "role": "target",
            "operator": m["operator"], "lat": m["latitude"], "lon": m["longitude"],
            "ward": m["ward_code"], "regime": m["micro_regime"],
            "morphology": {
                "alphaearth_roughness": m["alphaearth_roughness"],
                "alphaearth_green_idx": m["alphaearth_green_idx"],
                "open_buildings_density": m["open_buildings_density"],
                "coastal_dist_km": m["coastal_dist_km"], "deonar_dist_km": m["deonar_dist_km"],
            },
            "stats": {
                "n_obs": int(len(g)),
                "mean_all": r1(g["pm25_clean"].mean()),
                "mean_postmonsoon_2025": r1(nov25["pm25_clean"].mean()),
                "p95_postmonsoon_2025": r1(nov25["pm25_clean"].quantile(0.95)),
                "last_obs": iso(g["hour_start"].max()),
            },
            "diurnal": diurnal_profile(g, "pm25_clean"),
            "series_source": "forecasts_72h.json",
        })

    # auxiliary: ratio-calibrated downscaling of target ensemble via IDW
    tgt_coords = {lid: (tgt_meta[lid]["latitude"], tgt_meta[lid]["longitude"]) for lid in TARGET_IDS}
    ens = {lid: np.array([np.nan if v is None else v for v in station_payload[str(lid)]["models"]["ensemble"]])
           for lid in TARGET_IDS}
    aux_hist = aux[aux["hour_end"] <= CUTOFF]
    for a in AUXILIARY_STATIONS:
        lid = a["location_id"]
        g = aux_hist[aux_hist["location_id"] == lid].set_index("hour_start")
        g = g[~g.index.duplicated()]
        # calibration ratio: aux obs vs. IDW of simultaneous target obs
        joined = tgt_obs_wide.reindex(g.index)
        idw_obs = []
        for ts, row in joined.iterrows():
            idw_obs.append(idw(a["latitude"], a["longitude"],
                               [(*tgt_coords[t], row.get(t, np.nan)) for t in TARGET_IDS]))
        idw_obs = pd.Series(idw_obs, index=g.index)
        ok = (idw_obs > 3) & g["pm25"].notna()
        ratio = float(np.clip(np.median(g.loc[ok, "pm25"] / idw_obs[ok]), 0.5, 2.2)) if ok.sum() >= 30 else 1.0
        est = []
        for i in range(len(hour_index)):
            v = idw(a["latitude"], a["longitude"], [(*tgt_coords[t], ens[t][i]) for t in TARGET_IDS])
            est.append(r1(v * ratio) if np.isfinite(v) else None)
        obs = g["pm25"].reindex(hour_index)
        w = ward_lookup.get(a["ward_code"], {})
        stations.append({
            "id": lid, "name": a["station_name"], "short": a["short_name"], "role": "auxiliary",
            "operator": a["operator"], "lat": a["latitude"], "lon": a["longitude"], "ward": a["ward_code"],
            "morphology": {
                "alphaearth_roughness": w.get("alphaearth_impervious"),
                "open_buildings_density": w.get("open_bldg_density"),
            },
            "stats": {
                "n_obs": int(len(g)),
                "mean_all": r1(g["pm25"].mean()) if len(g) else None,
                "ratio_to_idw_targets": round(ratio, 3),
                "calibration_pairs": int(ok.sum()),
                "last_obs": iso(g.index.max()) if len(g) else None,
            },
            "diurnal": diurnal_profile(g.reset_index(), "pm25") if len(g) > 48 else None,
            "series": est,
            "obs": series_list(obs),
        })
    return {
        "meta": {
            "n_target": 4, "n_auxiliary": len(AUXILIARY_STATIONS),
            "aux_series_method": "IDW(4 target ensemble forecasts) × median(aux_obs / IDW(target_obs)) over 2025–2026 overlap",
            "morphology_source": "Curated AlphaEarth (doi:10.71646/GHSZ7J) / Open Buildings V3 (doi:10.71646/EV3RBJ) priors from app/spatial_catalog.py",
        },
        "stations": stations,
    }


# ----- ward geometry -----
def clip_polygon(subject: list[tuple[float, float]], clip: list[tuple[float, float]]):
    """Sutherland–Hodgman: clip (possibly concave) subject by a convex CCW clip polygon."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def inter(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-15:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(inter(s, e, a, b))
            s = e
    return out


def polygon_area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi:
            inside = not inside
        j = i
    return inside


def ward_polygons() -> dict[str, list[list[float]]]:
    pts = np.array([[w["lon"], w["lat"]] for w in BMC_24_WARDS])
    far = np.array([[72.0, 18.0], [74.0, 18.0], [74.0, 20.0], [72.0, 20.0]])
    vor = Voronoi(np.vstack([pts, far]))
    coast = MUMBAI_OUTLINE[::-1] if polygon_area(MUMBAI_OUTLINE) < 0 else MUMBAI_OUTLINE
    polys = {}
    for i, w in enumerate(BMC_24_WARDS):
        region = vor.regions[vor.point_region[i]]
        cell = [tuple(vor.vertices[k]) for k in region if k >= 0]
        if polygon_area(cell) < 0:
            cell = cell[::-1]
        clipped = clip_polygon(list(coast), cell)
        ring = [[round(x, 5), round(y, 5)] for x, y in clipped]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        polys[w["ward_code"]] = ring
    return polys


def build_wards(all_st, hour_index, blocks, wind_payload):
    from app.gemini_copilot import generate_deterministic_grap_briefing

    polys = ward_polygons()
    H = len(hour_index)
    rng = np.random.default_rng(42)
    wards = []
    private_idx = [i for i, b in enumerate(blocks) if b == "private"]
    city = wind_payload["city"]
    facilities = []
    for f in VULNERABLE_FACILITIES:
        facilities.append({
            "id": f["facility_id"], "name": f["facility_name"],
            "type": "school" if "School" in f["type"] else "hospital",
            "doi": f["doi"], "ward": f["ward_code"], "lat": f["lat"], "lon": f["lon"],
            "daily_population": f["daily_population"], "tier": f["vulnerability_tier"],
            "protocol": f["recommended_protocol"], "indicative": False,
        })
    for name, lat, lon, wc in MAJOR_BMC_HOSPITALS:
        facilities.append({"id": f"BMC-MAJOR-{wc}-{len(facilities)}", "name": name, "type": "hospital",
                           "doi": "doi:10.71646/1XSUVR", "ward": wc, "lat": lat, "lon": lon,
                           "tier": "Tertiary civic hospital", "indicative": False})

    station_pts = [(s["lat"], s["lon"], s) for s in all_st]
    for w in BMC_24_WARDS:
        code = w["ward_code"]
        ring = polys[code]
        # IDW over all 18 station series (targets use their ensemble via `series_full`)
        vals = []
        for i in range(H):
            pts = []
            for lat, lon, s in station_pts:
                pts.append((lat, lon, s["series_full"][i]))
            v = idw(w["lat"], w["lon"], pts, power=2.0)
            canyon = 1.0 + 0.08 * (0.5 * w["alphaearth_impervious"] + 0.5 * w["open_bldg_density"] - 0.75)
            vals.append(r1(v * canyon) if np.isfinite(v) else None)
        arr = np.array([np.nan if v is None else v for v in vals])
        priv = arr[private_idx]
        priv_mean = float(np.nanmean(priv)) if np.isfinite(priv).any() else float("nan")
        # daily (24h) exceedance days over the forecast horizon
        s = pd.Series(arr, index=hour_index)
        daily = s[s.index >= CUTOFF].resample("D").mean().dropna()
        vuln_pop = int(w["pop_2026"] * (w["vuln_children_pct"] + w["vuln_elderly_pct"]) / 100.0)
        peak_i = int(private_idx[int(np.nanargmax(priv))]) if np.isfinite(priv).any() else private_idx[0]
        peak_pm = float(arr[peak_i])
        stage = grap_stage(priv_mean)
        ward_row = {**w, "pm25_ugm3": round(priv_mean, 1), "naaqs_exceed_x": round(priv_mean / NAAQS_24H, 2),
                    "who_exceed_x": round(priv_mean / WHO_24H, 1), "grap_stage": stage, "vulnerable_pop": vuln_pop}
        snap = {"timestamp_ist": iso(hour_index[peak_i]),
                "pblh_m": city["pblh"][peak_i] or 0.0,
                "ventilation_coeff": city["vc"][peak_i] or 0.0,
                "wind_speed_ms": city["wind_ms"][peak_i] or 0.0,
                "wind_dir_from_deg": city["dir_from"][peak_i] or 0.0}
        fac_in = [{"facility_name": f["name"], "type": f["type"], "doi": f["doi"],
                   "pm25_ugm3": priv_mean, "who_multiple": priv_mean / WHO_24H,
                   "recommended_protocol": f.get("protocol", "Hourly telemetry watch.")}
                  for f in facilities if f["ward"] == code]
        directive = generate_deterministic_grap_briefing(ward_row, snap, fac_in)

        # density-sampled school pins (count from catalog, 1 pin per 12 schools)
        n_pins = max(2, int(round(w["udise_schools"] / 12)))
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        placed, tries = 0, 0
        while placed < n_pins and tries < 4000 and len(ring) > 3:
            tries += 1
            x = rng.uniform(min(xs), max(xs))
            y = rng.uniform(min(ys), max(ys))
            if point_in_poly(x, y, ring):
                facilities.append({"id": f"UDISE-DENS-{code}-{placed}", "name": f"UDISE+ school cluster · Ward {code}",
                                   "type": "school", "doi": "doi:10.71646/XJIY8W", "ward": code,
                                   "lat": round(y, 5), "lon": round(x, 5), "indicative": True,
                                   "represents": 12})
                placed += 1

        wards.append({
            "code": code, "name": w["ward_name"], "zone": w["zone"], "lat": w["lat"], "lon": w["lon"],
            "polygon": ring, "pop_2026": w["pop_2026"], "vulnerable_pop": vuln_pop,
            "children_pct": w["vuln_children_pct"], "elderly_pct": w["vuln_elderly_pct"],
            "slum_pct": w["slum_pct"], "maharera_sites": w["maharera_sites"],
            "alphaearth_impervious": w["alphaearth_impervious"], "open_bldg_density": w["open_bldg_density"],
            "udise_schools": w["udise_schools"], "civic_hospitals": w["civic_hospitals"],
            "series": vals,
            "private_mean": r1(priv_mean), "private_peak": r1(peak_pm), "private_peak_at": iso(hour_index[peak_i]),
            "naaqs_x": r2(priv_mean / NAAQS_24H), "who_x": r1(priv_mean / WHO_24H),
            "naaqs_exceed_days": int((daily > NAAQS_24H).sum()), "who_exceed_days": int((daily > WHO_24H).sum()),
            "forecast_days": int(len(daily)),
            "grap_stage": stage,
            "directive_md": directive,
        })
    return {
        "meta": {
            "naaqs_24h": NAAQS_24H, "who_24h": WHO_24H,
            "geometry": "Approximate: Voronoi cells of ward-office centroids clipped to a simplified BMC outline (doi:10.71646/EREDWQ for official polygons).",
            "population": "WorldPop 2026 ward aggregates (doi:10.71646/AAGK0G) via app/spatial_catalog.py",
            "ward_series": "IDW of 18 station series × AlphaEarth/Open-Buildings canyon factor",
            "grap_engine": "app/gemini_copilot.generate_deterministic_grap_briefing at the private-window peak hour",
            "school_pins": "Named facilities are curated; 'indicative' school pins are density-sampled (1 pin ≈ 12 UDISE+ schools).",
            "grap_thresholds": [
                {"stage": "Watch", "min": 0, "max": 60}, {"stage": "Stage I", "min": 60, "max": 90},
                {"stage": "Stage II", "min": 90, "max": 120}, {"stage": "Stage III", "min": 120, "max": 9999},
            ],
        },
        "wards": wards,
        "facilities": facilities,
        "outline": [[x, y] for x, y in MUMBAI_OUTLINE] + [[MUMBAI_OUTLINE[0][0], MUMBAI_OUTLINE[0][1]]],
    }


def grap_stage(pm: float) -> str:
    if not np.isfinite(pm):
        return "Unknown"
    if pm >= 120:
        return "Stage III (Severe+)"
    if pm >= 90:
        return "Stage II (Very Poor)"
    if pm >= 60:
        return "Stage I (Poor / Above NAAQS)"
    return "Watch (Moderate / Within NAAQS)"


CATEGORY_RULES = [
    ("Air Quality & Chemistry", ["air quality", "pm2", "aerosol", "cams", "pollut", "no2", "ozone", "tropomi", "sentinel-5", "emission", "methane"]),
    ("Meteorology & Climate", ["era5", "weather", "meteo", "rain", "precip", "temperature", "wind", "imd", "metar", "monsoon", "cyclone", "climate"]),
    ("Satellite & Earth Observation", ["satellite", "landsat", "sentinel", "modis", "viirs", "alphaearth", "embedding", "nightlight", "night light", "firms", "fire"]),
    ("Urban Form & Infrastructure", ["building", "road", "osm", "openstreetmap", "infrastructure", "construction", "rera", "land use", "land cover", "urban"]),
    ("Population & Health", ["population", "worldpop", "census", "health", "hospital", "school", "udise", "mortality", "demograph"]),
    ("Water & Coast", ["water", "ocean", "coast", "flood", "river", "groundwater", "sea"]),
    ("Land & Ecology", ["soil", "vegetation", "forest", "ndvi", "crop", "agricultur", "biodiversity", "landslide"]),
    ("Energy & Economy", ["energy", "power", "electric", "economic", "gdp", "industry", "industrial"]),
]


def categorize(text: str) -> str:
    t = text.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in t for k in keys):
            return cat
    return "Other"


def build_catalog():
    raw = json.loads((EXT / "climateverse_catalog.json").read_text())
    items = raw["data"]["items"]
    curated = {c["doi"]: c for c in CLIMATEVERSE_DOI_CATALOG}
    out = []
    seen = set()
    for it in items:
        doi = it.get("global_id")
        cur = curated.get(doi, {})
        desc = (it.get("description") or "").strip()
        out.append({
            "doi": doi, "name": it.get("name"), "url": it.get("url") or f"https://doi.org/{doi[4:]}",
            "published_at": it.get("published_at"), "publisher": it.get("publisher"),
            "subjects": it.get("subjects", []), "file_count": it.get("fileCount"),
            "category": categorize(f"{it.get('name', '')} {desc[:300]}"),
            "curated_category": cur.get("category"),
            "description": desc[:420], "integrated": bool(cur), "role": cur.get("role"),
        })
        seen.add(doi)
    for doi, cur in curated.items():
        if doi in seen:
            continue
        out.append({"doi": doi, "name": cur["title"], "url": cur["url"], "published_at": None,
                    "publisher": cur["provider"], "subjects": [], "file_count": None,
                    "category": categorize(f"{cur['title']} {cur['category']}"), "curated_category": cur["category"],
                    "description": cur["resolution"], "integrated": True,
                    "role": cur["role"]})
    out.sort(key=lambda d: (not d["integrated"], d["category"], d["name"] or ""))
    return {"meta": {"source": "ClimateVerse India Dataverse search API (data/external/climateverse_catalog.json) + curated roles",
                     "count": len(out), "integrated": sum(d["integrated"] for d in out)},
            "datasets": out}


if __name__ == "__main__":
    build()
