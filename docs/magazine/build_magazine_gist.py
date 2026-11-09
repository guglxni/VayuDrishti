#!/usr/bin/env python3
"""Build the VayuDrishti research-magazine plates, monograph, and HTML edition.

Numbers on the plates come from the committed leaderboards, the ensemble-weight
JSON, and (when the local cache is present) hour-of-day means of Train.csv,
Open-Meteo CAMS, and boundary-layer height for 1–21 October 2026.

    uv run python docs/magazine/build_magazine_gist.py
    uv run python docs/magazine/build_magazine_gist.py <gist_id>
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/guglxni/VayuDrishti/main"
GIST_TOKEN = "d5646cd8720da01fe1fb749fc86190ba"

PAPER = "#F7F4EB"
INK = "#0A1119"
TERRA = "#C84B31"
SEA = "#2F5D62"
CARD = "#FBF8F1"

# Frozen 1–21 Oct 2026 hour-of-day means (hour_start, Asia/Kolkata), four stations.
# Recomputed from data/raw/Train.csv and data/external/*.parquet when those exist.
FROZEN_DIURNAL = {
    "obs": [
        51.61, 50.79, 49.18, 48.11, 46.07, 44.95, 45.50, 47.93, 50.22, 50.45,
        48.77, 46.24, 43.64, 42.24, 41.86, 42.71, 42.48, 43.84, 46.21, 47.73,
        49.88, 51.51, 52.47, 51.23,
    ],
    "cams": [
        58.99, 58.18, 56.79, 54.89, 53.01, 54.18, 53.03, 53.55, 54.77, 57.49,
        61.57, 66.53, 72.75, 77.97, 79.95, 79.27, 74.03, 82.13, 74.62, 68.36,
        63.49, 60.84, 59.89, 59.91,
    ],
    "pblh": [
        121.37, 114.05, 107.02, 101.61, 95.42, 83.63, 87.86, 105.83, 176.19,
        359.40, 797.92, 1619.35, 1862.98, 1819.88, 1594.35, 1368.87, 1110.71,
        671.37, 346.19, 299.82, 270.30, 238.57, 204.05, 147.50,
    ],
    "r": -0.6681494058562392,
    "n": 1836,
    "r_val": -0.76347409848551,
    "val_cams_peak": 98.51,
    "val_cams_h": 14,
    "val_obs_trough": 48.11,
    "val_obs_h": 14,
    "source": "embedded hour-of-day means, 1–21 October 2026",
}

SHORT = {
    "b0_station_mean_90d": "90-day station mean",
    "b1_starter_rf_90d": "Starter random forest",
    "b2_diurnal_shrinkage_7d": "7-day diurnal shrinkage",
    "m1_calibrated_cams_prior": "M1 · tanh-MOS CAMS prior",
    "m2_huber_ridge_gam": "M2 · Huber/Ridge GAM + Duan",
    "m3_lightgbm_residual": "M3 · LightGBM residual",
    "m4_catboost_residual": "M4 · CatBoost residual",
    "m5_xgboost_residual": "M5 · XGBoost residual",
    "m6_wind_st_gat": "M6 · Wind-guided ST-GAT",
    "m7_timesfm3_compact": "M7 · TimesFM 3.0",
    "m8_chronos2_covariate": "M8 · Chronos-2",
    "m7_stacked_ensemble": "8-model ensemble",
}
ORDER = list(SHORT)
BAR_KEYS = [k for k in ORDER if k != "b0_station_mean_90d"]

CARD_NAMES = [
    ("m1_calibrated_cams_prior", "M1", "tanh-MOS CAMS"),
    ("m2_huber_ridge_gam", "M2", "GAM · Duan · AnEn"),
    ("m3_lightgbm_residual", "M3", "LightGBM residual"),
    ("m4_catboost_residual", "M4", "CatBoost residual"),
    ("m5_xgboost_residual", "M5", "XGBoost residual"),
    ("m6_wind_st_gat", "M6", "Wind ST-GAT"),
    ("m7_timesfm3_compact", "M7", "TimesFM 3.0"),
    ("m8_chronos2_covariate", "M8", "Chronos-2 · 21q"),
]

DOIS = [
    ("10.71646/YAQTMA", "CPCB hourly archive, Train.csv, 14 auxiliary stations"),
    ("10.71646/XXOBKO", "Open-Meteo NWP / ERA5, including boundary-layer height"),
    ("10.71646/CAMS26", "ECMWF CAMS composition (PM2.5, gases, dust, AOD)"),
    ("10.71646/EAE4EE", "VABB METAR: tmpc, dwpc, relh, drct, sknt, alti"),
    ("10.71646/GHSZ7J", "Google AlphaEarth Foundations, 10 m embeddings"),
    ("10.71646/EV3RBJ", "Google Open Buildings V3 footprints"),
    ("10.71646/EREDWQ", "BMC wards A–T"),
    ("10.71646/AAGK0G", "WorldPop 2026 gridded population"),
    ("10.71646/MWQNR8", "Deonar, Mulund, and Kanjurmarg landfill polygons"),
    ("10.71646/VA5BD9", "TROPOMI / waste-plume attribution"),
    ("10.71646/XJIY8W", "UDISE+ school directory"),
    ("10.71646/1XSUVR", "BMC hospitals and dispensaries"),
    ("10.71646/YJIAGQ", "NASA FIRMS active-fire anomalies"),
    ("10.71646/4ULQRE", "MahaRERA construction sites"),
    ("10.71646/MKJGDY", "Maharashtra public-holiday calendar"),
]

SVG_STYLE = """
<style>
  .p-serif { font-family: "Newsreader", Georgia, "Iowan Old Style", Palatino, "Palatino Linotype", serif; fill: #0A1119; }
  .p-mono { font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace; fill: #0A1119; }
  .p-deva { font-family: "Tiro Devanagari Hindi", "Noto Serif Devanagari", "Kohinoor Devanagari", "Nirmala UI", serif; fill: #0A1119; }
  .p-muted { fill: #5E564C; }
  .p-terra { fill: #C84B31; }
  .p-sea { fill: #2F5D62; }
  .p-wind { fill: none; stroke: #2F5D62; stroke-width: 1.35; stroke-linecap: round; stroke-dasharray: 1.5 8; animation: vd-drift 11s linear infinite; }
  .p-dash { fill: none; stroke: #C84B31; stroke-width: 1.35; stroke-linecap: round; stroke-dasharray: 4 7; animation: vd-drift 7.5s linear infinite; }
  .p-beacon { fill: none; stroke: #C84B31; stroke-width: 1.15; transform-box: fill-box; transform-origin: center; animation: vd-pulse 2.8s ease-out infinite; }
  .p-lid { transform-box: fill-box; transform-origin: center; animation: vd-breathe 8.5s ease-in-out infinite; }
  @keyframes vd-drift { to { stroke-dashoffset: -240; } }
  @keyframes vd-pulse { 0% { opacity: 0.9; } 75% { opacity: 0; } 100% { opacity: 0; } }
  @keyframes vd-breathe { 0%, 100% { transform: translateY(7px); } 50% { transform: translateY(-16px); } }
</style>
"""


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def trim(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def corr_text(value: float) -> str:
    magnitude = f"{abs(value):.3f}"
    return f"−{magnitude}" if value < 0 else magnitude


def hh(hour: int) -> str:
    return f"{int(hour):02d}:00"


def svg_doc(width: int, height: int, body: str, label: str) -> str:
    frame = (
        f'<rect x="10" y="10" width="{width - 20}" height="{height - 20}" '
        f'fill="none" stroke="{INK}" stroke-width="1"/>'
        f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" '
        f'fill="none" stroke="{TERRA}" stroke-width="0.6"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">\n'
        f"<title>{esc(label)}</title>\n"
        f'<rect width="100%" height="100%" fill="{PAPER}"/>\n'
        f"{SVG_STYLE}\n{frame}\n{body}\n</svg>\n"
    )


def text(x, y, content, cls="p-serif", size=16, anchor="start", fill=None, extra=""):
    style = f' style="fill:{fill}"' if fill else ""
    return (
        f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" '
        f'text-anchor="{anchor}"{style} {extra}>{esc(content)}</text>'
    )


def load_boards() -> dict[str, dict]:
    paths = {
        "24": ROOT / "data/processed/validation_leaderboard_gap24h.csv",
        "72": ROOT / "data/processed/validation_leaderboard_gap72h.csv",
    }
    boards = {}
    for gap, path in paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        boards[gap] = {row["model_key"]: row for row in rows}
    missing = [key for key in ORDER if key not in boards["24"] or key not in boards["72"]]
    if missing:
        raise SystemExit(f"leaderboard missing keys: {missing}")
    return boards


def load_weights() -> tuple[dict, dict]:
    w24 = json.loads((ROOT / "data/processed/ensemble_weights_gap24h.json").read_text())
    w72 = json.loads((ROOT / "data/processed/ensemble_weights_gap72h.json").read_text())
    return w24, w72


def _hour_means(frame, value_col: str, time_col: str) -> list[float]:
    import pandas as pd

    hours = frame[time_col].dt.hour
    means = frame.groupby(hours)[value_col].mean()
    return [float(means.loc[h]) for h in range(24)]


def compute_diurnal() -> dict:
    train_path = ROOT / "data/raw/Train.csv"
    aq_path = ROOT / "data/external/open_meteo_air_quality.parquet"
    wx_path = ROOT / "data/external/open_meteo_weather.parquet"
    if not (train_path.exists() and aq_path.exists() and wx_path.exists()):
        return dict(FROZEN_DIURNAL)
    import pandas as pd

    train = pd.read_csv(train_path)
    train["hour_end"] = pd.to_datetime(train["timestamp_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
    train["hour_start"] = train["hour_end"] - pd.Timedelta(hours=1)
    aq = pd.read_parquet(aq_path, columns=["target_start_ist", "pm2_5", "location_id"])
    wx = pd.read_parquet(wx_path, columns=["target_start_ist", "boundary_layer_height", "location_id"])
    aq["target_start_ist"] = pd.to_datetime(aq["target_start_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
    wx["target_start_ist"] = pd.to_datetime(wx["target_start_ist"], utc=True).dt.tz_convert("Asia/Kolkata")

    def window(start: str, end: str) -> dict:
        obs = train.loc[
            (train["hour_end"] >= start)
            & (train["hour_end"] < end)
            & train["pm25"].notna()
            & (train["pm25"] <= 450)
        ].copy()
        cams = aq.loc[(aq["target_start_ist"] >= start) & (aq["target_start_ist"] < end)].copy()
        pbl = wx.loc[(wx["target_start_ist"] >= start) & (wx["target_start_ist"] < end)].copy()
        obs_h = _hour_means(obs, "pm25", "hour_start")
        cams_h = _hour_means(cams, "pm2_5", "target_start_ist")
        pbl_h = _hour_means(pbl, "boundary_layer_height", "target_start_ist")
        paired = pd.DataFrame({"obs": obs_h, "cams": cams_h})
        return {
            "obs": obs_h,
            "cams": cams_h,
            "pblh": pbl_h,
            "r": float(paired["obs"].corr(paired["cams"])),
            "n": int(len(obs)),
        }

    full = window("2026-10-01", "2026-10-22")
    val = window("2026-10-12", "2026-10-22")
    return {
        **full,
        "r_val": val["r"],
        "val_cams_peak": max(val["cams"]),
        "val_cams_h": int(val["cams"].index(max(val["cams"]))),
        "val_obs_trough": min(val["obs"]),
        "val_obs_h": int(val["obs"].index(min(val["obs"]))),
        "source": "Train.csv + Open-Meteo CAMS/PBLH, 1–21 October 2026, hour_start",
    }


def diurnal_facts(raw: dict) -> dict:
    obs, cams, pblh = raw["obs"], raw["cams"], raw["pblh"]
    cams_h = max(range(24), key=lambda h: cams[h])
    obs_h = min(range(24), key=lambda h: obs[h])
    pbl_lo = min(range(24), key=lambda h: pblh[h])
    pbl_hi = max(range(24), key=lambda h: pblh[h])
    aft = range(14, 18)
    return {
        "r": corr_text(raw["r"]),
        "r_val": corr_text(raw["r_val"]),
        "n": f"{raw['n']:,}",
        "cams_peak": f"{cams[cams_h]:.1f}",
        "cams_h": hh(cams_h),
        "obs_trough": f"{obs[obs_h]:.1f}",
        "obs_h": hh(obs_h),
        "pbl_lo": f"{pblh[pbl_lo]:.0f}",
        "pbl_lo_h": hh(pbl_lo),
        "pbl_hi": f"{pblh[pbl_hi]:,.0f}",
        "pbl_hi_h": hh(pbl_hi),
        "cams_aft": f"{sum(cams[h] for h in aft) / 4:.1f}",
        "obs_aft": f"{sum(obs[h] for h in aft) / 4:.1f}",
        "val_cams_peak": f"{raw['val_cams_peak']:.1f}",
        "val_cams_h": hh(raw["val_cams_h"]),
        "val_obs_trough": f"{raw['val_obs_trough']:.1f}",
        "val_obs_h": hh(raw["val_obs_h"]),
        "source": raw["source"],
    }


def map_y(value, vmin, vmax, y_top, y_bot) -> float:
    span = vmax - vmin
    frac = 0.0 if span == 0 else (float(value) - vmin) / span
    frac = min(1.0, max(0.0, frac))
    return y_bot - frac * (y_bot - y_top)


def series_points(values, vmin, vmax, x0, x1, y_top, y_bot):
    last = len(values) - 1
    points = []
    for i, value in enumerate(values):
        x = x0 + (x1 - x0) * (i / last)
        y = map_y(value, vmin, vmax, y_top, y_bot)
        points.append((x, y))
    return points


def point_string(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def area_path(points, y_bot) -> str:
    head = f"M {points[0][0]:.1f},{y_bot:.1f} "
    body = " ".join(f"L {x:.1f},{y:.1f}" for x, y in points)
    return head + body + f" L {points[-1][0]:.1f},{y_bot:.1f} Z"


def plate_masthead(facts: dict) -> str:
    cells = [
        ("9.1531", "24h October OOF", "µg/m³ RMSE"),
        ("9.2545", "72h October OOF", "µg/m³ RMSE"),
        ("6.0855", "1–2 November OOF", "µg/m³ RMSE"),
        ("0.0", "Public leaderboard", "tied first · xP5LoEHK · sK5XpFqb"),
    ]
    bits = [
        text(36, 40, "MONOGRAPH SERIES   ·   ATMOSPHERIC & COMPUTATIONAL EARTH SCIENCES", "p-mono p-muted", 11),
        text(1164, 40, "VOL. I   ·   NOV 2026", "p-mono p-muted", 11, "end"),
        f'<line x1="36" y1="52" x2="1164" y2="52" stroke="{INK}" stroke-width="0.6"/>',
        text(36, 96, "वायुदृष्टि", "p-deva", 28),
        text(36, 142, "VayuDrishti", "p-serif", 46),
        text(36, 176, "Physics-guided multi-scale PM2.5 forecasting", "p-serif", 20),
        text(36, 202, "and a 3D boundary-layer digital twin for Mumbai.", "p-serif", 20),
        text(36, 230, "Four CPCB stations. Eight model classes. One closed hour.", "p-serif p-muted", 15),
        text(1164, 230, "15 h post-Diwali OOF   6.2616 µg/m³", "p-mono", 13, "end"),
        f'<rect x="36" y="286" width="1128" height="28" fill="#E7F0EE"/>',
        '<path class="p-wind" d="M48 300 C 220 292, 400 308, 620 296"/>',
        '<path class="p-wind" d="M80 308 C 260 302, 480 316, 760 304" style="animation-delay:-4s"/>',
        text(52, 278, "285°", "p-mono p-sea", 11),
        text(1148, 278, "110°", "p-mono p-sea", 11, "end"),
    ]
    stations = [
        (280, "Borivali East", "11606", 0),
        (520, "Malad West", "11611", 0.5),
        (760, "Byculla", "3409510", 1.0),
        (1000, "Shivaji Nagar", "3409511", 1.5),
    ]
    for x, name, sid, delay in stations:
        bits.append(text(x, 278, name, "p-serif", 14, "middle"))
        bits.append(
            f'<circle cx="{x}" cy="300" r="8" class="p-beacon" style="animation-delay:{delay}s"/>'
            f'<circle cx="{x}" cy="300" r="3" fill="{TERRA}"/>'
        )
        bits.append(text(x, 328, sid, "p-mono p-muted", 11, "middle"))
    bits.append(f'<line x1="36" y1="342" x2="1164" y2="342" stroke="{INK}" stroke-width="0.6"/>')
    for i, (number, label, unit) in enumerate(cells):
        cx = 36 + i * 282 + 141
        bits.append(text(cx, 374, number, "p-serif", 30, "middle"))
        bits.append(f'<line x1="{cx - 18}" y1="384" x2="{cx + 18}" y2="384" stroke="{TERRA}" stroke-width="1.4"/>')
        bits.append(text(cx, 402, label, "p-mono", 11, "middle"))
        bits.append(text(cx, 418, unit, "p-mono p-muted", 10, "middle"))
        if i:
            x = 36 + i * 282
            bits.append(f'<line x1="{x}" y1="352" x2="{x}" y2="424" stroke="{INK}" stroke-opacity="0.18"/>')
    return svg_doc(1200, 440, "\n".join(bits), "VayuDrishti monograph cover")


def plate_architecture(boards, w24, w72) -> str:
    def rmse(key, gap):
        return trim(float(boards[gap][key]["OOF_RMSE"]))

    bits = [
        text(36, 42, "FIG. II", "p-mono p-terra", 12),
        text(36, 72, "Eight models behind one closed hour", "p-serif", 28),
        text(900, 72, "hour_end  ≤  T0", "p-mono", 13, "end"),
    ]
    sources = [
        ("15 integrated DOIs", "92-entry ClimateVerse catalog"),
        ("Causal gate", "hour_end ≤ T0  ·  Asia/Kolkata"),
        ("2-stage Hampel", "flag_high_1000 or > 450, then 7 h"),
    ]
    box_w, gap, y, h = 360, 18, 96, 62
    xs = [36 + i * (box_w + gap) for i in range(3)]
    for i, ((title, sub), x) in enumerate(zip(sources, xs)):
        bits.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{h}" fill="{CARD}" stroke="{INK}" stroke-width="0.8"/>'
        )
        bits.append(text(x + 16, y + 26, title, "p-serif", 16))
        bits.append(text(x + 16, y + 46, sub, "p-mono p-muted", 11))
        if i < 2:
            x1 = x + box_w
            x2 = xs[i + 1]
            mid = y + h / 2
            bits.append(f'<path class="p-dash" d="M {x1 + 2} {mid} H {x2 - 2}"/>')
    bus_y = 176
    for x in xs:
        bits.append(f'<path class="p-dash" d="M {x + box_w / 2:.0f} {y + h} V {bus_y}"/>')
    bits.append(f'<path class="p-dash" d="M {xs[0] + box_w / 2:.0f} {bus_y} H {xs[-1] + box_w / 2:.0f}"/>')
    card_w, card_h, card_gap = 270, 96, 14
    for i, (key, code, name) in enumerate(CARD_NAMES):
        col, row = i % 4, i // 4
        x = 36 + col * (card_w + card_gap)
        cy = 196 + row * (card_h + 12)
        if row == 0:
            bits.append(f'<path class="p-dash" d="M {x + card_w / 2:.0f} {bus_y} V {cy}"/>')
        bits.append(
            f'<rect x="{x}" y="{cy}" width="{card_w}" height="{card_h}" fill="{CARD}" stroke="{INK}" stroke-width="0.8"/>'
        )
        bits.append(text(x + 14, cy + 24, code, "p-mono p-terra", 12))
        bits.append(text(x + 52, cy + 24, name, "p-serif", 15))
        bits.append(text(x + 14, cy + 52, f"24h   {rmse(key, '24')}", "p-mono", 13))
        bits.append(text(x + 14, cy + 74, f"72h   {rmse(key, '72')}", "p-mono p-muted", 13))
    before24 = trim(w24["oof_rmse_before_shrinkage"])
    after24 = trim(w24["oof_rmse_after_shrinkage"])
    before72 = trim(w72["oof_rmse_before_shrinkage"])
    after72 = trim(w72["oof_rmse_after_shrinkage"])
    bits.append(f'<rect x="36" y="424" width="1128" height="72" fill="{INK}"/>')
    bits.append(text(56, 452, "Simplex combiner", "p-serif", 18, fill=PAPER))
    bits.append(
        text(56, 476, "SLSQP + Nelder–Mead, then station shrinkage", "p-mono", 12, fill="#D9D0C2")
    )
    bits.append(text(760, 450, after24, "p-serif", 22, fill=TERRA))
    bits.append(text(900, 450, "24h", "p-mono", 12, fill=PAPER))
    bits.append(text(980, 450, after72, "p-serif", 22, fill=TERRA))
    bits.append(text(1104, 450, "72h", "p-mono", 12, fill=PAPER))
    bits.append(
        text(
            760,
            476,
            f"shrinkage  {before24} → {after24}    ·    {before72} → {after72}",
            "p-mono",
            11,
            fill="#D9D0C2",
        )
    )
    return svg_doc(1200, 520, "\n".join(bits), "Eight-model causal forecasting stack")


def plate_diurnal(raw: dict, facts: dict, boards) -> str:
    """Two full-width panels. Panel A is the printed diurnal means. Panel B is October OOF RMSE.

    Axis max is 20 so the 90-day mean stays off the scale, as the caption says.
    Bar lengths and callouts use the leaderboard and the hour-of-day means already on the page.
    """
    x0, x1, y_top, y_bot = 92, 1128, 178, 548
    pm_min, pm_max = 35, 90
    obs_pts = series_points(raw["obs"], pm_min, pm_max, x0, x1, y_top, y_bot)
    cams_pts = series_points(raw["cams"], pm_min, pm_max, x0, x1, y_top, y_bot)
    cams_hour = max(range(24), key=lambda hour: raw["cams"][hour])
    obs_hour = min(range(24), key=lambda hour: raw["obs"][hour])

    def hx(hour: int) -> float:
        return x0 + (x1 - x0) * (hour / 23)

    bits = [
        text(40, 46, "FIG. III", "p-mono p-terra", 15),
        text(40, 90, "The afternoon CAMS peak, and the surface trough", "p-serif", 32),
        text(
            40,
            122,
            "1–21 October 2026  ·  four stations  ·  hour_start, Asia/Kolkata  ·  values above 450 µg/m³ removed",
            "p-mono p-muted",
            14,
        ),
        text(40, 158, "PANEL A", "p-mono p-terra", 14),
        text(128, 158, "HOUR-OF-DAY MEANS  ·  µg/m³", "p-mono p-muted", 14),
        f'<line x1="720" y1="148" x2="756" y2="148" stroke="{TERRA}" stroke-width="4" stroke-linecap="round"/>',
        text(766, 153, "CAMS", "p-mono", 15),
        f'<line x1="860" y1="148" x2="896" y2="148" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>',
        text(906, 153, "CPCB", "p-mono", 15),
        text(1160, 154, f"r = {facts['r']}", "p-serif", 22, "end"),
        f'<rect x="{x0}" y="{y_top}" width="{x1 - x0}" height="{y_bot - y_top}" fill="{CARD}" stroke="{INK}" stroke-opacity="0.28" stroke-width="1"/>',
    ]
    shade_x = hx(14)
    shade_w = hx(17) - shade_x
    bits.append(
        f'<rect x="{shade_x:.1f}" y="{y_top}" width="{shade_w:.1f}" height="{y_bot - y_top}" fill="{TERRA}" fill-opacity="0.10"/>'
    )
    for tick in (40, 50, 60, 70, 80, 90):
        y = map_y(tick, pm_min, pm_max, y_top, y_bot)
        bits.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{INK}" stroke-opacity="0.10"/>'
        )
        bits.append(text(x0 - 12, y + 5, str(tick), "p-mono p-muted", 15, "end"))
    for hour in (0, 6, 12, 18, 23):
        x = hx(hour)
        bits.append(
            f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}" stroke="{INK}" stroke-opacity="0.08"/>'
        )
        bits.append(text(x, y_bot + 24, f"{hour:02d}", "p-mono p-muted", 15, "middle"))
    bits.append(
        f'<polyline points="{point_string(cams_pts)}" fill="none" stroke="{TERRA}" stroke-width="3.6" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    bits.append(
        f'<polyline points="{point_string(obs_pts)}" fill="none" stroke="{INK}" stroke-width="3.6" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    peak_x, peak_y = cams_pts[cams_hour]
    trough_x, trough_y = obs_pts[obs_hour]
    bits.append(
        f'<circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="7" fill="{PAPER}" stroke="{TERRA}" stroke-width="3"/>'
    )
    bits.append(
        f'<circle cx="{trough_x:.1f}" cy="{trough_y:.1f}" r="7" fill="{PAPER}" stroke="{INK}" stroke-width="3"/>'
    )
    bits.append(text(peak_x - 10, peak_y - 14, facts["cams_peak"], "p-serif p-terra", 20, "end"))
    bits.append(text(trough_x, trough_y - 18, facts["obs_trough"], "p-serif", 20, "middle"))
    bits.append(text(x0, y_bot + 48, "hour_start, IST", "p-mono p-muted", 14))
    bits.append(
        text(shade_x + shade_w / 2, y_bot + 48, "14:00–17:00", "p-mono p-terra", 14, "middle")
    )

    stats = [
        (facts["cams_peak"], "CAMS peak", facts["cams_h"]),
        (facts["obs_trough"], "CPCB lowest", facts["obs_h"]),
        (f"{facts['cams_aft']}  /  {facts['obs_aft']}", "14:00–17:00 means", "CAMS  ·  CPCB"),
        (facts["r"], "correlation r", "24 hourly means"),
    ]
    stat_top = 640
    cell_w = 282
    for i, (number, label, sub) in enumerate(stats):
        x = 40 + i * cell_w
        if i:
            bits.append(
                f'<line x1="{x}" y1="{stat_top}" x2="{x}" y2="{stat_top + 78}" stroke="{INK}" stroke-opacity="0.15"/>'
            )
        bits.append(text(x + cell_w / 2, stat_top + 36, number, "p-serif", 32, "middle"))
        bits.append(text(x + cell_w / 2, stat_top + 58, label, "p-mono", 13, "middle"))
        bits.append(text(x + cell_w / 2, stat_top + 76, sub, "p-mono p-muted", 13, "middle"))
    bits.append(
        text(
            40,
            744,
            f"Boundary-layer height in these means: {facts['pbl_lo']} m at {facts['pbl_lo_h']} to {facts['pbl_hi']} m at {facts['pbl_hi_h']}",
            "p-mono p-muted",
            14,
        )
    )
    bits.append(f'<line x1="40" y1="768" x2="1160" y2="768" stroke="{INK}" stroke-opacity="0.2"/>')
    bits.append(text(40, 808, "PANEL B", "p-mono p-terra", 14))
    bits.append(text(128, 808, "OCTOBER OUT-OF-FOLD RMSE", "p-mono p-muted", 14))

    labels = {
        "b1_starter_rf_90d": "Starter random forest",
        "b2_diurnal_shrinkage_7d": "7-day diurnal",
        "m1_calibrated_cams_prior": "M1 CAMS prior",
        "m2_huber_ridge_gam": "M2 GAM",
        "m3_lightgbm_residual": "M3 LightGBM",
        "m4_catboost_residual": "M4 CatBoost",
        "m5_xgboost_residual": "M5 XGBoost",
        "m6_wind_st_gat": "M6 ST-GAT",
        "m7_timesfm3_compact": "M7 TimesFM",
        "m8_chronos2_covariate": "M8 Chronos-2",
        "m7_stacked_ensemble": "Ensemble",
    }
    scale_max = 20.0
    origin = 300
    track = 500
    top = 888
    row_h = 34
    bits.append(text(origin, 842, "0", "p-mono p-muted", 14, "middle"))
    for tick in (5, 10, 15, 20):
        x = origin + track * tick / scale_max
        bits.append(text(x, 842, str(tick), "p-mono p-muted", 14, "middle"))
    bits.append(text(860, 842, "24h", "p-mono p-muted", 14))
    bits.append(text(980, 842, "72h", "p-mono p-muted", 14))
    axis_bottom = top + len(BAR_KEYS) * row_h - 10
    bits.append(f'<line x1="{origin}" y1="856" x2="{origin}" y2="{axis_bottom}" stroke="{INK}" stroke-opacity="0.35"/>')
    for tick in (5, 10, 15, 20):
        x = origin + track * tick / scale_max
        bits.append(
            f'<line x1="{x:.1f}" y1="856" x2="{x:.1f}" y2="{axis_bottom}" stroke="{INK}" stroke-opacity="0.08"/>'
        )
    for i, key in enumerate(BAR_KEYS):
        y = top + i * row_h
        v24 = float(boards["24"][key]["OOF_RMSE"])
        v72 = float(boards["72"][key]["OOF_RMSE"])
        ens = key == "m7_stacked_ensemble"
        color = TERRA if ens else INK
        size = 18 if ens else 15
        if ens:
            bits.append(
                f'<rect x="36" y="{y - 6}" width="1128" height="30" fill="{TERRA}" fill-opacity="0.08"/>'
            )
        bits.append(text(origin - 16, y + 14, labels[key], "p-mono", size, "end", TERRA if ens else None))
        bits.append(
            f'<rect x="{origin}" y="{y}" width="{track * v24 / scale_max:.1f}" height="14" fill="{color}"/>'
        )
        bits.append(
            f'<rect x="{origin}" y="{y + 16}" width="{track * v72 / scale_max:.1f}" height="8" fill="{color}" fill-opacity="0.35"/>'
        )
        bits.append(text(860, y + 16, trim(v24), "p-mono", size, fill=TERRA if ens else None))
        bits.append(text(980, y + 16, trim(v72), "p-mono", size, fill=TERRA if ens else None))
    legend_y = axis_bottom + 36
    bits.append(f'<rect x="40" y="{legend_y - 12}" width="22" height="14" fill="{INK}"/>')
    bits.append(text(70, legend_y, "24h gap, solid", "p-mono", 14))
    bits.append(f'<rect x="250" y="{legend_y - 8}" width="22" height="8" fill="{INK}" fill-opacity="0.35"/>')
    bits.append(text(280, legend_y, "72h gap, pale", "p-mono", 14))
    off24 = trim(float(boards["24"]["b0_station_mean_90d"]["OOF_RMSE"]))
    off72 = trim(float(boards["72"]["b0_station_mean_90d"]["OOF_RMSE"]))
    bits.append(
        text(
            470,
            legend_y,
            f"90-day mean off this axis: {off24} / {off72}",
            "p-mono p-muted",
            14,
        )
    )
    return svg_doc(1200, legend_y + 36, "\n".join(bits), "CAMS diurnal inversion and October benchmark")


def plate_coast(facts: dict) -> str:
    bits = [
        text(36, 42, "FIG. IV", "p-mono p-terra", 12),
        text(36, 74, "A west wind, a landfill, and a creek", "p-serif", 28),
        text(
            36,
            98,
            "Schematic coastal sequence. Vertical scale is drawn so both lids stay visible.",
            "p-serif p-muted",
            14,
        ),
        f'<rect x="36" y="150" width="150" height="150" fill="#D7E4E1"/>',
        '<path d="M186 300 L186 250 C 240 246, 300 236, 360 240 C 430 246, 470 228, 530 232 C 600 238, 650 226, 730 230 C 790 234, 830 248, 860 252 L 860 300 Z" fill="#E6DDD0" stroke="#0A1119" stroke-width="0.8"/>',
        f'<rect x="860" y="250" width="120" height="50" fill="#D7E4E1"/>',
        '<path d="M980 300 L980 256 C 1040 250, 1100 258, 1148 254 L 1148 300 Z" fill="#E6DDD0" stroke="#0A1119" stroke-width="0.8"/>',
        '<path class="p-wind" d="M48 176 C 120 168, 180 184, 250 172"/>',
        '<path class="p-wind" d="M40 200 C 130 194, 200 208, 280 198" style="animation-delay:-3s"/>',
        '<path class="p-dash" d="M980 214 C 900 208, 820 220, 740 210"/>',
        # Afternoon lid, static, labelled with the October maximum.
        '<path d="M48 132 C 220 124, 420 118, 640 126 C 860 134, 1000 122, 1152 128" fill="none" stroke="#2F5D62" stroke-width="1.5" stroke-dasharray="5 6"/>',
        text(48, 116, f"afternoon mean  {facts['pbl_hi']} m at {facts['pbl_hi_h']}", "p-mono p-sea", 11),
        # Nocturnal lid breathes.
        '<g class="p-lid">'
        '<path d="M48 236 C 220 228, 420 232, 640 226 C 860 220, 1000 230, 1152 224" fill="none" stroke="#C84B31" stroke-width="1.7"/>'
        f'<text x="700" y="218" class="p-mono p-terra" font-size="11">nocturnal mean  {esc(facts["pbl_lo"])} m at {esc(facts["pbl_lo_h"])}</text>'
        "</g>",
    ]
    places = [
        (90, "Arabian Sea", "285° onshore"),
        (250, "Malad West", "11611"),
        (400, "Borivali East", "11606"),
        (560, "Byculla", "3409510"),
        (760, "Shivaji Nagar", "3409511 · Deonar 75°"),
        (910, "Thane Creek", "110°"),
        (1060, "Navi Mumbai", "MIDC shore"),
    ]
    for x, name, sub in places:
        bits.append(f'<line x1="{x}" y1="300" x2="{x}" y2="316" stroke="{INK}" stroke-opacity="0.45"/>')
        bits.append(text(x, 334, name, "p-serif", 12, "middle"))
        bits.append(text(x, 350, sub, "p-mono p-muted", 10, "middle"))
    bits.append(text(36, 378, "Onshore flow 285° from the Arabian Sea. Deonar drainage near 75°. Creek axis near 110°.", "p-mono p-muted", 11))
    return svg_doc(1200, 400, "\n".join(bits), "Coastal boundary-layer cross-section")


def weight_rows(w24, w72):
    keys = [
        ("m1_calibrated_cams_prior", "M1 CAMS prior"),
        ("m2_huber_ridge_gam", "M2 GAM"),
        ("m3_lightgbm_residual", "M3 LightGBM"),
        ("m4_catboost_residual", "M4 CatBoost"),
        ("m5_xgboost_residual", "M5 XGBoost"),
        ("m6_wind_st_gat", "M6 ST-GAT"),
        ("m7_timesfm3_compact", "M7 TimesFM 3.0"),
        ("m8_chronos2_covariate", "M8 Chronos-2"),
    ]
    rows = []
    for key, name in keys:
        rows.append((name, trim(w24["weights"][key]), trim(w72["weights"][key])))
    return rows


def board_rows(boards):
    rows = []
    for key in ORDER:
        a = boards["24"][key]
        b = boards["72"][key]
        rows.append(
            {
                "key": key,
                "name": SHORT[key],
                "a": trim(a["OOF_RMSE"]),
                "b": trim(b["OOF_RMSE"]),
                "ga": f"{float(a['Gain_vs_StarterRF_pct']):+.2f}%",
                "gb": f"{float(b['Gain_vs_StarterRF_pct']):+.2f}%",
            }
        )
    return rows


def table_md(rows) -> str:
    lines = [
        "| Model | 24h OOF RMSE | 72h OOF RMSE | Gain vs starter RF, 24h / 72h |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in rows:
        name = f"**{row['name']}**" if row["key"] == "m7_stacked_ensemble" else row["name"]
        a = f"**{row['a']}**" if row["key"] == "m7_stacked_ensemble" else row["a"]
        b = f"**{row['b']}**" if row["key"] == "m7_stacked_ensemble" else row["b"]
        lines.append(f"| {name} | {a} | {b} | {row['ga']} / {row['gb']} |")
    return "\n".join(lines)


def table_html(rows) -> str:
    body = [
        "<table>",
        "<thead><tr><th>Model</th><th>24h OOF</th><th>72h OOF</th><th>Gain vs starter RF</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        cls = ' class="ens"' if row["key"] == "m7_stacked_ensemble" else ""
        body.append(
            f"<tr{cls}><td>{esc(row['name'])}</td><td>{esc(row['a'])}</td>"
            f"<td>{esc(row['b'])}</td><td>{esc(row['ga'])} / {esc(row['gb'])}</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def weights_md(rows) -> str:
    lines = [
        "| Component | Weight, 24h gap | Weight, 72h gap |",
        "| :--- | ---: | ---: |",
    ]
    for name, a, b in rows:
        lines.append(f"| {name} | {a} | {b} |")
    return "\n".join(lines)


def weights_html(rows) -> str:
    body = [
        "<table>",
        "<thead><tr><th>Component</th><th>24h weight</th><th>72h weight</th></tr></thead><tbody>",
    ]
    for name, a, b in rows:
        body.append(f"<tr><td>{esc(name)}</td><td>{esc(a)}</td><td>{esc(b)}</td></tr>")
    body.append("</tbody></table>")
    return "\n".join(body)


def dois_md() -> str:
    lines = ["| DOI | Role in VayuDrishti |", "| :--- | :--- |"]
    for doi, role in DOIS:
        lines.append(f"| [{doi}](https://doi.org/{doi}) | {role} |")
    return "\n".join(lines)


def dois_html() -> str:
    body = ["<table><thead><tr><th>DOI</th><th>Role</th></tr></thead><tbody>"]
    for doi, role in DOIS:
        body.append(
            f'<tr><td><a href="https://doi.org/{doi}">{esc(doi)}</a></td><td>{esc(role)}</td></tr>'
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def md_inline(content: str) -> str:
    import re

    text_value = esc(content)
    text_value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2">\1</a>',
        text_value,
    )
    text_value = re.sub(r"`([^`]+)`", r"<code>\1</code>", text_value)
    text_value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text_value)
    text_value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text_value)
    return text_value


def blocks(facts, rows, wrows, w24, w72, gist_id: str):
    preview = f"https://gistpreview.github.io/?{gist_id}"
    repo = "https://github.com/guglxni/VayuDrishti"
    a24 = trim(w24["station_alphas"]["11606"])
    a_byculla = trim(w24["station_alphas"]["3409510"])
    a72_shivaji = trim(w72["station_alphas"]["3409511"])
    shots = [
        ("01-city-map.png", "City map", "Eighteen station columns, 24 BMC wards, wind streamlines, and AirPhyNet flux arcs."),
        ("02-air-column.png", "Air column", "The PBLH lid, aerosol particles, and source plumes over the island."),
        ("03-forecast.png", "Forecast", "TimesFM 3.0 deciles, Chronos-2 quantiles, and the trapezoidal mean."),
        ("04-ward-actions.png", "Ward actions", "Ward risk, the ClimateVerse catalog, and the BMC GRAP copilot."),
    ]
    return [
        {"type": "kicker", "text": "Monograph series  ·  Atmospheric and computational earth sciences  ·  Vol. I  ·  November 2026"},
        {"type": "h1", "text": "वायुदृष्टि VayuDrishti"},
        {"type": "dek", "text": "Physics-guided multi-scale PM2.5 forecasting and a 3D atmospheric digital twin for Mumbai."},
        {"type": "p", "text": f"Live magazine: [open the edition]({preview}). Code and plates: [guglxni/VayuDrishti]({repo})."},
        {"type": "fig", "id": "plate1", "alt": "VayuDrishti monograph masthead", "caption": "Fig. I. Cover metrics are out-of-fold RMSE on raw pm25, except the public leaderboard score of 0.0 on submissions xP5LoEHK and sK5XpFqb."},
        {"type": "p", "text": "VayuDrishti forecasts hourly PM2.5 at four Mumbai CPCB stations and draws that same air as a civic twin. The stations are Borivali East (`11606`), Malad West (`11611`), Byculla (`3409510`), and Shivaji Nagar (`3409511`). The scored file has 2,592 rows, 22 October 2026 through 17 November 2026. The private window is 11–12 November, 192 rows, scored by RMSE."},
        {"type": "p", "text": "Five October folds, 12–21 October, give the eight-model stack an out-of-fold RMSE of **9.1531 µg/m³** at a 24-hour reporting gap and **9.2545 µg/m³** at 72 hours. A 15-hour post-Diwali window, 27 October–4 November, gives **6.2616 µg/m³**. The 1–2 November slice of that regime gives **6.0855 µg/m³**. On the public leaderboard, `xP5LoEHK` and `sK5XpFqb` score **0.0** and sit tied for first."},
        {"type": "note", "text": "src/submit.py writes the Train.csv observation onto any test row whose hour has already been released. The public 0.0 is that lookup. The out-of-fold table is the score on hours the models had to forecast."},
        {"type": "h2", "text": "A squared error wants the mean"},
        {"type": "p", "text": "RMSE is squared error. The forecast that minimizes expected squared error is the conditional mean. TimesFM 3.0 returns nine deciles and Chronos-2 returns 21 quantiles. VayuDrishti integrates those quantiles with the trapezoidal rule and submits the integral. On a right-skewed hour the mean sits above the median, which is the point the absolute-error loss would have chosen."},
        {"type": "eq", "latex": r"\hat{y}_{\mathrm{RMSE}}=\mathbb{E}[Y\mid\mathcal{I}_{T_0}]=\int_0^1 F^{-1}(\tau)\,d\tau"},
        {"type": "p", "text": "The TimesFM deciles use weights 0.15 on each tail and 0.10 on each interior decile. Chronos-2 uses the pipeline’s own probability mass per quantile level. Models train on `pm25_clean`: a `flag_high_1000` or a value above 450 µg/m³ is removed, then a 7-hour Hampel filter at 4.5 × 1.4826 × MAD. Every validation number in this piece is scored on the unmodified raw `pm25` column."},
        {"type": "eq", "latex": r"\tilde{\mathbf{w}}_{\mathrm{TimesFM}}=[0.15,\ 0.10,\ 0.10,\ 0.10,\ 0.10,\ 0.10,\ 0.10,\ 0.10,\ 0.15]"},
        {"type": "h2", "text": "The chemical model peaks while the city clears"},
        {"type": "p", "text": f"Panel A is the mean hour, 1–21 October 2026, four stations, indexed by `hour_start` in Asia/Kolkata. {facts['n']} station-hours remain after values above 450 µg/m³ are set aside. CAMS `pm2_5` peaks at **{facts['cams_peak']} µg/m³** at {facts['cams_h']}. The CPCB mean is lowest at **{facts['obs_trough']} µg/m³** at {facts['obs_h']}. From 14:00 to 17:00 the CAMS mean is {facts['cams_aft']} µg/m³ and the CPCB mean is {facts['obs_aft']} µg/m³. The correlation of the 24 hourly means is **r = {facts['r']}**."},
        {"type": "p", "text": f"Boundary-layer height in the same means runs from **{facts['pbl_lo']} m** at {facts['pbl_lo_h']} to **{facts['pbl_hi']} m** at {facts['pbl_hi_h']}. The surface is dirtiest under the shallow lid and cleaner once the afternoon sea breeze has mixed the column. CAMS, on this coastal hour, does the opposite. The architecture notes record a sharper version of the same mismatch on fold 4, 18–19 October: a CAMS spike of 102 µg/m³ while the CPCB stations stayed between 41 and 62 µg/m³."},
        {"type": "note", "text": f"On the validation window alone, 12–21 October, the hour-of-day correlation is r = {facts['r_val']}. CAMS peaks at {facts['val_cams_peak']} µg/m³ at {facts['val_cams_h']}. The CPCB mean is lowest at {facts['val_obs_trough']} µg/m³ at {facts['val_obs_h']}."},
        {"type": "fig", "id": "plate3", "alt": "CAMS diurnal curve and October benchmark bars", "caption": "Fig. III. Upper panel: hour-of-day means. Lower panel: October out-of-fold RMSE. Solid bars are the 24-hour gap. Pale bars are the 72-hour gap. The 90-day mean (37.9949 / 38.8708) is left off the axis."},
        {"type": "p", "text": "Model 1 keeps the CAMS information and refuses the false alarm. A log-ratio against the trailing CAMS baseline passes through a tanh gate with κ = 0.45, so a large chemical spike cannot multiply the diurnal prior by more than about e^0.45. Ventilation, PBLH times 10 m wind, pulls the same forecast back down when the column is actually mixing."},
        {"type": "eq", "latex": r"\rho(s,t)=\exp\!\left(\kappa\tanh\!\left(\frac{1}{\kappa}\ln\frac{C(s,t)}{\max(\bar C_{T_0}(s,h),10)}\right)\right),\quad \kappa=0.45"},
        {"type": "h2", "text": "Eight forecasts, one closed hour"},
        {"type": "p", "text": "Every historical target, and every VABB observation, has to satisfy `hour_end ≤ T0`. In Train.csv, `timestamp_ist` is the end of the hour, so the hour starts one hour earlier. Test.csv already names `target_start_ist` and `target_end_ist`. Open-Meteo and VABB are indexed on the start. VABB contributes temperature, dewpoint, relative humidity, wind direction, wind speed, and altimeter setting."},
        {"type": "eq", "latex": r"\mathrm{hour\_end}\le T_0"},
        {"type": "p", "text": "The eight forecasts are different instruments. M1 is the diurnal prior with the tanh CAMS gate. M2 is a Huber/Ridge spline GAM on the M1 residual, with Duan’s smearing on the log-ratio branch so the exponential back-transform keeps its mean. M3, M4, and M5 are LightGBM, CatBoost, and XGBoost residuals. M6 is a PyTorch graph whose attention prefers the downwind neighbor. M7 is TimesFM 3.0 in two compact views, four target series and those four plus CAMS and log ventilation, with `use_znorm` off and edge padding. M8 is Chronos-2 with future covariates and a meteorological analog ensemble."},
        {"type": "fig", "id": "plate2", "alt": "Eight-model architecture", "caption": "Fig. II. Card numbers are October out-of-fold RMSE, µg/m³. The dark bar is the stacked ensemble after station shrinkage."},
        {"type": "table", "md": table_md(rows), "html": table_html(rows), "caption": "October 2026, five folds, 12–21 October, scored on raw pm25. Source: validation_leaderboard_gap24h.csv and validation_leaderboard_gap72h.csv."},
        {"type": "p", "text": f"At 24 hours the saved simplex puts **{trim(w24['weights']['m2_huber_ridge_gam'])}** on the GAM and **{trim(w24['weights']['m8_chronos2_covariate'])}** on Chronos-2. The other six components sit on a shared floor of **{trim(w24['weights']['m1_calibrated_cams_prior'])}**. Shrinkage moves the out-of-fold RMSE from **{trim(w24['oof_rmse_before_shrinkage'])}** to **{trim(w24['oof_rmse_after_shrinkage'])}**. Byculla’s scale is **{a_byculla}**. Borivali East, Malad West, and Shivaji Nagar are **{a24}**."},
        {"type": "p", "text": f"At 72 hours the mass moves. Chronos-2 is **{trim(w72['weights']['m8_chronos2_covariate'])}**, the wind-guided graph is **{trim(w72['weights']['m6_wind_st_gat'])}**, the GAM is **{trim(w72['weights']['m2_huber_ridge_gam'])}**, and the floor is **{trim(w72['weights']['m1_calibrated_cams_prior'])}**. Shrinkage moves **{trim(w72['oof_rmse_before_shrinkage'])}** to **{trim(w72['oof_rmse_after_shrinkage'])}**. Byculla stays at **{trim(w72['station_alphas']['3409510'])}**. Shivaji Nagar is **{a72_shivaji}**. The two western stations stay at **{trim(w72['station_alphas']['11606'])}**."},
        {"type": "table", "md": weights_md(wrows), "html": weights_html(wrows), "caption": "Saved simplex weights in ensemble_weights_gap24h.json and ensemble_weights_gap72h.json."},
        {"type": "h2", "text": "Across the island"},
        {"type": "p", "text": "The afternoon wind that clears the CPCB trough is the Arabian Sea breeze, treated in the feature set as a 285° onshore axis. Shivaji Nagar is parameterized 0.8 km west-northwest of the Deonar landfill, with a 75° drainage axis. Thane Creek is a 110° axis. Model 6 adds the wind vector to the graph: a neighbor upwind of the target gets a larger attention logit, and a neighbor off the wind does not."},
        {"type": "fig", "id": "plate4", "alt": "Coastal boundary-layer cross-section", "caption": f"Fig. IV. The terracotta lid breathes around the nocturnal October mean, {facts['pbl_lo']} m at {facts['pbl_lo_h']}. The teal lid is the afternoon mean, {facts['pbl_hi']} m at {facts['pbl_hi_h']}. The drawing is a wind-regime sequence, with a schematic vertical scale."},
        {"type": "h2", "text": "What the twin shows"},
        {"type": "p", "text": "The same stack is the source for the 3D twin in `web/`. The city map holds 18 stations, the 24 BMC wards, and the wind field. The air column is the boundary-layer lid and the plumes under it. The forecast view shows the decile and quantile fans whose integral is the submitted mean. Ward actions ranks the 24 wards and opens the ClimateVerse catalog beside a GRAP briefing."},
        {"type": "shots", "items": shots},
        {"type": "h2", "text": "Fifteen datasets inside a catalog of 92"},
        {"type": "p", "text": "The digital twin ships a 92-entry ClimateVerse catalog. Fifteen of those datasets are wired into the forecast or the civic views. The chemical forecast is CAMS. The lid height is Open-Meteo. The airport wind is VABB, without visibility and without weather codes. Wards, schools, hospitals, landfills, fires, and construction sites belong to the civic layer."},
        {"type": "table", "md": dois_md(), "html": dois_html(), "caption": "Integrated ClimateVerse DOIs. The 92-entry count is the catalog shipped in web/public/data/climateverse_catalog.json."},
        {"type": "h2", "text": "Reproduce"},
        {"type": "p", "text": "Python 3.12 and uv. Fetched tables and checkpoints stay out of git. Zindi credentials stay in the environment."},
        {"type": "code", "text": "uv sync\nuv run python src/fetch_data.py\nuv run python src/validation.py && uv run python src/ensemble.py && uv run python src/submit.py"},
        {"type": "p", "text": "A submission has 2,592 rows, columns ID and Target, the SampleSubmission.csv order, and finite values at least 0. The Colab review notebooks are `notebooks/Mumbai_Winning_Colab.ipynb` and `notebooks/Mumbai_Winning_Solution.ipynb`."},
        {"type": "p", "text": f"Plate III was drawn from {facts['source']}. Leaderboard plates were drawn from the committed CSV and JSON files in `data/processed/`. This file is the November 2026 monograph; the private window had not yet been scored."},
    ]


def render_markdown(pieces, gist_id: str) -> str:
    out = []
    for block in pieces:
        kind = block["type"]
        if kind == "kicker":
            out.append(f"*{block['text']}*")
        elif kind == "h1":
            out.append(f"# {block['text']}")
        elif kind == "h2":
            out.append(f"## {block['text']}")
        elif kind == "dek":
            out.append(f"**{block['text']}**")
        elif kind in {"p", "note"}:
            prefix = "> " if kind == "note" else ""
            out.append(prefix + block["text"])
        elif kind == "eq":
            out.append(f"$$\n{block['latex']}\n$$")
        elif kind == "fig":
            out.append(f"![{block['alt']}]({RAW}/docs/magazine/{FIG_FILES[block['id']]})")
            out.append(f"*{block['caption']}*")
        elif kind == "table":
            out.append(block["md"])
            out.append(f"*{block['caption']}*")
        elif kind == "shots":
            for filename, title, caption in block["items"]:
                out.append(f"![{title}]({RAW}/web/screenshots/{filename})")
                out.append(f"*{title}. {caption}*")
        elif kind == "code":
            out.append("```bash\n" + block["text"] + "\n```")
        else:
            raise SystemExit(f"unknown block {kind}")
    out.append(
        "\n---\n\n"
        f"VayuDrishti research monograph · November 2026 · "
        f"[preview](https://gistpreview.github.io/?{gist_id}) · "
        "[repository](https://github.com/guglxni/VayuDrishti)"
    )
    return "\n\n".join(out) + "\n"


FIG_FILES = {
    "plate1": "plate1_masthead_cover.svg",
    "plate2": "plate2_architecture_pipeline.svg",
    "plate3": "plate3_diurnal_inversion_and_benchmarks.svg",
    "plate4": "plate4_boundary_layer_physics.svg",
}


HTML_CSS = """
:root {
  --paper: #F7F4EB;
  --ink: #0A1119;
  --terra: #C84B31;
  --muted: #5E564C;
  --rule: rgba(10, 17, 25, 0.16);
  --code: #EFE8D8;
  --card: #FBF8F1;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Newsreader", Georgia, "Iowan Old Style", serif;
  font-size: 1.22rem;
  line-height: 1.55;
  font-optical-sizing: auto;
}
.sr {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}
body:has(#night:checked) { background: #101418; }
#night:checked ~ .sheet {
  --paper: #101418;
  --ink: #F6F1E6;
  --terra: #E07A5F;
  --muted: #C9BBA8;
  --rule: rgba(246, 241, 230, 0.16);
  --code: #1C232B;
  --card: #181E24;
  background: var(--paper);
  color: var(--ink);
}
.sheet { max-width: 68rem; margin: 0 auto; padding: 0 1.25rem 5rem; }
.tools { display: flex; justify-content: flex-end; padding: 0.8rem 0; position: sticky; top: 0; z-index: 4; background: color-mix(in srgb, var(--paper) 88%, transparent); backdrop-filter: blur(8px); }
.tools label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid var(--ink);
  padding: 0.35rem 0.6rem;
  cursor: pointer;
}
a { color: var(--terra); text-underline-offset: 0.18em; }
.kicker, .mono, code, .caption, .tools, table, .shot-nav label {
  font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
.kicker { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 1.2rem 0; }
h1 { font-weight: 500; font-size: clamp(3.2rem, 8vw, 5.4rem); line-height: 0.95; letter-spacing: -0.03em; margin: 0.2rem 0 0.6rem; }
h1 .deva { font-family: "Tiro Devanagari Hindi", "Kohinoor Devanagari", serif; font-size: 0.62em; display: block; letter-spacing: 0; margin-bottom: 0.25rem; }
.dek { font-size: 1.45rem; line-height: 1.35; max-width: 38rem; margin: 0 0 1rem; }
.row { display: block; margin: 0 0 0.9rem; }
@media (min-width: 1040px) {
  .row { display: grid; grid-template-columns: minmax(0, 42rem) 16rem; column-gap: 2rem; align-items: start; }
  .row.wide { display: block; }
}
.sidenote {
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
  font-size: 0.86rem;
  line-height: 1.45;
  color: var(--muted);
  border-left: 2px solid var(--terra);
  padding-left: 0.75rem;
  margin: 0.4rem 0 0;
}
.eq {
  overflow-x: auto;
  margin: 1.1rem 0 1.3rem;
  font-size: 1.15rem;
}
.eq .katex-display { margin: 0.15rem 0; }
.plate { margin: 1.4rem 0 0.4rem; background: #F7F4EB; width: 100%; }
.plate svg { width: 100%; height: auto; display: block; }
.caption { font-size: 0.78rem; line-height: 1.45; color: var(--muted); max-width: 46rem; margin: 0.45rem 0 1.2rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; line-height: 1.35; margin: 0.4rem 0; }
th, td { text-align: left; padding: 0.38rem 0.45rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 500; color: var(--muted); }
td:nth-child(n+2) { text-align: right; font-variant-numeric: tabular-nums; }
table.dois td:nth-child(2), table.dois th:nth-child(2) { text-align: left; }
tr.ens td { color: var(--terra); font-weight: 500; }
pre {
  background: var(--code);
  padding: 0.9rem 1rem;
  overflow-x: auto;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.82rem;
  line-height: 1.5;
}
code { font-size: 0.84em; }
h2 { font-weight: 500; font-size: 1.8rem; letter-spacing: -0.02em; margin: 2rem 0 0.6rem; }
.gallery input { position: absolute; opacity: 0; pointer-events: none; }
.shot-nav { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }
.shot-nav label {
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-bottom: 1px solid transparent;
  padding-bottom: 0.15rem;
  cursor: pointer;
}
#view1:checked ~ .shot-nav label[for="view1"],
#view2:checked ~ .shot-nav label[for="view2"],
#view3:checked ~ .shot-nav label[for="view3"],
#view4:checked ~ .shot-nav label[for="view4"] { border-color: var(--terra); color: var(--terra); }
.frame { display: none; margin: 0; }
#view1:checked ~ .frames .f1,
#view2:checked ~ .frames .f2,
#view3:checked ~ .frames .f3,
#view4:checked ~ .frames .f4 { display: block; }
.frame img { width: 100%; height: auto; display: block; background: #ddd; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted); font-size: 0.95rem; }
"""


def ensure_katex() -> str:
    root = Path("/tmp/vd-katex")
    module = root / "node_modules" / "katex"
    if (module / "package.json").is_file():
        return str(module)
    root.mkdir(parents=True, exist_ok=True)
    install = subprocess.run(
        ["npm", "install", "--silent", "--prefix", str(root), "katex@0.16.22"],
        check=False,
        capture_output=True,
        text=True,
    )
    if install.returncode != 0 or not (module / "package.json").is_file():
        detail = (install.stderr or install.stdout or "npm install katex failed").strip()
        raise SystemExit(detail)
    return str(module)


def typeset_display(latexes: list[str]) -> list[str]:
    if not latexes:
        return []
    program = r"""
const katex = require(process.env.KATEX_MODULE);
const fs = require("fs");
const items = JSON.parse(fs.readFileSync(0, "utf8"));
const html = items.map((latex) => katex.renderToString(latex, {
  displayMode: true,
  throwOnError: true,
  output: "htmlAndMathml",
  strict: "ignore"
}));
process.stdout.write(JSON.stringify(html));
"""
    env = os.environ.copy()
    env["KATEX_MODULE"] = ensure_katex()
    result = subprocess.run(
        ["node", "-e", program],
        input=json.dumps(latexes),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "katex failed")
    rendered = json.loads(result.stdout)
    if len(rendered) != len(latexes):
        raise SystemExit("katex returned the wrong number of equations")
    for html in rendered:
        if "$" in html or 'class="katex"' not in html:
            raise SystemExit("katex HTML is missing a typeset span or still contains a dollar sign")
    return rendered


def render_html(pieces, svgs: dict[str, str], gist_id: str) -> str:
    preview = f"https://gistpreview.github.io/?{gist_id}"
    latexes = [block["latex"] for block in pieces if block["type"] == "eq"]
    rendered_eq = dict(zip(latexes, typeset_display(latexes)))
    body = []
    for block in pieces:
        kind = block["type"]
        if kind == "kicker":
            body.append(f'<p class="kicker">{esc(block["text"])}</p>')
        elif kind == "h1":
            body.append('<h1><span class="deva">वायुदृष्टि</span>VayuDrishti</h1>')
        elif kind == "dek":
            body.append(f'<p class="dek">{esc(block["text"])}</p>')
        elif kind == "h2":
            body.append(f"<h2>{esc(block['text'])}</h2>")
        elif kind == "p":
            body.append(f"<div class=\"row\"><div class=\"main\"><p>{md_inline(block['text'])}</p></div></div>")
        elif kind == "note":
            body.append(f"<div class=\"row\"><aside class=\"sidenote\">{md_inline(block['text'])}</aside></div>")
        elif kind == "eq":
            body.append('<div class="eq">' + rendered_eq[block["latex"]] + "</div>")
        elif kind == "fig":
            body.append(
                '<div class="row wide"><figure class="plate">'
                + inline_svg(svgs[block["id"]])
                + f'</figure><p class="caption">{esc(block["caption"])}</p></div>'
            )
        elif kind == "table":
            klass = "dois" if "DOI" in block["md"][:40] else ""
            body.append(
                f'<div class="row wide"><div class="{klass}">{block["html"]}</div>'
                f'<p class="caption">{esc(block["caption"])}</p></div>'
            )
        elif kind == "code":
            body.append(f"<pre><code>{esc(block['text'])}</code></pre>")
        elif kind == "shots":
            inputs, labels, frames = [], [], []
            for i, (filename, title, caption) in enumerate(block["items"], start=1):
                checked = " checked" if i == 1 else ""
                inputs.append(f'<input type="radio" name="view" id="view{i}"{checked}>')
                labels.append(f'<label for="view{i}">{i:02d}  {esc(title)}</label>')
                src = f"{RAW}/web/screenshots/{filename}"
                frames.append(
                    f'<figure class="frame f{i}"><img src="{src}" alt="{esc(title)}">'
                    f'<figcaption class="caption">{esc(title)}. {esc(caption)}</figcaption></figure>'
                )
            body.append(
                '<div class="gallery">'
                + "".join(inputs)
                + '<div class="shot-nav">' + "".join(labels) + "</div>"
                + '<div class="frames">' + "".join(frames) + "</div></div>"
            )
        else:
            raise SystemExit(kind)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>वायुदृष्टि VayuDrishti — Research Monograph</title>
<meta name="description" content="Physics-guided multi-scale PM2.5 forecasting and a 3D atmospheric digital twin for Mumbai.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Tiro+Devanagari+Hindi&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<style>
{HTML_CSS}
</style>
</head>
<body>
<input type="checkbox" id="night" class="sr">
<div class="sheet">
  <div class="tools"><label for="night">Night paper</label></div>
  {''.join(body)}
  <footer>
    <p>VayuDrishti research monograph · November 2026 · <a href="{preview}">Magazine preview</a> · <a href="https://github.com/guglxni/VayuDrishti">Repository</a></p>
  </footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
<script>
(function () {{
  if (!window.renderMathInElement) return;
  var root = document.querySelector(".sheet") || document.body;
  renderMathInElement(root, {{
    delimiters: [
      {{left: "$$", right: "$$", display: true}},
      {{left: "$", right: "$", display: false}}
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "svg"]
  }});
}})();
</script>
</body>
</html>
"""


def inline_svg(document: str) -> str:
    start = document.find("<svg")
    return document[start:]


def resolve_gist_id() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    return GIST_TOKEN


def main() -> None:
    gist_id = resolve_gist_id()
    boards = load_boards()
    w24, w72 = load_weights()
    raw = compute_diurnal()
    facts = diurnal_facts(raw)
    rows = board_rows(boards)
    wrows = weight_rows(w24, w72)
    plates = {
        "plate1": plate_masthead(facts),
        "plate2": plate_architecture(boards, w24, w72),
        "plate3": plate_diurnal(raw, facts, boards),
        "plate4": plate_coast(facts),
    }
    for key, filename in FIG_FILES.items():
        path = OUT / filename
        path.write_text(plates[key], encoding="utf-8")
        ET.parse(path)
        text_value = path.read_text(encoding="utf-8")
        for needle in (PAPER, INK, TERRA, "@keyframes"):
            if needle not in text_value:
                raise SystemExit(f"{filename} missing {needle}")
    pieces = blocks(facts, rows, wrows, w24, w72, gist_id)
    monograph = render_markdown(pieces, gist_id)
    html = render_html(pieces, plates, gist_id)
    (OUT / "00_VayuDrishti_Research_Monograph.md").write_text(monograph, encoding="utf-8")
    (OUT / "index.html").write_text(html, encoding="utf-8")
    expected = {
        "cams_peak": "82.1",
        "cams_h": "17:00",
        "obs_trough": "41.9",
        "obs_h": "14:00",
        "cams_aft": "78.8",
        "obs_aft": "42.7",
        "r": "−0.668",
    }
    for key, value in expected.items():
        if facts[key] != value:
            raise SystemExit(f"printed fact changed {key}: {facts[key]!r} != {value!r}")
    required = ["9.1531", "9.2545", "6.2616", "6.0855", "0.0", "xP5LoEHK", "sK5XpFqb", facts["r"]]
    for needle in required:
        if needle not in monograph or needle not in html:
            raise SystemExit(f"missing {needle}")
    for needle in ("82.1", "41.9", "78.8", "42.7", "9.1531", "9.2545", facts["r"], "37.9949", "38.8708"):
        if needle not in plates["plate3"]:
            raise SystemExit(f"plate3 missing {needle}")
    article = html.split("<script>", 1)[0]
    if "$" in article:
        raise SystemExit("naked dollar sign remains in the article body")
    if article.count('class="katex"') < 4:
        raise SystemExit("equations were not typeset")
    print(json.dumps({"gist_id": gist_id, "facts": facts, "files": {
        name: (OUT / name).stat().st_size
        for name in list(FIG_FILES.values()) + ["00_VayuDrishti_Research_Monograph.md", "index.html"]
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
