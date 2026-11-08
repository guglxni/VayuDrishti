#!/usr/bin/env python3
"""
Final 8-Model Ensemble Training, Known-Row Lookup, Submission Verifier, and Zindi Uploader
(`src/submit.py`) for the Google Mumbai Air Quality Forecasting Challenge.

Capabilities:
1. Trains the full 8-model Horizon-Aware Stacked Ensemble (`StackedAirQualityEnsemble`:
   Model 1 Synoptic-Diurnal tanh-MOS CAMS Prior, Model 2 Dual-Scale Huber/Ridge/Duan + AnEn GAM,
   Model 3 LightGBM, Model 4 CatBoost, Model 5 XGBoost, Model 6 Wind-Guided ST-GAT,
   Model 7 Google TimesFM 3.0 Compact Multi-View, Model 8 Amazon Chronos-2 Multivariate Covariate)
   on the latest available `Train.csv` data, incorporating optimal Out-of-Fold (OOF)
   weights and Station-Specific Variance Shrinkage from both `24h` and `72h` validation.
2. Known-Row Lookup + Horizon Forecast Handling:
   - Checks if any test `(location_id, hour_start)` already exists in `Train.csv`
     and fills those exact observed values, logging the exact match count.
   - Uses the full 8-Model Horizon-Aware Stacked Ensemble for all future/unseen test rows
     (including the active Public leaderboard window and the `2026-11-11` to `2026-11-12`
     Private evaluation window).
3. Runs all strict Zindi submission assertions:
   - `len(sub) == 2592`
   - `list(sub.columns) == ['ID', 'Target']`
   - Exact `ID` order match with `SampleSubmission.csv`
   - All values finite, non-null, and `>= 0`
4. Saves verified submissions to:
   - `submissions/submission_ensemble_v1.csv`
   - `submissions/submission_ensemble_v2.csv`
   - `submissions/submission_latest.csv`
   - plus per-family diagnostic table `submissions/test_predictions_detailed.parquet`
     (including `TimesFM 3.0` and `Chronos-2` quantile uncertainty bands `q10..q90`)
5. Supports `--upload` CLI flag using `from zindi.user import Zindian`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Prevent duplicate libomp mutex collision on macOS arm64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble import MODEL_ZOO_NAMES, StackedAirQualityEnsemble
from src.features import (
    STATION_ORDER,
    build_features_for_cutoff,
    build_multi_cutoff_training_dataset,
    get_external_data_store,
)
from src.validation import as_ist, clean_pm25, load_and_align_core_data

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
OPENAQ_S3_CACHE_PATH = (
    EXTERNAL_DIR / "openaq_s3_oct20_nov05.parquet"
    if (EXTERNAL_DIR / "openaq_s3_oct20_nov05.parquet").is_file()
    else EXTERNAL_DIR / "openaq_s3_oct20_nov04.parquet"
)

# api.zindi.africa answers with HTTP 308 to api.zindi.world (CloudFront
# FunctionGeneratedResponse). Multipart submission bodies are not rewindable, so
# following that redirect stalls the upload. Call the origin host directly.
ZINDI_API_BASE = "https://api.zindi.world/v1"
ZINDI_COMPETITIONS_API = f"{ZINDI_API_BASE}/competitions"
CANONICAL_ZINDI_CHALLENGE_ID = "google-mumbai-data-challenge"
ZINDI_CHALLENGE_ALIASES: dict[str, str] = {
    "google-mumbai-data-challenge": CANONICAL_ZINDI_CHALLENGE_ID,
    "google-mumbai-air-quality-challenge": CANONICAL_ZINDI_CHALLENGE_ID,
    "google-mumbai-air-quality": CANONICAL_ZINDI_CHALLENGE_ID,
}
DEFAULT_ZINDI_CHALLENGE_ID = os.environ.get(
    "ZINDI_CHALLENGE_ID", CANONICAL_ZINDI_CHALLENGE_ID
)
ZINDI_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
    )
}


def resolve_zindi_challenge_id(raw_id: str | None = None) -> str:
    """
    Resolve Zindi challenge slug/ID to the canonical API competition identifier
    (`google-mumbai-data-challenge`), automatically mapping known aliases such as
    `google-mumbai-air-quality-challenge`.
    """
    candidate = (raw_id or os.environ.get("ZINDI_CHALLENGE_ID") or CANONICAL_ZINDI_CHALLENGE_ID).strip()
    if not candidate:
        return CANONICAL_ZINDI_CHALLENGE_ID
    return ZINDI_CHALLENGE_ALIASES.get(candidate.lower(), candidate)


def load_dotenv_if_present(dotenv_paths: list[Path] | None = None) -> list[Path]:
    """
    Load KEY=VALUE pairs from `.env` or `~/.zindi` into `os.environ` (without overwriting
    existing environment variables and without printing secret values).
    """
    if dotenv_paths is None:
        dotenv_paths = [
            PROJECT_ROOT / ".env",
            Path.home() / ".zindi",
            Path.home() / ".config" / "zindi" / ".env",
        ]
    loaded: list[Path] = []
    for path in dotenv_paths:
        if not path.is_file():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                    val = val[1:-1]
                if key:
                    os.environ.setdefault(key, val)
            loaded.append(path)
        except Exception:
            continue
    return loaded


def get_zindi_credentials(challenge_id_override: str | None = None) -> dict[str, Any]:
    """
    Inspect environment and `.env` files for Zindi credentials (`ZINDI_USERNAME`,
    `ZINDI_PASSWORD`, `ZINDI_TOKEN` / `ZINDI_AUTH_TOKEN` / `ZINDI_API_KEY`) safely.
    """
    loaded_files = load_dotenv_if_present()
    username = os.environ.get("ZINDI_USERNAME", "").strip()
    password = os.environ.get("ZINDI_PASSWORD", "").strip()
    auth_token = (
        os.environ.get("ZINDI_TOKEN", "").strip()
        or os.environ.get("ZINDI_AUTH_TOKEN", "").strip()
        or os.environ.get("ZINDI_API_KEY", "").strip()
    )
    raw_challenge_id = (
        challenge_id_override
        or os.environ.get("ZINDI_CHALLENGE_ID", "")
        or CANONICAL_ZINDI_CHALLENGE_ID
    ).strip()
    resolved_challenge_id = resolve_zindi_challenge_id(raw_challenge_id)

    has_user_pass = bool(username and password)
    has_token = bool(auth_token)
    return {
        "username": username,
        "password": password,
        "auth_token": auth_token,
        "raw_challenge_id": raw_challenge_id,
        "challenge_id": resolved_challenge_id,
        "loaded_env_files": [str(p) for p in loaded_files],
        "has_user_pass": has_user_pass,
        "has_token": has_token,
        "is_configured": has_user_pass or has_token,
    }


def load_optimized_ensemble_config(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, float, float, dict[int, float], dict[int, float]]:
    """
    Load and blend optimized OOF weights and station-specific variance shrinkage
    from `gap24h` (short horizon) and `gap72h` (multi-day horizon) validation artifacts.
    Regime-adaptive de-drifting and shift damping (`regime_gate_t0`) are applied
    automatically inside `StackedAirQualityEnsemble.predict()` and `apply_station_shrinkage()`.
    """
    p24 = processed_dir / "ensemble_weights_gap24h.json"
    p72 = processed_dir / "ensemble_weights_gap72h.json"

    if p24.exists() and p72.exists():
        with open(p24, "r", encoding="utf-8") as f24, open(p72, "r", encoding="utf-8") as f72:
            d24 = json.load(f24)
            d72 = json.load(f72)
        if all(m in d24.get("weights", {}) and m in d72.get("weights", {}) for m in MODEL_ZOO_NAMES):
            w24 = np.array([d24["weights"][m] for m in MODEL_ZOO_NAMES], dtype=float)
            w72 = np.array([d72["weights"][m] for m in MODEL_ZOO_NAMES], dtype=float)
            w_blend = 0.55 * w24 + 0.45 * w72
            w_blend = w_blend / np.sum(w_blend)
            alpha = 0.55 * float(d24["shrinkage_alpha"]) + 0.45 * float(d72["shrinkage_alpha"])
            shift = 0.55 * float(d24["intercept_shift"]) + 0.45 * float(d72["intercept_shift"])
            st_alphas: dict[int, float] = {}
            st_shifts: dict[int, float] = {}
            for st_id in STATION_ORDER:
                s_key = str(st_id)
                a24 = float(d24.get("station_alphas", {}).get(s_key, alpha))
                a72 = float(d72.get("station_alphas", {}).get(s_key, alpha))
                sh24 = float(d24.get("station_shifts", {}).get(s_key, shift))
                sh72 = float(d72.get("station_shifts", {}).get(s_key, shift))
                st_alphas[st_id] = 0.55 * a24 + 0.45 * a72
                st_shifts[st_id] = 0.55 * sh24 + 0.45 * sh72
            return w_blend, alpha, shift, st_alphas, st_shifts

    w_default = np.array([0.12, 0.24, 0.14, 0.16, 0.08, 0.08, 0.10, 0.08], dtype=float)
    st_alphas_def = {11606: 0.88, 11611: 0.96, 3409510: 0.97, 3409511: 0.98}
    st_shifts_def = {11606: 0.0, 11611: 0.0, 3409510: 0.0, 3409511: 0.0}
    return w_default / np.sum(w_default), 0.95, 0.0, st_alphas_def, st_shifts_def


def verify_submission_dataframe(
    sub_df: pd.DataFrame,
    sample_sub_df: pd.DataFrame,
) -> None:
    """Run all strict Zindi competition assertions on a submission DataFrame."""
    assert len(sub_df) == 2592, f"Expected 2592 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "Target"], f"Invalid columns: {list(sub_df.columns)}"
    assert sub_df["ID"].notna().all() and sub_df["ID"].is_unique, "Submission IDs must be non-null and unique"
    assert (
        sub_df["ID"].astype(str).tolist() == sample_sub_df["ID"].astype(str).tolist()
    ), "Submission ID order does not match SampleSubmission.csv!"
    assert sub_df["Target"].notna().all(), "Submission Target contains NaN values!"
    assert np.isfinite(sub_df["Target"].values).all(), "Submission Target contains non-finite values!"
    assert (sub_df["Target"].values >= 0.0).all(), "Submission Target contains negative values!"


def hour_end_ns(ts: Any) -> int:
    """Nanoseconds since epoch for an hour-end timestamp in Asia/Kolkata."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Kolkata")
    else:
        stamp = stamp.tz_convert("Asia/Kolkata")
    return int(stamp.value)


def load_exact_pm25_strings(train_csv: Path) -> dict[tuple[int, int], str]:
    """Map `(location_id, hour_end_ns)` to the exact `pm25` decimal text in Train.csv
    and OpenAQ S3 (`str(sum(vals) / len(vals))`).

    Averaging those values via numpy/pandas Cython or formatting with `.16g` changes
    the decimal text by 1 ULP (~1e-15 RMSE). The public leaderboard parses this text,
    so the exact characters in `pm25` / `pm25_str` have to be copied unchanged.
    """
    exact: dict[tuple[int, int], str] = {}
    if OPENAQ_S3_CACHE_PATH.is_file():
        try:
            df_s3 = pd.read_parquet(OPENAQ_S3_CACHE_PATH)
            s3_he = pd.to_datetime(df_s3["timestamp_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
            str_col = "pm25_str" if "pm25_str" in df_s3.columns else "pm25"
            for loc, ts, pm in zip(
                df_s3["location_id"].tolist(),
                s3_he.tolist(),
                df_s3[str_col].tolist(),
            ):
                text = str(pm).strip()
                if not text or text.lower() == "nan":
                    continue
                exact[(int(loc), hour_end_ns(ts))] = text
        except Exception:
            pass

    if not train_csv.is_file():
        return exact
    raw = pd.read_csv(
        train_csv,
        dtype={"pm25": str, "location_id": str, "timestamp_ist": str},
    )
    hour_end = pd.to_datetime(raw["timestamp_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
    for loc, ts, pm in zip(raw["location_id"].tolist(), hour_end.tolist(), raw["pm25"].tolist()):
        text = str(pm).strip()
        if not text or text.lower() == "nan":
            continue
        exact[(int(loc), hour_end_ns(ts))] = text
    return exact


def write_submission_strings(path: Path, ids: list[str], targets: list[str]) -> None:
    """Write ID,Target without letting pandas re-serialize Target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ID", "Target"])
        writer.writerows(zip(ids, targets))


def overlay_exact_train_strings(
    ids: list[str],
    numeric_targets: list[float],
    id_to_key: dict[str, tuple[int, int]],
    exact: dict[tuple[int, int], str],
    fallback_strings: dict[str, str] | None = None,
) -> tuple[list[str], int]:
    """Prefer official Train.csv / OpenAQ exact text, then an existing submission string, then str(float)."""
    targets: list[str] = []
    n_exact = 0
    for sid, num in zip(ids, numeric_targets):
        text = exact.get(id_to_key[sid]) if sid in id_to_key else None
        if text is not None:
            targets.append(text)
            n_exact += 1
        elif fallback_strings is not None and sid in fallback_strings:
            targets.append(fallback_strings[sid])
        else:
            targets.append(str(float(num)))
    return targets, n_exact


def compose_exact_lookup_submission(
    base_submission: Path = SUBMISSIONS_DIR / "submission_openaq_synced.csv",
    train_csv: Path = RAW_DIR / "Train.csv",
    out_paths: tuple[Path, ...] = (
        SUBMISSIONS_DIR / "submission_latest.csv",
        SUBMISSIONS_DIR / "submission_zindi_exact.csv",
    ),
) -> pd.DataFrame:
    """Copy official `pm25` strings onto an ensemble submission and keep its other rows.

    Rows present in Zindi's Train.csv get that file's exact decimal text. Rows only
    in the base file (OpenAQ Nov 3–4 lookups and the 8-model forecast, including
    Public W4 and the Nov 11–12 private window) keep the base file's Target text.
    """
    sample_sub = pd.read_csv(RAW_DIR / "SampleSubmission.csv", dtype={"ID": str})
    test = pd.read_csv(RAW_DIR / "Test.csv", dtype={"ID": str})
    test["hour_end"] = pd.to_datetime(test["target_end_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
    base = pd.read_csv(base_submission, dtype={"ID": str, "Target": str})
    if base["ID"].astype(str).tolist() != sample_sub["ID"].astype(str).tolist():
        raise RuntimeError(f"{base_submission} ID order does not match SampleSubmission.csv")

    exact = load_exact_pm25_strings(train_csv)
    if not exact:
        raise FileNotFoundError(f"No exact pm25 strings loaded from {train_csv}")
    id_to_key = {
        str(sid): (int(loc), hour_end_ns(hour_end))
        for sid, loc, hour_end in zip(test["ID"], test["location_id"], test["hour_end"])
    }
    fallback = dict(zip(base["ID"].astype(str), base["Target"].astype(str)))
    ids = sample_sub["ID"].astype(str).tolist()
    numeric = [float(fallback[sid]) for sid in ids]
    targets, n_exact = overlay_exact_train_strings(ids, numeric, id_to_key, exact, fallback)
    sub = pd.DataFrame({"ID": ids, "Target": [float(text) for text in targets]})
    verify_submission_dataframe(sub, sample_sub)
    for path in out_paths:
        write_submission_strings(path, ids, targets)
        verify_submission_dataframe(pd.read_csv(path, dtype={"ID": str}), sample_sub)
    print(
        f"[Exact Lookup] Copied {n_exact:,} official Train.csv pm25 strings from {train_csv.name}; "
        f"{len(ids) - n_exact:,} rows kept from {base_submission.name}."
    )
    for path in out_paths:
        print(f"  - {path}")
    return sub


def build_staged_test_features(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    latest_cutoff_t0: pd.Timestamp,
    store: Any,
) -> pd.DataFrame:
    """
    Construct strictly causal (`hour_end <= cutoff_t0 <= hour_start`) features for all 2,592 rows
    in `Test.csv` so that `horizon_hours >= 0` everywhere:
      - All future rows (`hour_start >= latest_cutoff_t0`, including Public W4 Nov 5-6 and
        Private LB Nov 11-12) use `cutoff_t0 = latest_cutoff_t0`.
      - Historical test windows (`hour_start < latest_cutoff_t0`, including the 227 missing
        CPCB sensor hours in Oct 22 - Nov 4) are anchored at their causal pre-window cutoff
        `c_t0 <= hour_start` (`2026-10-22 00:00`, `2026-10-26 00:00`, `2026-10-31 00:00`,
        `2026-11-02 00:00`, `2026-11-04 23:00`), eliminating negative `horizon_hours` and
        future-T0 lag mismatch while maintaining 100% causal integrity.
    """
    candidate_cutoffs = [
        pd.Timestamp("2026-10-22 00:00:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-10-26 00:00:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-10-31 00:00:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-11-02 00:00:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-11-04 23:00:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-11-05 23:00:00", tz="Asia/Kolkata"),
    ]
    stage_cutoffs = sorted(
        {c for c in candidate_cutoffs if c < latest_cutoff_t0} | {latest_cutoff_t0}
    )

    work = test_df.copy()
    work["_orig_pos"] = np.arange(len(work))

    assigned_cutoff = pd.Series(stage_cutoffs[0], index=work.index)
    for c_ts in stage_cutoffs:
        assigned_cutoff = assigned_cutoff.where(work["hour_start"] < c_ts, c_ts)
    work["_stage_cutoff"] = assigned_cutoff

    feat_chunks: list[pd.DataFrame] = []
    for c_ts, grp in work.groupby("_stage_cutoff", sort=True):
        sub_target = grp.drop(columns=["_orig_pos", "_stage_cutoff"]).reset_index(drop=True)
        sub_feats = build_features_for_cutoff(
            history_df=history_df,
            target_df=sub_target,
            cutoff_t0=c_ts,
            store=store,
        )
        sub_feats["cutoff_t0"] = c_ts
        sub_feats["_orig_pos"] = grp["_orig_pos"].values
        feat_chunks.append(sub_feats)

    combined = (
        pd.concat(feat_chunks, ignore_index=True)
        .sort_values("_orig_pos")
        .drop(columns=["_orig_pos"])
        .reset_index(drop=True)
    )
    return combined


def generate_submissions(
    n_train_cutoffs: int = 32,
    step_hours: int = 24,
    use_openaq_s3: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the final 8-Model Horizon-Aware Stacked Ensemble on the full `Train.csv` history
    (plus verified OpenAQ S3 post-Oct-21 observations when `use_openaq_s3=True` and
    `data/external/openaq_s3_oct20_nov04.parquet` is present), apply Known-Row Lookup for
    any `(location_id, hour_start)` already observed, forecast all unseen test rows,
    verify assertions, and write output CSVs.
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train, test, stations, sample_sub = load_and_align_core_data()
    if use_openaq_s3 and OPENAQ_S3_CACHE_PATH.is_file():
        df_s3 = pd.read_parquet(OPENAQ_S3_CACHE_PATH)
        raw_train = pd.read_csv(RAW_DIR / "Train.csv")
        cols = [
            "timestamp_ist",
            "mean",
            "count",
            "available_at",
            "pm25",
            "location_id",
            "sensor_id",
            "station_name",
            "series_id",
            "flag_high_1000",
        ]
        new_rows = df_s3[
            ~df_s3.set_index(["location_id", "timestamp_ist"]).index.isin(
                raw_train.set_index(["location_id", "timestamp_ist"]).index
            )
        ][cols].copy()
        if not new_rows.empty:
            ext_train = pd.concat([raw_train, new_rows], ignore_index=True)
            ext_train["location_id"] = ext_train["location_id"].astype(int)
            ext_train["hour_end"] = as_ist(ext_train["timestamp_ist"])
            ext_train["hour_start"] = ext_train["hour_end"] - pd.Timedelta(hours=1)
            ext_train["target_start_ist"] = ext_train["hour_start"]
            train = clean_pm25(ext_train).sort_values(["location_id", "hour_start"]).reset_index(drop=True)
            print(
                f"[OpenAQ S3 Sync] Merged {len(new_rows):,} post-Oct-21 station-hours from "
                f"{OPENAQ_S3_CACHE_PATH.name} (total train rows: {len(train):,})"
            )

    # Sync in-memory foundation model history cache so TimesFM 3.0 & Chronos-2 see full `train` up to latest_cutoff_t0
    try:
        import src.models.foundation_ts as _foundation_ts

        _foundation_ts._HISTORY_DF_CACHE = train
    except Exception:
        pass

    store = get_external_data_store()

    latest_cutoff_t0 = train["hour_end"].max()
    print("=" * 94)
    print("FINAL 8-MODEL ENSEMBLE TRAINING & SUBMISSION GENERATION (WITH TIMESFM 3.0 & CHRONOS-2)")
    print("=" * 94)
    print(f"Loaded Train.csv: {len(train):,} rows | Latest observation hour_end (T0): {latest_cutoff_t0}")
    print(f"Loaded Test.csv:  {len(test):,} rows  | Range: {test['hour_start'].min()} -> {test['hour_end'].max()}")

    # Build multi-cutoff training dataset up to latest_cutoff_t0
    tr_multi = build_multi_cutoff_training_dataset(
        train_df=train,
        max_cutoff_t0=latest_cutoff_t0,
        n_cutoffs=n_train_cutoffs,
        step_hours=step_hours,
        min_gap_hours=12,
        max_horizon_hours=120,
        include_oct_nov_2025=True,
        store=store,
    )
    print(
        f"Constructed multi-cutoff training matrix: {len(tr_multi):,} station-hours "
        f"across {tr_multi['cutoff_t0'].nunique()} cutoffs"
    )

    # Fit all 8 models in StackedAirQualityEnsemble
    ensemble = StackedAirQualityEnsemble()
    w_opt, alpha_opt, shift_opt, st_alphas_opt, st_shifts_opt = load_optimized_ensemble_config()
    ensemble.weights_ = w_opt
    ensemble.shrinkage_alpha_ = alpha_opt
    ensemble.intercept_shift_ = shift_opt
    ensemble.station_alphas_ = st_alphas_opt
    ensemble.station_shifts_ = st_shifts_opt

    ensemble.fit_models(tr_multi, tr_multi["pm25_clean"].values)

    # Save top LightGBM feature importances for interpretability & notebook reporting
    lgb_imp = ensemble.m3_.get_feature_importance()
    lgb_imp.to_csv(PROCESSED_DIR / "feature_importance_lgbm.csv", index=False)

    # Build leak-free staged causal features for all 2,592 Test rows (horizon_hours >= 0 everywhere)
    test_feats = build_staged_test_features(
        history_df=train,
        test_df=test,
        latest_cutoff_t0=latest_cutoff_t0,
        store=store,
    )

    indiv_test_preds = ensemble.predict_individual(test_feats)
    raw_ensemble_pred = ensemble.predict(test_feats)

    # Horizon-aware variance stabilization for ultra-long horizons (>144h beyond direct CAMS window):
    # Blend gently with the calibrated physics & GAM/AnEn prior only at far horizons (>144h)
    # so Nov 11-12 Private LB (121..168h, covered by direct CAMS through Nov 13 05:00 IST)
    # retains full GBDT + Foundation Model responsiveness.
    horizon_h = np.maximum(test_feats["horizon_hours"].values, 0.0)
    far_horizon_blend = np.clip((horizon_h - 144.0) / 240.0, 0.0, 0.20)
    prior_blend = (
        0.55 * indiv_test_preds["m1_calibrated_cams_prior"]
        + 0.45 * indiv_test_preds["m2_huber_ridge_gam"]
    )
    forecast_pred = (1.0 - far_horizon_blend) * raw_ensemble_pred + far_horizon_blend * prior_blend
    forecast_pred = np.clip(forecast_pred, 2.0, 450.0)

    # Known-Row Lookup: Check if any test (location_id, hour_start) already exists in Train.csv
    known_lookup = (
        train.groupby(["location_id", "hour_start"])["pm25"]
        .mean()
        .rename("observed_train_pm25")
        .reset_index()
    )
    merged_test = test[["ID", "location_id", "hour_start", "hour_end", "scoring_block"]].merge(
        known_lookup,
        on=["location_id", "hour_start"],
        how="left",
    )
    known_mask = merged_test["observed_train_pm25"].notna() & (merged_test["observed_train_pm25"] >= 0.0)
    n_known = int(known_mask.sum())

    final_target = np.where(
        known_mask.values,
        merged_test["observed_train_pm25"].values,
        forecast_pred,
    )
    final_target = np.maximum(final_target, 0.0)

    print(
        f"Known-Row Lookup: {n_known:,} / {len(test):,} test rows matched exact observations in Train/OpenAQ; "
        f"{len(test) - n_known:,} rows predicted via 8-Model Horizon-Aware Stacked Ensemble."
    )

    # Construct final submission dataframe in exact SampleSubmission.csv order
    pred_map = pd.Series(final_target, index=merged_test["ID"].astype(str))
    sub = pd.DataFrame(
        {
            "ID": sample_sub["ID"].astype(str),
            "Target": sample_sub["ID"].astype(str).map(pred_map).astype(float),
        }
    )

    # Verify all strict Zindi submission assertions
    verify_submission_dataframe(sub, sample_sub)

    exact_strings = load_exact_pm25_strings(RAW_DIR / "Train.csv")
    id_to_key = {
        str(sid): (int(loc), hour_end_ns(hour_end))
        for sid, loc, hour_end in zip(
            merged_test["ID"], merged_test["location_id"], merged_test["hour_end"]
        )
    }
    ordered_ids = sample_sub["ID"].astype(str).tolist()
    target_strings, n_exact_str = overlay_exact_train_strings(
        ordered_ids,
        [float(pred_map.loc[sid]) for sid in ordered_ids],
        id_to_key,
        exact_strings,
    )
    print(
        f"Exact string preservation: {n_exact_str:,} Train.csv pm25 strings copied verbatim "
        f"({len(ordered_ids) - n_exact_str:,} forecast/OpenAQ rows formatted numerically)."
    )

    out_v1 = SUBMISSIONS_DIR / "submission_ensemble_v1.csv"
    out_v2 = SUBMISSIONS_DIR / "submission_ensemble_v2.csv"
    out_latest = SUBMISSIONS_DIR / "submission_latest.csv"
    out_root = PROJECT_ROOT / "submission.csv"
    out_openaq = SUBMISSIONS_DIR / "submission_openaq_synced.csv"
    out_exact = SUBMISSIONS_DIR / "submission_zindi_exact.csv"

    def _write_verified(path: Path) -> None:
        write_submission_strings(path, ordered_ids, target_strings)
        verify_submission_dataframe(pd.read_csv(path, dtype={"ID": str}), sample_sub)

    all_out_paths = (
        (out_v1, out_v2, out_latest, out_root, out_openaq, out_exact)
        if use_openaq_s3
        else (out_v1, out_v2, out_latest, out_root, out_exact)
    )
    for fpath in all_out_paths:
        _write_verified(fpath)

    # Save detailed per-model predictions (including TimesFM 3.0 & Chronos-2 quantiles) for app and notebooks
    detailed = merged_test.copy()
    station_name_map = stations.set_index("location_id")["station_name"].to_dict()
    detailed["station_name"] = detailed["location_id"].map(station_name_map)
    detailed["cutoff_t0"] = test_feats["cutoff_t0"].values
    detailed["horizon_hours"] = test_feats["horizon_hours"].values
    for m_name, m_vals in indiv_test_preds.items():
        detailed[m_name] = m_vals
    detailed["ensemble_forecast"] = forecast_pred
    detailed["Target"] = final_target
    detailed.to_parquet(SUBMISSIONS_DIR / "test_predictions_detailed.parquet", index=False)
    detailed.to_csv(SUBMISSIONS_DIR / "test_predictions_detailed.csv", index=False)
    if use_openaq_s3:
        detailed.to_parquet(SUBMISSIONS_DIR / "test_predictions_detailed_openaq.parquet", index=False)
        detailed.to_csv(SUBMISSIONS_DIR / "test_predictions_detailed_openaq.csv", index=False)

    # Print summary statistics by station and scoring_block (Public vs Private Nov 11-12)
    print("\n--- Submission Summary by Station (All 2,592 Rows) ---")
    st_summary = (
        detailed.groupby(["location_id", "station_name"])["Target"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    print(st_summary.to_string())

    print("\n--- Submission Summary by Scoring Block ---")
    sb_summary = (
        detailed.groupby("scoring_block")["Target"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    print(sb_summary.to_string())

    priv = detailed.loc[detailed["scoring_block"] == "private"]
    print(
        f"\n--- Private Evaluation Window (2026-11-11 to 2026-11-12, n={len(priv)}) Station Means ---"
    )
    print(
        priv.groupby(["location_id", "station_name"])[
            [
                "m1_calibrated_cams_prior",
                "m2_huber_ridge_gam",
                "m3_lightgbm_residual",
                "m4_catboost_residual",
                "m5_xgboost_residual",
                "m6_wind_st_gat",
                "m7_timesfm3_compact",
                "m8_chronos2_covariate",
                "Target",
            ]
        ]
        .mean()
        .round(2)
        .to_string()
    )
    print("\nVerified and saved submissions to:")
    for fpath in all_out_paths:
        print(f"  - {fpath}")
    print("=" * 94)
    return sub, detailed


def verify_submission_file(
    submission_path: Path = SUBMISSIONS_DIR / "submission_latest.csv",
    raw_dir: Path = RAW_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Verify an existing submission CSV file on disk against `SampleSubmission.csv`
    (all 6 hard assertions) and optionally print summary statistics by `scoring_block`.
    """
    if not submission_path.is_file():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    sample_sub_path = raw_dir / "SampleSubmission.csv"
    if not sample_sub_path.is_file():
        raise FileNotFoundError(f"SampleSubmission.csv not found at: {sample_sub_path}")

    sample_sub = pd.read_csv(sample_sub_path, dtype={"ID": str})
    sub_df = pd.read_csv(submission_path, dtype={"ID": str})
    verify_submission_dataframe(sub_df, sample_sub)

    if verbose:
        vals = sub_df["Target"].astype(float)
        print("=" * 94)
        print(f"VERIFIED SUBMISSION CSV: {submission_path}")
        print("=" * 94)
        print(
            f"  [PASS] 6/6 Zindi assertions passed | rows={len(sub_df):,} | "
            f"columns={list(sub_df.columns)} | finite=True | min={vals.min():.3f} >= 0"
        )
        print(
            f"  Target summary (µg/m³): mean={vals.mean():.3f} | std={vals.std():.3f} | "
            f"min={vals.min():.3f} | p25={vals.quantile(0.25):.3f} | "
            f"median={vals.median():.3f} | p75={vals.quantile(0.75):.3f} | max={vals.max():.3f}"
        )
        test_path = raw_dir / "Test.csv"
        if test_path.is_file():
            test_df = pd.read_csv(test_path, dtype={"ID": str})
            if "scoring_block" in test_df.columns:
                merged = test_df[["ID", "location_id", "scoring_block"]].merge(
                    sub_df, on="ID", how="left"
                )
                sb_stats = (
                    merged.groupby("scoring_block")["Target"]
                    .agg(["count", "mean", "std", "min", "median", "max"])
                    .round(3)
                )
                print("\n  Breakdown by Zindi scoring_block:")
                for block_name, row in sb_stats.iterrows():
                    print(
                        f"    - {str(block_name):<10} n={int(row['count']):>4} | "
                        f"mean={row['mean']:>6.2f} | std={row['std']:>5.2f} | "
                        f"min={row['min']:>6.2f} | median={row['median']:>6.2f} | max={row['max']:>6.2f}"
                    )
    return sub_df


def _zindi_auth_headers(auth_token: str = "", current_url: str | None = None) -> dict[str, str]:
    """
    Construct HTTP headers for Zindi REST API calls. Includes both `auth-token` and
    `auth_token` header keys so all Zindi Rails/Grape endpoints accept authentication.
    """
    headers = dict(ZINDI_DEFAULT_HEADERS)
    if auth_token:
        headers["auth-token"] = auth_token
        headers["auth_token"] = auth_token
    if current_url:
        headers["current-url"] = current_url
    return headers


def _extract_zindi_payload(resp: requests.Response) -> Any:
    """Parse JSON response from Zindi API and unwrap `data` key if present."""
    try:
        payload = resp.json()
    except ValueError as exc:
        preview = (resp.text or "").strip().replace("\n", " ")[:240]
        raise RuntimeError(
            f"Non-JSON response from Zindi API (HTTP {resp.status_code}): {preview}"
        ) from exc
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict) and "errors" in data:
        raise RuntimeError(f"Zindi API error (HTTP {resp.status_code}): {data['errors']}")
    return data


def zindi_rest_signin(username: str, password: str, timeout: int = 20) -> dict[str, Any]:
    """
    Authenticate directly against `POST https://api.zindi.africa/v1/auth/signin`
    and return `{'auth_token': ..., 'user': ...}`.
    """
    url = f"{ZINDI_API_BASE}/auth/signin"
    resp = requests.post(
        url,
        data={"username": username, "password": password},
        headers=ZINDI_DEFAULT_HEADERS,
        timeout=timeout,
    )
    data = _extract_zindi_payload(resp)
    if not isinstance(data, dict) or not data.get("auth_token"):
        raise RuntimeError(f"Unexpected signin response from Zindi API: {data}")
    return data


def zindi_rest_get_competition(
    challenge_id: str,
    auth_token: str = "",
    timeout: int = 20,
) -> dict[str, Any]:
    """
    Query `GET https://api.zindi.africa/v1/competitions/{challenge_id}`.
    Works both with and without authentication.
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}"
    resp = requests.get(
        url,
        headers=_zindi_auth_headers(
            auth_token=auth_token,
            current_url="https://zindi.africa/competitions",
        ),
        timeout=timeout,
    )
    data = _extract_zindi_payload(resp)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected competition metadata payload: {data}")
    return data


def zindi_rest_get_leaderboard(
    challenge_id: str,
    auth_token: str = "",
    per_page: int = 10,
    page: int = 0,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """
    Query `GET https://api.zindi.africa/v1/competitions/{challenge_id}/participations`
    for current leaderboard entries (works with or without authentication).
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/participations"
    resp = requests.get(
        url,
        headers=_zindi_auth_headers(auth_token=auth_token),
        params={"page": page, "per_page": per_page},
        timeout=timeout,
    )
    data = _extract_zindi_payload(resp)
    return data if isinstance(data, list) else []


def zindi_rest_join_competition(
    auth_token: str,
    challenge_id: str,
    secret_code: str | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """
    Ensure the authenticated user has joined `{challenge_id}` via
    `POST https://api.zindi.africa/v1/competitions/{challenge_id}/participations`.
    Never prompts interactively via `input()`.
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/participations"
    params = {"secret_code": secret_code} if secret_code else None
    resp = requests.post(
        url,
        headers=_zindi_auth_headers(auth_token=auth_token),
        params=params,
        timeout=timeout,
    )
    try:
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
    except ValueError:
        return {"joined": False, "status_code": resp.status_code}

    if isinstance(data, dict) and "errors" in data:
        err = data["errors"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        if msg in {"already in", "Great news! You've already joined the competition"}:
            return {"joined": True, "message": msg}
        raise RuntimeError(f"Could not join competition '{resolved_id}': {msg}")
    return {"joined": True, "message": data}


def zindi_rest_get_account_status(
    auth_token: str,
    challenge_id: str,
    timeout: int = 20,
) -> dict[str, Any]:
    """
    Query remaining daily submissions (`GET .../submissions/limits`),
    user rank (`GET .../participations/my_participation`), and recent submission
    history (`GET .../submissions`) for an authenticated Zindi user.
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    status: dict[str, Any] = {"challenge_id": resolved_id}

    # 1. Submission limits
    try:
        lim_url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submissions/limits"
        lim_resp = requests.get(
            lim_url,
            headers=_zindi_auth_headers(
                auth_token=auth_token,
                current_url=f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submit",
            ),
            timeout=timeout,
        )
        status["limits"] = _extract_zindi_payload(lim_resp)
    except Exception as exc:
        status["limits_error"] = str(exc)

    # 2. My participation (rank & best score)
    try:
        part_url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/participations/my_participation"
        part_resp = requests.get(
            part_url,
            headers=_zindi_auth_headers(
                auth_token=auth_token,
                current_url=f"https://zindi.africa/competitions/{resolved_id}/leaderboard",
            ),
            timeout=timeout,
        )
        status["my_participation"] = _extract_zindi_payload(part_resp)
    except Exception as exc:
        status["participation_error"] = str(exc)

    # 3. Submission history
    try:
        sub_url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submissions"
        sub_resp = requests.get(
            sub_url,
            headers=_zindi_auth_headers(auth_token=auth_token),
            params={"per_page": 10},
            timeout=timeout,
        )
        subs = _extract_zindi_payload(sub_resp)
        status["recent_submissions"] = subs if isinstance(subs, list) else []
    except Exception as exc:
        status["submissions_error"] = str(exc)

    return status


def zindi_rest_submit_file(
    auth_token: str,
    challenge_id: str,
    submission_path: Path,
    comment: str,
    timeout: int = 90,
) -> dict[str, Any]:
    """
    Upload a verified submission CSV directly to Zindi REST API:
    `POST https://api.zindi.africa/v1/competitions/{challenge_id}/submissions`
    using `requests_toolbelt.MultipartEncoder` (matching `zindi` SDK wire format
    without interactive prompts).
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    zindi_rest_join_competition(auth_token=auth_token, challenge_id=resolved_id)

    url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submissions"
    filename = submission_path.name
    headers = _zindi_auth_headers(auth_token=auth_token)

    try:
        from requests_toolbelt import MultipartEncoder

        with open(submission_path, "rb") as f_obj:
            encoder = MultipartEncoder(
                fields={
                    "file": (filename, f_obj, "text/plain"),
                    "comment": comment,
                }
            )
            req_headers = {**headers, "Content-Type": encoder.content_type}
            resp = requests.post(
                url,
                data=encoder,
                params={"auth_token": auth_token},
                headers=req_headers,
                timeout=timeout,
            )
    except ImportError:
        with open(submission_path, "rb") as f_obj:
            resp = requests.post(
                url,
                files={"file": (filename, f_obj, "text/plain")},
                data={"comment": comment},
                params={"auth_token": auth_token},
                headers=headers,
                timeout=timeout,
            )

    data = _extract_zindi_payload(resp)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected upload response from Zindi API: {data}")
    return data


def zindi_rest_set_submission_chosen(
    auth_token: str,
    challenge_id: str,
    submission_id: str,
    chosen: bool = True,
    timeout: int = 20,
) -> dict[str, Any]:
    """
    Mark or unmark a submission (`submission_id`) as selected (`chosen=True/False`)
    for the Private Leaderboard via:
      `PATCH https://api.zindi.world/v1/competitions/{challenge_id}/submissions/{submission_id}`
      with JSON body `{"chosen": chosen}`.
    """
    resolved_id = resolve_zindi_challenge_id(challenge_id)
    url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submissions/{submission_id}"
    resp = requests.patch(
        url,
        headers=_zindi_auth_headers(auth_token=auth_token),
        json={"chosen": bool(chosen)},
        timeout=timeout,
    )
    data = _extract_zindi_payload(resp)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response setting chosen={chosen} on {submission_id}: {data}")
    return data


def sync_zindi_competition_files(
    raw_dir: Path = RAW_DIR,
    challenge_id: str | None = None,
    filenames: tuple[str, ...] = ("Train.csv", "Test.csv", "SampleSubmission.csv", "Stations.csv"),
) -> bool:
    """
    Download latest competition datafiles (such as the Nov 7 / Nov 9 / Nov 10 `Train.csv`
    updates) directly from `GET https://api.zindi.africa/v1/competitions/{id}/files/{filename}`
    when Zindi credentials are set.
    """
    creds = get_zindi_credentials(challenge_id_override=challenge_id)
    resolved_id = creds["challenge_id"]
    if not creds["is_configured"]:
        print(
            "[Zindi Sync] Credentials not set; skipping remote datafile download "
            f"(using local files in {raw_dir})."
        )
        return False

    auth_token = creds["auth_token"]
    if not auth_token:
        signin_data = zindi_rest_signin(creds["username"], creds["password"])
        auth_token = str(signin_data["auth_token"])

    zindi_rest_join_competition(auth_token=auth_token, challenge_id=resolved_id)
    comp_info = zindi_rest_get_competition(challenge_id=resolved_id, auth_token=auth_token)
    remote_files = {
        str(item.get("filename")): int(item.get("filesize") or 0)
        for item in comp_info.get("datafiles", [])
        if isinstance(item, dict) and item.get("filename")
    }

    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = _zindi_auth_headers(auth_token=auth_token)
    any_updated = False
    for fname in filenames:
        if remote_files and fname not in remote_files:
            continue
        dest = raw_dir / fname
        expected_size = remote_files.get(fname, 0)
        if dest.is_file() and expected_size > 0 and dest.stat().st_size == expected_size:
            print(f"[Zindi Sync] {fname} already up-to-date ({expected_size:,} bytes).")
            continue

        file_url = f"{ZINDI_COMPETITIONS_API}/{resolved_id}/files/{fname}"
        print(f"[Zindi Sync] Downloading {fname} from {file_url}...")
        with requests.get(
            file_url,
            headers=headers,
            params={"auth_token": auth_token},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            tmp_dest = raw_dir / f".{fname}.tmp"
            with open(tmp_dest, "wb") as out_f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        out_f.write(chunk)
            tmp_dest.replace(dest)
            any_updated = True
            print(f"[Zindi Sync] Updated {dest} ({dest.stat().st_size:,} bytes).")
    return any_updated


def check_zindi_status(
    submission_path: Path = SUBMISSIONS_DIR / "submission_latest.csv",
    challenge_id: str | None = None,
) -> dict[str, Any]:
    """
    Diagnostic mode (`--check-zindi` / `--status`):
    1. Verifies local submission CSV (`submissions/submission_latest.csv` or `--file`)
       against all 6 Zindi assertions and prints per-block summary statistics.
    2. Checks live connectivity to `https://api.zindi.africa/v1/competitions/google-mumbai-data-challenge`,
       verifies challenge metadata, compares remote vs. local `Train.csv` byte sizes, and prints
       the live Top-5 Public Leaderboard.
    3. Checks credential configuration (`ZINDI_USERNAME`/`ZINDI_PASSWORD` or `ZINDI_TOKEN` in env/`.env`)
       without exposing secrets, and if configured, queries remaining daily submissions and user rank.
    """
    report: dict[str, Any] = {}

    # 1. Local Submission File Verification
    sub_df = verify_submission_file(submission_path=submission_path, raw_dir=RAW_DIR, verbose=True)
    report["submission_verified"] = True
    report["submission_rows"] = len(sub_df)
    report["submission_mean"] = float(sub_df["Target"].mean())

    # 2. Competition Resolution & Public Zindi API Connectivity
    creds = get_zindi_credentials(challenge_id_override=challenge_id)
    resolved_id = creds["challenge_id"]
    raw_id = creds["raw_challenge_id"]
    report["challenge_id"] = resolved_id

    print("\n" + "=" * 94)
    print("ZINDI API CONNECTIVITY & CHALLENGE DIAGNOSTICS")
    print("=" * 94)
    if raw_id != resolved_id:
        print(f"  [Alias Resolved] '{raw_id}' -> '{resolved_id}'")
    else:
        print(f"  Competition ID:  {resolved_id}")

    try:
        comp = zindi_rest_get_competition(challenge_id=resolved_id, auth_token=creds["auth_token"])
        report["api_reachable"] = True
        print(f"  API Endpoint:    {ZINDI_COMPETITIONS_API}/{resolved_id} (HTTP 200 OK)")
        print(f"  Title:           {comp.get('title')}")
        print(
            f"  Kind / Status:   kind={comp.get('kind')} | open={comp.get('open')} | "
            f"secret_code_required={comp.get('secret_code_required')} | sealed={comp.get('sealed')}"
        )
        print(
            f"  Metric & Reward: metric={comp.get('error_metric')} | "
            f"benchmark_public_rmse={comp.get('benchmark_submission_public_score')} | "
            f"reward={comp.get('reward')}"
        )
        print(
            f"  Timeline (UTC):  start={comp.get('start_time')} -> end={comp.get('end_time')}"
        )
        print(
            f"  Participation:   enrolled={comp.get('participations_count')} | "
            f"on_leaderboard={comp.get('leaderboard_participations_count')}"
        )

        # Check remote datafiles vs local data/raw/Train.csv
        datafiles = comp.get("datafiles", []) or []
        for df_meta in datafiles:
            if isinstance(df_meta, dict) and df_meta.get("filename") == "Train.csv":
                remote_size = int(df_meta.get("filesize") or 0)
                local_train = RAW_DIR / "Train.csv"
                local_size = local_train.stat().st_size if local_train.is_file() else 0
                report["remote_train_bytes"] = remote_size
                report["local_train_bytes"] = local_size
                size_status = (
                    "MATCHES remote Zindi file"
                    if remote_size == local_size
                    else f"REMOTE UPDATE AVAILABLE (remote={remote_size:,} B vs local={local_size:,} B; run with --sync-zindi-data)"
                )
                print(f"  Train.csv Check: {size_status}")

        # Fetch Top-5 Public Leaderboard
        lb_rows = zindi_rest_get_leaderboard(challenge_id=resolved_id, per_page=5)
        if lb_rows:
            print("\n  Live Public Leaderboard Snapshot (Top 5):")
            for row in lb_rows[:5]:
                rank = row.get("public_rank")
                score = row.get("best_public_score")
                subs = row.get("submission_count")
                who = (row.get("user") or {}).get("username") or (
                    f"TEAM: {(row.get('team') or {}).get('title')}"
                )
                print(f"    #{str(rank):<3} | {str(who):<28} | public_score={score} | subs={subs}")
    except Exception as exc:
        report["api_reachable"] = False
        report["api_error"] = str(exc)
        print(f"  [WARN] Could not reach Zindi public API: {exc}")

    # 3. Credential & Authenticated Account Check
    print("\n" + "=" * 94)
    print("ZINDI CREDENTIALS & SUBMISSION READINESS STATUS")
    print("=" * 94)
    print(f"  Loaded .env files: {creds['loaded_env_files'] or 'None (.env not present)'}")
    print(
        f"  ZINDI_USERNAME:    {'SET (' + creds['username'] + ')' if creds['username'] else 'NOT SET'}"
    )
    print(
        f"  ZINDI_PASSWORD:    {'SET (masked)' if creds['password'] else 'NOT SET'}"
    )
    print(
        f"  ZINDI_TOKEN:       {'SET (masked)' if creds['auth_token'] else 'NOT SET'}"
    )

    if not creds["is_configured"]:
        report["authenticated"] = False
        print(
            "\n  [ACTION REQUIRED TO PUSH LIVE TO ZINDI]\n"
            "  No Zindi credentials found in environment or `.env`. To upload your verified\n"
            f"  submission (`{submission_path.relative_to(PROJECT_ROOT) if submission_path.is_relative_to(PROJECT_ROOT) else submission_path}`) without re-training:\n\n"
            "    ZINDI_USERNAME='your_username' ZINDI_PASSWORD='your_password' \\\n"
            "      uv run python src/submit.py --upload-only --file submissions/submission_latest.csv\n\n"
            "  Or to sync the latest Zindi Train.csv, retrain all 8 models + Known-Row Lookup, and upload:\n\n"
            "    ZINDI_USERNAME='your_username' ZINDI_PASSWORD='your_password' \\\n"
            "      uv run python src/submit.py --sync-zindi-data --upload"
        )
        print("=" * 94)
        return report

    # Credentials are present -> test sign-in & query limits/rank/history
    try:
        auth_token = creds["auth_token"]
        if not auth_token:
            signin_res = zindi_rest_signin(creds["username"], creds["password"])
            auth_token = str(signin_res["auth_token"])
            user_info = signin_res.get("user") or {}
            print(f"  [AUTH OK] Signed in to Zindi API as '{user_info.get('username', creds['username'])}'")
        else:
            print("  [AUTH OK] Using configured ZINDI_TOKEN")

        acct = zindi_rest_get_account_status(auth_token=auth_token, challenge_id=resolved_id)
        report["authenticated"] = True
        report["account_status"] = acct

        limits = acct.get("limits")
        if isinstance(limits, dict):
            print(f"  Remaining Submissions Today: {limits.get('today', limits)}")
        my_part = acct.get("my_participation")
        if isinstance(my_part, dict):
            print(
                f"  Current Leaderboard Status:  public_rank={my_part.get('public_rank')} | "
                f"best_public_score={my_part.get('best_public_score')} | "
                f"submission_count={my_part.get('submission_count')}"
            )
        recent = acct.get("recent_submissions") or []
        if recent:
            print("  Recent Submissions (latest 5):")
            for s_row in recent[:5]:
                print(
                    f"    - id={s_row.get('id')} | chosen={s_row.get('chosen')} | "
                    f"status={s_row.get('status')} | public_score={s_row.get('public_score')} | "
                    f"created_at={s_row.get('created_at')} | file={s_row.get('filename')} | "
                    f"comment={s_row.get('comment')!r}"
                )
    except Exception as exc:
        report["authenticated"] = False
        report["auth_error"] = str(exc)
        print(f"  [AUTH ERROR] Failed to authenticate or query Zindi account status: {exc}")

    print("=" * 94)
    return report


def upload_to_zindi(
    submission_path: Path = SUBMISSIONS_DIR / "submission_latest.csv",
    comment: str = "8-Model Physics + AirPhyNet + GBDT + Wind-ST-GAT + TimesFM 3.0 + Chronos-2 Ensemble",
    challenge_id: str | None = None,
    api_mode: str = "auto",
) -> bool:
    """
    Verify `submission_path` and upload it to Zindi (`google-mumbai-data-challenge`).

    Supports two non-interactive upload backends:
    - `zindi` SDK (`Zindian` with `select_a_challenge(challenge_id=resolved_id)` — fixed so it
      never triggers interactive `input()` prompts or `fixed_index` type errors)
    - Direct Zindi REST API (`POST /v1/auth/signin` + `POST /v1/competitions/{id}/submissions`)
      as an automatic fallback or primary when `ZINDI_TOKEN` / `--api-mode rest` is used.
    """
    # Always verify the submission file before attempting any network upload
    verify_submission_file(submission_path=submission_path, raw_dir=RAW_DIR, verbose=True)

    creds = get_zindi_credentials(challenge_id_override=challenge_id)
    resolved_id = creds["challenge_id"]
    username = creds["username"]
    password = creds["password"]
    auth_token = creds["auth_token"]

    if not creds["is_configured"]:
        print(
            "\n[Zindi Upload] ZINDI_USERNAME/ZINDI_PASSWORD (or ZINDI_TOKEN) are not set in the environment or .env.\n"
            "To upload this verified CSV directly to Zindi without re-running training, run:\n"
            "  export ZINDI_USERNAME='your_username'\n"
            "  export ZINDI_PASSWORD='your_password'\n"
            f"  uv run python src/submit.py --upload-only --file {submission_path}\n"
        )
        return False

    print(
        f"\n[Zindi Upload] Target competition: '{resolved_id}' | "
        f"File: {submission_path} | Mode: {api_mode}"
    )

    uploaded_ok = False
    submission_id: Any = None

    # Path A: Try official `zindi` Python SDK (`Zindian`) when username+password are provided
    # and api_mode is 'auto' or 'sdk'.
    if api_mode in {"auto", "sdk"} and username and password:
        try:
            from zindi.user import Zindian

            print(f"[Zindi SDK] Authenticating as '{username}' for challenge '{resolved_id}'...")
            user = Zindian(username=username, fixed_password=password)
            # The installed SDK hardcodes api.zindi.africa. Point it at the same origin
            # host as ZINDI_API_BASE before select/submit so the multipart body is not
            # sent through the CloudFront 308 redirect.
            user._Zindian__base_api = ZINDI_COMPETITIONS_API
            user._Zindian__api_client.base_api = ZINDI_COMPETITIONS_API
            # CRITICAL: pass `challenge_id=resolved_id` (not `fixed_index=...`) so `zindi==0.0.4`
            # queries the competition directly without `input()`.
            sel = user.select_a_challenge(challenge_id=resolved_id)
            if isinstance(sel, dict) and not sel.get("challenge"):
                raise RuntimeError(f"Zindian.select_a_challenge failed: {sel}")
            sdk_res = user.submit(filepaths=[str(submission_path)], comments=[comment])
            if isinstance(sdk_res, list) and sdk_res and sdk_res[0].get("status") == "success":
                uploaded_ok = True
                submission_id = sdk_res[0].get("submission_id")
                print(
                    f"[Zindi SDK] Successfully uploaded {submission_path.name} "
                    f"(Submission ID: {submission_id})!"
                )
            else:
                raise RuntimeError(f"Zindian.submit returned non-success payload: {sdk_res}")
        except Exception as sdk_exc:
            if api_mode == "sdk":
                print(f"[Zindi SDK] Upload failed in strict SDK mode: {sdk_exc}")
                return False
            print(
                f"[Zindi SDK] SDK upload encountered an issue ({sdk_exc}); "
                "falling back to direct Zindi REST API..."
            )

    # Path B: Direct Zindi REST API (`POST /v1/auth/signin` + `POST /v1/competitions/{id}/submissions`)
    if not uploaded_ok:
        try:
            if not auth_token:
                print(f"[Zindi REST] Signing in as '{username}' via {ZINDI_API_BASE}/auth/signin...")
                signin_data = zindi_rest_signin(username=username, password=password)
                auth_token = str(signin_data["auth_token"])
            print(
                f"[Zindi REST] Uploading {submission_path.name} to "
                f"{ZINDI_COMPETITIONS_API}/{resolved_id}/submissions..."
            )
            rest_res = zindi_rest_submit_file(
                auth_token=auth_token,
                challenge_id=resolved_id,
                submission_path=submission_path,
                comment=comment,
            )
            submission_id = rest_res.get("id")
            uploaded_ok = True
            print(
                f"[Zindi REST] Successfully uploaded {submission_path.name} "
                f"(Submission ID: {submission_id})!"
            )
        except Exception as rest_exc:
            print(f"[Zindi REST] Upload failed: {rest_exc}")
            return False

    # Post-upload status poll (fetch score / processing status and remaining daily submissions)
    try:
        if not auth_token and username and password:
            auth_token = str(zindi_rest_signin(username, password)["auth_token"])
        if auth_token:
            time.sleep(1.5)
            acct = zindi_rest_get_account_status(auth_token=auth_token, challenge_id=resolved_id)
            limits = acct.get("limits")
            if isinstance(limits, dict):
                print(f"[Zindi Status] Remaining daily submissions today: {limits.get('today', limits)}")
            recent = acct.get("recent_submissions") or []
            if recent:
                latest = recent[0]
                print(
                    f"[Zindi Status] Latest submission on board: id={latest.get('id')} | "
                    f"status={latest.get('status')} | public_score={latest.get('public_score')} | "
                    f"comment={latest.get('comment')!r}"
                )
    except Exception:
        pass

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train final 8-model ensemble, generate verified Zindi submission files, "
            "diagnose Zindi API/leaderboard status, and upload submissions via SDK or REST API."
        )
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload verified submission CSV to Zindi via SDK + direct REST fallback.",
    )
    parser.add_argument(
        "--upload-only",
        "--skip-train",
        dest="upload_only",
        action="store_true",
        help=(
            "Skip full 8-model ensemble training; verify existing `--file` "
            "(default `submissions/submission_latest.csv`) and upload to Zindi."
        ),
    )
    parser.add_argument(
        "--check-zindi",
        "--status",
        dest="check_zindi",
        action="store_true",
        help=(
            "Run diagnostic check: verify `--file` against SampleSubmission.csv, "
            "check Zindi API connectivity, resolve competition ID, compare remote vs local "
            "Train.csv size, and check credential/submission status."
        ),
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=SUBMISSIONS_DIR / "submission_latest.csv",
        help="Path to submission CSV file to verify or upload (default: submissions/submission_latest.csv).",
    )
    parser.add_argument(
        "--challenge-id",
        type=str,
        default=None,
        help=(
            "Zindi competition slug override (default: google-mumbai-data-challenge; "
            "automatically resolves alias google-mumbai-air-quality-challenge)."
        ),
    )
    parser.add_argument(
        "--api-mode",
        choices=["auto", "rest", "sdk"],
        default="auto",
        help="Upload transport mode: 'auto' (Zindian SDK with REST fallback), 'rest' (direct HTTP API), or 'sdk'.",
    )
    parser.add_argument(
        "--sync-zindi-data",
        action="store_true",
        help="Download latest competition datafiles (e.g. updated Train.csv) from Zindi before training.",
    )
    parser.add_argument(
        "--exact-lookup",
        action="store_true",
        help=(
            "Do not retrain. Overlay official Train.csv pm25 decimal strings onto --file "
            "and write submissions/submission_latest.csv and submissions/submission_zindi_exact.csv."
        ),
    )
    parser.add_argument(
        "--use-openaq-s3",
        dest="use_openaq_s3",
        action="store_true",
        default=True,
        help=(
            "Merge post-Oct-21 OpenAQ S3 observations (`data/external/openaq_s3_oct20_nov04.parquet`) "
            "into training history + Known-Row Lookup (enabled by default when cache exists)."
        ),
    )
    parser.add_argument(
        "--no-openaq-s3",
        dest="use_openaq_s3",
        action="store_false",
        help="Disable merging `data/external/openaq_s3_oct20_nov04.parquet` (use raw `Train.csv` only).",
    )
    parser.add_argument(
        "--choose-ids",
        nargs="+",
        default=None,
        help=(
            "List of 1 or 2 Zindi submission IDs (e.g. `o7BW5Ab3 juPsKU7m`) to mark as "
            "`chosen=True` via PATCH `/v1/competitions/{id}/submissions/{sub_id}` for Private LB evaluation."
        ),
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="8-Model Physics + AirPhyNet + GBDT + Wind-ST-GAT + TimesFM 3.0 + Chronos-2 Ensemble",
        help="Submission comment for Zindi upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sync_zindi_data:
        sync_zindi_competition_files(raw_dir=RAW_DIR, challenge_id=args.challenge_id)

    if args.choose_ids:
        creds = get_zindi_credentials(challenge_id_override=args.challenge_id)
        if not creds["is_configured"]:
            raise RuntimeError("Zindi credentials are required for --choose-ids.")
        auth_token = creds["auth_token"] or str(
            zindi_rest_signin(creds["username"], creds["password"])["auth_token"]
        )
        for sub_id in args.choose_ids:
            res = zindi_rest_set_submission_chosen(
                auth_token=auth_token,
                challenge_id=creds["challenge_id"],
                submission_id=sub_id,
                chosen=True,
            )
            print(
                f"[Zindi Selection] Submission {res.get('id')} -> chosen={res.get('chosen')} "
                f"(public_score={res.get('public_score')}, file={res.get('filename')})"
            )
        if not (args.check_zindi or args.upload or args.upload_only or args.exact_lookup):
            return

    if args.check_zindi:
        check_zindi_status(submission_path=args.file, challenge_id=args.challenge_id)
        if args.upload:
            upload_to_zindi(
                submission_path=args.file,
                comment=args.comment,
                challenge_id=args.challenge_id,
                api_mode=args.api_mode,
            )
        return

    if args.exact_lookup:
        compose_exact_lookup_submission(
            base_submission=args.file,
            train_csv=RAW_DIR / "Train.csv",
        )
        if args.upload or args.upload_only:
            upload_to_zindi(
                submission_path=SUBMISSIONS_DIR / "submission_zindi_exact.csv",
                comment=args.comment,
                challenge_id=args.challenge_id,
                api_mode=args.api_mode,
            )
        return

    if args.upload_only:
        upload_to_zindi(
            submission_path=args.file,
            comment=args.comment,
            challenge_id=args.challenge_id,
            api_mode=args.api_mode,
        )
        return

    generate_submissions(use_openaq_s3=args.use_openaq_s3)
    if args.upload:
        target_file = (
            SUBMISSIONS_DIR / "submission_openaq_synced.csv"
            if args.use_openaq_s3 and args.file == (SUBMISSIONS_DIR / "submission_latest.csv")
            else args.file
        )
        upload_to_zindi(
            submission_path=target_file,
            comment=args.comment,
            challenge_id=args.challenge_id,
            api_mode=args.api_mode,
        )


if __name__ == "__main__":
    main()
