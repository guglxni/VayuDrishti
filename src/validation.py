#!/usr/bin/env python3
"""
Validation & Target Cleaning Module (`src/validation.py`)
for the Google Mumbai Air Quality Forecasting Challenge.

Implements:
1. Strict IST (`Asia/Kolkata`, UTC+05:30) timestamp alignment between `Train.csv`
   (`hour_end = timestamp_ist`, `hour_start = hour_end - 1h`) and `Test.csv`
   (`hour_start = target_start_ist`, `hour_end = target_end_ist`).
2. 2-Stage Target Outlier Cleaner (`clean_pm25`) for training targets:
   - Stage 1: Flag/replace extreme physical sensor glitches (`flag_high_1000 == True`
     or `pm25 > 450.0 ug/m3`).
   - Stage 2: Rolling 7-hour Hampel filter per station (replace spikes > 4.5 * MAD
     above the rolling median with interpolated rolling median).
3. 5-Fold Post-Monsoon Rolling 48-hour Validation Split (`get_rolling_validation_splits`)
   across all 4 target stations, supporting both short reporting gaps (24h-48h,
   matching Nov 10 -> Nov 11-12 Private evaluation) and medium gaps (72h, matching
   the Starter Notebook), plus optional Oct/Nov 2025 historical post-monsoon folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IST_TZ = "Asia/Kolkata"

# Post-monsoon October 2026 48-hour validation windows (each 48h across all 4 stations)
DEFAULT_OCT2026_WINDOWS: list[tuple[str, str]] = [
    ("2026-10-12 00:00:00", "2026-10-14 00:00:00"),
    ("2026-10-14 00:00:00", "2026-10-16 00:00:00"),
    ("2026-10-16 00:00:00", "2026-10-18 00:00:00"),
    ("2026-10-18 00:00:00", "2026-10-20 00:00:00"),
    ("2026-10-20 00:00:00", "2026-10-22 00:00:00"),
]

# Post-monsoon Oct/Nov 2025 48-hour validation windows ( matching Nov 11-12 Private season)
NOV2025_WINDOWS: list[tuple[str, str]] = [
    ("2025-10-28 00:00:00", "2025-10-30 00:00:00"),
    ("2025-11-02 00:00:00", "2025-11-04 00:00:00"),
    ("2025-11-06 00:00:00", "2025-11-08 00:00:00"),
    ("2025-11-11 00:00:00", "2025-11-13 00:00:00"),
]


@dataclass
class ValidationFold:
    """Represents a single leak-free 48-hour rolling validation split."""

    fold_id: int
    name: str
    cutoff_t0: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    gap_hours: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))


def as_ist(values: pd.Series | Any) -> pd.Series:
    """Convert timestamps to Asia/Kolkata (UTC+05:30)."""
    return pd.to_datetime(values, utc=True).dt.tz_convert(IST_TZ)


def clean_pm25(
    df: pd.DataFrame,
    target_col: str = "pm25",
    cap_ugm3: float = 450.0,
    hampel_window: int = 7,
    hampel_mad_mult: float = 4.5,
) -> pd.DataFrame:
    """
    2-Stage Target Outlier Cleaner for training PM2.5 observations:
      - Stage 1: Flag extreme physical sensor glitches (`flag_high_1000 == True`
        or `pm25 > cap_ugm3` = 450.0 ug/m3) and replace with local station rolling
        median / time-interpolated values (capped at `cap_ugm3`).
      - Stage 2: Rolling 7-hour Hampel filter per station (`pm25 - rolling_median > 4.5 * MAD`),
        replacing isolated transient sensor spikes with the rolling median.

    Returns a copy of `df` with:
      - `pm25_raw`: original raw PM2.5
      - `pm25_clean`: cleaned PM2.5
      - `is_glitch_stage1`: boolean flag for Stage 1 extreme glitches
      - `is_spike_stage2`: boolean flag for Stage 2 Hampel spikes
      - `is_outlier`: boolean union of Stage 1 and Stage 2
    """
    out = df.copy()
    out = out.sort_values(["location_id", "hour_start"]).reset_index(drop=True)
    raw_vals = pd.to_numeric(out[target_col], errors="coerce").astype(float)
    out["pm25_raw"] = raw_vals

    flag_1000 = (
        out["flag_high_1000"].fillna(False).astype(bool)
        if "flag_high_1000" in out.columns
        else pd.Series(False, index=out.index)
    )
    stage1_mask = flag_1000 | (raw_vals > cap_ugm3) | (raw_vals < 0.0)
    out["is_glitch_stage1"] = stage1_mask

    cleaned_chunks: list[pd.Series] = []
    stage2_masks: list[pd.Series] = []

    for _, grp in out.groupby("location_id", sort=False):
        s = grp["pm25_raw"].copy()
        s1_flag = grp["is_glitch_stage1"]

        # Stage 1: Mask extreme hardware glitches (>450 ug/m3 or flag_high_1000)
        # and replace via local rolling median / linear interpolation along station timeline
        s_stage1 = s.mask(s1_flag, np.nan)
        med_broad = s_stage1.rolling(window=25, center=True, min_periods=3).median()
        s_stage1 = s_stage1.fillna(med_broad).interpolate(method="linear", limit_direction="both")
        s_stage1 = s_stage1.clip(lower=0.5, upper=cap_ugm3)

        # Stage 2: Rolling 7-hour Hampel filter per station
        roll_med = s_stage1.rolling(window=hampel_window, center=True, min_periods=3).median()
        abs_dev = (s_stage1 - roll_med).abs()
        roll_mad = abs_dev.rolling(window=hampel_window, center=True, min_periods=3).median()
        # Use minimum MAD floor of 2.5 ug/m3 so low-variance periods are not over-filtered
        effective_mad = np.maximum(roll_mad * 1.4826, 2.5)

        # Replace upward spikes exceeding 4.5 * MAD above rolling median
        s2_flag = (s_stage1 - roll_med) > (hampel_mad_mult * effective_mad)
        s_stage2 = s_stage1.mask(s2_flag, roll_med)
        s_stage2 = s_stage2.interpolate(method="linear", limit_direction="both").clip(
            lower=0.5, upper=cap_ugm3
        )

        cleaned_chunks.append(s_stage2)
        stage2_masks.append(s2_flag)

    out["pm25_clean"] = pd.concat(cleaned_chunks).sort_index()
    out["is_spike_stage2"] = pd.concat(stage2_masks).sort_index()
    out["is_outlier"] = out["is_glitch_stage1"] | out["is_spike_stage2"]
    return out


def load_and_align_core_data(
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and align `Train.csv`, `Test.csv`, `Stations.csv`, and `SampleSubmission.csv`.

    - `Train.csv`:
        `hour_end = pd.to_datetime(timestamp_ist).dt.tz_convert('Asia/Kolkata')`
        `hour_start = hour_end - 1h`
        `target_start_ist = hour_start`
        Cleaned with `clean_pm25`.
    - `Test.csv`:
        `hour_start = pd.to_datetime(target_start_ist).dt.tz_convert('Asia/Kolkata')`
        `hour_end = pd.to_datetime(target_end_ist).dt.tz_convert('Asia/Kolkata')`
        `target_start_ist = hour_start`
    """
    train = pd.read_csv(raw_dir / "Train.csv")
    test = pd.read_csv(raw_dir / "Test.csv")
    stations = pd.read_csv(raw_dir / "Stations.csv")
    sample_sub = pd.read_csv(raw_dir / "SampleSubmission.csv")

    train["location_id"] = train["location_id"].astype(int)
    test["location_id"] = test["location_id"].astype(int)
    stations["location_id"] = stations["location_id"].astype(int)

    train["hour_end"] = as_ist(train["timestamp_ist"])
    train["hour_start"] = train["hour_end"] - pd.Timedelta(hours=1)
    train["target_start_ist"] = train["hour_start"]

    test["hour_start"] = as_ist(test["target_start_ist"])
    test["hour_end"] = as_ist(test["target_end_ist"])
    test["target_start_ist"] = test["hour_start"]

    train = clean_pm25(train)
    train = train.sort_values(["location_id", "hour_start"]).reset_index(drop=True)
    return train, test, stations, sample_sub


def get_rolling_validation_splits(
    train_df: pd.DataFrame,
    gap_hours: int = 24,
    windows: list[tuple[str, str]] | None = None,
    include_nov2025: bool = False,
) -> list[ValidationFold]:
    """
    Construct the 5-Fold Post-Monsoon Rolling 48-hour Validation Splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Aligned training dataframe containing `hour_start`, `hour_end`, `location_id`, `pm25`.
    gap_hours : int, default=24
        Reporting gap in hours between `cutoff_t0` (latest permitted `hour_end` in training history)
        and `val_start` (first `hour_start` in the 48-hour target validation window).
        - `gap_hours=24` or `48` simulates the short reporting gap (e.g. Nov 10 -> Nov 11-12 Private window,
          where horizons span 24h..72h or 48h..96h).
        - `gap_hours=72` simulates the medium 3-day reporting gap from the Starter Notebook.
    windows : list[tuple[str, str]] | None
        Optional custom list of (val_start_str, val_end_str) in IST. Defaults to the 5 post-monsoon
        October 2026 48h windows (`2026-10-12..13` through `2026-10-20..21`).
    include_nov2025 : bool, default=False
        If True, prepends the post-monsoon Oct/Nov 2025 48h windows before the Oct 2026 windows.
    """
    if windows is None:
        selected_windows = (NOV2025_WINDOWS + DEFAULT_OCT2026_WINDOWS) if include_nov2025 else list(DEFAULT_OCT2026_WINDOWS)
    else:
        selected_windows = list(windows)

    folds: list[ValidationFold] = []
    for idx, (vs_str, ve_str) in enumerate(selected_windows, start=1):
        val_start = pd.Timestamp(vs_str, tz=IST_TZ)
        val_end = pd.Timestamp(ve_str, tz=IST_TZ)
        cutoff_t0 = val_start - pd.Timedelta(hours=gap_hours)

        hist_mask = train_df["hour_end"] <= cutoff_t0
        val_mask = (train_df["hour_start"] >= val_start) & (train_df["hour_start"] < val_end)

        hist_df = train_df.loc[hist_mask].copy().reset_index(drop=True)
        val_df = train_df.loc[val_mask].copy().reset_index(drop=True)

        assert not hist_df.empty, f"Empty training history for fold {idx} ({vs_str})"
        assert not val_df.empty, f"Empty validation window for fold {idx} ({vs_str})"
        assert hist_df["hour_end"].max() <= cutoff_t0 < val_start, "Strict temporal anti-leakage assertion failed!"

        fold_name = f"Fold{idx}_{val_start.strftime('%Y%m%d')}_{(val_end - pd.Timedelta(days=1)).strftime('%m%d')}_gap{gap_hours}h"
        folds.append(
            ValidationFold(
                fold_id=idx,
                name=fold_name,
                cutoff_t0=cutoff_t0,
                val_start=val_start,
                val_end=val_end,
                gap_hours=gap_hours,
                train_df=hist_df,
                val_df=val_df,
            )
        )
    return folds


def predict_starter_rf_baseline(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cutoff_t0: pd.Timestamp,
    train_window_days: int = 90,
    use_clean_target: bool = False,
) -> np.ndarray:
    """
    Reproduce the exact Starter Notebook Random Forest Baseline (trained on last 90 days
    before `cutoff_t0` using calendar features + one-hot station ID).
    """
    hist = train_df.loc[
        (train_df["hour_end"] <= cutoff_t0)
        & (train_df["hour_end"] > cutoff_t0 - pd.Timedelta(days=train_window_days))
    ].copy()

    def _make_starter_features(rows: pd.DataFrame) -> pd.DataFrame:
        t = rows["hour_start"]
        return pd.DataFrame(
            {
                "location_id": rows["location_id"].astype(str),
                "hour": t.dt.hour,
                "day_of_week": t.dt.day_of_week,
                "month": t.dt.month,
                "day_of_year": t.dt.day_of_year,
            },
            index=rows.index,
        )

    X_tr = _make_starter_features(hist)
    y_tr = hist["pm25_clean"] if use_clean_target else hist["pm25"]
    X_te = _make_starter_features(target_df)

    preprocessor = ColumnTransformer(
        [("station", OneHotEncoder(handle_unknown="ignore"), ["location_id"])],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("features", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=float)


def predict_diurnal_shrinkage_baseline(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cutoff_t0: pd.Timestamp,
    diurnal_days: int = 7,
    level_days: int = 3,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Empirical Bayes / James-Stein 7-day Diurnal Shrinkage Baseline anchored at `cutoff_t0`:
      y_hat(s, h) = s_level_recent(s) + alpha * (sh_diurnal_7d(s, h) - s_mean_7d(s))
    Equivalent to shrinking the 7-day diurnal profile toward the fast-adapting recent station level
    computed strictly on cleaned observations (`pm25_clean`) with `hour_end <= cutoff_t0`.
    """
    hist = train_df.loc[train_df["hour_end"] <= cutoff_t0].copy()
    hist["hour"] = hist["hour_start"].dt.hour

    recent_diurnal = hist.loc[
        hist["hour_end"] > cutoff_t0 - pd.Timedelta(days=diurnal_days)
    ].copy()
    recent_level = hist.loc[
        hist["hour_end"] > cutoff_t0 - pd.Timedelta(days=level_days)
    ].copy()

    # Exponential time decay on 7d diurnal profile (half-life = 3.5 days)
    age_hours = (cutoff_t0 - recent_diurnal["hour_end"]).dt.total_seconds() / 3600.0
    recent_diurnal["w"] = np.exp(-np.log(2.0) * age_hours / (3.5 * 24.0))
    recent_diurnal["wy"] = recent_diurnal["w"] * recent_diurnal["pm25_clean"]

    sh_sum = recent_diurnal.groupby(["location_id", "hour"])[["wy", "w"]].sum()
    sh_mean = (sh_sum["wy"] / sh_sum["w"]).rename("sh_mean")

    s7_sum = recent_diurnal.groupby("location_id")[["wy", "w"]].sum()
    s7_mean = (s7_sum["wy"] / s7_sum["w"]).rename("s7_mean")

    # Fast-adapting recent station level over last `level_days` (half-life = 18 hours)
    age_level = (cutoff_t0 - recent_level["hour_end"]).dt.total_seconds() / 3600.0
    recent_level["w"] = np.exp(-np.log(2.0) * age_level / 18.0)
    recent_level["wy"] = recent_level["w"] * recent_level["pm25_clean"]
    s_sum = recent_level.groupby("location_id")[["wy", "w"]].sum()
    s_level = (s_sum["wy"] / s_sum["w"]).rename("s_level")

    global_mean = float(recent_diurnal["pm25_clean"].mean())

    tmp = pd.DataFrame(
        {
            "location_id": target_df["location_id"].astype(int).values,
            "hour": target_df["hour_start"].dt.hour.values,
        },
        index=target_df.index,
    )
    tmp = tmp.merge(sh_mean.reset_index(), on=["location_id", "hour"], how="left")
    tmp = tmp.merge(s7_mean.reset_index(), on="location_id", how="left")
    tmp = tmp.merge(s_level.reset_index(), on="location_id", how="left")

    tmp["s7_mean"] = tmp["s7_mean"].fillna(global_mean)
    tmp["s_level"] = tmp["s_level"].fillna(tmp["s7_mean"])
    tmp["sh_mean"] = tmp["sh_mean"].fillna(tmp["s7_mean"])

    # Shrinkage combination of recent station level and 7d diurnal shape:
    diurnal_anom = tmp["sh_mean"].values - tmp["s7_mean"].values
    level_blend = 0.65 * tmp["s_level"].values + 0.35 * tmp["s7_mean"].values
    pred = level_blend + alpha * diurnal_anom
    return np.asarray(np.clip(pred, 1.0, 450.0), dtype=float)


def main() -> None:
    train, test, stations, _ = load_and_align_core_data()
    print(f"Loaded Train: {len(train):,} rows | Test: {len(test):,} rows | Stations: {len(stations)}")
    print(
        f"Outlier Cleaning Summary: Stage1 Glitches={train['is_glitch_stage1'].sum()}, "
        f"Stage2 Hampel Spikes={train['is_spike_stage2'].sum()}, "
        f"Total Cleaned={train['is_outlier'].sum()} ({100.0 * train['is_outlier'].mean():.2f}%)"
    )
    print(f"Raw max PM2.5: {train['pm25_raw'].max():.1f} ug/m3 -> Cleaned max PM2.5: {train['pm25_clean'].max():.1f} ug/m3")

    for gap in [24, 48, 72]:
        folds = get_rolling_validation_splits(train, gap_hours=gap)
        rf_preds, diurnal_preds, actuals = [], [], []
        print(f"\n--- 5-Fold Post-Monsoon Validation (Reporting Gap = {gap}h) ---")
        for f in folds:
            p_rf = predict_starter_rf_baseline(f.train_df, f.val_df, f.cutoff_t0)
            p_diurnal = predict_diurnal_shrinkage_baseline(f.train_df, f.val_df, f.cutoff_t0)
            y_val = f.val_df["pm25"].values
            rf_preds.append(p_rf)
            diurnal_preds.append(p_diurnal)
            actuals.append(y_val)
            print(
                f"  {f.name}: n={len(y_val)} | Starter RF RMSE={rmse(y_val, p_rf):.3f} | "
                f"Diurnal Shrinkage RMSE={rmse(y_val, p_diurnal):.3f}"
            )
        y_all = np.concatenate(actuals)
        rf_all = np.concatenate(rf_preds)
        di_all = np.concatenate(diurnal_preds)
        print(
            f"  >> OOF Overall (gap={gap}h, N={len(y_all)}): "
            f"Starter RF RMSE = {rmse(y_all, rf_all):.4f} | "
            f"Diurnal Shrinkage RMSE = {rmse(y_all, di_all):.4f}"
        )


if __name__ == "__main__":
    main()
