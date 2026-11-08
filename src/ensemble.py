#!/usr/bin/env python3
r"""
Out-of-Fold (OOF) 8-Model Ensemble Optimizer, Station-Specific Variance Shrinkage,
and Validation Leaderboard (`src/ensemble.py`) for the Google Mumbai Air Quality Challenge.

Evaluates all 11 baseline, physics, GAM/AnEn, GBDT, Spatiotemporal GAT, Pretrained Time-Series
Foundation Models (`TimesFM 3.0` + `Chronos-2`), and Stacked Ensemble architectures across the
5-Fold Post-Monsoon Rolling 48-hour Validation Split:
  0. 90-Day Station Mean Reference (Starter Benchmark Proxy)
  1. Starter Notebook Random Forest Baseline (90d window)
  2. 7-day Diurnal Shrinkage Baseline
  3. Calibrated Synoptic-Diurnal + tanh-MOS CAMS Physics Prior (Model 1)
  4. Dual-Scale Huber/Ridge/Duan-Smearing + AnEn Physics GAM (Model 2)
  5. LightGBM Residual Forecaster (Model 3)
  6. CatBoost Residual Forecaster (Model 4)
  7. XGBoost Residual Forecaster (Model 5)
  8. Wind-Guided ST-GAT (PyTorch) (Model 6)
  9. Google TimesFM 3.0 Compact Multi-View Forecaster (Model 7)
 10. Amazon Chronos-2 Multivariate + Covariates Forecaster (Model 8)
 11. Optimal 8-Model Stacked Ensemble (SLSQP + Station-Specific Variance Shrinkage $\alpha_s(h)$)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

# Prevent duplicate libomp mutex collision on macOS arm64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    STATION_ORDER,
    build_features_for_cutoff,
    build_multi_cutoff_training_dataset,
    get_external_data_store,
)
from src.models.foundation_ts import (
    Chronos2CovariateForecaster,
    TimesFM3CompactForecaster,
    attach_foundation_features,
)
from src.models.gbdt_zoo import (
    CatBoostResidualForecaster,
    LightGBMResidualForecaster,
    XGBoostResidualForecaster,
)
from src.models.priors_and_linear import (
    CalibratedDiurnalCAMSPriorForecaster,
    HuberRidgePhysicsGAMForecaster,
)
from src.models.st_gat import WindGuidedSTGATForecaster
from src.validation import (
    ValidationFold,
    get_rolling_validation_splits,
    load_and_align_core_data,
    predict_diurnal_shrinkage_baseline,
    predict_starter_rf_baseline,
    rmse,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_ZOO_NAMES: list[str] = [
    "m1_calibrated_cams_prior",
    "m2_huber_ridge_gam",
    "m3_lightgbm_residual",
    "m4_catboost_residual",
    "m5_xgboost_residual",
    "m6_wind_st_gat",
    "m7_timesfm3_compact",
    "m8_chronos2_covariate",
]

LEADERBOARD_DISPLAY_NAMES: dict[str, str] = {
    "b0_station_mean_90d": "0. 90-Day Station Mean Reference (Starter Benchmark Proxy)",
    "b1_starter_rf_90d": "1. Starter Notebook Random Forest Baseline (90d window)",
    "b2_diurnal_shrinkage_7d": "2. 7-day Diurnal Shrinkage Baseline",
    "m1_calibrated_cams_prior": "3. Synoptic-Diurnal + tanh-MOS CAMS Prior (Model 1)",
    "m2_huber_ridge_gam": "4. Dual-Scale Huber/Ridge/Duan + AnEn GAM (Model 2)",
    "m3_lightgbm_residual": "5. LightGBM Residual Forecaster (Model 3)",
    "m4_catboost_residual": "6. CatBoost Residual Forecaster (Model 4)",
    "m5_xgboost_residual": "7. XGBoost Residual Forecaster (Model 5)",
    "m6_wind_st_gat": "8. Wind-Guided ST-GAT (PyTorch) (Model 6)",
    "m7_timesfm3_compact": "9. Google TimesFM 3.0 Compact Multi-View (Model 7)",
    "m8_chronos2_covariate": "10. Amazon Chronos-2 Multivariate + Covariates (Model 8)",
    "m7_stacked_ensemble": "11. Optimal 8-Model Stacked Ensemble (SLSQP + Station Shrinkage)",
    "m9_stacked_ensemble": "11. Optimal 8-Model Stacked Ensemble (SLSQP + Station Shrinkage)",
}


@dataclass
class StackedAirQualityEnsemble:
    """
    8-Model Multi-Paradigm Stacked Ensemble combining:
      - Model 1: Synoptic-Diurnal + tanh-Saturating CAMS MOS Physics Prior
      - Model 2: Dual-Scale Huber/Ridge/Duan-Smearing + Meteorological Analog (`AnEn`) GAM
      - Model 3: LightGBM Residual Forecaster
      - Model 4: CatBoost Residual Forecaster
      - Model 5: XGBoost Residual Forecaster
      - Model 6: Wind-Guided ST-GAT (PyTorch)
      - Model 7: Google TimesFM 3.0 Compact Multi-View (`google/timesfm-3.0-pytorch`)
      - Model 8: Amazon Chronos-2 Multivariate + Covariates (`amazon/chronos-2`)
    with non-negative simplex weights and Station-Specific & Horizon-Decaying Variance
    Shrinkage $\alpha_s(h_{\text{lead}})$.
    """

    weights_: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.12, 0.24, 0.14, 0.16, 0.08, 0.08, 0.10, 0.08], dtype=float
        )
    )
    shrinkage_alpha_: float = 0.96
    intercept_shift_: float = 0.0
    station_alphas_: dict[int, float] = field(
        default_factory=lambda: {
            11606: 0.88,    # Borivali East (coastal/park variance damping)
            11611: 0.96,    # Malad West
            3409510: 0.97,  # Byculla
            3409511: 0.98,  # Shivaji Nagar
        }
    )
    station_shifts_: dict[int, float] = field(
        default_factory=lambda: {11606: 0.0, 11611: 0.0, 3409510: 0.0, 3409511: 0.0}
    )
    m1_: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    m2_: HuberRidgePhysicsGAMForecaster = field(
        default_factory=HuberRidgePhysicsGAMForecaster
    )
    m3_: LightGBMResidualForecaster = field(
        default_factory=LightGBMResidualForecaster
    )
    m4_: CatBoostResidualForecaster = field(
        default_factory=CatBoostResidualForecaster
    )
    m5_: XGBoostResidualForecaster = field(
        default_factory=XGBoostResidualForecaster
    )
    m6_: WindGuidedSTGATForecaster = field(
        default_factory=WindGuidedSTGATForecaster
    )
    m7_: TimesFM3CompactForecaster = field(
        default_factory=TimesFM3CompactForecaster
    )
    m8_: Chronos2CovariateForecaster = field(
        default_factory=Chronos2CovariateForecaster
    )

    def fit_models(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> "StackedAirQualityEnsemble":
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        sw = (
            train_feat_df["sample_weight"].values
            if "sample_weight" in train_feat_df.columns
            else np.ones(len(train_feat_df))
        )
        self.m1_.fit(train_feat_df, target)
        self.m2_.fit(train_feat_df, target, sample_weight=sw)
        self.m3_.fit(train_feat_df, target, sample_weight=sw)
        self.m4_.fit(train_feat_df, target, sample_weight=sw)
        self.m5_.fit(train_feat_df, target, sample_weight=sw)
        self.m6_.fit(train_feat_df, target, sample_weight=sw)
        self.m7_.fit(train_feat_df, target, sample_weight=sw)
        self.m8_.fit(train_feat_df, target, sample_weight=sw)
        return self

    def predict_individual(self, feat_df: pd.DataFrame) -> dict[str, np.ndarray]:
        tfm_out = self.m7_.predict_with_quantiles(feat_df)
        c2_out = self.m8_.predict_with_quantiles(feat_df)
        p_m1 = self.m1_.predict(feat_df)
        p_m2 = self.m2_.predict(feat_df)
        p_m3 = self.m3_.predict(feat_df)
        p_m4 = self.m4_.predict(feat_df)
        p_m5 = self.m5_.predict(feat_df)
        p_m6 = self.m6_.predict(feat_df)
        p_m7 = tfm_out["pred"]
        p_m8 = c2_out["pred"]

        # Regime-aware station-day de-drifting for residual models (M2..M6) during post-episode cleansing:
        # When training cutoffs span a rising buildup (e.g. Oct 5-23) and the forecast cutoff enters a
        # cleansing/flushing regime (regime_gate_t0 > 0), remove the rising-window station-day drift
        # from M2..M6 while preserving their within-day diurnal residual innovations.
        if "regime_gate_t0" in feat_df.columns and np.any(feat_df["regime_gate_t0"].values > 0.05):
            rg = np.clip(feat_df["regime_gate_t0"].values, 0.0, 1.0)
            tmp = pd.DataFrame(
                {
                    "loc": feat_df["location_id"].values,
                    "dt": pd.to_datetime(feat_df["target_start_ist"]).dt.date.values,
                    "m1": p_m1,
                    "m2": p_m2,
                    "m3": p_m3,
                    "m4": p_m4,
                    "m5": p_m5,
                    "m6": p_m6,
                    "m7": p_m7,
                }
            )
            grp = tmp.groupby(["loc", "dt"])
            m1_st = grp["m1"].transform("mean").values
            m7_st = grp["m7"].transform("mean").values
            clean_st_anchor = 0.20 * m1_st + 0.80 * m7_st
            m7_dm = p_m7 - m7_st

            def _dedrift_residual_model(p_mod: np.ndarray, col: str, dm_blend: float = 0.78) -> np.ndarray:
                mod_st = grp[col].transform("mean").values
                mod_dm = p_mod - mod_st
                target_st = (1.0 - 0.96 * rg) * mod_st + (0.96 * rg) * clean_st_anchor
                combined_dm = (1.0 - dm_blend * rg) * mod_dm + (dm_blend * rg) * m7_dm
                return np.asarray(np.clip(target_st + combined_dm, 2.0, 450.0), dtype=float)

            p_m2 = _dedrift_residual_model(p_m2, "m2", dm_blend=0.72)
            p_m3 = _dedrift_residual_model(p_m3, "m3", dm_blend=0.78)
            p_m4 = _dedrift_residual_model(p_m4, "m4", dm_blend=0.78)
            p_m5 = _dedrift_residual_model(p_m5, "m5", dm_blend=0.78)
            p_m6 = _dedrift_residual_model(p_m6, "m6", dm_blend=0.78)

        return {
            "m1_calibrated_cams_prior": p_m1,
            "m2_huber_ridge_gam": p_m2,
            "m3_lightgbm_residual": p_m3,
            "m4_catboost_residual": p_m4,
            "m5_xgboost_residual": p_m5,
            "m6_wind_st_gat": p_m6,
            "m7_timesfm3_compact": p_m7,
            "m8_chronos2_covariate": p_m8,
            # Uncertainty fan quantiles for downstream Streamlit & notebook diagnostics
            "tfm3_q10": tfm_out["q10"],
            "tfm3_q25": tfm_out["q25"],
            "tfm3_q50": tfm_out["q50"],
            "tfm3_q75": tfm_out["q75"],
            "tfm3_q90": tfm_out["q90"],
            "tfm3_iqr_80": tfm_out["iqr_80"],
            "tfm3_skewness": tfm_out["skewness"],
            "chronos2_q10": c2_out["q10"],
            "chronos2_q50": c2_out["q50"],
            "chronos2_q90": c2_out["q90"],
            "chronos2_iqr_80": c2_out["iqr_80"],
        }

    def apply_station_shrinkage(
        self,
        raw_blend: np.ndarray,
        prior_anchor: np.ndarray,
        location_ids: np.ndarray,
        horizon_hours: np.ndarray | None = None,
        regime_gate: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply Station-Specific & Horizon-Decaying Variance Shrinkage $\alpha_s(h_{\text{lead}})$
        toward the conditional physics prior, plus regime-aware station bias calibration.
        """
        out = np.zeros_like(raw_blend, dtype=float)
        locs = np.asarray(location_ids, dtype=int)
        if horizon_hours is not None:
            h = np.maximum(np.asarray(horizon_hours, dtype=float), 0.0)
            h_decay = np.clip(1.0 - 0.04 * np.maximum(h - 72.0, 0.0) / 72.0, 0.90, 1.0)
        else:
            h_decay = np.ones_like(raw_blend, dtype=float)

        rg = (
            np.clip(np.asarray(regime_gate, dtype=float), 0.0, 1.0)
            if regime_gate is not None
            else np.zeros_like(raw_blend, dtype=float)
        )

        for st_id in STATION_ORDER:
            mask = locs == st_id
            if not np.any(mask):
                continue
            base_alpha = float(self.station_alphas_.get(st_id, self.shrinkage_alpha_))
            eff_alpha = 1.0 + (base_alpha - 1.0) * (1.0 - 0.92 * rg[mask])
            alpha_s = eff_alpha * h_decay[mask]
            base_shift = float(self.station_shifts_.get(st_id, self.intercept_shift_))
            # Damp rising-regime station shifts during cleansing/flushing regimes
            shift_s = base_shift * (1.0 - 0.95 * rg[mask])
            out[mask] = (
                alpha_s * raw_blend[mask]
                + (1.0 - alpha_s) * prior_anchor[mask]
                + shift_s
            )
        return np.asarray(np.clip(out, 2.0, 450.0), dtype=float)

    def optimize_weights_on_oof(
        self,
        oof_preds_matrix: np.ndarray,
        y_true: np.ndarray,
        prior_anchor: np.ndarray,
        location_ids: np.ndarray | None = None,
        horizon_hours: np.ndarray | None = None,
        min_diversity_floor: float = 0.03,
    ) -> dict[str, Any]:
        """
        Optimize non-negative ensemble weights `w` across all 8 models (with diversity floor
        so all paradigms contribute) and Station-Specific Variance Shrinkage $\alpha_s$
        via `scipy.optimize.minimize` (SLSQP followed by Nelder-Mead refinement).
        """
        n_models = oof_preds_matrix.shape[1]
        y_vec = np.asarray(y_true, dtype=float)
        p_anchor = np.asarray(prior_anchor, dtype=float)

        def loss_fn(w: np.ndarray) -> float:
            blend = oof_preds_matrix @ w
            return float(np.sqrt(np.mean((y_vec - blend) ** 2)))

        init_w = np.ones(n_models, dtype=float) / n_models
        bounds = [(min_diversity_floor, 0.48) for _ in range(n_models)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res_slsqp = minimize(
            loss_fn,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 600, "ftol": 1e-9},
        )
        w_opt = np.clip(res_slsqp.x, min_diversity_floor, 0.48)
        w_opt = w_opt / np.sum(w_opt)

        # Refine with Nelder-Mead in simplex-projected coordinates
        def simplex_loss(raw_logits: np.ndarray) -> float:
            w = np.clip(raw_logits, min_diversity_floor, 0.48)
            w = w / np.sum(w)
            blend = oof_preds_matrix @ w
            return float(np.sqrt(np.mean((y_vec - blend) ** 2)))

        res_nm = minimize(
            simplex_loss,
            w_opt,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-7},
        )
        w_final = np.clip(res_nm.x, min_diversity_floor, 0.48)
        w_final = w_final / np.sum(w_final)
        self.weights_ = w_final

        raw_blend = oof_preds_matrix @ self.weights_
        rmse_before_shrink = float(loss_fn(self.weights_))

        # Global shrinkage optimization
        def shrink_loss(params: np.ndarray) -> float:
            alpha, shift = params
            calibrated = alpha * raw_blend + (1.0 - alpha) * p_anchor + shift
            return float(np.sqrt(np.mean((y_vec - calibrated) ** 2)))

        res_cal = minimize(
            shrink_loss,
            np.array([0.96, 0.0]),
            method="Nelder-Mead",
            bounds=[(0.75, 1.05), (-2.0, 2.0)],
        )
        alpha_opt = float(np.clip(res_cal.x[0], 0.80, 1.02))
        shift_opt = float(np.clip(res_cal.x[1], -1.5, 1.5))
        self.shrinkage_alpha_ = alpha_opt
        self.intercept_shift_ = shift_opt

        # Station-Specific Variance Shrinkage optimization (fixing Borivali East 11606 over-dispersion)
        if location_ids is not None:
            locs = np.asarray(location_ids, dtype=int)
            for st_id in STATION_ORDER:
                m_st = locs == st_id
                if not np.any(m_st):
                    continue
                yb = y_vec[m_st]
                rb = raw_blend[m_st]
                pb = p_anchor[m_st]

                def st_loss(p_st: np.ndarray) -> float:
                    a_s, sh_s = p_st
                    cal_s = a_s * rb + (1.0 - a_s) * pb + sh_s
                    # Regularize gently toward global alpha_opt, shift_opt
                    reg = 0.08 * ((a_s - alpha_opt) ** 2) + 0.02 * (sh_s ** 2)
                    return float(np.sqrt(np.mean((yb - cal_s) ** 2)) + reg)

                r_st = minimize(
                    st_loss,
                    np.array([alpha_opt, shift_opt]),
                    method="Nelder-Mead",
                    bounds=[(0.72, 1.04), (-1.8, 1.8)],
                )
                self.station_alphas_[st_id] = float(np.clip(r_st.x[0], 0.75, 1.02))
                self.station_shifts_[st_id] = float(np.clip(r_st.x[1], -1.5, 1.5))

            cal_full = self.apply_station_shrinkage(
                raw_blend=raw_blend,
                prior_anchor=p_anchor,
                location_ids=locs,
                horizon_hours=horizon_hours,
            )
            rmse_after_shrink = float(np.sqrt(np.mean((y_vec - cal_full) ** 2)))
        else:
            for st_id in STATION_ORDER:
                self.station_alphas_[st_id] = alpha_opt
                self.station_shifts_[st_id] = shift_opt
            rmse_after_shrink = float(shrink_loss([alpha_opt, shift_opt]))

        return {
            "weights": {
                name: float(round(w, 4))
                for name, w in zip(MODEL_ZOO_NAMES, self.weights_)
            },
            "shrinkage_alpha": round(self.shrinkage_alpha_, 4),
            "intercept_shift": round(self.intercept_shift_, 4),
            "station_alphas": {str(k): round(v, 4) for k, v in self.station_alphas_.items()},
            "station_shifts": {str(k): round(v, 4) for k, v in self.station_shifts_.items()},
            "oof_rmse_before_shrinkage": round(rmse_before_shrink, 4),
            "oof_rmse_after_shrinkage": round(rmse_after_shrink, 4),
        }

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        indiv = self.predict_individual(feat_df)
        mat = np.column_stack([indiv[name] for name in MODEL_ZOO_NAMES])
        raw_blend = mat @ self.weights_
        p_anchor = 0.55 * indiv["m1_calibrated_cams_prior"] + 0.45 * indiv["m2_huber_ridge_gam"]
        if "regime_gate_t0" in feat_df.columns:
            rg = np.clip(feat_df["regime_gate_t0"].values, 0.0, 1.0)
            p_clean_blend = (
                0.04 * indiv["m1_calibrated_cams_prior"]
                + 0.12 * indiv["m3_lightgbm_residual"]
                + 0.24 * indiv["m6_wind_st_gat"]
                + 0.45 * indiv["m7_timesfm3_compact"]
                + 0.15 * indiv["m8_chronos2_covariate"]
            )
            raw_blend = (1.0 - 0.94 * rg) * raw_blend + (0.94 * rg) * p_clean_blend
            p_anchor = (1.0 - rg) * p_anchor + rg * p_clean_blend
        else:
            rg = None
        return self.apply_station_shrinkage(
            raw_blend=raw_blend,
            prior_anchor=p_anchor,
            location_ids=feat_df["location_id"].values,
            horizon_hours=feat_df["horizon_hours"].values if "horizon_hours" in feat_df.columns else None,
            regime_gate=rg,
        )


def run_rolling_cv_evaluation(
    gap_hours: int = 24,
    n_train_cutoffs: int = 24,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Execute the complete 5-Fold Post-Monsoon Rolling 48-hour Validation Split,
    training all 8 Model Zoo architectures (`Models 1..8`) per fold strictly on data prior
    to `cutoff_t0`, and computing Out-of-Fold (OOF) RMSE + optimal ensemble weights.
    """
    train, _, _, _ = load_and_align_core_data()
    store = get_external_data_store()
    folds: list[ValidationFold] = get_rolling_validation_splits(train, gap_hours=gap_hours)

    oof_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    print("=" * 98)
    print(
        f"RUNNING 8-MODEL 5-FOLD POST-MONSOON 48H ROLLING VALIDATION (REPORTING GAP = {gap_hours}h)"
    )
    print("=" * 98)

    for fold in folds:
        print(
            f"\n[{fold.name}] Cutoff T0={fold.cutoff_t0} -> Validation=[{fold.val_start} .. {fold.val_end}) (n={len(fold.val_df)})"
        )
        # 0. Reference 90-day station mean (demonstrating monsoon-to-post-monsoon regime shift)
        h90 = fold.train_df.loc[
            fold.train_df["hour_end"] > fold.cutoff_t0 - pd.Timedelta(days=90)
        ]
        s_mean_90 = h90.groupby("location_id")["pm25"].mean()
        pred_b0 = (
            fold.val_df["location_id"]
            .map(s_mean_90)
            .fillna(h90["pm25"].mean())
            .values
        )

        # 1. Starter Notebook Random Forest Baseline (90d window)
        pred_b1 = predict_starter_rf_baseline(
            fold.train_df, fold.val_df, fold.cutoff_t0
        )

        # 2. 7-day Diurnal Shrinkage Baseline
        pred_b2 = predict_diurnal_shrinkage_baseline(
            fold.train_df, fold.val_df, fold.cutoff_t0
        )

        # Build multi-cutoff historical training dataset strictly prior to fold.cutoff_t0
        tr_multi = build_multi_cutoff_training_dataset(
            train_df=fold.train_df,
            max_cutoff_t0=fold.cutoff_t0,
            n_cutoffs=n_train_cutoffs,
            step_hours=24,
            min_gap_hours=max(12, gap_hours - 12),
            max_horizon_hours=gap_hours + 60,
            include_oct_nov_2025=True,
            store=store,
        )
        val_feats = build_features_for_cutoff(
            history_df=fold.train_df,
            target_df=fold.val_df,
            cutoff_t0=fold.cutoff_t0,
            store=store,
        )
        val_feats["cutoff_t0"] = fold.cutoff_t0

        # Train Models 1..8 on tr_multi
        ens = StackedAirQualityEnsemble()
        ens.fit_models(tr_multi, tr_multi["pm25_clean"].values)
        indiv_preds = ens.predict_individual(val_feats)

        rec = fold.val_df[
            ["location_id", "station_name", "hour_start", "hour_end", "pm25", "pm25_clean"]
        ].copy()
        rec["fold_id"] = fold.fold_id
        rec["fold_name"] = fold.name
        rec["horizon_hours"] = val_feats["horizon_hours"].values
        rec["b0_station_mean_90d"] = pred_b0
        rec["b1_starter_rf_90d"] = pred_b1
        rec["b2_diurnal_shrinkage_7d"] = pred_b2
        for m_name, m_pred in indiv_preds.items():
            rec[m_name] = m_pred

        oof_records.append(rec)

        y_val = rec["pm25"].values
        f_row: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "fold_name": fold.name,
            "n_obs": len(y_val),
            "b0_station_mean_90d": rmse(y_val, pred_b0),
            "b1_starter_rf_90d": rmse(y_val, pred_b1),
            "b2_diurnal_shrinkage_7d": rmse(y_val, pred_b2),
        }
        for m_name in MODEL_ZOO_NAMES:
            f_row[m_name] = rmse(y_val, rec[m_name].values)
        fold_metrics.append(f_row)

        print(
            f"  RMSEs -> StarterRF: {f_row['b1_starter_rf_90d']:.3f} | "
            f"M1_Prior: {f_row['m1_calibrated_cams_prior']:.3f} | "
            f"M2_GAM_AnEn: {f_row['m2_huber_ridge_gam']:.3f} | "
            f"M3_LGBM: {f_row['m3_lightgbm_residual']:.3f} | "
            f"M4_CatB: {f_row['m4_catboost_residual']:.3f} | "
            f"M5_XGB: {f_row['m5_xgboost_residual']:.3f} | "
            f"M6_STGAT: {f_row['m6_wind_st_gat']:.3f} | "
            f"M7_TimesFM3: {f_row['m7_timesfm3_compact']:.3f} | "
            f"M8_Chronos2: {f_row['m8_chronos2_covariate']:.3f}"
        )

    oof_df = pd.concat(oof_records, ignore_index=True)
    oof_mat = np.column_stack([oof_df[m].values for m in MODEL_ZOO_NAMES])
    y_oof = oof_df["pm25"].values
    p_anchor_oof = (
        0.55 * oof_df["m1_calibrated_cams_prior"].values
        + 0.45 * oof_df["m2_huber_ridge_gam"].values
    )

    # Optimize 8-model ensemble weights and station-specific variance shrinkage
    meta_ens = StackedAirQualityEnsemble()
    opt_info = meta_ens.optimize_weights_on_oof(
        oof_preds_matrix=oof_mat,
        y_true=y_oof,
        prior_anchor=p_anchor_oof,
        location_ids=oof_df["location_id"].values,
        horizon_hours=oof_df["horizon_hours"].values,
    )

    raw_blend = oof_mat @ meta_ens.weights_
    ens_oof_pred = meta_ens.apply_station_shrinkage(
        raw_blend=raw_blend,
        prior_anchor=p_anchor_oof,
        location_ids=oof_df["location_id"].values,
        horizon_hours=oof_df["horizon_hours"].values,
    )
    oof_df["m7_stacked_ensemble"] = ens_oof_pred
    oof_df["m9_stacked_ensemble"] = ens_oof_pred

    for idx_f, f_row in enumerate(fold_metrics, start=1):
        sub_f = oof_df.loc[oof_df["fold_id"] == idx_f]
        f_rmse = rmse(sub_f["pm25"].values, sub_f["m7_stacked_ensemble"].values)
        f_row["m7_stacked_ensemble"] = f_rmse
        f_row["m9_stacked_ensemble"] = f_rmse

    # Build comprehensive leaderboard summary table
    all_keys = [
        "b0_station_mean_90d",
        "b1_starter_rf_90d",
        "b2_diurnal_shrinkage_7d",
        *MODEL_ZOO_NAMES,
        "m7_stacked_ensemble",
    ]
    lb_rows: list[dict[str, Any]] = []
    starter_oof_rmse = rmse(y_oof, oof_df["b1_starter_rf_90d"].values)
    ref90_oof_rmse = rmse(y_oof, oof_df["b0_station_mean_90d"].values)

    for k in all_keys:
        overall_rmse = rmse(y_oof, oof_df[k].values)
        fold_vals = [fm[k] for fm in fold_metrics]
        row_dict: dict[str, Any] = {
            "model_key": k,
            "Model": LEADERBOARD_DISPLAY_NAMES[k],
            "OOF_RMSE": round(overall_rmse, 4),
            "Mean_Fold_RMSE": round(float(np.mean(fold_vals)), 4),
            "Std_Fold_RMSE": round(float(np.std(fold_vals)), 4),
            "Fold1_Oct12_13": round(fold_vals[0], 3),
            "Fold2_Oct14_15": round(fold_vals[1], 3),
            "Fold3_Oct16_17": round(fold_vals[2], 3),
            "Fold4_Oct18_19": round(fold_vals[3], 3),
            "Fold5_Oct20_21": round(fold_vals[4], 3),
            "Gain_vs_90d_Ref_pct": round(100.0 * (ref90_oof_rmse - overall_rmse) / ref90_oof_rmse, 2),
            "Gain_vs_StarterRF_pct": round(100.0 * (starter_oof_rmse - overall_rmse) / starter_oof_rmse, 2),
        }
        lb_rows.append(row_dict)

    leaderboard_df = pd.DataFrame(lb_rows)

    print("\n" + "=" * 98)
    print(f"5-FOLD POST-MONSOON VALIDATION LEADERBOARD (GAP = {gap_hours}h, N = {len(oof_df)} STATION-HOURS)")
    print("=" * 98)
    print(
        leaderboard_df[
            [
                "Model",
                "OOF_RMSE",
                "Mean_Fold_RMSE",
                "Fold1_Oct12_13",
                "Fold2_Oct14_15",
                "Fold3_Oct16_17",
                "Fold4_Oct18_19",
                "Fold5_Oct20_21",
                "Gain_vs_StarterRF_pct",
            ]
        ].to_string(index=False)
    )
    print("-" * 98)
    print("Optimized 8-Model Ensemble Weights:", opt_info["weights"])
    print(
        f"Global Shrinkage Alpha: {opt_info['shrinkage_alpha']} | "
        f"Station Alphas: {opt_info['station_alphas']}"
    )
    print("=" * 98)

    if save_artifacts:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        oof_df.to_parquet(PROCESSED_DIR / f"oof_predictions_gap{gap_hours}h.parquet", index=False)
        oof_df.to_csv(PROCESSED_DIR / f"oof_predictions_gap{gap_hours}h.csv", index=False)
        leaderboard_df.to_csv(
            PROCESSED_DIR / f"validation_leaderboard_gap{gap_hours}h.csv", index=False
        )
        with open(PROCESSED_DIR / f"ensemble_weights_gap{gap_hours}h.json", "w", encoding="utf-8") as f:
            json.dump(opt_info, f, indent=2)

    return leaderboard_df, oof_df, opt_info


def main() -> None:
    # Primary evaluation on short reporting gap (24h, matching Nov 10 -> Nov 11-12 Private window)
    lb_24, _, _ = run_rolling_cv_evaluation(gap_hours=24, n_train_cutoffs=24, save_artifacts=True)
    # Also evaluate on 72h reporting gap (matching Starter Notebook 3-day gap)
    lb_72, _, _ = run_rolling_cv_evaluation(gap_hours=72, n_train_cutoffs=20, save_artifacts=True)


if __name__ == "__main__":
    main()
