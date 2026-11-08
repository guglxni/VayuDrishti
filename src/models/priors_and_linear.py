#!/usr/bin/env python3
r"""
Model 1 & Model 2: TimeMixer-Inspired Synoptic-Diurnal Physics Prior &
Dual-Scale Huber/Ridge/Duan-Smearing + Meteorological Analog Ensemble (`AnEn`) GAM
(`src/models/priors_and_linear.py`) for the Google Mumbai Air Quality Challenge.

Implements:
- Model 1 (`CalibratedDiurnalCAMSPriorForecaster`):
  1. TimeMixer-Inspired Synoptic-Diurnal Decomposition anchored at `T0`.
  2. Bounded $\tanh$-Saturating CAMS MOS Multiplier on log CAMS ratio (`cams_log_ratio_h`
     and `cams_log_ratio_24h`) so extreme CAMS dust hallucinations ($> +45\%$, e.g. Fold 4)
     smoothly saturate instead of exploding linearly.
  3. Hygroscopic (`RH > 68-75%`), Boundary-Layer (`PBLH`), and Column-AOD MOS correction
     plus ventilation anomaly $\exp(-\beta_{\text{vc}} \Delta \log(1 + \text{VC}))$ and
     `AirPhyNet` upwind advection flux coupling.
- Model 2 (`HuberRidgePhysicsGAMForecaster`):
  1. Raw-scale Additive Spline GAM (Ridge + HuberRegressor) over periodic hour splines,
     PBLH/CAMS cubic splines, `AirPhyNet` advection-diffusion PDE fluxes, and station interactions.
  2. Log-Ratio Ridge GAM predicting $\log((y + 15) / (\hat{y}_{\text{prior}} + 15))$
     back-transformed with **Duan's Smearing Estimator** $\hat{S}_{\text{Duan}} = \frac{1}{W}\sum_i w_i \exp(\hat{\varepsilon}_i)$.
  3. Non-Parametric **Historical Meteorological Analog Ensemble (`AnEn`)** matching historical
     post-monsoon `(location_id, hour)` trajectories in standardized meteorological + CAMS space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CalibratedDiurnalCAMSPriorForecaster:
    """
    Model 1: TimeMixer-Inspired Synoptic-Diurnal + tanh-Saturating CAMS MOS Prior Forecaster.
    Decomposes the physics prior into:
      1. Multi-scale Empirical Bayes + fast-adapting diurnal profile anchored at `T0`.
      2. Bounded tanh-saturating log-CAMS synoptic multiplier with hygroscopic & PBLH MOS damping.
      3. Boundary-layer ventilation anomaly scaling + AirPhyNet upwind advection flux adjustment.
    """

    cams_exponent: float = 0.42
    vent_sensitivity: float = 0.075
    diurnal_weight_eb: float = 0.55
    diurnal_weight_adaptive: float = 0.45
    cams_direct_blend: float = 0.22
    advection_gain: float = 0.06

    def _compute_prior_from_params(
        self,
        feat_df: pd.DataFrame,
        c_exp: float,
        v_sens: float,
        c_blend: float,
        adv_gain: float,
    ) -> np.ndarray:
        h = np.maximum(feat_df["horizon_hours"].values, 0.0)
        h_long = np.clip((h - 48.0) / 48.0, 0.0, 1.0)
        rg = (
            feat_df["regime_gate_t0"].values
            if "regime_gate_t0" in feat_df.columns
            else np.zeros(len(feat_df))
        )
        regime_damp = np.clip(1.0 - rg, 0.0, 1.0)

        # Start from the regime-adaptive Spatial James-Stein + HOD-debiased physics prior
        base_phys = feat_df["prior_physics_calibrated"].values
        cams_rel = np.clip(feat_df["cams_rel_factor"].values, 0.40, 2.2)
        vent_anom = feat_df["wx_anom_log1p_vc"].values

        # Apply differential parameter adjustments around the reference (0.42, 0.075, 0.22)
        delta_exp = (c_exp - 0.42) * regime_damp
        delta_vsens = (v_sens - 0.075) * regime_damp
        mod_factor = np.clip(
            (cams_rel ** delta_exp) * np.exp(-delta_vsens * vent_anom),
            0.88,
            1.14,
        )
        pred = base_phys * mod_factor

        cams_cal = (
            0.65 * feat_df["cams_pm25_calibrated"].values
            + 0.35 * feat_df["cams_pm25_offset_calibrated"].values
        )
        delta_blend = (c_blend - 0.22) * regime_damp
        pred = pred + delta_blend * (cams_cal - base_phys)

        # AirPhyNet upwind advection PDE correction (bounded so local spikes never overshoot)
        if "airphynet_pde_total_flux" in feat_df.columns and adv_gain > 0.0:
            pde_adj = np.clip(
                adv_gain * regime_damp * feat_df["airphynet_pde_total_flux"].values,
                -3.5,
                3.5,
            )
            pred = pred + pde_adj

        # Gentle climatological drift adjustment for ultra-long horizons (>96h)
        long_h_weight = np.clip((h - 96.0) / 260.0, 0.0, 0.22)
        clim_prior = feat_df["clim_scaled_prior"].values
        pred = (1.0 - long_h_weight) * pred + long_h_weight * clim_prior
        return np.asarray(np.clip(pred, 2.0, 450.0), dtype=float)

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> "CalibratedDiurnalCAMSPriorForecaster":
        """Fit optimal exponent, ventilation sensitivity, CAMS MOS blend, and AirPhyNet advection gain."""
        if train_feat_df is None or len(train_feat_df) == 0:
            return self
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )

        best_rmse = float("inf")
        best_params = (
            self.cams_exponent,
            self.vent_sensitivity,
            self.cams_direct_blend,
            self.advection_gain,
        )
        for c_exp in [0.38, 0.42, 0.46]:
            for v_sens in [0.06, 0.075, 0.09]:
                for c_blend in [0.16, 0.22, 0.26]:
                    for adv_g in [0.0, 0.05, 0.10]:
                        pred = self._compute_prior_from_params(
                            train_feat_df, c_exp, v_sens, c_blend, adv_g
                        )
                        err = float(np.sqrt(np.mean((target - pred) ** 2)))
                        if err < best_rmse:
                            best_rmse = err
                            best_params = (c_exp, v_sens, c_blend, adv_g)

        (
            self.cams_exponent,
            self.vent_sensitivity,
            self.cams_direct_blend,
            self.advection_gain,
        ) = best_params
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        return self._compute_prior_from_params(
            feat_df,
            self.cams_exponent,
            self.vent_sensitivity,
            self.cams_direct_blend,
            self.advection_gain,
        )


GAM_CONT_FEATURES: list[str] = [
    "prior_eb_diurnal",
    "prior_diurnal_adaptive",
    "prior_physics_calibrated",
    "cams_pm25_calibrated",
    "cams_pm25_offset_calibrated",
    "cams_pm25_rel_change",
    "cams_rel_factor",
    "cams_log_ratio_mos",
    "cams_mos_damp_factor",
    "cams_pm25_diff3h",
    "cams_pm25_to_pm10",
    "airphynet_advection_flux",
    "airphynet_cams_advection_flux",
    "airphynet_diffusion_laplacian",
    "airphynet_pde_total_flux",
    "wx_anom_pblh",
    "wx_ratio_pblh",
    "wx_anom_log1p_vc",
    "wx_anom_temp",
    "wx_anom_rh",
    "wx_anom_wind",
    "wx_anom_onshore",
    "wx_anom_dewdep",
    "log1p_vc",
    "log_pblh",
    "u10",
    "v10",
    "onshore_breeze",
    "deonar_wind_cos",
    "thane_creek_wind_cos",
    "deonar_transport_proxy",
    "dewpoint_depression",
    "delta_t_3h",
    "delta_rh_3h",
    "stagnation_hours",
    "station_trend_3d_vs_14d",
    "station_trend_1d_vs_7d",
    "station_to_city_ratio_t0",
    "hour_sin_1",
    "hour_cos_1",
    "hour_sin_2",
    "hour_cos_2",
    "hour_sin_3",
    "hour_cos_3",
    "log1p_horizon",
]

ANALOG_MET_FEATURES: list[str] = [
    "cams_log_ratio_mos",
    "wx_anom_log1p_vc",
    "wx_anom_rh",
    "onshore_breeze",
    "dewpoint_depression",
    "airphynet_pde_total_flux",
]


@dataclass
class HuberRidgePhysicsGAMForecaster:
    """
    Model 2: Dual-Scale Huber/Ridge Physics GAM + Log-Ratio Duan Smearing +
    Historical Meteorological Analog Ensemble (`AnEn`).

    Combines three complementary statistical physics estimators:
      1. Additive Raw-Residual GAM (`Ridge` + `HuberRegressor` over periodic hour splines,
         PBLH/CAMS splines, and `AirPhyNet` advection-diffusion PDE features).
      2. Multiplicative Log-Ratio Ridge GAM predicting `log((y + 15) / (y_prior + 15))`
         back-transformed via **Duan's Smearing Estimator** `S_duan = E[exp(eps)]`.
      3. Non-Parametric **Historical Meteorological Analog Ensemble (`AnEn`)** using Gaussian
         RBF kernel matching over historical `(location_id, hour)` post-monsoon trajectories.
    """

    ridge_alpha: float = 650.0
    huber_alpha: float = 250.0
    huber_epsilon: float = 1.35
    log_ridge_alpha: float = 800.0
    huber_weight: float = 0.50
    raw_gam_weight: float = 0.45
    log_duan_weight: float = 0.35
    analog_weight: float = 0.20
    residual_damping: float = 0.40
    log_offset: float = 15.0
    analog_bandwidth: float = 1.65
    prior_model: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    ridge_pipe_: Pipeline | None = field(default=None, init=False)
    huber_pipe_: Pipeline | None = field(default=None, init=False)
    log_ridge_pipe_: Pipeline | None = field(default=None, init=False)
    duan_smearing_factor_: float = field(default=1.0, init=False)
    used_cont_cols_: list[str] = field(default_factory=list, init=False)
    analog_cols_: list[str] = field(default_factory=list, init=False)
    analog_scaler_: RobustScaler | None = field(default=None, init=False)
    analog_bank_: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False
    )

    def _prepare_gam_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.used_cont_cols_].copy()
        X["location_id_str"] = df["location_id"].astype(str)
        X["hour_num"] = df["hour"].astype(float)
        # Station x CAMS, Station x Ventilation, and Station x AirPhyNet Advection interactions
        for st_id in [11606, 11611, 3409510, 3409511]:
            ind = (df["location_id"] == st_id).astype(float)
            X[f"st_{st_id}_cams_diff"] = ind * (
                df["cams_pm25_calibrated"] - df["prior_eb_diurnal"]
            )
            X[f"st_{st_id}_vc_anom"] = ind * df["wx_anom_log1p_vc"]
            X[f"st_{st_id}_rh_anom"] = ind * df["wx_anom_rh"]
            if "airphynet_pde_total_flux" in df.columns:
                X[f"st_{st_id}_pde_flux"] = ind * df["airphynet_pde_total_flux"]
        return X

    def _make_preprocessor(self, num_cols: list[str]) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("station", OneHotEncoder(handle_unknown="ignore"), ["location_id_str"]),
                (
                    "hour_spline",
                    SplineTransformer(
                        n_knots=8, degree=3, knots="uniform", extrapolation="periodic"
                    ),
                    ["hour_num"],
                ),
                (
                    "pblh_spline",
                    SplineTransformer(n_knots=5, degree=2),
                    ["log_pblh", "cams_rel_factor"],
                ),
                ("num", RobustScaler(), num_cols),
            ],
            remainder="drop",
        )

    def _fit_analog_bank(
        self,
        train_feat_df: pd.DataFrame,
        target: np.ndarray,
        y_prior: np.ndarray,
        sw: np.ndarray,
    ) -> None:
        """Build Historical Meteorological Analog (`AnEn`) lookup bank per `(location_id, hour)`."""
        self.analog_cols_ = [c for c in ANALOG_MET_FEATURES if c in train_feat_df.columns]
        if not self.analog_cols_:
            return
        self.analog_scaler_ = RobustScaler()
        Z_all = np.clip(
            self.analog_scaler_.fit_transform(train_feat_df[self.analog_cols_].values),
            -4.0,
            4.0,
        )
        # Target residual relative to y_prior (damped for outlier robustness)
        res_clipped = np.clip(target - y_prior, -35.0, 35.0)
        locs = train_feat_df["location_id"].astype(int).values
        hrs = train_feat_df["hour"].astype(int).values

        self.analog_bank_ = {}
        for st_id in [11606, 11611, 3409510, 3409511]:
            for hr in range(24):
                mask = (locs == st_id) & (hrs == hr)
                if np.any(mask):
                    self.analog_bank_[(st_id, hr)] = (
                        Z_all[mask],
                        res_clipped[mask],
                        sw[mask],
                    )

    def _predict_analog_residual(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Compute Gaussian RBF kernel-weighted historical meteorological analog residual."""
        n = len(feat_df)
        out_res = np.zeros(n, dtype=float)
        if not self.analog_cols_ or self.analog_scaler_ is None or not self.analog_bank_:
            return out_res

        Z_query = np.clip(
            self.analog_scaler_.transform(feat_df[self.analog_cols_].values),
            -4.0,
            4.0,
        )
        locs = feat_df["location_id"].astype(int).values
        hrs = feat_df["hour"].astype(int).values
        inv_2sigma2 = 1.0 / (2.0 * (self.analog_bandwidth ** 2) * len(self.analog_cols_))

        for (st_id, hr), (Z_hist, res_hist, w_hist) in self.analog_bank_.items():
            q_mask = (locs == st_id) & (hrs == hr)
            if not np.any(q_mask):
                continue
            Zq = Z_query[q_mask]  # (M, D)
            # Squared Euclidean distance (M, N_hist)
            dist_sq = np.sum((Zq[:, None, :] - Z_hist[None, :, :]) ** 2, axis=-1)
            kern = np.exp(-dist_sq * inv_2sigma2) * w_hist[None, :]
            denom = np.sum(kern, axis=-1) + 1e-6
            out_res[q_mask] = np.sum(kern * res_hist[None, :], axis=-1) / denom

        return out_res

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "HuberRidgePhysicsGAMForecaster":
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        self.prior_model.fit(train_feat_df, target)
        y_prior = self.prior_model.predict(train_feat_df)
        residual = np.clip(target - y_prior, -55.0, 55.0)

        self.used_cont_cols_ = [c for c in GAM_CONT_FEATURES if c in train_feat_df.columns]
        X_gam = self._prepare_gam_frame(train_feat_df)
        num_cols = [c for c in X_gam.columns if c not in ("location_id_str", "hour_num")]

        self.ridge_pipe_ = Pipeline(
            [
                ("pre", self._make_preprocessor(num_cols)),
                ("reg", Ridge(alpha=self.ridge_alpha, random_state=42)),
            ]
        )
        self.huber_pipe_ = Pipeline(
            [
                ("pre", self._make_preprocessor(num_cols)),
                (
                    "reg",
                    HuberRegressor(
                        alpha=self.huber_alpha,
                        epsilon=self.huber_epsilon,
                        max_iter=500,
                    ),
                ),
            ]
        )
        self.log_ridge_pipe_ = Pipeline(
            [
                ("pre", self._make_preprocessor(num_cols)),
                ("reg", Ridge(alpha=self.log_ridge_alpha, random_state=42)),
            ]
        )

        sw = (
            sample_weight
            if sample_weight is not None
            else (
                train_feat_df["sample_weight"].values
                if "sample_weight" in train_feat_df.columns
                else np.ones(len(train_feat_df))
            )
        )
        self.ridge_pipe_.fit(X_gam, residual, reg__sample_weight=sw)
        self.huber_pipe_.fit(X_gam, residual, reg__sample_weight=sw)

        # Fit Log-Ratio Ridge model and compute Duan's Smearing Estimator
        c0 = self.log_offset
        z_log_target = np.clip(
            np.log(np.maximum(target + c0, 2.0) / np.maximum(y_prior + c0, 2.0)),
            -0.85,
            0.85,
        )
        self.log_ridge_pipe_.fit(X_gam, z_log_target, reg__sample_weight=sw)
        z_pred_train = np.clip(self.log_ridge_pipe_.predict(X_gam), -0.65, 0.65)
        eps_log = np.clip(z_log_target - z_pred_train, -0.75, 0.75)
        self.duan_smearing_factor_ = float(
            np.clip(np.average(np.exp(eps_log), weights=sw), 0.96, 1.06)
        )

        # Fit non-parametric Historical Meteorological Analog (`AnEn`) bank
        self._fit_analog_bank(train_feat_df, target, y_prior, sw)
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        y_prior = self.prior_model.predict(feat_df)
        if self.ridge_pipe_ is None or self.huber_pipe_ is None or self.log_ridge_pipe_ is None:
            return y_prior

        X_gam = self._prepare_gam_frame(feat_df)
        # 1. Raw additive spline GAM residual
        res_ridge = self.ridge_pipe_.predict(X_gam)
        res_huber = self.huber_pipe_.predict(X_gam)
        res_raw_gam = self.huber_weight * res_huber + (1.0 - self.huber_weight) * res_ridge

        # 2. Multiplicative Log-Ratio Ridge GAM with Duan's Smearing Estimator
        c0 = self.log_offset
        z_hat = np.clip(
            self.log_ridge_pipe_.predict(X_gam) * self.residual_damping,
            -0.35,
            0.35,
        )
        # Normalize smearing factor smoothly with damping so zero-residual limit is unbiased
        eff_duan = 1.0 + self.residual_damping * (self.duan_smearing_factor_ - 1.0)
        y_log_duan = (y_prior + c0) * np.exp(z_hat) * eff_duan - c0
        res_log_duan = y_log_duan - y_prior

        # 3. Historical Meteorological Analog Ensemble (`AnEn`) residual
        res_analog = self._predict_analog_residual(feat_df)

        # Combine all three physics-statistical components
        total_res = (
            self.raw_gam_weight * (res_raw_gam * self.residual_damping)
            + self.log_duan_weight * res_log_duan
            + self.analog_weight * (res_analog * self.residual_damping)
        )
        if "regime_gate_t0" in feat_df.columns:
            rg = feat_df["regime_gate_t0"].values
            total_res = np.where(total_res > 0.0, total_res * (1.0 - 0.85 * rg), total_res)
        res_clipped = np.clip(total_res, -16.0, 16.0)
        return np.asarray(np.clip(y_prior + res_clipped, 2.0, 450.0), dtype=float)
