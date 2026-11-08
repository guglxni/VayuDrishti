#!/usr/bin/env python3
"""
Model 3, Model 4, & Model 5: Gradient Boosted Decision Tree Residual Zoo
(`src/models/gbdt_zoo.py`) for the Google Mumbai Air Quality Challenge.

Implements:
- Model 3 (`LightGBMResidualForecaster`):
  Blends `regression` (RMSE) and `huber` objectives predicting the residual
  `y - y_hat_prior` over the physics-calibrated diurnal + CAMS prior, with L1/L2
  leaf regularization and feature subsampling for tabular time series.
- Model 4 (`CatBoostResidualForecaster`):
  Symmetric oblivious trees (`Depth=5`) with `RMSE` + `Huber` blend on station,
  weather anomaly, CAMS chemical forecast, and multi-scale lag features.
- Model 5 (`XGBoostResidualForecaster`):
  Depth-regularized gradient boosted trees (`reg:squarederror` + `reg:pseudohubererror`)
  predicting residuals over the physics prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

# Prevent duplicate libomp (PyTorch + LightGBM/XGBoost) mutex collision on macOS arm64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import get_feature_columns
from src.models.priors_and_linear import CalibratedDiurnalCAMSPriorForecaster

EXCLUDED_GBDT_COLS: set[str] = {
    "sample_weight",
    "cutoff_t0",
    "wx_source",
    "cams_last_direct_ist",
}


def _select_gbdt_features(df: pd.DataFrame) -> list[str]:
    return [c for c in get_feature_columns(df) if c not in EXCLUDED_GBDT_COLS]


@dataclass
class LightGBMResidualForecaster:
    """
    Model 3: LightGBM Residual Forecaster.
    Predicts residual `y - y_hat_prior` using a dual-objective blend of L2 (`regression`)
    and robust Huber (`huber`) LightGBM boosters.
    """

    n_estimators: int = 240
    learning_rate: float = 0.03
    num_leaves: int = 23
    max_depth: int = 5
    min_child_samples: int = 45
    subsample: float = 0.80
    colsample_bytree: float = 0.72
    reg_alpha: float = 4.0
    reg_lambda: float = 12.0
    residual_damping: float = 0.72
    random_state: int = 42
    prior_model: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    model_rmse_: lgb.LGBMRegressor | None = field(default=None, init=False)
    model_huber_: lgb.LGBMRegressor | None = field(default=None, init=False)
    feature_cols_: list[str] = field(default=list, init=False)

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "LightGBMResidualForecaster":
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        self.prior_model.fit(train_feat_df, target)
        y_prior = self.prior_model.predict(train_feat_df)
        residual = np.clip(target - y_prior, -65.0, 65.0)

        self.feature_cols_ = _select_gbdt_features(train_feat_df)
        X = train_feat_df[self.feature_cols_].values
        sw = (
            sample_weight
            if sample_weight is not None
            else (
                train_feat_df["sample_weight"].values
                if "sample_weight" in train_feat_df.columns
                else np.ones(len(train_feat_df))
            )
        )

        common_params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbose=-1,
        )

        self.model_rmse_ = lgb.LGBMRegressor(objective="regression", **common_params)
        self.model_huber_ = lgb.LGBMRegressor(
            objective="huber", alpha=12.0, **common_params
        )

        self.model_rmse_.fit(X, residual, sample_weight=sw)
        self.model_huber_.fit(X, residual, sample_weight=sw)
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        y_prior = self.prior_model.predict(feat_df)
        if self.model_rmse_ is None or self.model_huber_ is None:
            return y_prior

        X = feat_df[self.feature_cols_].values
        res_rmse = self.model_rmse_.predict(X)
        res_huber = self.model_huber_.predict(X)
        res = 0.55 * res_rmse + 0.45 * res_huber
        res_damped = np.clip(res * self.residual_damping, -35.0, 35.0)
        return np.asarray(np.clip(y_prior + res_damped, 2.0, 450.0), dtype=float)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model_rmse_ is None or self.model_huber_ is None:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = 0.5 * (
            self.model_rmse_.feature_importances_ + self.model_huber_.feature_importances_
        )
        return (
            pd.DataFrame({"feature": self.feature_cols_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


@dataclass
class CatBoostResidualForecaster:
    """
    Model 4: CatBoost Residual Forecaster.
    Uses symmetric oblivious decision trees (`depth=5`) with L2 leaf regularization
    predicting the residual `y - y_hat_prior`.
    """

    iterations: int = 320
    learning_rate: float = 0.035
    depth: int = 5
    l2_leaf_reg: float = 10.0
    subsample: float = 0.80
    residual_damping: float = 0.72
    random_state: int = 42
    prior_model: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    model_rmse_: CatBoostRegressor | None = field(default=None, init=False)
    model_huber_: CatBoostRegressor | None = field(default=None, init=False)
    feature_cols_: list[str] = field(default=list, init=False)

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "CatBoostResidualForecaster":
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        self.prior_model.fit(train_feat_df, target)
        y_prior = self.prior_model.predict(train_feat_df)
        residual = np.clip(target - y_prior, -65.0, 65.0)

        self.feature_cols_ = _select_gbdt_features(train_feat_df)
        X = train_feat_df[self.feature_cols_].values
        sw = (
            sample_weight
            if sample_weight is not None
            else (
                train_feat_df["sample_weight"].values
                if "sample_weight" in train_feat_df.columns
                else np.ones(len(train_feat_df))
            )
        )

        self.model_rmse_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE",
            subsample=self.subsample,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        self.model_huber_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Huber:delta=12.0",
            subsample=self.subsample,
            random_seed=self.random_state + 7,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

        self.model_rmse_.fit(X, residual, sample_weight=sw)
        self.model_huber_.fit(X, residual, sample_weight=sw)
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        y_prior = self.prior_model.predict(feat_df)
        if self.model_rmse_ is None or self.model_huber_ is None:
            return y_prior

        X = feat_df[self.feature_cols_].values
        res_rmse = self.model_rmse_.predict(X)
        res_huber = self.model_huber_.predict(X)
        res = 0.55 * res_rmse + 0.45 * res_huber
        res_damped = np.clip(res * self.residual_damping, -35.0, 35.0)
        return np.asarray(np.clip(y_prior + res_damped, 2.0, 450.0), dtype=float)


@dataclass
class XGBoostResidualForecaster:
    """
    Model 5: XGBoost Residual Forecaster.
    Depth-regularized gradient boosted trees (`max_depth=4`, `colsample_bytree=0.70`)
    blending `reg:squarederror` and `reg:pseudohubererror` on the physics residual.
    """

    n_estimators: int = 220
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 25.0
    subsample: float = 0.80
    colsample_bytree: float = 0.70
    reg_alpha: float = 4.0
    reg_lambda: float = 14.0
    residual_damping: float = 0.72
    random_state: int = 42
    prior_model: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    model_rmse_: xgb.XGBRegressor | None = field(default=None, init=False)
    model_huber_: xgb.XGBRegressor | None = field(default=None, init=False)
    feature_cols_: list[str] = field(default=list, init=False)

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "XGBoostResidualForecaster":
        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        self.prior_model.fit(train_feat_df, target)
        y_prior = self.prior_model.predict(train_feat_df)
        residual = np.clip(target - y_prior, -65.0, 65.0)

        self.feature_cols_ = _select_gbdt_features(train_feat_df)
        X = train_feat_df[self.feature_cols_].values
        sw = (
            sample_weight
            if sample_weight is not None
            else (
                train_feat_df["sample_weight"].values
                if "sample_weight" in train_feat_df.columns
                else np.ones(len(train_feat_df))
            )
        )

        common_params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
        )

        self.model_rmse_ = xgb.XGBRegressor(
            objective="reg:squarederror", **common_params
        )
        self.model_huber_ = xgb.XGBRegressor(
            objective="reg:pseudohubererror", huber_slope=10.0, **common_params
        )

        self.model_rmse_.fit(X, residual, sample_weight=sw)
        self.model_huber_.fit(X, residual, sample_weight=sw)
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        y_prior = self.prior_model.predict(feat_df)
        if self.model_rmse_ is None or self.model_huber_ is None:
            return y_prior

        X = feat_df[self.feature_cols_].values
        res_rmse = self.model_rmse_.predict(X)
        res_huber = self.model_huber_.predict(X)
        res = 0.55 * res_rmse + 0.45 * res_huber
        res_damped = np.clip(res * self.residual_damping, -35.0, 35.0)
        return np.asarray(np.clip(y_prior + res_damped, 2.0, 450.0), dtype=float)
