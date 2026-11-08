#!/usr/bin/env python3
r"""
Model 6: Wind-Guided Spatiotemporal Graph Attention Network (`src/models/st_gat.py`)
inspired by `PM2.5-GNN` (Wang et al.), modeling the 4 Mumbai CAAQMS target stations
(`11606` Borivali East, `11611` Malad West, `3409510` Byculla, `3409511` Shivaji Nagar)
as a dynamic wind-directed graph in PyTorch.

Architecture:
1. Pairwise Haversine distances $d_{ij}$ and azimuth bearings $\theta_{j \to i}$ between the
   4 target stations.
2. Dynamic Wind-Advection Edge Weights:
   At each target hour $t$, given wind speed $w_j(t)$ and wind direction $\theta_{\text{wind}, j}(t)$,
   computes directional advection transport affinity:
     $A_{j \to i}(t) = \text{softmax}_j \left( -\frac{d_{ij}}{d_0} + \gamma \cdot w_j(t) \cos(\theta_{\text{wind}}^{\text{to}}(t) - \theta_{j \to i}) + W_e h_j \right)$
3. Multi-head Wind-Guided Graph Attention Layer + Residual Gated MLP predicting the
   station-hour residual over the physics-calibrated diurnal + CAMS prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import lightgbm  # noqa: F401
import xgboost  # noqa: F401
import catboost  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import STATION_ORDER
from src.models.priors_and_linear import CalibratedDiurnalCAMSPriorForecaster

# Coordinates of the 4 Mumbai target stations in `STATION_ORDER` = [11606, 11611, 3409510, 3409511]
STATION_COORDS: dict[int, tuple[float, float]] = {
    11606: (19.23241, 72.86895),    # Borivali East
    11611: (19.19709, 72.82204),    # Malad West
    3409510: (18.97670, 72.83800),  # Byculla
    3409511: (19.060498, 72.923356),  # Shivaji Nagar
}

ST_GAT_NODE_FEATURES: list[str] = [
    "prior_eb_diurnal",
    "prior_diurnal_adaptive",
    "prior_physics_calibrated",
    "cams_pm25_calibrated",
    "cams_pm25_offset_calibrated",
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
    "log1p_vc",
    "log_pblh",
    "u10",
    "v10",
    "onshore_breeze",
    "deonar_wind_cos",
    "thane_creek_wind_cos",
    "dewpoint_depression",
    "stagnation_hours",
    "station_trend_3d_vs_14d",
    "station_to_city_ratio_t0",
    "hour_sin_1",
    "hour_cos_1",
    "hour_sin_2",
    "hour_cos_2",
    "log1p_horizon",
]


def _build_static_station_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise Haversine distance matrix (km) and unit direction vectors (dx_east, dy_north)
    from station j to station i across the 4 Mumbai stations.
    """
    n = len(STATION_ORDER)
    dist_km = np.zeros((n, n), dtype=np.float32)
    dir_east = np.zeros((n, n), dtype=np.float32)
    dir_north = np.zeros((n, n), dtype=np.float32)

    for i, loc_i in enumerate(STATION_ORDER):
        lat_i, lon_i = STATION_COORDS[loc_i]
        for j, loc_j in enumerate(STATION_ORDER):
            if i == j:
                continue
            lat_j, lon_j = STATION_COORDS[loc_j]
            dy_km = (lat_i - lat_j) * 111.0
            dx_km = (lon_i - lon_j) * 111.0 * np.cos(np.deg2rad(0.5 * (lat_i + lat_j)))
            d = float(np.hypot(dx_km, dy_km))
            dist_km[i, j] = d
            dir_east[i, j] = dx_km / max(d, 1e-3)
            dir_north[i, j] = dy_km / max(d, 1e-3)

    return dist_km, dir_east, dir_north


class WindGuidedGATNet(nn.Module):
    """
    PyTorch Wind-Guided Spatiotemporal Graph Attention Network.
    Input per timestamp batch B:
      - `x`: (B, 4, F) standardized node features for the 4 stations
      - `u_wind`: (B, 4) eastward wind velocity component (m/s or km/h)
      - `v_wind`: (B, 4) northward wind velocity component (m/s or km/h)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 48,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        dist_km, dir_east, dir_north = _build_static_station_graph()
        self.register_buffer("dist_km", torch.tensor(dist_km, dtype=torch.float32))
        self.register_buffer("dir_east", torch.tensor(dir_east, dtype=torch.float32))
        self.register_buffer("dir_north", torch.tensor(dir_north, dtype=torch.float32))

        self.station_emb = nn.Embedding(len(STATION_ORDER), 8)
        self.node_enc = nn.Sequential(
            nn.Linear(in_features + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.wind_gain = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))
        self.dist_scale = nn.Parameter(torch.tensor(12.0, dtype=torch.float32))

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        u_wind: torch.Tensor,
        v_wind: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, F), N=4
        bsz, n_nodes, _ = x.shape
        st_ids = torch.arange(n_nodes, device=x.device).unsqueeze(0).expand(bsz, n_nodes)
        h_in = torch.cat([x, self.station_emb(st_ids)], dim=-1)
        h = self.node_enc(h_in)  # (B, N, H)

        q = self.q_proj(h)  # (B, N, H)
        k = self.k_proj(h)  # (B, N, H)
        v = self.v_proj(h)  # (B, N, H)

        # Content attention logits: (B, i, j)
        attn_logits = torch.einsum("bih,bjh->bij", q, k) / np.sqrt(h.shape[-1])

        # Wind-directed physical transport score from source station j to target station i:
        # dot((u_j, v_j), dir_{j -> i})
        u_src = u_wind.unsqueeze(1).expand(bsz, n_nodes, n_nodes)  # (B, i, j)
        v_src = v_wind.unsqueeze(1).expand(bsz, n_nodes, n_nodes)  # (B, i, j)
        wind_proj = u_src * self.dir_east.unsqueeze(0) + v_src * self.dir_north.unsqueeze(0)
        wind_advection = torch.relu(wind_proj)

        dist_penalty = -self.dist_km.unsqueeze(0) / torch.clamp(self.dist_scale, min=3.0)
        phys_logits = attn_logits + dist_penalty + self.wind_gain * wind_advection

        attn_weights = torch.softmax(phys_logits, dim=-1)  # (B, i, j)
        h_graph = torch.einsum("bij,bjh->bih", attn_weights, v)  # (B, N, H)

        combined = torch.cat([h, h_graph], dim=-1)  # (B, N, 2H)
        out = self.out_head(combined).squeeze(-1)  # (B, N)
        return out


@dataclass
class WindGuidedSTGATForecaster:
    """
    Model 6: Wind-Guided Spatiotemporal Graph Attention Network Forecaster (PyTorch).
    Groups observations by `(cutoff_t0, target_start_ist)` across the 4 Mumbai stations
    so the graph layer processes all 4 stations simultaneously with dynamic wind-advection edges.
    """

    hidden_dim: int = 36
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.5e-2
    residual_damping: float = 0.42
    random_state: int = 42
    prior_model: CalibratedDiurnalCAMSPriorForecaster = field(
        default_factory=CalibratedDiurnalCAMSPriorForecaster
    )
    scaler_: RobustScaler | None = field(default=None, init=False)
    net_: WindGuidedGATNet | None = field(default=None, init=False)
    feature_cols_: list[str] = field(default=list, init=False)

    def _build_graph_tensors(
        self,
        df: pd.DataFrame,
        y_res: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[tuple]]:
        """
        Reshape tabular station-hour dataframe into `(B, 4, F)` graph snapshots
        indexed by `(cutoff_key, target_start_ist)`.
        """
        work = df.copy()
        if "cutoff_t0" not in work.columns:
            work["cutoff_t0"] = pd.Timestamp("2026-10-22 00:00:00", tz="Asia/Kolkata")
        work["_row_id"] = np.arange(len(work))
        if y_res is not None:
            work["_y_res"] = y_res
        if sample_weight is not None:
            work["_sw"] = sample_weight
        else:
            work["_sw"] = 1.0

        X_scaled = np.clip(
            self.scaler_.transform(work[self.feature_cols_].values).astype(np.float32),
            -4.0,
            4.0,
        )

        # Fast vectorized integer factorize on (cutoff_t0, target_start_ist)
        work["node_idx"] = work["location_id"].map({loc: i for i, loc in enumerate(STATION_ORDER)})
        snap_idx, uniques = pd.factorize(
            pd.MultiIndex.from_arrays([work["cutoff_t0"], work["target_start_ist"]])
        )
        n_snaps = len(uniques)
        n_nodes = len(STATION_ORDER)
        n_feats = len(self.feature_cols_)

        X_tensor = np.zeros((n_snaps, n_nodes, n_feats), dtype=np.float32)
        U_tensor = np.zeros((n_snaps, n_nodes), dtype=np.float32)
        V_tensor = np.zeros((n_snaps, n_nodes), dtype=np.float32)
        Y_tensor = np.zeros((n_snaps, n_nodes), dtype=np.float32) if y_res is not None else None
        M_tensor = np.zeros((n_snaps, n_nodes), dtype=np.float32)

        node_indices = work["node_idx"].values
        X_tensor[snap_idx, node_indices, :] = X_scaled
        U_tensor[snap_idx, node_indices] = work["u10"].values.astype(np.float32)
        V_tensor[snap_idx, node_indices] = work["v10"].values.astype(np.float32)
        if y_res is not None:
            Y_tensor[snap_idx, node_indices] = work["_y_res"].values.astype(np.float32)
        M_tensor[snap_idx, node_indices] = work["_sw"].values.astype(np.float32)

        # Fill any missing station in a partial snapshot with the mean of present stations in that snapshot
        present_mask = M_tensor > 0
        for s_i in range(n_snaps):
            if not np.all(present_mask[s_i]) and np.any(present_mask[s_i]):
                mean_feat = X_tensor[s_i, present_mask[s_i]].mean(axis=0)
                X_tensor[s_i, ~present_mask[s_i]] = mean_feat

        row_lookup = list(zip(snap_idx.tolist(), node_indices.tolist()))
        return X_tensor, U_tensor, V_tensor, Y_tensor, M_tensor, row_lookup

    def fit(
        self,
        train_feat_df: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "WindGuidedSTGATForecaster":
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        target = (
            np.asarray(y, dtype=float)
            if y is not None
            else train_feat_df["pm25_clean"].values
        )
        self.prior_model.fit(train_feat_df, target)
        y_prior = self.prior_model.predict(train_feat_df)
        residual = np.clip(target - y_prior, -60.0, 60.0)

        self.feature_cols_ = [c for c in ST_GAT_NODE_FEATURES if c in train_feat_df.columns]
        self.scaler_ = RobustScaler()
        self.scaler_.fit(train_feat_df[self.feature_cols_].values)

        sw = (
            sample_weight
            if sample_weight is not None
            else (
                train_feat_df["sample_weight"].values
                if "sample_weight" in train_feat_df.columns
                else np.ones(len(train_feat_df))
            )
        )

        X_np, U_np, V_np, Y_np, M_np, _ = self._build_graph_tensors(
            train_feat_df, y_res=residual, sample_weight=sw
        )

        self.net_ = WindGuidedGATNet(
            in_features=len(self.feature_cols_),
            hidden_dim=self.hidden_dim,
            dropout=0.20,
        )
        optimizer = torch.optim.AdamW(
            self.net_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        huber_loss = nn.HuberLoss(reduction="none", delta=12.0)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        U_t = torch.tensor(U_np, dtype=torch.float32)
        V_t = torch.tensor(V_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        M_t = torch.tensor(M_np, dtype=torch.float32)

        n_samples = X_t.shape[0]
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                pred_res = self.net_(X_t[idx], U_t[idx], V_t[idx])
                mse_term = (pred_res - Y_t[idx]) ** 2
                hub_term = huber_loss(pred_res, Y_t[idx]) * 10.0
                loss_mat = 0.55 * mse_term + 0.45 * hub_term
                weighted_loss = (loss_mat * M_t[idx]).sum() / torch.clamp(M_t[idx].sum(), min=1.0)
                weighted_loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=2.0)
                optimizer.step()

        self.net_.eval()
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        y_prior = self.prior_model.predict(feat_df)
        if self.net_ is None or self.scaler_ is None:
            return y_prior

        X_np, U_np, V_np, _, _, row_lookup = self._build_graph_tensors(feat_df)
        self.net_.eval()
        with torch.no_grad():
            pred_mat = (
                self.net_(
                    torch.tensor(X_np, dtype=torch.float32),
                    torch.tensor(U_np, dtype=torch.float32),
                    torch.tensor(V_np, dtype=torch.float32),
                )
                .cpu()
                .numpy()
            )

        res_flat = np.array([pred_mat[s_i, n_i] for s_i, n_i in row_lookup], dtype=float)
        res_damped = np.clip(res_flat * self.residual_damping, -18.0, 18.0)
        return np.asarray(np.clip(y_prior + res_damped, 2.0, 450.0), dtype=float)
