# Mathematical Foundations — Spatiemporal Physics, Foundation Models & Optimal Ensembling

**Reference Manual for Engineers, Researchers, and AI Coding Agents**  
**Project:** Google Mumbai Air Quality Forecasting Challenge & VayuDrishti Mumbai (`2026`)

---

## 1. Bayes-Optimal Point Estimation Under RMSE vs. MAE & Trapezoidal Quantile Quadrature

### 1.1 Why Median Forecasts Are Suboptimal Under RMSE on Right-Skewed $\text{PM}_{2.5}$
Let $Y_{s,t} \in \mathbb{R}_+$ denote the ground-truth hourly $\text{PM}_{2.5}$ concentration at station $s$ and hour $t$, and let $F_{Y \mid \mathcal{I}_{T_0}}(y)$ denote its conditional cumulative distribution function given information set $\mathcal{I}_{T_0}$ available at cutoff $T_0$.
- Under **Mean Absolute Error (MAE)** loss $\mathcal{L}_1(y, \hat{y}) = |y - \hat{y}|$, the Bayes risk $\mathbb{E}[|Y - \hat{y}| \mid \mathcal{I}_{T_0}]$ is minimized by the **conditional median**:
  $$\hat{y}_{\text{L1}}^* = F_{Y \mid \mathcal{I}_{T_0}}^{-1}(0.5) = q_{0.50}$$
- Under the competition's evaluation metric, **Root Mean Squared Error (RMSE)**, minimizing $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$ is strictly equivalent to minimizing expected squared error $\mathcal{L}_2(y, \hat{y}) = (y - \hat{y})^2$. Setting the Fréchet derivative $\frac{\partial}{\partial \hat{y}} \mathbb{E}[(Y - \hat{y})^2 \mid \mathcal{I}_{T_0}] = -2(\mathbb{E}[Y \mid \mathcal{I}_{T_0}] - \hat{y}) = 0$ yields the unique Bayes-optimal point estimator:
  $$\hat{y}_{\text{RMSE}}^* = \mathbb{E}[Y \mid \mathcal{I}_{T_0}] = \int_0^1 F_{Y \mid \mathcal{I}_{T_0}}^{-1}(\tau)\,d\tau$$

Because urban $\text{PM}_{2.5}$ concentrations are approximately log-normally distributed (strictly right-skewed during nocturnal boundary-layer inversions), if $\ln Y \sim \mathcal{N}(\mu_\ell, \sigma_\ell^2)$, then:
$$q_{0.50} = \exp(\mu_\ell) < \mathbb{E}[Y] = \exp\left(\mu_\ell + \frac{1}{2}\sigma_\ell^2\right)$$
Using the default `median` point forecast (`q=0.5`) from pretrained quantile foundation models (`TimesFM 3.0` or `Chronos-2`) introduces a systematic downward bias of $\mathbb{E}[Y] - q_{0.50} = \mathbb{E}[Y]\big(1 - \exp(-\sigma_\ell^2/2)\big) \approx 3.5\text{--}7.0\;\mu\text{g/m}^3$ during post-monsoon smog episodes.

### 1.2 Trapezoidal Numerical Quadrature Over Predicted Quantiles
Given a strictly increasing grid of $K$ quantile levels $0 < \tau_1 < \tau_2 < \dots < \tau_K < 1$ with corresponding predicted quantiles $\hat{q}_{\tau_1}, \dots, \hat{q}_{\tau_K}$, we approximate the Riemann–Stieltjes integral $\mathbb{E}[Y \mid \mathcal{I}_{T_0}] = \int_0^1 F^{-1}(\tau)\,d\tau$ using piecewise-uniform trapezoidal probability masses with boundary extensions $\tau_0 = 0$ and $\tau_{K+1} = 1$:
$$w_k = \frac{\tau_{k+1} - \tau_{k-1}}{2}, \qquad k = 1, \dots, K, \qquad \sum_{k=1}^K w_k = \frac{\tau_2 - 0 + \tau_3 - \tau_1 + \dots + 1 - \tau_{K-1}}{2} = \frac{1 + \tau_K - \tau_1}{2}$$
Normalizing by $\sum_{j=1}^K w_j$ yields the exact trapezoidal expectation implemented in `Chronos2Pipeline._get_prob_mass_per_quantile_level` (`external_repos/chronos-forecasting/src/chronos/chronos2/pipeline.py:51-77`):
$$\hat{y}_{\text{RMSE}} = \sum_{k=1}^K \tilde{w}_k \hat{q}_{\tau_k}, \qquad \tilde{w}_k = \frac{\tau_{k+1} - \tau_{k-1}}{2 \sum_{j=1}^K (\tau_{j+1} - \tau_{j-1})/2}$$
- **For `TimesFM 3.0` ($K=9$ deciles, $\tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]$):**
  Assigning tail masses $[0, 0.15]$ to $\tau_1=0.1$ and $[0.85, 1.0]$ to $\tau_9=0.9$ yields exact normalized weights:
  $$\tilde{\mathbf{w}}_{\text{TimesFM3}} = [0.15,\, 0.10,\, 0.10,\, 0.10,\, 0.10,\, 0.10,\, 0.10,\, 0.10,\, 0.15]^\top$$
- **For `Chronos-2` ($K=21$ quantiles, $\tau = [0.01, 0.05, 0.10, 0.15, \dots, 0.90, 0.95, 0.99]$):**
  Applying `boundaries = [0.0, 0.01, 0.05, ..., 0.95, 0.99, 1.0]` and $\tilde{w}_k \propto (\tau_{k+1} - \tau_{k-1})/2$ integrates the full 1st-to-99th percentile distribution.
- **Distributional Meta-Features for Downstream GBDTs (`Models 3–5`):**
  From the quantile vector $\hat{\mathbf{q}}$, we also extract three scale- and shape-aware summary statistics:
  $$\text{Mean}_{s,t} = \sum_{k=1}^K \tilde{w}_k \hat{q}_{\tau_k}, \qquad \text{IQR}_{s,t} = \hat{q}_{0.80} - \hat{q}_{0.20}, \qquad \text{Skew}_{s,t} = \frac{\hat{q}_{0.90} - \hat{q}_{0.50}}{\max(\hat{q}_{0.50} - \hat{q}_{0.10}, 1.0)}$$

---

## 2. Google Research `TimesFM 3.0` (`330.71M`) Mathematical Mechanics

All equations below correspond directly to the reference implementation in `external_repos/timesfm/src/timesfm3/torch/`.

### 2.1 Patch Tokenization, Causal Running RevIN, and Future Covariate Roll (`src/timesfm3/torch/model.py:191-263`)
Let the input tensor have shape $(B, V, N, P)$, where $B$ is batch size, $V$ is the number of variates, $N$ is the number of patches, and $P = 32$ (`input_patch_len`) is the patch length, with boolean mask $M \in \{0, 1\}^{B \times V \times N \times P}$ (`True` = masked) and target indicator $I_{\text{target}} \in \{0, 1\}^{B \times V \times N}$.
1. **Causal Patch-by-Patch Running Statistics (`util.get_running_stats`, `src/timesfm3/torch/util.py:163-224`):**
   For each variate $v$ and patch $n \in \{0, \dots, N-1\}$, let $\Omega_{v,n} = \{p \in \{0,\dots,P-1\} : \neg M_{b,v,n,p}\}$ be the set of valid time steps in patch $n$ with count $c_{v,n} = |\Omega_{v,n}|$, patch mean $\bar{x}_{v,n}$, and patch variance $s_{v,n}^2$. Welford's parallel running update accumulates causal statistics $(\eta_{v,n}, \mu_{v,n}, \sigma_{v,n})$ over patches $0 \dots n$:
   $$\eta_{v,n} = \eta_{v,n-1} + c_{v,n}, \qquad \mu_{v,n} = \frac{\eta_{v,n-1}\mu_{v,n-1} + c_{v,n}\bar{x}_{v,n}}{\eta_{v,n}}$$
   $$\sigma_{v,n} = \sqrt{\frac{\eta_{v,n-1}\big(\sigma_{v,n-1}^2 + (\mu_{v,n-1} - \mu_{v,n})^2\big) + c_{v,n}\big(s_{v,n}^2 + (\bar{x}_{v,n} - \mu_{v,n})^2\big)}{\eta_{v,n}}}$$
2. **Reversible Instance Normalization (`util.revin`, `src/timesfm3/torch/util.py:227-257`):**
   $$\tilde{x}_{b,v,n,p} = \begin{cases} \dfrac{x_{b,v,n,p} - \mu_{b,v,n}}{\max(\sigma_{b,v,n}, 10^{-6})} & \text{if } \neg M_{b,v,n,p} \\ 0 & \text{if } M_{b,v,n,p} \end{cases}$$
3. **Future Covariate Lookahead Roll (`util.get_output_patch_via_roll`, `src/timesfm3/torch/util.py:260-296`):**
   With `output_patch_len` $O = 64$ and $R = O / P = 2$ rolls, `get_output_patch_via_roll` shifts the patch tensor leftward along the patch axis by $r \in \{1, 2\}$ patches to construct the $O=64$-step lookahead tensor $x^{\text{fcov}}_{b,v,n,1:64} = [x_{b,v,n+1,:} \,\|\, x_{b,v,n+2,:}]$, normalized by the current patch's running statistics $(\mu_{b,v,n}, \sigma_{b,v,n})$. Crucially, its mask is:
   $$M^{\text{fcov}}_{b,v,n,:} = M^{\text{roll}}_{b,v,n,:} \lor I_{\text{target}}(b,v,n) \lor M^{\text{wrap}}_{n,:}$$
   Thus, **for target variates ($I_{\text{target}} = \text{True}$), future values are strictly zeroed out**, whereas **for known future covariates ($I_{\text{target}} = \text{False}$, such as calibrated CAMS $\text{PM}_{2.5}$ and Open-Meteo `log1p_vc`), the model directly receives the next 64 hours of normalized covariate trajectory inside patch $n$'s token!**
4. **192-Dimensional `ResidualBlock` Tokenizer (`src/timesfm3/torch/model.py:146-149, 248-252`):**
   The input to `pre_transformer_resblock` at each patch $(b, v, n)$ concatenates four vectors of lengths $32, 64, 32, 64$:
   $$\mathbf{z}_{b,v,n}^{\text{in}} = \big[\tilde{\mathbf{x}}_{b,v,n} \in \mathbb{R}^{32} \;\big\|\; \tilde{\mathbf{x}}^{\text{fcov}}_{b,v,n} \in \mathbb{R}^{64} \;\big\|\; \mathbf{M}_{b,v,n} \in \{0,1\}^{32} \;\big\|\; \mathbf{M}^{\text{fcov}}_{b,v,n} \in \{0,1\}^{64}\big] \in \mathbb{R}^{192}$$
   mapping $\mathbb{R}^{192} \to \mathbb{R}^{1280}$ via a two-layer ReLU residual projection (`hidden_dims=1280, output_dims=1280`).

### 2.2 Alternating `SequenceAttention` and `VariateAttention` (`src/timesfm3/torch/transformer.py`)
Each of the $L = 20$ `MixingTransformer` layers applies:
1. **Causal Sequence Self-Attention with Rotary Position Embeddings (`use_rope_seq=True`):**
   Operating along the patch sequence dimension $n \in \{0, \dots, N-1\}$ independently per variate $v$ with RMSNorm on Queries/Keys (`qk_norm="rms"`), $H = 16$ heads ($d_h = 80$), and causal mask $n_{\text{query}} \ge n_{\text{kv}} \ge n_{\text{front\_masked}}$:
   $$\mathbf{H}^{(l,\text{seq})}_{b,v,:,:} = \mathbf{H}^{(l-1)}_{b,v,:,:} + \text{MHA}_{\text{seq}}\big(\text{RMSNorm}(\mathbf{H}^{(l-1)}_{b,v,:,:});\, \text{RoPE},\, \text{causal}=\text{True}\big)$$
2. **Permutation-Equivariant Cross-Variate Self-Attention (`use_rope_var=False`):**
   Operating along the variate dimension $v \in \{0, \dots, V-1\}$ at each fixed patch index $n$ with bidirectional (non-causal) attention:
   $$\mathbf{H}^{(l,\text{var})}_{b,:,n,:} = \mathbf{H}^{(l,\text{seq})}_{b,:,n,:} + \text{MHA}_{\text{var}}\big(\text{RMSNorm}(\mathbf{H}^{(l,\text{seq})}_{b,:,n,:});\, \text{RoPE}=\text{False},\, \text{causal}=\text{False}\big)$$
   **Proof of Softmax Dilution at Large $V$:** In $\text{MHA}_{\text{var}}$, the attention weights for target variate $i \in \{1..4\}$ over all $V$ variates at patch $n$ are:
   $$\alpha_{i \to j}^{(n)} = \frac{\exp(\mathbf{q}_{i,n}^\top \mathbf{k}_{j,n} / \sqrt{d_h})}{\sum_{v=1}^V \exp(\mathbf{q}_{i,n}^\top \mathbf{k}_{v,n} / \sqrt{d_h})}$$
   Because `use_rope_var=False` and no static variate-type embedding exists prior to attention, adding $V_{\text{noise}}$ weakly correlated exogenous covariates inflates the denominator by $\sum_{v \in \mathcal{V}_{\text{noise}}} \exp(\mathbf{q}_{i,n}^\top \mathbf{k}_{v,n} / \sqrt{d_h})$, shrinking the cross-station target attention mass $\sum_{j=1}^4 \alpha_{i \to j}^{(n)}$ from $1.0$ (in our `V=4` view) and $\approx 0.42$ (in our `V=12` physics-coupled view) down to $\approx 0.14$ in a 28-variate configuration.

### 2.3 Iterative CPM RevIN Refinement (`cpm_iterative_revin_refine`, `src/timesfm3/torch/cpm_revin_refine.py:27-149`)
When forecasting multi-patch horizons using Continuous Patch Masking (`patch_cpm_mask`), freezing the running normalization statistics $(\mu_{v,n_0}, \sigma_{v,n_0})$ at the last observed context patch $n_0$ causes scale drift if the predicted series trends upward during post-monsoon build-up. `cpm_iterative_revin_refine` iteratively updates $(\tilde{\eta}_{v,n}, \tilde{\mu}_{v,n}, \tilde{\sigma}_{v,n})$ at each horizon patch $n > n_0$ using the reverse-RevIN median prediction ($\tau = 0.5$) of the preceding anchor patch before un-normalizing all 9 quantile heads.

### 2.4 Conditional OLS Linear Detrending & Linear Overlap Patch Stitching
- **Conditional OLS Linear Detrending (`timesfm3_forecaster.py`, `linear_detrending_threshold = 0.5`):**
  Fits an Ordinary Least Squares line $\hat{\ell}_v(t) = \hat{a}_v t + \hat{b}_v$ on the unmasked context of variate $v$. Let $\sigma_{\text{orig}, v} = \text{std}(x_v)$ and $\sigma_{\text{det}, v} = \text{std}(x_v - \hat{\ell}_v)$. Detrending is applied **if and only if** the linear trend accounts for $>75\%$ of the variance:
  $$\frac{\sigma_{\text{det}, v}}{\sigma_{\text{orig}, v}} < 0.50 \iff R^2 = 1 - \left(\frac{\sigma_{\text{det}, v}}{\sigma_{\text{orig}, v}}\right)^2 > 0.75$$
- **Linear Overlap Patch Stitching (`util.stitch_patches`, `src/timesfm3/torch/util.py:337-387`):**
  Because each patch step advances by $P = 32$ steps (`stride=32`) while emitting $O = 64$ steps, consecutive patch predictions $\hat{\mathbf{Y}}^{(m)}$ and $\hat{\mathbf{Y}}^{(m+1)}$ overlap over $\Delta = O - P = 32$ hours. `stitch_patches` blends the overlapping region with linear weights $w_r = 1 - \frac{r}{\Delta - 1}$ for $r \in \{0, \dots, 31\}$:
  $$\hat{Y}_{\text{stitched}}(mP + P + r, q) = w_r \hat{Y}^{(m)}(P + r, q) + (1 - w_r) \hat{Y}^{(m+1)}(r, q)$$
  eliminating 32-hour block boundary jumps.

---

## 3. `TimeMixer` Multiscale Synoptic-Diurnal Decomposition & $\tanh$-Saturating CAMS MOS

### 3.1 Two-Scale Frequency Decomposition (`TimeMixer`, Wang et al., ICLR 2024)
Following `TimeMixer` (`external_repos/TimeMixer`), an atmospheric time series $x_{s,t}$ is decomposed via a causal 24-hour boxcar low-pass filter into a **synoptic trend component** ($f < \frac{1}{24}\text{h}^{-1}$) and a **diurnal seasonal component** ($f \ge \frac{1}{24}\text{h}^{-1}$):
$$x_{s,t}^{\text{syn}} = \frac{1}{24}\sum_{\ell=0}^{23} x_{s, t-\ell}, \qquad x_{s,t}^{\text{diu}} = x_{s,t} - x_{s,t}^{\text{syn}}$$

### 3.2 Empirical Bayes James-Stein Diurnal Shrinkage & $\tanh$-Saturating CAMS MOS
At forecast cutoff $T_0$, let $\bar{y}_{s,h}^{(W)}$ denote the exponentially decayed (half-life $\tau_{1/2} = 120\text{h}$) station-hour mean over window $W \in \{7\text{d}, 10\text{d}, 14\text{d}, 28\text{d}\}$ for observations with `hour_end <= T_0`, and let $\bar{y}_s^{(7\text{d})}$ denote the overall station mean. The **James-Stein Empirical Bayes prior** shrinks noisy single-hour estimates toward the station mean:
$$\hat{y}_{\text{EB}}(s, h) = \alpha_{\text{EB}} \bar{y}_{s,h}^{(10\text{d})} + (1 - \alpha_{\text{EB}}) \bar{y}_s^{(7\text{d})}, \qquad \alpha_{\text{EB}} = 0.65$$
Combined with the fast-adapting 18-hour half-life station level $\bar{y}_s^{\text{fast}}(T_0)$:
$$\hat{y}_{\text{adapt}}(s, h) = \big(0.60\bar{y}_s^{\text{fast}}(T_0) + 0.40\bar{y}_s^{(7\text{d})}\big) + 0.65\big(\bar{y}_{s,h}^{(7\text{d})} - \bar{y}_s^{(7\text{d})}\big)$$
$$\hat{y}_{\text{base}}(s, h) = 0.55\hat{y}_{\text{EB}}(s, h) + 0.45\hat{y}_{\text{adapt}}(s, h)$$

To incorporate the future CAMS chemical forecast $C(s, t)$ without vulnerability to regional CAMS dust false alarms (such as Fold 4's `102 µg/m³` spike), let $\bar{C}_{T_0}(s, h) = 0.65\bar{C}_{s,h}^{(7\text{d})}(T_0) + 0.35\bar{C}_s^{(7\text{d})}(T_0)$ be CAMS's own trailing baseline at $T_0$. The **$\tanh$-Saturating Model Output Statistics (MOS)** scaling factor is:
$$\rho_{\text{CAMS}}(s, t) = \exp\left( \kappa \tanh\left(\frac{1}{\kappa}\ln\frac{C(s, t)}{\max(\bar{C}_{T_0}(s, h), 10.0)}\right) \right), \qquad \kappa = 0.45$$
When $C(s,t) \approx \bar{C}_{T_0}(s,h)$, $\tanh(u/\kappa) \approx u/\kappa$ so $\rho_{\text{CAMS}}(s,t) \approx C(s,t)/\bar{C}_{T_0}(s,h)$ (linear sensitivity to moderate synoptic changes). When CAMS suffers a $3\times$ dust false alarm ($\ln 3 \approx 1.10$), $\kappa \tanh(1.10/0.45) \approx 0.443$, smoothly bounding the maximum multiplicative excursion to $\exp(0.45) \approx 1.568$! Coupled with boundary-layer ventilation anomaly $\Delta \text{VC}_{s,t} = \ln(1 + \text{VC}_{s,t}) - \overline{\ln(1 + \text{VC})}_{s,h}^{(7\text{d})}(T_0)$:
$$\hat{y}_{\text{M1}}(s, t) = (1 - \lambda_c)\,\hat{y}_{\text{base}}(s, h)\,\big(\rho_{\text{CAMS}}(s,t)\big)^{\beta_c} \exp(-\gamma_v \Delta \text{VC}_{s,t}) + \lambda_c C_{\text{cal}}(s, t)$$

---

## 4. `AirPhyNet` & `PM2.5-GNN` Atmospheric Advection-Diffusion PDE on Mumbai's Coastal Graph

### 4.1 Governing Advection-Diffusion-Reaction PDE (`AirPhyNet`, Hettige et al., ICLR 2024)
The spatiotemporal evolution of particulate concentration $C(\mathbf{x}, t)$ in the planetary boundary layer satisfies the conservation PDE (`external_repos/AirPhyNet`):
$$\frac{\partial C}{\partial t} = -\underbrace{(\mathbf{v} \cdot \nabla) C}_{\text{Horizontal Wind Advection}} + \underbrace{\nabla \cdot (D \nabla C)}_{\text{Turbulent Diffusion}} - \underbrace{\frac{w_e}{\text{PBLH}} C}_{\text{Vertical Dilution / Scavenging}} + \underbrace{S(\mathbf{x}, t)}_{\text{Local Emissions}}$$
where $\mathbf{v}(t) = (u_{10}(t), v_{10}(t))$ is the horizontal wind velocity vector (direction air is blowing **toward**):
$$u_{10} = -w_{10\text{m}} \sin\theta_{\text{from}}, \qquad v_{10} = -w_{10\text{m}} \cos\theta_{\text{from}}$$

### 4.2 Coastal Land-Sea Breeze & Point-Source Plume Projections (`src/features.py:114-135`)
1. **Mumbai Coastline Normal Projection ($15^\circ$ NNE–SSW Coastline):**
   Mumbai's western Arabian Sea coastline is oriented along azimuth $\approx 15^\circ$ NNE–SSW. The onshore normal corresponds to wind blowing **from** $\theta_{\text{coast}} = 285^\circ$ (WNW). The onshore sea-breeze velocity component is:
   $$v_{\text{onshore}}(t) = w_{10\text{m}}(t) \cos\big(\theta_{\text{from}}(t) - 285^\circ\big)$$
   When $v_{\text{onshore}}(t) > 0$ (afternoon sea breeze, `13:00–19:00 IST`), clean marine air flushes the urban canopy; when $v_{\text{onshore}}(t) < 0$ (nocturnal/morning land breeze, `00:00–10:00 IST`), continental drainage traps urban and industrial plumes against the coast.
2. **Deonar Landfill ($75^\circ$ ENE) & Thane Creek Industrial Belt ($110^\circ$ ESE) Transport Proxies:**
   The 132-hectare Deonar Dumping Ground (`doi:10.71646/MWQNR8`, `19.056°N, 72.928°E`) lies `0.8 km` ENE ($\approx 75^\circ$) of `Shivaji Nagar (3409511)`. The directional plume transport proxy is:
   $$\Phi_{\text{Deonar}}(t) = w_{10\text{m}}(t) \cdot \max\Big(0,\, \cos\big(\theta_{\text{from}}(t) - 75^\circ\big)\Big)$$
3. **Ventilation Coefficient ($\text{VC}$):**
   $$\text{VC}_{s,t} = \max(\text{PBLH}_{s,t}, 15\text{ m}) \times \max(w_{10\text{m}, s, t}, 0.1\text{ km/h}), \qquad \text{Stagnation Indicator } \mathbb{I}_{\text{stag}} = \mathbb{I}(\text{PBLH}_{s,t} < 350\text{ m} \land w_{10\text{m}, s, t} < 8.5\text{ km/h})$$

### 4.3 Graph Discretization Over the 4 Target + 14 Auxiliary Mumbai CAAQMS Stations (`src/models/st_gat.py`)
For any target station $i$ at $(\phi_i, \lambda_i)$ and source station $j$ at $(\phi_j, \lambda_j)$, the local tangent-plane displacement vector from $j \to i$ is:
$$\Delta x_{j \to i} = 111.320 \cos\left(\frac{\phi_i + \phi_j}{2}\right)(\lambda_i - \lambda_j), \qquad \Delta y_{j \to i} = 110.574(\phi_i - \phi_j), \qquad d_{ij} = \sqrt{\Delta x_{j \to i}^2 + \Delta y_{j \to i}^2}$$
$$\hat{\mathbf{e}}_{j \to i} = \left(\frac{\Delta x_{j \to i}}{d_{ij}},\, \frac{\Delta y_{j \to i}}{d_{ij}}\right)$$
The discrete upwind advection operator (`WindGuidedGATNet.forward`, `src/models/st_gat.py:170-184`) weights neighbor features $\mathbf{v}_j$ by:
$$A_{j \to i}(t) = \text{softmax}_j \left( \frac{(\mathbf{W}_Q \mathbf{h}_i)^\top (\mathbf{W}_K \mathbf{h}_j)}{\sqrt{d_h}} - \frac{d_{ij}}{\ell_0} + \gamma_{\text{wind}} \,\text{ReLU}\big(u_{10,j}(t)\hat{e}_{j\to i, x} + v_{10,j}(t)\hat{e}_{j\to i, y}\big) \right)$$

---

## 5. Historical Meteorological Analog Ensemble (`AnEn`, Delle Monache et al.)

Given a target forecast hour $t$ at station $s$ with hour-of-day $h = t \bmod 24$, let $\mathbf{z}_{s,t} \in \mathbb{R}^D$ denote the standardized physical feature vector:
$$\mathbf{z}_{s,t} = \big[\ln(\text{PBLH}_{s,t}),\, \ln(1 + \text{VC}_{s,t}),\, u_{10,s,t},\, v_{10,s,t},\, \text{RH}_{s,t},\, T_{2\text{m},s,t},\, \rho_{\text{CAMS}}(s,t)\big]^\top$$
To capture temporal tendency (e.g., whether the boundary layer is collapsing or expanding), we compare the 5-hour trajectory window $[t-2, t-1, t, t+1, t+2]$ against all historical post-monsoon candidate timestamps $\tau \in \mathcal{H}_{s,h}(T_0)$ (restricted to the same station $s$, same hour of day $h$, months Oct–Feb in `2018–2025` with $\tau + 1\text{h} \le T_0$) using the weighted trajectory distance:
$$d_{\text{AnEn}}(t, \tau) = \left( \sum_{\nu=-2}^{2} \omega_\nu \big\| \mathbf{W}^{1/2} (\mathbf{z}_{s, t+\nu} - \mathbf{z}_{s, \tau+\nu}) \big\|_2^2 \right)^{1/2}, \qquad \boldsymbol{\omega} = [0.10, 0.20, 0.40, 0.20, 0.10]$$
Let $\{\tau_{(1)}, \dots, \tau_{(k)}\}$ be the $k = 15$ nearest historical analogs minimizing $d_{\text{AnEn}}(t, \tau)$. To account for secular emission changes between historical years and cutoff $T_0$, each analog observation $y_{s, \tau_{(j)}}$ is ratio-calibrated by the trailing 7-day station level at $T_0$ relative to $\tau_{(j)}$:
$$\hat{y}_{\text{AnEn}}(s, t) = \sum_{j=1}^k \pi_j \, y_{s, \tau_{(j)}} \cdot \text{clip}\left(\frac{\bar{y}_s^{(7\text{d})}(T_0)}{\bar{y}_s^{(7\text{d})}(\tau_{(j)})},\, 0.55,\, 1.80\right), \qquad \pi_j = \frac{\exp(-d_{\text{AnEn}}(t, \tau_{(j)})^2 / 2\sigma_d^2)}{\sum_{m=1}^k \exp(-d_{\text{AnEn}}(t, \tau_{(m)})^2 / 2\sigma_d^2)}$$

---

## 6. Duan's Smearing Estimator & Optimal L2 Variance Shrinkage Calibration

### 6.1 Duan's (1983) Nonparametric Smearing Estimator for Log-Ratio Regression
Suppose a model predicts the log-ratio $u_{s,t} = \ln(y_{s,t} / \mu_{0,s,t}) = \mathbf{x}_{s,t}^\top \boldsymbol{\beta} + \varepsilon_{s,t}$ with zero-mean error $\mathbb{E}[\varepsilon_{s,t}] = 0$. Because the exponential function is strictly convex, Jensen's inequality guarantees $\mathbb{E}[y_{s,t} \mid \mathbf{x}_{s,t}] = \mu_{0,s,t}\exp(\mathbf{x}_{s,t}^\top \boldsymbol{\beta})\,\mathbb{E}[\exp(\varepsilon_{s,t})] > \mu_{0,s,t}\exp(\mathbf{x}_{s,t}^\top \boldsymbol{\beta})$. Without assuming Gaussianity of $\varepsilon$, **Duan's (1983) Smearing Estimator** consistently estimates the re-transformation factor using the empirical OOF residuals $\{\hat{\varepsilon}_i = u_i - \hat{u}_i\}_{i=1}^n$:
$$\hat{\phi}_{\text{Duan}, s} = \frac{1}{n_s} \sum_{i \in \mathcal{I}_s} \exp(\hat{\varepsilon}_i) \implies \hat{y}_{\text{Duan}}(s, t) = \mu_{0,s,t} \exp(\hat{u}_{s,t}) \cdot \hat{\phi}_{\text{Duan}, s}$$

### 6.2 Theorem: Optimal Finite-Sample L2 Variance Shrinkage ($\alpha_s^*(h)$)
Let $\mu_0 = \mu_{0,s}(t)$ be the physics/diurnal anchor prior, let $\hat{y} = \sum_{m=1}^M w_m \hat{y}_m(s,t)$ be the stacked ensemble prediction, and let $\Delta y = y - \mu_0$ and $\Delta \hat{y} = \hat{y} - \mu_0$ denote innovations over the prior. Consider the affine shrinkage estimator:
$$\tilde{y}(\alpha, \delta) = \mu_0 + \alpha \Delta \hat{y} + \delta = \alpha \hat{y} + (1 - \alpha)\mu_0 + \delta$$
Minimizing expected squared error $\mathbb{E}\big[(y - \tilde{y}(\alpha, \delta))^2\big] = \mathbb{E}\big[(\Delta y - \alpha \Delta \hat{y} - \delta)^2\big]$ with respect to $(\alpha, \delta)$ yields the closed-form OLS optimal shrinkage coefficient:
$$\alpha^* = \frac{\text{Cov}(\Delta y,\, \Delta \hat{y})}{\text{Var}(\Delta \hat{y})} = \rho(\Delta y, \Delta \hat{y}) \cdot \frac{\sigma_{\Delta y}}{\sigma_{\Delta \hat{y}}}, \qquad \delta^* = \mathbb{E}[\Delta y] - \alpha^* \mathbb{E}[\Delta \hat{y}]$$
**Key Consequence for `Borivali East (11606)` and Long Lead Horizons ($h_{\text{lead}}$):**
1. At `Borivali East (11606)`, ground-truth innovation dispersion is small ($\sigma_{\Delta y} \approx 12.23\;\mu\text{g/m}^3$) while raw model innovations have $\sigma_{\Delta \hat{y}} \approx 15.86\;\mu\text{g/m}^3$ ($\sigma_{\Delta y} / \sigma_{\Delta \hat{y}} \approx 0.77$). Thus $\alpha_{\text{Borivali}}^* < 1$ is mathematically guaranteed even if $\rho$ is high!
2. As forecast lead time $h_{\text{lead}} = t - T_0$ increases from $24\text{h}$ to $120\text{h}$, the predictive correlation $\rho(\Delta y, \Delta \hat{y};\, h_{\text{lead}}) \approx \rho_0 \exp(-\gamma h_{\text{lead}})$ decays exponentially. Therefore, the Bayes-optimal shrinkage factor decays with horizon:
   $$\alpha_s^*(h_{\text{lead}}) = \alpha_{0,s} \exp(-\gamma_s h_{\text{lead}})$$
   reducing ensemble MSE by $\text{Var}(\Delta \hat{y})(1 - \alpha^*)^2 > 0$.

---

## 7. Split Conformal Prediction & Horizon-Scaled Intervals for Track 2 (`app/physics_engine.py`)

For Track 2 (`VayuDrishti Mumbai`), civic decision-makers (BMC Ward Officers, School Principals, Hospital Superintendents) require calibrated uncertainty bounds $[L_{1-\alpha}(s,t),\, U_{1-\alpha}(s,t)]$ satisfying marginal coverage $\mathbb{P}\big(Y_{s,t} \in [L_{1-\alpha}(s,t), U_{1-\alpha}(s,t)]\big) \ge 1 - \alpha$ at $1 - \alpha \in \{0.80, 0.95\}$.

Let $\hat{\sigma}_{s,t}(h_{\text{lead}})$ be the heteroscedastic physics-and-horizon local scale estimator (`app/physics_engine.py:471-486`):
$$\hat{\sigma}_{s,t}(h_{\text{lead}}) = \big(6.8 + 0.065\,\hat{y}_{s,t}\big) \cdot \left(1 + 0.28\sqrt{\frac{h_{\text{lead}}}{24}}\right) \cdot \begin{cases} 1.22 & \text{if } \text{PBLH}_{s,t} < 300\text{ m} \\ 1.00 & \text{otherwise} \end{cases}$$
On the out-of-fold calibration set $\mathcal{D}_{\text{cal}}$ ($N_{\text{cal}} = 960$), the standardized nonconformity scores are:
$$S_i = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}_i}, \qquad i \in \mathcal{D}_{\text{cal}}$$
Let $\hat{Q}_{1-\alpha}$ denote the $\lceil (N_{\text{cal}} + 1)(1 - \alpha) \rceil / N_{\text{cal}}$ empirical quantile of $\{S_i\}_{i=1}^{N_{\text{cal}}}$. By exchangeability over rolling post-monsoon blocks (Romano et al., 2019; Lei et al., 2018), the prediction interval:
$$\mathcal{C}_{1-\alpha}(s, t) = \Big[\max\big(3.0,\, \hat{y}_{s,t} - \hat{Q}_{1-\alpha}\hat{\sigma}_{s,t}(h_{\text{lead}})\big),\; \hat{y}_{s,t} + \hat{Q}_{1-\alpha}\hat{\sigma}_{s,t}(h_{\text{lead}})\Big]$$
provides finite-sample coverage while automatically widening during nocturnal boundary-layer inversions ($\text{PBLH} < 300\text{ m}$) and long lead horizons.

---

## 8. Primary Scientific Literature Citations

1. **TimesFM (Google Research):** Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only foundation model for time-series forecasting*. ICML 2024 (`external_repos/timesfm`).
2. **Chronos & Chronos-2 (Amazon Science):** Ansari, A. F., Stella, L., Turkmen, C., et al. (2024/2025). *Chronos: Learning the Language of Time Series* (TMLR) & *Chronos-2 Multivariate Group Self-Attention* (`external_repos/chronos-forecasting`).
3. **TimeMixer (ICLR 2024):** Wang, S., Wu, H., Shi, X., et al. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*. ICLR 2024 (`external_repos/TimeMixer`).
4. **AirPhyNet (ICLR 2024):** Hettige, K. H., Ji, J., Xiang, S., Long, C., Cong, G., & Wang, J. (2024). *AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction*. ICLR 2024 (`external_repos/AirPhyNet`).
5. **PM2.5-GNN:** Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). *PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting*. ACM SIGSPATIAL 2020 (`external_repos/PM2.5-GNN`).
6. **TabPFN v2 (Nature 2025):** Hollmann, N., Müller, S., Purucker, L., et al. (2025). *Accurate predictions on small data with a tabular foundation model*. Nature, 637, 319–326 (`external_repos/TabPFN`).
7. **Analog Ensemble (AnEn):** Delle Monache, L., Eckel, F. A., Rife, D. L., Nagarajan, B., & Searight, K. (2013). *Probabilistic Weather Prediction with an Analog Ensemble*. Monthly Weather Review, 141(10), 3498–3516.
8. **Duan's Smearing Estimator:** Duan, N. (1983). *Smearing Estimate: A Nonparametric Retransformation Method*. Journal of the American Statistical Association, 78(383), 605–610.
9. **Conformalized Quantile Regression (CQR):** Romano, Y., Patterson, E., & Candès, E. (2019). *Conformalized Quantile Regression*. NeurIPS 2019.
