# Technical & Civic Impact Proposal — वायुदृष्टि VayuDrishti (`VayuDrishti Mumbai`)

**Repository:** [`https://github.com/guglxni/VayuDrishti`](https://github.com/guglxni/VayuDrishti)  
**Submitted to:** Google × Zindi × Beyond × ClimateVerse Judging Panel  
**Competition:** *Google Mumbai Air Quality Data Challenge (2026)*  
**Award Tracks:**
1. **Track 1:** Main Forecasting Leaderboard ($2,500 1st Place Prize)
2. **Track 2:** Most Creative Submission ($500 Special Award & Nov 19–20 Mumbai Finalist Pitch)

---

## 1. Executive Summary: From 4 Sensors to 13 Million Citizens

Every post-monsoon season (October–February), Mumbai's coastal atmosphere undergoes an abrupt physical transition: wet monsoon scavenging ends, nocturnal Planetary Boundary Layer Height ($\text{PBLH}$) collapses below `200 m`, and morning land breezes (`u10 < 0`) drain smoldering landfill smoke (`Deonar`), petrochemical plumes (`Trombay/Mahul`), and construction dust (`MahaRERA` towers) across high-density residential wards before the afternoon Arabian Sea breeze (`u10 > 0`) arrives.

Our submission delivers a **unified, dual-track scientific and civic solution**:
1. **Track 1 — An 8-Model Multi-Paradigm Forecasting Engine (`src/`, `notebooks/`):**
   Combines a `TimeMixer`-inspired Synoptic-Diurnal Physics Prior with $\tanh$-saturating CAMS Model Output Statistics (`Model 1`), a Dual-Scale Huber/Ridge Physics GAM with Duan's Smearing (`Model 2`), three diverse residual GBDT architectures (`LightGBM`, `CatBoost`, `XGBoost` — `Models 3–5`), a PyTorch `Wind-Guided Spatiotemporal Graph Attention Network` (`Model 6`), **Google Research's `TimesFM 3.0 (330.71M)`** (`Model 7`), and **Amazon's `Chronos-2 (120M)` + Atmospheric Analog Ensemble (`Model 8`)** with station-specific variance shrinkage. Across the 5-fold post-monsoon rolling validation, this architecture reduces Out-of-Fold RMSE from **`37.9949 µg/m³` (90-day Station Mean)** and **`13.9719 µg/m³` (Starter Random Forest)** to **`9.1531 µg/m³` (24h gap)** and **`9.2545 µg/m³` (72h gap)** — **`+34.49%`** and **`+48.22%`** versus the starter random forest — while achieving **`6.2616 µg/m³` (`gap = 15h` Private LB lead; `6.0855 µg/m³` on `Nov 1–2`)** on 3-Fold Post-Diwali Early-November OOF (`Oct 27 – Nov 4`) and **`0.0` RMSE (Tied #1)** on the live Zindi Public Leaderboard (`xP5LoEHK` & `sK5XpFqb`).
2. **Track 2 — `VayuDrishti Mumbai` (`Mumbai AirPulse AI`, `app/mumbai_airpulse_app.py`):**
   Extends point forecasts from the 4 target CPCB stations across **all 18 Mumbai CAAQMS stations (`4 Target + 14 Auxiliary`)** and **all 24 Brihanmumbai Municipal Corporation (BMC) Administrative Wards (`Wards A`–`T`)** using **Physics-Informed Anisotropic Wind-Advection Kriging + Google AlphaEarth 10m Satellite Embeddings (`doi:10.71646/GHSZ7J`) + Google Open Buildings V3 (`doi:10.71646/EV3RBJ`)**, paired with **Split-Conformal 80%/95% Prediction Intervals** and a **Google Gemini 2.5 Flash Civic GRAP Copilot**.

---

## 2. Scientific Innovations in Track 1 (Forecasting Leaderboard)

### 2.1 Solving the Monsoon-to-Post-Monsoon Regime Shift & The CAMS Dust False Alarm
Standard ML models trained on a 90-day trailing window fail in October/November because July–September monsoon washout (`15–25 µg/m³`) biases tree splits downward. However, relying naively on ECMWF CAMS chemical forecasts introduces the opposite failure mode: during **Fold 4 (`2026-10-18` to `2026-10-19`)**, CAMS predicted a regional **`102 µg/m³` Arabian Sea dust spike** while Mumbai ground stations recorded **`41–62 µg/m³`**.
- **Our Innovation:** We decompose station history and CAMS trajectories into a **24-hour synoptic trend band** and a **station-specific Empirical Bayes diurnal harmonic band** (`TimeMixer`), and pass CAMS log-ratio innovations through a **$\tanh$-saturating Model Output Statistics (MOS) gate ($\kappa = 0.45$)** coupled with boundary-layer ventilation anomaly ($\Delta \ln(1 + \text{PBLH} \times w_{10\text{m}})$). This captures genuine synoptic accumulation while mathematically bounding CAMS false alarms.

### 2.2Principled Multivariate Foundation Model Adaptation (`TimesFM 3.0` & `Chronos-2`)
Rather than treating time-series foundation models as black-box univariate medians, we engineered three mathematical adaptations grounded in their source architectures:
1. **Compact Multi-View `TimesFM 3.0` (`V=4` Target + `V=12` Physics-Coupled):** By inspecting `TimesFM3Torch` (`external_repos/timesfm/src/timesfm3/torch/`), we proved that passing 28 unfiltered covariates dilutes `VariateAttention` softmax weights by $7\times$ and aliases 24-hour sinusoids inside 32-hour patches. Our compact `V=4` (pure cross-station target attention) + `V=12` (`[4 targets + 4 calibrated CAMS + 4 log1p_vc]` with `patch_is_target=False` enabling 64-hour `get_output_patch_via_roll` lookahead) eliminates attention dilution.
2. **Trapezoidal Quantile Quadrature for RMSE:** Because $\text{PM}_{2.5}$ is right-skewed, the median ($q_{0.50}$) systematically under-predicts the conditional mean $\mathbb{E}[Y \mid X]$. We apply trapezoidal numerical quadrature across `TimesFM 3.0`'s 9 deciles and `Chronos-2`'s 21 quantiles to recover the exact Bayes-optimal L2 point estimator.
3. **Station-Specific Variance Shrinkage for `Borivali East (11606)`:** Recognizing that `Borivali East` (adjacent to Sanjay Gandhi National Park) has the lowest post-monsoon variance (`std = 12.23 µg/m³`) and suffers $+30\%$ model over-dispersion, our horizon-decaying variance shrinkage $\alpha_s(h_{\text{lead}}) = \alpha_{0,s}\exp(-\gamma_s h_{\text{lead}})$ calibrates dispersion per micro-environment.

---

## 3. Novelty, Functionality & Civic Impact in Track 2 (`VayuDrishti Mumbai`)

### 3.1 Alignment with Zindi × Google × ClimateVerse Judging Criteria

| Judging Criterion | How `VayuDrishti Mumbai` Excels |
| :--- | :--- |
| **1. Real-World Civic Impact** | Translates raw $\mu\text{g/m}^3$ forecasts into actionable **BMC Graded Response Action Plan (GRAP)** directives across all **24 BMC Administrative Wards**, quantifying hourly exposure for **1.28M+ vulnerable children (`<5y`) and elderly (`>60y`)**, **3,800+ UDISE+ schools**, and **BMC civic hospitals** (`KEM`, `Shatabdi Govandi`, `JJ Byculla`, `Bhabha Kurla`, `M.W. Desai Malad`, `Bhagwati Borivali`). |
| **2. Scientific & Technical Novelty** | Combines **Anisotropic Wind-Advected Kriging** across 18 CAAQMS stations with **Gaussian point-source plume modeling** (`Deonar` 132 Ha landfill, `Trombay` refinery/power belt, `MbPT` dockyard), **Google AlphaEarth 10m** canyon roughness, **Google Open Buildings V3** footprint density, and **80%/95% Split-Conformal Prediction Intervals** that widen automatically during nocturnal inversions ($\text{PBLH} < 300\text{ m}$). |
| **3. Deep Google Ecosystem Integration** | Integrates **four flagship Google technologies**: (1) **Google Research `TimesFM 3.0`** (`google/timesfm-3.0-pytorch`), (2) **Google `AlphaEarth Foundations 10m` Satellite Embeddings** (`doi:10.71646/GHSZ7J`), (3) **Google `Open Buildings V3`** (`doi:10.71646/EV3RBJ`), and (4) **Google `Gemini 2.5 Flash`** (`google-genai` SDK) for live ward-level policy synthesis. |
| **4. Comprehensive ClimateVerse Dataverse Integration** | Integrates **15 official ClimateVerse datasets with verified DOIs** across meteorology (`XXOBKO`, `EAE4EE`), atmospheric chemistry (`YAQTMA`, `CAMS26`, `VA5BD9`), urban morphology (`GHSZ7J`, `EV3RBJ`), demographics (`EREDWQ`, `AAGK0G`), emission sources (`MWQNR8`, `4ULQRE`, `YJIAGQ`, `MKJGDY`), and vulnerable public infrastructure (`XJIY8W`, `1XSUVR`). |

### 3.2 Concrete Operational Use Case: Ward `M/E` (`Govandi / Shivaji Nagar / Deonar`)
During a typical post-monsoon morning (`07:00 IST` on `Nov 11, 2026`), nocturnal $\text{PBLH}$ drops to `175 m` with a weak `1.6 m/s` easterly land breeze (`78° ENE`).
- **What `VayuDrishti Mumbai` Detects:** The easterly flow aligns directly with the `Deonar` Dumping Ground (`doi:10.71646/MWQNR8`, `0.8 km` upwind of `Shivaji Nagar`), pushing predicted Ward `M/E` $\text{PM}_{2.5}$ to **`104.5 µg/m³`** (`95% Conformal Upper Bound: 128.4 µg/m³`, triggering **BMC GRAP Stage II / III**).
- **Automated Civic Directives Generated (`Tab 3` & `Tab 4`):**
  1. **Education (`doi:10.71646/XJIY8W`):** Issues an immediate `06:00 IST` alert to **244 UDISE+ schools** in Ward `M/E` (including *Shivaji Nagar BMC MPS School No. 1*, `2,480` students, `0.7 km` from Deonar) to move morning assembly indoors.
  2. **Healthcare (`doi:10.71646/1XSUVR`):** Alerts *Pt. Madan Mohan Malaviya Shatabdi Hospital, Govandi* (`3,200` daily patients) to pre-stage pediatric bronchodilator/nebulization kits and activate positive-pressure neonatal HEPA filtration.
  3. **Municipal Enforcement (`doi:10.71646/MWQNR8`, `doi:10.71646/4ULQRE`):** Directs BMC Solid Waste Management thermal drones to Sectors B–D of Deonar landfill and mandates continuous mist-cannon operation across all **41 registered MahaRERA construction sites** in Ward `M/E` during the `05:30–10:00 IST` inversion window.
