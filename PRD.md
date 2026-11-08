# Product Requirements Document (PRD) — Google Mumbai Air Quality Forecasting Engine & VayuDrishti Mumbai

**Product Name:** **VayuDrishti Mumbai (`Mumbai AirPulse AI`) & Multi-Paradigm Forecasting Engine**  
**Competition:** Zindi × Google × Beyond × ClimateVerse — *Google Mumbai Air Quality Data Challenge (2026)*  
**Target Tracks:**
1. **Track 1 — Main Forecasting Leaderboard ($2,500 1st / $1,500 2nd / $1,000 3rd):** Hourly $\text{PM}_{2.5}$ forecasting across 4 Mumbai CPCB stations (`2,592` test rows; Private LB evaluated on `2026-11-11` to `2026-11-12`).
2. **Track 2 — Most Creative Submission Award ($500 USD):** Interactive ward-level air quality intelligence, conformal risk warning, vulnerable facility exposure audit, and Google Gemini 2.5 Flash civic action copilot (`app/mumbai_airpulse_app.py`).

---

## 1. Problem Statement & Atmospheric Context

Every October through February, the Mumbai Metropolitan Region (13+ million citizens across 24 Brihanmumbai Municipal Corporation administrative wards) experiences a severe **post-monsoon air quality regime shift**:
1. **Meteorological Regime Collapse:** As the southwest monsoon withdraws in early October, wet deposition ceases, nocturnal Planetary Boundary Layer Height ($\text{PBLH}$) drops from $>800\text{ m}$ to `<180 m`, and morning **easterly land-breeze drainage (`u10 < 0`)** advects smoldering waste plumes from the 132-hectare `Deonar` Dumping Ground, industrial emissions from `Trombay/Mahul`, and construction dust from `MahaRERA` corridors across dense residential wards before the afternoon **Arabian Sea breeze (`u10 > 0`)** restores ventilation.
2. **Failure of Naive Machine Learning Models:** Standard tabular regressors trained on a trailing 90-day window (e.g., the competition Starter Notebook Random Forest) are dominated by July–September monsoon observations (`15–25 µg/m³`) and under-predict post-monsoon concentrations (`60–140 µg/m³`) by `>25 µg/m³` (`24h OOF RMSE = 13.97 µg/m³`, `72h OOF RMSE = 17.87 µg/m³`). Conversely, uncalibrated regional chemical transport models (ECMWF CAMS) suffer from regional Arabian Sea dust false alarms (such as the `102 µg/m³` spike on `2026-10-18..19`).
3. **Civic Monitoring Blind Spots:** While the Zindi leaderboard scores 4 target CAAQMS stations (`Malad West`, `Shivaji Nagar`, `Borivali East`, `Byculla`), 20 of Mumbai's 24 BMC wards lack target stations, leaving municipal ward officers, 3,800+ UDISE+ school principals, and civic hospitals without hyperlocal, uncertainty-calibrated 72-hour decision support.

---

## 2. User Personas & Jobs-to-be-Done (JTBD)

| Persona | Primary Role | Core Needs & Jobs-to-be-Done | How Our Product Solves It |
| :--- | :--- | :--- | :--- |
| **1. Zindi Evaluators & Top-10 Code Auditors** | Competition Judges & Technical Verification Team | Deterministic (`SEED = 42`), leak-free (`hour_end <= cutoff_t0`), reproducible code that executes end-to-end on a free Google Colab T4 GPU in `<20 minutes` and achieves #1 Private LB RMSE on `Nov 11–12, 2026`. | Self-contained `notebooks/Mumbai_Winning_Colab.ipynb` + modular CLI pipeline (`src/`) with automated schema/assertion checks and zero manual steps. |
| **2. BMC Environment Dept & 24 Ward Officers** | Municipal Enforcement & GRAP Operations (`Wards A`–`T`) | Know **which wards** will breach India NAAQS (`60 µg/m³`) or Severe (`120 µg/m³`) thresholds 48–72 hours ahead, and **when** to schedule mechanical water sprinkling, `MahaRERA` construction misting, and `Deonar` landfill drone patrols. | **Tab 1** (24-Ward Anisotropic Kriging Map) + **Tab 4** (Google Gemini 2.5 Flash Civic GRAP Copilot citing exact ClimateVerse DOIs and operational time windows). |
| **3. UDISE+ School Principals & Education Officers** | Child Health Protection across 3,800+ Mumbai Schools | Decide by `06:00 IST` whether to suspend outdoor morning assembly (`07:30–09:00 IST`) or physical education due to morning land-breeze inversion trapping. | **Tab 2** (80%/95% Conformal Prediction Intervals) + **Tab 3** (Georeferenced UDISE+ School Exposure Audit & automated assembly alerts). |
| **4. BMC Civic Hospital Superintendents** | Healthcare Capacity Planning (`KEM`, `Shatabdi`, `JJ`, `Bhabha`, `M.W. Desai`, `Bhagwati`) | Anticipate pediatric nebulization and COPD/asthma OPD surges 48 hours ahead and activate positive-pressure neonatal HEPA filtration during low-ventilation (`VC < 1,000 m²/s`) windows. | **Tab 3** (BMC Civic Hospital Respiratory Surge Matrix & inhaled dose estimation in `µg/hr`). |

---

## 3. Competition Rolling Evaluation Schedule & Target Windows

The `Test.csv` grid contains `2,592` hourly rows (`4 stations × 648 hours` from `2026-10-22 00:00 IST` to `2026-11-17 23:00 IST`). Zindi evaluates active submissions on rolling 48-hour (`192`-row) scoring blocks following periodic `Train.csv` updates:

| Window Phase | `Train.csv` Cutoff ($T_0$) | Active 48h Scoring Window (IST) | Row Count | Strategic Significance |
| :--- | :--- | :--- | :---: | :--- |
| **Public Window 1** | `2026-10-22 00:00+05:30` | `2026-10-22 00:00` $\to$ `2026-10-23 23:00` | `192` | Initial Public LB benchmark (Starter RF `~37.9` vs. TopLB `~16.46`). |
| **Public Window 2** | `2026-10-26` update | `2026-10-27 00:00` $\to$ `2026-10-28 23:00` | `192` | Early post-monsoon buildup evaluation. |
| **Public Window 3** | `2026-10-31` update | `2026-11-01 00:00` $\to$ `2026-11-02 23:00` | `192` | Early November diurnal inversion regime. |
| **Public Window 4** | `2026-11-05` update | `2026-11-05 00:00` $\to$ `2026-11-06 23:00` | `192` | Active Public LB window prior to final weekend. |
| **Pre-Final Data Sync** | `2026-11-09` & **`2026-11-10 09:00 IST`** | Final `Train.csv` release before forecasting deadline (`Nov 10 23:59 IST`). | — | Critical 15-hour window to re-sync `Train.csv`, Open-Meteo/CAMS, and VABB METAR. |
| **Final Private LB Window** | **`2026-11-10 09:00+05:30`** | **`2026-11-11 00:00` $\to$ `2026-11-12 23:00`** | **`192`** | **100% of Final Track 1 Prize Ranking ($2,500 / $1,500 / $1,000).** |

---

## 4. Product Scope & Feature Matrix

### 4.1 Track 1: Multi-Paradigm Forecasting Engine (`src/` & `notebooks/`)
1. **Automated Multi-Source Data Ingestion (`src/fetch_data.py`):** One-command sync of Zindi/Hugging Face core data, Open-Meteo Historical Archive + Historical Forecast + 16-Day NWP Forecast (`doi:10.71646/XXOBKO`), Open-Meteo CAMS Air Quality (`doi:10.71646/CAMS26`), VABB Airport METAR (`doi:10.71646/EAE4EE`), and ClimateVerse Dataverse catalog (`15` DOIs).
2. **Causal Anti-Leakage & Hampel Cleaning (`src/validation.py`):** Strict `hour_end <= cutoff_t0` filtering, 2-stage hardware glitch (`flag_high_1000` / `>450 µg/m³`) + 7-hour Hampel (`4.5 MAD`) cleaner, and 5-fold post-monsoon rolling 48h cross-validation across `24h`, `48h`, and `72h` reporting gaps.
3. **8-Model Multi-Paradigm Zoo (`src/models/`, `src/ensemble.py`, `notebooks/`):**
   - `Model 1`: `TimeMixer` Synoptic-Diurnal Decomposed Physics Prior with $\tanh$-saturating CAMS MOS.
   - `Model 2`: Dual-Scale Huber/Ridge Physics GAM with periodic splines and Duan's smearing estimator.
   - `Model 3`: `LightGBM` Residual Forecaster (`deterministic=True`, L2 + Huber blend).
   - `Model 4`: `CatBoost` Symmetric Oblivious-Tree Residual Forecaster.
   - `Model 5`: `XGBoost` Depth-Regularized Histogram Residual Forecaster.
   - `Model 6`: `Wind-Guided Spatiotemporal Graph Attention Network (ST-GAT)` in PyTorch.
   - `Model 7`: `Google Research TimesFM 3.0 (330.71M)` Compact `V=4` + `V=12` Multi-View Forecaster with 9-decile trapezoidal expectation.
   - `Model 8`: `Amazon Chronos-2 (120M)` (`cross_learning=True` + future covariates, 21-quantile trapezoidal expectation) + `Meteorological Analog Ensemble (AnEn)`.
4. **Simplex Stacking, Variance Shrinkage & Known-Row Lookup (`src/submit.py`):** SLSQP + Nelder-Mead weight optimization, station-specific horizon-decaying shrinkage (correcting `Borivali East` over-dispersion), exact `Train.csv` lookup for observed historical rows, and 6-point submission verification.

### 4.2 Track 2: VayuDrishti Mumbai Civic Intelligence Platform (`app/`)
1. **Tab 1 — Hyperlocal 24-Ward Spatiotemporal Kriging & Wind-Plume GIS Map:** Interactive Folium + Plotly Mapbox surface fusing all 18 Mumbai CAAQMS stations (`4 Target + 14 Auxiliary`), wind-advection vectors, hotspot Gaussian plumes (`Deonar`, `Trombay`, `MbPT Dockyard`), `AlphaEarth 10m` roughness, and `Open Buildings V3` canyon density.
2. **Tab 2 — 72-Hour Conformal Forecast & Physical Driver Explorer:** Station-level fan charts displaying point forecasts, calibrated **80% and 95% Split-Conformal Prediction Intervals**, Hampel glitch diagnostics, and synchronized panels for $\text{PBLH}$, Ventilation Coefficient, coastal onshore breeze, and CAMS trajectories across the `Nov 11–12` Private Evaluation Window.
3. **Tab 3 — Vulnerable Population, UDISE+ School & BMC Civic Hospital Exposure Audit:** Ward-by-ward ranking of `WorldPop 2026` pediatric (`<5y`) and geriatric (`>60y`) exposure, plus facility-level alerts for high-risk UDISE+ schools and BMC civic hospitals.
4. **Tab 4 — Google Gemini 2.5 Flash Civic Action Copilot & ClimateVerse DOI Explorer:** Generates structured BMC Graded Response Action Plan (GRAP) operational orders via `google-genai` (`gemini-2.5-flash`) with a 100% deterministic offline fallback, plus an interactive searchable catalog of all 15 integrated ClimateVerse DOIs.

---

## 5. Key Success Metrics & KPIs

| Metric Category | Target KPI | Current Verified Status |
| :--- | :--- | :--- |
| **5-Fold Post-Monsoon OOF RMSE (24h Gap)** | `< 11.00 µg/m³` (`>20%` gain over Starter RF `13.97 µg/m³`) | **`9.4265 µg/m³`** (`+32.53%` vs Starter RF, `+75.19%` vs 90d Mean) |
| **5-Fold Post-Monsoon OOF RMSE (72h Gap)** | `< 10.80 µg/m³` (`>38%` gain over Starter RF `17.87 µg/m³`) | **`9.5401 µg/m³`** (`+46.63%` vs Starter RF, `+75.46%` vs 90d Mean) |
| **Submission Integrity & Row Count** | `2,592` rows, exact `SampleSubmission.csv` ID order, `0` NaNs/negatives | **100% Verified (`submissions/submission_latest.csv`)** |
| **Top-10 Code Review Runtime (Colab T4)** | `< 20 minutes` end-to-end with `SEED = 42` determinism | **Verified (`~14–18 min` on single T4 GPU / `<3 min` CPU tabular+GAT)** |
| **ClimateVerse & Google Tool Integration** | $\ge 10$ ClimateVerse DOIs + $\ge 3$ Google tools/models | **15 ClimateVerse DOIs + 4 Google Tools (`TimesFM 3.0`, `AlphaEarth 10m`, `Open Buildings V3`, `Gemini 2.5 Flash`)** |
