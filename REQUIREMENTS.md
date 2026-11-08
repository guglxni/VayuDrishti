# Formal System Requirements (EARS Specification)

**Project:** Google Mumbai Air Quality Forecasting Challenge & VayuDrishti Mumbai (`2026`)  
**Notation:** **EARS** (*Easy Approach to Requirements Syntax* — `WHEN <trigger> / WHILE <precondition>, THE SYSTEM SHALL <response>`)

---

## 1. Data Ingestion, Timestamp Alignment & Anti-Leakage Requirements (`REQ-DATA`)

- **`REQ-DATA-001` (Canonical Timezone Alignment):**  
  **WHEN** ingesting any timestamp from `Train.csv`, `Test.csv`, `AdditionalHistory.zip`, Open-Meteo NWP (`doi:10.71646/XXOBKO`), Open-Meteo CAMS (`doi:10.71646/CAMS26`), or VABB METAR (`doi:10.71646/EAE4EE`), **THE SYSTEM SHALL** convert and localize all timestamps to `Asia/Kolkata` (`UTC+05:30`) and define `hour_start` (interval start) and `hour_end = hour_start + 1h` (interval end).
- **`REQ-DATA-002` (Train vs. Test 1-Hour Convention Reconciliation):**  
  **WHEN** loading `Train.csv`, **THE SYSTEM SHALL** map `hour_end = timestamp_ist` and compute `hour_start = hour_end - 1 hour` so that every training record aligns identically with `Test.csv` (`hour_start = target_start_ist`, `hour_end = target_end_ist`) and Open-Meteo hourly timestamps (`target_start_ist == hour_start`).
- **`REQ-DATA-003` (Strict Causal Anti-Leakage Invariant):**  
  **WHEN** constructing features or fitting models for any validation or test forecast cutoff timestamp `cutoff_t0` ($T_0$), **THE SYSTEM SHALL** access historical target observations (`Train.csv` and `Other_Mumbai_Stations_2025_2026.csv.gz`) and VABB METAR ground observations **if and only if** `hour_end <= cutoff_t0` (`target_start_ist + 1h <= cutoff_t0`), and **SHALL** assert `hist_df["hour_end"].max() <= cutoff_t0 < val_start`.
- **`REQ-DATA-004` (VABB METAR Zero-Leakage Variable Filter):**  
  **WHEN** ingesting Chhatrapati Shivaji Maharaj International Airport (`VABB`) METAR observations (`doi:10.71646/EAE4EE`), **THE SYSTEM SHALL** restrict variables exclusively to physical meteorology (`tmpc`, `dwpc`, `relh`, `drct`, `sknt`, `alti`) and **SHALL NOT** ingest visibility or weather-code (`wxcodes`) fields.
- **`REQ-DATA-005` (NWP-Consistent Weather Continuity):**  
  **WHEN** merging historical and forecast weather from Open-Meteo (`src/fetch_data.py`), **THE SYSTEM SHALL** construct NWP-consistent series (`*_nwp` combining Historical Forecast API and 16-Day Live Forecast API, backfilled with ERA5 Archive) and **SHALL** fill any CAMS horizon beyond `+120h` using the station's trailing 120-hour hour-of-day diurnal median so that `0` NaN values exist across `2025-02-01` through `2026-11-21`.

---

## 2. Target Outlier Cleaning & Validation Split Requirements (`REQ-VAL`)

- **`REQ-VAL-001` (2-Stage Target Outlier Cleaner on Training Data Only):**  
  **WHEN** preparing training targets (`clean_pm25` in `src/validation.py`), **THE SYSTEM SHALL** execute a 2-stage causal cleaner per station:
  1. **Stage 1 (Hardware Glitch Filter):** Mask observations where `flag_high_1000 == True` or `pm25 > 450.0 µg/m³` or `pm25 < 0.0` (catching extreme sensor malfunctions such as the `1,985.12 µg/m³` spike at `Shivaji Nagar` on `2026-01-17 17:00 IST`) and impute via local station rolling median and linear interpolation.
  2. **Stage 2 (Rolling 7-Hour Hampel Filter):** Compute the 7-hour rolling median $m_{s,t}$ and scaled MAD $\hat{\sigma}_{\text{MAD}} = \max(1.4826 \times \text{MAD}_7, 2.5\;\mu\text{g/m}^3)$ and replace upward spikes exceeding $m_{s,t} + 4.5\,\hat{\sigma}_{\text{MAD}}$ with $m_{s,t}$.
- **`REQ-VAL-002` (Unmodified Raw Validation Ground Truth):**  
  **WHEN** computing validation RMSE (`rmse` in `src/validation.py` and `src/ensemble.py`), **THE SYSTEM SHALL** evaluate predictions against the **raw, unmodified** `val_df["pm25"]` column in `Train.csv` to mirror Zindi's exact leaderboard scoring.
- **`REQ-VAL-003` (5-Fold Post-Monsoon 48-Hour Rolling Validation):**  
  **WHEN** evaluating model architectures and optimizing ensemble weights (`src/ensemble.py`), **THE SYSTEM SHALL** evaluate across 5 non-overlapping 48-hour post-monsoon October 2026 windows (`Oct 12–13`, `Oct 14–15`, `Oct 16–17`, `Oct 18–19`, `Oct 20–21`, `N = 960` station-hours) across both `gap_hours = 24` (matching the Nov 10 $\to$ Nov 11–12 Private window) and `gap_hours = 72`.

---

## 3. Model Zoo, Foundation Model & Stacking Requirements (`REQ-MOD`)

- **`REQ-MOD-001` ($\tanh$-Saturating CAMS MOS in Model 1):**  
  **WHEN** computing the physics-calibrated CAMS prior (`Model 1`), **THE SYSTEM SHALL** bound the log-ratio CAMS innovation relative to its trailing 7-day baseline at $T_0$ via a smooth $\tanh$ saturation gate ($\kappa = 0.45$) or clipped power law (`cams_rel_factor ** c_exp`), preventing regional CAMS dust false alarms (such as Fold 4's `102 µg/m³` spike) from corrupting surface $\text{PM}_{2.5}$ forecasts.
- **`REQ-MOD-002` (Multi-Paradigm Residual Diversity):**  
  **THE SYSTEM SHALL** train at least six distinct model families on multi-cutoff historical training datasets (`Oct–Nov 2025` + rolling `2026` cutoffs) and integrate two zero-shot foundation/analog families:
  - `Model 1`: `CalibratedDiurnalCAMSPriorForecaster` (`TimeMixer` synoptic-diurnal physics prior)
  - `Model 2`: `HuberRidgePhysicsGAMForecaster` (Periodic B-spline Huber + Ridge GAM with Duan's smearing)
  - `Model 3`: `LightGBMResidualForecaster` (`deterministic=True`, L2 + Huber blend)
  - `Model 4`: `CatBoostResidualForecaster` (Symmetric oblivious depth-5 trees, L2 + Huber blend)
  - `Model 5`: `XGBoostResidualForecaster` (Histogram depth-4 trees, L2 + PseudoHuber blend)
  - `Model 6`: `WindGuidedSTGATForecaster` (PyTorch dynamic wind-advection graph attention network)
  - `Model 7`: `TimesFM 3.0 (330.71M)` Compact `V=4` + `V=12` Multi-View Forecaster (`use_znorm=False`, `padding_mode="edge"`)
  - `Model 8`: `Amazon Chronos-2 (120M)` (`cross_learning=True` + future covariates) + `MeteorologicalAnalogForecaster (AnEn)`
- **`REQ-MOD-003` (Trapezoidal Quantile Expectation for RMSE):**  
  **WHEN** extracting point forecasts from quantile foundation models (`TimesFM 3.0` 9 deciles or `Chronos-2` 21 quantiles), **THE SYSTEM SHALL** compute the trapezoidal numerical integral over all predicted quantiles to approximate the Bayes-optimal conditional mean $\mathbb{E}[Y \mid X]$ under RMSE rather than using the median ($q_{0.50}$).
- **`REQ-MOD-004` (Simplex Stacking & Station-Specific Variance Shrinkage):**  
  **WHEN** blending individual model predictions (`src/ensemble.py` and `src/submit.py`), **THE SYSTEM SHALL** optimize non-negative simplex weights ($\sum_m w_m = 1, w_m \ge 0.04$) via `SLSQP` + `Nelder-Mead` and apply variance shrinkage $\alpha_s(h_{\text{lead}})$ toward the physics prior anchor, specifically correcting the $+30\%$ forecast over-dispersion at `Borivali East (11606)` and damping residual drift at ultra-long horizons ($h_{\text{lead}} > 96\text{h}$).

---

## 4. Zindi Submission & Top-10 Code Review Reproducibility Requirements (`REQ-SUB`)

- **`REQ-SUB-001` (Known-Row Lookup for Updated `Train.csv` Snapshots):**  
  **WHEN** generating a submission file (`src/submit.py`), **THE SYSTEM SHALL** check whether any `(location_id, hour_start)` in `Test.csv` is already present with a valid non-negative `pm25` observation in `Train.csv`, populate those exact known values, log the exact count of matched vs. forecasted rows, and forecast all remaining unseen rows via the stacked ensemble.
- **`REQ-SUB-002` (Hard 6-Point Zindi Submission Verification Gate):**  
  **BEFORE** writing or uploading any submission CSV (`submissions/submission_latest.csv`), **THE SYSTEM SHALL** assert:
  1. `len(sub_df) == 2592`
  2. `list(sub_df.columns) == ["ID", "Target"]`
  3. `sub_df["ID"].notna().all() and sub_df["ID"].is_unique`
  4. `sub_df["ID"].astype(str).tolist() == sample_sub_df["ID"].astype(str).tolist()`
  5. `sub_df["Target"].notna().all() and np.isfinite(sub_df["Target"].values).all()`
  6. `(sub_df["Target"].values >= 0.0).all()`
- **`REQ-SUB-003` (Top-10 Code Review Determinism & Colab Self-Containment):**  
  **WHEN** executed on a fresh Google Colab T4 GPU runtime (`notebooks/Mumbai_Winning_Colab.ipynb` or `notebooks/Mumbai_Winning_Solution.ipynb`), **THE SYSTEM SHALL** fix all random seeds (`PYTHONHASHSEED=42`, `np.random.seed(42)`, `torch.manual_seed(42)`, `deterministic=True`), automatically fetch required datasets/models from Hugging Face and Open-Meteo without manual intervention, and produce a byte-verified `submission.csv` within 20 minutes.

---

## 5. Track 2 Creative Submission Requirements (`REQ-APP`)

- **`REQ-APP-001` (Zero-Crash Offline & Live Execution):**  
  **WHEN** launched via `uv run streamlit run app/mumbai_airpulse_app.py`, **THE SYSTEM SHALL** render all 4 interactive tabs with zero runtime exceptions whether `GEMINI_API_KEY` is configured (using live `gemini-2.5-flash` via `google-genai`) or absent (using the deterministic physics-grounded BMC GRAP policy engine in `app/gemini_copilot.py`).
- **`REQ-APP-002` (18-Station & 24-Ward Anisotropic Kriging Coverage):**  
  **WHEN** a user selects any timestamp across the test horizon (`2026-10-22` to `2026-11-17`, defaulting to the `Nov 11–12` Private Evaluation Window), **THE SYSTEM SHALL** compute wind-advected anisotropic kriging + `AlphaEarth 10m` morphology + hotspot Gaussian plume surfaces across all 18 Mumbai CAAQMS stations, 24 BMC administrative wards, and georeferenced UDISE+ schools and BMC civic hospitals.
- **`REQ-APP-003` (Explicit ClimateVerse DOI Provenance):**  
  **THE SYSTEM SHALL** cite and link the exact ClimateVerse DOIs (`doi:10.71646/YAQTMA`, `XXOBKO`, `EAE4EE`, `GHSZ7J`, `EV3RBJ`, `EREDWQ`, `AAGK0G`, `MWQNR8`, `VA5BD9`, `XJIY8W`, `1XSUVR`, `YJIAGQ`, `4ULQRE`, `MKJGDY`, `CAMS26`) inline within the UI and inside every generated BMC GRAP operational briefing.
