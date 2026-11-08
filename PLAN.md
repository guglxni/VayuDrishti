# Operational Execution & Competition Runbook (`PLAN.md`)

**Project:** Google Mumbai Air Quality Forecasting Challenge & VayuDrishti Mumbai (`2026`)  
**Current Date:** Saturday, Nov 7, 2026 (IST, `UTC+05:30`)

---

## 1. Master Competition Timeline & Critical Milestones

```mermaid
gantt
    title Google Mumbai Air Quality Challenge — Final Countdown Runbook (Nov 2026)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Data Updates
    Nov 5 Train.csv Release (Completed)       :done,    d1, 2026-11-05, 1d
    Nov 7 Pipeline & 8-Model Zoo Verification :active,  d2, 2026-11-07, 2d
    Nov 9 Pre-Final Train.csv Update          :         d3, 2026-11-09, 1d
    Nov 10 09:00 IST Final Train.csv Release  :crit,    d4, 2026-11-10, 1d

    section Track 1: Forecasting LB
    Public Window 4 (Nov 5–6 Scoring)         :done,    p1, 2026-11-05, 2d
    Final Ensemble Retrain & Zindi Upload     :crit,    p2, 2026-11-10, 1d
    Forecasting Submission Deadline (23:59)   :milestone, m1, 2026-11-10, 0d
    Private LB Evaluation Window (Nov 11–12)  :crit,    p3, 2026-11-11, 2d

    section Track 2 & Code Review
    Top-10 48h Code Review Verification       :crit,    c1, 2026-11-11, 2d
    Creative Submission Deadline (Nov 13)     :crit,    c2, 2026-11-11, 3d
    Finalist Pitch in Mumbai (Nov 19–20)      :         c3, 2026-11-19, 2d
```

---

## 2. Phase-by-Phase Execution Checklist

### Phase 1: Baseline, Physics & Multi-Paradigm Zoo Verification (Completed — Nov 7, 2026)
- [x] **Data Ingestion & IST Alignment (`src/fetch_data.py`, `src/validation.py`):**
  - Synced `Train.csv` (`25,204` rows), `Test.csv` (`2,592` rows), `Stations.csv`, and `AdditionalHistory.zip` (`Other_Mumbai_Stations_2025_2026.csv.gz`, `History_Mumbai_2018_2020.csv.gz`, `History_Mumbai_2024.csv.gz`).
  - Fetched and cached `open_meteo_weather.parquet` (`63,264` rows, `0` nulls), `open_meteo_air_quality.parquet` (`63,264` rows, `0` nulls), `vabb_metar_hourly.parquet` (`15,436` rows, `0` nulls), and `climateverse_catalog.json` (`15` DOIs).
- [x] **5-Fold Post-Monsoon Rolling Validation (`src/ensemble.py`):**
  - Verified 8-model 24h reporting gap OOF RMSE = **`9.4265 µg/m³`** (`+32.53%` vs. Starter RF `13.9719 µg/m³`; `+75.19%` vs. 90d Station Mean `37.9949 µg/m³`).
  - Verified 8-model 72h reporting gap OOF RMSE = **`9.5401 µg/m³`** (`+46.63%` vs. Starter RF `17.8742 µg/m³`; `+75.46%` vs. 90d Station Mean `38.8708 µg/m³`).
- [x] **FOSS Reference Ecosystem (`external_repos/`):**
  - Cloned all 10 verified repositories (`timesfm`, `chronos-forecasting`, `TabPFN`, `TimeMixer`, `AirPhyNet`, `PM2.5-GNN`, `Layer.ai-Air-Quality-Prediction-Challenge`, `climateverse_mcp`, `google-colab-cli`, `aidlc-workflows`).
- [x] **Track 2 Creative Platform (`app/mumbai_airpulse_app.py`):**
  - Built and verified all 4 interactive tabs, 18-station + 24-ward anisotropic kriging, 80%/95% conformal intervals, UDISE+ school & BMC hospital exposure matrix, and Google Gemini 2.5 Flash GRAP Copilot.

### Phase 2: Nov 9 Pre-Final Data Update Runbook (`2026-11-09`)
When Zindi posts the penultimate `Train.csv` update on **Monday, Nov 9, 2026**:
1. Pull updated `Train.csv` and refresh live Open-Meteo NWP, CAMS AQ, and VABB METAR forecasts:
   ```bash
   uv run python src/fetch_data.py --zindi-sync --weather --air-quality --metar
   ```
2. Run the 5-fold validation suite and regenerate OOF weights:
   ```bash
   uv run python src/ensemble.py
   ```
3. Generate and verify `submissions/submission_latest.csv` and upload to Zindi to benchmark on the penultimate Public Leaderboard window:
   ```bash
   uv run python src/submit.py --upload --comment "Nov 9 Sync: 8-Model Physics + CAMS MOS + GBDT + ST-GAT + Foundation Ensemble"
   ```

### Phase 3: Nov 10 Final Release Day Runbook (`2026-11-10 09:00 IST` $\to$ `23:59 IST`)
**CRITICAL DEADLINE:** Zindi releases the **final `Train.csv` update at `09:00 IST` (`03:30 UTC`) on Tuesday, Nov 10, 2026**. The final forecasting submission deadline is **`23:59 IST` (`18:29 UTC`) on Tuesday, Nov 10, 2026**, immediately preceding the **Nov 11–12 Private Leaderboard Evaluation Window (`192` rows)**.

Execute the following exact 5-step sequence on **Nov 10 after `09:05 IST`**:

```bash
# Step 1: Pull the final Nov 10 09:00 IST Train.csv + live 16-day Open-Meteo NWP + CAMS AQ + VABB METAR
export ZINDI_USERNAME="your_zindi_username"
export ZINDI_PASSWORD="your_zindi_password"
export ZINDI_CHALLENGE_ID="google-mumbai-data-challenge"
uv run python src/fetch_data.py --zindi-sync --weather --air-quality --metar

# Step 2: Confirm latest cutoff_t0 in Train.csv is >= 2026-11-09 / 2026-11-10
uv run python -c "
from src.validation import load_and_align_core_data
train, test, _, _ = load_and_align_core_data()
print('Final Train.csv rows:', len(train), '| Max hour_end (T0):', train['hour_end'].max())
"

# Step 3: Re-optimize OOF ensemble weights and generate final verified submission
uv run python src/ensemble.py
uv run python src/submit.py --upload --comment "FINAL NOV 10 SUBMISSION: Physics + CAMS MOS + GBDT + Wind-ST-GAT + TimesFM3/Chronos2 Ensemble"

# Step 4: Execute the self-contained Colab notebook on a T4 GPU via google-colab-cli for Code Review parity
colab new -s mumbai-final --gpu T4
colab upload -s mumbai-final notebooks/Mumbai_Winning_Colab.ipynb /content/Mumbai_Winning_Colab.ipynb
colab exec -s mumbai-final -f /content/Mumbai_Winning_Colab.ipynb
colab download -s mumbai-final /content/submission.csv submissions/submission_colab_final.csv
colab stop -s mumbai-final

# Step 5: Verify zero NaNs, exact 2,592 rows, and Private Window (Nov 11-12) station means
uv run python -c "
import pandas as pd
sub = pd.read_csv('submissions/submission_latest.csv')
sample = pd.read_csv('data/raw/SampleSubmission.csv')
assert len(sub) == 2592 and list(sub.columns) == ['ID', 'Target']
assert (sub['ID'] == sample['ID']).all() and sub['Target'].notna().all() and (sub['Target'] >= 0).all()
print('ALL 6 ZINDI SUBMISSION ASSERTIONS PASSED! Mean Target:', round(sub['Target'].mean(), 3))
"
```

### Phase 4: Top-10 Code Review & Track 2 Creative Submission (`Nov 11–13, 2026`)
1. **Top-10 Code Review Verification (48 Hours Post-Close):**
   - Archive directory containing `notebooks/Mumbai_Winning_Colab.ipynb`, `notebooks/Mumbai_Winning_Solution.ipynb`, `src/`, `app/`, `pyproject.toml`, `README.md`, `ARCHITECTURE.md`, and `MATH_FOUNDATIONS.md`.
   - Ensure `SEED = 42` is fixed across Python, NumPy, LightGBM, CatBoost, XGBoost, and PyTorch.
2. **Track 2 Creative Submission Form (`Due Nov 13, 2026 23:59 IST`):**
   - Submit the `VayuDrishti Mumbai` application (`app/mumbai_airpulse_app.py`), architectural & scientific proposal (`PROPOSAL.md`), and demo walkthrough highlighting all 15 integrated ClimateVerse DOIs and 4 Google technologies (`TimesFM 3.0`, `AlphaEarth 10m`, `Open Buildings V3`, `Gemini 2.5 Flash`).
