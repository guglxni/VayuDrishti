# AI-DLC Phase 3 (Operations): Deterministic Verification Sensors & Release Runbook

**Lifecycle Stage:** `Operations -> Automated Sensors, Gates & Live Competition Runbook`  
**Status:** `ACTIVE`

---

## 1. Automated Deterministic Sensors (AI-DLC Guard Gates)

In accordance with AI-DLC v2's second tenet (*"Tools for determinism"*), every Unit of Work is guarded by an automated command-line sensor that must exit with code `0` before any submission is uploaded to Zindi:

| Sensor ID | Gate Name | Deterministic Command | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **`SENSOR-1`** | **Dataset & IST Schema Integrity** | `uv run python src/fetch_data.py` | `Train.csv` (`>=25,204` rows), `Test.csv` (`2,592` rows), `open_meteo_weather.parquet` (`0` nulls), `open_meteo_air_quality.parquet` (`0` nulls), `vabb_metar_hourly.parquet` (`0` nulls), `climateverse_catalog.json` (`15` DOIs). |
| **`SENSOR-2`** | **Causal Anti-Leakage & Hampel Gate** | `uv run python src/validation.py` | Asserts `hist_df["hour_end"].max() <= cutoff_t0 < val_start` across all 5 post-monsoon folds (`24h`, `48h`, `72h` gaps); verifies Stage 1 (`flag_high_1000`) + Stage 2 (`7h Hampel`) cleaning. |
| **`SENSOR-3`** | **Feature Completeness Gate** | `uv run python src/features.py` | Constructs `85+` numeric features for all `2,592` test rows at `latest_cutoff_t0` with `int(test_feats[feat_cols].isna().sum().sum()) == 0`. |
| **`SENSOR-4`** | **5-Fold OOF RMSE Benchmark Gate** | `uv run python src/ensemble.py` | 24h Gap Stacked OOF RMSE `<= 10.70 µg/m³` (`10.6835 µg/m³`); 72h Gap Stacked OOF RMSE `<= 10.35 µg/m³` (`10.3254 µg/m³`). Writes `data/processed/ensemble_weights_gap{24,72}h.json`. |
| **`SENSOR-5`** | **6-Point Zindi Submission Gate** | `uv run python src/submit.py` | Verifies `len(sub) == 2592`, `columns == ['ID', 'Target']`, exact `SampleSubmission.csv` ID order, `0` NaNs, `0` infinities, and `Target >= 0.0` on re-read from disk. |
| **`SENSOR-6`** | **VayuDrishti App Health Gate** | `uv run python -c "import app.physics_engine, app.spatial_catalog, app.gemini_copilot; print('App modules OK')"` | All Track 2 creative platform modules import cleanly and compute 24-ward anisotropic kriging + GRAP briefings with zero errors. |

---

## 2. Live Operations Runbook for Nov 9 & Nov 10, 2026 (`09:00 IST` Final Release)

### 2.1 Pre-Final Sync (`Monday, Nov 9, 2026`)
1. Pull updated `Train.csv` and live NWP/CAMS forecasts:
   ```bash
   uv run python src/fetch_data.py --zindi-sync --weather --air-quality --metar
   ```
2. Re-run OOF validation and submission generation:
   ```bash
   uv run python src/ensemble.py && uv run python src/submit.py --upload
   ```

### 2.2 Final Release Day Protocol (`Tuesday, Nov 10, 2026, 09:00 IST -> 23:59 IST`)
Immediately after Zindi publishes the final `Train.csv` update at **`09:00 IST` (`03:30 UTC`) on Nov 10, 2026**:
1. **Sync Final Data & Live 16-Day Weather/CAMS Forecasts:**
   ```bash
   export ZINDI_USERNAME="your_zindi_username"
   export ZINDI_PASSWORD="your_zindi_password"
   export ZINDI_CHALLENGE_ID="google-mumbai-data-challenge"
   uv run python src/fetch_data.py --zindi-sync --weather --air-quality --metar
   ```
2. **Verify Cutoff $T_0$ Advancement & Known-Row Coverage:**
   Confirm that `train["hour_end"].max()` has advanced to the Nov 9/10 cutoff and check how many historical test rows in `Test.csv` (`2026-10-22` to `T0`) are matched by Known-Row Lookup.
3. **Execute Full Validation, Final Training & Zindi Upload:**
   ```bash
   uv run python src/ensemble.py
   uv run python src/submit.py --upload --comment "FINAL NOV 10: 8-Model Physics + CAMS MOS + GBDT + Wind-ST-GAT + TimesFM3/Chronos2 Ensemble"
   ```
4. **Verify Google Colab T4 Notebook Parity (`google-colab-cli`):**
   ```bash
   colab new -s mumbai-final --gpu T4
   colab upload -s mumbai-final notebooks/Mumbai_Winning_Colab.ipynb /content/Mumbai_Winning_Colab.ipynb
   colab exec -s mumbai-final -f /content/Mumbai_Winning_Colab.ipynb
   colab download -s mumbai-final /content/submission.csv submissions/submission_colab_final.csv
   colab stop -s mumbai-final
   ```
