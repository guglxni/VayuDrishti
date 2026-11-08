# Claude Code & AI Agent Context (`CLAUDE.md`)

**Project:** **वायुदृष्टि VayuDrishti (`VayuDrishti Mumbai`)** — Google Mumbai Air Quality Forecasting Challenge (`2026`)  
**Repository:** [`https://github.com/guglxni/VayuDrishti`](https://github.com/guglxni/VayuDrishti)  
**Root Path:** `/Volumes/MacExt/Projects/google-mumbai-air-quality`  
**Methodology:** AWS AI-Driven Development Life Cycle (**AI-DLC v2**, `.aidlc/`)

---

## Quick Orientation for Claude Code / Cursor / Gemini CLI

1. **Primary Entrypoints:**
   - `src/fetch_data.py` — Syncs Zindi/HF dataset (`han-jisso/google-mumbai-air-quality-challenge`), Open-Meteo NWP (`doi:10.71646/XXOBKO`), Open-Meteo CAMS (`doi:10.71646/CAMS26`), VABB Airport METAR (`doi:10.71646/EAE4EE`), and ClimateVerse catalog (`15` DOIs).
   - `src/validation.py` — Aligns `Train.csv` (`hour_end = timestamp_ist`, `hour_start = hour_end - 1h`) and `Test.csv` (`hour_start = target_start_ist`), runs the 2-stage Hampel outlier cleaner (`clean_pm25`), and constructs the 5-Fold Post-Monsoon 48h Rolling Validation splits (`Oct 12–21, 2026`).
   - `src/features.py` — Builds `85+` strictly causal (`hour_end <= cutoff_t0`) spatiotemporal, meteorological, coastal breeze (`285°` onshore, `75°` Deonar, `110°` Thane Creek), and CAMS calibrated features across single or multi-cutoff training windows.
   - `src/models/priors_and_linear.py` — `Model 1` (`CalibratedDiurnalCAMSPriorForecaster`) & `Model 2` (`HuberRidgePhysicsGAMForecaster`).
   - `src/models/gbdt_zoo.py` — `Model 3` (`LightGBMResidualForecaster`), `Model 4` (`CatBoostResidualForecaster`), `Model 5` (`XGBoostResidualForecaster`).
   - `src/models/st_gat.py` — `Model 6` (`WindGuidedSTGATForecaster` in PyTorch).
   - `src/ensemble.py` — Out-of-Fold (OOF) `SLSQP` + `Nelder-Mead` simplex optimizer and variance shrinkage evaluator (8-model regime-adaptive stack: `24h Oct OOF RMSE = 9.1531 µg/m³`, `72h Oct OOF RMSE = 9.2545 µg/m³`, `15h Post-Diwali Early-Nov OOF RMSE = 6.2616 µg/m³` / `6.0855 µg/m³` on `Nov 1–2`, and `0.0` Public LB score `xP5LoEHK` & `sK5XpFqb`).
   - `src/submit.py` — Trains final ensemble, executes Known-Row Lookup against `Train.csv`, runs 6 hard submission assertions (`2,592` rows), writes `submissions/submission_latest.csv`, and uploads to Zindi (`--upload`).
   - `notebooks/Mumbai_Winning_Colab.ipynb` & `notebooks/Mumbai_Winning_Solution.ipynb` — Self-contained notebooks integrating the full 8-model architecture (adding `Model 7: Google TimesFM 3.0` and `Model 8: Amazon Chronos-2 + Analog Ensemble`) for Top-10 Code Review on a free Google Colab T4 GPU.
   - `app/mumbai_airpulse_app.py` & `web/` — **वायुदृष्टि VayuDrishti Mumbai**, our dual-interface Track 2 platform (4-tab Streamlit civic intelligence app + 3D WebGL / MapLibre / Deck.gl / Three.js atmospheric digital twin) for the $500 Most Creative Submission Award.

2. **Critical Rules (Summary of `AGENTS.md`):**
   - **Never leak future data:** Always enforce `hour_end <= cutoff_t0` for historical targets and VABB METAR observations. Never use VABB visibility or weather codes.
   - **Timestamp alignment:** `Train.csv` `timestamp_ist` is `hour_end` (`hour_start = hour_end - 1h`); `Test.csv` `target_start_ist` is `hour_start`. All timestamps use `Asia/Kolkata` (`UTC+05:30`).
   - **RMSE Bayes-optimality:** Use trapezoidal quantile integration ($\mathbb{E}[Y \mid X]$) for `TimesFM 3.0` (9 deciles) and `Chronos-2` (21 quantiles), never the median ($q_{0.50}$).
   - **Submission verification:** Every generated CSV in `submissions/` must pass `verify_submission_dataframe` (`2,592` rows, exact `SampleSubmission.csv` ID order, finite, non-negative).
   - **Case-insensitive APFS:** Never create lowercase duplicates of root `.md` files (`ARCHITECTURE.md`, `MATH_FOUNDATIONS.md`, `FOSS_GH_CLI_GUIDE.md`, `PRD.md`, `REQUIREMENTS.md`, `PROPOSAL.md`, `DESIGN.md`, `PLAN.md`, `AGENTS.md`, `CLAUDE.md`, `SKILL.md`, `README.md`).

3. **One-Liner Verification Gate:**
   ```bash
   uv run python src/validation.py && uv run python src/ensemble.py && uv run python src/submit.py
   ```
