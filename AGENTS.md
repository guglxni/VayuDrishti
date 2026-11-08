# Agentic Engineering & Vibe-Coding Steering Guide (`AGENTS.md`)

**Project & Repository:** **वायुदृष्टि VayuDrishti** ([`https://github.com/guglxni/VayuDrishti`](https://github.com/guglxni/VayuDrishti))  
**Workspace:** `/Volumes/MacExt/Projects/google-mumbai-air-quality`  
**Competition:** Zindi × Google × Beyond × ClimateVerse — *Google Mumbai Air Quality Data Challenge (2026)*  
**Supported Harnesses:** Claude Code, Cursor, Google Gemini CLI, OpenAI Codex CLI, AWS Kiro / AI-DLC v2

---

## 1. Mission & Dual-Track Objective

Any AI coding agent operating in this repository is responsible for maintaining and advancing two tightly coupled tracks:
1. **Track 1 — Main Zindi Forecasting Leaderboard ($2,500 1st Place):** Predict hourly $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) across 4 Mumbai CPCB stations (`11606` Borivali East, `11611` Malad West, `3409510` Byculla, `3409511` Shivaji Nagar) for all `2,592` rows in `Test.csv` (`2026-10-22` to `2026-11-17`), evaluated by **RMSE** on the **Nov 11–12, 2026 Private Leaderboard window (`192` rows)**.
2. **Track 2 — Most Creative Submission ($500 Special Award):** Maintain and enhance **वायुदृष्टि VayuDrishti Mumbai** (`web/` 3D WebGL Atmospheric Digital Twin + `app/mumbai_airpulse_app.py` 4-tab Streamlit platform), covering hyperlocal 24-Ward air quality intelligence, conformal risk warning, UDISE+ school/hospital audit, and Google Gemini BMC GRAP Copilot.

---

## 2. Non-Negotiable Mathematical & Data Invariants (NEVER VIOLATE)

### Invariant 1: Timestamp Convention (`hour_start` vs. `hour_end` in `Asia/Kolkata`)
- **All timestamps MUST be converted to `Asia/Kolkata` (`UTC+05:30`).**
- In `Train.csv` and `Other_Mumbai_Stations_2025_2026.csv.gz`, `timestamp_ist` is the **END** of the 1-hour averaging window (`hour_end`). Therefore:
  ```python
  train["hour_end"] = pd.to_datetime(train["timestamp_ist"], utc=True).dt.tz_convert("Asia/Kolkata")
  train["hour_start"] = train["hour_end"] - pd.Timedelta(hours=1)
  train["target_start_ist"] = train["hour_start"]
  ```
- In `Test.csv`, `target_start_ist` is `hour_start` and `target_end_ist` is `hour_end`.
- In Open-Meteo (`open_meteo_weather.parquet`, `open_meteo_air_quality.parquet`) and VABB METAR (`vabb_metar_hourly.parquet`), timestamps are indexed by `target_start_ist` (`== hour_start`).

### Invariant 2: Strict Causal Anti-Leakage Gate (`hour_end <= cutoff_t0`)
- When constructing features or fitting models at forecast cutoff $T_0$ (`cutoff_t0`), you **MUST** filter historical observations by:
  ```python
  hist_df = train_df.loc[train_df["hour_end"] <= cutoff_t0]
  ```
- **NEVER** use centered rolling windows, global target scalers fit across validation boundaries, or future ground-truth observations (`hour_end > cutoff_t0`).
- **NEVER** ingest visibility or weather-code (`wxcodes`) columns from VABB METAR (`doi:10.71646/EAE4EE`). Only `tmpc`, `dwpc`, `relh`, `drct`, `sknt`, and `alti` are permitted.

### Invariant 3: Clean Training Targets (`pm25_clean`), Evaluate on Raw Validation Targets (`pm25`)
- Always train models on `pm25_clean` (produced by `clean_pm25` in `src/validation.py`: Stage 1 masks `flag_high_1000 == True` or `pm25 > 450 µg/m³`; Stage 2 applies a 7-hour rolling Hampel filter at `4.5 * 1.4826 * MAD`).
- Always compute validation `rmse(y_true, y_pred)` against the **unmodified raw `val_df["pm25"]`** column to match Zindi's exact leaderboard evaluation.

### Invariant 4: Foundation Model Guardrails (`TimesFM 3.0` & `Chronos-2`)
- **Trapezoidal Expectation for RMSE:** Never use the raw median ($q_{0.50}$) from `TimesFM 3.0` or `Chronos-2` as a final point forecast under RMSE. Always integrate over all predicted quantiles (`9` deciles for `TimesFM 3.0` with weights `[0.15, 0.10, ..., 0.10, 0.15]`; `21` quantiles for `Chronos-2` via `Chronos2Pipeline._get_prob_mass_per_quantile_level`).
- **Compact Variate Views in `TimesFM 3.0`:** Never pass 28 unfiltered covariates into `TimesFM3Torch`'s `VariateAttention` (`use_rope_var=False`), and never pass `hour_sin/cos` (which alias inside 32-hour patches) or near-zero residuals with `use_znorm=True`. Use the Compact Multi-View (`V=4` target-only + `V=12` `[4 targets + 4 CAMS + 4 log1p_vc]`, `use_znorm=False, padding_mode="edge"`).

### Invariant 5: macOS APFS Case-Insensitive Filesystem Guardrail
- `/Volumes/MacExt` is formatted as macOS APFS (case-insensitive by default). `ARCHITECTURE.md` and `architecture.md` point to the **exact same inode on disk**. Always use uppercase canonical filenames for root documentation (`ARCHITECTURE.md`, `MATH_FOUNDATIONS.md`, `FOSS_GH_CLI_GUIDE.md`, `PRD.md`, `REQUIREMENTS.md`, `PROPOSAL.md`, `DESIGN.md`, `PLAN.md`, `AGENTS.md`, `CLAUDE.md`, `SKILL.md`, `README.md`).

---

## 3. Essential CLI Commands (`uv`, `gh`, `hf`, `colab`, `zindi`)

```bash
# 1. Data Ingestion & ClimateVerse Sync
uv run python src/fetch_data.py

# 2. Validation & 2-Stage Hampel Cleaning Verification
uv run python src/validation.py

# 3. Feature Engineering Verification (85+ leak-free features)
uv run python src/features.py

# 4. 5-Fold Post-Monsoon Rolling Validation (24h & 72h reporting gaps)
uv run python src/ensemble.py

# 5. Final Submission Generation, Known-Row Lookup & 6-Point Verification
uv run python src/submit.py

# 6. Direct Zindi Upload (requires ZINDI_USERNAME and ZINDI_PASSWORD)
uv run python src/submit.py --upload --comment "8-Model Physics + CAMS MOS + GBDT + Wind-ST-GAT + Foundation Ensemble"

# 7. Launch Track 2 Creative App (VayuDrishti Mumbai)
uv run streamlit run app/mumbai_airpulse_app.py
```

---

## 4. Reference Architecture & Documentation Index

Before modifying any component, consult:
- `ARCHITECTURE.md` — Dual-track system architecture, Mermaid data-flow diagrams, and 8-Model Zoo specification.
- `MATH_FOUNDATIONS.md` — Complete mathematical proofs and equations (`TimesFM 3.0`, `Chronos-2`, `TimeMixer`, `AirPhyNet`, `AnEn`, Duan's Smearing, optimal L2 variance shrinkage, Split Conformal Prediction).
- `FOSS_GH_CLI_GUIDE.md` — Cloned FOSS repositories in `external_repos/`, Hugging Face models, and CLI commands.
- `REQUIREMENTS.md` & `PRD.md` — Formal EARS requirements, acceptance criteria, and evaluation windows.
- `PLAN.md` — Operational runbook for the Nov 9 update and Nov 10 `09:00 IST` final release.
- `.aidlc/` — AWS AI-DLC v2 methodology artifacts (`intent.md`, `units_of_work.md`, `domain_model.md`, `adr_log.md`, `verification_and_ops.md`).
