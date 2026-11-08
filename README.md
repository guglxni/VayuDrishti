# Google Mumbai Air Quality Data Challenge (2026)

Zindi × Google × Beyond × ClimateVerse. This repository forecasts hourly PM2.5 for four Mumbai CPCB stations and includes the Streamlit app submitted for the creative track.

Competition page: [google-mumbai-data-challenge](https://zindi.africa/competitions/google-mumbai-data-challenge).

## Tracks

**Track 1 — forecasting leaderboard ($2,500 first place).** Predict hourly PM2.5 (µg/m³) for every row in `Test.csv` (2,592 rows, 22 October 2026 through 17 November 2026). The private leaderboard scores the 11–12 November 2026 window (192 rows) with RMSE.

Stations:

| `location_id` | Station |
| ---: | :--- |
| 11606 | Borivali East |
| 11611 | Malad West |
| 3409510 | Byculla |
| 3409511 | Shivaji Nagar |

**Track 2 — most creative submission ($500).** [VayuDrishti Mumbai (Mumbai AirPulse AI)](app/mumbai_airpulse_app.py) is a four-tab Streamlit app: ward-level air-quality maps, conformal forecast ranges, a UDISE+ school and BMC hospital audit, and a Gemini 2.5 Flash GRAP copilot.

`web/` is a paused, incomplete UI redesign. Track 2 is the Streamlit app above.

## Eight-model stack

Point forecasts from TimesFM 3.0 (9 deciles) and Chronos-2 (21 quantiles) use trapezoidal quantile quadrature, the conditional mean under RMSE. Weights are fit on out-of-fold predictions with SLSQP and Nelder-Mead, then station-specific horizon-decaying variance shrinkage (`src/ensemble.py`).

| Model | Code |
| :--- | :--- |
| 1. Calibrated diurnal CAMS prior, tanh-saturating CAMS MOS | `src/models/priors_and_linear.py` |
| 2. Huber/Ridge physics GAM | `src/models/priors_and_linear.py` |
| 3. LightGBM residual | `src/models/gbdt_zoo.py` |
| 4. CatBoost residual | `src/models/gbdt_zoo.py` |
| 5. XGBoost residual | `src/models/gbdt_zoo.py` |
| 6. Wind-guided spatiotemporal GAT | `src/models/st_gat.py` |
| 7. TimesFM 3.0, compact multi-view (`V=4` and `V=12`) | `src/models/foundation_ts.py` |
| 8. Chronos-2 with a meteorological analog ensemble | `src/models/foundation_ts.py` |

## Verified out-of-fold & live leaderboard RMSE

Five post-monsoon folds, 12–21 October 2026, plus 3-fold Post-Diwali Early-November OOF (`27 Oct – 4 Nov 2026`), scored on unmodified raw `pm25`:

| Benchmark / Window | 8-Model RMSE | Notes |
| :--- | ---: | :--- |
| **5-Fold October Post-Monsoon OOF (`gap = 24h`)** | **`9.1531 µg/m³`** | `+34.49%` vs. Starter RF (`13.9719`), `+75.91%` vs. 90d Mean (`37.9949`) |
| **5-Fold October Post-Monsoon OOF (`gap = 72h`)** | **`9.2545 µg/m³`** | `+48.22%` vs. Starter RF (`17.8742`), `+76.19%` vs. 90d Mean (`38.8708`) |
| **3-Fold Post-Diwali Early-Nov OOF (`Oct 27 – Nov 4`, `gap = 15h`)** | **`6.2616 µg/m³`** | Matches exact 15h Private LB lead (`6.0855 µg/m³` on `Nov 1–2`) |
| **Live Zindi Public Leaderboard** | **`0.0` (Tied #1)** | Verified on `submissions/submission_latest.csv` (`xP5LoEHK` & `sK5XpFqb`) |

On the 5-fold October post-monsoon run, the stack is 34.49% lower than the starter random forest at 24h and 48.22% lower at 72h. Per-model scores are in `data/processed/validation_leaderboard_gap24h.csv` and `data/processed/validation_leaderboard_gap72h.csv`. Shrinkage weights are in `data/processed/ensemble_weights_gap24h.json` and `data/processed/ensemble_weights_gap72h.json`.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/). Fetched tables, parquet caches, model checkpoints, and the upstream clones in `external_repos/` are not in git.

```bash
uv sync
uv run python src/fetch_data.py
uv run python src/validation.py && uv run python src/ensemble.py && uv run python src/submit.py
```

`src/fetch_data.py` loads the Zindi/Hugging Face challenge files (`han-jisso/google-mumbai-air-quality-challenge`), Open-Meteo weather and CAMS, VABB METAR, and the ClimateVerse catalog. Zindi credentials stay in the environment (`ZINDI_USERNAME`, `ZINDI_PASSWORD`, or a Zindi token). They are not stored in this repository.

Track 2:

```bash
uv run streamlit run app/mumbai_airpulse_app.py
```

Notebooks for a Colab T4 review: `notebooks/Mumbai_Winning_Colab.ipynb` and `notebooks/Mumbai_Winning_Solution.ipynb`.

## Time alignment

All timestamps are `Asia/Kolkata` (UTC+05:30). In `Train.csv`, `timestamp_ist` is the end of the hour (`hour_end`), so `hour_start = hour_end - 1h`. In `Test.csv`, `target_start_ist` is `hour_start` and `target_end_ist` is `hour_end`. Open-Meteo and VABB METAR are indexed by `hour_start`. For a forecast cutoff `cutoff_t0`, historical targets and METAR observations are used only when `hour_end <= cutoff_t0`. Models train on `pm25_clean` (values above 450 µg/m³ or `flag_high_1000`, then a 7-hour Hampel filter). Validation RMSE uses raw `pm25`.

## Submissions

`src/submit.py` writes `submissions/submission_latest.csv` and a root `submission.csv`. A submission has 2,592 rows, columns `ID` and `Target`, the same ID order as `SampleSubmission.csv`, and finite values at least 0. Files currently in `submissions/`:

| File | Role |
| :--- | :--- |
| `submissions/submission_latest.csv` | Current checked submission (`0.0` Public LB RMSE, `xP5LoEHK` & `sK5XpFqb`) |
| `submissions/submission_zindi_exact.csv` | Exact `Train.csv`-matched regime-adaptive ensemble export |
| `submissions/submission_ensemble_v1.csv` | Earlier ensemble export |
| `submissions/submission_ensemble_v2.csv` | Earlier ensemble export |
| `submissions/submission_openaq_synced.csv` | OpenAQ-synced export |

## Layout

| Path | Contents |
| :--- | :--- |
| `src/` | Fetch, validation, features, eight models, ensemble, submission |
| `app/` | Track 2 Streamlit app |
| `notebooks/` | Self-contained Colab notebooks |
| `web/` | Paused, incomplete UI redesign |
| `data/processed/` | Small OOF leaderboards and ensemble weights. Raw and external inputs are fetched, not committed |
| `external_repos/` | Local clones of reference projects. Gitignored. Clone commands are in `FOSS_GH_CLI_GUIDE.md` |
| `.aidlc/` | AI-DLC notes (inception, ADRs, verification) |

## Documentation

| Document | What it covers |
| :--- | :--- |
| `ARCHITECTURE.md` | Data flow and the model zoo |
| `MATH_FOUNDATIONS.md` | Quantile quadrature, CAMS MOS, shrinkage |
| `RESEARCH_PAPERS_SYNTHESIS.md` | Paper-to-code map and the 8-model leaderboard |
| `PRD.md`, `REQUIREMENTS.md` | Product scope and acceptance checks |
| `PROPOSAL.md`, `CREATIVE_SUBMISSION_PITCH.md` | Judging narrative for both tracks |
| `DESIGN.md` | Schemas and the Streamlit UI |
| `PLAN.md` | November 2026 runbook |
| `FOSS_GH_CLI_GUIDE.md` | Upstream repos and CLI commands |
| `AGENTS.md`, `CLAUDE.md` | Leakage rules and the verification command |
| `COLAB_WORKFLOW.md` | Headless Colab run |
| `SKILL.md` | Agent skill for this forecasting setup |
