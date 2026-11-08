# VayuDrishti Mumbai (Mumbai AirPulse AI) — Most Creative Submission Guide

**Competition:** Zindi × Google × Beyond × ClimateVerse — *Google Mumbai Air Quality Data Challenge (2026)*  
**Award Track:** **$500 USD Most Creative Submission** (Judged by Google, Zindi & ClimateVerse on Impact, Novelty, Functionality, Google AI Stack & ClimateVerse Dataset Integration)  
**Application Entrypoint:** `app/mumbai_airpulse_app.py`  
**Finalist Pitch Deck & Form Text:** `CREATIVE_SUBMISSION_PITCH.md`

---

## 1. Executive Summary & Real-World Civic Impact

Mumbai's post-monsoon air quality crisis (October–February) is governed by a sharp **atmospheric regime shift**: as the southwest monsoon withdraws, wet scavenging stops, nocturnal Planetary Boundary Layer Height (`PBLH`) collapses below `200 m`, and morning **land-breeze drainage (`u10 < 0`)** transports landfill smoke (`Deonar`), petrochemical plumes (`Trombay/Mahul`), and construction dust (`MahaRERA` corridors) across dense residential wards before the afternoon **Arabian Sea breeze (`u10 > 0`)** restores marine ventilation.

While the core Zindi leaderboard evaluates 4 target CPCB stations (`11611 Malad West - IITM`, `3409511 Shivaji Nagar - BMC`, `11606 Borivali East - IITM`, `3409510 Byculla - BMC`), **13+ million citizens across Mumbai's 24 Brihanmumbai Municipal Corporation (BMC) Administrative Wards** live, attend school, and seek hospital care in unmonitored micro-environments between stations.

**VayuDrishti Mumbai (`Mumbai AirPulse AI`)** transforms our multi-horizon physics-guided foundation model ensemble into a live, ward-level civic decision-support platform featuring four integrated interactive modules:

1. **Tab 1 — Hyperlocal 100m Wind-Advected Kriging & AlphaEarth Urban Morphology Map**:
   - Fuses **4 Target CPCB Stations + 14 Auxiliary Mumbai CAAQMS Stations** (`doi:10.71646/YAQTMA`) with directional wind transport vectors (`u10, v10`, `doi:10.71646/XXOBKO`), **Google AlphaEarth Foundations 10m Surface Roughness (`doi:10.71646/GHSZ7J`)**, **Google Open Buildings V3 2.5D Street-Canyon Density (`doi:10.71646/EV3RBJ`)**, and **Deonar/Trombay Gaussian Plume Dispersion (`doi:10.71646/MWQNR8`)** across all **24 BMC Wards (`doi:10.71646/EREDWQ`)**.
2. **Tab 2 — 72-Hour Foundation Quantile Fan (`TimesFM 3.0` + `Chronos-2`), `TimeMixer`/`AirPhyNet` Wave Decomposition & `AnEn` Analog Matcher**:
   - **Probabilistic Quantile Fan Charts**: Interactive selector switching between **Google Research `TimesFM 3.0` (`google/timesfm-3.0-pytorch`, 9-decile `q10..q90`)**, **Amazon `Chronos-2` (`amazon/chronos-2`, 21-quantile `q05..q95`)**, and **Split-Conformal (`80%` & `95%`)** uncertainty envelopes.
   - **RMSE-Optimal Distributional Mean ($\mathbb{E}[Y\mid X]$) vs. MAE-Optimal Median ($Q_{0.50}$)**: Interactive toggle and mathematical callout demonstrating why right-skewed nocturnal inversions (`PBLH < 250 m`) require trapezoidal integration over the full quantile fan ($\hat{\mu}_{\text{L2}} = \int_0^1 Q_\tau(Y\mid X)\,d\tau$) rather than median point predictions ($Q_{0.50}$).
   - **`TimeMixer` Two-Scale Synoptic vs. Diurnal Wave Decomposition & `AirPhyNet` Plume Attribution**: Decomposes hourly $\text{PM}_{2.5}$ into the **Macroscopic 24h Synoptic Regime Wave $L_s(t)$** ($\tanh$-saturating CAMS MOS + 24h mean PBLH) and the **Microscopic 24h Diurnal Wave $D_s(t)$** (partitioned into morning/evening traffic rush, Arabian Sea onshore breeze clearing, and `AirPhyNet` Deonar/Thane Creek land-breeze advection-diffusion plumes).
   - **Historical Meteorological Analog (`AnEn`) 5-Hour Trajectory Matcher (`2024–2026`)**: Matches the active 5-hour `(PBLH, Wind Vector, RH, CAMS)` trajectory against historical post-monsoon dates in Mumbai (`2024–2026`) using the Delle Monache (2013) Analog Ensemble metric.
   - **2-Stage Hampel Sensor Glitch Inspector (`flag_high_1000`)**: Audits and cleans extreme CPCB telemetry hardware spikes (e.g., `241,587.6 µg/m³` at Shivaji Nagar and `75,203.5 µg/m³` at Borivali East).
3. **Tab 3 — Vulnerable Population, UDISE+ School & BMC Civic Hospital Exposure Audit**:
   - Cross-references ward-level $\text{PM}_{2.5}$ against **WorldPop 2026 pediatric (`<5y`) and geriatric (`>60y`) demographics (`doi:10.71646/AAGK0G`)**, **Mumbai UDISE+ 2024–25 schools (`doi:10.71646/XJIY8W`)**, and **MyBMC civic hospitals (`doi:10.71646/1XSUVR`)** (`Shatabdi Govandi`, `KEM Parel`, `JJ Byculla`, `Bhabha Kurla`, `M.W. Desai Malad`, `Bhagwati Borivali`).
4. **Tab 4 — Google Gemini 4 Argon + Universal LiteLLM Civic Action Copilot & ClimateVerse Dataverse Explorer**:
   - Generates ward-specific **BMC Graded Response Action Plan (GRAP)** operational directives through a 3-tier router: **LiteLLM** (`litellm.completion`, flagship **Google Gemini 4 Argon** / `gemini/gemini-4-argon`, plus Gemini 3.0, Gemini 2.5, OpenRouter, Vertex AI, OpenAI, Anthropic, xAI Grok, and local Ollama), a native **`google-genai`** fallback with live `models.list()` discovery, and a deterministic physics-grounded offline engine. Paired with an interactive search engine across all **15 integrated ClimateVerse DOIs**.

---

## 2. Quickstart & Verification Commands

From the repository root (`/Volumes/MacExt/Projects/google-mumbai-air-quality`):

```bash
# 1. Launch the interactive Streamlit platform (http://localhost:8501)
uv run streamlit run app/mumbai_airpulse_app.py

# 2. Optional: Enable live Gemini 4 Argon (or any LiteLLM model) generation
export GEMINI_API_KEY="your-google-ai-studio-key"
export COPILOT_MODEL="gemini/gemini-4-argon"   # optional; this is the default
export COPILOT_PROVIDER="Google Gemini (Native + LiteLLM)"  # optional
# export OPENROUTER_API_KEY="..."  OPENAI_API_KEY="..."  ANTHROPIC_API_KEY="..."  XAI_API_KEY="..."
# export COPILOT_API_BASE="https://openrouter.ai/api/v1"   # optional custom endpoint
uv run streamlit run app/mumbai_airpulse_app.py

# 3. Run automated headless end-to-end verification via Streamlit AppTest
/Volumes/MacExt/Projects/google-mumbai-air-quality/.venv/bin/python -c '
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/mumbai_airpulse_app.py").run(timeout=30)
assert len(at.exception) == 0, f"Exceptions: {at.exception}"
print("AppTest passed with 0 exceptions!")
'
```

---

## 3. The 5-Pillar Google AI Stack

| Pillar | Google Technology | Scientific & Civic Role in VayuDrishti Mumbai |
| :--- | :--- | :--- |
| **Pillar 1** | **Google Research `TimesFM 3.0` (`google/timesfm-3.0-pytorch`)** | Pretrained patched-decoder time-series foundation model generating zero-shot 72-hour 9-decile (`q10..q90`) probabilistic fan forecasts and trapezoidal $L_2$ expectations $\hat{\mu}_{\text{L2}}$ across all 4 target CPCB stations (`app/physics_engine.py`, Tab 2). |
| **Pillar 2** | **Google `Gemini 4 Argon` (`gemini/gemini-4-argon`) + Universal LiteLLM** | Multi-model Civic Action Copilot (`app/gemini_copilot.py`, Tab 4). LiteLLM routes Gemini 4 Argon, Gemini 3.0, Gemini 2.5, OpenRouter, Vertex, OpenAI, Anthropic, xAI, and Ollama; `google-genai` discovers live models and falls back on new Gemini slugs. Offline, the same tab emits a deterministic BMC GRAP from `PBLH`, ventilation, TimesFM 3.0 & Chronos-2 quantiles, Deonar advection cosine, UDISE+ schools, and civic hospitals. |
| **Pillar 3** | **Google `AlphaEarth Foundations` 10m Satellite Embeddings (`doi:10.71646/GHSZ7J`)** | 64-D 10m annual learned surface representations encoding vegetative canopy (Sanjay Gandhi National Park at `Borivali East`) vs. impervious urban canyons (`Byculla` & `Shivaji Nagar`) inside the 100m anisotropic wind-advected kriging surface (`app/physics_engine.py`, Tab 1). |
| **Pillar 4** | **Google `Open Buildings V3` (`doi:10.71646/EV3RBJ`)** | 0.5m footprint polygons and 2.5D urban morphology quantifying street-canyon recirculation and nocturnal particulate trapping across all 24 BMC administrative wards (`app/spatial_catalog.py`, Tabs 1 & 3). |
| **Pillar 5** | **Google `Colab` T4 GPU & `Cloud Run` Architecture** | Self-contained, deterministic (`SEED = 42`) pipeline verified for single-notebook execution on Google Colab T4 GPUs (for Zindi Top-10 verification) and stateless containerization on Google Cloud Run. |

---

## 4. All 15 Integrated ClimateVerse Datasets (With Persistent DOIs)

Every external dataset integrated into **VayuDrishti Mumbai** is sourced from the official **ClimateVerse Google Air Quality Forecasting Challenge Dataverse (`https://india.climateverse.net/dataverse/google-air-quality-forecasting-challenge`)**:

| # | ClimateVerse DOI | Dataset Title | Role in Forecasting & VayuDrishti Decision Support |
| :---: | :--- | :--- | :--- |
| **1** | **`doi:10.71646/YAQTMA`** | **Mumbai Hourly Air Quality & CPCB CAAQMS Archive (2018–2026)** | Primary target $\text{PM}_{2.5}$ series (`Train.csv`, 4 target stations) + 14 auxiliary Mumbai CAAQMS stations (`Other_Mumbai_Stations_2025_2026.csv.gz`) for Hampel cleaning and cross-station upwind advection lags. |
| **2** | **`doi:10.71646/XXOBKO`** | **Open-Meteo Historical Forecast & High-Resolution NWP Meteorology** | Hourly `boundary_layer_height` (`PBLH`), `u10`, `v10`, `wind_speed_10m`, `wind_direction_10m`, `temperature_2m`, `relative_humidity_2m`, and `Ventilation Coefficient (PBLH × WindSpeed)`. |
| **3** | **`doi:10.71646/EAE4EE`** | **Chhatrapati Shivaji Maharaj Intl Airport (VABB) Half-Hourly METAR** | Ground-truth synoptic wind speed (`sknt`), direction (`drct`), temperature, dewpoint depression, and altimeter pressure (`alti`), strictly excluding visibility/wxcodes for zero target leakage. |
| **4** | **`doi:10.71646/GHSZ7J`** | **Google AlphaEarth Foundations 10m Annual Satellite Embeddings** | 10m learned surface roughness, vegetation fraction, and impervious surface priors for 100m anisotropic kriging. |
| **5** | **`doi:10.71646/EV3RBJ`** | **Google Open Buildings V3 Polygons & 2.5D Urban Morphology** | Quantifies building footprint density and street-canyon recirculation across Mumbai's 24 BMC wards. |
| **6** | **`doi:10.71646/EREDWQ`** | **BMC 24 Administrative Wards & Demographic Boundaries** | Spatial aggregation polygons & centroids (`Wards A` through `T`) for population-weighted $\text{PM}_{2.5}$ and GRAP civic orders. |
| **7** | **`doi:10.71646/AAGK0G`** | **WorldPop 2026 High-Resolution Gridded Age-Stratified Population** | Weights ward-level $\text{PM}_{2.5}$ concentrations by vulnerable pediatric (`<5y`) and geriatric (`>60y`) cohorts. |
| **8** | **`doi:10.71646/MWQNR8`** | **Mumbai Solid Waste Landfills: Deonar, Mulund & Kanjurmarg Polygons** | Models nocturnal land-breeze Gaussian smoke/particulate drainage from the 132-hectare Deonar Dumping Ground into `Shivaji Nagar (3409511)`. |
| **9** | **`doi:10.71646/VA5BD9`** | **Sentinel-5P TROPOMI & Waste Sector Methane / Smoke Attribution** | Triggers automated Deonar subsurface smoldering & thermal infrared drone patrol alerts in M-East Ward. |
| **10** | **`doi:10.71646/XJIY8W`** | **Mumbai UDISE+ 2024–25 Georeferenced School Directory** | Computes school-hour (`07:30–13:30 IST`) child exposure and morning assembly suspension alerts across 3,800+ schools. |
| **11** | **`doi:10.71646/1XSUVR`** | **MyBMC Georeferenced Civic Hospitals, Maternity Homes & Dispensaries** | Forecasts respiratory OPD surge risk and HEPA filtration protocols for `Shatabdi`, `KEM`, `JJ`, `Bhabha`, `M.W. Desai`, and `Bhagwati` civic hospitals. |
| **12** | **`doi:10.71646/YJIAGQ`** | **NASA FIRMS VIIRS 375m & MODIS Active Fire / Thermal Anomalies** | Detects peri-urban biomass burning, Thane/Bhiwandi brick kilns, and landfill thermal hotspots. |
| **13** | **`doi:10.71646/4ULQRE`** | **MahaRERA Registered Real-Estate & Construction Projects in MMR** | Targets BMC GRAP Stage-II/III anti-smog mist gun and wheel-washing compliance notices in Malad, Byculla, and Worli. |
| **14** | **`doi:10.71646/MKJGDY`** | **Maharashtra Official Public Holidays & Festival Calendar** | Captures post-monsoon festival emissions and weekend commercial vehicle regime shifts. |
| **15** | **`doi:10.71646/CAMS26`** | **ECMWF CAMS Global & Regional Atmospheric Composition Forecasts** | Provides forward chemical boundary conditions (`PM2.5`, `PM10`, `NO2`, `AOD550`) with $\tanh$-saturating Model Output Statistics across the Nov 11–12 Private Window. |

---

## 5. Modular Codebase Structure (`app/`)

- `app/mumbai_airpulse_app.py` — 4-tab Streamlit application featuring Plotly anisotropic kriging surfaces, Folium GIS maps, `TimesFM 3.0` + `Chronos-2` probabilistic quantile fan charts, `TimeMixer` / `AirPhyNet` wave decomposition, `AnEn` historical trajectory matcher, and BMC ward exposure matrices.
- `app/physics_engine.py` — Data ingestion, 2-stage Hampel sensor glitch filter (`flag_high_1000` + `500 µg/m³` cap), log-normal 9-decile/21-quantile fan & trapezoidal $L_2$ expectation solver, `TimeMixer` synoptic/diurnal wave decomposer, `AirPhyNet` advection-diffusion attribution, `AnEn` 5-hour analog trajectory matcher, and anisotropic wind-advected kriging engine.
- `app/spatial_catalog.py` — Georeferenced catalog of all 18 Mumbai CAAQMS stations, 24 BMC Administrative Wards, 6 major emission hotspots, 10 flagship UDISE+ schools & BMC civic hospitals, and all 15 ClimateVerse DOIs.
- `app/gemini_copilot.py` — Gemini 4 Argon flagship plus Universal LiteLLM router, `google-genai` live model discovery, and a deterministic physics-grounded BMC GRAP policy fallback.
