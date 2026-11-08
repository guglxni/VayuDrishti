# VayuDrishti Mumbai (Mumbai AirPulse AI) — Finalist Pitch Deck & Most Creative Submission Text

**Competition:** Zindi × Google × Beyond × ClimateVerse — *Google Mumbai Air Quality Data Challenge (2026)*  
**Award Track:** **$500 USD Most Creative Submission Award**  
**Judges:** Google Research / Google India, Zindi, Beyond, and ClimateVerse (XKDR Forum / IIT Bombay)

---

## PART 1: Ready-to-Paste Zindi / Google Form Submission Text

### 1.1 Project Title
**VayuDrishti Mumbai (`Mumbai AirPulse AI`): Physics-Guided Foundation Quantile Forecasting (`TimesFM 3.0` + `Chronos-2`), 100m `AlphaEarth` Wind-Advected Kriging & `Gemini 4 Argon` + LiteLLM Civic GRAP Decision-Support Across Mumbai's 24 BMC Wards**

### 1.2 One-Line Elevator Pitch (Under 200 Characters)
Fusing TimesFM 3.0, AlphaEarth, Open Buildings, Gemini 4 Argon + LiteLLM, and 15 ClimateVerse DOIs into a 100m ward-level Mumbai forecast and BMC GRAP platform.

### 1.3 Executive Abstract (Copy-Paste for Submission Form Description)
Mumbai's post-monsoon air quality crisis (October–February) is driven by a severe **atmospheric regime shift**: when the southwest monsoon withdraws, wet scavenging ceases, nocturnal Planetary Boundary Layer Height (`PBLH`) collapses below `200 m`, and easterly land-breeze drainage (`u10 < 0`) funnels smoldering landfill plumes from the 132-hectare **Deonar Dumping Ground (`doi:10.71646/MWQNR8`)**, industrial stack emissions from **Trombay (`doi:10.71646/VA5BD9`)**, and fugitive dust from **1,700+ MahaRERA construction sites (`doi:10.71646/4ULQRE`)** across dense residential wards before the afternoon **Arabian Sea breeze (`u10 > 0`)** restores ventilation.

While the core forecasting benchmark evaluates 4 CPCB stations (`11611 Malad West`, `3409511 Shivaji Nagar`, `11606 Borivali East`, `3409510 Byculla`), **13+ million citizens, 3,800+ UDISE+ schools (`doi:10.71646/XJIY8W`), and 160+ BMC civic hospitals (`doi:10.71646/1XSUVR`)** sit in unmonitored micro-environments across Mumbai's **24 BMC Administrative Wards (`doi:10.71646/EREDWQ`)**.

**VayuDrishti Mumbai (`Mumbai AirPulse AI`)** bridges the last mile between machine learning and municipal climate action through four breakthroughs:
1. **Probabilistic Foundation Model Quantile Fan & $L_2$-Optimal Expectation**: Integrates **Google Research `TimesFM 3.0` (`google/timesfm-3.0-pytorch`, 9-decile `q10..q90`)** and **Amazon `Chronos-2` (`amazon/chronos-2`, 21-quantile `q05..q95`)** with residual GBDTs and Wind-Guided Spatiotemporal Graph Attention (`ST-GAT`). Instead of using the MAE-optimal median $Q_{0.50}$—which systematically under-predicts right-skewed nocturnal inversions—we integrate the full quantile fan via trapezoidal quadrature to recover the **RMSE-optimal conditional expectation $\hat{\mu}_{\text{L2}} = \mathbb{E}[Y\mid X] = \int_0^1 Q_\tau(Y\mid X)\,d\tau$**, cutting post-monsoon 72h-gap OOF RMSE from `38.8708` (90-day reference) and `17.8742` (Starter RF) down to **`9.5401 µg/m³`** (`+46.63%` versus the starter random forest).
2. **`TimeMixer` Two-Scale Wave Decomposition, `AirPhyNet` Plume Attribution & `AnEn` Analog Matcher**: Decomposes hourly station $\text{PM}_{2.5}$ into a **Macroscopic 24h Synoptic Regime Wave $L_s(t)$** ($\tanh$-saturating CAMS MOS + 24h mean PBLH) and a **Microscopic 24h Diurnal Wave $D_s(t)$** (morning/evening traffic rush, Arabian Sea onshore breeze clearing, and `AirPhyNet` Deonar/Thane Creek land-breeze advection-diffusion), backed by a **5-Hour Historical Meteorological Analog (`AnEn`) Matcher (`2024–2026`)**.
3. **100m Anisotropic Wind-Advected + Google `AlphaEarth` Kriging**: Fuses all **18 Mumbai CAAQMS stations (4 Target + 14 Auxiliary, `doi:10.71646/YAQTMA`)** with directional wind vectors (`doi:10.71646/XXOBKO`), **Google `AlphaEarth` 10m surface roughness (`doi:10.71646/GHSZ7J`)**, and **Google `Open Buildings V3` 2.5D street-canyon density (`doi:10.71646/EV3RBJ`)** across all 24 BMC wards.
4. **Google `Gemini 4 Argon` + Universal LiteLLM Civic Action Copilot**: Tab 4 routes ward-specific **BMC Graded Response Action Plan (GRAP)** directives through LiteLLM (flagship `gemini/gemini-4-argon`, with Gemini 3.0, Gemini 2.5, OpenRouter, Vertex AI, OpenAI, Anthropic, xAI Grok, and local Ollama selectable in the UI) and falls back to native `google-genai` for newly released Gemini slugs. With no API key, a deterministic physics engine still schedules STP tanker water sprinkling during `05:30–09:30 IST` inversions, enforces MahaRERA mist-cannon compliance, dispatches Deonar thermal drone patrols, and issues UDISE+ morning assembly suspension alerts.

### 1.4 The 5-Pillar Google AI Stack (Form Field: "How did you use Google Tools?")
1. **Google Research `TimesFM 3.0` (`google/timesfm-3.0-pytorch`)**: Provides zero-shot multi-horizon 9-decile (`q10..q90`) probabilistic fan forecasts and trapezoidal $L_2$ conditional expectations across the 72-hour forecast horizon and Nov 11–12 Private Evaluation Window.
2. **Google `Gemini 4 Argon` (`gemini/gemini-4-argon`) + Universal LiteLLM (`google-genai` discovery)**: Powers the Tab 4 Civic Action Copilot. Officers pick the provider, model slug, API key, and optional API base in the UI (or via `GEMINI_API_KEY`, `COPILOT_MODEL`, `COPILOT_API_BASE`, and the other provider keys). The copilot synthesizes live `PBLH`, `Ventilation Coefficient`, `TimesFM 3.0` and `Chronos-2` quantiles, Deonar advection cosine, and `AlphaEarth` urban morphology into operational BMC GRAP orders with inline ClimateVerse DOI citations.
3. **Google `AlphaEarth Foundations` 10m Satellite Embeddings (`doi:10.71646/GHSZ7J`)**: Supplies 64-D 10m surface roughness, vegetative canopy (Sanjay Gandhi National Park vs. urban core), and imperviousness priors for our 100m anisotropic kriging engine.
4. **Google `Open Buildings V3` (`doi:10.71646/EV3RBJ`)**: Quantifies 0.5m building footprint density and 2.5D street-canyon trapping across Mumbai's 24 administrative wards.
5. **Google `Colab` T4 GPU & `Cloud Run` Deployment Stack**: Deterministic (`SEED = 42`), leak-free pipeline verified for reproducible execution on Google Colab and one-command deployment on Google Cloud Run.

### 1.5 All 15 Integrated ClimateVerse Datasets (Form Field: "Which ClimateVerse datasets did you use?")
1. `doi:10.71646/YAQTMA` — Mumbai Hourly Air Quality & CPCB CAAQMS Archive (4 Target + 14 Auxiliary Stations)
2. `doi:10.71646/XXOBKO` — Open-Meteo Historical Forecast & High-Resolution NWP Meteorology (`PBLH`, `u10`, `v10`, `Temp`, `RH`)
3. `doi:10.71646/EAE4EE` — Chhatrapati Shivaji Maharaj Intl Airport (`VABB`) Half-Hourly METAR Archive
4. `doi:10.71646/GHSZ7J` — Google AlphaEarth Foundations 10m Annual Satellite Embeddings
5. `doi:10.71646/EV3RBJ` — Google Open Buildings V3 Polygons & 2.5D Urban Morphology
6. `doi:10.71646/EREDWQ` — Brihanmumbai Municipal Corporation (`BMC`) 24 Administrative Wards
7. `doi:10.71646/AAGK0G` — WorldPop 2026 Gridded Population & Age-Stratified Vulnerability (`<5y` & `>60y`)
8. `doi:10.71646/MWQNR8` — Mumbai Solid Waste Landfills (`Deonar`, `Mulund`, `Kanjurmarg` Polygons)
9. `doi:10.71646/VA5BD9` — Sentinel-5P TROPOMI & Waste Sector Methane / Smoke Plume Attribution
10. `doi:10.71646/XJIY8W` — Mumbai `UDISE+` 2024–25 Georeferenced School Directory & Enrollment
11. `doi:10.71646/1XSUVR` — `MyBMC` Georeferenced Civic Hospitals, Maternity Homes & Dispensaries
12. `doi:10.71646/YJIAGQ` — NASA FIRMS VIIRS 375m & MODIS Active Fire / Thermal Anomalies
13. `doi:10.71646/4ULQRE` — `MahaRERA` Registered Real-Estate & Construction Projects in MMR
14. `doi:10.71646/MKJGDY` — Maharashtra Official Public Holidays, Festival Calendar & Traffic Regimes
15. `doi:10.71646/CAMS26` — ECMWF CAMS Global & Regional Atmospheric Composition Forecasts

---

## PART 2: 5-Minute Finalist Pitch Deck (Slide-by-Slide Script & Visual Blueprint)

### Slide 1 (`0:00 – 0:40`) — Title & The Post-Monsoon Mumbai Paradox
- **Slide Headline:** **VayuDrishti Mumbai (`Mumbai AirPulse AI`): From 4 Sensors to 24 BMC Wards & 13 Million Citizens**
- **Visual on Screen:** Split-screen showing (Left) Mumbai's `3.5×–4.5×` post-monsoon $\text{PM}_{2.5}$ jump in October–November as the SW Monsoon withdraws and nocturnal `PBLH` collapses below `200 m`; (Right) Live VayuDrishti 100m anisotropic kriging map across Mumbai's 24 BMC wards.
- **Speaker Script (40 sec):**
  > "Namaste and hello to the Google, Zindi, Beyond, and ClimateVerse judges. Every October, as the southwest monsoon withdraws from Mumbai, average $\text{PM}_{2.5}$ surges four-fold from `16 µg/m³` to over `85 µg/m³`, with nocturnal spikes exceeding `180 µg/m³` at Shivaji Nagar. Standard models trained on 90-day monsoon windows fail catastrophically (`38.9 RMSE`), and worse, 4 point-forecasts leave Mumbai's 24 municipal wards, 3,800 UDISE+ schools, and 160 civic hospitals flying blind. We built **VayuDrishti Mumbai (`Mumbai AirPulse AI`)**—powered by a 5-pillar Google AI stack and 15 ClimateVerse DOIs—to solve both the multi-horizon forecasting problem and hyperlocal civic action."

### Slide 2 (`0:40 – 1:30`) — Data Integrity & The Monsoon-to-Post-Monsoon Regime Shift
- **Slide Headline:** **Solving Sensor Glitches (`241,587 µg/m³`) & Non-Stationary Regime Shift**
- **Visual on Screen:** Tab 2 Hampel Sensor Glitch Inspector table (`flag_high_1000`, `241,587.6 µg/m³` at Shivaji Nagar, `75,203.5 µg/m³` at Borivali East) + 5-Fold Expanding-Window Post-Monsoon CV diagram (`72h` gap matching Nov 8 cutoff $\to$ Nov 11–12 Private Window).
- **Speaker Script (50 sec):**
  > "First, raw CPCB telemetry (`doi:10.71646/YAQTMA`) contains severe hardware glitches up to `241,587 µg/m³` that blow up squared-error loss. Our 2-stage Hampel median-absolute-deviation filter cleans 140+ telemetry spikes while preserving true Diwali and landfill smoke peaks. Second, because Zindi evaluates a 72-hour blind jump from the November 8 cutoff to the November 11–12 Private Window, we designed a strict 5-fold expanding-window post-monsoon CV with a 72-hour purge gap and regime-aware sample weighting (`3.5×` post-monsoon vs `0.20×` monsoon wash-out)."

### Slide 3 (`1:30 – 2:30`) — Breakthrough #1: `TimesFM 3.0` + `Chronos-2` Quantile Fans & Why RMSE Needs $\mathbb{E}[Y\mid X] > Q_{0.50}$
- **Slide Headline:** **Probabilistic Foundation Models: Trapezoidal $L_2$ Expectation Beats the Median**
- **Visual on Screen:** Tab 2 `Google Research TimesFM 3.0` 9-decile (`q10..q90`) & `Amazon Chronos-2` 21-quantile (`q05..q95`) fan chart + Mathematical callout card comparing $\hat{\mu}_{\text{L2}} = \int_0^1 Q_\tau\,d\tau$ (blue solid) against $Q_{0.50}$ (crimson dashed).
- **Speaker Script (60 sec):**
  > "Here is our core statistical breakthrough. During calm nocturnal boundary-layer inversions (`PBLH < 250 m`), urban $\text{PM}_{2.5}$ follows a heavily right-skewed log-normal distribution. Foundation time-series models like **Google Research `TimesFM 3.0`** and **Amazon `Chronos-2`** output quantiles centered on the conditional median $Q_{0.50}$, which minimizes MAE ($L_1$ loss) but systematically under-predicts nocturnal smog spikes by `4 to 8 µg/m³` under RMSE ($L_2$ loss). By integrating the full 9-decile `TimesFM 3.0` and 21-quantile `Chronos-2` fan via trapezoidal quadrature, $\hat{\mu}_{\text{L2}} = \int_0^1 Q_\tau(Y\mid X)\,d\tau$, and stacking with physics-residual GBDTs and Wind-Guided `ST-GAT`, we cut 72h-gap OOF RMSE from `17.8742` (Starter RF) to **`9.5401 µg/m³`**."

### Slide 4 (`2:30 – 3:25`) — Breakthrough #2: `TimeMixer` Wave Decomposition, `AirPhyNet` Plume Attribution & `AnEn` Analog Matcher
- **Slide Headline:** **Inside the Black Box: Synoptic vs. Diurnal Waves & Historical Analog Matching (`2024–2026`)**
- **Visual on Screen:** Tab 2 `TimeMixer` Two-Scale Wave Decomposition chart ($L_s(t)$ + $D_s(t)$ partitioned into Traffic Rush, Arabian Sea Breeze Clearing, and `AirPhyNet` Deonar Land-Breeze Plume) + Top-3 Historical Analog (`AnEn`) Matcher table.
- **Speaker Script (55 sec):**
  > "Municipal commissioners don't just need a number—they need physical attribution. Using our **`TimeMixer` Two-Scale Wave Decomposition**, VayuDrishti separates any station's forecast into a **Macroscopic 24h Synoptic Regime Wave $L_s(t)$**—driven by $\tanh$-saturating CAMS MOS and 24-hour mean PBLH—and a **Microscopic 24h Diurnal Wave $D_s(t)$**. Using **`AirPhyNet` advection-diffusion physics**, we partition the diurnal wave into morning/evening vehicular rush, afternoon Arabian Sea breeze clearing (`u10 > 0`), and nocturnal Deonar/Thane Creek land-breeze smoke drainage (`u10 < 0`). Below that, our **Historical Meteorological Analog (`AnEn`) Matcher** retrieves the top-3 post-monsoon dates from `2024–2026` whose 5-hour `(PBLH, Wind Vector, RH, CAMS)` trajectory best matches the active forecast hour."

### Slide 5 (`3:25 – 4:15`) — Breakthrough #3: 100m `AlphaEarth` + `Open Buildings V3` Kriging & School/Hospital Audit
- **Slide Headline:** **Hyperlocal 100m Kriging Across 24 BMC Wards, UDISE+ Schools & Civic Hospitals**
- **Visual on Screen:** Tab 1 Anisotropic Kriging Surface + Tab 3 BMC 24-Ward Vulnerability Matrix (`doi:10.71646/EREDWQ`, `AAGK0G`, `XJIY8W`, `1XSUVR`).
- **Speaker Script (50 sec):**
  > "To scale from 4 target sensors to all 24 BMC wards, we fuse the **14 auxiliary Mumbai CAAQMS stations** (`Deonar`, `Chembur`, `Sewri`, `BKC`, `Worli`, `Kurla`, `Airport T2`, `Colaba`, `Powai`, `Mulund`, `Ghatkopar`, `Mindspace Malad`) into a physics-informed anisotropic wind-advection kriging solver. We weight upwind stations along the wind vector (`doi:10.71646/XXOBKO`, `EAE4EE`) and adjust for street-canyon trapping using **Google `AlphaEarth` 10m embeddings (`doi:10.71646/GHSZ7J`)** and **Google `Open Buildings V3` (`doi:10.71646/EV3RBJ`)**. Tab 3 immediately flags high-risk **UDISE+ schools** and **BMC civic hospitals** like Shatabdi Hospital in M-East Ward and KEM Hospital in F-South Ward when local $\text{PM}_{2.5}$ crosses India's `60 µg/m³` NAAQS threshold."

### Slide 6 (`4:15 – 4:50`) — Breakthrough #4: Google `Gemini 4 Argon` + LiteLLM Civic Action Copilot & 15 ClimateVerse DOIs
- **Slide Headline:** **Automated BMC GRAP Operational Directives via Google `Gemini 4 Argon` + Universal LiteLLM**
- **Visual on Screen:** Tab 4 router panel defaulting to `gemini/gemini-4-argon`, with provider, custom model ID, API key, and live model discovery, generating an M-East Ward GRAP order + the 15-DOI ClimateVerse Dataverse Explorer.
- **Speaker Script (35 sec):**
  > "Finally, Tab 4 routes the Civic Action Copilot through **Google `Gemini 4 Argon`** on a Universal LiteLLM router—with native `google-genai` fallback and live model discovery—and a 100% deterministic offline policy engine. A ward officer picks the model, pastes a key or points at OpenRouter, Vertex, or a local Ollama endpoint, and gets exact `05:30–09:30 IST` STP water-sprinkling schedules, compliance notices for active `MahaRERA` construction sites (`doi:10.71646/4ULQRE`), thermal drone patrols for `Deonar` landfill (`doi:10.71646/MWQNR8`, `VA5BD9`), and morning assembly advisories for `UDISE+` schools—every recommendation citing its persistent ClimateVerse DOI."

### Slide 7 (`4:50 – 5:00`) — Summary & Impact Call-to-Action
- **Slide Headline:** **VayuDrishti Mumbai: Production-Verified, Reproducible & Ready for BMC Deployment**
- **Visual on Screen:** 5-Pillar Google AI badge row + 15 ClimateVerse DOI pills + `AppTest` 0-exception verification badge.
- **Speaker Script (10 sec):**
  > "VayuDrishti Mumbai unites 5 Google AI pillars, 15 ClimateVerse datasets, and rigorous atmospheric physics into a production-tested civic platform. Thank you!"

---

## PART 3: 90-Second Live Interactive Demo Script for Judges

1. **`0:00 – 0:20` (Sidebar & Tab 1 — 100m Kriging Map):**
   - Select **`Private Evaluation Window (Nov 11–12, 2026)`** and slide `target_start_ist` to **`Nov 11, 08:00 IST`** (morning inversion peak).
   - Point out the top KPI cards (`4 Target CPCB Mean PM2.5`, `Peak Hotspot Ward M/E`, `PBLH & Ventilation Coefficient`, and `Population Exceeding 24h NAAQS`).
   - Hover over `★ Shivaji Nagar (BMC)` and the `Deonar Dumping Ground` plume on the Plotly Anisotropic Surface, then toggle to `Folium Interactive GIS`.
2. **`0:20 – 0:55` (Tab 2 — `TimesFM 3.0` + `Chronos-2` Fan, $\mathbb{E}[Y\mid X]$ vs. $Q_{0.50}$, `TimeMixer` & `AnEn`):**
   - Switch Probabilistic Quantile Fan Engine between **`Google TimesFM 3.0 (9-Decile q10..q90)`** and **`Amazon Chronos-2 (21-Quantile q05..q95)`**.
   - Point to the **RMSE-Optimal Trapezoidal Expectation $\mathbb{E}[Y\mid X]$ (blue solid)** sitting above the **MAE-Optimal Median $Q_{0.50}$ (crimson dashed)** during nocturnal low-PBLH windows, and highlight the mathematical callout card.
   - Scroll down to the **`TimeMixer` Two-Scale Synoptic vs. Diurnal Wave Decomposition & `AirPhyNet` Plume Attribution** chart and the **Top-3 Historical Meteorological Analog (`AnEn`) Matcher (`2024–2026`)** card.
3. **`0:55 – 1:15` (Tab 3 — Vulnerable Wards, UDISE+ Schools & BMC Hospitals):**
   - Inspect the 24 BMC Wards bar chart and filter the High-Risk Facilities table to `Ward M/E` and `Ward E` to view automated protocols for `Shivaji Nagar BMC MPS School No. 1` and `Pt. Madan Mohan Malaviya Shatabdi Hospital`.
4. **`1:15 – 1:30` (Tab 4 — Google `Gemini 4 Argon` + LiteLLM Copilot & 15-DOI ClimateVerse Explorer):**
   - Select `Ward M/E — Govandi / Shivaji Nagar / Deonar` and click **Generate BMC GRAP Civic Directive**, then type `"AlphaEarth"` or `"Deonar"` into the interactive ClimateVerse Dataverse Explorer.
