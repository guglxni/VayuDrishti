# वायुदृष्टि VayuDrishti // Mumbai 3D Atmospheric Digital Twin (`web/`)

**Repository:** [`https://github.com/guglxni/VayuDrishti`](https://github.com/guglxni/VayuDrishti)  
**Companion Streamlit Platform:** [`app/mumbai_airpulse_app.py`](../app/mumbai_airpulse_app.py)

**VayuDrishti (`वायुदृष्टि — VayuDrishti Mumbai`)** is our interactive 3D WebGL / MapLibre GL / Deck.gl / Three.js (`@react-three/fiber`) atmospheric digital twin and civic intelligence platform built for the **Google Mumbai Air Quality Data Challenge (2026)** (*Zindi × Google × Beyond × ClimateVerse*).

---

## 1. Architecture & 4-View Tactical Atmospheric Observatory

All four views are synchronized to a global **960-hour IST (`Asia/Kolkata`) timeline scrubber** (`2026-10-12 00:00 IST` to `2026-11-20 23:00 IST`), spanning the 5-Fold Out-of-Fold Validation window, Public Leaderboard windows, and the **Nov 11–12, 2026 Private Leaderboard evaluation window (`192` rows)**:

1. **View 1 — `City map` (`src/map/MapView.tsx`, `src/map/LayerPanel.tsx`, `src/map/StationCard.tsx`, `src/map/fields.ts`):**
   - `maplibre-gl` 6.16 + `@deck.gl/mapbox` 9.4 3D tilted observatory (`pitch=58°`) with 3D extruded pollution columns across **18 Mumbai stations** (`4 Target + 14 Auxiliary` on `main`, and **34 Greater MMR stations** on `mmr-extended`), dynamic **24 BMC administrative ward** choropleths (`wards-fill`), **UDISE+ school & BMC civic hospital** pins, and **Deonar / Trombay / Mahul** emission plume rings.
   - Synchronized 60fps `TripsLayer` **900-streamline Wind-Advection overlay** driven by Open-Meteo 10m wind vectors and VABB Airport METAR (`u, v`) observations.
   - Directed **`AirPhyNet` station-to-station advection-diffusion flux arcs** (`ArcLayer`, $\text{flux}_{ij}(t) \propto C_i(t)\max(0,\cos(\theta_{\text{wind}} - \theta_{ij}))|U(t)|/d_{ij} - \kappa(C_i - C_j)/d_{ij}^2$).
   - Six tactical camera presets: `From sea`, `North up`, `Shivaji / Deonar`, `Malad West`, `Borivali East`, and `Byculla`.

2. **View 2 — `Air column` (`src/views/column/ColumnView.tsx`, `src/views/column/Scene.tsx`, `src/views/column/ColumnHud.tsx`):**
   - Interactive Three.js / React Three Fiber 3D vertical atmospheric column (`3 km × 3 km × 3,000 m`) with OrbitControls and 4 camera presets (`Isometric`, `Canyon`, `Lid`, `Top`).
   - Renders 3D urban building canyon blocks scaled by **Google Open Buildings V3** (`doi:10.71646/EV3RBJ`) and **Google AlphaEarth 10m Embeddings** (`doi:10.71646/GHSZ7J`), **5,200 aerosol particles** trapped beneath a dynamic translucent **Boundary Layer Height (`PBLH`) inversion ceiling**, directional **3D emission source plume cones**, ground-level **`AirPhyNet` flux vectors**, and a continuous **3D vertical concentration profile curve $C(z)$** along the altitude axis.

3. **View 3 — `Forecast` (`src/views/forecast/ForecastView.tsx`, `src/views/forecast/FanChart.tsx`, `src/views/forecast/TimeMixerSection.tsx`, `src/views/forecast/AccuracySection.tsx`, `src/views/forecast/BenchmarkStrip.tsx`):**
   - **`TimesFM 3.0` (9-Decile)** and **`Chronos-2` (21-Quantile)** split-conformal fan charts alongside all individual models (`M1` CAMS MOS Prior through `M8` `Chronos-2` and the `8-Model Stacked Ensemble`), plus raw uncalibrated CAMS overlay.
   - **RMSE Bayes-Optimal Expectation ($\mathbb{E}[Y\mid X] = \int_0^1 Q(p)\,dp$) vs. MAE Median ($Q_{0.50}$)** trapezoidal quantile quadrature comparison.
   - **`TimeMixer` Multi-Scale Decomposition (`TimeMixerSection.tsx`)** separating the 24h synoptic baseline $L_s(t)$ and $\tanh$-MOS CAMS calibration from the diurnal sea-breeze/traffic wave $D_s(t) = Y_s(t) - L_s(t)$.
   - **Verified Competition Benchmark Strip**: `24h Oct OOF RMSE = 9.1531 µg/m³`, `72h Oct OOF RMSE = 9.2545 µg/m³`, `15h Post-Diwali Early-Nov OOF RMSE = 6.2616 µg/m³` (`6.0855 µg/m³` on `Nov 1–2`), and `0.0000` Public LB (`xP5LoEHK` & `sK5XpFqb`).
   - **2-Stage Hampel Sensor Glitch Inspector** and **`AnEn` Historical Analog Matcher** (`hour_end <= cutoff_t0`).

4. **View 4 — `Ward actions` (`src/views/wards/WardsView.tsx`, `src/views/wards/WardList.tsx`, `src/views/wards/WardProfile.tsx`, `src/views/wards/GrapDrawer.tsx`, `src/ui/CatalogExplorer.tsx`):**
   - Multi-criteria 24-ward risk ranking (`PM2.5 now`, `Nov 11–12 peak`, `Vulnerable pop`, `MahaRERA sites`).
   - Embedded **92-Dataset ClimateVerse Catalog Explorer** (`15` directly integrated DOIs + `77` auxiliary regional datasets).
   - Multi-provider **BMC GRAP Civic Copilot** (`Gemini`, `OpenAI`, `Anthropic`, `Grok`, `OpenRouter`, `LiteLLM`) with zero-latency deterministic physics directives.

---

## 2. Local Development & Build Commands

From `web/`:

```bash
# 1. Start the Vite development server on http://127.0.0.1:5179
npm run dev

# 2. Type-check with TypeScript 7 and build production bundle into web/dist/
npm run typecheck && npm run build

# 3. Preview the production build locally
npm run preview

# 4. Regenerate public/data/*.json from root data/ and submissions/
npm run export-data
```
