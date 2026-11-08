# VayuDrishti // Mumbai 3D Atmospheric Digital Twin (`web/`)

**Status:** paused, incomplete redesign. The Track 2 submission is the Streamlit app `app/mumbai_airpulse_app.py`.

**VayuDrishti Mumbai (`vayudrishti-3d-twin`)** is the interactive 3D WebGL / MapLibre / Three.js atmospheric digital twin and civic intelligence platform built for the **Google Mumbai Air Quality Data Challenge (2026)** (*Zindi × Google × Beyond × ClimateVerse*).

---

## 1. Architecture & 4-Tab Interactive Views

All views are synchronized to a global **960-hour IST (`Asia/Kolkata`) timeline scrubber** (`Oct 12 – Nov 20, 2026`), spanning the 5-Fold Out-of-Fold Validation window, Public Leaderboard window, and **Nov 11–12 Private Leaderboard evaluation window**:

1. **Tab 1 — 3D Mumbai Spatial Digital Twin (`src/components/SpatialTwinMap.tsx`):**
   - `maplibre-gl` 3D tilted map (`pitch=60°`) with 3D extruded pollution beacons across all **18 Mumbai stations** (`4 Target + 14 Auxiliary`), **24 BMC administrative ward** choropleth extrusions, **342 UDISE+ school & civic hospital** pins, and **Deonar / Trombay / Mahul** emission plume rings.
   - Synchronized 60fps HTML5 Canvas **2,500-particle Wind-Advection Streamline overlay** driven by Open-Meteo 10m wind vectors and VABB Airport METAR (`u, v`) observations.
   - One-click camera fly-to presets: `Mumbai Overview`, `Shivaji Nagar / Deonar`, `Malad West`, `Borivali East`, and `Byculla`.

2. **Tab 2 — Interactive Three.js 3D Boundary-Layer Column & `AirPhyNet` Graph (`src/components/BoundaryLayerScene.tsx`):**
   - Interactive WebGL `THREE.Scene` with pointer drag orbit/rotation and scroll zoom.
   - Renders a 3D vertical atmospheric column above the selected station: 3D urban building canyon blocks scaled by **Google Open Buildings V3** (`doi:10.71646/EV3RBJ`) and **AlphaEarth 10m Embeddings** (`doi:10.71646/GHSZ7J`), **1,200 glowing 3D aerosol particles** trapped beneath a dynamic translucent 3D **Boundary Layer Height (`PBLH`) inversion ceiling** (`~140m` nocturnal collapse vs `~1,200m` afternoon sea-breeze mixing), 3D wind vector arrows, and directed inter-station **`AirPhyNet` advection-diffusion flux tubes**.

3. **Tab 3 — 72-Hour Multi-Horizon Telemetry Scrubber & Foundation Model Lab (`src/components/TelemetryLab.tsx`):**
   - **`TimesFM 3.0` (9-Decile)** and **`Chronos-2` (21-Quantile)** split-conformal fan charts alongside all individual models (`M1` CAMS MOS Prior through `M6` `AirPhyNet Wind-ST-GAT` and the `8-Model Stacked Ensemble`).
   - **RMSE Bayes-Optimal Expectation (`E[Y|X]`) vs MAE Median (`Q50`)** trapezoidal quantile quadrature comparison.
   - **`TimeMixer` Wave Decomposer** separating the 24h synoptic plume envelope $L_s(t)$ from the diurnal sea-breeze/traffic oscillation $D_s(t)$.
   - **2-Stage Hampel Sensor Glitch Inspector** highlighting extreme CPCB raw telemetry glitches (up to **`241,587.6 µg/m³`** at Shivaji Nagar `3409511` and **`75,203.5 µg/m³`** at Borivali East `11606`) vs cleaned training series.
   - **`AnEn` Historical Analog Time-Machine** matching multivariate post-monsoon meteorological signatures strictly within `hour_end <= cutoff_t0`.

4. **Tab 4 — Configurable Frontier AI Civic GRAP Copilot & 92-Dataset ClimateVerse Catalog (`src/components/CivicCopilotAndCatalog.tsx`):**
   - Multi-provider router supporting **`Google Gemini 4 Argon`**, **`OpenAI GPT-6.2 / GPT-6.1`**, **`Anthropic Claude Opus 5.5`**, **`xAI Grok 4.7`**, **OpenRouter**, and **LiteLLM** endpoints, plus zero-latency **instant physics-computed BMC GRAP operational directives** for all 24 wards.
   - Searchable **92-dataset ClimateVerse India Dataverse catalog** (`15` directly integrated DOIs + `77` auxiliary regional datasets).

---

## 2. Local Development & Build Commands

From `/Volumes/MacExt/Projects/google-mumbai-air-quality/web/`:

```bash
# 1. Start the Vite development server (with optional /api/gemini dev proxy)
npm run dev

# 2. Type-check with TypeScript 7 and build production bundle into web/dist/
npm run build

# 3. Preview the production build locally
npm run preview

# 4. (Optional) Regenerate public/data/*.json from root data/ and submissions/
npm run export-data
```

To enable the local Vite dev proxy for Gemini without exposing your key in the browser bundle, set `GEMINI_API_KEY` in your shell or `web/.env.local` before running `npm run dev`.

---

## 3. Static & Cloud Deployment (`web/dist/`)

Running `npm run build` outputs a self-contained static bundle in `web/dist/` (including `web/public/data/*.json`):

- **Vercel / Netlify / Cloudflare Pages:** Set root directory to `web`, build command to `npm run build`, and output directory to `dist`.
- **GitHub Pages / Subpath Hosting:** Build with `VITE_BASE=/your-repo-name/ npm run build` so `fetch(\`${import.meta.env.BASE_URL}data/...\`)` resolves relative to your subpath.
