# AI-DLC Phase 1 (Inception): System Intent & Non-Functional Requirements

**Lifecycle Stage:** `Inception -> Intent & Requirements Elaboration`  
**Status:** `APPROVED`  
**Last Updated:** `2026-11-07`

---

## 1. High-Level Intent Statement

Deliver a **scientifically rigorous, leak-free, deterministic dual-track system** for the **Google × Zindi × Beyond × ClimateVerse Mumbai Air Quality Data Challenge (2026)** that:
1. **Wins Track 1 (Main Forecasting Leaderboard — $2,500 1st Place):** Achieves the lowest Root Mean Squared Error ($\text{RMSE}$) across the 4 target Mumbai CPCB stations (`11606` Borivali East, `11611` Malad West, `3409510` Byculla, `3409511` Shivaji Nagar) during both rolling public windows and the decisive **2026-11-11 to 2026-11-12 Private Evaluation Window (`192` rows within `2,592` test rows)**, while passing Zindi's 48-hour Top-10 Code Review on a free Google Colab T4 GPU without a single modification.
2. **Wins Track 2 (Most Creative Submission — $500 Special Award):** Transforms station-level point forecasts into **VayuDrishti Mumbai (`Mumbai AirPulse AI`)** (`app/mumbai_airpulse_app.py`), an interactive 24-Ward civic decision-support platform fusing **15 official ClimateVerse DOIs** and **4 Google technologies** (`TimesFM 3.0`, `AlphaEarth Foundations 10m`, `Open Buildings V3`, `Gemini 2.5 Flash`) to protect 13+ million citizens, 3,800+ UDISE+ schools, and BMC civic hospitals.

---

## 2. Measurable Success Outcomes (KPIs)

| ID | Outcome Metric | Baseline (Starter RF / 90d Mean) | Target Threshold | Achieved / Verified |
| :--- | :--- | :---: | :---: | :---: |
| **`KPI-1`** | 5-Fold Post-Monsoon OOF RMSE (`24h` Reporting Gap, $N=960$) | `13.9719` / `37.9949 µg/m³` | `< 11.00 µg/m³` | **`10.6835 µg/m³`** (`+23.54%` vs RF) |
| **`KPI-2`** | 5-Fold Post-Monsoon OOF RMSE (`72h` Reporting Gap, $N=960$) | `17.8742` / `38.8708 µg/m³` | `< 10.80 µg/m³` | **`10.3254 µg/m³`** (`+42.23%` vs RF) |
| **`KPI-3`** | Fold 4 (`Oct 18–19` CAMS Dust False Alarm) Resilience | `16.013 µg/m³` (Linear CAMS) | `< 13.50 µg/m³` | **`13.332 µg/m³`** (Stacked + MOS) |
| **`KPI-4`** | Zindi Submission Schema & Assertion Compliance | `2,592` rows | `6 / 6` assertions | **`6 / 6` (`submission_latest.csv`)** |
| **`KPI-5`** | ClimateVerse & Google Ecosystem Integration | `1` dataset | $\ge 10$ DOIs + $\ge 3$ Google tools | **15 DOIs + 4 Google Tools** |

---

## 3. Non-Functional Requirements (NFRs) & Guardrails

1. **`NFR-CAUSAL` (Zero Temporal Leakage):** Every historical target aggregation (`Train.csv`, `Other_Mumbai_Stations_2025_2026.csv.gz`) and airport observation (`VABB METAR`) must strictly satisfy `hour_end <= cutoff_t0`. VABB visibility and weather codes (`wxcodes`) are permanently banned.
2. **`NFR-DETERMINISM` (Bitwise & Statistical Reproducibility):** All random seeds (`PYTHONHASHSEED=42`, `np.random.seed(42)`, `torch.manual_seed(42)`, `deterministic=True` in LightGBM) must be fixed so Top-10 Code Review auditors reproduce our exact submission within floating-point tolerance (`< 1e-4`).
3. **`NFR-COMPUTE` (Colab T4 GPU Budget):** Full end-to-end execution in `notebooks/Mumbai_Winning_Colab.ipynb` must complete in `< 20 minutes` on a single NVIDIA T4 GPU (`16 GB` VRAM) and `< 3 minutes` for the CPU tabular + PyTorch ST-GAT core (`src/submit.py`).
4. **`NFR-RESILIENCE` (Offline-First Creative App):** `app/mumbai_airpulse_app.py` must run 100% offline using cached Parquet/CSV artifacts and the deterministic physics-grounded BMC GRAP engine when `GEMINI_API_KEY` or internet access is unavailable.
