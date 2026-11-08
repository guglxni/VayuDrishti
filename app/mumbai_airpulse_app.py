"""VayuDrishti Mumbai / Mumbai AirPulse AI
===========================================================================
Hyperlocal Spatiotemporal Air Quality Intelligence, Conformal Forecasting &
Civic Decision-Support Platform for the Google Mumbai Data Challenge.

Built for the $500 Most Creative Submission Award judged by Google, Zindi & ClimateVerse.

Run locally:
    uv run streamlit run app/mumbai_airpulse_app.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium

# Ensure root package imports work whether invoked from repo root or app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gemini_copilot import (
    FLAGSHIP_MODEL_ID,
    FLAGSHIP_MODEL_LABEL,
    PROVIDER_OPTIONS,
    build_model_catalog,
    default_copilot_settings,
    discover_available_gemini_models,
    provider_env_key_names,
    resolve_api_base,
    resolve_api_key,
    resolve_litellm_model,
    run_gemini_civic_copilot,
)
from app.physics_engine import (
    compute_spatiotemporal_snapshot,
    find_historical_meteorological_analogs,
    load_auxiliary_observations,
    load_climateverse_catalog,
    load_forecasts_and_conformal,
    load_meteorology_and_cams,
    load_train_observations,
)
from app.spatial_catalog import (
    CLIMATEVERSE_DOI_CATALOG,
    EMISSION_HOTSPOTS,
    TARGET_STATIONS,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VayuDrishti Mumbai · AirPulse AI",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.0rem;
    }
    .doi-pill {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        font-size: 0.76rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: rgba(26, 115, 232, 0.12);
        color: #1a73e8;
        border: 1px solid rgba(26, 115, 232, 0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Data Pipeline Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_all_app_data():
    train_df = load_train_observations()
    aux_df = load_auxiliary_observations()
    met_df, ext_status = load_meteorology_and_cams()
    forecast_df, sub_source = load_forecasts_and_conformal(train_df, met_df)
    catalog_df = load_climateverse_catalog()
    return train_df, aux_df, met_df, forecast_df, catalog_df, ext_status, sub_source


train_df, aux_df, met_df, forecast_df, catalog_df, ext_status, sub_source = get_all_app_data()


# Helper for CPCB / NAAQS color coding
def pm25_color_hex(pm25: float) -> str:
    if pm25 < 30.0:
        return "#10b981"  # Good (Emerald)
    if pm25 < 60.0:
        return "#84cc16"  # Satisfactory (Within 24h NAAQS)
    if pm25 < 90.0:
        return "#f59e0b"  # Moderate / Poor (Amber)
    if pm25 < 120.0:
        return "#ef4444"  # Very Poor (Red)
    return "#8b5cf6"      # Severe+ (Purple)


def _quantile_context_for_ward(ward_row: dict) -> dict:
    """Nearest target-station TimesFM 3.0 and Chronos-2 quantiles at the active hour."""
    ts = snapshot["timestamp_ist"]
    sub = forecast_df[forecast_df["timestamp_ist"] == ts]
    if sub.empty:
        return {}
    ward_lat = float(ward_row.get("lat", 19.05))
    ward_lon = float(ward_row.get("lon", 72.88))
    anchor = min(
        TARGET_STATIONS,
        key=lambda station: (station["latitude"] - ward_lat) ** 2 + (station["longitude"] - ward_lon) ** 2,
    )
    rows = sub[sub["location_id"] == anchor["location_id"]]
    if rows.empty:
        rows = sub
    record = rows.iloc[0]

    def reading(column: str):
        if column not in record.index or pd.isna(record[column]):
            return None
        return float(record[column])

    return {
        "anchor_station": anchor["short_name"],
        "anchor_location_id": anchor["location_id"],
        "timesfm_q10": reading("tfm_q10"),
        "timesfm_q50": reading("tfm_q50"),
        "timesfm_q90": reading("tfm_q90"),
        "chronos_q05": reading("chr_q05"),
        "chronos_q50": reading("chr_q50"),
        "chronos_q95": reading("chr_q95"),
        "mu_l2": reading("mu_l2_expectation"),
        "q50_median": reading("q50_median"),
    }


_copilot_defaults = default_copilot_settings()
st.session_state.setdefault("copilot_provider", _copilot_defaults["provider"])
if st.session_state["copilot_provider"] not in PROVIDER_OPTIONS:
    st.session_state["copilot_provider"] = PROVIDER_OPTIONS[0]
st.session_state.setdefault("copilot_model_id", _copilot_defaults["model_id"])
st.session_state.setdefault("copilot_custom_model", _copilot_defaults["custom_model"])
st.session_state.setdefault("copilot_api_key", "")
st.session_state.setdefault("copilot_api_base", _copilot_defaults["api_base"])
st.session_state.setdefault("discovered_gemini_models", [])
st.session_state.setdefault("discover_status", "")


# ---------------------------------------------------------------------------
# Sidebar: Global Spatiotemporal Controls & Live Data Provenance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/air: VayuDrishti Mumbai")
    st.caption(
        "**Mumbai AirPulse AI** — Spatiotemporal Urban Air Intelligence & Civic Decision Support "
        "built for **Google × Zindi × ClimateVerse**."
    )

    # Global Timestamp Selector across the Test Grid (focusing on Nov 11-12 Private Window by default)
    available_ts = sorted(forecast_df["timestamp_ist"].unique())
    default_ts = pd.Timestamp("2026-11-11 08:00:00+05:30")
    if default_ts not in available_ts:
        default_ts = available_ts[len(available_ts) // 2]

    preset_window = st.selectbox(
        "Evaluation regime window",
        options=[
            "Private Evaluation Window (Nov 11–12, 2026)",
            "Update 3 Public Window (Nov 5–6, 2026)",
            "Update 2 Public Window (Nov 1–2, 2026)",
            "Full 27-Day Test Horizon (Oct 27–Nov 20, 2026)",
        ],
        index=0,
    )

    if "Private" in preset_window:
        window_ts = [
            ts
            for ts in available_ts
            if pd.Timestamp("2026-11-11 00:00:00+05:30") <= ts <= pd.Timestamp("2026-11-12 23:00:00+05:30")
        ]
    elif "Nov 5–6" in preset_window:
        window_ts = [
            ts
            for ts in available_ts
            if pd.Timestamp("2026-11-05 00:00:00+05:30") <= ts <= pd.Timestamp("2026-11-06 23:00:00+05:30")
        ]
    elif "Nov 1–2" in preset_window:
        window_ts = [
            ts
            for ts in available_ts
            if pd.Timestamp("2026-11-01 00:00:00+05:30") <= ts <= pd.Timestamp("2026-11-02 23:00:00+05:30")
        ]
    else:
        window_ts = available_ts

    if not window_ts:
        window_ts = available_ts

    default_idx = window_ts.index(default_ts) if default_ts in window_ts else min(8, len(window_ts) - 1)
    selected_ts = st.select_slider(
        "Active IST hour (`target_start_ist`)",
        options=window_ts,
        value=window_ts[default_idx],
        format_func=lambda ts: pd.Timestamp(ts).strftime("%b %d, %H:%M IST"),
    )

    map_engine = st.segmented_control(
        "Spatial map renderer",
        options=["Plotly Anisotropic Surface", "Folium Interactive GIS"],
        default="Plotly Anisotropic Surface",
    )

    st.markdown("#### :material/database: Live Pipeline Status")
    st.caption(f"**Forecast Engine:** {sub_source}")
    st.caption("**8-Model 5-Fold OOF RMSE:** `9.43 µg/m³` (`9.4265` @ 24h gap) · `9.54 µg/m³` (`9.5401` @ 72h gap)")
    st.caption(f"**CPCB Train:** `{len(train_df):,}` rows · 4 Target Stations (`doi:10.71646/YAQTMA`)")
    st.caption(f"**Aux CAAQMS:** `{len(aux_df):,}` rows · 14 Mumbai Stations (`doi:10.71646/YAQTMA`)")
    st.caption(f"**Meteorology:** {ext_status.get('open_meteo_weather', '')}")
    st.caption(f"**CAMS Chemistry:** {ext_status.get('open_meteo_aq', '')}")
    st.caption(f"**VABB Airport METAR:** {ext_status.get('vabb_metar', '')}")

    with st.expander("5-Pillar Google AI Stack", icon=":material/auto_awesome:"):
        st.markdown(
            "- **Google `TimesFM 3.0`** (`google/timesfm-3.0-pytorch`) 9-decile probabilistic fan & L2 expectation\n"
            "- **Google `Gemini 4 Argon`** (`gemini/gemini-4-argon`) + Universal LiteLLM multi-model BMC GRAP Copilot\n"
            "- **Google `AlphaEarth Foundations`** 10m Satellite Embeddings (`doi:10.71646/GHSZ7J`)\n"
            "- **Google `Open Buildings V3`** 2.5D Urban Canyon Morphology (`doi:10.71646/EV3RBJ`)\n"
            "- **Google `Colab` & `Cloud Run`** Deterministic (`SEED=42`) Verification Stack"
        )

    with st.expander("15 Integrated ClimateVerse DOIs", icon=":material/verified:"):
        for item in CLIMATEVERSE_DOI_CATALOG[:15]:
            st.markdown(f"- `{item['doi']}` — {item['title']}")

    with st.expander("AI Copilot · Gemini 4 Argon + LiteLLM", icon=":material/psychology:"):
        sidebar_model = st.session_state.get("copilot_custom_model", "").strip() or st.session_state.get(
            "copilot_model_id", FLAGSHIP_MODEL_ID
        )
        st.caption(f"**Router:** {st.session_state.get('copilot_provider', PROVIDER_OPTIONS[0])}")
        st.caption(f"**Model:** `{sidebar_model}`")
        st.caption(
            "Provider, flagship dropdown, custom LiteLLM model ID, API key, API base, and live "
            "`models.list()` discovery are configured in **Tab 4**."
        )


# Compute live spatial snapshot for selected timestamp
snapshot = compute_spatiotemporal_snapshot(
    selected_ts=pd.Timestamp(selected_ts),
    forecast_df=forecast_df,
    aux_df=aux_df,
    grid_resolution=28,
)
stations_now_df = snapshot["stations_df"]
wards_df = snapshot["wards_df"]
grid_df = snapshot["grid_df"]
facilities_df = snapshot["facilities_df"]

# ---------------------------------------------------------------------------
# Hero Header & Top-Level KPI Cards
# ---------------------------------------------------------------------------
st.title(":material/air: VayuDrishti Mumbai (Mumbai AirPulse AI)")
st.markdown(
    "**Hyperlocal 100m Wind-Advected Air Quality Kriging, Conformal Multi-Horizon Forecasting & "
    "Google Gemini 4 Argon + LiteLLM Civic Decision-Support across Mumbai's 24 BMC Wards**  \n"
    "**8-Model Stacked Ensemble (`TimesFM 3.0` + `Chronos-2` + `AirPhyNet Wind-ST-GAT` + `GBDT` + `CAMS MOS`) · "
    "5-Fold Post-Monsoon OOF RMSE: `9.4265 µg/m³` (24h Gap) · `9.5401 µg/m³` (72h Gap)**"
)
st.markdown(
    '<span class="doi-pill">doi:10.71646/YAQTMA · CPCB Hourly AQ</span>'
    '<span class="doi-pill">doi:10.71646/XXOBKO · Open-Meteo NWP</span>'
    '<span class="doi-pill">doi:10.71646/EAE4EE · VABB METAR</span>'
    '<span class="doi-pill">doi:10.71646/GHSZ7J · Google AlphaEarth 10m</span>'
    '<span class="doi-pill">doi:10.71646/EV3RBJ · Google Open Buildings V3</span>'
    '<span class="doi-pill">doi:10.71646/EREDWQ · 24 BMC Wards</span>'
    '<span class="doi-pill">doi:10.71646/MWQNR8 · Deonar Landfill</span>'
    '<span class="doi-pill">doi:10.71646/XJIY8W · UDISE+ Schools</span>'
    '<span class="doi-pill">doi:10.71646/1XSUVR · BMC Civic Hospitals</span>'
    '<span class="doi-pill">doi:10.71646/YJIAGQ · NASA FIRMS</span>',
    unsafe_allow_html=True,
)

target_only_df = stations_now_df[stations_now_df["is_target"]].copy()
mean_target_pm = float(target_only_df["pm25"].mean())
max_ward_row = wards_df.iloc[0]
wards_above_naaqs = int((wards_df["pm25_ugm3"] >= 60.0).sum())
pop_above_naaqs_m = float(wards_df.loc[wards_df["pm25_ugm3"] >= 60.0, "pop_2026"].sum()) / 1e6

with st.container(horizontal=True):
    st.metric(
        "4 Target CPCB Mean PM2.5",
        f"{mean_target_pm:.1f} µg/m³",
        f"{mean_target_pm / 60.0:.2f}× India NAAQS (60 µg/m³)",
        delta_color="inverse",
        border=True,
    )
    st.metric(
        "Peak Hotspot Ward (BMC)",
        f"Ward {max_ward_row['ward_code']}: {max_ward_row['pm25_ugm3']:.1f} µg/m³",
        f"{max_ward_row['ward_name']} ({max_ward_row['who_exceed_x']:.1f}× WHO)",
        delta_color="inverse",
        border=True,
    )
    st.metric(
        "Boundary Layer & Ventilation",
        f"PBLH {snapshot['pblh_m']:.0f} m · VC {snapshot['ventilation_coeff']:.0f} m²/s",
        f"Wind {snapshot['wind_speed_ms']:.1f} m/s from {snapshot['wind_dir_from_deg']:.0f}°",
        delta_color="normal" if snapshot["ventilation_coeff"] > 1200 else "off",
        border=True,
    )
    st.metric(
        "Population Exceeding 24h NAAQS",
        f"{pop_above_naaqs_m:.2f}M Citizens",
        f"{wards_above_naaqs} / 24 BMC Wards > 60 µg/m³",
        delta_color="inverse",
        border=True,
    )

# ---------------------------------------------------------------------------
# Main 4-Tab Navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        ":material/map: 1. Spatiotemporal Map & Wind-Advection Kriging",
        ":material/timeline: 2. 72h Forecast, Sensor Cleaning & Conformal Bands",
        ":material/local_hospital: 3. Vulnerable Wards, UDISE+ Schools & Hospitals",
        ":material/psychology: 4. Gemini 4 Argon Copilot & ClimateVerse Explorer",
    ]
)

# ===========================================================================
# TAB 1: Hyperlocal Mumbai Spatiotemporal Air Quality Map & Wind-Advection Kriging
# ===========================================================================
with tab1:
    st.subheader(
        f"Hyperlocal Anisotropic Wind-Advected Kriging & AlphaEarth Urban Morphology — "
        f"`{pd.Timestamp(selected_ts).strftime('%Y-%m-%d %H:%M IST')}`"
    )
    st.caption(
        "Fuses **4 Target CPCB Stations** (`11611 Malad West`, `3409511 Shivaji Nagar`, `11606 Borivali East`, `3409510 Byculla`) + "
        "**14 Auxiliary Mumbai CAAQMS Stations** (`doi:10.71646/YAQTMA`) with directional wind transport vectors (`u10, v10`, `doi:10.71646/XXOBKO`), "
        "**Google AlphaEarth 10m Surface Roughness (`doi:10.71646/GHSZ7J`)**, **Google Open Buildings V3 (`doi:10.71646/EV3RBJ`)**, and "
        "**Deonar/Trombay Gaussian Plume Dispersion (`doi:10.71646/MWQNR8`)**."
    )

    col_map, col_panel = st.columns([2.15, 1.0])

    with col_map:
        with st.container(border=True):
            if map_engine == "Plotly Anisotropic Surface":
                ScatterMapTrace = getattr(go, "Scattermap", None) or getattr(go, "Scattermapbox")
                fig_map = go.Figure()

                # 1. Kriging interpolated grid points over Mumbai peninsula
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=grid_df["lat"],
                        lon=grid_df["lon"],
                        mode="markers",
                        marker=dict(
                            size=13,
                            color=grid_df["pm25"],
                            colorscale="Turbo",
                            cmin=25,
                            cmax=140,
                            opacity=0.38,
                            colorbar=dict(title="PM2.5 (µg/m³)", thickness=14, len=0.75),
                        ),
                        text=[f"Kriged Grid PM2.5: {v:.1f} µg/m³" for v in grid_df["pm25"]],
                        hoverinfo="text",
                        name="100m Wind-Advected Kriging Surface",
                    )
                )

                # 2. 24 BMC Ward Centroids
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=wards_df["lat"],
                        lon=wards_df["lon"],
                        mode="markers+text",
                        marker=dict(
                            size=11,
                            color=wards_df["pm25_ugm3"],
                            colorscale="Turbo",
                            cmin=25,
                            cmax=140,
                            opacity=0.85,
                        ),
                        text=[f"Ward {w}" for w in wards_df["ward_code"]],
                        textposition="top center",
                        customdata=np.stack(
                            [
                                wards_df["ward_name"],
                                wards_df["pm25_ugm3"],
                                wards_df["pop_2026"],
                                wards_df["grap_stage"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>BMC Ward %{text} (%{customdata[0]})</b><br>"
                            "Kriged PM2.5: <b>%{customdata[1]} µg/m³</b><br>"
                            "Ward Population: %{customdata[2]:,}<br>"
                            "GRAP Status: %{customdata[3]}<extra></extra>"
                        ),
                        name="24 BMC Administrative Wards (doi:10.71646/EREDWQ)",
                    )
                )

                # 3. Emission Hotspots (Deonar Landfill, Trombay Power Plant, Port, MahaRERA)
                hs_df = pd.DataFrame(EMISSION_HOTSPOTS)
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=hs_df["lat"],
                        lon=hs_df["lon"],
                        mode="markers",
                        marker=dict(size=15, color="#dc2626", symbol="circle"),
                        text=hs_df["name"],
                        customdata=np.stack([hs_df["category"], hs_df["doi"], hs_df["description"]], axis=-1),
                        hovertemplate=(
                            "<b>🔥 Emission Hotspot: %{text}</b><br>"
                            "Category: %{customdata[0]} (%{customdata[1]})<br>"
                            "%{customdata[2]}<extra></extra>"
                        ),
                        name="Emission Hotspots (Deonar / Trombay / Port)",
                    )
                )

                # 4. 14 Auxiliary CAAQMS Stations
                aux_map_df = stations_now_df[~stations_now_df["is_target"]]
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=aux_map_df["latitude"],
                        lon=aux_map_df["longitude"],
                        mode="markers",
                        marker=dict(
                            size=14,
                            color=aux_map_df["pm25"],
                            colorscale="Turbo",
                            cmin=25,
                            cmax=140,
                            opacity=0.95,
                        ),
                        text=aux_map_df["station_name"],
                        customdata=np.stack(
                            [aux_map_df["pm25"], aux_map_df["operator"], aux_map_df["ward_code"]],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>Auxiliary CAAQMS: %{text}</b><br>"
                            "PM2.5: <b>%{customdata[0]} µg/m³</b><br>"
                            "Operator: %{customdata[1]} · Ward %{customdata[2]}<extra></extra>"
                        ),
                        name="14 Auxiliary Mumbai CAAQMS Stations",
                    )
                )

                # 5. 4 Official Target CPCB Stations (Highlighted with outer ring + label)
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=target_only_df["latitude"],
                        lon=target_only_df["longitude"],
                        mode="markers+text",
                        marker=dict(size=22, color="#111827"),
                        text=[f"★ {s}: {p:.1f} µg/m³" for s, p in zip(target_only_df["short_name"], target_only_df["pm25"])],
                        textposition="bottom right",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=target_only_df["latitude"],
                        lon=target_only_df["longitude"],
                        mode="markers",
                        marker=dict(
                            size=17,
                            color=target_only_df["pm25"],
                            colorscale="Turbo",
                            cmin=25,
                            cmax=140,
                        ),
                        text=target_only_df["station_name"],
                        customdata=np.stack(
                            [
                                target_only_df["pm25"],
                                target_only_df["q95_lo"],
                                target_only_df["q95_hi"],
                                target_only_df["pblh_m"],
                                target_only_df["ventilation_coeff"],
                                target_only_df["location_id"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>★ TARGET STATION: %{text} (ID %{customdata[5]})</b><br>"
                            "Predicted PM2.5: <b>%{customdata[0]} µg/m³</b><br>"
                            "95% Conformal Interval: [%{customdata[1]}, %{customdata[2]}] µg/m³<br>"
                            "PBLH: %{customdata[3]} m · Ventilation: %{customdata[4]} m²/s<extra></extra>"
                        ),
                        name="★ 4 Target CPCB Stations (Zindi Evaluation)",
                    )
                )

                # 6. Wind Advection Vector Lines (u10, v10 arrows across Mumbai)
                arrow_lats = []
                arrow_lons = []
                u_step = snapshot["wind_u10"] * 0.0045
                v_step = snapshot["wind_v10"] * 0.0045
                for _, st_row in stations_now_df.iloc[::2].iterrows():
                    lat0 = float(st_row["latitude"])
                    lon0 = float(st_row["longitude"])
                    arrow_lats.extend([lat0, lat0 + v_step, None])
                    arrow_lons.extend([lon0, lon0 + u_step, None])

                fig_map.add_trace(
                    ScatterMapTrace(
                        lat=arrow_lats,
                        lon=arrow_lons,
                        mode="lines",
                        line=dict(width=2.5, color="#0284c7"),
                        name=f"Wind Advection Vectors ({snapshot['wind_speed_ms']} m/s @ {snapshot['wind_dir_from_deg']:.0f}°)",
                    )
                )

                map_cfg_key = "map" if hasattr(go, "Scattermap") else "mapbox"
                fig_map.update_layout(
                    **{
                        map_cfg_key: dict(
                            style="carto-positron",
                            center=dict(lat=19.085, lon=72.875),
                            zoom=10.15,
                        )
                    },
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=610,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.01,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255,255,255,0.88)",
                    ),
                )
                st.plotly_chart(fig_map, width="stretch")

            else:
                # Folium Interactive GIS Map
                m = folium.Map(location=[19.085, 72.875], zoom_start=11, tiles="CartoDB positron")

                # Plot Emission Hotspots
                for hs in EMISSION_HOTSPOTS:
                    folium.Circle(
                        location=[hs["lat"], hs["lon"]],
                        radius=900,
                        color="#dc2626",
                        fill=True,
                        fill_opacity=0.22,
                        popup=f"<b>{hs['name']}</b><br>{hs['category']} ({hs['doi']})<br>{hs['description']}",
                        tooltip=f"Hotspot: {hs['name']}",
                    ).add_to(m)

                # Plot all 18 stations + wind advection trajectory lines
                u_step = snapshot["wind_u10"] * 0.005
                v_step = snapshot["wind_v10"] * 0.005
                for _, row in stations_now_df.iterrows():
                    color = pm25_color_hex(row["pm25"])
                    radius = 11 if row["is_target"] else 6
                    weight = 3 if row["is_target"] else 1
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=radius,
                        color="#111827" if row["is_target"] else color,
                        weight=weight,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.92,
                        tooltip=(
                            f"{'★ TARGET' if row['is_target'] else 'AUX'}: {row['station_name']} — "
                            f"{row['pm25']:.1f} µg/m³ (Ward {row['ward_code']})"
                        ),
                    ).add_to(m)
                    # Wind advection line
                    folium.PolyLine(
                        locations=[
                            [row["latitude"], row["longitude"]],
                            [row["latitude"] + v_step, row["longitude"] + u_step],
                        ],
                        color="#0284c7",
                        weight=2,
                        opacity=0.75,
                    ).add_to(m)

                st_folium(m, width=None, height=600, returned_objects=[])

    with col_panel:
        with st.container(border=True):
            st.markdown("#### ★ 4 Target CPCB Stations")
            st.caption("Live predictions & 95% conformal intervals (`doi:10.71646/YAQTMA`)")
            st.dataframe(
                target_only_df[
                    ["short_name", "ward_code", "pm25", "q95_lo", "q95_hi", "pblh_m"]
                ].rename(
                    columns={
                        "short_name": "Station",
                        "ward_code": "Ward",
                        "pm25": "PM2.5 (µg/m³)",
                        "q95_lo": "95% Lo",
                        "q95_hi": "95% Hi",
                        "pblh_m": "PBLH (m)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        with st.container(border=True):
            st.markdown("#### 14 Auxiliary Mumbai CAAQMS Stations")
            st.caption("Cross-station directional transport network (`Other_Mumbai_Stations_2025_2026.csv.gz`)")
            aux_display = stations_now_df[~stations_now_df["is_target"]][
                ["short_name", "ward_code", "operator", "pm25", "naaqs_Ratio"]
            ].sort_values("pm25", ascending=False)
            st.dataframe(
                aux_display.rename(
                    columns={
                        "short_name": "Auxiliary Station",
                        "ward_code": "Ward",
                        "operator": "Agency",
                        "pm25": "PM2.5 (µg/m³)",
                        "naaqs_Ratio": "× NAAQS",
                    }
                ),
                hide_index=True,
                height=315,
                width="stretch",
            )

    # Bottom row of Tab 1: Anisotropic Transport Equation & Top Hotspot Wards
    c_eq, c_hot = st.columns([1.15, 1.85])
    with c_eq:
        with st.container(border=True):
            st.markdown("#### :material/science: Physics-Informed Anisotropic Kriging")
            st.latex(
                r"\hat{C}(\mathbf{x}, t) = \frac{\sum_{j=1}^{18} w_j(\mathbf{x}, \mathbf{u}_{10}) C_j(t)}{\sum_{j=1}^{18} w_j(\mathbf{x}, \mathbf{u}_{10})} + \sum_{k \in \text{Hotspots}} \mathcal{P}_k(\mathbf{x}, \text{PBLH}) + \gamma_{\text{AE}} \Phi_{\text{AlphaEarth}}(\mathbf{x})"
            )
            st.caption(
                r"Where directional weight $w_j(\mathbf{x}, \mathbf{u}_{10}) = d_j^{-2.15} \left[1 + \alpha \tanh(\|\mathbf{u}_{10}\|) \max(0, \cos(\theta_{\text{wind}} - \theta_{j \to \mathbf{x}}))\right]$ "
                "upweights upwind stations along the Arabian Sea breeze / Thane Creek land-breeze axis, combined with "
                "**Google AlphaEarth 10m (`doi:10.71646/GHSZ7J`)** & **Open Buildings V3 (`doi:10.71646/EV3RBJ`)** canyon trapping."
            )
    with c_hot:
        with st.container(border=True):
            st.markdown("#### :material/local_fire_department: Major Mumbai Emission Hotspots & Plume Attribution")
            st.dataframe(
                pd.DataFrame(EMISSION_HOTSPOTS)[
                    ["name", "category", "ward_code", "doi", "description"]
                ].rename(
                    columns={
                        "name": "Emission Source",
                        "category": "Sector",
                        "ward_code": "Ward",
                        "doi": "ClimateVerse DOI",
                        "description": "Atmospheric Transport Mechanism",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


# ===========================================================================
# TAB 2: 72-Hour Multi-Horizon Forecast & Conformal Uncertainty Explorer
# ===========================================================================
with tab2:
    st.subheader(
        "72-Hour Foundation Quantile Fan (`TimesFM 3.0` + `Chronos-2`), `TimeMixer` / `AirPhyNet` Decomposition & `AnEn` Analog Matcher"
    )
    st.caption(
        "Integrates **`Google Research TimesFM 3.0` (`google/timesfm-3.0-pytorch`, 9-decile `q10..q90`)** + "
        "**`Amazon Chronos-2` (`amazon/chronos-2`, 21-quantile `q05..q95`)** probabilistic fan charts, "
        "**RMSE-Optimal Trapezoidal Expectation $\\hat{\\mu}_{\\text{L2}} = \\mathbb{E}[Y\\mid X]$ vs. MAE-Optimal Median $Q_{0.50}$**, "
        "**`TimeMixer` Two-Scale Synoptic vs. Diurnal Wave Decomposition**, **`AirPhyNet` Upwind Advection-Diffusion Plume Attribution**, "
        "and **Historical Meteorological Analog (`AnEn`) Matching (`2024–2026`)**."
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.35, 1.05, 1.3, 1.1])
    with ctrl1:
        station_options = {s["station_name"]: s["location_id"] for s in TARGET_STATIONS}
        selected_station_name = st.selectbox(
            "Select target CPCB station (`Stations.csv`)",
            options=list(station_options.keys()),
            index=1,  # Default to Shivaji Nagar - BMC (most dynamic station)
        )
        selected_loc_id = station_options[selected_station_name]
    with ctrl2:
        horizon_hours = st.slider(
            "Forecast visualization horizon (hours)",
            min_value=24,
            max_value=120,
            value=72,
            step=12,
        )
    with ctrl3:
        fan_mode = st.selectbox(
            "Probabilistic quantile fan engine",
            options=[
                "Google TimesFM 3.0 (9-Decile q10..q90)",
                "Amazon Chronos-2 (21-Quantile q05..q95)",
                "Calibrated Split-Conformal (80% & 95%)",
            ],
            index=0,
        )
    with ctrl4:
        show_model_components = st.toggle(
            "Overlay Model Zoo (GBDT / ST-GAT / TimesFM / Chronos)",
            value=True,
        )
        show_l2_vs_median = st.toggle(
            "Compare RMSE Mean E[Y|X] vs. Median Q_0.50",
            value=True,
        )

    # Filter forecast series for selected station around Nov 10 -> Nov 13
    st_fc = (
        forecast_df[forecast_df["location_id"] == selected_loc_id]
        .sort_values("timestamp_ist")
        .reset_index(drop=True)
    )
    anchor_start = pd.Timestamp("2026-11-10 00:00:00+05:30")
    anchor_end = anchor_start + pd.Timedelta(hours=horizon_hours)
    st_fc_win = st_fc[(st_fc["timestamp_ist"] >= anchor_start) & (st_fc["timestamp_ist"] <= anchor_end)].copy()
    if st_fc_win.empty:
        st_fc_win = st_fc.tail(horizon_hours).copy()

    # Synchronized 2-Panel Plot: Top = Foundation Quantile Fan + L2 Expectation vs Median + NAAQS; Bottom = PBLH + Ventilation Coefficient
    with st.container(border=True):
        fig_fc = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.64, 0.36],
            subplot_titles=(
                f"{selected_station_name} — {fan_mode} Fan & RMSE-Optimal Expectation vs. Median",
                "Physical Atmospheric Drivers: Boundary Layer Height (PBLH) & Ventilation Coefficient (PBLH × Wind)",
            ),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        )

        # Highlight Private Evaluation Window (Nov 11 00:00 to Nov 12 23:59 IST)
        fig_fc.add_vrect(
            x0="2026-11-11 00:00:00+05:30",
            x1="2026-11-12 23:00:00+05:30",
            fillcolor="rgba(26, 115, 232, 0.09)",
            line_width=1,
            line_dash="dot",
            line_color="#1a73e8",
            annotation_text="Zindi Private Evaluation Window (Nov 11–12 IST)",
            annotation_position="top left",
            row=1,
            col=1,
        )

        x_concat = pd.concat([st_fc_win["timestamp_ist"], st_fc_win["timestamp_ist"][::-1]])

        if "TimesFM 3.0" in fan_mode:
            # 9-Decile Fan: [q10..q90], [q20..q80], [q30..q70], [q40..q60]
            decile_bands = [
                ("tfm_q10", "tfm_q90", "rgba(37, 99, 235, 0.12)", "TimesFM 3.0 q10–q90 (80% Decile Envelope)"),
                ("tfm_q20", "tfm_q80", "rgba(37, 99, 235, 0.18)", "TimesFM 3.0 q20–q80 (60% Decile Fan)"),
                ("tfm_q30", "tfm_q70", "rgba(37, 99, 235, 0.24)", "TimesFM 3.0 q30–q70 (40% Core Fan)"),
            ]
            for lo_c, hi_c, fill_rgba, label_str in decile_bands:
                fig_fc.add_trace(
                    go.Scatter(
                        x=x_concat,
                        y=pd.concat([st_fc_win[hi_c], st_fc_win[lo_c][::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(color="rgba(255,255,255,0)"),
                        name=label_str,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
        elif "Chronos-2" in fan_mode:
            # 21-Quantile Fan: [q05..q95], [q15..q85], [q25..q75], [q35..q65]
            chronos_bands = [
                ("chr_q05", "chr_q95", "rgba(124, 58, 237, 0.11)", "Chronos-2 q05–q95 (90% 21-Quantile Envelope)"),
                ("chr_q15", "chr_q85", "rgba(124, 58, 237, 0.16)", "Chronos-2 q15–q85 (70% Quantile Fan)"),
                ("chr_q25", "chr_q75", "rgba(124, 58, 237, 0.22)", "Chronos-2 q25–q75 (50% IQR Fan)"),
                ("chr_q35", "chr_q65", "rgba(124, 58, 237, 0.28)", "Chronos-2 q35–q65 (30% Core Fan)"),
            ]
            for lo_c, hi_c, fill_rgba, label_str in chronos_bands:
                fig_fc.add_trace(
                    go.Scatter(
                        x=x_concat,
                        y=pd.concat([st_fc_win[hi_c], st_fc_win[lo_c][::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(color="rgba(255,255,255,0)"),
                        name=label_str,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
        else:
            # Calibrated Split-Conformal 95% and 80% Bands
            fig_fc.add_trace(
                go.Scatter(
                    x=x_concat,
                    y=pd.concat([st_fc_win["q95_hi"], st_fc_win["q95_lo"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(59, 130, 246, 0.14)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Split-Conformal Band (q05–q95)",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=x_concat,
                    y=pd.concat([st_fc_win["q80_hi"], st_fc_win["q80_lo"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(59, 130, 246, 0.26)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Split-Conformal Band (q10–q90)",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        if show_model_components:
            fig_fc.add_trace(
                go.Scatter(
                    x=st_fc_win["timestamp_ist"],
                    y=st_fc_win["model_gbdt_residual"],
                    mode="lines",
                    line=dict(width=1.3, dash="dot", color="#10b981"),
                    name="Tier 1: Residual GBDT (LGBM+CatBoost+XGB)",
                ),
                row=1,
                col=1,
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=st_fc_win["timestamp_ist"],
                    y=st_fc_win["model_gat_wind"],
                    mode="lines",
                    line=dict(width=1.3, dash="dash", color="#f59e0b"),
                    name="Tier 2: Wind-Guided ST-GAT + AirPhyNet",
                ),
                row=1,
                col=1,
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=st_fc_win["timestamp_ist"],
                    y=st_fc_win["model_timesfm_3"],
                    mode="lines",
                    line=dict(width=1.3, dash="longdash", color="#0284c7"),
                    name="Tier 3A: Google TimesFM 3.0 (Patched Decoder)",
                ),
                row=1,
                col=1,
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=st_fc_win["timestamp_ist"],
                    y=st_fc_win["model_chronos_bolt"],
                    mode="lines",
                    line=dict(width=1.3, dash="dashdot", color="#8b5cf6"),
                    name="Tier 3B: Amazon Chronos-2 + EB Diurnal Prior",
                ),
                row=1,
                col=1,
            )

        if show_l2_vs_median:
            # MAE-Optimal Conditional Median Q_0.50
            fig_fc.add_trace(
                go.Scatter(
                    x=st_fc_win["timestamp_ist"],
                    y=st_fc_win["q50_median"],
                    mode="lines",
                    line=dict(width=2.3, dash="dash", color="#e11d48"),
                    name="MAE-Optimal Conditional Median Q_0.50 (µg/m³)",
                ),
                row=1,
                col=1,
            )

        # Main Ensemble Forecast Curve = RMSE-Optimal Trapezoidal Expectation E[Y|X]
        fig_fc.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["mu_l2_expectation"],
                mode="lines+markers",
                marker=dict(size=5, color="#1d4ed8"),
                line=dict(width=3.1, color="#1d4ed8"),
                name="RMSE-Optimal Trapezoidal Expectation E[Y|X] (µg/m³)",
            ),
            row=1,
            col=1,
        )

        # India NAAQS (60 ug/m3) reference line
        fig_fc.add_hline(
            y=60.0,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="India 24h NAAQS (60 µg/m³)",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

        # Row 2: Physical Drivers (PBLH on primary axis, Ventilation Coefficient on secondary)
        fig_fc.add_trace(
            go.Bar(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["boundary_layer_height"],
                marker_color="rgba(14, 165, 233, 0.35)",
                name="Boundary Layer Height PBLH (m)",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig_fc.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["ventilation_coeff"],
                mode="lines",
                line=dict(width=2.2, color="#0f766e"),
                name="Ventilation Coeff (PBLH × Wind, m²/s)",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        fig_fc.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
        fig_fc.update_yaxes(title_text="PBLH (m)", row=2, col=1, secondary_y=False)
        fig_fc.update_yaxes(title_text="Ventilation (m²/s)", row=2, col=1, secondary_y=True)
        fig_fc.update_layout(
            height=630,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, width="stretch")

    # -----------------------------------------------------------------------
    # Visual Callout Card: Why RMSE Requires Distributional Mean E[Y|X] Over Median Q_0.50
    # -----------------------------------------------------------------------
    if show_l2_vs_median:
        mean_gap = float(st_fc_win["l2_vs_median_gap"].mean())
        max_inv_gap = float(st_fc_win["l2_vs_median_gap"].max())
        mean_skew_ratio = float(st_fc_win["upper_tail_skew_ratio"].mean())
        nocturnal_mask = st_fc_win["boundary_layer_height"] < 300.0
        nocturnal_gap = (
            float(st_fc_win.loc[nocturnal_mask, "l2_vs_median_gap"].mean())
            if nocturnal_mask.any()
            else max_inv_gap
        )

        with st.container(border=True):
            st.markdown(
                "#### :material/functions: Why RMSE Requires the Distributional Mean ($\\mathbb{E}[Y\\mid X]$) Over the Median ($Q_{0.50}$) During Nocturnal Inversions"
            )
            c_math, c_kpi = st.columns([1.65, 1.35])
            with c_math:
                st.latex(
                    r"\hat{\mu}_{\text{L2}} = \arg\min_{\hat{y}} \mathbb{E}\!\left[(Y - \hat{y})^2 \mid X\right] = \mathbb{E}[Y \mid X] = \int_0^1 Q_\tau(Y \mid X)\,d\tau \approx \sum_{k=1}^{K-1} \frac{\tau_{k+1}-\tau_k}{2}\bigl(Q_{\tau_k} + Q_{\tau_{k+1}}\bigr) > Q_{0.50}(Y \mid X)"
                )
                st.caption(
                    "When Mumbai's nocturnal Planetary Boundary Layer collapses (`PBLH < 250 m`) under easterly Deonar/Thane Creek land-breeze drainage, "
                    "particulate concentrations follow a **positively skewed log-normal distribution** $\\ln(\\text{PM}_{2.5}) \\sim \\mathcal{N}(\\mu_{\\ln}, \\sigma_{\\ln}^2)$. "
                    "Foundation models (**Google `TimesFM 3.0`** & **Amazon `Chronos-2`**) default to outputting the **median $Q_{0.50} = \\exp(\\mu_{\\ln})$** (optimal only for **MAE / $L_1$ loss**), "
                    "which systematically under-predicts nocturnal smog spikes by $\\exp(\\frac{1}{2}\\sigma_{\\ln}^2) - 1$. "
                    "Integrating the full 9-decile / 21-quantile fan via trapezoidal quadrature recovers the **exact $L_2$-optimal conditional mean $\\hat{\\mu}_{\\text{L2}}$**!"
                )
            with c_kpi:
                k1, k2 = st.columns(2)
                with k1:
                    st.metric(
                        "Nocturnal Inversion Skew Gap",
                        f"+{nocturnal_gap:.2f} µg/m³",
                        f"Peak +{max_inv_gap:.2f} µg/m³ (E[Y|X] − Q_0.50)",
                        border=True,
                    )
                with k2:
                    st.metric(
                        "Upper/Lower Tail Asymmetry",
                        f"{mean_skew_ratio:.2f}× Right-Skew",
                        f"(Q_0.90 − Q_0.50) / (Q_0.50 − Q_0.10) · Mean +{mean_gap:.2f}",
                        border=True,
                    )

    # -----------------------------------------------------------------------
    # Panel B: TimeMixer Two-Scale Synoptic vs. Diurnal Wave Decomposition & AirPhyNet Plume Attribution
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "#### :material/waves: `TimeMixer` Two-Scale Synoptic vs. Diurnal Wave Decomposition & `AirPhyNet` Advection-Diffusion Plume Attribution"
        )
        st.caption(
            "Decomposes hourly $\\text{PM}_{2.5}$ at **"
            f"{selected_station_name}** into: "
            "(1) **Macroscopic 24h Synoptic Regime Wave $L_s(t)$** (driven by post-monsoon multi-day accumulation, $\\tanh$-saturating CAMS MOS $115\\tanh(\\text{CAMS}/115)$, and 24h mean PBLH $\\overline{\\text{PBLH}}_{24\\text{h}}$), and "
            "(2) **Microscopic 24h Diurnal Wave $D_s(t)$** partitioned into **Morning/Evening Traffic Rush Hours**, **Arabian Sea Onshore Breeze Clearing ($u_{10} > 0$)**, and "
            "**`AirPhyNet` Upwind Advection-Diffusion Plume Attribution ($-\\mathbf{u}_{10}\\cdot\\nabla C + \\nabla\\cdot(\\kappa\\nabla C)$)** from Deonar Landfill & Thane Creek."
        )

        fig_tm = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.50, 0.50],
            subplot_titles=(
                "Macroscopic 24h Synoptic Regime Wave L_s(t) vs. Total PM2.5 & tanh-Saturating CAMS MOS",
                "Microscopic 24h Diurnal Wave D_s(t) & AirPhyNet Physical Driver Attribution (µg/m³)",
            ),
        )

        # Top subplot: Total PM2.5, Macroscopic Synoptic Wave L_s(t), and tanh-CAMS MOS
        fig_tm.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["pred_pm25"],
                mode="lines",
                line=dict(width=1.8, dash="dot", color="#64748b"),
                name="Total Forecast PM2.5: L_s(t) + D_s(t)",
            ),
            row=1,
            col=1,
        )
        fig_tm.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["cams_tanh_mos"],
                mode="lines",
                line=dict(width=1.8, dash="dash", color="#f59e0b"),
                name="tanh-Saturating CAMS MOS Prior",
            ),
            row=1,
            col=1,
        )
        fig_tm.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["synoptic_wave_L"],
                mode="lines",
                line=dict(width=3.2, color="#0f766e"),
                name="Macroscopic 24h Synoptic Regime Wave L_s(t)",
            ),
            row=1,
            col=1,
        )

        # Bottom subplot: Microscopic Diurnal Wave D_s(t) & 3 Physical Sub-Waves
        fig_tm.add_trace(
            go.Bar(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["diurnal_traffic_rush"],
                marker_color="rgba(245, 158, 11, 0.55)",
                name="Morning/Evening Traffic Rush Wave (+µg/m³)",
            ),
            row=2,
            col=1,
        )
        fig_tm.add_trace(
            go.Bar(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["airphynet_advection_plume"],
                marker_color="rgba(220, 38, 38, 0.55)",
                name="AirPhyNet Deonar/Thane Land-Breeze Advection Plume",
            ),
            row=2,
            col=1,
        )
        fig_tm.add_trace(
            go.Bar(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["diurnal_seabreeze_clearing"],
                marker_color="rgba(14, 165, 233, 0.55)",
                name="Arabian Sea Onshore Breeze Clearing (−µg/m³)",
            ),
            row=2,
            col=1,
        )
        fig_tm.add_trace(
            go.Scatter(
                x=st_fc_win["timestamp_ist"],
                y=st_fc_win["diurnal_wave_D"],
                mode="lines+markers",
                marker=dict(size=4, color="#1e1b4b"),
                line=dict(width=2.6, color="#1e1b4b"),
                name="Net Microscopic 24h Diurnal Wave D_s(t)",
            ),
            row=2,
            col=1,
        )
        fig_tm.add_hline(y=0.0, line_dash="solid", line_color="#94a3b8", row=2, col=1)

        fig_tm.update_yaxes(title_text="Synoptic PM2.5 (µg/m³)", row=1, col=1)
        fig_tm.update_yaxes(title_text="Diurnal Δ PM2.5 (µg/m³)", row=2, col=1)
        fig_tm.update_layout(
            barmode="relative",
            height=540,
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_tm, width="stretch")

    # -----------------------------------------------------------------------
    # Panel C: Interactive Historical Meteorological Analog (AnEn) Matcher Card (2024–2026)
    # -----------------------------------------------------------------------
    anen_result = find_historical_meteorological_analogs(
        selected_ts=pd.Timestamp(selected_ts),
        location_id=selected_loc_id,
        forecast_df=forecast_df,
        train_df=train_df,
        top_k=3,
    )
    target_sum = anen_result["target_summary"]
    analogs_df = anen_result["analogs_df"]
    analogs_list = anen_result["analogs_list"]

    with st.container(border=True):
        st.markdown(
            f"#### :material/history: Historical Meteorological Analog (`AnEn`) 5-Hour Trajectory Matcher (`2024–2026`) — Active Timestamp `{target_sum['timestamp_ist']}`"
        )
        st.caption(
            "Matches the active 5-hour `[t−4h .. t]` trajectory of **`(PBLH, Wind Vector [u10, v10], Relative Humidity, CAMS PM2.5)`** at "
            f"**{selected_station_name}** against historical post-monsoon regimes in Mumbai (`2024–2026`) using the "
            "**Delle Monache (2013) Analog Ensemble (`AnEn`)** trajectory metric."
        )

        col_anen_tbl, col_anen_plot = st.columns([1.45, 1.15])
        with col_anen_tbl:
            st.markdown(
                f"**Active 5h Target Trajectory (`{target_sum['timestamp_ist']}`):** "
                f"`PBLH = {target_sum['mean_5h_pblh_m']:.0f} m` · "
                f"`Wind = {target_sum['mean_5h_wind']}` · "
                f"`RH = {target_sum['mean_5h_rh_pct']:.1f}%` · "
                f"`CAMS = {target_sum['mean_5h_cams_ugm3']:.1f} µg/m³` → "
                f"**Ensemble Forecast: `{target_sum['pred_pm25_ugm3']:.1f} µg/m³`** "
                f"(Top-3 `AnEn` Analog Consensus: **`{target_sum['anen_consensus_ugm3']:.1f} µg/m³`**)"
            )
            st.dataframe(
                analogs_df[
                    [
                        "rank",
                        "analog_date",
                        "similarity_pct",
                        "anen_distance",
                        "mean_5h_pblh_m",
                        "mean_5h_wind",
                        "mean_5h_rh_pct",
                        "mean_5h_cams_ugm3",
                        "observed_pm25_ugm3",
                        "synoptic_regime",
                    ]
                ].rename(
                    columns={
                        "rank": "Match Rank",
                        "analog_date": "Historical Post-Monsoon Date",
                        "similarity_pct": "Similarity (%)",
                        "anen_distance": "AnEn Dist",
                        "mean_5h_pblh_m": "5h PBLH (m)",
                        "mean_5h_wind": "5h Wind Vector",
                        "mean_5h_rh_pct": "5h RH (%)",
                        "mean_5h_cams_ugm3": "5h CAMS (µg/m³)",
                        "observed_pm25_ugm3": "Hist. PM2.5 (µg/m³)",
                        "synoptic_regime": "Matched Synoptic Regime",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        with col_anen_plot:
            lag_labels = ["t−4h", "t−3h", "t−2h", "t−1h", "t (Active)"]
            fig_anen = go.Figure()
            fig_anen.add_trace(
                go.Scatter(
                    x=lag_labels,
                    y=target_sum["pm25_trajectory_5h"],
                    mode="lines+markers",
                    marker=dict(size=8, color="#1d4ed8"),
                    line=dict(width=3.2, color="#1d4ed8"),
                    name=f"Target 5h Trajectory ({target_sum['timestamp_ist']})",
                )
            )
            analog_colors = ["#10b981", "#f59e0b", "#8b5cf6"]
            for idx_a, a_rec in enumerate(analogs_list):
                fig_anen.add_trace(
                    go.Scatter(
                        x=lag_labels,
                        y=a_rec["pm25_trajectory_5h"],
                        mode="lines+markers",
                        line=dict(width=2.0, dash="dash", color=analog_colors[idx_a % len(analog_colors)]),
                        name=f"{a_rec['rank']}: {a_rec['analog_date']} ({a_rec['similarity_pct']}%)",
                    )
                )
            fig_anen.update_layout(
                height=265,
                margin=dict(l=10, r=10, t=25, b=10),
                yaxis_title="5h PM2.5 Trajectory (µg/m³)",
                legend=dict(orientation="h", y=1.18, font=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_anen, width="stretch")

    # Fourth Row of Tab 2: Monsoon-to-Post-Monsoon Regime Shift & Hampel Sensor Glitch Inspector
    col_regime, col_glitch = st.columns(2)

    with col_regime:
        with st.container(border=True):
            st.markdown("#### :material/wb_sunny: Monsoon-to-Post-Monsoon Regime Shift (`Train.csv`)")
            st.caption(
                "In July–September (SW Monsoon wash-out), Mumbai PM2.5 averages `11–20 µg/m³`. In October–November "
                "(coastal boundary-layer inversion + eastern land-breeze), PM2.5 jumps `3.5×–4.5×` to `54–98 µg/m³`."
            )
            st_train = train_df[train_df["location_id"] == selected_loc_id].copy()
            st_train["year_month"] = st_train["timestamp_ist"].dt.strftime("%Y-%m")
            monthly_stats = (
                st_train.groupby("year_month", as_index=False)
                .agg(
                    clean_mean=("pm25_clean", "mean"),
                    clean_p90=("pm25_clean", lambda s: float(np.nanpercentile(s, 90))),
                    raw_max=("pm25_raw", "max"),
                )
                .tail(12)
            )
            fig_reg = go.Figure()
            fig_reg.add_trace(
                go.Bar(
                    x=monthly_stats["year_month"],
                    y=monthly_stats["clean_mean"],
                    name="Monthly Mean PM2.5 (Cleaned)",
                    marker_color="#2563eb",
                )
            )
            fig_reg.add_trace(
                go.Scatter(
                    x=monthly_stats["year_month"],
                    y=monthly_stats["clean_p90"],
                    mode="lines+markers",
                    name="90th Percentile PM2.5",
                    line=dict(color="#dc2626", width=2.5),
                )
            )
            fig_reg.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="PM2.5 (µg/m³)",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_reg, width="stretch")

    with col_glitch:
        with st.container(border=True):
            st.markdown("#### :material/cleaning_services: Hampel Sensor Glitch Inspector (`flag_high_1000`)")
            glitches_df = (
                train_df[train_df["is_sensor_glitch"]]
                .sort_values("pm25_raw", ascending=False)[
                    ["timestamp_ist", "station_name", "pm25_raw", "pm25_clean", "flag_high_1000", "hampel_score"]
                ]
            )
            st.caption(
                f"Detected **{len(glitches_df)} extreme sensor anomalies** across `Train.csv` (including hardware spikes up to "
                f"`241,587.6 µg/m³` at Shivaji Nagar and `75,203.5 µg/m³` at Borivali East) that severely bias uncleaned RMSE models."
            )
            st.dataframe(
                glitches_df.head(15).rename(
                    columns={
                        "timestamp_ist": "Timestamp (IST)",
                        "station_name": "Station",
                        "pm25_raw": "Raw Glitch (µg/m³)",
                        "pm25_clean": "Hampel Cleaned (µg/m³)",
                        "flag_high_1000": "flag_high_1000",
                        "hampel_score": "Hampel z-Score",
                    }
                ),
                hide_index=True,
                height=265,
                width="stretch",
            )


# ===========================================================================
# TAB 3: Vulnerable Population, UDISE+ School & Civic Hospital Exposure Audit
# ===========================================================================
with tab3:
    st.subheader("Vulnerable Population, UDISE+ School & BMC Civic Hospital Exposure Audit")
    st.caption(
        "Combines **BMC 24 Administrative Wards (`doi:10.71646/EREDWQ`)**, **WorldPop 2026 Age-Stratified Demographics (`doi:10.71646/AAGK0G`)**, "
        "**Mumbai UDISE+ 2024–25 Schools (`doi:10.71646/XJIY8W`)**, and **MyBMC Georeferenced Civic Hospitals (`doi:10.71646/1XSUVR`)** "
        "against India NAAQS (`60 µg/m³`) and WHO (`15 µg/m³`) guidelines."
    )

    col_wchart, col_wtable = st.columns([1.25, 1.35])

    with col_wchart:
        with st.container(border=True):
            st.markdown("#### Population-Weighted PM2.5 Across Mumbai's 24 BMC Wards")
            fig_wards = px.bar(
                wards_df.head(16).iloc[::-1],
                x="pm25_ugm3",
                y="ward_code",
                color="risk_level",
                orientation="h",
                hover_data=["ward_name", "pop_2026", "vulnerable_pop", "udise_schools", "civic_hospitals"],
                color_discrete_map={
                    "Critical": "#8b5cf6",
                    "High": "#ef4444",
                    "Elevated": "#f59e0b",
                    "Moderate": "#10b981",
                },
                labels={
                    "pm25_ugm3": "Kriged Ward PM2.5 (µg/m³)",
                    "ward_code": "BMC Ward",
                    "risk_level": "GRAP Risk Tier",
                },
            )
            fig_wards.add_vline(
                x=60.0,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text="NAAQS 60 µg/m³",
            )
            fig_wards.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_wards, width="stretch")

    with col_wtable:
        with st.container(border=True):
            st.markdown("#### 24 BMC Wards Demographic & Infrastructure Vulnerability Matrix")
            st.dataframe(
                wards_df[
                    [
                        "ward_code",
                        "ward_name",
                        "pm25_ugm3",
                        "naaqs_exceed_x",
                        "who_exceed_x",
                        "vulnerable_pop",
                        "udise_schools",
                        "civic_hospitals",
                        "maharera_sites",
                        "grap_stage",
                    ]
                ].rename(
                    columns={
                        "ward_code": "Ward",
                        "ward_name": "Administrative Area",
                        "pm25_ugm3": "PM2.5 (µg/m³)",
                        "naaqs_exceed_x": "× NAAQS",
                        "who_exceed_x": "× WHO",
                        "vulnerable_pop": "Children (<5y) & Elderly (>60y)",
                        "udise_schools": "UDISE+ Schools",
                        "civic_hospitals": "Civic Hospitals",
                        "maharera_sites": "MahaRERA Sites",
                        "grap_stage": "BMC GRAP Trigger",
                    }
                ),
                hide_index=True,
                height=440,
                width="stretch",
            )

    # High-Risk Schools & Civic Hospitals Actionable Alert Table
    with st.container(border=True):
        st.markdown(
            "#### :material/school: High-Risk UDISE+ Schools (`doi:10.71646/XJIY8W`) & BMC Civic Hospitals (`doi:10.71646/1XSUVR`) — Real-Time Advisories"
        )
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            fac_filter = st.multiselect(
                "Filter by facility type",
                options=["UDISE+ School", "BMC Civic Hospital"],
                default=["UDISE+ School", "BMC Civic Hospital"],
            )
        with f_col2:
            ward_filter = st.multiselect(
                "Filter by high-risk BMC Ward",
                options=sorted(facilities_df["ward_code"].unique()),
                default=sorted(facilities_df["ward_code"].unique()),
            )

        filtered_fac = facilities_df[
            facilities_df["type"].isin(fac_filter) & facilities_df["ward_code"].isin(ward_filter)
        ]
        st.dataframe(
            filtered_fac[
                [
                    "facility_name",
                    "type",
                    "ward_name",
                    "nearest_station",
                    "pm25_ugm3",
                    "who_multiple",
                    "daily_population",
                    "recommended_protocol",
                    "doi",
                ]
            ].rename(
                columns={
                    "facility_name": "Facility Name",
                    "type": "Category",
                    "ward_name": "BMC Ward",
                    "nearest_station": "Nearest CPCB Anchor",
                    "pm25_ugm3": "PM2.5 (µg/m³)",
                    "who_multiple": "× WHO (15 µg/m³)",
                    "daily_population": "Daily Children / Patients",
                    "recommended_protocol": "Automated Civic Health Protocol",
                    "doi": "ClimateVerse DOI",
                }
            ),
            hide_index=True,
            width="stretch",
        )


# ===========================================================================
# TAB 4: Gemini 4 Argon + LiteLLM Civic Action Copilot & ClimateVerse Explorer
# ===========================================================================
with tab4:
    st.subheader("Google Gemini 4 Argon + Universal LiteLLM Civic Action Copilot")
    st.caption(
        "Ward-level **BMC Graded Response Action Plan (GRAP)** orders route through a 3-tier chain: "
        "**Tier 1** `litellm.completion` (flagship default **Google Gemini 4 Argon**, `gemini/gemini-4-argon`), "
        "**Tier 2** native `google-genai` when a new Gemini slug is not yet prefixed, and "
        "**Tier 3** a deterministic physics engine (PM2.5, PBLH, ventilation, TimesFM 3.0 & Chronos-2 quantiles, "
        "Deonar advection cosine, UDISE+ schools, civic hospitals) when no API key is set."
    )

    with st.container(border=True):
        st.markdown("#### :material/auto_awesome: AI Copilot Model & LiteLLM Router Configuration")
        st.caption(
            "Choose a provider, a preset flagship model, or any custom LiteLLM model string "
            "(`gemini/gemini-4-argon-high`, `openrouter/google/gemini-4-argon`, `vertex_ai/gemini-4-argon`). "
            "Keys can also come from `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, or `OPENROUTER_API_KEY`."
        )
        route_c1, route_c2, route_c3 = st.columns(3)
        with route_c1:
            copilot_provider = st.selectbox(
                "Provider / router",
                options=list(PROVIDER_OPTIONS),
                key="copilot_provider",
                help="Google Gemini uses LiteLLM first, then the native google-genai SDK. Other providers stay on LiteLLM.",
            )
        env_key_names = provider_env_key_names(copilot_provider)
        with route_c2:
            copilot_api_key = st.text_input(
                "API key (optional — overrides environment)",
                type="password",
                key="copilot_api_key",
                help="Leave blank to use " + ", ".join(f"`{name}`" for name in env_key_names) + ".",
            )
        with route_c3:
            copilot_api_base = st.text_input(
                "Optional API base URL",
                key="copilot_api_base",
                placeholder="https://openrouter.ai/api/v1 · http://127.0.0.1:11434",
                help="Set COPILOT_API_BASE or LITELLM_API_BASE to supply this without typing it. Blank uses the provider default.",
            )
        discover_col, status_col = st.columns([1, 2])
        with discover_col:
            discover_clicked = st.button(
                "Discover Live Models from API Key",
                icon=":material/travel_explore:",
                width="stretch",
            )
        if discover_clicked:
            discovery_key = resolve_api_key(copilot_provider, copilot_api_key)
            try:
                st.session_state["discovered_gemini_models"] = discover_available_gemini_models(discovery_key)
                st.session_state["discover_status"] = (
                    f"Discovered {len(st.session_state['discovered_gemini_models'])} live Gemini models "
                    "via google-genai models.list()."
                )
            except ValueError as exc:
                st.session_state["discover_status"] = str(exc)
            except Exception as exc:
                st.session_state["discover_status"] = (
                    f"Live discovery unavailable ({type(exc).__name__}). The preset catalog stays available."
                )
        with status_col:
            if st.session_state.get("discover_status"):
                st.caption(st.session_state["discover_status"])
            else:
                active_env = next((name for name in env_key_names if os.environ.get(name, "").strip()), "")
                if active_env and not copilot_api_key.strip():
                    st.caption(f"Environment key detected: `{active_env}` (hidden). A typed key overrides it.")
                else:
                    st.caption("Presets include Gemini 4 Argon through Gemini 2.5 Flash, Grok, Claude, GPT-5, OpenRouter, and Ollama.")

        catalog = build_model_catalog(
            discovered=st.session_state.get("discovered_gemini_models") or [],
            extra_model_id=st.session_state.get("copilot_model_id", FLAGSHIP_MODEL_ID),
        )
        catalog_ids = [model_id for model_id, _label in catalog]
        if st.session_state.get("copilot_model_id") not in catalog_ids:
            st.session_state["copilot_model_id"] = FLAGSHIP_MODEL_ID
        label_by_id = {model_id: label for model_id, label in catalog}
        model_col, custom_col = st.columns(2)
        with model_col:
            copilot_model_id = st.selectbox(
                "Flagship model",
                options=catalog_ids,
                format_func=lambda model_id: f"{label_by_id.get(model_id, model_id)} · {model_id}",
                key="copilot_model_id",
                help=f"Default is {FLAGSHIP_MODEL_LABEL} (`{FLAGSHIP_MODEL_ID}`).",
            )
        with custom_col:
            copilot_custom_model = st.text_input(
                "Custom LiteLLM model ID (overrides the dropdown)",
                key="copilot_custom_model",
                placeholder="gemini/gemini-4-argon-high",
                help="Any litellm model string. Leave blank to use the dropdown.",
            )
        resolved_model = resolve_litellm_model(copilot_provider, copilot_model_id, copilot_custom_model)
        resolved_base = resolve_api_base(copilot_provider, copilot_api_base, resolved_model)
        base_caption = resolved_base or "LiteLLM provider default"
        st.caption(f"Active route: `{resolved_model}` via **{copilot_provider}** · API base `{base_caption}` · temperature `0.2`.")

    cop_left, cop_right = st.columns([1.05, 1.95])

    with cop_left:
        with st.container(border=True):
            st.markdown("#### :material/tune: Configure Ward GRAP Directive")
            ward_labels = {
                f"Ward {r['ward_code']} — {r['ward_name']} ({r['pm25_ugm3']:.1f} µg/m³)": r["ward_code"]
                for _, r in wards_df.iterrows()
            }
            selected_ward_label = st.selectbox(
                "Target BMC Administrative Ward (`doi:10.71646/EREDWQ`)",
                options=list(ward_labels.keys()),
                index=0,
            )
            selected_ward_code = ward_labels[selected_ward_label]
            selected_ward_row = wards_df[wards_df["ward_code"] == selected_ward_code].iloc[0].to_dict()

            custom_query = st.text_area(
                "Custom policy or mitigation question for the Civic Copilot",
                value=(
                    "Design a 24-hour BMC water sprinkling schedule, MahaRERA construction dust enforcement plan, "
                    "Deonar landfill smoke patrol protocol, and UDISE+ morning assembly advisory for this ward."
                ),
                height=110,
            )
            generate_btn = st.button(
                "Generate BMC GRAP Civic Directive",
                type="primary",
                icon=":material/bolt:",
                width="stretch",
            )

    with cop_right:
        with st.container(border=True):
            fac_in_ward = facilities_df[facilities_df["ward_code"] == selected_ward_code].to_dict("records")
            key_tag = (
                hashlib.sha256(copilot_api_key.strip().encode()).hexdigest()[:8]
                if copilot_api_key.strip()
                else "nokey"
            )
            route_fingerprint = "|".join(
                [
                    selected_ward_code,
                    copilot_provider,
                    resolved_model,
                    copilot_api_base.strip(),
                    key_tag,
                ]
            )
            if (
                generate_btn
                or "last_copilot_md" not in st.session_state
                or st.session_state.get("last_copilot_fingerprint") != route_fingerprint
            ):
                md_out, engine_used = run_gemini_civic_copilot(
                    ward_row=selected_ward_row,
                    snapshot_meta=snapshot,
                    facilities_in_ward=fac_in_ward,
                    user_question=custom_query,
                    api_key_override=copilot_api_key,
                    model=copilot_model_id,
                    provider=copilot_provider,
                    api_base=copilot_api_base,
                    custom_model=copilot_custom_model,
                    quantile_context=_quantile_context_for_ward(selected_ward_row),
                )
                st.session_state["last_copilot_md"] = md_out
                st.session_state["last_copilot_engine"] = engine_used
                st.session_state["last_copilot_fingerprint"] = route_fingerprint

            engine_label = st.session_state["last_copilot_engine"]
            badge_color = "green" if engine_label.startswith("Tier 1") else "orange" if engine_label.startswith("Tier 2") else "blue"
            st.badge(engine_label, icon=":material/verified:", color=badge_color)
            st.caption(
                "Flagship default is **Google Gemini 4 Argon** (`gemini/gemini-4-argon`). "
                "LiteLLM covers Gemini, OpenRouter, Vertex AI, OpenAI, Anthropic, xAI Grok, and local Ollama / vLLM."
            )
            st.markdown(st.session_state["last_copilot_md"])

    # Searchable ClimateVerse Dataverse Catalog Explorer
    with st.container(border=True):
        st.markdown(
            "#### :material/travel_explore: Interactive ClimateVerse Google Air Quality Challenge Dataverse Explorer"
        )
        st.caption(
            "Search and filter the curated datasets from `https://india.climateverse.net/dataverse/google-air-quality-forecasting-challenge` "
            "and `data/external/climateverse_catalog.json`."
        )
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            search_q = st.text_input(
                "Search by DOI, title, provider, or physical variable",
                placeholder="e.g., AlphaEarth, Deonar, UDISE, METAR, Open-Meteo, 10.71646...",
            )
        with s_col2:
            cat_options = ["All Categories"] + sorted(catalog_df["category"].unique().tolist())
            selected_cat = st.selectbox("Filter by domain category", options=cat_options)

        filt_cat = catalog_df.copy()
        if selected_cat != "All Categories":
            filt_cat = filt_cat[filt_cat["category"] == selected_cat]
        if search_q.strip():
            q_low = search_q.strip().lower()
            mask = (
                filt_cat["doi"].str.lower().str.contains(q_low, na=False)
                | filt_cat["title"].str.lower().str.contains(q_low, na=False)
                | filt_cat["provider"].str.lower().str.contains(q_low, na=False)
                | filt_cat["role"].str.lower().str.contains(q_low, na=False)
            )
            filt_cat = filt_cat[mask]

        st.dataframe(
            filt_cat[
                ["doi", "title", "category", "provider", "resolution", "integrated_in_app", "role", "url"]
            ].rename(
                columns={
                    "doi": "ClimateVerse DOI",
                    "title": "Dataset Title",
                    "category": "Domain Category",
                    "provider": "Curator / Source",
                    "resolution": "Spatiotemporal Resolution",
                    "integrated_in_app": "Active in VayuDrishti",
                    "role": "Modeling & Decision-Support Role",
                    "url": "Persistent Link",
                }
            ),
            column_config={
                "Persistent Link": st.column_config.LinkColumn("DOI URL"),
                "Active in VayuDrishti": st.column_config.CheckboxColumn("Integrated"),
            },
            hide_index=True,
            width="stretch",
        )
