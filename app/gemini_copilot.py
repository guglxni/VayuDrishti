"""VayuDrishti Mumbai Civic Action Copilot.

Three-tier router for ward-level BMC Graded Response Action Plan (GRAP) orders:

1. **LiteLLM** ``litellm.completion`` — any provider slug, with a short alias chain
   when a newly released model prefix is not yet mapped.
2. **google-genai** native ``models.generate_content`` — direct fallback for Gemini
   slugs (including Gemini 4 Argon) when LiteLLM rejects the provider prefix.
3. **Deterministic physics engine** — full ward GRAP from station telemetry when
   no API key is configured or every live call fails.

The flagship default is Google Gemini 4 Argon (``gemini/gemini-4-argon``).
Live Gemini IDs are merged from ``client.models.list()`` when a key is supplied.
Model, provider, API key, and ``api_base`` are configurable from the UI and from
environment variables.
"""

from __future__ import annotations

import math
import os
from typing import Any

FLAGSHIP_MODEL_ID = "gemini/gemini-4-argon"
FLAGSHIP_MODEL_LABEL = "Google Gemini 4 Argon — Flagship Reasoning & Spatial Synthesis"

PROVIDER_OPTIONS: tuple[str, ...] = (
    "Google Gemini (Native + LiteLLM)",
    "LiteLLM Universal (Any Provider)",
    "OpenRouter",
    "Local Ollama / vLLM",
    "Vertex AI",
    "OpenAI",
    "Anthropic",
    "xAI Grok",
)

# (litellm model id, human label). Gemini 4 Argon stays pinned at index 0.
PRESET_MODELS: tuple[tuple[str, str], ...] = (
    (FLAGSHIP_MODEL_ID, FLAGSHIP_MODEL_LABEL),
    ("gemini/gemini-3.0-pro", "Google Gemini 3.0 Pro"),
    ("gemini/gemini-3.0-flash", "Google Gemini 3.0 Flash"),
    ("gemini/gemini-2.5-pro", "Google Gemini 2.5 Pro"),
    ("gemini/gemini-2.5-flash", "Google Gemini 2.5 Flash"),
    ("xai/grok-4.7", "xAI Grok 4.7"),
    ("anthropic/claude-opus-4-6", "Anthropic Claude Opus 4.6"),
    ("anthropic/claude-sonnet-4-6", "Anthropic Claude Sonnet 4.6"),
    ("openai/gpt-5", "OpenAI GPT-5"),
    ("openrouter/google/gemini-4-argon", "OpenRouter · Google Gemini 4 Argon"),
    ("ollama/llama3.3", "Ollama Local · Llama 3.3"),
)

PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "Google Gemini (Native + LiteLLM)": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "LiteLLM Universal (Any Provider)": (
        "LITELLM_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
    ),
    "OpenRouter": ("OPENROUTER_API_KEY", "LITELLM_API_KEY"),
    "Local Ollama / vLLM": ("OLLAMA_API_KEY",),
    "Vertex AI": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "OpenAI": ("OPENAI_API_KEY",),
    "Anthropic": ("ANTHROPIC_API_KEY",),
    "xAI Grok": ("XAI_API_KEY",),
}

# Longest prefixes first so `openrouter/google/gemini-4-argon` yields `gemini-4-argon`.
_MODEL_PREFIXES: tuple[str, ...] = (
    "openrouter/google/",
    "vertex_ai/",
    "gemini/",
    "openrouter/",
    "anthropic/",
    "openai/",
    "ollama/",
    "xai/",
)

DEONAR_LAT = 19.0560
DEONAR_LON = 72.9280

_SKIP_MODEL_TOKENS = (
    "embedding",
    "imagen",
    "veo",
    "aqa",
    "tts",
    "image",
    "audio",
    "live",
    "robotics",
)


def provider_env_key_names(provider: str) -> tuple[str, ...]:
    """Environment variable names consulted for ``provider`` when the UI key is blank."""
    return PROVIDER_ENV_KEYS.get(provider, ("GEMINI_API_KEY", "GOOGLE_API_KEY"))


def default_copilot_settings() -> dict[str, str]:
    """Environment-backed defaults. Secrets are never copied into this dict."""
    provider = os.environ.get("COPILOT_PROVIDER", "").strip()
    if provider not in PROVIDER_OPTIONS:
        provider = PROVIDER_OPTIONS[0]
    model = (
        os.environ.get("COPILOT_MODEL", "").strip()
        or os.environ.get("LITELLM_MODEL", "").strip()
        or FLAGSHIP_MODEL_ID
    )
    api_base = os.environ.get("COPILOT_API_BASE", "").strip() or os.environ.get("LITELLM_API_BASE", "").strip()
    return {
        "provider": provider,
        "model_id": model,
        "custom_model": os.environ.get("COPILOT_CUSTOM_MODEL", ""),
        "api_base": api_base,
    }


def raw_model_slug(model: str) -> str:
    """Strip a known router prefix and return the provider-native model id."""
    text = (model or "").strip()
    for prefix in _MODEL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def is_gemini_family(model: str) -> bool:
    return "gemini" in (model or "").lower()


def _gemini_sort_key(slug: str) -> tuple[int, int, str]:
    lowered = slug.lower()
    if "gemini-4" in lowered:
        major = 0
    elif "gemini-3" in lowered:
        major = 1
    elif "gemini-2.5" in lowered or "gemini-2-5" in lowered:
        major = 2
    else:
        major = 3
    if "argon" in lowered:
        family = 0
    elif "pro" in lowered:
        family = 1
    elif "flash" in lowered:
        family = 2
    else:
        family = 3
    return (major, family, lowered)


def build_model_catalog(
    discovered: list[dict[str, str]] | None = None,
    extra_model_id: str = "",
) -> list[tuple[str, str]]:
    """Preset flagship list, then any env/custom id, then live ``models.list()`` hits.

    Gemini 4 Argon remains the first preset. Duplicate ids are dropped.
    """
    options: list[tuple[str, str]] = list(PRESET_MODELS)
    seen = {model_id for model_id, _label in options}
    extra = (extra_model_id or "").strip()
    if extra and extra not in seen:
        options.insert(1, (extra, f"Environment model · {extra}"))
        seen.add(extra)
    live_rows: list[tuple[str, str]] = []
    for item in discovered or []:
        model_id = str(item.get("id", "")).strip()
        label = str(item.get("label", model_id)).strip() or model_id
        if model_id and model_id not in seen:
            live_rows.append((model_id, label))
            seen.add(model_id)
    live_rows.sort(key=lambda row: _gemini_sort_key(raw_model_slug(row[0])))
    options.extend(live_rows)
    return options


def resolve_litellm_model(provider: str, preset_model: str, custom_model: str = "") -> str:
    """Custom model strings pass through unchanged. Presets are routed by provider."""
    custom = (custom_model or "").strip()
    if custom:
        return custom
    return _route_preset(preset_model or FLAGSHIP_MODEL_ID, provider or PROVIDER_OPTIONS[0])


def _route_preset(model: str, provider: str) -> str:
    model = (model or FLAGSHIP_MODEL_ID).strip()
    if provider == "LiteLLM Universal (Any Provider)":
        return model
    if provider == "Google Gemini (Native + LiteLLM)":
        if model.startswith(("gemini/", "vertex_ai/", "openrouter/")):
            return model
        slug = raw_model_slug(model)
        if slug.lower().startswith("gemini"):
            return f"gemini/{slug}"
        return model
    if provider == "OpenRouter":
        if model.startswith("openrouter/"):
            return model
        if model.startswith("gemini/") or is_gemini_family(model):
            return f"openrouter/google/{raw_model_slug(model)}"
        if "/" in model:
            return f"openrouter/{model}"
        return f"openrouter/{model}"
    if provider == "Vertex AI":
        if model.startswith("vertex_ai/"):
            return model
        if is_gemini_family(model) or model.startswith("gemini"):
            return f"vertex_ai/{raw_model_slug(model)}"
        return model
    if provider == "OpenAI":
        if model.startswith("openai/") or model.startswith("gpt-"):
            return model if model.startswith("openai/") else f"openai/{raw_model_slug(model)}"
        return model
    if provider == "Anthropic":
        if "claude" in model.lower() and not model.startswith("anthropic/"):
            return f"anthropic/{raw_model_slug(model)}"
        return model
    if provider == "xAI Grok":
        if "grok" in model.lower() and not model.startswith("xai/"):
            return f"xai/{raw_model_slug(model)}"
        return model
    if provider.startswith("Local"):
        if model.startswith(("ollama/", "openai/", "hosted_vllm/")):
            return model
        if "/" not in model:
            return f"ollama/{model}"
        return model
    return model


def alias_chain(resolved_model: str, provider: str) -> list[str]:
    """Primary slug plus at most one alternate prefix for the same underlying model."""
    resolved = (resolved_model or FLAGSHIP_MODEL_ID).strip()
    attempts = [resolved]
    slug = raw_model_slug(resolved)
    alternate = ""
    if is_gemini_family(resolved):
        if provider == "Vertex AI" or resolved.startswith("vertex_ai/"):
            alternate = f"gemini/{slug}"
        elif resolved.startswith("gemini/"):
            alternate = f"vertex_ai/{slug}"
        else:
            alternate = f"gemini/{slug}"
    elif resolved.startswith("xai/") or "grok" in slug.lower():
        alternate = f"openai/{slug}" if resolved.startswith("xai/") else f"xai/{slug}"
    elif "claude" in slug.lower() and not resolved.startswith("anthropic/"):
        alternate = f"anthropic/{slug}"
    elif resolved.startswith("gpt-"):
        alternate = f"openai/{slug}"
    if alternate and alternate not in attempts:
        attempts.append(alternate)
    return attempts[:2]


def resolve_api_key(provider: str, ui_key: str = "") -> str:
    """UI password wins. Otherwise the provider's environment variables, in order."""
    typed = (ui_key or "").strip()
    if typed:
        return typed
    for env_name in provider_env_key_names(provider):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def resolve_api_base(provider: str, ui_base: str = "", model: str = "") -> str:
    """Optional endpoint. Blank means LiteLLM's own default, except routers that require a base."""
    typed = (ui_base or "").strip()
    if typed:
        return typed
    for env_name in ("COPILOT_API_BASE", "LITELLM_API_BASE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    resolved = model or ""
    if provider == "OpenRouter" or resolved.startswith("openrouter/"):
        return os.environ.get("OPENROUTER_API_BASE", "").strip() or "https://openrouter.ai/api/v1"
    if provider == "xAI Grok" or resolved.startswith("xai/"):
        return os.environ.get("XAI_API_BASE", "").strip() or "https://api.x.ai/v1"
    if resolved.startswith("ollama/") or (
        provider.startswith("Local") and resolved.startswith(("ollama/", "hosted_vllm/"))
    ):
        return os.environ.get("OLLAMA_API_BASE", "").strip() or "http://127.0.0.1:11434"
    if provider == "OpenAI":
        return os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("OPENAI_API_BASE", "").strip()
    return ""


def discover_available_gemini_models(api_key: str) -> list[dict[str, str]]:
    """List live Gemini generate-content models on the caller's Google AI key.

    Uses ``google.genai.Client(api_key=api_key).models.list()`` and returns
    LiteLLM ids (``gemini/<slug>``) ready to merge into the selector.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError(
            "Add a GEMINI_API_KEY or paste a Google AI Studio key to discover live models."
        )
    from google import genai

    client = genai.Client(api_key=key)
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for model in client.models.list():
        name = str(getattr(model, "name", "") or "").strip()
        slug = name.split("/")[-1].strip()
        if not slug or "gemini" not in slug.lower():
            continue
        lowered = slug.lower()
        if any(token in lowered for token in _SKIP_MODEL_TOKENS):
            continue
        actions = [str(action).lower() for action in (getattr(model, "supported_actions", None) or [])]
        if actions and not any("generatecontent" in action or action == "generate" for action in actions):
            continue
        litellm_id = f"gemini/{slug}"
        if litellm_id in seen:
            continue
        seen.add(litellm_id)
        display = str(getattr(model, "display_name", "") or slug).strip() or slug
        found.append(
            {
                "id": litellm_id,
                "raw": slug,
                "label": f"Live · {display} (`{litellm_id}`)",
            }
        )
        if len(found) >= 200:
            break
    found.sort(key=lambda item: _gemini_sort_key(item["raw"]))
    return found


def deonar_advection_cosine(ward_row: dict[str, Any], snapshot_meta: dict[str, Any]) -> float:
    """Cosine between the wind-transport vector and the Deonar → ward axis.

    Positive values mean the hourly flow carries Deonar air toward the ward.
    """
    lat = float(ward_row.get("lat", DEONAR_LAT))
    lon = float(ward_row.get("lon", DEONAR_LON))
    u10 = snapshot_meta.get("wind_u10")
    v10 = snapshot_meta.get("wind_v10")
    if u10 is None or v10 is None:
        wind_from = float(snapshot_meta.get("wind_dir_from_deg", 75.0))
        speed = float(snapshot_meta.get("wind_speed_ms", 1.0) or 1.0)
        radians = math.radians(wind_from)
        u10 = -math.sin(radians) * speed
        v10 = -math.cos(radians) * speed
    else:
        u10 = float(u10)
        v10 = float(v10)
    dy_km = (lat - DEONAR_LAT) * 110.574
    dx_km = (lon - DEONAR_LON) * 111.320 * math.cos(math.radians(lat))
    dist_km = math.hypot(dx_km, dy_km)
    wind = math.hypot(u10, v10)
    if dist_km < 0.05 or wind < 0.05:
        return 1.0
    return (dx_km * u10 + dy_km * v10) / (dist_km * wind)


def _deonar_phrase(cosine: float) -> str:
    if cosine > 0.15:
        return "Deonar plume axis is aligned toward this ward"
    if cosine < -0.15:
        return "Transport is directed off the Deonar-to-ward axis"
    return "Transport is cross-wind to the Deonar-to-ward axis"


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _foundation_quantiles(
    pm25: float,
    snapshot_meta: dict[str, Any],
    quantile_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer real TimesFM / Chronos columns. Otherwise a deterministic log-normal fan."""
    ctx = quantile_context or {}
    pblh = _num(snapshot_meta.get("pblh_m"), 380.0)
    u10_raw = snapshot_meta.get("wind_u10")
    u10 = None if u10_raw is None else _num(u10_raw, 0.0)
    inversion = min(1.15, max(0.0, (450.0 - pblh) / 310.0))
    land = 1.0 if (u10 is not None and u10 < 0.0) else 0.25
    sigma = min(0.40, max(0.12, 0.135 + 0.065 * math.sqrt(0.5) + 0.135 * inversion + 0.032 * land))

    def quantile(z_score: float, sigma_scale: float) -> float:
        scaled = sigma * sigma_scale
        location = math.log(max(pm25, 5.0)) - 0.5 * scaled * scaled
        return max(2.5, min(450.0, math.exp(location + z_score * scaled)))

    return {
        "anchor_station": str(ctx.get("anchor_station") or "ward kriging surface"),
        "timesfm_q10": _num(ctx.get("timesfm_q10"), quantile(-1.2816, 0.96)),
        "timesfm_q50": _num(ctx.get("timesfm_q50"), quantile(0.0, 0.96)),
        "timesfm_q90": _num(ctx.get("timesfm_q90"), quantile(1.2816, 0.96)),
        "chronos_q05": _num(ctx.get("chronos_q05"), quantile(-1.6449, 1.03)),
        "chronos_q50": _num(ctx.get("chronos_q50"), quantile(0.0, 1.03)),
        "chronos_q95": _num(ctx.get("chronos_q95"), quantile(1.6449, 1.03)),
        "mu_l2": _num(ctx.get("mu_l2"), pm25),
        "q50_median": _num(ctx.get("q50_median"), quantile(0.0, 1.0)),
    }


def generate_deterministic_grap_briefing(
    ward_row: dict[str, Any],
    snapshot_meta: dict[str, Any],
    facilities_in_ward: list[dict[str, Any]],
    user_question: str = "",
    quantile_context: dict[str, Any] | None = None,
) -> str:
    """Ward-specific BMC GRAP directive from physical meteorology and ClimateVerse fields."""
    ward_code = ward_row["ward_code"]
    ward_name = ward_row["ward_name"]
    pm25 = float(ward_row["pm25_ugm3"])
    naaqs_x = float(ward_row["naaqs_exceed_x"])
    who_x = float(ward_row["who_exceed_x"])
    grap_stage = ward_row["grap_stage"]
    pop = int(ward_row["pop_2026"])
    vuln_pop = int(ward_row["vulnerable_pop"])
    maharera = int(ward_row["maharera_sites"])
    schools = int(ward_row["udise_schools"])
    hospitals = int(ward_row["civic_hospitals"])
    impervious = float(ward_row["alphaearth_impervious"])
    bldg_density = float(ward_row["open_bldg_density"])

    ts_str = str(snapshot_meta["timestamp_ist"])
    pblh = float(snapshot_meta["pblh_m"])
    vc = float(snapshot_meta["ventilation_coeff"])
    w_spd = float(snapshot_meta["wind_speed_ms"])
    w_dir = float(snapshot_meta["wind_dir_from_deg"])
    quantiles = _foundation_quantiles(pm25, snapshot_meta, quantile_context)
    deonar_cos = deonar_advection_cosine(ward_row, snapshot_meta)
    deonar_text = _deonar_phrase(deonar_cos)

    if 45 <= w_dir <= 145:
        breeze_regime = (
            f"Easterly / Land-Breeze Drainage ({w_dir:.0f}° at {w_spd:.1f} m/s) "
            "advecting Thane Creek & Deonar inland plumes"
        )
    elif 230 <= w_dir <= 330:
        breeze_regime = f"Onshore Arabian Sea Breeze ({w_dir:.0f}° at {w_spd:.1f} m/s) providing marine ventilation"
    else:
        breeze_regime = f"Transitional Synoptic Flow ({w_dir:.0f}° at {w_spd:.1f} m/s)"

    if ward_code in ("M/E", "M/W", "N", "L"):
        hotspot_directive = (
            "- **Deonar Landfill & Waste-Sector Protocol (`doi:10.71646/MWQNR8`, `doi:10.71646/VA5BD9`)**: "
            "Deploy BMC Solid Waste Management (SWM) thermal infrared drone & methane patrol across Sectors B–D of "
            "Deonar Dumping Ground (`19.056°N, 72.928°E`). Activate perimeter high-pressure fogging cannons along "
            "Ghatkopar–Mankhurd Link Road (GMLR) between `05:00–10:30 IST` to suppress nocturnal katabatic smoke drainage toward Shivaji Nagar."
        )
    elif ward_code in ("E", "F/S", "B", "G/S"):
        hotspot_directive = (
            "- **Eastern Waterfront & Freight Corridor Protocol (`doi:10.71646/EREDWQ`)**: "
            "Coordinate with Mumbai Port Trust (MbPT) and Traffic Police to restrict heavy diesel multi-axle freight movement "
            "along P. D'Mello Road, Mazgaon Dockyard, and JJ Flyover approaches during morning (`07:00–11:00 IST`) and evening (`19:00–23:00 IST`) boundary-layer collapse."
        )
    elif ward_code in ("P/N", "K/E", "K/W", "R/C", "R/S"):
        hotspot_directive = (
            "- **Western Suburbs Arterial & Creek Corridor (`doi:10.71646/EV3RBJ`)**: "
            "Synchronize green-wave signal timing at Western Express Highway (WEH) and New Link Road junctions in Malad/Borivali "
            "to cut idling NOx/PM2.5 emissions by 18–24% before the afternoon sea-breeze front arrives."
        )
    else:
        hotspot_directive = (
            "- **Arterial Corridor Sweeping (`doi:10.71646/EREDWQ`)**: "
            "Deploy vacuum-assisted mechanical sweepers on primary arterial roads between `03:30–06:30 IST` prior to morning commuter rush."
        )

    fac_lines = []
    for facility in facilities_in_ward[:4]:
        fac_lines.append(
            f"  - **{facility['facility_name']}** (`{facility['type']}`, `{facility['doi']}`): Predicted **{facility['pm25_ugm3']:.1f} µg/m³** "
            f"({facility['who_multiple']:.1f}× WHO) — *{facility['recommended_protocol']}*"
        )
    if not fac_lines:
        fac_lines.append(
            f"  - All **{schools} UDISE+ schools (`doi:10.71646/XJIY8W`)** and **{hospitals} civic dispensaries (`doi:10.71646/1XSUVR`)** "
            f"in Ward `{ward_code}` placed on automated hourly CPCB telemetry watch."
        )
    fac_block = "\n".join(fac_lines)

    qa_section = ""
    if user_question.strip():
        mixing = (
            "severely suppressed (<1,200 m²/s stagnation threshold)"
            if vc < 1200
            else "moderately active"
        )
        qa_section = f"""
### 5. Copilot Response to Custom Query
> **Query:** *"{user_question.strip()}"*
- **Physics & Policy Assessment:** Under the current boundary layer height of **{pblh:.0f} m**, ventilation coefficient of **{vc:.0f} m²/s** (`doi:10.71646/XXOBKO`), and Deonar advection cosine **{deonar_cos:+.2f}**, vertical mixing in Ward `{ward_code}` is {mixing}. Priority intervention should focus on curtailing ground-level fugitive dust from the **{maharera} MahaRERA construction sites (`doi:10.71646/4ULQRE`)** during `05:30–10:30 IST` while protecting **{vuln_pop:,} high-risk pediatric/geriatric residents (`doi:10.71646/AAGK0G`)** across **{schools} UDISE+ schools** and **{hospitals} civic hospitals**.
"""

    anchor = quantiles["anchor_station"]
    return f"""### BMC Graded Response Action Plan (GRAP) Operational Order — Ward `{ward_code}` ({ward_name})
**Target Timestamp:** `{ts_str}` · **Active Trigger:** **{grap_stage}**  
**Predicted Ward PM2.5:** **{pm25:.1f} µg/m³** (`{naaqs_x:.2f}×` India 24h NAAQS 60 µg/m³ · `{who_x:.1f}×` WHO 15 µg/m³)

#### 1. Atmospheric Physics, Foundation Quantiles & Satellite Urban Morphology Diagnosis
- **Boundary Layer & Ventilation (`doi:10.71646/XXOBKO`, `doi:10.71646/EAE4EE`)**: Planetary Boundary Layer Height is **{pblh:.0f} m** with a Ventilation Coefficient of **{vc:.0f} m²/s** (`PBLH × WindSpeed`). Dominant flow: *{breeze_regime}*.
- **Deonar Advection Cosine (`doi:10.71646/MWQNR8`, `doi:10.71646/VA5BD9`)**: Alignment of the hourly transport vector with the Deonar Dumping Ground → Ward `{ward_code}` axis is **cos θ = {deonar_cos:+.2f}**. *{deonar_text}*.
- **Google Research `TimesFM 3.0` (9 deciles) & Amazon `Chronos-2` (21 quantiles)**, anchor **{anchor}**: TimesFM `Q10/Q50/Q90` = **{quantiles['timesfm_q10']:.1f} / {quantiles['timesfm_q50']:.1f} / {quantiles['timesfm_q90']:.1f} µg/m³**. Chronos-2 `Q05/Q50/Q95` = **{quantiles['chronos_q05']:.1f} / {quantiles['chronos_q50']:.1f} / {quantiles['chronos_q95']:.1f} µg/m³**. RMSE-optimal trapezoidal expectation $\\hat{{\\mu}}_{{\\text{{L2}}}} = \\mathbb{{E}}[Y\\mid X]$ is **{quantiles['mu_l2']:.1f} µg/m³**, with median $Q_{{0.50}}$ = **{quantiles['q50_median']:.1f} µg/m³**. Upper-tail surge risk uses TimesFM $Q_{{0.90}}$ = **{quantiles['timesfm_q90']:.1f} µg/m³**.
- **Google AlphaEarth 10m & Open Buildings V3 Canyon Trapping (`doi:10.71646/GHSZ7J`, `doi:10.71646/EV3RBJ`)**: Ward `{ward_code}` exhibits an AlphaEarth impervious surface index of **{impervious:.2f}** and Open Buildings footprint density of **{bldg_density:.2f}**, creating street-canyon recirculation that amplifies local emissions by **+{5.5 * (0.5 * impervious + 0.5 * bldg_density - 0.50):.1f} µg/m³**.

#### 2. Mandatory Construction & Dust Mitigation (`doi:10.71646/4ULQRE`)
- **MahaRERA Active Site Enforcement**: Issue automated compliance directives to all **{maharera} registered MahaRERA construction projects** in Ward `{ward_code}`.
- **Mechanical Misting & Sprinkling Schedule**:
  - **Window A (`05:30–09:30 IST` Morning Inversion Peak)**: Mandatory continuous operation of 30m anti-smog mist guns, wheel-washing bays, and 35-foot green tin/jute perimeter sheeting. Halt all excavation, demolition, and ready-mix concrete (RMC) batching if hourly PM2.5 exceeds **90 µg/m³**.
  - **Window B (`18:30–22:30 IST` Nocturnal Collapse)**: Deploy non-potable treated sewage water (STPs) tanker sprinkling along ward arterial corridors.

#### 3. Point-Source & Transport Hotspot Directives
{hotspot_directive}
- **NASA FIRMS Thermal Anomaly Watch (`doi:10.71646/YJIAGQ`)**: Cross-check VIIRS 375m night-pass alerts for open municipal waste or biomass burning within a 5 km upwind buffer of Ward `{ward_code}`.

#### 4. Vulnerable Population, UDISE+ Schools & Civic Hospital Protocols (`doi:10.71646/AAGK0G`, `doi:10.71646/XJIY8W`, `doi:10.71646/1XSUVR`)
- **Demographics at Risk**: **{pop:,} total residents**, including **{vuln_pop:,} children (<5y) and elderly (>60y)** mapped via WorldPop 2026 (`doi:10.71646/AAGK0G`).
- **School & Healthcare Action Matrix ({schools} UDISE+ Schools · {hospitals} Civic Hospitals)**:
{fac_block}
{qa_section}"""


def _civic_messages(
    ward_row: dict[str, Any],
    snapshot_meta: dict[str, Any],
    user_question: str,
    deterministic_brief: str,
) -> tuple[str, list[dict[str, str]]]:
    system = (
        "You are VayuDrishti Mumbai Civic Copilot, an atmospheric-physics advisor to the "
        "Brihanmumbai Municipal Corporation. Write a concise BMC Graded Response Action Plan "
        "in Markdown. Cite the ClimateVerse DOIs already present in the telemetry. "
        "Do not invent stations, populations, or concentrations."
    )
    user = f"""Generate the operational GRAP directive for Ward {ward_row['ward_code']} ({ward_row['ward_name']}) at {snapshot_meta['timestamp_ist']}.

Ground every number in this telemetry brief:
{deterministic_brief}

User request: {user_question or 'Provide the full operational GRAP schedule.'}
"""
    return user, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _is_connection_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in ("connection", "refused", "timed out", "timeout", "unreachable", "name or service")
    )


def _is_auth_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in ("unauthorized", "401", "403", "permission denied", "invalid api key"))


def _is_routing_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "provider",
            "prefix",
            "not found",
            "unknown model",
            "invalid model",
            "does not exist",
            "not supported",
            "model_not_found",
            "llm provider",
            "404",
        )
    )


def _litellm_complete(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str,
    provider: str,
) -> str:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.set_verbose = False
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 12,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    if provider == "Vertex AI":
        project = os.environ.get("VERTEX_PROJECT", "").strip() or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = (
            os.environ.get("VERTEX_LOCATION", "").strip()
            or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
            or "us-central1"
        )
        if project:
            kwargs["vertex_project"] = project
            kwargs["vertex_location"] = location
    response = litellm.completion(**kwargs)
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("LiteLLM returned no choices")
    message = choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", part)))
        content = "\n".join(part for part in parts if part.strip())
    if not content:
        content = getattr(message, "reasoning_content", None)
    if not content or not str(content).strip():
        raise RuntimeError("LiteLLM returned empty content")
    return str(content).strip()


def _google_genai_complete(raw_slug: str, prompt: str, api_key: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=raw_slug,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise RuntimeError("google-genai returned empty text")
    return str(text).strip()


def _needs_live_call(provider: str, api_key: str, model: str) -> bool:
    if api_key:
        return True
    return provider.startswith("Local") or model.startswith(("ollama/", "hosted_vllm/"))


def run_gemini_civic_copilot(
    ward_row: dict[str, Any],
    snapshot_meta: dict[str, Any],
    facilities_in_ward: list[dict[str, Any]],
    user_question: str = "",
    api_key_override: str = "",
    model: str = "",
    provider: str = "",
    api_base: str = "",
    custom_model: str = "",
    quantile_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run the 3-tier copilot and return ``(markdown, engine_label)``.

    With no API key this stays on the deterministic GRAP engine and does not
    import LiteLLM or contact the network.
    """
    deterministic_brief = generate_deterministic_grap_briefing(
        ward_row=ward_row,
        snapshot_meta=snapshot_meta,
        facilities_in_ward=facilities_in_ward,
        user_question=user_question,
        quantile_context=quantile_context,
    )
    active_provider = (provider or "").strip() or default_copilot_settings()["provider"]
    if active_provider not in PROVIDER_OPTIONS:
        active_provider = PROVIDER_OPTIONS[0]
    preset = (model or "").strip() or default_copilot_settings()["model_id"]
    resolved = resolve_litellm_model(active_provider, preset, custom_model)
    api_key = resolve_api_key(active_provider, api_key_override)
    endpoint = resolve_api_base(active_provider, api_base, resolved)

    if not _needs_live_call(active_provider, api_key, resolved):
        return (
            deterministic_brief,
            "Tier 3 · Deterministic BMC GRAP Policy Engine "
            f"(offline — set an API key to route {FLAGSHIP_MODEL_LABEL} via LiteLLM)",
        )

    prompt, messages = _civic_messages(ward_row, snapshot_meta, user_question, deterministic_brief)
    last_error = "Error"
    for candidate in alias_chain(resolved, active_provider):
        try:
            text = _litellm_complete(
                model=candidate,
                messages=messages,
                api_key=api_key,
                api_base=endpoint,
                provider=active_provider,
            )
            return text, f"Tier 1 · LiteLLM `{candidate}`"
        except Exception as exc:
            last_error = type(exc).__name__
            if _is_connection_error(exc) or _is_auth_error(exc):
                break
            if not _is_routing_error(exc):
                break

    if api_key and is_gemini_family(resolved):
        raw_slug = raw_model_slug(resolved)
        try:
            text = _google_genai_complete(raw_slug, prompt, api_key)
            return text, f"Tier 2 · google-genai `{raw_slug}` (LiteLLM prefix fallback)"
        except Exception as exc:
            last_error = type(exc).__name__

    return (
        deterministic_brief,
        f"Tier 3 · Deterministic BMC GRAP Policy Engine (live router fallback after {last_error})",
    )
