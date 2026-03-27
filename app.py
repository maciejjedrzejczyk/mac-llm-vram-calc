"""
LLM VRAM Calculator Dashboard
Uses LM Studio's lms CLI as primary data source, with fallback to file scanning.
Multi-step wizard UI with direct dashboard access.
"""

import streamlit as st
import plotly.graph_objects as go
import os
import time

from gguf_scanner import scan_models, get_lm_studio_models_dir, ModelInfo
from vram_calc import estimate_vram, quant_to_bpw, VRAMEstimate
from lmstudio_config import read_model_config
from system_info import detect_system_memory
from lms_cli import (
    is_available as lms_available,
    list_models as lms_list_models,
    list_loaded_models as lms_list_loaded,
    read_global_settings,
    get_loaded_runtime_configs,
    LMSModelInfo,
    LMSRuntimeConfig,
)
from benchmark import (
    check_server,
    get_all_model_ids,
    get_model_catalog,
    load_model,
    unload_model,
    unload_all_models,
    run_simulation,
    run_multi_model_comparison,
    annotate_session_vram,
    trace_via_log_stream,
    parse_trace_entry,
    SessionStats,
    RoundStats,
    ComparisonResult,
    ModelCatalogEntry,
    TOPIC_PRESETS,
)

st.set_page_config(page_title="LLM VRAM Calculator", page_icon="🧠", layout="wide")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #3d3d5c;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c8aff; }
    .metric-label { font-size: 0.85rem; color: #a0a0b8; margin-top: 0.3rem; }
    .fit-yes { color: #4ade80; font-weight: 700; }
    .fit-tight { color: #facc15; font-weight: 700; }
    .fit-no { color: #f87171; font-weight: 700; }
    .tag { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 6px;
           font-size: 0.75rem; font-weight: 600; margin-right: 0.3rem; }
    .tag-vision { background: #3b2f7a; color: #a78bfa; }
    .tag-tools { background: #1e3a2f; color: #6ee7b7; }
    .tag-loaded { background: #3b1e1e; color: #fca5a5; }
    .tag-moe { background: #1e2e3b; color: #7dd3fc; }
    .rope-warning { background: #2d2a1e; border: 1px solid #facc15; border-radius: 8px;
                    padding: 0.6rem 1rem; margin: 0.5rem 0; font-size: 0.85rem; color: #fde68a; }
    .sweet-spot { background: #1e2e1e; border: 1px solid #4ade80; border-radius: 8px;
                  padding: 0.6rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #bbf7d0; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'


def fit_badge(total_gb: float, available_gb: float) -> str:
    if total_gb <= available_gb * 0.85:
        return '<span class="fit-yes">✅ Fits comfortably</span>'
    elif total_gb <= available_gb:
        return '<span class="fit-tight">⚠️ Tight fit</span>'
    return '<span class="fit-no">❌ Won\'t fit</span>'


def _parse_params(s: str) -> int:
    if not s:
        return 0
    s = s.strip().upper()
    if "-" in s:
        s = s.split("-")[0]
    try:
        if s.endswith("B"):
            return int(float(s[:-1]) * 1e9)
        elif s.endswith("M"):
            return int(float(s[:-1]) * 1e6)
    except (ValueError, IndexError):
        pass
    return 0


# ── Unified Model Wrapper ────────────────────────────────────────────────────

class UnifiedModel:
    """Common model representation regardless of data source."""
    def __init__(self, *, name="", display_name="", model_key="", format="",
                 architecture="", params_string="", size_bytes=0,
                 quant_name="", quant_bits=0, bits_per_weight=0.0,
                 max_context_length=0, is_vision=False, is_tool_use=False,
                 hidden_dim=0, num_layers=0, num_heads=0, num_kv_heads=0,
                 attention_layer_count=0, parameter_count=0,
                 feed_forward_length=0,
                 expert_count=0, expert_used_count=0,
                 rope_scaling_type="", rope_scaling_factor=0.0,
                 rope_original_context_length=0,
                 variants=None,
                 source="", path=""):
        self.name = name
        self.display_name = display_name
        self.model_key = model_key
        self.format = format
        self.architecture = architecture
        self.params_string = params_string
        self.size_bytes = size_bytes
        self.quant_name = quant_name
        self.quant_bits = quant_bits
        self.bits_per_weight = bits_per_weight or quant_to_bpw(quant_name, quant_bits)
        self.max_context_length = max_context_length
        self.is_vision = is_vision
        self.is_tool_use = is_tool_use
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attention_layer_count = attention_layer_count
        self.parameter_count = parameter_count or _parse_params(params_string)
        self.feed_forward_length = feed_forward_length
        self.expert_count = expert_count
        self.expert_used_count = expert_used_count
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_original_context_length = rope_original_context_length
        self.variants = variants or []
        self.source = source
        self.path = path

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1 and self.expert_used_count > 0

    @staticmethod
    def from_lms(m: LMSModelInfo) -> "UnifiedModel":
        return UnifiedModel(
            name=m.model_key, display_name=m.display_name, model_key=m.model_key,
            format=m.format, architecture=m.architecture, params_string=m.params_string,
            size_bytes=m.size_bytes, quant_name=m.quant_name, quant_bits=m.quant_bits,
            max_context_length=m.max_context_length, is_vision=m.is_vision,
            is_tool_use=m.trained_for_tool_use, parameter_count=_parse_params(m.params_string),
            variants=m.variants, source="lms", path=m.path,
        )

    @staticmethod
    def from_scanner(m: ModelInfo) -> "UnifiedModel":
        return UnifiedModel(
            name=m.file_name, display_name=m.model_name or m.file_name, model_key="",
            format=m.model_format, architecture=m.architecture,
            params_string=f"{m.parameter_count / 1e9:.1f}B" if m.parameter_count else "",
            size_bytes=m.file_size_bytes, quant_name=m.quantization_type,
            bits_per_weight=m.bits_per_weight, max_context_length=m.context_length,
            hidden_dim=m.embedding_length, num_layers=m.num_layers,
            num_heads=m.num_heads, num_kv_heads=m.num_kv_heads,
            attention_layer_count=m.metadata.get("_attention_layer_count", 0),
            parameter_count=m.parameter_count, feed_forward_length=m.feed_forward_length,
            expert_count=m.expert_count, expert_used_count=m.expert_used_count,
            rope_scaling_type=m.rope_scaling_type, rope_scaling_factor=m.rope_scaling_factor,
            rope_original_context_length=m.rope_original_context_length,
            source="scanner", path=m.file_path,
        )


# ── Model Loading ────────────────────────────────────────────────────────────

def load_models(models_dir: str) -> tuple[list[UnifiedModel], set[str], dict]:
    """Scan and return (models, loaded_keys, runtime_configs)."""
    has_lms = lms_available()
    unified: list[UnifiedModel] = []

    if has_lms:
        lms_models = [m for m in lms_list_models() if m.model_type == "llm"]
        unified = [UnifiedModel.from_lms(m) for m in lms_models]
        scanner_models = scan_models(models_dir)
        for um in unified:
            for sm in scanner_models:
                if (sm.architecture == um.architecture or
                        um.name.lower() in sm.file_path.lower() or
                        sm.file_name.lower().startswith(um.name.lower().replace("/", "-"))):
                    um.hidden_dim = um.hidden_dim or sm.embedding_length
                    um.num_layers = um.num_layers or sm.num_layers
                    um.num_heads = um.num_heads or sm.num_heads
                    um.num_kv_heads = um.num_kv_heads or sm.num_kv_heads
                    um.attention_layer_count = um.attention_layer_count or sm.metadata.get("_attention_layer_count", 0)
                    um.feed_forward_length = um.feed_forward_length or sm.feed_forward_length
                    um.expert_count = um.expert_count or sm.expert_count
                    um.expert_used_count = um.expert_used_count or sm.expert_used_count
                    um.rope_scaling_type = um.rope_scaling_type or sm.rope_scaling_type
                    um.rope_scaling_factor = um.rope_scaling_factor or sm.rope_scaling_factor
                    um.rope_original_context_length = um.rope_original_context_length or sm.rope_original_context_length
                    break
    else:
        unified = [UnifiedModel.from_scanner(m) for m in scan_models(models_dir)]

    loaded_keys = {lm.model_key for lm in lms_list_loaded()}
    runtime_configs = get_loaded_runtime_configs()
    return unified, loaded_keys, runtime_configs


def ensure_models_loaded():
    """Load models into session state if not already present."""
    if "models" not in st.session_state or not st.session_state.models:
        global_settings = read_global_settings()
        models_dir = st.session_state.get("models_dir",
                     global_settings.downloads_folder or get_lm_studio_models_dir())
        with st.spinner("Scanning models..."):
            models, loaded_keys, runtime_configs = load_models(models_dir)
        st.session_state.models = models
        st.session_state.loaded_keys = loaded_keys
        st.session_state.runtime_configs = runtime_configs


# ── Navigation ───────────────────────────────────────────────────────────────

def go_to(page: str):
    st.session_state.page = page

# Initialize session state defaults (auto-detect system memory)
if "page" not in st.session_state:
    st.session_state.page = "home"
if "system_detected" not in st.session_state:
    sys_info = detect_system_memory()
    st.session_state.system_info = sys_info
    st.session_state.system_ram_gb = max(4, int(sys_info["total_ram_gb"])) if sys_info["total_ram_gb"] else 32
    st.session_state.gpu_available_pct = sys_info["estimated_available_pct"] or 75
    st.session_state.system_detected = True


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("🧠 LLM VRAM Calculator")
    st.caption("Estimate memory requirements for running local LLMs on Apple Silicon")

    st.markdown("")  # spacer

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        ### 🔧 Setup Wizard
        Step-by-step configuration:
        1. Configure your system memory
        2. Select a model to analyze
        3. View detailed VRAM estimate
        """)
        st.button("Start Setup →", type="primary", use_container_width=True,
                   on_click=go_to, args=("setup",))

    with col2:
        st.markdown("""
        ### 📊 Dashboard
        Jump straight to the full comparison
        of all your local models with VRAM
        estimates at a glance.
        """)
        st.button("Go to Dashboard →", use_container_width=True,
                   on_click=go_to, args=("dashboard",))

    st.markdown("")
    col3, _ = st.columns([1, 1])
    with col3:
        st.markdown("""
        ### 🏋️ Trace & Simulate
        Benchmark a loaded model with live
        inference stats — passively trace your
        conversations or run automated simulations.
        """)
        st.button("Open Benchmark →", use_container_width=True,
                   on_click=go_to, args=("benchmark",))

    # Quick status
    st.divider()
    has_lms = lms_available()
    sys_info = st.session_state.get("system_info", {})
    if sys_info.get("chip"):
        ram = sys_info.get("total_ram_gb", 0)
        vram_note = ""
        if sys_info.get("gpu_vram_limit_gb"):
            vram_note = f" · VRAM limit: {sys_info['gpu_vram_limit_gb']:.1f} GB"
        st.info(f"🖥️ {sys_info['chip']} — {ram:.0f} GB unified memory{vram_note}")
    if has_lms:
        st.success("✅ LM Studio `lms` CLI detected — rich model metadata available")
    else:
        st.warning("⚠️ `lms` CLI not found — will use file scanner (less metadata)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Setup (Step 1 — System Configuration)
# ══════════════════════════════════════════════════════════════════════════════

def page_setup():
    st.title("🔧 Step 1: System Configuration")
    st.caption("Tell us about your hardware so we can estimate VRAM accurately.")

    global_settings = read_global_settings()
    has_lms = lms_available()
    sys_info = st.session_state.get("system_info", {})

    # Data source indicator
    if has_lms:
        st.success("✅ lms CLI detected — using rich model data")
    else:
        st.warning("⚠️ lms CLI not found — using file scanner (less metadata)")

    # Auto-detected system info
    if sys_info.get("is_apple_silicon"):
        chip = sys_info.get("chip", "Apple Silicon")
        st.info(f"🖥️ Detected: **{chip}** — {sys_info.get('total_ram_gb', 0):.0f} GB unified memory")
        if sys_info.get("gpu_vram_limit_mb"):
            st.warning(
                f"⚡ Manual GPU VRAM limit detected: **{sys_info['gpu_vram_limit_gb']:.1f} GB** "
                f"(`iogpu.wired_limit_mb = {sys_info['gpu_vram_limit_mb']}`)"
            )

    st.subheader("Your System")

    col1, col2 = st.columns(2)
    with col1:
        detected_ram = max(4, int(sys_info.get("total_ram_gb", 0))) if sys_info.get("total_ram_gb") else 32
        system_ram_gb = st.number_input(
            "Total Unified Memory (GB)", min_value=4, max_value=512,
            value=st.session_state.system_ram_gb, step=8,
            help=f"Auto-detected: {detected_ram} GB. Apple Silicon shares RAM between CPU and GPU.",
        )
    with col2:
        detected_pct = sys_info.get("estimated_available_pct", 75)
        gpu_available_pct = st.slider(
            "% Available for LLM", 50, 95,
            value=st.session_state.gpu_available_pct,
            help=f"Auto-detected: {detected_pct}%."
                 + (f" Based on iogpu.wired_limit_mb = {sys_info['gpu_vram_limit_mb']}"
                    if sys_info.get("gpu_vram_limit_mb") else
                    " Default heuristic: ~75% of unified memory for GPU."),
        )

    available_vram_gb = system_ram_gb * gpu_available_pct / 100
    st.markdown(f"**Available for LLM:** {available_vram_gb:.1f} GB")

    if global_settings.guardrail_mode:
        threshold_gb = global_settings.guardrail_threshold_bytes / (1024**3)
        st.caption(f"LM Studio guardrail: {global_settings.guardrail_mode} (threshold: {threshold_gb:.0f} GB)")

    st.divider()

    # Model directory
    if not has_lms:
        st.subheader("Model Directory")
        default_dir = global_settings.downloads_folder or get_lm_studio_models_dir()
        models_dir = st.text_input("LM Studio models path", value=default_dir)
    else:
        models_dir = global_settings.downloads_folder or get_lm_studio_models_dir()

    st.markdown("")  # spacer

    col_back, col_next = st.columns([1, 1])
    with col_back:
        st.button("← Back to Home", use_container_width=True, on_click=go_to, args=("home",))
    with col_next:
        def _save_setup_and_next():
            st.session_state.system_ram_gb = system_ram_gb
            st.session_state.gpu_available_pct = gpu_available_pct
            st.session_state.models_dir = models_dir
            # Force model reload with new directory
            st.session_state.pop("models", None)
            go_to("model_select")
        st.button("Next: Select Model →", type="primary", use_container_width=True,
                   on_click=_save_setup_and_next)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Selection (Step 2)
# ══════════════════════════════════════════════════════════════════════════════

def page_model_select():
    st.title("🔍 Step 2: Select Model")
    st.caption("Choose a model and adjust parameters that affect VRAM usage.")

    ensure_models_loaded()
    models: list[UnifiedModel] = st.session_state.models
    loaded_keys: set[str] = st.session_state.loaded_keys
    available_vram_gb = st.session_state.system_ram_gb * st.session_state.gpu_available_pct / 100

    if not models:
        st.info("No models found. Check your model directory in Setup.")
        st.button("← Back to Setup", on_click=go_to, args=("setup",))
        st.stop()

    # Refresh button
    def _refresh():
        st.session_state.pop("models", None)
    st.button("🔄 Refresh Models", on_click=_refresh)

    # Format filter
    available_formats = sorted(set(m.format for m in models))
    if len(available_formats) > 1:
        format_filter = st.radio("Format", ["All"] + available_formats, horizontal=True)
    else:
        format_filter = "All"

    filtered = [m for m in models if format_filter == "All" or m.format == format_filter]

    # Build display names
    model_display = {}
    for m in filtered:
        size_gb = m.size_bytes / (1024**3)
        tags = f"[{m.format}]"
        if m.is_moe:
            tags += " 🧩"
        if m.is_vision:
            tags += " 👁️"
        if m.model_key in loaded_keys:
            tags += " 🟢"
        display = f"{tags} {m.display_name}  ({size_gb:.1f} GB, {m.quant_name})"
        model_display[display] = m

    if not model_display:
        st.warning("No models match the selected filter.")
        st.stop()

    selected_display = st.selectbox("Select a model", list(model_display.keys()))
    sel = model_display[selected_display]

    # Tags
    tags_html = ""
    if sel.is_moe:
        tags_html += f'<span class="tag tag-moe">🧩 MoE {sel.expert_used_count}/{sel.expert_count} experts</span>'
    if sel.is_vision:
        tags_html += '<span class="tag tag-vision">👁️ Vision</span>'
    if sel.is_tool_use:
        tags_html += '<span class="tag tag-tools">🔧 Tool Use</span>'
    if sel.model_key in loaded_keys:
        tags_html += '<span class="tag tag-loaded">🟢 Loaded</span>'
    if tags_html:
        st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()

    # Adjustable parameters
    global_settings = read_global_settings()
    config_path = sel.model_key if sel.model_key else sel.path
    lms_config = read_model_config(config_path)

    st.subheader("Inference Parameters")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        default_ctx = lms_config.context_length or global_settings.default_context_length or 4096
        ctx_options = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        default_ctx_snapped = min(ctx_options, key=lambda x: abs(x - default_ctx))
        ctx_length = st.select_slider("Context Length", options=ctx_options, value=default_ctx_snapped,
            help=f"Max supported: {sel.max_context_length:,}" if sel.max_context_length else "")

    with col_b:
        kv_source = st.radio("KV Cache Precision", ["Manual", "From LM Studio"], horizontal=True)
        if kv_source == "From LM Studio":
            if lms_config.kv_cache_quant_enabled is not None:
                if lms_config.kv_cache_quant_enabled:
                    kv_bits = lms_config.kv_cache_quant_bits or 8
                    st.caption(f"✅ KV quant ON — {kv_bits}-bit (group {lms_config.kv_cache_quant_group_size})")
                else:
                    kv_bits = 16
                    st.caption("KV quant OFF — FP16")
            else:
                kv_bits = 16
                st.caption("⚠️ No config found — defaulting to FP16")
        else:
            kv_bits = st.selectbox("Bits", [16, 8, 4], index=0, label_visibility="collapsed")

    with col_c:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=1)

    with col_d:
        override_bpw = st.number_input("Bits per Weight",
            min_value=1.0, max_value=32.0, value=float(sel.bits_per_weight), step=0.5)

    st.markdown("")

    col_back, col_next = st.columns([1, 1])
    with col_back:
        st.button("← Back to Setup", use_container_width=True, on_click=go_to, args=("setup",))
    with col_next:
        def _save_model_and_next():
            st.session_state.sel_display = selected_display
            st.session_state.sel_model_display_map = model_display
            st.session_state.ctx_length = ctx_length
            st.session_state.kv_bits = kv_bits
            st.session_state.batch_size = batch_size
            st.session_state.override_bpw = override_bpw
            st.session_state.format_filter = format_filter
            go_to("vram_estimate")
        st.button("Next: View VRAM Estimate →", type="primary", use_container_width=True,
                   on_click=_save_model_and_next)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VRAM Estimate (Step 3)
# ══════════════════════════════════════════════════════════════════════════════

def page_vram_estimate():
    st.title("📊 Step 3: VRAM Estimate")

    # Recover selected model from session state
    model_display = st.session_state.get("sel_model_display_map", {})
    sel_display = st.session_state.get("sel_display", "")
    if not model_display or sel_display not in model_display:
        st.warning("No model selected. Please go back to Step 2.")
        st.button("← Back to Model Selection", on_click=go_to, args=("model_select",))
        st.stop()

    sel = model_display[sel_display]
    ctx_length = st.session_state.get("ctx_length", 4096)
    kv_bits = st.session_state.get("kv_bits", 16)
    batch_size = st.session_state.get("batch_size", 1)
    override_bpw = st.session_state.get("override_bpw", sel.bits_per_weight)
    available_vram_gb = st.session_state.system_ram_gb * st.session_state.gpu_available_pct / 100
    loaded_keys: set[str] = st.session_state.get("loaded_keys", set())
    runtime_configs: dict[str, LMSRuntimeConfig] = st.session_state.get("runtime_configs", {})

    st.caption(f"Model: **{sel.display_name}** — {sel.quant_name} ({override_bpw:.1f} bpw) — Context: {ctx_length:,}")

    # Tags
    tags_html = ""
    if sel.is_moe:
        tags_html += f'<span class="tag tag-moe">🧩 MoE {sel.expert_used_count}/{sel.expert_count} experts</span>'
    if sel.is_vision:
        tags_html += '<span class="tag tag-vision">👁️ Vision</span>'
    if sel.is_tool_use:
        tags_html += '<span class="tag tag-tools">🔧 Tool Use</span>'
    if sel.model_key in loaded_keys:
        tags_html += '<span class="tag tag-loaded">🟢 Loaded</span>'
    if tags_html:
        st.markdown(tags_html, unsafe_allow_html=True)

    # Calculate
    est = estimate_vram(
        parameter_count=sel.parameter_count, bits_per_weight=override_bpw,
        context_length=ctx_length, hidden_dim=sel.hidden_dim, num_layers=sel.num_layers,
        num_heads=sel.num_heads, num_kv_heads=sel.num_kv_heads,
        kv_cache_bits=kv_bits, batch_size=batch_size,
        file_size_bytes=sel.size_bytes, attention_layer_count=sel.attention_layer_count,
        is_vision=sel.is_vision, expert_count=sel.expert_count,
        expert_used_count=sel.expert_used_count,
    )

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("Total VRAM", f"{est.total_gb:.1f} GB"), unsafe_allow_html=True)
    c2.markdown(metric_card("Model Weights", f"{est.model_weights_gb:.1f} GB"), unsafe_allow_html=True)
    c3.markdown(metric_card("KV Cache", f"{est.kv_cache_gb:.2f} GB"), unsafe_allow_html=True)
    overhead = est.activation_overhead_gb + est.runtime_overhead_gb + est.vision_overhead_gb
    c4.markdown(metric_card("Overhead", f"{overhead:.2f} GB"), unsafe_allow_html=True)
    c5.markdown(metric_card("Fit?", fit_badge(est.total_gb, available_vram_gb)), unsafe_allow_html=True)

    # Usage bar
    usage_pct = min(est.total_gb / available_vram_gb * 100, 100) if available_vram_gb > 0 else 100
    bar_color = "#4ade80" if usage_pct < 85 else "#facc15" if usage_pct <= 100 else "#f87171"
    st.markdown(f"""
<div style="background: #1e1e2e; border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
    <div style="background: {bar_color}; width: {min(usage_pct, 100):.0f}%; height: 24px; border-radius: 6px;
                display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; color: #1e1e2e;">
        {est.total_gb:.1f} / {available_vram_gb:.1f} GB ({usage_pct:.0f}%)
    </div>
</div>
""", unsafe_allow_html=True)

    # RoPE Scaling Warning
    if sel.rope_scaling_type and sel.rope_original_context_length:
        native_ctx = sel.rope_original_context_length
        if ctx_length > native_ctx:
            st.markdown(f"""
<div class="rope-warning">
    ⚠️ <b>RoPE scaling active</b> — Trained at <b>{native_ctx:,}</b> context,
    extended to {sel.max_context_length:,} via <b>{sel.rope_scaling_type}</b>
    ({sel.rope_scaling_factor:.1f}×).
    Your selected context ({ctx_length:,}) exceeds native training length.
    Quality may degrade beyond {native_ctx:,} tokens.
</div>""", unsafe_allow_html=True)
        else:
            st.caption(f"ℹ️ RoPE: native {native_ctx:,} context, extended via {sel.rope_scaling_type} ({sel.rope_scaling_factor:.1f}×)")
    elif sel.rope_scaling_type:
        st.caption(f"ℹ️ RoPE scaling: {sel.rope_scaling_type}" +
                   (f" ({sel.rope_scaling_factor:.1f}×)" if sel.rope_scaling_factor else ""))

    # Context Sweet Spot
    ctx_opts_for_sweet = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    sweet_spot_ctx = 0
    for ctx in reversed(ctx_opts_for_sweet):
        e = estimate_vram(
            parameter_count=sel.parameter_count, bits_per_weight=override_bpw,
            context_length=ctx, hidden_dim=sel.hidden_dim, num_layers=sel.num_layers,
            num_heads=sel.num_heads, num_kv_heads=sel.num_kv_heads,
            kv_cache_bits=kv_bits, batch_size=batch_size,
            file_size_bytes=sel.size_bytes, attention_layer_count=sel.attention_layer_count,
            is_vision=sel.is_vision, expert_count=sel.expert_count,
            expert_used_count=sel.expert_used_count,
        )
        if e.total_gb <= available_vram_gb * 0.85:
            sweet_spot_ctx = ctx
            break

    if sweet_spot_ctx > 0:
        st.markdown(f"""
<div class="sweet-spot">
    🎯 <b>Max comfortable context: {sweet_spot_ctx:,} tokens</b> — fits within 85% of your {available_vram_gb:.0f} GB budget
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="rope-warning">
    ⚠️ This model doesn't fit comfortably at any context length with your current memory budget.
</div>""", unsafe_allow_html=True)

    # Runtime Config Comparison
    rt_cfg = runtime_configs.get(sel.model_key)
    if rt_cfg and rt_cfg.context_length:
        with st.expander("⚡ Loaded Model — Estimated vs Actual Config", expanded=True):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Your Estimate**")
                st.markdown(f"| Setting | Value |\n|---|---|\n| Context Length | {ctx_length:,} |\n| KV Cache | {kv_bits}-bit |\n| Batch Size | {batch_size} |\n| VRAM Estimate | {est.total_gb:.1f} GB |")
            with rc2:
                st.markdown("**Actual (LM Studio Server)**")
                flash_str = "✅ On" if rt_cfg.flash_attention else "❌ Off"
                kv_offload_str = "GPU" if rt_cfg.offload_kv_cache_to_gpu else "CPU (RAM)"
                experts_str = str(rt_cfg.num_experts) if rt_cfg.num_experts else "N/A"
                st.markdown(f"| Setting | Value |\n|---|---|\n| Context Length | {rt_cfg.context_length:,} |\n| Flash Attention | {flash_str} |\n| KV Cache Location | {kv_offload_str} |\n| Active Experts | {experts_str} |")
            if rt_cfg.context_length != ctx_length:
                st.caption(f"💡 The loaded model is using {rt_cfg.context_length:,} context — adjust the slider to match for an accurate estimate.")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        pie_labels = ["Model Weights", "KV Cache", "Activation", "Runtime"]
        pie_values = [est.model_weights_gb, est.kv_cache_gb, est.activation_overhead_gb, est.runtime_overhead_gb]
        pie_colors = ["#7c8aff", "#4ade80", "#facc15", "#f87171"]
        if est.vision_overhead_gb > 0:
            pie_labels.append("Vision Encoder")
            pie_values.append(est.vision_overhead_gb)
            pie_colors.append("#a78bfa")
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_values, hole=0.5,
            marker_colors=pie_colors, textinfo="label+percent",
        )])
        fig_pie.update_layout(title="VRAM Breakdown", template="plotly_dark", height=350,
                              margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        ctx_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        ctx_vrams = []
        for ctx in ctx_opts:
            e = estimate_vram(
                parameter_count=sel.parameter_count, bits_per_weight=override_bpw,
                context_length=ctx, hidden_dim=sel.hidden_dim, num_layers=sel.num_layers,
                num_heads=sel.num_heads, num_kv_heads=sel.num_kv_heads,
                kv_cache_bits=kv_bits, batch_size=batch_size,
                file_size_bytes=sel.size_bytes, attention_layer_count=sel.attention_layer_count,
                is_vision=sel.is_vision, expert_count=sel.expert_count,
                expert_used_count=sel.expert_used_count,
            )
            ctx_vrams.append(e.total_gb)
        fig_ctx = go.Figure()
        fig_ctx.add_trace(go.Scatter(
            x=[str(c) for c in ctx_opts], y=ctx_vrams,
            mode="lines+markers", line=dict(color="#7c8aff", width=2), marker=dict(size=8),
        ))
        fig_ctx.add_hline(y=available_vram_gb, line_dash="dash", line_color="#f87171",
                          annotation_text=f"Available: {available_vram_gb:.0f} GB")
        fig_ctx.update_layout(title="VRAM vs Context Length", xaxis_title="Context (tokens)",
                              yaxis_title="VRAM (GB)", template="plotly_dark", height=350,
                              margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_ctx, use_container_width=True)

    # Model Details
    with st.expander("📋 Model Details", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            moe_str = f"{sel.expert_used_count}/{sel.expert_count} experts active" if sel.is_moe else "No (dense)"
            st.markdown(f"| Property | Value |\n|---|---|\n| Name | {sel.display_name} |\n| Format | {sel.format} |\n| Size on Disk | {sel.size_bytes / (1024**3):.2f} GB |\n| Architecture | {sel.architecture or 'Unknown'} |\n| Quantization | {sel.quant_name} ({sel.bits_per_weight:.1f} bpw) |\n| MoE | {moe_str} |\n| Vision | {'Yes' if sel.is_vision else 'No'} |\n| Tool Use | {'Yes' if sel.is_tool_use else 'No'} |")
        with d2:
            rope_str = "None"
            if sel.rope_scaling_type:
                rope_str = f"{sel.rope_scaling_type}"
                if sel.rope_original_context_length:
                    rope_str += f" (native {sel.rope_original_context_length:,}, {sel.rope_scaling_factor:.1f}×)"
            st.markdown(f"| Property | Value |\n|---|---|\n| Parameters | {sel.params_string or 'Unknown'} |\n| Hidden Dim | {est.hidden_dim or 'Unknown'} |\n| FFN Dim | {sel.feed_forward_length or 'Unknown'} |\n| Layers | {est.num_layers or 'Unknown'} |\n| Attention Heads | {est.num_heads or 'Unknown'} |\n| KV Heads | {est.num_kv_heads or 'Unknown'} |\n| Max Context | {sel.max_context_length:,} |\n| RoPE Scaling | {rope_str} |\n| Data Source | {sel.source} |")

    # Quant Variant Comparison
    if sel.variants and len(sel.variants) > 1:
        st.divider()
        st.subheader("Quantization Variants")
        st.caption(f"Compare VRAM across available quantizations for {sel.display_name}")
        models = st.session_state.models
        variant_models = []
        base_key = sel.model_key.rsplit("/", 1)[0] if "/" in sel.model_key else sel.model_key
        for m in models:
            if m.model_key.startswith(base_key) or m.display_name == sel.display_name:
                variant_models.append(m)
        if len(variant_models) <= 1:
            variant_set = set(sel.variants)
            for m in models:
                if m.model_key != sel.model_key and any(v in m.model_key for v in variant_set):
                    variant_models.append(m)
        if len(variant_models) > 1:
            variant_data = []
            for vm in variant_models:
                ve = estimate_vram(
                    parameter_count=vm.parameter_count, bits_per_weight=vm.bits_per_weight,
                    context_length=ctx_length, hidden_dim=vm.hidden_dim, num_layers=vm.num_layers,
                    num_heads=vm.num_heads, num_kv_heads=vm.num_kv_heads,
                    kv_cache_bits=kv_bits, batch_size=batch_size,
                    file_size_bytes=vm.size_bytes, attention_layer_count=vm.attention_layer_count,
                    is_vision=vm.is_vision, expert_count=vm.expert_count,
                    expert_used_count=vm.expert_used_count,
                )
                fits = "✅" if ve.total_gb <= available_vram_gb * 0.85 else "⚠️" if ve.total_gb <= available_vram_gb else "❌"
                current = " ← selected" if vm.model_key == sel.model_key else ""
                variant_data.append({"Variant": f"{vm.quant_name}{current}", "BPW": vm.bits_per_weight,
                    "Disk (GB)": round(vm.size_bytes / (1024**3), 1), "VRAM (GB)": ve.total_gb,
                    "KV Cache (GB)": ve.kv_cache_gb, "Fits?": fits})
            variant_data.sort(key=lambda d: d["BPW"])
            st.dataframe(variant_data, use_container_width=True, hide_index=True)
        else:
            variant_data = []
            for v_name in sorted(sel.variants):
                v_bpw = quant_to_bpw(v_name, 0)
                if v_bpw == 4.5 and v_name.upper() not in ("Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "IQ4_NL", "IQ4_XS"):
                    continue
                ve = estimate_vram(
                    parameter_count=sel.parameter_count, bits_per_weight=v_bpw,
                    context_length=ctx_length, hidden_dim=sel.hidden_dim, num_layers=sel.num_layers,
                    num_heads=sel.num_heads, num_kv_heads=sel.num_kv_heads,
                    kv_cache_bits=kv_bits, batch_size=batch_size,
                    attention_layer_count=sel.attention_layer_count,
                    is_vision=sel.is_vision, expert_count=sel.expert_count,
                    expert_used_count=sel.expert_used_count,
                )
                fits = "✅" if ve.total_gb <= available_vram_gb * 0.85 else "⚠️" if ve.total_gb <= available_vram_gb else "❌"
                current = " ← selected" if v_name == sel.quant_name else ""
                variant_data.append({"Variant": f"{v_name}{current}", "BPW": v_bpw,
                    "Est. VRAM (GB)": ve.total_gb, "Fits?": fits})
            if variant_data:
                variant_data.sort(key=lambda d: d["BPW"])
                st.dataframe(variant_data, use_container_width=True, hide_index=True)

    # Navigation
    st.divider()
    col_back, col_dash = st.columns([1, 1])
    with col_back:
        st.button("← Back to Model Selection", use_container_width=True, on_click=go_to, args=("model_select",))
    with col_dash:
        st.button("📊 View All Models Dashboard", use_container_width=True, on_click=go_to, args=("dashboard",))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard (All Models Comparison)
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.title("📊 All Models Dashboard")
    st.caption("Compare VRAM estimates across all your local models")

    ensure_models_loaded()
    models: list[UnifiedModel] = st.session_state.models
    loaded_keys: set[str] = st.session_state.get("loaded_keys", set())
    available_vram_gb = st.session_state.system_ram_gb * st.session_state.gpu_available_pct / 100

    if not models:
        st.info("No models found. Run Setup to configure your model directory.")
        st.button("← Go to Setup", on_click=go_to, args=("setup",))
        st.stop()

    # Sidebar-style controls inline
    with st.expander("⚙️ Settings", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            system_ram_gb = st.number_input("Unified Memory (GB)", min_value=4, max_value=512,
                                            value=st.session_state.system_ram_gb, step=8, key="dash_ram")
            st.session_state.system_ram_gb = system_ram_gb
        with sc2:
            gpu_pct = st.slider("% Available for LLM", 50, 95,
                                value=st.session_state.gpu_available_pct, key="dash_pct")
            st.session_state.gpu_available_pct = gpu_pct
        with sc3:
            def _refresh_dash():
                st.session_state.pop("models", None)
            st.button("🔄 Refresh Models", on_click=_refresh_dash, key="dash_refresh")
        available_vram_gb = system_ram_gb * gpu_pct / 100
        st.markdown(f"**Available for LLM:** {available_vram_gb:.1f} GB")

    # Summary metrics
    total_size_gb = sum(m.size_bytes for m in models) / (1024**3)
    gguf_count = sum(1 for m in models if m.format == "GGUF")
    mlx_count = sum(1 for m in models if m.format == "MLX")
    moe_count = sum(1 for m in models if m.is_moe)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(metric_card("Models", str(len(models))), unsafe_allow_html=True)
    mc2.markdown(metric_card("Disk Usage", f"{total_size_gb:.1f} GB"), unsafe_allow_html=True)
    mc3.markdown(metric_card("Formats", f"{gguf_count} GGUF · {mlx_count} MLX"), unsafe_allow_html=True)
    mc4.markdown(metric_card("MoE Models", str(moe_count)), unsafe_allow_html=True)

    st.divider()

    # Format filter
    available_formats = sorted(set(m.format for m in models))
    if len(available_formats) > 1:
        format_filter = st.radio("Format", ["All"] + available_formats, horizontal=True, key="dash_fmt")
    else:
        format_filter = "All"

    filtered = [m for m in models if format_filter == "All" or m.format == format_filter]

    # Context length for comparison
    ctx_length = st.select_slider("Context Length for comparison",
        options=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        value=4096, key="dash_ctx")

    # Comparison table
    comparison_data = []
    for m in filtered:
        e = estimate_vram(
            parameter_count=m.parameter_count, bits_per_weight=m.bits_per_weight,
            context_length=ctx_length, hidden_dim=m.hidden_dim, num_layers=m.num_layers,
            num_heads=m.num_heads, num_kv_heads=m.num_kv_heads,
            file_size_bytes=m.size_bytes, attention_layer_count=m.attention_layer_count,
            is_vision=m.is_vision, expert_count=m.expert_count,
            expert_used_count=m.expert_used_count,
        )
        fits = "✅" if e.total_gb <= available_vram_gb * 0.85 else "⚠️" if e.total_gb <= available_vram_gb else "❌"
        tags = ""
        if m.is_moe:
            tags += "🧩 "
        if m.is_vision:
            tags += "👁️ "
        if m.is_tool_use:
            tags += "🔧 "
        if m.model_key in loaded_keys:
            tags += "🟢 "
        comparison_data.append({
            "Model": m.display_name, "Tags": tags.strip(), "Format": m.format,
            "Quant": m.quant_name, "Params": m.params_string,
            "Disk (GB)": round(m.size_bytes / (1024**3), 1),
            "VRAM (GB)": e.total_gb, "Fits?": fits,
        })

    if comparison_data:
        st.dataframe(
            comparison_data, use_container_width=True, hide_index=True,
            column_config={
                "VRAM (GB)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=max(available_vram_gb, max((d["VRAM (GB)"] for d in comparison_data), default=1)),
                    format="%.1f GB",
                ),
            },
        )

    # Bar chart
    if len(comparison_data) > 1:
        names = [d["Model"][:25] for d in comparison_data]
        vrams = [d["VRAM (GB)"] for d in comparison_data]
        colors = ["#4ade80" if v <= available_vram_gb * 0.85 else "#facc15" if v <= available_vram_gb else "#f87171" for v in vrams]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=names, y=vrams, marker_color=colors,
                                 text=[f"{v:.1f}" for v in vrams], textposition="outside"))
        fig_bar.add_hline(y=available_vram_gb, line_dash="dash", line_color="#f87171",
                          annotation_text=f"Available: {available_vram_gb:.0f} GB")
        fig_bar.update_layout(title=f"VRAM @ {ctx_length:,} context", yaxis_title="VRAM (GB)",
                              template="plotly_dark", height=400,
                              margin=dict(t=40, b=80, l=40, r=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("← Back to Home", use_container_width=True, on_click=go_to, args=("home",))
    with col2:
        st.button("🔧 Run Setup Wizard", use_container_width=True, on_click=go_to, args=("setup",))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Trace & Simulate (Benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def _render_session_results(session: SessionStats):
    """Render session statistics as metrics, tables, and charts."""
    st.subheader("Session Summary")

    # Check if VRAM data is available
    has_vram = session.peak_total_vram_gb > 0

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.markdown(metric_card("Rounds", str(len(session.rounds))), unsafe_allow_html=True)
    mc2.markdown(metric_card("Total Tokens", f"{session.total_tokens:,}"), unsafe_allow_html=True)
    mc3.markdown(metric_card("Avg tok/s", f"{session.avg_tokens_per_second:.1f}"), unsafe_allow_html=True)
    mc4.markdown(metric_card("Avg TTFT", f"{session.avg_time_to_first_token:.2f}s"), unsafe_allow_html=True)
    mc5.markdown(metric_card("Duration", f"{session.session_duration:.1f}s"), unsafe_allow_html=True)

    # VRAM summary row
    if has_vram:
        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.markdown(metric_card("Model Weights", f"{session.model_weights_gb:.1f} GB"), unsafe_allow_html=True)
        vm2.markdown(metric_card("Peak KV Cache", f"{session.peak_kv_cache_gb:.2f} GB"), unsafe_allow_html=True)
        vm3.markdown(metric_card("Peak Total VRAM", f"{session.peak_total_vram_gb:.1f} GB"), unsafe_allow_html=True)
        last_round = session.rounds[-1] if session.rounds else None
        fill_pct = last_round.context_fill_pct if last_round else 0
        vm4.markdown(metric_card("Context Fill", f"{fill_pct:.0f}%"), unsafe_allow_html=True)

    # Per-round table
    st.subheader("Per-Round Statistics")
    round_data = []
    for r in session.rounds:
        row = {
            "Round": r.round_number,
            "Prompt Tokens": r.prompt_tokens,
            "Completion Tokens": r.completion_tokens,
            "tok/s": round(r.tokens_per_second, 1),
            "TTFT (s)": round(r.time_to_first_token, 3),
            "Gen Time (s)": round(r.generation_time, 2),
            "Stop Reason": r.stop_reason,
        }
        if has_vram:
            row["Context Tokens"] = r.cumulative_context_tokens
            row["KV Cache (GB)"] = round(r.estimated_kv_cache_gb, 3)
            row["VRAM (GB)"] = r.estimated_total_vram_gb
            row["Ctx Fill %"] = r.context_fill_pct
        round_data.append(row)
    if round_data:
        st.dataframe(round_data, use_container_width=True, hide_index=True)

    # Charts
    if len(session.rounds) > 1:
        chart_col1, chart_col2 = st.columns(2)
        rounds_x = [r.round_number for r in session.rounds]

        with chart_col1:
            tps_y = [r.tokens_per_second for r in session.rounds]
            fig_tps = go.Figure()
            fig_tps.add_trace(go.Scatter(
                x=rounds_x, y=tps_y, mode="lines+markers",
                line=dict(color="#7c8aff", width=2), marker=dict(size=8),
                name="tok/s",
            ))
            fig_tps.update_layout(title="Tokens/Second per Round", xaxis_title="Round",
                                  yaxis_title="tok/s", template="plotly_dark", height=300,
                                  margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(fig_tps, use_container_width=True)

        with chart_col2:
            ttft_y = [r.time_to_first_token for r in session.rounds]
            fig_ttft = go.Figure()
            fig_ttft.add_trace(go.Scatter(
                x=rounds_x, y=ttft_y, mode="lines+markers",
                line=dict(color="#facc15", width=2), marker=dict(size=8),
                name="TTFT",
            ))
            fig_ttft.update_layout(title="Time to First Token per Round", xaxis_title="Round",
                                   yaxis_title="Seconds", template="plotly_dark", height=300,
                                   margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(fig_ttft, use_container_width=True)

        # VRAM growth chart
        if has_vram:
            vram_col1, vram_col2 = st.columns(2)

            with vram_col1:
                fig_vram = go.Figure()
                fig_vram.add_trace(go.Scatter(
                    x=rounds_x,
                    y=[r.estimated_total_vram_gb for r in session.rounds],
                    mode="lines+markers", name="Total VRAM",
                    line=dict(color="#f87171", width=2), marker=dict(size=8),
                    fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
                ))
                fig_vram.add_trace(go.Scatter(
                    x=rounds_x,
                    y=[r.estimated_kv_cache_gb for r in session.rounds],
                    mode="lines+markers", name="KV Cache",
                    line=dict(color="#4ade80", width=2), marker=dict(size=6),
                ))
                available_gb = st.session_state.get("system_ram_gb", 32) * st.session_state.get("gpu_available_pct", 75) / 100
                fig_vram.add_hline(y=available_gb, line_dash="dash", line_color="#facc15",
                                   annotation_text=f"Available: {available_gb:.0f} GB")
                fig_vram.update_layout(title="VRAM Usage Growth", xaxis_title="Round",
                                       yaxis_title="GB", template="plotly_dark", height=300,
                                       margin=dict(t=40, b=40, l=40, r=20))
                st.plotly_chart(fig_vram, use_container_width=True)

            with vram_col2:
                fig_fill = go.Figure()
                fig_fill.add_trace(go.Bar(
                    x=rounds_x,
                    y=[r.cumulative_context_tokens for r in session.rounds],
                    marker_color=["#4ade80" if r.context_fill_pct < 75 else "#facc15" if r.context_fill_pct < 95 else "#f87171"
                                  for r in session.rounds],
                    text=[f"{r.context_fill_pct:.0f}%" for r in session.rounds],
                    textposition="outside",
                ))
                if session.max_context_length:
                    fig_fill.add_hline(y=session.max_context_length, line_dash="dash",
                                       line_color="#f87171",
                                       annotation_text=f"Max: {session.max_context_length:,}")
                fig_fill.update_layout(title="Context Fill per Round", xaxis_title="Round",
                                       yaxis_title="Tokens", template="plotly_dark", height=300,
                                       margin=dict(t=40, b=40, l=40, r=20))
                st.plotly_chart(fig_fill, use_container_width=True)
        else:
            # Token usage chart (fallback when no VRAM data)
            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Bar(
                x=rounds_x, y=[r.prompt_tokens for r in session.rounds],
                name="Prompt", marker_color="#7c8aff",
            ))
            fig_ctx.add_trace(go.Bar(
                x=rounds_x, y=[r.completion_tokens for r in session.rounds],
                name="Completion", marker_color="#4ade80",
            ))
            fig_ctx.update_layout(title="Token Usage per Round", xaxis_title="Round",
                                  yaxis_title="Tokens", template="plotly_dark", height=300,
                                  barmode="stack", margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(fig_ctx, use_container_width=True)

    # Conversation log
    with st.expander("💬 Conversation Log", expanded=False):
        for r in session.rounds:
            st.markdown(f"**Round {r.round_number} — User:**")
            st.text(r.user_message[:500] if r.user_message else "(no message)")
            if r.reasoning_content:
                st.markdown("**Thinking:**")
                st.text(r.reasoning_content[:500])
            st.markdown("**Assistant:**")
            st.text(r.assistant_message[:500] if r.assistant_message else "(no response)")
            st.divider()


def _render_comparison_results(comparison: ComparisonResult):
    """Render multi-model comparison results with side-by-side metrics and charts."""
    st.subheader("Model Comparison Summary")
    st.caption(f"Total benchmark duration: {comparison.total_duration:.1f}s")

    # Summary comparison table
    summary_data = []
    for model_id in comparison.model_order:
        s = comparison.sessions.get(model_id)
        if not s:
            continue
        summary_data.append({
            "Model": model_id,
            "Rounds": len(s.rounds),
            "Total Tokens": s.total_tokens,
            "Avg tok/s": round(s.avg_tokens_per_second, 1),
            "Min tok/s": round(s.min_tokens_per_second, 1),
            "Max tok/s": round(s.max_tokens_per_second, 1),
            "Avg TTFT (s)": round(s.avg_time_to_first_token, 3),
            "Total Gen Time (s)": round(s.total_generation_time, 1),
            "Duration (s)": round(s.session_duration, 1),
        })

    if summary_data:
        st.dataframe(summary_data, use_container_width=True, hide_index=True)

    # Comparison bar charts
    if len(comparison.model_order) > 1:
        valid_sessions = [(mid, comparison.sessions[mid]) for mid in comparison.model_order
                          if mid in comparison.sessions and comparison.sessions[mid].avg_tokens_per_second > 0]

        if valid_sessions:
            chart_col1, chart_col2 = st.columns(2)

            model_names = [mid[:30] for mid, _ in valid_sessions]
            colors = ["#7c8aff", "#4ade80", "#facc15", "#f87171", "#a78bfa",
                      "#7dd3fc", "#fca5a5", "#6ee7b7", "#fde68a", "#c4b5fd"]

            with chart_col1:
                avg_tps = [s.avg_tokens_per_second for _, s in valid_sessions]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=model_names, y=avg_tps,
                    marker_color=colors[:len(model_names)],
                    text=[f"{v:.1f}" for v in avg_tps], textposition="outside",
                ))
                fig.update_layout(title="Avg Tokens/Second", yaxis_title="tok/s",
                                  template="plotly_dark", height=350,
                                  margin=dict(t=40, b=80, l=40, r=20), xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                avg_ttft = [s.avg_time_to_first_token for _, s in valid_sessions]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=model_names, y=avg_ttft,
                    marker_color=colors[:len(model_names)],
                    text=[f"{v:.3f}" for v in avg_ttft], textposition="outside",
                ))
                fig.update_layout(title="Avg Time to First Token", yaxis_title="Seconds",
                                  template="plotly_dark", height=350,
                                  margin=dict(t=40, b=80, l=40, r=20), xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            # Overlay: tok/s per round across models
            fig_overlay = go.Figure()
            for i, (mid, s) in enumerate(valid_sessions):
                rounds_x = [r.round_number for r in s.rounds]
                tps_y = [r.tokens_per_second for r in s.rounds]
                fig_overlay.add_trace(go.Scatter(
                    x=rounds_x, y=tps_y, mode="lines+markers",
                    name=mid[:25], line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                ))
            fig_overlay.update_layout(title="Tokens/Second per Round (all models)",
                                      xaxis_title="Round", yaxis_title="tok/s",
                                      template="plotly_dark", height=350,
                                      margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(fig_overlay, use_container_width=True)

    # Per-model detail expanders
    for model_id in comparison.model_order:
        s = comparison.sessions.get(model_id)
        if not s:
            continue
        with st.expander(f"📋 {model_id} — detailed results", expanded=False):
            _render_session_results(s)


def _annotate_session_from_model(session: SessionStats, model_id: str):
    """Look up model architecture from session state and annotate VRAM estimates."""
    # Ensure models are loaded in session state
    if not st.session_state.get("models"):
        ensure_models_loaded()

    models = st.session_state.get("models", [])
    um = _find_model_by_id(models, model_id)

    if um and (um.hidden_dim or um.num_layers):
        annotate_session_vram(
            session,
            model_weights_gb=um.size_bytes / (1024**3) if um.size_bytes else 0,
            hidden_dim=um.hidden_dim,
            num_layers=um.num_layers,
            num_heads=um.num_heads,
            num_kv_heads=um.num_kv_heads,
            attention_layer_count=um.attention_layer_count,
            max_context_length=um.max_context_length,
        )
    elif session.rounds:
        # Fallback: track cumulative context even without architecture info
        cumulative = 0
        for r in session.rounds:
            cumulative += r.prompt_tokens + r.completion_tokens
            r.cumulative_context_tokens = cumulative


def _find_model_by_id(models: list, model_id: str):
    """
    Find a UnifiedModel matching a server model ID with flexible matching.
    Server IDs can be 'gemma-3n-e4b' while model_key is 'google/gemma-3n-e4b'.
    """
    if not model_id:
        return None
    mid_lower = model_id.lower().strip()

    # Pass 1: exact match on key, name, or display_name
    for m in models:
        if mid_lower in (m.model_key.lower(), m.name.lower(), m.display_name.lower()):
            return m

    # Pass 2: server ID matches the part after '/' in model_key (publisher/model -> model)
    for m in models:
        key_parts = m.model_key.lower().split("/")
        if len(key_parts) >= 2 and mid_lower == key_parts[-1]:
            return m
        name_parts = m.name.lower().split("/")
        if len(name_parts) >= 2 and mid_lower == name_parts[-1]:
            return m

    # Pass 3: substring containment (model_id appears in key or vice versa)
    for m in models:
        if mid_lower in m.model_key.lower() or m.model_key.lower() in mid_lower:
            return m
        if mid_lower in m.name.lower() or m.name.lower() in mid_lower:
            return m

    return None


def page_benchmark():
    st.title("🏋️ Trace & Simulate")
    st.caption("Benchmark inference performance with live statistics")

    # Server status check
    server = check_server()
    if not server["running"]:
        st.error("❌ LM Studio server is not running. Start it in LM Studio → Developer → Start Server.")
        st.button("← Back to Home", on_click=go_to, args=("home",))
        st.stop()

    if server["models_loaded"] == 0:
        st.warning("⚠️ No models loaded. Load a model in LM Studio before benchmarking.")
    else:
        st.success(f"✅ Server running — {server['models_loaded']} model(s) loaded: {', '.join(server['model_ids'])}")

    # Tabs
    tab_simulate, tab_trace = st.tabs(["🚀 Simulate", "📡 Trace"])

    # ── Simulate Tab ─────────────────────────────────────────────────────
    with tab_simulate:
        sim_mode = st.radio("Mode", ["Single Model", "Multi-Model Comparison"], horizontal=True, key="sim_mode")

        # ── Shared config (used by both modes) ───────────────────────────
        st.subheader("Conversation Config")
        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            sim_topic = st.selectbox("Topic", list(TOPIC_PRESETS.keys()) + ["Custom"], key="sim_topic")
            sim_rounds = st.slider("Conversation rounds", 1, 20, 3, key="sim_rounds")

        with cfg_col2:
            sim_temp = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1, key="sim_temp")
            sim_max_tokens = st.number_input("Max tokens per response", 64, 8192, 1024, step=128, key="sim_max_tok")

        if sim_topic == "Custom":
            sim_system = st.text_area("System prompt", value="You are a helpful assistant.", key="sim_sys")
            sim_custom_msgs_raw = st.text_area(
                "User messages (one per line)", key="sim_custom_msgs",
                placeholder="Enter one message per line, one for each round",
            )
            sim_custom_msgs = [m.strip() for m in sim_custom_msgs_raw.split("\n") if m.strip()] if sim_custom_msgs_raw else None
        else:
            sim_system = ""
            sim_custom_msgs = None
            with st.expander("Preview conversation starters"):
                for i, msg in enumerate(TOPIC_PRESETS[sim_topic]["starters"][:sim_rounds]):
                    st.caption(f"Round {i+1}: {msg}")

        st.divider()

        # ── Single Model Mode ────────────────────────────────────────────
        if sim_mode == "Single Model":
            st.subheader("Model")
            catalog = get_model_catalog()
            loaded_entries = [e for e in catalog if e.is_loaded]
            if loaded_entries:
                label_map = {e.selection_label: e.model_id for e in loaded_entries}
                selected_label = st.selectbox("Select loaded model", list(label_map.keys()), key="sim_model_single")
                sim_model = label_map[selected_label]
            elif catalog:
                label_map = {e.selection_label: e.model_id for e in catalog}
                selected_label = st.selectbox("Select model (will need loading)", list(label_map.keys()), key="sim_model_single_all")
                sim_model = label_map[selected_label]
                st.caption("⚠️ This model is not loaded. It must be loaded in LM Studio before running.")
            else:
                sim_model = st.text_input("Model identifier", key="sim_model_input",
                                          placeholder="e.g. qwen2.5-7b-instruct")

            if st.button("▶️ Run Simulation", type="primary", use_container_width=True, key="run_sim_single"):
                if not sim_model:
                    st.error("Please select or enter a model identifier.")
                else:
                    progress_bar = st.progress(0, text="Starting simulation...")

                    def _progress(current, total):
                        progress_bar.progress(current / total, text=f"Round {current}/{total}...")

                    try:
                        session = run_simulation(
                            model=sim_model, num_rounds=sim_rounds, topic=sim_topic,
                            system_prompt=sim_system, custom_messages=sim_custom_msgs,
                            temperature=sim_temp, max_tokens=sim_max_tokens,
                            progress_callback=_progress,
                        )
                        progress_bar.progress(0.95, text="Unloading model...")
                        unload_all_models()
                        progress_bar.progress(1.0, text="Complete!")
                        _annotate_session_from_model(session, sim_model)
                        st.session_state.last_sim_session = session
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        unload_all_models()

            if "last_sim_session" in st.session_state:
                st.divider()
                _render_session_results(st.session_state.last_sim_session)

        # ── Multi-Model Comparison Mode ──────────────────────────────────
        else:
            st.subheader("Select Models to Compare")

            # Fetch rich model catalog
            catalog = get_model_catalog()
            if not catalog:
                st.warning("No models found. Make sure LM Studio has models downloaded.")
                st.stop()

            # Build selection UI with rich labels
            label_to_entry: dict[str, ModelCatalogEntry] = {}
            for entry in catalog:
                label_to_entry[entry.selection_label] = entry

            all_labels = list(label_to_entry.keys())
            # Default to first loaded model if any
            default_labels = [lbl for lbl, e in label_to_entry.items() if e.is_loaded][:1]

            selected_labels = st.multiselect(
                "Models (select 2 or more)",
                options=all_labels,
                default=default_labels,
                key="multi_model_select",
                help="🟢 = currently loaded in memory. Unloaded models will be loaded automatically if auto-load is enabled.",
            )
            selected_entries = [label_to_entry[lbl] for lbl in selected_labels]
            selected_model_ids = [e.model_id for e in selected_entries]

            # Show selection summary
            if selected_entries:
                loaded_count = sum(1 for e in selected_entries if e.is_loaded)
                unloaded_count = len(selected_entries) - loaded_count
                summary_parts = [f"{len(selected_entries)} models selected"]
                if loaded_count:
                    summary_parts.append(f"{loaded_count} loaded")
                if unloaded_count:
                    summary_parts.append(f"{unloaded_count} will need loading")
                st.caption(" · ".join(summary_parts))

                # Detail table of selected models
                with st.expander("Selected models detail", expanded=False):
                    sel_data = []
                    for e in selected_entries:
                        sel_data.append({
                            "Status": "🟢 Loaded" if e.is_loaded else "⬚ Not loaded",
                            "Model": e.model_id,
                            "Arch": e.architecture or "—",
                            "Quant": e.quantization or "—",
                            "Params": e.params_string or "—",
                            "Size": e.size_label or "—",
                            "Max Context": f"{e.max_context_length:,}" if e.max_context_length else "—",
                        })
                    st.dataframe(sel_data, use_container_width=True, hide_index=True)

            auto_load = st.checkbox(
                "Auto load/unload models",
                value=True, key="multi_auto_load",
                help="Automatically load each model before its benchmark run and unload it after. "
                     "Models that were already loaded will not be unloaded.",
            )

            if len(selected_model_ids) < 2:
                st.info("Select at least 2 models to compare.")
            else:
                st.caption(f"Will benchmark {len(selected_model_ids)} models × {sim_rounds} rounds each")

                if st.button("▶️ Run Comparison", type="primary", use_container_width=True, key="run_multi"):
                    progress_bar = st.progress(0, text="Starting comparison...")
                    status_text = st.empty()

                    def _multi_progress(model_id, model_idx, total_models, status):
                        overall = (model_idx / total_models)
                        if "round" in status:
                            # Parse "round X/Y" for finer progress
                            try:
                                parts = status.replace("round ", "").split("/")
                                r_cur, r_tot = int(parts[0]), int(parts[1])
                                overall = (model_idx + r_cur / r_tot) / total_models
                            except (ValueError, IndexError):
                                pass
                        progress_bar.progress(
                            min(overall, 0.99),
                            text=f"Model {model_idx+1}/{total_models}: {model_id[:30]} — {status}",
                        )
                        status_text.caption(f"Current: {model_id} — {status}")

                    try:
                        comparison = run_multi_model_comparison(
                            model_ids=selected_model_ids,
                            num_rounds=sim_rounds, topic=sim_topic,
                            system_prompt=sim_system, custom_messages=sim_custom_msgs,
                            temperature=sim_temp, max_tokens=sim_max_tokens,
                            auto_load_unload=auto_load,
                            progress_callback=_multi_progress,
                        )
                        progress_bar.progress(1.0, text="Comparison complete!")
                        status_text.empty()
                        # Annotate each model's session with VRAM data
                        for mid, sess in comparison.sessions.items():
                            _annotate_session_from_model(sess, mid)
                        st.session_state.last_comparison = comparison

                        # Also save individual sessions for history
                        if "comparison_history" not in st.session_state:
                            st.session_state.comparison_history = []
                        st.session_state.comparison_history.append(comparison)

                    except Exception as e:
                        st.error(f"Comparison failed: {e}")

            # Show comparison results
            if "last_comparison" in st.session_state:
                st.divider()
                _render_comparison_results(st.session_state.last_comparison)

            # Show history of past comparisons
            if st.session_state.get("comparison_history") and len(st.session_state.comparison_history) > 1:
                with st.expander(f"📜 Comparison History ({len(st.session_state.comparison_history)} runs)", expanded=False):
                    for i, comp in enumerate(reversed(st.session_state.comparison_history)):
                        models_str = ", ".join(m[:20] for m in comp.model_order)
                        st.caption(f"Run {len(st.session_state.comparison_history) - i}: "
                                   f"{models_str} — {comp.total_duration:.1f}s")
                        # Quick summary row
                        row = {}
                        for mid in comp.model_order:
                            s = comp.sessions.get(mid)
                            if s:
                                row[mid[:20]] = f"{s.avg_tokens_per_second:.1f} tok/s"
                        st.json(row)

    # ── Trace Tab ────────────────────────────────────────────────────────
    with tab_trace:
        st.markdown(
            "Passively monitor your conversations in LM Studio. "
            "Chat with a model in LM Studio's UI while this captures the stats."
        )

        trace_col1, trace_col2 = st.columns(2)
        with trace_col1:
            trace_duration = st.slider("Trace duration (seconds)", 10, 300, 60, step=10, key="trace_dur")
        with trace_col2:
            st.markdown("")
            st.caption("Stats are captured from `lms log stream`. "
                       "Start a conversation in LM Studio, then click Trace.")

        if st.button("📡 Start Trace", type="primary", use_container_width=True, key="run_trace"):
            progress_bar = st.progress(0, text="Listening for model output...")
            status_text = st.empty()
            collected_rounds: list[RoundStats] = []
            round_num = 0

            start = time.time()
            try:
                for entry in trace_via_log_stream(timeout_seconds=trace_duration):
                    elapsed = time.time() - start
                    if elapsed > trace_duration:
                        break

                    progress_bar.progress(
                        min(elapsed / trace_duration, 1.0),
                        text=f"Tracing... {elapsed:.0f}s / {trace_duration}s — {len(collected_rounds)} responses captured",
                    )

                    parsed = parse_trace_entry(entry)
                    if parsed:
                        round_num += 1
                        parsed.round_number = round_num
                        collected_rounds.append(parsed)

            except Exception as e:
                st.warning(f"Trace ended: {e}")

            progress_bar.progress(1.0, text=f"Trace complete — {len(collected_rounds)} responses captured")

            if collected_rounds:
                session = SessionStats(rounds=collected_rounds)
                session.session_duration = time.time() - start
                session.compute()
                st.session_state.last_trace_session = session
            else:
                st.info("No prediction stats captured. Make sure you're chatting with a model in LM Studio during the trace.")

        # Show trace results
        if "last_trace_session" in st.session_state:
            st.divider()
            _render_session_results(st.session_state.last_trace_session)

    # Navigation
    st.divider()
    st.button("← Back to Home", use_container_width=True, on_click=go_to, args=("home",))


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "home": page_home,
    "setup": page_setup,
    "model_select": page_model_select,
    "vram_estimate": page_vram_estimate,
    "dashboard": page_dashboard,
    "benchmark": page_benchmark,
}

current_page = st.session_state.get("page", "home")
PAGES.get(current_page, page_home)()
