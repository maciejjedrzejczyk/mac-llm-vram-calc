"""
Benchmark engine for LM Studio inference statistics.

Provides two modes:
- Trace: passive monitoring via lms log stream (captures stats from user-driven conversations)
- Simulate: active benchmarking via REST API (drives multi-turn conversations and collects stats)

Requires LM Studio server running on localhost:1234.
"""

import json
import subprocess
import time
import urllib.request
import urllib.error
import os
from dataclasses import dataclass, field
from typing import Generator


def _api_base(port: int = 1234) -> str:
    """Return the LM Studio API base URL, respecting LMSTUDIO_HOST env var for Docker."""
    host = os.environ.get("LMSTUDIO_HOST", "localhost")
    return f"http://{host}:{port}"


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RoundStats:
    """Statistics for a single conversation round."""
    round_number: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    generation_time: float = 0.0
    stop_reason: str = ""
    user_message: str = ""
    assistant_message: str = ""
    reasoning_content: str = ""
    timestamp: float = 0.0
    # VRAM tracking (annotated post-simulation)
    cumulative_context_tokens: int = 0  # total tokens in context at this round
    estimated_kv_cache_gb: float = 0.0  # KV cache size at this context length
    estimated_total_vram_gb: float = 0.0  # total VRAM (weights + KV + overhead)
    context_fill_pct: float = 0.0  # % of max context used


@dataclass
class SessionStats:
    """Aggregated statistics for an entire session."""
    rounds: list[RoundStats] = field(default_factory=list)
    model_id: str = ""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_generation_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    min_tokens_per_second: float = 0.0
    max_tokens_per_second: float = 0.0
    avg_time_to_first_token: float = 0.0
    min_time_to_first_token: float = 0.0
    max_time_to_first_token: float = 0.0
    stop_reason_counts: dict[str, int] = field(default_factory=dict)
    total_reasoning_tokens: int = 0
    session_duration: float = 0.0
    # VRAM tracking (annotated post-simulation)
    model_weights_gb: float = 0.0
    peak_kv_cache_gb: float = 0.0
    peak_total_vram_gb: float = 0.0
    max_context_length: int = 0  # model's max context for fill % calc

    def compute(self):
        """Recompute aggregates from rounds."""
        if not self.rounds:
            return
        self.total_prompt_tokens = sum(r.prompt_tokens for r in self.rounds)
        self.total_completion_tokens = sum(r.completion_tokens for r in self.rounds)
        self.total_tokens = sum(r.total_tokens for r in self.rounds)
        self.total_generation_time = sum(r.generation_time for r in self.rounds)

        tps_values = [r.tokens_per_second for r in self.rounds if r.tokens_per_second > 0]
        if tps_values:
            self.avg_tokens_per_second = sum(tps_values) / len(tps_values)
            self.min_tokens_per_second = min(tps_values)
            self.max_tokens_per_second = max(tps_values)

        ttft_values = [r.time_to_first_token for r in self.rounds if r.time_to_first_token > 0]
        if ttft_values:
            self.avg_time_to_first_token = sum(ttft_values) / len(ttft_values)
            self.min_time_to_first_token = min(ttft_values)
            self.max_time_to_first_token = max(ttft_values)

        self.stop_reason_counts = {}
        for r in self.rounds:
            if r.stop_reason:
                self.stop_reason_counts[r.stop_reason] = self.stop_reason_counts.get(r.stop_reason, 0) + 1


# ── Server Detection ─────────────────────────────────────────────────────────

def check_server(port: int = 1234) -> dict:
    """
    Check if LM Studio server is running and return status info.
    Returns {"running": bool, "models_loaded": int, "model_ids": list[str],
             "loaded_instances": list[dict]}.
    """
    result = {"running": False, "models_loaded": 0, "model_ids": [], "loaded_instances": []}

    # Try v1 first (has instance-level detail), fall back to v0
    for endpoint, parse_fn in [
        ("/api/v1/models", _parse_v1_loaded),
        ("/api/v0/models", _parse_v0_loaded),
    ]:
        try:
            req = urllib.request.Request(
                f"{_api_base(port)}{endpoint}",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            result["running"] = True
            ids, instances = parse_fn(data)
            result["model_ids"] = ids
            result["loaded_instances"] = instances
            result["models_loaded"] = len(instances)
            return result
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            continue

    return result


def _parse_v1_loaded(data: dict) -> tuple[list[str], list[dict]]:
    """Parse v1 /api/v1/models response for loaded model info."""
    ids = []
    instances = []
    for m in data.get("models", []):
        key = m.get("key", "")
        for inst in m.get("loaded_instances", []):
            # The instance "id" is what unload needs as "instance_id"
            inst_id = inst.get("id", key)
            ids.append(key)
            instances.append({"id": inst_id, "key": key, "instance_id": inst_id})
    return list(dict.fromkeys(ids)), instances


def _parse_v0_loaded(data: dict) -> tuple[list[str], list[dict]]:
    """Parse v0 /api/v0/models response for loaded model info."""
    ids = []
    instances = []
    for m in data.get("data", []):
        if m.get("state") == "loaded":
            mid = m.get("id", "")
            ids.append(mid)
            instances.append({"id": mid, "key": mid, "instance_id": mid})
    return ids, instances


def get_available_models(port: int = 1234) -> list[dict]:
    """Get all models (loaded and unloaded) from the REST API with full metadata."""
    # Try v1 API first (richer data), fall back to v0
    for endpoint in ["/api/v1/models", "/api/v0/models"]:
        try:
            req = urllib.request.Request(
                f"{_api_base(port)}{endpoint}",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            # v1 returns {"models": [...]}, v0 returns {"data": [...]}
            models = data.get("models", data.get("data", []))
            if models:
                return models
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            continue
    return []


@dataclass
class ModelCatalogEntry:
    """Rich model info for selection UI."""
    model_id: str = ""
    display_name: str = ""
    model_type: str = ""  # "llm", "vlm", "embedding"
    architecture: str = ""
    quantization: str = ""
    state: str = ""  # "loaded", "not-loaded"
    max_context_length: int = 0
    format: str = ""  # "gguf", "mlx"
    publisher: str = ""
    params_string: str = ""
    size_bytes: int = 0

    @property
    def is_loaded(self) -> bool:
        return self.state == "loaded"

    @property
    def size_label(self) -> str:
        if self.size_bytes > 0:
            gb = self.size_bytes / (1024**3)
            return f"{gb:.1f} GB" if gb >= 1 else f"{self.size_bytes / (1024**2):.0f} MB"
        return ""

    @property
    def selection_label(self) -> str:
        """Human-readable label for multiselect UI."""
        parts = []
        if self.is_loaded:
            parts.append("🟢")
        parts.append(self.model_id)
        details = []
        if self.architecture:
            details.append(self.architecture)
        if self.quantization:
            details.append(self.quantization)
        if self.params_string:
            details.append(self.params_string)
        if self.size_label:
            details.append(self.size_label)
        if details:
            parts.append(f"({', '.join(details)})")
        return " ".join(parts)


def get_model_catalog(port: int = 1234) -> list[ModelCatalogEntry]:
    """
    Get a rich catalog of all available models for selection UI.
    Combines REST API data with lms CLI data for maximum coverage.
    """
    entries: dict[str, ModelCatalogEntry] = {}

    # REST API models
    for m in get_available_models(port):
        mid = m.get("id", m.get("key", ""))
        if not mid:
            continue
        # v0 fields
        quant = m.get("quantization", "")
        if isinstance(quant, dict):
            quant = quant.get("name", "") or f"{quant.get('bits_per_weight', '')}bpw"
        # v1 has params_string, size_bytes directly; v0 may not
        entries[mid] = ModelCatalogEntry(
            model_id=mid,
            display_name=m.get("display_name", mid),
            model_type=m.get("type", ""),
            architecture=m.get("arch", m.get("architecture", "")),
            quantization=str(quant),
            state=m.get("state", "not-loaded"),
            max_context_length=m.get("max_context_length", 0),
            format=m.get("compatibility_type", m.get("format", "")),
            publisher=m.get("publisher", ""),
            params_string=m.get("params_string", ""),
            size_bytes=m.get("size_bytes", 0),
        )

    # Enrich with lms CLI data if available (has size_bytes, params, etc.)
    import shutil, os
    lms_path = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if os.path.isfile(lms_path):
        try:
            result = subprocess.run(
                [lms_path, "ls", "--json", "--llm"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                for item in json.loads(result.stdout):
                    mid = item.get("modelKey", "")
                    if not mid:
                        continue
                    quant_obj = item.get("quantization", {})
                    if mid in entries:
                        # Enrich existing entry
                        e = entries[mid]
                        e.size_bytes = e.size_bytes or item.get("sizeBytes", 0)
                        e.params_string = e.params_string or item.get("paramsString", "")
                        e.architecture = e.architecture or item.get("architecture", "")
                        e.quantization = e.quantization or quant_obj.get("name", "")
                        e.max_context_length = e.max_context_length or item.get("maxContextLength", 0)
                    else:
                        entries[mid] = ModelCatalogEntry(
                            model_id=mid,
                            display_name=item.get("displayName", mid),
                            model_type=item.get("type", "llm"),
                            architecture=item.get("architecture", ""),
                            quantization=quant_obj.get("name", ""),
                            state="not-loaded",
                            max_context_length=item.get("maxContextLength", 0),
                            format=item.get("format", "").upper(),
                            publisher=item.get("publisher", ""),
                            params_string=item.get("paramsString", ""),
                            size_bytes=item.get("sizeBytes", 0),
                        )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
            pass

    # Filter to LLMs only (exclude embeddings)
    result = [e for e in entries.values() if e.model_type in ("llm", "vlm", "")]
    # Sort: loaded first, then by model_id
    result.sort(key=lambda e: (0 if e.is_loaded else 1, e.model_id.lower()))
    return result


def get_all_model_ids(port: int = 1234) -> list[str]:
    """Get all model identifiers (loaded + downloaded) from the REST API."""
    return [e.model_id for e in get_model_catalog(port)]


# ── Model Load / Unload ─────────────────────────────────────────────────────

def load_model(model_id: str, port: int = 1234, context_length: int | None = None) -> dict:
    """
    Load a model via the v1 REST API.
    Returns the response dict with load_time_seconds, status, etc.
    Raises on failure.
    """
    body: dict = {"model": model_id}
    if context_length:
        body["context_length"] = context_length
    body["echo_load_config"] = True

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_api_base(port)}/api/v1/models/load",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def unload_model(instance_id: str, port: int = 1234) -> dict:
    """
    Unload a model instance via the v1 REST API.
    The instance_id comes from loaded_instances[].id in GET /api/v1/models.
    """
    payload = json.dumps({"instance_id": instance_id}).encode()
    req = urllib.request.Request(
        f"{_api_base(port)}/api/v1/models/unload",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def unload_all_models(port: int = 1234) -> int:
    """
    Unload ALL currently loaded model instances.
    Returns the number of instances successfully unloaded.
    """
    server = check_server(port)
    unloaded = 0

    for inst in server.get("loaded_instances", []):
        inst_id = inst.get("instance_id", "")
        if not inst_id:
            continue
        try:
            unload_model(inst_id, port)
            unloaded += 1
        except Exception:
            pass

    return unloaded


def is_model_loaded(model_id: str, port: int = 1234) -> bool:
    """Check if a specific model is currently loaded."""
    server = check_server(port)
    return model_id in server.get("model_ids", [])


# ── Simulate Mode ────────────────────────────────────────────────────────────

# Preset conversation topics with system prompts and starter messages
TOPIC_PRESETS = {
    "General Q&A": {
        "system": "You are a helpful assistant. Answer questions clearly and concisely.",
        "starters": [
            "What are the main differences between compiled and interpreted programming languages?",
            "Explain how a hash table works and when you'd use one.",
            "What is the CAP theorem in distributed systems?",
            "How does garbage collection work in modern programming languages?",
            "What are the tradeoffs between SQL and NoSQL databases?",
        ],
    },
    "Creative Writing": {
        "system": "You are a creative writing assistant. Write vivid, engaging prose.",
        "starters": [
            "Write a short scene about a detective arriving at an abandoned lighthouse.",
            "Continue the story. The detective finds a journal inside.",
            "Now introduce a second character who arrives unexpectedly.",
            "Write a tense dialogue between the two characters.",
            "Write the final scene where the mystery is resolved.",
        ],
    },
    "Coding": {
        "system": "You are an expert programmer. Write clean, well-documented code.",
        "starters": [
            "Write a Python function that implements binary search on a sorted list.",
            "Now add comprehensive error handling and type hints to that function.",
            "Write unit tests for the binary search function using pytest.",
            "Refactor the code to also support a custom comparison function.",
            "Add a performance benchmark that compares it against Python's bisect module.",
        ],
    },
    "Reasoning": {
        "system": "You are a logical reasoning assistant. Think step by step.",
        "starters": [
            "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9, totaling $27, plus the $2 the bellboy kept equals $29. Where did the missing dollar go?",
            "You have 8 balls. One is heavier. You have a balance scale. What's the minimum number of weighings to find the heavy ball?",
            "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        ],
    },
}


def _api_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    port: int = 1234,
) -> dict:
    """Send a chat completion request to LM Studio REST API and return the full response."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{_api_base(port)}/api/v0/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _parse_response(response: dict, round_number: int) -> RoundStats:
    """Parse a chat completion response into RoundStats."""
    stats = response.get("stats", {})
    usage = response.get("usage", {})
    choices = response.get("choices", [{}])
    message = choices[0].get("message", {}) if choices else {}

    return RoundStats(
        round_number=round_number,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        tokens_per_second=stats.get("tokens_per_second", 0.0),
        time_to_first_token=stats.get("time_to_first_token", 0.0),
        generation_time=stats.get("generation_time", 0.0),
        stop_reason=stats.get("stop_reason", ""),
        assistant_message=message.get("content", ""),
        reasoning_content=message.get("reasoning_content", ""),
        timestamp=time.time(),
    )


def run_simulation(
    model: str,
    num_rounds: int = 3,
    topic: str = "General Q&A",
    system_prompt: str = "",
    custom_messages: list[str] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    port: int = 1234,
    progress_callback=None,
) -> SessionStats:
    """
    Run a multi-turn conversation simulation and collect statistics.

    Args:
        model: Model identifier (as shown in LM Studio)
        num_rounds: Number of conversation rounds
        topic: Preset topic name or "Custom"
        system_prompt: Override system prompt (uses preset if empty)
        custom_messages: Custom user messages (one per round)
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        port: LM Studio server port
        progress_callback: Called with (round_number, num_rounds) after each round
    """
    session = SessionStats(model_id=model)
    start_time = time.time()

    # Build system prompt and user messages
    preset = TOPIC_PRESETS.get(topic, TOPIC_PRESETS["General Q&A"])
    sys_prompt = system_prompt or preset["system"]

    if custom_messages:
        user_messages = custom_messages[:num_rounds]
        # Pad with preset starters if not enough custom messages
        while len(user_messages) < num_rounds:
            idx = len(user_messages) % len(preset["starters"])
            user_messages.append(preset["starters"][idx])
    else:
        user_messages = []
        for i in range(num_rounds):
            idx = i % len(preset["starters"])
            user_messages.append(preset["starters"][idx])

    # Build conversation
    messages = [{"role": "system", "content": sys_prompt}]

    for i, user_msg in enumerate(user_messages):
        messages.append({"role": "user", "content": user_msg})

        try:
            response = _api_chat(messages, model, temperature, max_tokens, port)
            round_stats = _parse_response(response, i + 1)
            round_stats.user_message = user_msg
            session.rounds.append(round_stats)

            # Add assistant response to conversation history for multi-turn
            messages.append({"role": "assistant", "content": round_stats.assistant_message})
        except Exception as e:
            # Record failed round
            session.rounds.append(RoundStats(
                round_number=i + 1,
                user_message=user_msg,
                stop_reason=f"error: {str(e)}",
                timestamp=time.time(),
            ))

        if progress_callback:
            progress_callback(i + 1, num_rounds)

    session.session_duration = time.time() - start_time
    session.compute()
    return session


@dataclass
class ComparisonResult:
    """Results from a multi-model comparison benchmark."""
    sessions: dict[str, SessionStats] = field(default_factory=dict)  # model_id -> SessionStats
    model_order: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)  # shared config used for all models
    total_duration: float = 0.0


def run_multi_model_comparison(
    model_ids: list[str],
    num_rounds: int = 3,
    topic: str = "General Q&A",
    system_prompt: str = "",
    custom_messages: list[str] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    auto_load_unload: bool = True,
    port: int = 1234,
    progress_callback=None,
) -> ComparisonResult:
    """
    Run the same conversation against multiple models sequentially.
    Handles loading/unloading models automatically if auto_load_unload is True.
    All tested models are unloaded after their run completes.

    Args:
        model_ids: List of model identifiers to benchmark
        auto_load_unload: If True, load each model before its run and unload after
        progress_callback: Called with (model_id, model_index, total_models, status_message)
    """
    result = ComparisonResult(
        model_order=list(model_ids),
        config={
            "num_rounds": num_rounds, "topic": topic, "temperature": temperature,
            "max_tokens": max_tokens, "system_prompt": system_prompt,
        },
    )
    start_time = time.time()

    for idx, model_id in enumerate(model_ids):
        if progress_callback:
            progress_callback(model_id, idx, len(model_ids), "preparing")

        if auto_load_unload:
            # Unload ALL currently loaded models to free VRAM before loading the next one
            if progress_callback:
                progress_callback(model_id, idx, len(model_ids), "clearing VRAM")
            unload_all_models(port)

            # Small delay to let the runtime release memory
            time.sleep(1)

            # Load the target model into clean VRAM
            if progress_callback:
                progress_callback(model_id, idx, len(model_ids), "loading")
            try:
                load_model(model_id, port)
            except Exception as e:
                failed = SessionStats(model_id=model_id)
                failed.rounds.append(RoundStats(
                    round_number=0, stop_reason=f"load_error: {e}", timestamp=time.time(),
                ))
                result.sessions[model_id] = failed
                continue

        # Run simulation
        if progress_callback:
            progress_callback(model_id, idx, len(model_ids), "running")

        def _inner_progress(current, total):
            if progress_callback:
                progress_callback(model_id, idx, len(model_ids),
                                  f"round {current}/{total}")

        session = run_simulation(
            model=model_id, num_rounds=num_rounds, topic=topic,
            system_prompt=system_prompt, custom_messages=custom_messages,
            temperature=temperature, max_tokens=max_tokens, port=port,
            progress_callback=_inner_progress,
        )
        result.sessions[model_id] = session

        # Unload after run
        if auto_load_unload:
            if progress_callback:
                progress_callback(model_id, idx, len(model_ids), "unloading")
            unload_all_models(port)

    # Final cleanup: unload anything still loaded
    if auto_load_unload:
        unload_all_models(port)

    result.total_duration = time.time() - start_time
    return result


# ── VRAM Annotation ──────────────────────────────────────────────────────────

def annotate_session_vram(
    session: SessionStats,
    model_weights_gb: float = 0.0,
    hidden_dim: int = 0,
    num_layers: int = 0,
    num_heads: int = 0,
    num_kv_heads: int = 0,
    kv_cache_bits: int = 16,
    attention_layer_count: int = 0,
    max_context_length: int = 0,
    activation_overhead_pct: float = 0.15,
    runtime_overhead_gb: float = 0.3,
):
    """
    Annotate each round in a session with estimated VRAM usage based on
    cumulative context growth. Call this after run_simulation() with the
    model's architecture info.
    """
    if not session.rounds or not hidden_dim or not num_layers:
        return

    if num_kv_heads == 0:
        num_kv_heads = num_heads
    kv_layers = attention_layer_count if attention_layer_count > 0 else num_layers
    gqa_ratio = num_kv_heads / num_heads if num_heads > 0 else 1.0
    bytes_per_element = kv_cache_bits / 8
    kv_per_token_bytes = 2 * hidden_dim * kv_layers * bytes_per_element * gqa_ratio

    activation_gb = model_weights_gb * activation_overhead_pct
    runtime_gb = runtime_overhead_gb + model_weights_gb * 0.02
    base_vram = model_weights_gb + activation_gb + runtime_gb

    cumulative = 0
    for r in session.rounds:
        cumulative += r.prompt_tokens + r.completion_tokens
        r.cumulative_context_tokens = cumulative
        r.estimated_kv_cache_gb = round((cumulative * kv_per_token_bytes) / (1024**3), 4)
        r.estimated_total_vram_gb = round(base_vram + r.estimated_kv_cache_gb, 2)
        if max_context_length > 0:
            r.context_fill_pct = round(cumulative / max_context_length * 100, 1)

    session.model_weights_gb = model_weights_gb
    session.max_context_length = max_context_length
    if session.rounds:
        session.peak_kv_cache_gb = session.rounds[-1].estimated_kv_cache_gb
        session.peak_total_vram_gb = session.rounds[-1].estimated_total_vram_gb


# ── Trace Mode ───────────────────────────────────────────────────────────────

def trace_poll_loaded_models(port: int = 1234) -> list[dict]:
    """
    Poll the REST API for currently loaded models and their state.
    Returns list of model dicts with id, state, type, etc.
    """
    try:
        req = urllib.request.Request(
            f"{_api_base(port)}/api/v0/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return [m for m in data.get("data", []) if m.get("state") == "loaded"]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return []


def trace_via_log_stream(timeout_seconds: int = 60) -> Generator[dict, None, None]:
    """
    Start `lms log stream --source model --filter output --stats --json`
    and yield parsed JSON log entries as they arrive.

    Each yielded dict may contain prediction stats when --stats captures them.
    Stops after timeout_seconds of inactivity or if the process ends.
    """
    import shutil
    import os

    lms_path = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if not os.path.isfile(lms_path):
        return

    try:
        proc = subprocess.Popen(
            [lms_path, "log", "stream", "--source", "model", "--filter", "output", "--stats", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except (FileNotFoundError, OSError):
        return

    try:
        import select
        while True:
            # Use select for non-blocking read with timeout
            ready, _, _ = select.select([proc.stdout], [], [], timeout_seconds)
            if not ready:
                break  # timeout
            line = proc.stdout.readline()
            if not line:
                break  # process ended
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                yield entry
            except json.JSONDecodeError:
                continue
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_trace_entry(entry: dict) -> RoundStats | None:
    """
    Parse a trace log entry into RoundStats if it contains prediction stats.
    Returns None if the entry doesn't have stats.
    """
    # lms log stream --stats --json emits entries with prediction stats
    # The exact format depends on LM Studio version, but typically includes:
    # - "stats" or "predictionStats" with tokens_per_second, time_to_first_token, etc.
    stats = entry.get("stats", entry.get("predictionStats", {}))
    if not stats:
        return None

    return RoundStats(
        tokens_per_second=stats.get("tokens_per_second", stats.get("tokensPerSecond", 0.0)),
        time_to_first_token=stats.get("time_to_first_token", stats.get("timeToFirstToken", 0.0)),
        generation_time=stats.get("generation_time", stats.get("generationTime", 0.0)),
        stop_reason=stats.get("stop_reason", stats.get("stopReason", "")),
        prompt_tokens=stats.get("prompt_tokens", stats.get("promptTokens", 0)),
        completion_tokens=stats.get("completion_tokens", stats.get("completionTokens", 0)),
        total_tokens=stats.get("total_tokens", stats.get("totalTokens", 0)),
        assistant_message=entry.get("content", entry.get("text", "")),
        timestamp=time.time(),
    )
