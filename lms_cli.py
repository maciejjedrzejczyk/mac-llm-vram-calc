"""
Integration with LM Studio's `lms` CLI tool.
Provides richer model metadata than raw file parsing.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMSModelInfo:
    """Model info from `lms ls --json`."""
    model_key: str = ""
    display_name: str = ""
    publisher: str = ""
    model_type: str = ""  # "llm" or "embedding"
    format: str = ""  # "gguf" or "safetensors"
    architecture: str = ""
    params_string: str = ""  # e.g. "7B", "122B-A10B"
    size_bytes: int = 0
    quant_name: str = ""
    quant_bits: int = 0
    max_context_length: int = 0
    is_vision: bool = False
    trained_for_tool_use: bool = False
    path: str = ""
    indexed_model_id: str = ""
    variants: list[str] = field(default_factory=list)
    selected_variant: str = ""


@dataclass
class LMSLoadedModel:
    """Runtime info from `lms ps --json` for a currently loaded model."""
    model_key: str = ""
    display_name: str = ""
    # Additional fields will depend on what lms ps returns


@dataclass
class LMSRuntimeConfig:
    """Actual runtime config for a loaded model from the REST API."""
    context_length: int = 0
    eval_batch_size: int = 0
    flash_attention: bool = False
    num_experts: int = 0
    offload_kv_cache_to_gpu: bool = True


@dataclass
class LMSGlobalSettings:
    """Global settings from ~/.lmstudio/settings.json."""
    default_context_length: int = 4096
    downloads_folder: str = ""
    guardrail_mode: str = ""
    guardrail_threshold_bytes: int = 0


def find_lms_cli() -> str | None:
    """Find the lms CLI binary. Returns path or None."""
    # Check common locations
    candidates = [
        os.path.expanduser("~/.lmstudio/bin/lms"),
        shutil.which("lms"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _run_lms(args: list[str], timeout: int = 15) -> str | None:
    """Run an lms CLI command and return stdout, or None on failure."""
    lms_path = find_lms_cli()
    if not lms_path:
        return None
    try:
        result = subprocess.run(
            [lms_path] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def list_models() -> list[LMSModelInfo]:
    """
    Get all models from `lms ls --json`.
    Returns parsed model info list, or empty list if CLI unavailable.
    """
    output = _run_lms(["ls", "--json"])
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    models = []
    for item in data:
        quant = item.get("quantization", {})
        info = LMSModelInfo(
            model_key=item.get("modelKey", ""),
            display_name=item.get("displayName", ""),
            publisher=item.get("publisher", ""),
            model_type=item.get("type", ""),
            format=item.get("format", "").upper().replace("SAFETENSORS", "MLX"),
            architecture=item.get("architecture", ""),
            params_string=item.get("paramsString", ""),
            size_bytes=item.get("sizeBytes", 0),
            quant_name=quant.get("name", ""),
            quant_bits=quant.get("bits", 0),
            max_context_length=item.get("maxContextLength", 0),
            is_vision=item.get("vision", False),
            trained_for_tool_use=item.get("trainedForToolUse", False),
            path=item.get("path", ""),
            indexed_model_id=item.get("indexedModelIdentifier", ""),
            variants=item.get("variants", []),
            selected_variant=item.get("selectedVariant", ""),
        )
        models.append(info)

    return models


def list_loaded_models() -> list[LMSLoadedModel]:
    """Get currently loaded models from `lms ps --json`."""
    output = _run_lms(["ps", "--json"])
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    loaded = []
    for item in data:
        info = LMSLoadedModel(
            model_key=item.get("modelKey", item.get("identifier", "")),
            display_name=item.get("displayName", ""),
        )
        loaded.append(info)
    return loaded


def read_global_settings() -> LMSGlobalSettings:
    """Read LM Studio's global settings.json."""
    settings = LMSGlobalSettings()

    candidates = [
        Path.home() / ".lmstudio" / "settings.json",
        Path.home() / ".cache" / "lm-studio" / "settings.json",
    ]

    for path in candidates:
        if path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)

                # Default context length
                ctx_config = data.get("defaultContextLength", {})
                if isinstance(ctx_config, dict):
                    if ctx_config.get("type") == "custom":
                        settings.default_context_length = ctx_config.get("value", 4096)
                elif isinstance(ctx_config, int):
                    settings.default_context_length = ctx_config

                # Downloads folder
                settings.downloads_folder = data.get("downloadsFolder", "")

                # Guardrails
                guardrails = data.get("modelLoadingGuardrails", {})
                settings.guardrail_mode = guardrails.get("mode", "")
                settings.guardrail_threshold_bytes = guardrails.get("customThresholdBytes", 0)

                return settings
            except (json.JSONDecodeError, OSError):
                continue

    return settings


def is_available() -> bool:
    """Check if the lms CLI is available."""
    return find_lms_cli() is not None


def get_loaded_runtime_configs(port: int = 1234) -> dict[str, LMSRuntimeConfig]:
    """
    Query the LM Studio REST API for actual runtime config of loaded models.
    Returns {model_key: LMSRuntimeConfig} for each loaded instance.
    Requires the LM Studio server to be running on localhost.
    """
    import urllib.request
    import urllib.error

    configs: dict[str, LMSRuntimeConfig] = {}
    host = os.environ.get("LMSTUDIO_HOST", "localhost")
    url = f"http://{host}:{port}/api/v1/models"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return configs

    for model in data.get("models", data.get("data", [])):
        key = model.get("key", model.get("id", ""))
        loaded_instances = model.get("loaded_instances", [])
        if not loaded_instances:
            continue
        # Use the first loaded instance's config
        instance = loaded_instances[0]
        cfg = instance.get("config", {})
        configs[key] = LMSRuntimeConfig(
            context_length=cfg.get("context_length", 0),
            eval_batch_size=cfg.get("eval_batch_size", 0),
            flash_attention=cfg.get("flash_attention", False),
            num_experts=cfg.get("num_experts", 0),
            offload_kv_cache_to_gpu=cfg.get("offload_kv_cache_to_gpu", True),
        )

    return configs
