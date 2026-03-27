"""
Reads LM Studio per-model configuration (context length, KV cache settings, etc.)
from ~/.lmstudio/.internal/user-concrete-model-default-config/
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LMStudioModelConfig:
    """Per-model load settings from LM Studio."""
    context_length: int | None = None
    kv_cache_quant_enabled: bool | None = None
    kv_cache_quant_bits: int | None = None
    kv_cache_quant_group_size: int | None = None
    raw_fields: dict | None = None


def get_lmstudio_config_dir() -> Path:
    """Return the path to LM Studio's per-model config directory."""
    # Try newer path first, then legacy
    candidates = [
        Path.home() / ".lmstudio" / ".internal" / "user-concrete-model-default-config",
        Path.home() / ".cache" / "lm-studio" / ".internal" / "user-concrete-model-default-config",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def _extract_fields(load_fields: list[dict]) -> dict:
    """Convert LM Studio's field list [{key, value}, ...] into a flat dict."""
    return {f["key"]: f["value"] for f in load_fields if "key" in f and "value" in f}


def read_model_config(model_path: str) -> LMStudioModelConfig:
    """
    Try to find and read LM Studio's config for a given model.

    LM Studio stores configs at:
      ~/.lmstudio/.internal/user-concrete-model-default-config/{org}/{model}.json

    The org/model name is derived from the model's path in the models directory.
    For example:
      models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit/
      -> looks for configs matching "qwen3-coder-30b" under any org folder

      models/unsloth/Qwen3.5-122B-A10B-GGUF/
      -> looks for configs matching "qwen3.5-122b-a10b" under any org folder
    """
    config = LMStudioModelConfig()
    config_dir = get_lmstudio_config_dir()

    if not config_dir.is_dir():
        return config

    # Extract a base model name from the path for fuzzy matching
    # Strip common suffixes: -GGUF, -MLX-4bit, -Q4_K_M, quantization tags, etc.
    model_name = Path(model_path).name
    base_name = _normalize_model_name(model_name)

    # Search all org folders for a matching config
    # LM Studio config names can be abbreviated (e.g. "devstral-small-2-2512" for
    # "Devstral-Small-2-24B-Instruct-2512-4bit"), so we use flexible matching.
    best_match: Path | None = None
    best_score = 0

    for org_dir in config_dir.iterdir():
        if not org_dir.is_dir():
            continue
        for config_file in org_dir.glob("*.json"):
            config_name = config_file.stem.lower()
            score = _match_score(base_name, config_name)
            if score > best_score:
                best_score = score
                best_match = config_file

    if best_match and best_score >= 0.6:
        return _parse_config_file(best_match)

    return config


def _match_score(model_name: str, config_name: str) -> float:
    """
    Score how well a config name matches a model name.
    Returns 0.0-1.0 where 1.0 is a perfect match.
    """
    if model_name == config_name:
        return 1.0

    scores = []

    # Prefix matching in both directions
    if model_name.startswith(config_name) or config_name.startswith(model_name):
        shorter = min(len(model_name), len(config_name))
        longer = max(len(model_name), len(config_name))
        scores.append(shorter / longer)

    # Check if all segments of the config name appear in the model name
    config_parts = set(config_name.replace("-", " ").replace("_", " ").split())
    model_parts = set(model_name.replace("-", " ").replace("_", " ").split())

    if config_parts:
        overlap = config_parts & model_parts
        scores.append(len(overlap) / len(config_parts))

    return max(scores) if scores else 0.0


def _normalize_model_name(name: str) -> str:
    """
    Normalize a model directory/file name to match LM Studio's config naming.
    Strips GGUF/MLX suffixes, quantization tags, shard numbers, etc.
    """
    import re
    name = name.lower()
    # Remove file extension
    name = re.sub(r"\.gguf$", "", name)
    # Remove shard suffix (-00001-of-00003)
    name = re.sub(r"-\d{5}-of-\d{5}$", "", name)
    # Remove " (N shards)" suffix from our grouped display name
    name = re.sub(r"\s*\(\d+ shards\)$", "", name)
    # Remove common quantization suffixes (can appear anywhere after model name)
    name = re.sub(r"[-_](q\d+_k_[sml]|q\d+_\d+|f16|f32|bf16|fp16|fp32)(?=[-_]|$)", "", name)
    # Remove -GGUF, -MLX-Nbit suffixes
    name = re.sub(r"[-_](gguf|mlx[-_]?\d+bit)(?=[-_]|$)", "", name)
    # Remove trailing -Nbit (e.g. -4bit, -8bit)
    name = re.sub(r"[-_]\d+bit$", "", name)
    # Remove -MXFP4-Q8 style suffixes
    name = re.sub(r"[-_]mxfp\d+[-_]?q\d+$", "", name)
    return name.strip("-_ ")


def _parse_config_file(config_path: Path) -> LMStudioModelConfig:
    """Parse a single LM Studio model config JSON file."""
    config = LMStudioModelConfig()
    try:
        with open(config_path) as f:
            data = json.load(f)

        load_section = data.get("load", {})
        fields = load_section.get("fields", [])
        field_map = _extract_fields(fields)
        config.raw_fields = field_map

        # Context length
        if "llm.load.contextLength" in field_map:
            config.context_length = int(field_map["llm.load.contextLength"])

        # MLX KV cache quantization
        kv_config = field_map.get("llm.load.mlx.kvCacheQuantization")
        if isinstance(kv_config, dict):
            config.kv_cache_quant_enabled = kv_config.get("enabled", False)
            config.kv_cache_quant_bits = kv_config.get("bits", 8)
            config.kv_cache_quant_group_size = kv_config.get("groupSize", 64)

        # llama.cpp flash attention (implies KV cache quantization support)
        if "llm.load.llama.flashAttention" in field_map:
            flash = field_map["llm.load.llama.flashAttention"]
            if isinstance(flash, bool) and flash:
                # Flash attention enabled — KV cache is typically FP16
                # but llama.cpp can also do Q8/Q4 KV cache with flash attention
                pass

    except Exception:
        pass

    return config


def scan_all_configs() -> dict[str, LMStudioModelConfig]:
    """Read all per-model configs from LM Studio. Returns {config_name: config}."""
    config_dir = get_lmstudio_config_dir()
    configs = {}

    if not config_dir.is_dir():
        return configs

    for org_dir in config_dir.iterdir():
        if not org_dir.is_dir():
            continue
        for config_file in org_dir.glob("*.json"):
            config = _parse_config_file(config_file)
            key = f"{org_dir.name}/{config_file.stem}"
            configs[key] = config

    return configs
