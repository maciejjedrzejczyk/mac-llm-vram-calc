"""
Scans LM Studio's model directory for GGUF and MLX model files and extracts metadata.
"""

import json
import os
import struct
from pathlib import Path
from dataclasses import dataclass, field

# GGUF magic number and constants
GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types and their bits-per-weight
GGML_TYPES_BPW = {
    0: 32.0,   # F32
    1: 16.0,   # F16
    2: 4.5,    # Q4_0
    3: 4.5,    # Q4_1
    6: 5.5,    # Q5_0
    7: 5.5,    # Q5_1
    8: 8.5,    # Q8_0
    9: 8.5,    # Q8_1
    10: 2.5625,  # Q2_K
    11: 3.4375,  # Q3_K_S (approx)
    12: 4.5,    # Q4_K_S (approx)
    13: 4.5,    # Q4_K_M (approx, same block structure)
    14: 5.5,    # Q5_K_S (approx)
    15: 5.5,    # Q5_K_M (approx)
    16: 6.5625,  # Q6_K
    17: 8.5,    # Q8_K
    18: 2.0,    # IQ2_XXS
    19: 2.3125,  # IQ2_XS
    20: 3.0,    # IQ3_XXS
    21: 1.625,  # IQ1_S
    22: 4.5,    # IQ4_NL
    23: 3.5,    # IQ3_S
    24: 2.5,    # IQ2_S
    25: 4.5,    # IQ4_XS
    26: 1.75,   # IQ1_M
    28: 16.0,   # BF16
    29: 4.875,  # Q4_0_4_4
    30: 4.875,  # Q4_0_4_8
    31: 4.875,  # Q4_0_8_8
    32: 10.0,   # TQ1_0
    33: 12.0,   # TQ2_0
}


@dataclass
class ModelInfo:
    """Parsed metadata from a GGUF or MLX model."""
    file_path: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    model_name: str = ""
    architecture: str = ""
    model_format: str = ""  # "GGUF" or "MLX"
    parameter_count: int = 0
    quantization_type: str = ""
    bits_per_weight: float = 0.0
    context_length: int = 0
    embedding_length: int = 0  # hidden dimension
    feed_forward_length: int = 0  # FFN intermediate dimension
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    tensor_count: int = 0
    # MoE (Mixture of Experts) fields
    expert_count: int = 0  # total experts in MoE models
    expert_used_count: int = 0  # experts active per token
    # RoPE scaling info
    rope_scaling_type: str = ""  # "none", "linear", "yarn", etc.
    rope_scaling_factor: float = 0.0
    rope_original_context_length: int = 0
    metadata: dict = field(default_factory=dict)
    parse_error: str = ""


# Keep backward-compatible alias
GGUFModelInfo = ModelInfo


def _read_string(f) -> str:
    """Read a GGUF string (uint64 length + bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    """Read a single GGUF metadata value based on its type."""
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_ARRAY:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def parse_gguf_metadata(file_path: str) -> ModelInfo:
    """
    Parse GGUF file header and metadata without loading tensors.
    Only reads the header + KV pairs, so it's fast even for huge files.
    """
    info = ModelInfo(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        file_size_bytes=os.path.getsize(file_path),
        model_format="GGUF",
    )

    try:
        with open(file_path, "rb") as f:
            # Read magic
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                info.parse_error = "Not a valid GGUF file"
                return info

            # Read version
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                info.parse_error = f"Unsupported GGUF version: {version}"
                return info

            # Read tensor count and metadata KV count
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            info.tensor_count = tensor_count

            # Read all metadata KV pairs
            metadata = {}
            for _ in range(kv_count):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = _read_value(f, vtype)
                metadata[key] = value

            info.metadata = metadata

            # Extract well-known fields
            arch = metadata.get("general.architecture", "")
            info.architecture = arch
            info.model_name = metadata.get("general.name", "")
            info.quantization_type = metadata.get("general.file_type", "")

            # Architecture-specific keys use the pattern: {arch}.{key}
            prefix = f"{arch}." if arch else ""
            info.context_length = metadata.get(f"{prefix}context_length", 0)
            info.embedding_length = metadata.get(f"{prefix}embedding_length", 0)
            info.num_layers = metadata.get(f"{prefix}block_count", 0)
            info.num_heads = metadata.get(f"{prefix}attention.head_count", 0)

            # KV heads can be an int (uniform) or a per-layer list (hybrid models)
            raw_kv_heads = metadata.get(
                f"{prefix}attention.head_count_kv",
                info.num_heads,
            )
            if isinstance(raw_kv_heads, list):
                # Hybrid model (e.g. Mamba + Attention): some layers have 0 KV heads.
                # Use the max non-zero value for KV cache estimation.
                non_zero = [h for h in raw_kv_heads if h > 0]
                info.num_kv_heads = max(non_zero) if non_zero else 0
                # Store the count of attention layers for more accurate KV cache calc
                info.metadata["_attention_layer_count"] = len(non_zero)
            else:
                info.num_kv_heads = raw_kv_heads

            info.vocab_size = metadata.get(f"{prefix}vocab_size", 0)
            info.feed_forward_length = metadata.get(f"{prefix}feed_forward_length", 0)

            # MoE (Mixture of Experts) fields
            info.expert_count = metadata.get(f"{prefix}expert_count", 0)
            info.expert_used_count = metadata.get(f"{prefix}expert_used_count", 0)

            # RoPE scaling info
            info.rope_scaling_type = metadata.get(f"{prefix}rope.scaling.type", "")
            info.rope_scaling_factor = metadata.get(f"{prefix}rope.scaling.factor", 0.0)
            info.rope_original_context_length = metadata.get(
                f"{prefix}rope.scaling.original_context_length", 0
            )
            # Fallback: older models use rope.scale_linear instead
            if not info.rope_scaling_type and metadata.get(f"{prefix}rope.scale_linear", 0) > 0:
                info.rope_scaling_type = "linear"
                info.rope_scaling_factor = metadata.get(f"{prefix}rope.scale_linear", 0.0)

            # Estimate parameter count and quantization info
            _estimate_params_and_quant(info)

    except Exception as e:
        info.parse_error = str(e)

    return info


# Map GGUF general.file_type integer codes to human-readable names and avg BPW
FILE_TYPE_MAP = {
    0: ("F32", 32.0),
    1: ("F16", 16.0),
    2: ("Q4_0", 4.5),
    3: ("Q4_1", 5.0),
    7: ("Q8_0", 8.5),
    8: ("Q8_1", 9.0),
    10: ("Q2_K", 2.5625),
    11: ("Q3_K_S", 3.4375),
    12: ("Q3_K_M", 3.9),
    13: ("Q3_K_L", 4.3),
    14: ("Q4_K_S", 4.5),
    15: ("Q4_K_M", 4.8),
    16: ("Q5_K_S", 5.5),
    17: ("Q5_K_M", 5.7),
    18: ("Q6_K", 6.5625),
    19: ("IQ2_XXS", 2.0),
    20: ("IQ2_XS", 2.3),
    21: ("IQ2_S", 2.5),
    22: ("IQ3_XXS", 3.0),
    23: ("IQ1_S", 1.625),
    24: ("IQ4_NL", 4.5),
    25: ("IQ3_S", 3.5),
    26: ("IQ4_XS", 4.25),
    27: ("IQ1_M", 1.75),
    28: ("BF16", 16.0),
    29: ("Q4_0_4_4", 4.5),
    30: ("Q4_0_4_8", 4.5),
    31: ("Q4_0_8_8", 4.5),
}


def _estimate_params_and_quant(info: ModelInfo):
    """Estimate parameter count and bits-per-weight from metadata."""
    file_type = info.metadata.get("general.file_type", None)

    if file_type is not None and file_type in FILE_TYPE_MAP:
        name, bpw = FILE_TYPE_MAP[file_type]
        info.quantization_type = name
        info.bits_per_weight = bpw
    else:
        # Try to infer from filename
        fname = info.file_name.upper()
        for code, (name, bpw) in sorted(FILE_TYPE_MAP.items(), key=lambda x: -len(x[1][0])):
            if name.replace("_", "").upper() in fname.replace("_", "").replace("-", ""):
                info.quantization_type = name
                info.bits_per_weight = bpw
                break
        if not info.bits_per_weight:
            # Default assumption
            info.quantization_type = "Unknown"
            info.bits_per_weight = 4.5  # conservative Q4 estimate

    # Estimate parameter count using the best available source:
    # 1. general.size_label from metadata (e.g. "7B", "1.5B") — most reliable
    # 2. Architecture-based calculation from hidden_dim and layers
    # 3. File size / bytes-per-param — least reliable (inflated by MoE, large vocab, etc.)

    size_label = info.metadata.get("general.size_label", "")
    if size_label and not info.parameter_count:
        info.parameter_count = _parse_size_label(size_label)

    if not info.parameter_count and info.embedding_length and info.num_layers:
        # Use feed_forward_length for a more accurate estimate when available.
        # Standard transformer per-layer params:
        #   Attention: 4 × hidden² (Q, K, V, O projections) adjusted for GQA
        #   FFN: 3 × hidden × ffn_dim (gate, up, down for SwiGLU) or 2 × hidden × ffn_dim
        h = info.embedding_length
        n = info.num_layers
        if info.feed_forward_length:
            ffn = info.feed_forward_length
            # Assume SwiGLU (3 matrices) which is standard for modern LLMs
            ffn_params = 3 * h * ffn
        else:
            # Fallback: assume ffn_dim ≈ 4 × hidden (classic transformer)
            ffn_params = 3 * h * (h * 4)
        # Attention params: Q, K, V, O — adjust K/V for GQA
        kv_h = info.num_kv_heads if info.num_kv_heads else info.num_heads
        n_h = info.num_heads if info.num_heads else 1
        head_dim = h // n_h if n_h else h
        attn_params = h * (n_h * head_dim) + 2 * h * (kv_h * head_dim) + (n_h * head_dim) * h
        per_layer = attn_params + ffn_params
        # MoE: multiply FFN params by expert_count (each expert has its own FFN)
        if info.expert_count > 1:
            per_layer = attn_params + ffn_params * info.expert_count
        # Add embedding + output layers
        vocab_params = info.vocab_size * h * 2 if info.vocab_size else h * 32000 * 2
        info.parameter_count = int(per_layer * n + vocab_params)

    if not info.parameter_count and info.bits_per_weight > 0:
        bytes_per_param = info.bits_per_weight / 8
        effective_size = max(info.file_size_bytes - 1_000_000, info.file_size_bytes * 0.98)
        info.parameter_count = int(effective_size / bytes_per_param)


def _parse_size_label(label: str) -> int:
    """Parse a size label like '7B', '1.5B', '400M', '122B-A10B' into total parameter count."""
    label = label.strip().upper()
    # Handle MoE labels like "122B-A10B" — use the first number (total params)
    if "-" in label:
        label = label.split("-")[0]
    try:
        if label.endswith("B"):
            return int(float(label[:-1]) * 1e9)
        elif label.endswith("M"):
            return int(float(label[:-1]) * 1e6)
    except (ValueError, IndexError):
        pass
    return 0


def get_lm_studio_models_dir() -> str:
    """Return the default LM Studio models directory on macOS."""
    # LM Studio has used different paths across versions
    candidates = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]  # default to newer path


def _dir_total_size(dir_path: Path) -> int:
    """Sum the size of all files in a directory (non-recursive into subdirs of subdirs)."""
    total = 0
    for f in dir_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def parse_mlx_model(model_dir: str) -> ModelInfo:
    """
    Parse an MLX model directory containing config.json and .safetensors files.

    MLX models use the HuggingFace config.json format with fields like:
    - hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads
    - max_position_embeddings, vocab_size, model_type
    - quantization_config.bits (if quantized)
    """
    dir_path = Path(model_dir)
    config_path = dir_path / "config.json"

    # Use the directory name as the display name
    # For LM Studio, paths look like: author/model-name/
    dir_name = dir_path.name

    info = ModelInfo(
        file_path=model_dir,
        file_name=dir_name,
        file_size_bytes=_dir_total_size(dir_path),
        model_format="MLX",
    )

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        info.metadata = config
        info.architecture = config.get("model_type", "")
        info.model_name = config.get("_name_or_path", dir_name)
        info.embedding_length = config.get("hidden_size", 0)
        info.num_layers = config.get("num_hidden_layers", 0)
        info.num_heads = config.get("num_attention_heads", 0)
        info.num_kv_heads = config.get("num_key_value_heads", info.num_heads)
        info.vocab_size = config.get("vocab_size", 0)
        info.context_length = config.get(
            "max_position_embeddings",
            config.get("max_sequence_length", 0),
        )

        # MoE fields (HuggingFace config format)
        info.expert_count = config.get("num_local_experts", config.get("num_experts", 0))
        info.expert_used_count = config.get(
            "num_experts_per_tok",
            config.get("num_selected_experts", 0),
        )
        info.feed_forward_length = config.get("intermediate_size", 0)

        # RoPE scaling (HuggingFace config format)
        rope_scaling = config.get("rope_scaling", {})
        if isinstance(rope_scaling, dict) and rope_scaling:
            info.rope_scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type", ""))
            info.rope_scaling_factor = rope_scaling.get("factor", 0.0)
            info.rope_original_context_length = rope_scaling.get("original_max_position_embeddings", 0)

        # Determine quantization from quantization_config or directory name
        quant_config = config.get("quantization_config", {})
        quant_bits = quant_config.get("bits", None)

        if quant_bits:
            group_size = quant_config.get("group_size", 64)
            info.quantization_type = f"MLX-W{quant_bits}G{group_size}"
            # MLX quantization: bits per weight + small overhead for scales/zeros
            # Each group of `group_size` weights stores 2 extra FP16 values (scale + zero)
            # Overhead per weight = 2 * 16 / group_size bits
            overhead_bpw = 2 * 16 / group_size
            info.bits_per_weight = quant_bits + overhead_bpw
        else:
            # Check if directory name hints at quantization
            dir_upper = dir_name.upper()
            if "4BIT" in dir_upper or "4-BIT" in dir_upper or "W4" in dir_upper:
                info.quantization_type = "MLX-4bit"
                info.bits_per_weight = 4.5
            elif "8BIT" in dir_upper or "8-BIT" in dir_upper or "W8" in dir_upper:
                info.quantization_type = "MLX-8bit"
                info.bits_per_weight = 8.5
            elif "3BIT" in dir_upper or "3-BIT" in dir_upper or "W3" in dir_upper:
                info.quantization_type = "MLX-3bit"
                info.bits_per_weight = 3.5
            else:
                # Check torch_dtype for unquantized models
                dtype = config.get("torch_dtype", "float16")
                if dtype in ("float32", "fp32"):
                    info.quantization_type = "F32"
                    info.bits_per_weight = 32.0
                elif dtype in ("bfloat16", "bf16"):
                    info.quantization_type = "BF16"
                    info.bits_per_weight = 16.0
                else:
                    info.quantization_type = "F16"
                    info.bits_per_weight = 16.0

        # Estimate parameter count
        if info.bits_per_weight > 0 and info.file_size_bytes > 0:
            # Safetensors files are the bulk of the size
            safetensor_size = sum(
                f.stat().st_size for f in dir_path.glob("*.safetensors")
            )
            if safetensor_size == 0:
                safetensor_size = info.file_size_bytes * 0.95  # fallback
            bytes_per_param = info.bits_per_weight / 8
            info.parameter_count = int(safetensor_size / bytes_per_param)

        # Count tensor files
        info.tensor_count = len(list(dir_path.glob("*.safetensors")))

    except Exception as e:
        info.parse_error = str(e)

    return info


def _is_mlx_model_dir(dir_path: Path) -> bool:
    """Check if a directory looks like an MLX model (has config.json + safetensors)."""
    has_config = (dir_path / "config.json").exists()
    has_safetensors = any(dir_path.glob("*.safetensors"))
    return has_config and has_safetensors


import re

# Pattern for split GGUF files: name-00001-of-00003.gguf
_SPLIT_PATTERN = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Files to skip (vision projectors, tokenizers, etc. — not the main LLM)
_SKIP_PREFIXES = ("mmproj",)


def scan_models(models_dir: str | None = None) -> list[ModelInfo]:
    """
    Recursively scan a directory for GGUF files and MLX model directories.
    Handles split GGUF shards by grouping them into a single model entry.
    Returns a list of ModelInfo objects.
    """
    if models_dir is None:
        models_dir = get_lm_studio_models_dir()

    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        return models

    # ── GGUF files ───────────────────────────────────────────────────────
    # Group split shards: {base_name: [shard_paths]}
    split_groups: dict[str, list[Path]] = {}
    standalone_gguf: list[Path] = []

    for gguf_file in models_path.rglob("*.gguf"):
        if not gguf_file.is_file():
            continue

        # Skip non-LLM files (vision projectors, etc.)
        if any(gguf_file.name.lower().startswith(p) for p in _SKIP_PREFIXES):
            continue

        match = _SPLIT_PATTERN.match(gguf_file.name)
        if match:
            base_name = match.group(1)
            # Use parent dir + base name as the group key
            group_key = str(gguf_file.parent / base_name)
            split_groups.setdefault(group_key, []).append(gguf_file)
        else:
            standalone_gguf.append(gguf_file)

    # Parse standalone GGUF files
    for gguf_file in standalone_gguf:
        info = parse_gguf_metadata(str(gguf_file))
        models.append(info)

    # Parse split GGUF groups — metadata comes from shard 1, file size is the sum
    for group_key, shards in split_groups.items():
        shards.sort(key=lambda p: p.name)  # ensure shard order

        # Parse metadata from the first shard (shard 00001 has the header)
        info = parse_gguf_metadata(str(shards[0]))

        # Sum file sizes across all shards
        total_size = sum(s.stat().st_size for s in shards)
        info.file_size_bytes = total_size

        # Update display name to show it's a combined model
        shard_count = len(shards)
        base_name = shards[0].name.rsplit("-", 2)[0]  # strip the -00001-of-00003 part
        info.file_name = f"{base_name}.gguf ({shard_count} shards)"
        info.file_path = str(shards[0].parent / base_name)

        models.append(info)

    # ── MLX model directories ────────────────────────────────────────────
    # MLX models are directories containing config.json + *.safetensors
    found_mlx_dirs: set[str] = set()
    for config_file in models_path.rglob("config.json"):
        model_dir = config_file.parent
        # Skip if this dir is inside an already-found MLX dir
        if any(str(model_dir).startswith(d + os.sep) for d in found_mlx_dirs):
            continue
        if _is_mlx_model_dir(model_dir):
            found_mlx_dirs.add(str(model_dir))
            info = parse_mlx_model(str(model_dir))
            models.append(info)

    # Sort by file name
    models.sort(key=lambda m: m.file_name.lower())
    return models
