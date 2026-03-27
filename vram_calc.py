"""
VRAM estimation engine for local LLM inference.

Formulas based on:
- Model weights = parameters × (bits_per_weight / 8)
- KV cache = 2 × num_layers × hidden_dim × context_length × bytes_per_element × (num_kv_heads / num_heads)
- Activation overhead ≈ 0.2 × model_weights (for inference)
- Compute buffer / runtime overhead ≈ fixed + proportional

References:
- https://twm.me/how-to-calculate-vram-requirement-local-llm-advanced/
- https://www.ywian.com/blog/llm-vram-calculator-guide
"""

from dataclasses import dataclass

# Default hidden dimensions and layer counts by approximate parameter count
# Used as fallback when GGUF metadata doesn't include these
DEFAULT_ARCH_PARAMS = {
    1:   {"hidden_dim": 2048,  "layers": 22, "heads": 16,  "kv_heads": 16},
    3:   {"hidden_dim": 3072,  "layers": 26, "heads": 24,  "kv_heads": 24},
    7:   {"hidden_dim": 4096,  "layers": 32, "heads": 32,  "kv_heads": 8},
    8:   {"hidden_dim": 4096,  "layers": 32, "heads": 32,  "kv_heads": 8},
    13:  {"hidden_dim": 5120,  "layers": 40, "heads": 40,  "kv_heads": 40},
    14:  {"hidden_dim": 5120,  "layers": 40, "heads": 40,  "kv_heads": 8},
    27:  {"hidden_dim": 5120,  "layers": 48, "heads": 40,  "kv_heads": 8},
    30:  {"hidden_dim": 7168,  "layers": 60, "heads": 56,  "kv_heads": 8},
    32:  {"hidden_dim": 5120,  "layers": 64, "heads": 40,  "kv_heads": 8},
    65:  {"hidden_dim": 8192,  "layers": 80, "heads": 64,  "kv_heads": 8},
    70:  {"hidden_dim": 8192,  "layers": 80, "heads": 64,  "kv_heads": 8},
    72:  {"hidden_dim": 8192,  "layers": 80, "heads": 64,  "kv_heads": 8},
    120: {"hidden_dim": 12288, "layers": 96, "heads": 96,  "kv_heads": 16},
    405: {"hidden_dim": 16384, "layers": 126, "heads": 128, "kv_heads": 16},
}


@dataclass
class VRAMEstimate:
    """Breakdown of VRAM usage components."""
    model_weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    activation_overhead_gb: float = 0.0
    vision_overhead_gb: float = 0.0
    runtime_overhead_gb: float = 0.0
    total_gb: float = 0.0

    # Input parameters (for display)
    parameter_count_b: float = 0.0
    bits_per_weight: float = 0.0
    context_length: int = 0
    hidden_dim: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    kv_cache_bits: int = 16
    is_vision: bool = False
    # MoE info
    expert_count: int = 0
    expert_used_count: int = 0
    is_moe: bool = False


# Map quantization names to average bits-per-weight
QUANT_BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q8_1": 9.0,
    "Q6_K": 6.56, "Q5_K_M": 5.7, "Q5_K_S": 5.5,
    "Q4_K_M": 4.8, "Q4_K_S": 4.5, "Q4_0": 4.5, "Q4_1": 5.0,
    "Q3_K_M": 3.9, "Q3_K_L": 4.3, "Q3_K_S": 3.44,
    "Q2_K": 2.56,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "IQ3_S": 3.5, "IQ3_XXS": 3.0,
    "IQ2_XS": 2.3, "IQ2_XXS": 2.0, "IQ2_S": 2.5,
    "IQ1_S": 1.625, "IQ1_M": 1.75,
    "MXFP4": 4.5,
    # Generic bit-level fallbacks (from lms quant_bits)
    "16bit": 16.0, "8bit": 8.5, "6bit": 6.56,
    "4bit": 4.5, "3bit": 3.9, "2bit": 2.56,
}


def quant_to_bpw(quant_name: str, quant_bits: int = 0) -> float:
    """Convert a quantization name or bit count to bits-per-weight."""
    if quant_name:
        # Try exact match first
        name_upper = quant_name.upper().replace("-", "_")
        if name_upper in QUANT_BPW:
            return QUANT_BPW[name_upper]
        # Try with common suffixes stripped
        for key, bpw in QUANT_BPW.items():
            if key in name_upper or name_upper in key:
                return bpw
    # Fall back to bit count
    if quant_bits > 0:
        key = f"{quant_bits}bit"
        return QUANT_BPW.get(key, float(quant_bits))
    return 4.5  # conservative default


def _closest_arch(param_billions: float) -> dict:
    """Find the closest default architecture params for a given parameter count."""
    if not DEFAULT_ARCH_PARAMS:
        return {"hidden_dim": 4096, "layers": 32, "heads": 32, "kv_heads": 8}
    closest_key = min(DEFAULT_ARCH_PARAMS.keys(), key=lambda k: abs(k - param_billions))
    return DEFAULT_ARCH_PARAMS[closest_key]


def estimate_vram(
    parameter_count: int,
    bits_per_weight: float,
    context_length: int = 4096,
    hidden_dim: int = 0,
    num_layers: int = 0,
    num_heads: int = 0,
    num_kv_heads: int = 0,
    kv_cache_bits: int = 16,
    batch_size: int = 1,
    file_size_bytes: int = 0,
    attention_layer_count: int = 0,
    is_vision: bool = False,
    expert_count: int = 0,
    expert_used_count: int = 0,
) -> VRAMEstimate:
    """
    Estimate total VRAM required for inference of a GGUF/MLX model.

    Args:
        parameter_count: Total number of model parameters
        bits_per_weight: Average bits per weight (from quantization)
        context_length: Maximum sequence length / context window
        hidden_dim: Model hidden dimension (embedding length)
        num_layers: Number of transformer layers (total blocks)
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA; equals num_heads for MHA)
        kv_cache_bits: Precision for KV cache (16 = FP16, 8 = INT8)
        batch_size: Number of concurrent sequences
        file_size_bytes: Actual file size on disk (used as ground truth for weights)
        attention_layer_count: Number of layers that have attention/KV cache
                               (for hybrid models like Mamba+Attention; 0 = all layers)
        is_vision: Whether this is a vision-language model (adds image encoder overhead)
        expert_count: Total number of experts in MoE models (0 = dense model)
        expert_used_count: Number of experts active per token in MoE models
    """
    is_moe = expert_count > 1 and expert_used_count > 0
    param_b = parameter_count / 1e9

    # Fill in missing architecture params from defaults
    if hidden_dim == 0 or num_layers == 0:
        defaults = _closest_arch(param_b)
        if hidden_dim == 0:
            hidden_dim = defaults["hidden_dim"]
        if num_layers == 0:
            num_layers = defaults["layers"]
        if num_heads == 0:
            num_heads = defaults["heads"]
        if num_kv_heads == 0:
            num_kv_heads = defaults["kv_heads"]

    if num_kv_heads == 0:
        num_kv_heads = num_heads

    # 1. Model weights — use actual file size when available (most accurate),
    #    otherwise derive from parameter count × bits per weight
    if file_size_bytes > 0:
        model_weights_gb = file_size_bytes / (1024**3)
    else:
        model_weights_gb = (parameter_count * bits_per_weight / 8) / (1024**3)

    # 2. KV cache
    # For hybrid models (Mamba + Attention), only attention layers have KV cache
    kv_layers = attention_layer_count if attention_layer_count > 0 else num_layers
    bytes_per_element = kv_cache_bits / 8
    gqa_ratio = num_kv_heads / num_heads if num_heads > 0 else 1.0
    kv_per_token_bytes = 2 * hidden_dim * kv_layers * bytes_per_element * gqa_ratio
    kv_cache_gb = (batch_size * context_length * kv_per_token_bytes) / (1024**3)

    # 3. Activation overhead (inference ≈ 10-20% of model weights)
    # MoE models only activate a subset of experts per token, so the activation
    # memory is proportional to the active parameters, not total parameters.
    if is_moe:
        # Active fraction: shared layers (attention) + active experts' FFN
        # Rough heuristic: activation scales with active params / total params
        active_ratio = expert_used_count / expert_count
        # Attention is shared (always active), FFN is per-expert.
        # In typical MoE, FFN is ~2/3 of per-layer params, attention is ~1/3.
        effective_active_ratio = 0.33 + 0.67 * active_ratio
        activation_overhead_gb = model_weights_gb * 0.15 * effective_active_ratio
    else:
        activation_overhead_gb = model_weights_gb * 0.15

    # 4. Vision encoder overhead (VLMs load an image encoder on top of the LLM)
    # Typically 300MB-1.5GB depending on the vision encoder (SigLIP, CLIP, etc.)
    vision_overhead_gb = 0.0
    if is_vision:
        # Rough estimate: ~10-15% of model weights for the vision encoder + projector
        vision_overhead_gb = max(0.3, model_weights_gb * 0.12)

    # 5. Runtime overhead (llama.cpp / LM Studio compute buffers)
    # Typically 200-500MB fixed + small proportional amount
    runtime_overhead_gb = 0.3 + model_weights_gb * 0.02

    total_gb = (model_weights_gb + kv_cache_gb + activation_overhead_gb
                + vision_overhead_gb + runtime_overhead_gb)

    return VRAMEstimate(
        model_weights_gb=round(model_weights_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        activation_overhead_gb=round(activation_overhead_gb, 2),
        vision_overhead_gb=round(vision_overhead_gb, 2),
        runtime_overhead_gb=round(runtime_overhead_gb, 2),
        total_gb=round(total_gb, 2),
        parameter_count_b=round(param_b, 2),
        bits_per_weight=bits_per_weight,
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kv_cache_bits=kv_cache_bits,
        is_vision=is_vision,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        is_moe=is_moe,
    )
