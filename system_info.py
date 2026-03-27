"""
Detect system memory configuration on macOS (Apple Silicon).

Reads:
- Total unified memory via sysctl hw.memsize
- GPU VRAM allocation via sysctl iogpu.wired_limit_mb (if manually set)
- Memory pressure and usage via vm_stat / sysctl
"""

import subprocess
import platform


def _sysctl_value(key: str) -> str | None:
    """Read a sysctl value. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def detect_system_memory() -> dict:
    """
    Detect system memory configuration.

    Returns:
        {
            "total_ram_gb": float,          # Total unified memory in GB
            "gpu_vram_limit_mb": int | None, # Manual iogpu.wired_limit_mb if set, else None
            "gpu_vram_limit_gb": float | None,
            "estimated_available_gb": float, # Estimated memory available for LLM
            "estimated_available_pct": int,  # As percentage of total
            "is_apple_silicon": bool,
            "chip": str,                    # e.g. "Apple M2 Max"
            "os_version": str,
        }
    """
    info = {
        "total_ram_gb": 0.0,
        "gpu_vram_limit_mb": None,
        "gpu_vram_limit_gb": None,
        "estimated_available_gb": 0.0,
        "estimated_available_pct": 75,
        "is_apple_silicon": False,
        "chip": "",
        "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else "",
    }

    if platform.system() != "Darwin":
        return info

    # Detect Apple Silicon
    arch = platform.machine()
    info["is_apple_silicon"] = arch == "arm64"

    # Total RAM
    memsize = _sysctl_value("hw.memsize")
    if memsize:
        try:
            info["total_ram_gb"] = int(memsize) / (1024 ** 3)
        except ValueError:
            pass

    # Chip name
    chip = _sysctl_value("machdep.cpu.brand_string")
    if chip:
        info["chip"] = chip

    # GPU VRAM limit (only present if manually set via iogpu.wired_limit_mb)
    vram_limit = _sysctl_value("iogpu.wired_limit_mb")
    if vram_limit:
        try:
            limit_mb = int(vram_limit)
            # A value of 0 means no manual limit (system default)
            if limit_mb > 0:
                info["gpu_vram_limit_mb"] = limit_mb
                info["gpu_vram_limit_gb"] = limit_mb / 1024
        except ValueError:
            pass

    # Estimate available memory for LLM
    if info["gpu_vram_limit_gb"]:
        # User has explicitly set a VRAM limit
        info["estimated_available_gb"] = info["gpu_vram_limit_gb"]
        if info["total_ram_gb"] > 0:
            info["estimated_available_pct"] = int(
                info["gpu_vram_limit_gb"] / info["total_ram_gb"] * 100
            )
    elif info["total_ram_gb"] > 0:
        # Default heuristic: ~75% of unified memory is available for GPU
        info["estimated_available_pct"] = 75
        info["estimated_available_gb"] = info["total_ram_gb"] * 0.75

    return info
