#!/usr/bin/env python3
"""Curses-based TUI for viewing LLM memory requirements."""

import curses
import sys

from gguf_scanner import scan_models, ModelInfo
from vram_calc import estimate_vram
from system_info import detect_system_memory
from lms_cli import list_models, is_available as lms_available, LMSModelInfo


def _load_models() -> list[ModelInfo]:
    """Load models via GGUF scanner (file-based)."""
    try:
        return scan_models()
    except Exception:
        return []


def _load_lms_models() -> list[LMSModelInfo]:
    """Load models via lms CLI if available."""
    if lms_available():
        try:
            return list_models()
        except Exception:
            pass
    return []


def _estimate_for_model(model: ModelInfo, context_length: int):
    """Run VRAM estimation for a model at a given context length."""
    attn_layers = model.metadata.get("_attention_layer_count", 0) if model.metadata else 0
    return estimate_vram(
        parameter_count=model.parameter_count,
        bits_per_weight=model.bits_per_weight,
        context_length=context_length,
        hidden_dim=model.embedding_length,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads,
        file_size_bytes=model.file_size_bytes,
        attention_layer_count=attn_layers,
        expert_count=model.expert_count,
        expert_used_count=model.expert_used_count,
    )


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def _format_params(count: int) -> str:
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    if count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return str(count)


def _bar(ratio: float, width: int) -> str:
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _draw(stdscr, models, sys_info, selected, scroll, ctx_length):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    if h < 10 or w < 40:
        stdscr.addstr(0, 0, "Terminal too small")
        return

    # Colors
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)

    GREEN = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    RED = curses.color_pair(3)
    CYAN = curses.color_pair(4)
    HEADER = curses.color_pair(5) | curses.A_BOLD

    avail = sys_info["estimated_available_gb"]

    # ── Header ──
    title = " LLM VRAM Calculator "
    chip = sys_info.get("chip", "Unknown")
    ram = sys_info["total_ram_gb"]
    hdr = f"{title} │ {chip} │ RAM: {ram:.0f} GB │ Available: {avail:.1f} GB │ Context: {ctx_length}"
    stdscr.addstr(0, 0, hdr[:w].ljust(w), HEADER)

    if not models:
        stdscr.addstr(2, 1, "No models found. Ensure LM Studio models directory exists.")
        stdscr.addstr(3, 1, "Press 'q' to quit.")
        stdscr.refresh()
        return

    # ── Left pane: model list ──
    pane_w = min(w // 2, 50)
    detail_x = pane_w + 1
    list_h = h - 3  # rows available for model list

    stdscr.addstr(1, 1, "Models (↑↓ navigate, ←→ context, q quit)", CYAN)

    visible = models[scroll:scroll + list_h]
    for i, m in enumerate(visible):
        y = i + 2
        if y >= h:
            break
        idx = scroll + i
        name = m.file_name[:pane_w - 4]
        attr = curses.A_REVERSE if idx == selected else 0
        line = f" {name}".ljust(pane_w)
        stdscr.addstr(y, 0, line[:pane_w], attr)

    # ── Right pane: detail for selected model ──
    if selected < len(models):
        m = models[selected]
        est = _estimate_for_model(m, ctx_length)
        dw = w - detail_x - 1
        if dw < 20:
            stdscr.refresh()
            return

        row = 1
        def put(text, attr=0):
            nonlocal row
            if row < h:
                stdscr.addstr(row, detail_x, text[:dw], attr)
                row += 1

        put(f"┌{'─' * (dw - 2)}┐")
        put(f"│ {m.model_name or m.file_name}", curses.A_BOLD)
        put(f"│ Format: {m.model_format}  Arch: {m.architecture}")
        put(f"│ Params: {_format_params(m.parameter_count)}  Quant: {m.quantization_type} ({m.bits_per_weight:.1f} bpw)")
        put(f"│ File: {_format_size(m.file_size_bytes)}  Layers: {m.num_layers}  Heads: {m.num_heads}/{m.num_kv_heads}")
        if m.expert_count > 1:
            put(f"│ MoE: {m.expert_used_count}/{m.expert_count} experts active")
        if m.context_length:
            put(f"│ Native context: {m.context_length:,}")
        put(f"├{'─' * (dw - 2)}┤")
        put(f"│ VRAM Estimate @ {ctx_length:,} ctx", curses.A_BOLD)
        put(f"│  Model weights:  {est.model_weights_gb:6.2f} GB")
        put(f"│  KV cache:       {est.kv_cache_gb:6.2f} GB")
        put(f"│  Activations:    {est.activation_overhead_gb:6.2f} GB")
        put(f"│  Runtime:        {est.runtime_overhead_gb:6.2f} GB")
        put(f"│  ─────────────────────")
        put(f"│  TOTAL:          {est.total_gb:6.2f} GB", curses.A_BOLD)
        put(f"├{'─' * (dw - 2)}┤")

        # Fit indicator
        if avail > 0:
            ratio = min(est.total_gb / avail, 1.0)
            bar_w = dw - 20
            if bar_w > 5:
                bar = _bar(ratio, bar_w)
                pct = ratio * 100
                color = GREEN if pct <= 75 else YELLOW if pct <= 95 else RED
                put(f"│  [{bar}] {pct:.0f}%", color)
                if est.total_gb <= avail:
                    put(f"│  ✓ Fits ({avail - est.total_gb:.1f} GB headroom)", GREEN)
                else:
                    put(f"│  ✗ Exceeds by {est.total_gb - avail:.1f} GB", RED)

        # Context sweet spot
        if avail > 0 and m.parameter_count > 0:
            lo, hi, best = 512, max(ctx_length, m.context_length or 131072), 512
            while lo <= hi:
                mid = (lo + hi) // 2
                e = _estimate_for_model(m, mid)
                if e.total_gb <= avail:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            put(f"│  Max context in budget: ~{best:,}")

        put(f"└{'─' * (dw - 2)}┘")

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    curses.use_default_colors()
    stdscr.timeout(100)

    sys_info = detect_system_memory()
    models = _load_models()

    # Enrich with lms data if available
    lms_models = _load_lms_models()
    lms_by_path = {}
    for lm in lms_models:
        if lm.path:
            lms_by_path[lm.path] = lm

    selected = 0
    scroll = 0
    ctx_length = 4096

    while True:
        h, _ = stdscr.getmaxyx()
        list_h = h - 3
        if list_h < 1:
            list_h = 1

        # Keep scroll/selection in bounds
        if models:
            selected = max(0, min(selected, len(models) - 1))
            if selected < scroll:
                scroll = selected
            if selected >= scroll + list_h:
                scroll = selected - list_h + 1

        _draw(stdscr, models, sys_info, selected, scroll, ctx_length)

        key = stdscr.getch()
        if key == -1:
            continue
        if key in (ord('q'), ord('Q')):
            break
        elif key == curses.KEY_UP and selected > 0:
            selected -= 1
        elif key == curses.KEY_DOWN and selected < len(models) - 1:
            selected += 1
        elif key == curses.KEY_RIGHT:
            ctx_length = min(ctx_length * 2, 262144)
        elif key == curses.KEY_LEFT:
            ctx_length = max(ctx_length // 2, 512)
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - list_h)
        elif key == curses.KEY_NPAGE:
            selected = min(len(models) - 1, selected + list_h)
        elif key == curses.KEY_HOME:
            selected = 0
        elif key == curses.KEY_END:
            selected = len(models) - 1
        elif key == ord('r') or key == ord('R'):
            models = _load_models()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
