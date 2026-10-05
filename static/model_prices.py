"""OpenRouter prices for cost estimates, shared by the canvas benchmark and the scenario harness.

Used by ``scripts/benchmark_canvas_formats.py`` and by ``python -m cli.main test
scenarios --mode live --provider openrouter --dry-run``. Models missing from the
table have no estimate.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# USD per 1M tokens (input, output) on OpenRouter, as of PRICES_AS_OF.
PRICES_AS_OF = "2026-09-25"
PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.099, 0.60),
    "xiaomi/mimo-v2.6-pro": (0.435, 0.87),
}
# Dry-run cost estimates assume this many completion tokens per request (reasoning included).
ESTIMATED_COMPLETION_TOKENS = 400


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Cost of ``prompt_tokens`` in and ``completion_tokens`` out on ``model``; None when unpriced."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000
