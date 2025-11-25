from __future__ import annotations

import math
from typing import Any

from constants import label_min_screen_font_px, label_vanish_threshold_px


def _coerce_font_size(candidate: Any, fallback: Any, default_value: float = 14.0) -> float:
    try:
        value = float(candidate)
    except Exception:
        value = None
    if value is not None and math.isfinite(value) and value > 0:
        return value
    if isinstance(fallback, (int, float)):
        fallback_value = float(fallback)
        if math.isfinite(fallback_value) and fallback_value > 0:
            return fallback_value
    return float(default_value)


def _compute_zoom_adjusted_font_size(base_size: float, label: Any, coordinate_mapper: Any) -> float:
    reference_scale = getattr(label, "reference_scale_factor", None)
    try:
        reference_scale_value = float(reference_scale)
    except Exception:
        reference_scale_value = 1.0
    if not math.isfinite(reference_scale_value) or reference_scale_value <= 0:
        reference_scale_value = 1.0

    current_scale = getattr(coordinate_mapper, "scale_factor", 1.0)
    try:
        current_scale_value = float(current_scale)
    except Exception:
        current_scale_value = 1.0
    if not math.isfinite(current_scale_value) or current_scale_value <= 0:
        current_scale_value = 1.0

    ratio = current_scale_value / reference_scale_value if reference_scale_value else 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        ratio = 1.0

    if ratio >= 1.0:
        return base_size

    scaled = base_size * ratio
    if scaled <= label_vanish_threshold_px:
        return 0.0
    return max(scaled, label_min_screen_font_px)

