"""Font helper utilities for zoom-adjusted text sizing.

This module provides functions for computing font sizes that adapt to
zoom level, ensuring labels remain readable at all scales.

Key Features:
    - Font size coercion with fallback chain
    - Zoom-aware font scaling based on reference scale
    - Minimum size thresholds to maintain readability
    - Vanishing threshold for hiding labels at extreme zoom-out
"""

from __future__ import annotations

import math
from typing import Any

from constants import label_min_screen_font_px, label_vanish_threshold_px


def _coerce_font_size(candidate: Any, fallback: Any, default_value: float = 14.0) -> float:
    """Coerce a font size value to a valid float with fallbacks.

    Args:
        candidate: Primary font size value to try.
        fallback: Secondary value to try if candidate is invalid.
        default_value: Final fallback if both values are invalid.

    Returns:
        Valid positive float font size.
    """
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
    """Compute font size adjusted for current zoom level.

    Scales the font size based on the ratio between the current scale
    and the reference scale stored when the label was created.

    Args:
        base_size: Original font size in pixels.
        label: Label object with optional reference_scale_factor attribute.
        coordinate_mapper: Mapper with current scale_factor.

    Returns:
        Adjusted font size. Returns 0.0 if below vanish threshold,
        or at least label_min_screen_font_px if visible.
    """
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

