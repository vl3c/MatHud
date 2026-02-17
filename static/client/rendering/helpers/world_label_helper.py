"""World-space label rendering helper for drawing labels at math coordinates.

This module provides functions for rendering labels that scale with zoom
and are positioned in world (mathematical) coordinate space.

Key Features:
    - Zoom-adjusted font sizing based on reference scale
    - Multi-line text support with automatic line spacing
    - Rotation support for angled label text
    - Font size quantization for cleaner rendering
    - Metadata emission for label reprojection during pan/zoom
"""

from __future__ import annotations

import math
from typing import Any

from constants import default_font_family, label_min_screen_font_px, label_vanish_threshold_px
from rendering.helpers.font_helpers import _coerce_font_size, _compute_zoom_adjusted_font_size
from rendering.primitives import FontStyle, TextAlignment

_FONT_SIZE_QUANTUM_PX: float = 0.25
_FONT_SIZE_EPS: float = 1e-6


def _quantize_font_size_px(value: Any) -> Any:
    """Quantize font size to reduce cache fragmentation.

    Rounds font sizes to the nearest 0.25px quantum to improve
    caching efficiency while maintaining visual quality.

    Args:
        value: Font size value to quantize.

    Returns:
        Quantized font size as int (if whole) or float.
    """
    try:
        size_float = float(value)
    except Exception:
        return value
    if not math.isfinite(size_float) or size_float <= 0:
        return value
    nearest_int = round(size_float)
    if abs(size_float - nearest_int) <= _FONT_SIZE_EPS:
        return int(nearest_int)
    quantum = float(_FONT_SIZE_QUANTUM_PX)
    if quantum <= 0:
        return size_float
    k = int(round(size_float / quantum))
    quantized = float(k) * quantum
    nearest_int_q = round(quantized)
    if abs(quantized - nearest_int_q) <= _FONT_SIZE_EPS:
        return int(nearest_int_q)
    return round(quantized, 6)


def get_label_lines(label: Any):
    """Extract text lines from a label.

    Uses the lines attribute if available, otherwise splits
    the text attribute on newlines.

    Args:
        label: Label object with optional lines or text attribute.

    Returns:
        List of text line strings.
    """
    try:
        lines = list(getattr(label, "lines", []))
    except Exception:
        lines = []
    if not lines:
        text_value = str(getattr(label, "text", ""))
        lines = text_value.split("\n") if text_value else [""]
    return lines


def compute_world_label_font(label: Any, style: dict, coordinate_mapper: Any):
    """Compute the font for a world-space label with zoom adjustment.

    Args:
        label: Label object with font_size and reference_scale_factor.
        style: Style dictionary with label_font_size and label_font_family.
        coordinate_mapper: Mapper with current scale_factor.

    Returns:
        Tuple of (FontStyle or None, base_font_size, effective_font_size).
        FontStyle is None if the label should be hidden.
    """
    raw_font_size = getattr(label, "font_size", style.get("label_font_size", 14))
    fallback_font = style.get("label_font_size", 14)
    base_font_size = _coerce_font_size(raw_font_size, fallback_font)
    effective_font_size = _compute_zoom_adjusted_font_size(base_font_size, label, coordinate_mapper)

    if effective_font_size <= 0:
        return None, base_font_size, effective_font_size

    if math.isfinite(effective_font_size) and effective_font_size.is_integer():
        font_size_final = int(effective_font_size)
    else:
        font_size_final = _quantize_font_size_px(effective_font_size)

    font_family = style.get("label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size_final)
    return font, base_font_size, effective_font_size


def build_world_label_metadata(
    index: int, position: Any, offset_y: float, rotation_degrees: float, label: Any, base_font_size: float
):
    """Build metadata for world-space label reprojection.

    Args:
        index: Line index for multi-line labels.
        position: Label position with x, y attributes.
        offset_y: Vertical offset for this line in pixels.
        rotation_degrees: Label rotation angle.
        label: Label object with reference_scale_factor.
        base_font_size: Original font size before zoom adjustment.

    Returns:
        Dict with label metadata for reprojection.
    """
    return {
        "label": {
            "line_index": index,
            "math_position": (position.x, position.y),
            "screen_offset": (0.0, float(offset_y)),
            "rotation_degrees": rotation_degrees,
            "reference_scale_factor": getattr(label, "reference_scale_factor", 1.0),
            "base_font_size": base_font_size,
            "min_font_size": label_min_screen_font_px,
            "vanish_threshold_px": label_vanish_threshold_px,
        }
    }


def render_world_label_at_screen_point(
    primitives: Any,
    label: Any,
    coordinate_mapper: Any,
    style: dict,
    *,
    screen_x: float,
    screen_y: float,
) -> None:
    """Render a world-space label at the given screen coordinates.

    Args:
        primitives: The renderer primitives interface.
        label: Label object with text, position, color, rotation.
        coordinate_mapper: Mapper for zoom-adjusted font sizing.
        style: Style dictionary with label settings.
        screen_x: Pre-computed screen x coordinate.
        screen_y: Pre-computed screen y coordinate.
    """
    font, base_font_size, effective_font_size = compute_world_label_font(label, style, coordinate_mapper)
    if font is None:
        return

    position = getattr(label, "position", None)
    if position is None:
        return

    color = str(getattr(label, "color", style.get("label_text_color", "#000")))
    alignment = TextAlignment(horizontal="left", vertical="alphabetic")
    lines = get_label_lines(label)

    size_numeric = font.size if isinstance(font.size, (int, float)) else effective_font_size
    line_height = float(size_numeric) * 1.2

    try:
        rotation_degrees = float(getattr(label, "rotation_degrees", 0.0))
    except Exception:
        rotation_degrees = 0.0
    if not math.isfinite(rotation_degrees):
        rotation_degrees = 0.0

    for index, line in enumerate(lines):
        current_text = str(line)
        offset_y = index * line_height
        metadata = build_world_label_metadata(index, position, offset_y, rotation_degrees, label, base_font_size)
        primitives.draw_text(
            current_text,
            (screen_x, screen_y + offset_y),
            font,
            color,
            alignment,
            metadata=metadata,
        )
