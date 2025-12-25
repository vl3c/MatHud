from __future__ import annotations

import math
from typing import Any

from constants import default_font_family, label_min_screen_font_px, label_vanish_threshold_px
from rendering.helpers.font_helpers import _coerce_font_size, _compute_zoom_adjusted_font_size
from rendering.primitives import FontStyle, TextAlignment


def get_label_lines(label: Any):
    try:
        lines = list(getattr(label, "lines", []))
    except Exception:
        lines = []
    if not lines:
        text_value = str(getattr(label, "text", ""))
        lines = text_value.split("\n") if text_value else [""]
    return lines


def compute_world_label_font(label: Any, style: dict, coordinate_mapper: Any):
    raw_font_size = getattr(label, "font_size", style.get("label_font_size", 14))
    fallback_font = style.get("label_font_size", 14)
    base_font_size = _coerce_font_size(raw_font_size, fallback_font)
    effective_font_size = _compute_zoom_adjusted_font_size(base_font_size, label, coordinate_mapper)

    if effective_font_size <= 0:
        return None, base_font_size, effective_font_size

    if math.isfinite(effective_font_size) and effective_font_size.is_integer():
        font_size_final = int(effective_font_size)
    else:
        font_size_final = effective_font_size

    font_family = style.get("label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size_final)
    return font, base_font_size, effective_font_size


def build_world_label_metadata(index: int, position: Any, offset_y: float, rotation_degrees: float, label: Any, base_font_size: float):
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


