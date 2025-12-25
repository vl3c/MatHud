from __future__ import annotations

import math
from typing import Any, Dict, Optional

from constants import default_font_family
from rendering.primitives import FontStyle, TextAlignment


def draw_point_style_label_with_coords(
    primitives: Any,
    *,
    anchor_screen_x: float,
    anchor_screen_y: float,
    anchor_math_x: float,
    anchor_math_y: float,
    label: str,
    radius: float,
    color: str,
    style: Dict[str, Any],
    coord_precision: int = 3,
    non_selectable: bool = True,
    layout_group: Optional[Any] = None,
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Draw a point-style label with coordinates.

    This function intentionally preserves the existing logic from
    point_renderer._render_point_label to avoid visual drift.
    """

    if not label:
        return

    precision = int(coord_precision) if coord_precision is not None else 3
    label_text = f"{label}({round(anchor_math_x, precision)}, {round(anchor_math_y, precision)})"

    font_size_value = style.get("point_label_font_size", 10)
    try:
        font_size_float = float(font_size_value)
    except Exception:
        font_size = font_size_value
    else:
        if math.isfinite(font_size_float) and font_size_float.is_integer():
            font_size = int(font_size_float)
        else:
            font_size = font_size_float

    font_family = style.get("point_label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size)

    label_metadata: Dict[str, Any] = {
        "point_label": {
            "math_position": (float(anchor_math_x), float(anchor_math_y)),
            "screen_offset": (float(radius), float(-radius)),
            "layout_line_index": 0,
            "layout_line_count": 1,
            "layout_max_line_len": int(len(label_text)),
        }
    }
    if layout_group is not None:
        label_metadata["point_label"]["layout_group"] = layout_group
    if metadata_overrides:
        label_metadata.update(metadata_overrides)

    style_overrides = None
    if bool(non_selectable):
        style_overrides = {
            "user-select": "none",
            "-webkit-user-select": "none",
            "-moz-user-select": "none",
            "-ms-user-select": "none",
        }

    primitives.draw_text(
        label_text,
        (anchor_screen_x + radius, anchor_screen_y - radius),
        font,
        color,
        TextAlignment(horizontal="left", vertical="alphabetic"),
        style_overrides,
        screen_space=True,
        metadata=label_metadata,
    )


