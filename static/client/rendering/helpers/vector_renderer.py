from __future__ import annotations

import math

from rendering.helpers.shape_decorator import _manages_shape
from rendering.renderer_primitives import FillStyle, StrokeStyle


@_manages_shape
def _render_vector(primitives, start, end, seg, color, stroke, style):
    primitives.stroke_line(start, end, stroke)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    side_length = float(style.get("vector_tip_size", (style.get("point_radius", 2) or 2) * 4))
    half_base = side_length / 2
    height_sq = max(side_length * side_length - half_base * half_base, 0.0)
    height = math.sqrt(height_sq)

    tip = end
    base1 = (
        end[0] - height * math.cos(angle) - half_base * math.sin(angle),
        end[1] - height * math.sin(angle) + half_base * math.cos(angle),
    )
    base2 = (
        end[0] - height * math.cos(angle) + half_base * math.sin(angle),
        end[1] - height * math.sin(angle) - half_base * math.cos(angle),
    )
    metadata = {
        "vector_arrow": {
            "start_math": (getattr(seg.point1, "x", 0.0), getattr(seg.point1, "y", 0.0)),
            "end_math": (getattr(seg.point2, "x", 0.0), getattr(seg.point2, "y", 0.0)),
            "tip_size": side_length,
        }
    }
    primitives.fill_polygon(
        [tip, base1, base2],
        FillStyle(color=color),
        StrokeStyle(color=color, width=1),
        screen_space=True,
        metadata=metadata,
    )


def render_vector_helper(primitives, vector, coordinate_mapper, style):
    seg = getattr(vector, "segment", None)
    if seg is None:
        return
    try:
        start = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
        end = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
    except Exception:
        return
    if not start or not end:
        return

    color = str(getattr(vector, "color", getattr(seg, "color", style.get("vector_color", "#000"))))
    stroke = StrokeStyle(color=color, width=float(style.get("segment_stroke_width", 1) or 1))
    _render_vector(primitives, start, end, seg, color, stroke, style)

