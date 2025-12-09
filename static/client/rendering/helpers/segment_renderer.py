from __future__ import annotations

from constants import point_label_font_size
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment


@_manages_shape
def _render_segment(primitives, start, end, stroke):
    primitives.stroke_line(start, end, stroke, include_width=False)

def _render_segment_label(primitives, segment, stroke, start, end, style):
    """Render the segment's embedded label at the midpoint with a fixed screen offset."""
    label = getattr(segment, "label", None)
    if not label:
        return
    try:
        text_value = str(getattr(label, "text", "") or "")
        visible_value = bool(getattr(label, "visible", True))
    except Exception:
        return
    if not visible_value or not text_value:
        return
    try:
        # Compute midpoint in MATH space (for metadata used in reprojection).
        math_mid_x = (segment.point1.x + segment.point2.x) / 2
        math_mid_y = (segment.point1.y + segment.point2.y) / 2

        # Compute midpoint in screen space from the already-converted endpoints.
        screen_mid_x = (start[0] + end[0]) / 2
        screen_mid_y = (start[1] + end[1]) / 2

        radius_raw = style.get("point_radius", 0) or 0
        try:
            radius = float(radius_raw)
        except Exception:
            radius = 0.0

        # Apply fixed pixel offset (same as point labels).
        draw_x = screen_mid_x + radius
        draw_y = screen_mid_y - radius

        label_lines = []
        try:
            label_lines = list(getattr(label, "lines", []))
        except Exception:
            pass
        if not label_lines:
            label_lines = text_value.split("\n") if text_value else [""]

        font_size = getattr(label, "font_size", point_label_font_size)
        label_font_size = float(font_size) if isinstance(font_size, (int, float)) else point_label_font_size

        font_family = style.get("label_font_family", style.get("font_family"))
        font = FontStyle(family=font_family, size=label_font_size)
        line_height = float(label_font_size) * 1.2
        alignment = TextAlignment(horizontal="left", vertical="alphabetic")

        for idx, line in enumerate(label_lines):
            # Metadata for the reprojection system: use point_label format so the
            # cached_render_plan reprojects using math_position + fixed screen_offset.
            # Each line has its own y-offset to account for line_height.
            line_offset_y = float(-radius) + idx * line_height
            label_metadata = {
                "point_label": {
                    "math_position": (float(math_mid_x), float(math_mid_y)),
                    "screen_offset": (float(radius), line_offset_y),
                }
            }
            primitives.draw_text(
                str(line),
                (draw_x, draw_y + idx * line_height),
                font,
                stroke.color,
                alignment,
                screen_space=True,
                metadata=label_metadata,
            )
    except Exception:
        return


def render_segment_helper(primitives, segment, coordinate_mapper, style):
    try:
        start = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        end = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
    except Exception:
        return
    if not start or not end:
        return
    stroke = StrokeStyle(
        color=str(getattr(segment, "color", style.get("segment_color", "#000"))),
        width=float(style.get("segment_stroke_width", 1) or 1),
    )
    _render_segment(primitives, start, end, stroke)
    _render_segment_label(primitives, segment, stroke, start, end, style)

