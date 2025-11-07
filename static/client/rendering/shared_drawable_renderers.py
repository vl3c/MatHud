from __future__ import annotations

import math

Point2D = tuple

class StrokeStyle:
    __slots__ = ("color", "width", "line_join", "line_cap")

    def __init__(self, color, width, line_join=None, line_cap=None, **kwargs):
        self.color = str(color)
        self.width = float(width)
        self.line_join = line_join
        self.line_cap = line_cap

class FillStyle:
    __slots__ = ("color", "opacity")

    def __init__(self, color, opacity=None, **kwargs):
        self.color = str(color)
        self.opacity = None if opacity is None else float(opacity)

class FontStyle:
    __slots__ = ("family", "size", "weight")

    def __init__(self, family, size, weight=None):
        self.family = family
        try:
            size_float = float(size)
        except Exception:
            self.size = size
        else:
            if math.isfinite(size_float) and size_float.is_integer():
                self.size = int(size_float)
            else:
                self.size = size_float
        self.weight = weight

class TextAlignment:
    __slots__ = ("horizontal", "vertical")

    def __init__(self, horizontal="left", vertical="alphabetic"):
        self.horizontal = horizontal
        self.vertical = vertical

class RendererPrimitives:
    """Backend-specific primitive surface consumed by shared helpers."""

    def stroke_line(self, start, end, stroke, *, include_width=True):
        raise NotImplementedError

    def stroke_polyline(self, points, stroke):
        raise NotImplementedError

    def stroke_circle(self, center, radius, stroke):
        raise NotImplementedError

    def fill_circle(self, center, radius, fill, stroke=None):
        raise NotImplementedError

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        raise NotImplementedError

    def fill_polygon(self, points, fill, stroke=None):
        raise NotImplementedError

    def fill_joined_area(self, forward, reverse, fill):
        raise NotImplementedError

    def stroke_arc(self, center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke):
        raise NotImplementedError

    def draw_text(self, text, position, font, color, alignment, style_overrides=None):
        raise NotImplementedError

    def clear_surface(self):
        raise NotImplementedError

    def resize_surface(self, width, height):
        raise NotImplementedError

from rendering.function_renderable import FunctionRenderable
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from rendering.primitives import ClosedArea
from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable


def render_point_helper(primitives, point, coordinate_mapper, style):
    try:
        sx_sy = coordinate_mapper.math_to_screen(point.x, point.y)  # type: ignore[attr-defined]
    except Exception:
        return
    if not sx_sy:
        return
    sx, sy = sx_sy
    radius_raw = style.get("point_radius", 0) or 0
    try:
        radius = float(radius_raw)
    except Exception:
        return
    if radius <= 0:
        return

    fill = FillStyle(color=str(getattr(point, "color", style.get("point_color", "#000"))), opacity=None)

    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.fill_circle((sx, sy), radius_raw, fill)

        label = getattr(point, "name", "")
        if label:
            # Match existing behavior: include coordinates in label text
            label_text = f"{label}({round(getattr(point, 'x', 0), 3)}, {round(getattr(point, 'y', 0), 3)})"
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
            font = FontStyle(family="Inter, sans-serif", size=font_size)
            primitives.draw_text(
                label_text,
                (sx + radius, sy - radius),
                font,
                fill.color,
                TextAlignment(horizontal="left", vertical="alphabetic"),
                {
                    "user-select": "none",
                    "-webkit-user-select": "none",
                    "-moz-user-select": "none",
                    "-ms-user-select": "none",
                },
            )
    finally:
        if managing_shape:
            end_shape()


def render_segment_helper(primitives, segment, coordinate_mapper, style):
    try:
        start = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)  # type: ignore[attr-defined]
        end = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
    except Exception:
        return
    if not start or not end:
        return
    stroke = StrokeStyle(
        color=str(getattr(segment, "color", style.get("segment_color", "#000"))),
        width=float(style.get("segment_stroke_width", 1) or 1),
    )
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.stroke_line(start, end, stroke, include_width=False)
    finally:
        if managing_shape:
            end_shape()


def render_circle_helper(primitives, circle, coordinate_mapper, style):
    try:
        center = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)  # type: ignore[attr-defined]
        radius = coordinate_mapper.scale_value(circle.radius)  # type: ignore[attr-defined]
    except Exception:
        return
    if not center or radius is None:
        return
    stroke = StrokeStyle(
        color=str(getattr(circle, "color", style.get("circle_color", "#000"))),
        width=float(style.get("circle_stroke_width", 1) or 1),
    )
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.stroke_circle(center, float(radius), stroke)
    finally:
        if managing_shape:
            end_shape()


def render_vector_helper(primitives, vector, coordinate_mapper, style):
    seg = getattr(vector, "segment", None)
    if seg is None:
        return
    try:
        start = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)  # type: ignore[attr-defined]
        end = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
    except Exception:
        return
    if not start or not end:
        return

    color = str(getattr(vector, "color", getattr(seg, "color", style.get("vector_color", "#000"))))
    stroke = StrokeStyle(color=color, width=float(style.get("segment_stroke_width", 1) or 1))
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
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
        primitives.fill_polygon([tip, base1, base2], FillStyle(color=color), StrokeStyle(color=color, width=1))
    finally:
        if managing_shape:
            end_shape()


def render_angle_helper(primitives, angle_obj, coordinate_mapper, style):
    try:
        vx, vy = coordinate_mapper.math_to_screen(angle_obj.vertex_point.x, angle_obj.vertex_point.y)  # type: ignore[attr-defined]
        p1x, p1y = coordinate_mapper.math_to_screen(angle_obj.arm1_point.x, angle_obj.arm1_point.y)
        p2x, p2y = coordinate_mapper.math_to_screen(angle_obj.arm2_point.x, angle_obj.arm2_point.y)
    except Exception:
        return

    params = angle_obj._calculate_arc_parameters(  # type: ignore[attr-defined]
        vx,
        vy,
        p1x,
        p1y,
        p2x,
        p2y,
        arc_radius=style.get("angle_arc_radius"),
    )
    if not params:
        return

    display_degrees = getattr(angle_obj, "angle_degrees", None)
    if display_degrees is None:
        return
    color = str(getattr(angle_obj, "color", style.get("angle_color", "#000")))
    stroke = StrokeStyle(color=color, width=float(style.get("angle_stroke_width", 1) or 1))
    text_radius = params["arc_radius_on_screen"] * float(style.get("angle_text_arc_radius_factor", 1.8))
    display_angle_rad = math.radians(display_degrees)
    text_delta = display_angle_rad / 2.0
    if params.get("final_sweep_flag", "0") == "0":
        text_delta = -text_delta
    text_angle = params["angle_v_p1_rad"] + text_delta
    tx = vx + text_radius * math.cos(text_angle)
    ty = vy + text_radius * math.sin(text_angle)
    font_size_value = style.get("angle_label_font_size", 12)
    try:
        font_size_float = float(font_size_value)
    except Exception:
        font_size = font_size_value
    else:
        if math.isfinite(font_size_float) and font_size_float.is_integer():
            font_size = int(font_size_float)
        else:
            font_size = font_size_float
    font = FontStyle(family="Inter, sans-serif", size=font_size)
    start_angle = math.atan2(p1y - vy, p1x - vx)
    sweep_cw = params.get("final_sweep_flag", "0") == "1"
    delta = math.radians(display_degrees)
    direction = 1 if sweep_cw else -1
    end_angle = start_angle + direction * delta
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        css_class = "angle-arc" if hasattr(primitives, "_surface") else None
        primitives.stroke_arc(
            (vx, vy),
            float(params["arc_radius_on_screen"]),
            float(start_angle),
            float(end_angle),
            sweep_cw,
            stroke,
            css_class=css_class,
        )
    finally:
        if managing_shape:
            end_shape()

    primitives.draw_text(
        f"{display_degrees:.1f}°",
        (tx, ty),
        font,
        color,
        TextAlignment(horizontal="center", vertical="middle"),
    )


def render_function_helper(primitives, func, coordinate_mapper, style):
    try:
        canvas = getattr(func, "canvas", None)
        cartesian = getattr(canvas, "cartesian2axis", None) if canvas is not None else None
        renderable = FunctionRenderable(func, coordinate_mapper, cartesian)  # type: ignore[attr-defined]
        screen_paths = renderable.build_screen_paths().paths
    except Exception:
        return
    if not screen_paths:
        return

    stroke = StrokeStyle(
        color=str(getattr(func, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
    )
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        for path in screen_paths:
            if len(path) < 2:
                continue
            primitives.stroke_polyline(path, stroke)
    finally:
        if managing_shape:
            end_shape()

    if getattr(func, "name", "") and screen_paths and screen_paths[0]:
        font_size_value = style.get("function_label_font_size", 12)
        try:
            font_size_float = float(font_size_value)
        except Exception:
            font_size = font_size_value
        else:
            if math.isfinite(font_size_float) and font_size_float.is_integer():
                font_size = int(font_size_float)
            else:
                font_size = font_size_float
        first_point = screen_paths[0][0]
        label_offset_x = (1 + len(func.name)) * font_size / 2.0
        position = (first_point[0] - label_offset_x, max(first_point[1], font_size))
        font = FontStyle(family="Inter, sans-serif", size=font_size)
        primitives.draw_text(
            func.name,
            position,
            font,
            stroke.color,
            TextAlignment(horizontal="left", vertical="alphabetic"),
        )


def render_colored_area_helper(primitives, closed_area, coordinate_mapper, style):
    if closed_area is None or not closed_area.forward_points or not closed_area.reverse_points:
        return
    forward = list(closed_area.forward_points)
    reverse = list(closed_area.reverse_points)
    if not getattr(closed_area, "is_screen", False):
        try:
            forward = [coordinate_mapper.math_to_screen(x, y) for (x, y) in forward]
            reverse = [coordinate_mapper.math_to_screen(x, y) for (x, y) in reverse]
        except Exception:
            return

    forward = _filter_valid_points(forward)
    reverse = _filter_valid_points(reverse)
    if len(forward) < 2 or len(reverse) < 1:
        return
    fill = FillStyle(
        color=str(style.get("area_fill_color", "lightblue")),
        opacity=float(style.get("area_opacity", 0.3)),
    )
    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.fill_joined_area(forward, reverse, fill)
    finally:
        if managing_shape:
            end_shape()


# ----------------------------------------------------------------------------
# High-level colored area helpers
# ----------------------------------------------------------------------------


def render_functions_bounded_area_helper(primitives, area_model, coordinate_mapper, style):
    area = build_functions_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_function_segment_area_helper(primitives, area_model, coordinate_mapper, style, *, num_points=100):
    area = build_function_segment_colored_area(area_model, coordinate_mapper, num_points=num_points)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_segments_bounded_area_helper(primitives, area_model, coordinate_mapper, style):
    area = build_segments_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


def build_functions_colored_area(area_model, coordinate_mapper):
    try:
        renderable = FunctionsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_function_segment_colored_area(area_model, coordinate_mapper, *, num_points=100):
    try:
        renderable = FunctionSegmentAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area(num_points=num_points)
    except Exception:
        return None


def build_segments_colored_area(area_model, coordinate_mapper):
    try:
        renderable = SegmentsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def _approximate_circle(cx, cy, radius, segments=24):
    if radius <= 0:
        return []
    seg = max(int(segments), 8)
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / seg),
            cy + radius * math.sin(2 * math.pi * i / seg),
        )
        for i in range(seg)
    ]


def _filter_valid_points(points):
    filtered = []
    for pt in points:
        if not pt:
            continue
        x, y = pt
        if x is None or y is None:
            continue
        filtered.append((float(x), float(y)))
    return filtered

