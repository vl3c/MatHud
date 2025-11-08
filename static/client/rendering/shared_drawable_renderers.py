from __future__ import annotations

import math

from typing import Any, Dict, Optional, Tuple

from utils.math_utils import MathUtils

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

    def fill_circle(self, center, radius, fill, stroke=None, *, screen_space=False):
        raise NotImplementedError

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        raise NotImplementedError

    def fill_polygon(
        self,
        points,
        fill,
        stroke=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def fill_joined_area(self, forward, reverse, fill):
        raise NotImplementedError

    def stroke_arc(
        self,
        center,
        radius,
        start_angle_rad,
        end_angle_rad,
        sweep_clockwise,
        stroke,
        css_class=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def draw_text(
        self,
        text,
        position,
        font,
        color,
        alignment,
        style_overrides=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def clear_surface(self):
        raise NotImplementedError

    def resize_surface(self, width, height):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional lifecycle hooks used by optimized rendering paths.
    # Default implementations are no-ops so legacy adapters do not need
    # to override them.
    # ------------------------------------------------------------------

    def begin_frame(self):
        """Hook invoked at the start of a full canvas render cycle."""
        return None

    def end_frame(self):
        """Hook invoked at the end of a full canvas render cycle."""
        return None

    def begin_batch(self, plan=None):
        """Hook invoked before executing a batched render plan."""
        return None

    def end_batch(self, plan=None):
        """Hook invoked after executing a batched render plan."""
        return None

    def execute_optimized(self, command):
        """Fallback execution path for optimized commands."""
        handler = getattr(self, getattr(command, "op", ""), None)
        if callable(handler):
            handler(*getattr(command, "args", ()), **getattr(command, "kwargs", {}))

from rendering.function_renderable import FunctionRenderable
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from rendering.primitives import ClosedArea
from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable


def render_point_helper(primitives, point, coordinate_mapper, style):
    try:
        sx_sy = coordinate_mapper.math_to_screen(point.x, point.y)  # type: ignore[attr-defined]
    except Exception as exc:
        return
    if not sx_sy:
        return
    sx, sy = sx_sy
    radius_raw = style.get("point_radius", 0) or 0
    try:
        radius = float(radius_raw)
    except Exception as exc:
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
        primitives.fill_circle((sx, sy), radius_raw, fill, screen_space=True)

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
            label_metadata = {
                "point_label": {
                    "math_position": (float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))),
                    "screen_offset": (float(radius), float(-radius)),
                }
            }
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
                screen_space=True,
                metadata=label_metadata,
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


def render_triangle_helper(primitives, triangle, coordinate_mapper, style):
    segments = []
    for attr in ("segment1", "segment2", "segment3"):
        seg = getattr(triangle, attr, None)
        if seg is None:
            return
        segments.append(seg)

    color = str(getattr(triangle, "color", style.get("segment_color", "#000")))
    stroke_width_raw = style.get("segment_stroke_width", 1) or 1
    try:
        stroke_width = float(stroke_width_raw)
    except Exception:
        stroke_width = 1.0
    stroke = StrokeStyle(color=color, width=stroke_width)

    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        for seg in segments:
            try:
                start = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)  # type: ignore[attr-defined]
                end = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
            except Exception:
                continue
            if not start or not end:
                continue
            primitives.stroke_line(start, end, stroke, include_width=False)
    finally:
        if managing_shape:
            end_shape()


def render_rectangle_helper(primitives, rectangle, coordinate_mapper, style):
    segments = []
    for attr in ("segment1", "segment2", "segment3", "segment4"):
        seg = getattr(rectangle, attr, None)
        if seg is None:
            return
        segments.append(seg)

    color = str(getattr(rectangle, "color", style.get("segment_color", "#000")))
    stroke_width_raw = style.get("segment_stroke_width", 1) or 1
    try:
        stroke_width = float(stroke_width_raw)
    except Exception:
        stroke_width = 1.0
    stroke = StrokeStyle(color=color, width=stroke_width)

    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        for seg in segments:
            try:
                start = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)  # type: ignore[attr-defined]
                end = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
            except Exception:
                continue
            if not start or not end:
                continue
            primitives.stroke_line(start, end, stroke, include_width=False)
    finally:
        if managing_shape:
            end_shape()


def render_ellipse_helper(primitives, ellipse, coordinate_mapper, style):
    try:
        center = coordinate_mapper.math_to_screen(ellipse.center.x, ellipse.center.y)  # type: ignore[attr-defined]
        radius_x = coordinate_mapper.scale_value(getattr(ellipse, "radius_x", None))  # type: ignore[attr-defined]
        radius_y = coordinate_mapper.scale_value(getattr(ellipse, "radius_y", None))
    except Exception:
        return
    if not center or radius_x is None or radius_y is None:
        return

    try:
        rx = float(radius_x)
    except Exception:
        rx = None
    try:
        ry = float(radius_y)
    except Exception:
        ry = None
    if rx is None or ry is None or rx <= 0 or ry <= 0:
        return

    color = str(getattr(ellipse, "color", style.get("ellipse_color", "#000")))
    stroke_width_raw = style.get("ellipse_stroke_width", 1) or 1
    try:
        stroke_width = float(stroke_width_raw)
    except Exception:
        stroke_width = 1.0
    stroke = StrokeStyle(color=color, width=stroke_width)

    rotation_deg = getattr(ellipse, "rotation_angle", 0) or 0
    try:
        rotation_rad = -math.radians(float(rotation_deg))
    except Exception:
        rotation_rad = 0.0

    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.stroke_ellipse(center, rx, ry, rotation_rad, stroke)
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
        angle_metadata = {
            "angle": {
                "vertex_math": (getattr(angle_obj.vertex_point, "x", 0.0), getattr(angle_obj.vertex_point, "y", 0.0)),
                "arm1_math": (getattr(angle_obj.arm1_point, "x", 0.0), getattr(angle_obj.arm1_point, "y", 0.0)),
                "arm2_math": (getattr(angle_obj.arm2_point, "x", 0.0), getattr(angle_obj.arm2_point, "y", 0.0)),
                "arc_radius_on_screen": float(params["arc_radius_on_screen"]),
                "display_degrees": float(display_degrees),
                "final_sweep_flag": params.get("final_sweep_flag", "0"),
                "text_radius_factor": float(style.get("angle_text_arc_radius_factor", 1.8)),
            }
        }
        primitives.stroke_arc(
            (vx, vy),
            float(params["arc_radius_on_screen"]),
            float(start_angle),
            float(end_angle),
            sweep_cw,
            stroke,
            css_class=css_class,
            screen_space=True,
            metadata=angle_metadata,
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
        screen_space=True,
        metadata=angle_metadata,
    )


def render_cartesian_helper(primitives, cartesian, coordinate_mapper, style):
    width = getattr(cartesian, "width", None)
    height = getattr(cartesian, "height", None)
    if width is None or height is None:
        return
    try:
        width_px = float(width)
        height_px = float(height)
    except Exception:
        return
    if not math.isfinite(width_px) or not math.isfinite(height_px) or width_px <= 0 or height_px <= 0:
        return

    try:
        ox, oy = coordinate_mapper.math_to_screen(0, 0)
    except Exception:
        return
    try:
        ox = float(ox)
        oy = float(oy)
    except Exception:
        return

    scale_factor = getattr(coordinate_mapper, "scale_factor", 1)
    try:
        scale = float(scale_factor)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale == 0:
        scale = 1.0

    tick_spacing_raw = getattr(cartesian, "current_tick_spacing", None)
    if tick_spacing_raw is None:
        tick_spacing_raw = getattr(cartesian, "default_tick_spacing", 1)
    try:
        tick_spacing = float(tick_spacing_raw)
    except Exception:
        tick_spacing = 1.0
    if not math.isfinite(tick_spacing) or tick_spacing <= 0:
        tick_spacing = 1.0

    display_tick = tick_spacing * scale
    try:
        display_tick = float(display_tick)
    except Exception:
        display_tick = scale
    if not math.isfinite(display_tick) or display_tick <= 0:
        display_tick = scale if math.isfinite(scale) and scale > 0 else 1.0
    if not math.isfinite(display_tick) or display_tick <= 0:
        display_tick = 1.0

    axis_color = str(style.get("cartesian_axis_color", "#000"))
    grid_color = str(style.get("cartesian_grid_color", "lightgrey"))
    label_color = str(style.get("cartesian_label_color", "grey"))

    tick_size_raw = style.get("cartesian_tick_size", 3)
    try:
        tick_size = float(tick_size_raw)
    except Exception:
        tick_size = 3.0
    if not math.isfinite(tick_size):
        tick_size = 3.0
    tick_size = max(tick_size, 0.0)

    tick_font_raw = style.get("cartesian_tick_font_size", 8)
    try:
        tick_font_float = float(tick_font_raw)
    except Exception:
        tick_font_float = 8.0
    if not math.isfinite(tick_font_float):
        tick_font_float = 8.0
    font = FontStyle(family="Inter, sans-serif", size=tick_font_raw)
    label_alignment = TextAlignment(horizontal="left", vertical="alphabetic")

    axis_stroke = StrokeStyle(color=axis_color, width=1)
    grid_stroke = StrokeStyle(color=grid_color, width=1)
    tick_stroke = StrokeStyle(color=axis_color, width=1)

    begin_shape = getattr(primitives, "begin_shape", None)
    end_shape = getattr(primitives, "end_shape", None)
    managing_shape = callable(begin_shape) and callable(end_shape)
    if managing_shape:
        begin_shape()
    try:
        primitives.stroke_line((0.0, oy), (width_px, oy), axis_stroke)
        primitives.stroke_line((ox, 0.0), (ox, height_px), axis_stroke)

        def draw_tick_x(x_pos: float) -> None:
            primitives.stroke_line((x_pos, oy - tick_size), (x_pos, oy + tick_size), tick_stroke)
            if abs(x_pos - ox) < 1e-6:
                primitives.draw_text(
                    "O",
                    (x_pos + 2, oy + tick_size + tick_font_float),
                    font,
                    label_color,
                    label_alignment,
                )
            else:
                value = (x_pos - ox) / scale
                label = MathUtils.format_number_for_cartesian(value)
                primitives.draw_text(
                    label,
                    (x_pos + 2, oy + tick_size + tick_font_float),
                    font,
                    label_color,
                    label_alignment,
                )

        def draw_tick_y(y_pos: float) -> None:
            primitives.stroke_line((ox - tick_size, y_pos), (ox + tick_size, y_pos), tick_stroke)
            if abs(y_pos - oy) >= 1e-6:
                value = (oy - y_pos) / scale
                label = MathUtils.format_number_for_cartesian(value)
                primitives.draw_text(
                    label,
                    (ox + tick_size + 2, y_pos - tick_size),
                    font,
                    label_color,
                    label_alignment,
                )

        def draw_grid_line_x(x_pos: float) -> None:
            primitives.stroke_line((x_pos, 0.0), (x_pos, height_px), grid_stroke)

        def draw_grid_line_y(y_pos: float) -> None:
            primitives.stroke_line((0.0, y_pos), (width_px, y_pos), grid_stroke)

        x = ox
        while x <= width_px:
            draw_grid_line_x(x)
            draw_tick_x(x)
            x += display_tick

        x = ox - display_tick
        while x >= 0.0:
            draw_grid_line_x(x)
            draw_tick_x(x)
            x -= display_tick

        y = oy
        while y <= height_px:
            draw_grid_line_y(y)
            draw_tick_y(y)
            y += display_tick

        y = oy - display_tick
        while y >= 0.0:
            draw_grid_line_y(y)
            draw_tick_y(y)
            y -= display_tick
    finally:
        if managing_shape:
            end_shape()


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
    raw_color = getattr(closed_area, "color", None)
    if not raw_color:
        raw_color = style.get("area_fill_color", "lightblue")
    raw_opacity = getattr(closed_area, "opacity", None)
    if raw_opacity is None:
        raw_opacity = style.get("area_opacity", 0.3)
    try:
        opacity = float(raw_opacity)
    except Exception:
        opacity = 0.3
    if not math.isfinite(opacity):
        opacity = 0.3
    else:
        opacity = max(0.0, min(opacity, 1.0))
    fill = FillStyle(
        color=str(raw_color),
        opacity=opacity,
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

