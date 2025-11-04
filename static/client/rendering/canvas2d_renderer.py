from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from browser import document, html, console

from constants import (
    default_color,
    default_point_size,
    point_label_font_size,
    DEFAULT_ANGLE_ARC_SCREEN_RADIUS,
    DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR,
)
from utils.math_utils import MathUtils
from rendering.function_renderable import FunctionRenderable
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable
from rendering.interfaces import RendererProtocol


class Canvas2DRenderer(RendererProtocol):
    """Experimental renderer backed by the Canvas 2D API."""

    def __init__(self, canvas_id: str = "math-canvas-2d") -> None:
        self.canvas_el = self._ensure_canvas(canvas_id)
        self._log("### Canvas2DRenderer.__init__: canvas element", self.canvas_el)
        self.ctx = self.canvas_el.getContext("2d")
        if self.ctx is None:
            raise RuntimeError("Canvas 2D context unavailable")
        self._log("### Canvas2DRenderer.__init__: context acquired")
        self.style: Dict[str, Any] = {
            "point_color": default_color,
            "point_radius": default_point_size,
            "point_label_font_size": point_label_font_size,
            "segment_color": default_color,
            "segment_stroke_width": 1,
            "vector_color": default_color,
            "vector_tip_size": default_point_size * 4,
            "fill_style": "rgba(0, 0, 0, 0)",
            "circle_color": default_color,
            "circle_stroke_width": 1,
            "cartesian_axis_color": default_color,
            "cartesian_grid_color": "lightgrey",
            "cartesian_tick_size": 3,
            "cartesian_tick_font_size": 8,
            "cartesian_label_color": "grey",
            "angle_color": default_color,
            "angle_arc_radius": DEFAULT_ANGLE_ARC_SCREEN_RADIUS,
            "angle_label_font_size": point_label_font_size,
            "angle_text_arc_radius_factor": DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR,
            "function_color": default_color,
            "function_stroke_width": 1,
            "function_label_font_size": point_label_font_size,
            "area_fill_color": "lightblue",
            "area_opacity": 0.3,
        }
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self.register_default_drawables()

    def _log(self, *args: Any) -> None:
        try:
            # console.log(*args)
            pass
        except Exception:
            pass

    def clear(self) -> None:
        width = self.canvas_el.width
        height = self.canvas_el.height
        self.ctx.clearRect(0, 0, width, height)
        self._log("### Canvas2DRenderer.clear: cleared", width, height)

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            self._log("### Canvas2DRenderer.render: missing handler for", type(drawable))
            return False
        self._log("### Canvas2DRenderer.render: rendering", type(drawable))
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        self._resize_to_container()
        width = self.canvas_el.width
        height = self.canvas_el.height
        cartesian.width = width
        cartesian.height = height

        axis_color = self.style.get("cartesian_axis_color", default_color)
        grid_color = self.style.get("cartesian_grid_color", "lightgrey")
        tick_size = int(self.style.get("cartesian_tick_size", 3))
        tick_font_size = int(self.style.get("cartesian_tick_font_size", 8))
        label_color = self.style.get("cartesian_label_color", "grey")

        try:
            ox, oy = coordinate_mapper.math_to_screen(0, 0)
        except Exception:
            return

        display_tick = cartesian.current_tick_spacing * getattr(coordinate_mapper, "scale_factor", 1)
        if not display_tick or display_tick <= 0 or math.isinf(display_tick) or math.isnan(display_tick):
            display_tick = getattr(coordinate_mapper, "scale_factor", 1) or 1

        self.ctx.save()
        self.ctx.strokeStyle = axis_color
        self.ctx.lineWidth = 1
        self.ctx.beginPath()
        self.ctx.moveTo(0, oy)
        self.ctx.lineTo(width, oy)
        self.ctx.moveTo(ox, 0)
        self.ctx.lineTo(ox, height)
        self.ctx.stroke()

        self.ctx.font = f"{tick_font_size}px Inter, sans-serif"
        self.ctx.fillStyle = label_color

        def draw_tick_x(x: float) -> None:
            self.ctx.beginPath()
            self.ctx.moveTo(x, oy - tick_size)
            self.ctx.lineTo(x, oy + tick_size)
            self.ctx.stroke()
            if abs(x - ox) < 1e-6:
                self.ctx.fillText("O", x + 2, oy + tick_size + tick_font_size)
            else:
                value = (x - ox) / getattr(coordinate_mapper, "scale_factor", 1)
                label = MathUtils.format_number_for_cartesian(value)
                self.ctx.fillText(label, x + 2, oy + tick_size + tick_font_size)

        def draw_tick_y(y: float) -> None:
            self.ctx.beginPath()
            self.ctx.moveTo(ox - tick_size, y)
            self.ctx.lineTo(ox + tick_size, y)
            self.ctx.stroke()
            if abs(y - oy) >= 1e-6:
                value = (oy - y) / getattr(coordinate_mapper, "scale_factor", 1)
                label = MathUtils.format_number_for_cartesian(value)
                self.ctx.fillText(label, ox + tick_size + 2, y - tick_size)

        def draw_grid_line_x(x: float) -> None:
            self.ctx.save()
            self.ctx.strokeStyle = grid_color
            self.ctx.beginPath()
            self.ctx.moveTo(x, 0)
            self.ctx.lineTo(x, height)
            self.ctx.stroke()
            self.ctx.restore()

        def draw_grid_line_y(y: float) -> None:
            self.ctx.save()
            self.ctx.strokeStyle = grid_color
            self.ctx.beginPath()
            self.ctx.moveTo(0, y)
            self.ctx.lineTo(width, y)
            self.ctx.stroke()
            self.ctx.restore()

        # Positive X direction including origin
        x = ox
        while x <= width:
            draw_grid_line_x(x)
            self.ctx.strokeStyle = axis_color
            draw_tick_x(x)
            x += display_tick

        # Negative X direction
        x = ox - display_tick
        while x >= 0:
            draw_grid_line_x(x)
            self.ctx.strokeStyle = axis_color
            draw_tick_x(x)
            x -= display_tick

        # Positive Y direction including origin
        y = oy
        while y <= height:
            draw_grid_line_y(y)
            self.ctx.strokeStyle = axis_color
            draw_tick_y(y)
            y += display_tick

        # Negative Y direction
        y = oy - display_tick
        while y >= 0:
            draw_grid_line_y(y)
            self.ctx.strokeStyle = axis_color
            draw_tick_y(y)
            y -= display_tick

        self.ctx.restore()
        self._log("### Canvas2DRenderer.render_cartesian: rendered axis", width, height)

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        self._handlers_by_type[cls] = handler
        self._log("### Canvas2DRenderer.register: handler for", cls)

    def register_default_drawables(self) -> None:
        try:
            from drawables.point import Point as PointDrawable
            self.register(PointDrawable, self._render_point)
        except Exception:
            pass
        try:
            from drawables.segment import Segment as SegmentDrawable
            self.register(SegmentDrawable, self._render_segment)
        except Exception:
            pass
        try:
            from drawables.circle import Circle as CircleDrawable
            self.register(CircleDrawable, self._render_circle)
        except Exception:
            pass
        try:
            from drawables.vector import Vector as VectorDrawable
            self.register(VectorDrawable, self._render_vector)
        except Exception:
            pass
        try:
            from drawables.angle import Angle as AngleDrawable
            self.register(AngleDrawable, self._render_angle)
        except Exception:
            pass
        try:
            from drawables.function import Function as FunctionDrawable
            self.register(FunctionDrawable, self._render_function)
        except Exception:
            pass
        try:
            from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea as FunctionsAreaDrawable
            self.register(FunctionsAreaDrawable, self._render_functions_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea as FunctionSegmentAreaDrawable
            self.register(FunctionSegmentAreaDrawable, self._render_function_segment_bounded_colored_area)
        except Exception:
            pass
        try:
            from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea as SegmentsAreaDrawable
            self.register(SegmentsAreaDrawable, self._render_segments_bounded_colored_area)
        except Exception:
            pass
        self._log("### Canvas2DRenderer.register_default_drawables: completed")

    # ------------------------------------------------------------------
    # Handlers

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        sx, sy = coordinate_mapper.math_to_screen(point.x, point.y)
        radius = self.style.get("point_radius", default_point_size)
        color = getattr(point, "color", self.style.get("point_color", default_color))

        self.ctx.save()
        self.ctx.fillStyle = color
        self.ctx.beginPath()
        self.ctx.arc(sx, sy, radius, 0, 2 * math.pi)
        self.ctx.fill()

        label_name = point.name or ""
        if label_name:
            label_text = f"{label_name}({round(point.x, 3)}, {round(point.y, 3)})"
            self.ctx.fillStyle = color
            font_size = self.style.get("point_label_font_size", point_label_font_size)
            self.ctx.font = f"{font_size}px Inter, sans-serif"
            self.ctx.fillText(label_text, sx + radius, sy - radius)
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_point: rendered", point.name)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        x1, y1 = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        x2, y2 = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
        color = getattr(segment, "color", self.style.get("segment_color", default_color))

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = self.style.get("segment_stroke_width", 1)
        self.ctx.beginPath()
        self.ctx.moveTo(x1, y1)
        self.ctx.lineTo(x2, y2)
        self.ctx.stroke()
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_segment: rendered", getattr(segment, "name", None))

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        cx, cy = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        radius = coordinate_mapper.scale_value(circle.radius)
        color = getattr(circle, "color", self.style.get("circle_color", default_color))

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = self.style.get("circle_stroke_width", 1)
        self.ctx.beginPath()
        self.ctx.arc(cx, cy, radius, 0, 2 * math.pi)
        self.ctx.stroke()
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_circle: rendered", getattr(circle, "name", None))

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        seg = vector.segment
        color = getattr(vector, "color", getattr(seg, "color", self.style.get("vector_color", default_color)))
        x1, y1 = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
        x2, y2 = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = self.style.get("segment_stroke_width", 1)
        self.ctx.beginPath()
        self.ctx.moveTo(x1, y1)
        self.ctx.lineTo(x2, y2)
        self.ctx.stroke()

        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        side_length = self.style.get("vector_tip_size", default_point_size * 4)
        half_base = side_length / 2
        height = side_length if side_length < half_base else math.sqrt(max(side_length * side_length - half_base * half_base, 0))

        p1x = x2
        p1y = y2
        p2x = x2 - height * math.cos(angle) - half_base * math.sin(angle)
        p2y = y2 - height * math.sin(angle) + half_base * math.cos(angle)
        p3x = x2 - height * math.cos(angle) + half_base * math.sin(angle)
        p3y = y2 - height * math.sin(angle) - half_base * math.cos(angle)

        self.ctx.fillStyle = color
        self.ctx.beginPath()
        self.ctx.moveTo(p1x, p1y)
        self.ctx.lineTo(p2x, p2y)
        self.ctx.lineTo(p3x, p3y)
        self.ctx.closePath()
        self.ctx.fill()
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_vector: rendered", getattr(vector, "name", None))

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        vx, vy = coordinate_mapper.math_to_screen(angle.vertex_point.x, angle.vertex_point.y)
        p1x, p1y = coordinate_mapper.math_to_screen(angle.arm1_point.x, angle.arm1_point.y)
        p2x, p2y = coordinate_mapper.math_to_screen(angle.arm2_point.x, angle.arm2_point.y)

        params = angle._calculate_arc_parameters(vx, vy, p1x, p1y, p2x, p2y, arc_radius=self.style.get("angle_arc_radius", DEFAULT_ANGLE_ARC_SCREEN_RADIUS))
        if not params:
            return

        color = getattr(angle, "color", self.style.get("angle_color", default_color))
        radius = params["arc_radius_on_screen"]

        start_angle = math.atan2(p1y - vy, p1x - vx)
        sweep_cw = params.get("final_sweep_flag", '0') == '1'
        display_degrees = getattr(angle, "angle_degrees", None)
        if display_degrees is None:
            return
        delta = math.radians(display_degrees)
        direction = 1 if sweep_cw else -1
        end_angle = start_angle + direction * delta

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = 1
        self.ctx.beginPath()
        self.ctx.arc(vx, vy, radius, start_angle, end_angle, not sweep_cw)
        self.ctx.stroke()
        self.ctx.restore()

        text_radius = radius * self.style.get("angle_text_arc_radius_factor", DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR)
        text_angle = start_angle + direction * delta / 2
        tx = vx + text_radius * math.cos(text_angle)
        ty = vy + text_radius * math.sin(text_angle)
        label = f"{display_degrees:.1f}°"

        self.ctx.save()
        self.ctx.fillStyle = color
        font_size = int(self.style.get("angle_label_font_size", point_label_font_size))
        self.ctx.font = f"{font_size}px Inter, sans-serif"
        self.ctx.textAlign = "center"
        self.ctx.textBaseline = "middle"
        self.ctx.fillText(label, tx, ty)
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_angle: rendered", getattr(angle, "name", None))

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        color = getattr(func, "color", self.style.get("function_color", default_color))
        cartesian = getattr(getattr(func, "canvas", None), "cartesian2axis", None)
        renderable = FunctionRenderable(func, coordinate_mapper, cartesian)
        screen_paths = renderable.build_screen_paths().paths
        if not screen_paths:
            return

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = self.style.get("function_stroke_width", 1)
        for path in screen_paths:
            if not path:
                continue
            self.ctx.beginPath()
            self.ctx.moveTo(path[0][0], path[0][1])
            for x, y in path[1:]:
                self.ctx.lineTo(x, y)
            self.ctx.stroke()
        self.ctx.restore()

        if getattr(func, "name", "") and screen_paths and screen_paths[0]:
            first = screen_paths[0][0]
            font_size = self.style.get("function_label_font_size", point_label_font_size)
            label_offset_x = (1 + len(func.name)) * font_size / 2
            label_x = first[0] - label_offset_x
            label_y = max(first[1], font_size)
            self.ctx.save()
            self.ctx.fillStyle = color
            self.ctx.font = f"{font_size}px Inter, sans-serif"
            self.ctx.fillText(func.name, label_x, label_y)
            self.ctx.restore()
        self._log("### Canvas2DRenderer._render_function: rendered", getattr(func, "name", None))

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        renderable = FunctionsBoundedAreaRenderable(area, coordinate_mapper)
        closed_area = renderable.build_screen_area()
        if not closed_area or not closed_area.forward_points or not closed_area.reverse_points:
            return
        forward = closed_area.forward_points
        reverse = closed_area.reverse_points
        if not getattr(closed_area, "is_screen", False):
            reverse = [coordinate_mapper.math_to_screen(x, y) for (x, y) in reverse]
        self._fill_area_path(forward, reverse, getattr(area, "color", self.style.get("area_fill_color", "lightblue")), getattr(area, "opacity", self.style.get("area_opacity", 0.3)))
        self._log("### Canvas2DRenderer._render_functions_bounded_colored_area: rendered", getattr(area, "name", None))

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        renderable = FunctionSegmentAreaRenderable(area, coordinate_mapper)
        closed_area = renderable.build_screen_area(num_points=100)
        if not closed_area or not closed_area.forward_points or not closed_area.reverse_points:
            return
        self._fill_area_path(
            closed_area.forward_points,
            closed_area.reverse_points,
            getattr(area, "color", self.style.get("area_fill_color", "lightblue")),
            getattr(area, "opacity", self.style.get("area_opacity", 0.3)),
        )
        self._log("### Canvas2DRenderer._render_function_segment_bounded_colored_area: rendered", getattr(area, "name", None))

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        renderable = SegmentsBoundedAreaRenderable(area, coordinate_mapper)
        closed_area = renderable.build_screen_area()
        if not closed_area or not closed_area.forward_points or not closed_area.reverse_points:
            return
        self._fill_area_path(
            closed_area.forward_points,
            closed_area.reverse_points,
            getattr(area, "color", self.style.get("area_fill_color", "lightblue")),
            getattr(area, "opacity", self.style.get("area_opacity", 0.3)),
        )
        self._log("### Canvas2DRenderer._render_segments_bounded_colored_area: rendered", getattr(area, "name", None))

    # ------------------------------------------------------------------
    # Helpers

    def _fill_area_path(
        self,
        forward: List[Tuple[float, float]],
        reverse: List[Tuple[float, float]],
        fill_color: str,
        opacity: float,
    ) -> None:
        if not forward:
            return
        self.ctx.save()
        self.ctx.fillStyle = fill_color
        self.ctx.globalAlpha = max(0.0, min(1.0, float(opacity)))
        self.ctx.beginPath()
        self.ctx.moveTo(forward[0][0], forward[0][1])
        for x, y in forward[1:]:
            if x is None or y is None:
                continue
            self.ctx.lineTo(x, y)
        for x, y in reverse:
            if x is None or y is None:
                continue
            self.ctx.lineTo(x, y)
        self.ctx.closePath()
        self.ctx.fill()
        self.ctx.restore()

    def _ensure_canvas(self, canvas_id: str):
        canvas_el = document.getElementById(canvas_id)
        if canvas_el is None:
            canvas_el = html.CANVAS(id=canvas_id)
            container = document.getElementById("math-container")
            if container is None:
                document <= canvas_el
            else:
                container <= canvas_el
        container = getattr(canvas_el, "parentElement", None)
        rect = container.getBoundingClientRect() if hasattr(container, "getBoundingClientRect") else None
        if rect:
            pixel_width = int(rect.width)
            pixel_height = int(rect.height)
            canvas_el.width = pixel_width
            canvas_el.height = pixel_height
            canvas_el.attrs["width"] = pixel_width
            canvas_el.attrs["height"] = pixel_height
            self._log("### Canvas2DRenderer._ensure_canvas: sized to", pixel_width, pixel_height)
        else:
            self._log("### Canvas2DRenderer._ensure_canvas: no rect for container", container)
        canvas_el.style.width = f"{int(canvas_el.width)}px"
        canvas_el.style.height = f"{int(canvas_el.height)}px"
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "block"
        canvas_el.style.zIndex = "10"
        self._log("### Canvas2DRenderer._ensure_canvas: parent", container)
        return canvas_el

    def _resize_to_container(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        if container is None or not hasattr(container, "getBoundingClientRect"):
            self._log("### Canvas2DRenderer._resize_to_container: no container")
            return
        rect = container.getBoundingClientRect()
        if rect.width != self.canvas_el.width or rect.height != self.canvas_el.height:
            pixel_width = int(rect.width)
            pixel_height = int(rect.height)
            self.canvas_el.width = pixel_width
            self.canvas_el.height = pixel_height
            self.canvas_el.attrs["width"] = pixel_width
            self.canvas_el.attrs["height"] = pixel_height
            self._log("### Canvas2DRenderer._resize_to_container: resized", pixel_width, pixel_height)
        else:
            self._log("### Canvas2DRenderer._resize_to_container: size unchanged")
        self.canvas_el.style.width = f"{int(self.canvas_el.width)}px"
        self.canvas_el.style.height = f"{int(self.canvas_el.height)}px"


