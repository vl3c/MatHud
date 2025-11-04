from __future__ import annotations

import math
from typing import Any, Callable, Dict

from browser import document, html, console

from constants import default_color, default_point_size, point_label_font_size
from utils.math_utils import MathUtils
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
            "fill_style": "rgba(0, 0, 0, 0)",
            "cartesian_axis_color": default_color,
            "cartesian_grid_color": "lightgrey",
            "cartesian_tick_size": 3,
            "cartesian_tick_font_size": 8,
            "cartesian_label_color": "grey",
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
        from drawables.point import Point as PointDrawable
        self.register(PointDrawable, self._render_point)
        from drawables.segment import Segment as SegmentDrawable
        self.register(SegmentDrawable, self._render_segment)
        from drawables.circle import Circle as CircleDrawable
        self.register(CircleDrawable, self._render_circle)
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
        self.ctx.lineWidth = 1
        self.ctx.beginPath()
        self.ctx.moveTo(x1, y1)
        self.ctx.lineTo(x2, y2)
        self.ctx.stroke()
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_segment: rendered", getattr(segment, "name", None))

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        cx, cy = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        radius = coordinate_mapper.scale_value(circle.radius)
        color = getattr(circle, "color", self.style.get("segment_color", default_color))

        self.ctx.save()
        self.ctx.strokeStyle = color
        self.ctx.lineWidth = 1
        self.ctx.beginPath()
        self.ctx.arc(cx, cy, radius, 0, 2 * math.pi)
        self.ctx.stroke()
        self.ctx.restore()
        self._log("### Canvas2DRenderer._render_circle: rendered", getattr(circle, "name", None))

    # ------------------------------------------------------------------
    # Helpers

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


