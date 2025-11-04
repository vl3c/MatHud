from __future__ import annotations

import math
from typing import Any, Callable, Dict

from browser import document, html

from constants import default_color, default_point_size, point_label_font_size
from rendering.interfaces import RendererProtocol


class Canvas2DRenderer(RendererProtocol):
    """Experimental renderer backed by the Canvas 2D API."""

    def __init__(self, canvas_id: str = "math-canvas-2d") -> None:
        self.canvas_el = self._ensure_canvas(canvas_id)
        self.ctx = self.canvas_el.getContext("2d")
        if self.ctx is None:
            raise RuntimeError("Canvas 2D context unavailable")
        self.style: Dict[str, Any] = {
            "point_color": default_color,
            "point_radius": default_point_size,
            "point_label_font_size": point_label_font_size,
            "segment_color": default_color,
            "fill_style": "rgba(0, 0, 0, 0)",
        }
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self.register_default_drawables()

    def clear(self) -> None:
        width = self.canvas_el.width
        height = self.canvas_el.height
        self.ctx.clearRect(0, 0, width, height)

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        self._resize_to_container()
        width = self.canvas_el.width
        height = self.canvas_el.height
        self.ctx.save()
        self.ctx.strokeStyle = self.style["segment_color"]
        self.ctx.lineWidth = 1

        ox, oy = coordinate_mapper.math_to_screen(0, 0)
        self.ctx.beginPath()
        self.ctx.moveTo(0, oy)
        self.ctx.lineTo(width, oy)
        self.ctx.moveTo(ox, 0)
        self.ctx.lineTo(ox, height)
        self.ctx.stroke()
        self.ctx.restore()

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        self._handlers_by_type[cls] = handler

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

        label = point.name or ""
        if label:
            self.ctx.fillStyle = color
            font_size = self.style.get("point_label_font_size", point_label_font_size)
            self.ctx.font = f"{font_size}px Inter, sans-serif"
            self.ctx.fillText(label, sx + radius, sy - radius)
        self.ctx.restore()

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
            canvas_el.width = rect.width
            canvas_el.height = rect.height
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "none"  # opt-in activation later
        return canvas_el

    def _resize_to_container(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        if container is None or not hasattr(container, "getBoundingClientRect"):
            return
        rect = container.getBoundingClientRect()
        if rect.width != self.canvas_el.width or rect.height != self.canvas_el.height:
            self.canvas_el.width = rect.width
            self.canvas_el.height = rect.height


