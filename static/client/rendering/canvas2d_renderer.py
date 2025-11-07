from __future__ import annotations

from typing import Any, Callable, Dict

from browser import document, html

from rendering.style_manager import get_renderer_style
from rendering.interfaces import RendererProtocol
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.shared_drawable_renderers import (
    render_point_helper,
    render_segment_helper,
    render_circle_helper,
    render_vector_helper,
    render_angle_helper,
    render_function_helper,
    render_functions_bounded_area_helper,
    render_function_segment_area_helper,
    render_segments_bounded_area_helper,
    render_cartesian_helper,
)


class Canvas2DRenderer(RendererProtocol):
    """Experimental renderer backed by the Canvas 2D API using shared primitives."""

    def __init__(self, canvas_id: str = "math-canvas-2d") -> None:
        self.canvas_el = self._ensure_canvas(canvas_id)
        self.ctx = self.canvas_el.getContext("2d")
        if self.ctx is None:
            raise RuntimeError("Canvas 2D context unavailable")
        self.style: Dict[str, Any] = get_renderer_style()
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self.register_default_drawables()
        self._shared_primitives: Canvas2DPrimitiveAdapter = Canvas2DPrimitiveAdapter(self.canvas_el)

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
        cartesian.width = width
        cartesian.height = height
        render_cartesian_helper(self._shared_primitives, cartesian, coordinate_mapper, self.style)

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

    # ------------------------------------------------------------------
    # Handlers

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        render_point_helper(self._shared_primitives, point, coordinate_mapper, self.style)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        render_segment_helper(self._shared_primitives, segment, coordinate_mapper, self.style)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        render_circle_helper(self._shared_primitives, circle, coordinate_mapper, self.style)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        render_vector_helper(self._shared_primitives, vector, coordinate_mapper, self.style)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        render_angle_helper(self._shared_primitives, angle, coordinate_mapper, self.style)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        render_function_helper(self._shared_primitives, func, coordinate_mapper, self.style)

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_functions_bounded_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_function_segment_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_segments_bounded_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)

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
            canvas_el.attrs["width"] = str(pixel_width)
            canvas_el.attrs["height"] = str(pixel_height)
        canvas_el.style.width = f"{int(canvas_el.width)}px"
        canvas_el.style.height = f"{int(canvas_el.height)}px"
        canvas_el.style.position = "absolute"
        canvas_el.style.top = "0"
        canvas_el.style.left = "0"
        canvas_el.style.pointerEvents = "none"
        canvas_el.style.display = "block"
        canvas_el.style.zIndex = "10"
        return canvas_el

    def _resize_to_container(self) -> None:
        container = getattr(self.canvas_el, "parentElement", None)
        if container is None or not hasattr(container, "getBoundingClientRect"):
            return
        rect = container.getBoundingClientRect()
        if rect.width != self.canvas_el.width or rect.height != self.canvas_el.height:
            pixel_width = int(rect.width)
            pixel_height = int(rect.height)
            self.canvas_el.width = pixel_width
            self.canvas_el.height = pixel_height
            self.canvas_el.attrs["width"] = str(pixel_width)
            self.canvas_el.attrs["height"] = str(pixel_height)
        self.canvas_el.style.width = f"{int(self.canvas_el.width)}px"
        self.canvas_el.style.height = f"{int(self.canvas_el.height)}px"


