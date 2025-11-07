"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch maps model classes to handler methods so drawables stay
renderer-agnostic. Each handler delegates to the shared primitive helpers,
ensuring every renderer uses the same fundamental drawing algorithms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from browser import document, svg

from rendering.interfaces import RendererProtocol
from rendering.style_manager import get_renderer_style
from rendering.svg_primitive_adapter import SvgPrimitiveAdapter
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
    render_triangle_helper,
    render_rectangle_helper,
    render_ellipse_helper,
    render_cartesian_helper,
)


class SvgRenderer(RendererProtocol):
    """SVG-based renderer.

    Parameters
    ----------
    style_config : Optional[dict]
        Dict holding visual configuration such as colors, sizes, font sizes.
        Optional and can be extended per shape over time.
    """

    def __init__(self, style_config: Optional[Dict[str, Any]] = None) -> None:
        self.style: Dict[str, Any] = get_renderer_style(style_config)
        # Handlers will be populated incrementally per shape
        # Example shape registrations will be added in later steps
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}
        self._shared_primitives: SvgPrimitiveAdapter = SvgPrimitiveAdapter("math-svg")

    def register_default_drawables(self) -> None:
        self._register_shape("drawables.point", "Point", self._render_point)
        self._register_shape("drawables.segment", "Segment", self._render_segment)
        self._register_shape("drawables.circle", "Circle", self._render_circle)
        self._register_shape("drawables.ellipse", "Ellipse", self._render_ellipse)
        self._register_shape("drawables.vector", "Vector", self._render_vector)
        self._register_shape("drawables.angle", "Angle", self._render_angle)
        self._register_shape("drawables.function", "Function", self._render_function)
        self._register_shape("drawables.triangle", "Triangle", self._render_triangle)
        self._register_shape("drawables.rectangle", "Rectangle", self._render_rectangle)
        self._register_shape(
            "drawables.functions_bounded_colored_area",
            "FunctionsBoundedColoredArea",
            self._render_functions_bounded_colored_area,
        )
        self._register_shape(
            "drawables.function_segment_bounded_colored_area",
            "FunctionSegmentBoundedColoredArea",
            self._render_function_segment_bounded_colored_area,
        )
        self._register_shape(
            "drawables.segments_bounded_colored_area",
            "SegmentsBoundedColoredArea",
            self._render_segments_bounded_colored_area,
        )

    def _register_shape(self, module_path: str, class_name: str, handler: Callable[[Any, Any], None]) -> None:
        try:
            module = __import__(module_path, fromlist=[class_name])
            drawable_cls = getattr(module, class_name)
            self.register(drawable_cls, handler)
        except Exception:
            pass

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None:
        """Register a handler for a given drawable class."""
        self._handlers_by_type[cls] = handler

    def clear(self) -> None:
        try:
            document["math-svg"].clear()
        except Exception:
            # In non-browser environments, silently ignore
            pass

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        handler: Optional[Callable[[Any, Any], None]] = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    # ----------------------- Point -----------------------
    def register_point(self, point_cls: type) -> None:
        self.register(point_cls, self._render_point)

    def _render_point(self, point: Any, coordinate_mapper: Any) -> None:
        render_point_helper(self._shared_primitives, point, coordinate_mapper, self.style)

    # ----------------------- Segment -----------------------
    def register_segment(self, segment_cls: type) -> None:
        self.register(segment_cls, self._render_segment)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        render_segment_helper(self._shared_primitives, segment, coordinate_mapper, self.style)

    # ----------------------- Circle -----------------------
    def register_circle(self, circle_cls: type) -> None:
        self.register(circle_cls, self._render_circle)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        render_circle_helper(self._shared_primitives, circle, coordinate_mapper, self.style)

    # ----------------------- Ellipse -----------------------
    def register_ellipse(self, ellipse_cls: type) -> None:
        self.register(ellipse_cls, self._render_ellipse)

    def _render_ellipse(self, ellipse: Any, coordinate_mapper: Any) -> None:
        render_ellipse_helper(self._shared_primitives, ellipse, coordinate_mapper, self.style)

    # ----------------------- Vector -----------------------
    def register_vector(self, vector_cls: type) -> None:
        self.register(vector_cls, self._render_vector)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        render_vector_helper(self._shared_primitives, vector, coordinate_mapper, self.style)

    # ----------------------- Angle -----------------------
    def register_angle(self, angle_cls: type) -> None:
        self.register(angle_cls, self._render_angle)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        render_angle_helper(self._shared_primitives, angle, coordinate_mapper, self.style)

    # ----------------------- Triangle -----------------------
    def register_triangle(self, triangle_cls: type) -> None:
        self.register(triangle_cls, self._render_triangle)

    def _render_triangle(self, triangle: Any, coordinate_mapper: Any) -> None:
        render_triangle_helper(self._shared_primitives, triangle, coordinate_mapper, self.style)

    # ----------------------- Rectangle -----------------------
    def register_rectangle(self, rectangle_cls: type) -> None:
        self.register(rectangle_cls, self._render_rectangle)

    def _render_rectangle(self, rectangle: Any, coordinate_mapper: Any) -> None:
        render_rectangle_helper(self._shared_primitives, rectangle, coordinate_mapper, self.style)

    # ----------------------- Function -----------------------
    def register_function(self, function_cls: type) -> None:
        self.register(function_cls, self._render_function)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        render_function_helper(self._shared_primitives, func, coordinate_mapper, self.style)

    # ----------------------- Cartesian Grid -----------------------
    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None:
        render_cartesian_helper(self._shared_primitives, cartesian, coordinate_mapper, self.style)

    # ----------------------- Colored Areas: FunctionsBoundedColoredArea -----------------------
    def register_functions_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_functions_bounded_colored_area)

    def _render_functions_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_functions_bounded_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)

    # ----------------------- Colored Areas: FunctionSegmentBoundedColoredArea -----------------
    def register_function_segment_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_function_segment_bounded_colored_area)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_function_segment_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)

    # ----------------------- Colored Areas: SegmentsBoundedColoredArea -----------------------
    def register_segments_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_segments_bounded_colored_area)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        render_segments_bounded_area_helper(self._shared_primitives, area, coordinate_mapper, self.style)


