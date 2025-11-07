"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch maps model classes to handler methods so drawables stay
renderer-agnostic. Each handler delegates to the shared primitive helpers,
ensuring every renderer uses the same fundamental drawing algorithms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from browser import document, svg

from utils.math_utils import MathUtils
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
        """Render axes, ticks, labels, and grid lines for the Cartesian system.

        This mirrors the logic in Cartesian2Axis.draw(), using the cartesian object's
        current_tick_spacing, colors, and the canvas size.
        """
        try:
            width: float = cartesian.width
            height: float = cartesian.height
            # Origin in screen space via mapper
            ox: float
            oy: float
            ox, oy = coordinate_mapper.math_to_screen(0, 0)
            # Axes
            axis_color: str = self.style['cartesian_axis_color']
            document["math-svg"] <= svg.line(x1=str(0), y1=str(oy), x2=str(width), y2=str(oy), stroke=axis_color)
            document["math-svg"] <= svg.line(x1=str(ox), y1=str(0), x2=str(ox), y2=str(height), stroke=axis_color)

            # Tick spacing displayed in pixels
            display_tick: float = cartesian.current_tick_spacing * coordinate_mapper.scale_factor

            def draw_tick_x(x: float) -> None:
                tick_size: int = self.style['cartesian_tick_size']
                document["math-svg"] <= svg.line(x1=str(x), y1=str(oy - tick_size), x2=str(x), y2=str(oy + tick_size), stroke=axis_color)
                if abs(x - ox) > 1e-6:
                    val: float = (x - ox) / coordinate_mapper.scale_factor
                    label: str = MathUtils.format_number_for_cartesian(val)
                    tx: float = x + 2
                    ty: float = oy + tick_size + self.style['cartesian_tick_font_size']
                    t: Any = svg.text(label, x=str(tx), y=str(ty), fill=self.style['cartesian_label_color'])
                    t.setAttribute('font-size', str(self.style['cartesian_tick_font_size']))
                    document["math-svg"] <= t
                else:
                    t = svg.text('O', x=str(x + 2), y=str(oy + tick_size + self.style['cartesian_tick_font_size']), fill=self.style['cartesian_label_color'])
                    t.setAttribute('font-size', str(self.style['cartesian_tick_font_size']))
                    document["math-svg"] <= t

            def draw_tick_y(y: float) -> None:
                tick_size: int = self.style['cartesian_tick_size']
                document["math-svg"] <= svg.line(x1=str(ox - tick_size), y1=str(y), x2=str(ox + tick_size), y2=str(y), stroke=axis_color)
                if abs(y - oy) > 1e-6:
                    val: float = (oy - y) / coordinate_mapper.scale_factor
                    label: str = MathUtils.format_number_for_cartesian(val)
                    tx: float = ox + tick_size
                    ty: float = y - tick_size
                    t: Any = svg.text(label, x=str(tx), y=str(ty), fill=self.style['cartesian_label_color'])
                    t.setAttribute('font-size', str(self.style['cartesian_tick_font_size']))
                    document["math-svg"] <= t

            # Grid lines
            def draw_grid_line_x(x: float) -> None:
                document["math-svg"] <= svg.line(x1=str(x), y1=str(0), x2=str(x), y2=str(height), stroke=self.style['cartesian_grid_color'])

            def draw_grid_line_y(y: float) -> None:
                document["math-svg"] <= svg.line(x1=str(0), y1=str(y), x2=str(width), y2=str(y), stroke=self.style['cartesian_grid_color'])

            # Iterate ticks and grid in both directions
            # X positive
            x: float = ox
            while x < width:
                draw_grid_line_x(x)
                draw_tick_x(x)
                x += display_tick
            # X negative
            x = ox - display_tick
            while x > 0:
                draw_grid_line_x(x)
                draw_tick_x(x)
                x -= display_tick
            # Y positive (down)
            y: float = oy
            while y < height:
                draw_grid_line_y(y)
                draw_tick_y(y)
                y += display_tick
            # Y negative (up)
            y = oy - display_tick
            while y > 0:
                draw_grid_line_y(y)
                draw_tick_y(y)
                y -= display_tick
        except Exception:
            return

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


