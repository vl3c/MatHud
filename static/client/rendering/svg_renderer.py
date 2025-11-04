"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch: the renderer maintains a mapping from model classes
to handler methods. This keeps models strictly math-space and renderer-agnostic.

Initially only provides clear() and a minimal registry without handlers. Shapes
will be added incrementally (Point first), ensuring non-breaking integration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from browser import document, svg

from constants import default_color, default_point_size, point_label_font_size
from utils.math_utils import MathUtils
from rendering.function_renderable import FunctionRenderable
from rendering.interfaces import RendererProtocol
from rendering.style_manager import get_renderer_style, get_default_style_value


class SvgRenderer(RendererProtocol):
    """SVG-based renderer.

    Parameters
    ----------
    style_config : dict | None
        Dict holding visual configuration such as colors, sizes, font sizes.
        Optional and can be extended per shape over time.
    """

    def __init__(self, style_config: Optional[Dict[str, Any]] = None) -> None:
        self.style: Dict[str, Any] = get_renderer_style(style_config)
        # Handlers will be populated incrementally per shape
        # Example shape registrations will be added in later steps
        self._handlers_by_type: Dict[type, Callable[[Any, Any], None]] = {}

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
        # Prefer model color, then style, then default
        color: str = getattr(point, 'color', self.style.get('point_color', default_color))
        radius_val: float = self.style.get('point_radius', default_point_size)
        font_size_val: float = self.style.get('point_label_font_size', point_label_font_size)

        x: float
        y: float
        x, y = coordinate_mapper.math_to_screen(point.x, point.y)

        # Draw point
        circle_el: Any = svg.circle(cx=str(x), cy=str(y), r=str(radius_val), fill=color)
        document["math-svg"] <= circle_el

        # Draw label including math coordinates, matching existing format
        label_text: str = point.name + f'({round(point.x, 3)}, {round(point.y, 3)})'
        label_offset: int = int(radius_val)
        text_el: Any = svg.text(label_text, x=str(x + label_offset), y=str(y - label_offset), fill=color)
        text_el.style['user-select'] = 'none'
        text_el.style['-webkit-user-select'] = 'none'
        text_el.style['-moz-user-select'] = 'none'
        text_el.style['-ms-user-select'] = 'none'
        text_el.setAttribute('font-size', f'{int(font_size_val)}px')
        document["math-svg"] <= text_el

    # ----------------------- Segment -----------------------
    def register_segment(self, segment_cls: type) -> None:
        self.register(segment_cls, self._render_segment)

    def _render_segment(self, segment: Any, coordinate_mapper: Any) -> None:
        # Prefer model color, then style, then default
        color: str = getattr(segment, 'color', self.style.get('segment_color', default_color))

        x1: float
        y1: float
        x1, y1 = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        x2: float
        y2: float
        x2, y2 = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)

        line_el: Any = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=color)
        document["math-svg"] <= line_el

    # ----------------------- Circle -----------------------
    def register_circle(self, circle_cls: type) -> None:
        self.register(circle_cls, self._render_circle)

    def _render_circle(self, circle: Any, coordinate_mapper: Any) -> None:
        color: str = getattr(circle, 'color', self.style.get('circle_color', default_color))
        cx: float
        cy: float
        cx, cy = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        r_screen: float = coordinate_mapper.scale_value(circle.radius)
        circle_el: Any = svg.circle(cx=str(cx), cy=str(cy), r=str(r_screen), fill="none", stroke=color)
        if self.style['circle_stroke_width']:
            circle_el.setAttribute('stroke-width', str(self.style['circle_stroke_width']))
        document["math-svg"] <= circle_el

    # ----------------------- Ellipse -----------------------
    def register_ellipse(self, ellipse_cls: type) -> None:
        self.register(ellipse_cls, self._render_ellipse)

    def _render_ellipse(self, ellipse: Any, coordinate_mapper: Any) -> None:
        color: str = getattr(ellipse, 'color', self.style.get('ellipse_color', default_color))
        cx: float
        cy: float
        cx, cy = coordinate_mapper.math_to_screen(ellipse.center.x, ellipse.center.y)
        rx: float = coordinate_mapper.scale_value(ellipse.radius_x)
        ry: float = coordinate_mapper.scale_value(ellipse.radius_y)
        # Apply rotation around center if needed
        transform: Optional[str] = None
        if getattr(ellipse, 'rotation_angle', 0) not in (0, None):
            transform = f"rotate({-ellipse.rotation_angle} {cx} {cy})"
        el: Any
        if transform:
            el = svg.ellipse(cx=str(cx), cy=str(cy), rx=str(rx), ry=str(ry), fill="none", stroke=color, transform=transform)
        else:
            el = svg.ellipse(cx=str(cx), cy=str(cy), rx=str(rx), ry=str(ry), fill="none", stroke=color)
        if self.style['ellipse_stroke_width']:
            el.setAttribute('stroke-width', str(self.style['ellipse_stroke_width']))
        document["math-svg"] <= el

    # ----------------------- Vector -----------------------
    def register_vector(self, vector_cls: type) -> None:
        self.register(vector_cls, self._render_vector)

    def _render_vector(self, vector: Any, coordinate_mapper: Any) -> None:
        # Draw underlying segment
        seg: Any = vector.segment
        seg_color: str = getattr(vector, 'color', getattr(seg, 'color', self.style.get('vector_color', default_color)))

        x1: float
        y1: float
        x1, y1 = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
        x2: float
        y2: float
        x2, y2 = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)

        line_el: Any = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=seg_color)
        if self.style['segment_stroke_width']:
            line_el.setAttribute('stroke-width', str(self.style['segment_stroke_width']))
        document["math-svg"] <= line_el

        # Draw arrow tip at the end (near tip point)
        import math as _math
        dx: float = x2 - x1
        dy: float = y2 - y1
        angle: float = _math.atan2(dy, dx)
        side_length: float = self.style.get('vector_tip_size', default_point_size * 4)
        half_base: float = side_length / 2
        height: float = (_math.sqrt(side_length * side_length - half_base * half_base)
                  if side_length >= half_base else side_length)

        # Triangle points around (x2, y2)
        p1x: float = x2
        p1y: float = y2
        p2x: float = x2 - height * _math.cos(angle) - half_base * _math.sin(angle)
        p2y: float = y2 - height * _math.sin(angle) + half_base * _math.cos(angle)
        p3x: float = x2 - height * _math.cos(angle) + half_base * _math.sin(angle)
        p3y: float = y2 - height * _math.sin(angle) - half_base * _math.cos(angle)

        points_str: str = f"{p1x},{p1y} {p2x},{p2y} {p3x},{p3y}"
        poly_el: Any = svg.polygon(points=points_str, fill=seg_color, stroke=seg_color)
        document["math-svg"] <= poly_el

    # ----------------------- Angle -----------------------
    def register_angle(self, angle_cls: type) -> None:
        self.register(angle_cls, self._render_angle)

    def _render_angle(self, angle: Any, coordinate_mapper: Any) -> None:
        # Compute screen coordinates for vertex and arms
        vx: float
        vy: float
        vx, vy = coordinate_mapper.math_to_screen(angle.vertex_point.x, angle.vertex_point.y)
        p1x: float
        p1y: float
        p1x, p1y = coordinate_mapper.math_to_screen(angle.arm1_point.x, angle.arm1_point.y)
        p2x: float
        p2y: float
        p2x, p2y = coordinate_mapper.math_to_screen(angle.arm2_point.x, angle.arm2_point.y)

        # Use model's precise arc parameter computation for correct curvature and flags
        # Provide arc radius from style if set
        style_radius: Optional[float] = self.style.get('angle_arc_radius') if isinstance(self.style, dict) else None
        arc_params: Optional[Dict[str, Any]] = angle._calculate_arc_parameters(vx, vy, p1x, p1y, p2x, p2y, arc_radius=style_radius)
        if not arc_params:
            return

        color: str = getattr(angle, 'color', self.style.get('angle_color', default_color))
        # Build SVG path from parameters
        d: str = (
            f"M {arc_params['arc_start_x']} {arc_params['arc_start_y']} "
            f"A {arc_params['arc_radius_on_screen']} {arc_params['arc_radius_on_screen']} 0 "
            f"{arc_params['final_large_arc_flag']} {arc_params['final_sweep_flag']} "
            f"{arc_params['arc_end_x']} {arc_params['arc_end_y']}"
        )
        path_el: Any = svg.path(d=d, stroke=color, **{'stroke-width': '1', 'fill': 'none', 'class': 'angle-arc'})
        document["math-svg"] <= path_el

        # Label placement mirroring model logic
        if getattr(angle, 'angle_degrees', None) is not None:
            import math as _math
            display_angle_for_text_rad: float = _math.radians(angle.angle_degrees)
            effective_text_mid_arc_delta_rad: float = display_angle_for_text_rad / 2.0
            if arc_params["final_sweep_flag"] == '0':  # CW
                effective_text_mid_arc_delta_rad = -effective_text_mid_arc_delta_rad
            text_angle_rad: float = arc_params["angle_v_p1_rad"] + effective_text_mid_arc_delta_rad
            text_r: float = arc_params["arc_radius_on_screen"] * self.style.get(
                'angle_text_arc_radius_factor',
                get_default_style_value('angle_text_arc_radius_factor'),
            )
            tx: float = vx + text_r * _math.cos(text_angle_rad)
            ty: float = vy + text_r * _math.sin(text_angle_rad)
            text: str = f"{angle.angle_degrees:.1f}°"
            text_el: Any = svg.text(text, x=str(tx), y=str(ty), fill=color)
            text_el.setAttribute('font-size', str(int(self.style['angle_label_font_size'])))
            text_el.style['text-anchor'] = 'middle'
            text_el.style['dominant-baseline'] = 'middle'
            document["math-svg"] <= text_el

    # ----------------------- Triangle -----------------------
    def register_triangle(self, triangle_cls: type) -> None:
        self.register(triangle_cls, self._render_triangle)

    def _render_triangle(self, triangle: Any, coordinate_mapper: Any) -> None:
        # Triangles consist of three segments; render each as a line
        segs: tuple[Any, Any, Any] = (triangle.segment1, triangle.segment2, triangle.segment3)
        color: str = getattr(triangle, 'color', self.style.get('segment_color', default_color))
        for seg in segs:
            x1: float
            y1: float
            x1, y1 = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
            x2: float
            y2: float
            x2, y2 = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
            line_el: Any = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=color)
            document["math-svg"] <= line_el

    # ----------------------- Rectangle -----------------------
    def register_rectangle(self, rectangle_cls: type) -> None:
        self.register(rectangle_cls, self._render_rectangle)

    def _render_rectangle(self, rectangle: Any, coordinate_mapper: Any) -> None:
        segs: tuple[Any, Any, Any, Any] = (rectangle.segment1, rectangle.segment2, rectangle.segment3, rectangle.segment4)
        color: str = getattr(rectangle, 'color', self.style.get('segment_color', default_color))
        for seg in segs:
            x1: float
            y1: float
            x1, y1 = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
            x2: float
            y2: float
            x2, y2 = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
            line_el: Any = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=color)
            document["math-svg"] <= line_el

    # ----------------------- Function -----------------------
    def register_function(self, function_cls: type) -> None:
        self.register(function_cls, self._render_function)

    def _render_function(self, func: Any, coordinate_mapper: Any) -> None:
        # Build screen-space paths using validated generator to match original behavior
        try:
            # Math models are canvas-free; do not read func.canvas
            renderable: FunctionRenderable = FunctionRenderable(func, coordinate_mapper, None)
            screen_poly: Any = renderable.build_screen_paths()
            screen_paths: list[list[tuple[float, float]]] = screen_poly.paths if screen_poly else []
            if not screen_paths:
                return
            color: str = getattr(func, 'color', self.style.get('function_color', default_color))
            for sp in screen_paths:
                d: str = "M" + " L".join(f"{x},{y}" for x,y in sp)
                path_el: Any = svg.path(d=d, stroke=color, fill="none")
                if self.style['function_stroke_width']:
                    path_el.setAttribute('stroke-width', str(self.style['function_stroke_width']))
                document["math-svg"] <= path_el
            # Label at first point of first screen path
            first: list[tuple[float, float]] = screen_paths[0]
            label_offset_x: float = (1 + len(func.name)) * self.style['function_label_font_size'] / 2
            label_x: float = first[0][0] - label_offset_x
            label_y: float = max(first[0][1], self.style['function_label_font_size'])
            text_el: Any = svg.text(func.name, x=str(label_x), y=str(label_y), fill=color)
            text_el.setAttribute('font-size', str(int(self.style['function_label_font_size'])))
            document["math-svg"] <= text_el
        except Exception:
            return

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
        try:
            # Use renderable to build screen-space closed area
            from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
            renderable: FunctionsBoundedAreaRenderable = FunctionsBoundedAreaRenderable(area, coordinate_mapper)
            closed_area: Optional[Any] = renderable.build_screen_area()
            if not closed_area or not closed_area.forward_points or not closed_area.reverse_points:
                return
            # Forward and reverse points from this renderable are already in screen space
            d: str = f"M {closed_area.forward_points[0][0]},{closed_area.forward_points[0][1]}" + "".join(
                f" L {x},{y}" for x, y in closed_area.forward_points[1:]
            )
            rev_pts: List[Tuple[float, float]] = closed_area.reverse_points
            # Defensive: only convert if the area is not flagged as screen-space
            if not getattr(closed_area, 'is_screen', False):
                rev_pts = [coordinate_mapper.math_to_screen(x, y) for (x, y) in closed_area.reverse_points]
            d += "".join(f" L {x},{y}" for x, y in rev_pts)
            d += " Z"
            fill_color: str = getattr(area, 'color', 'lightblue')
            fill_opacity: str = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return

    # ----------------------- Colored Areas: FunctionSegmentBoundedColoredArea -----------------
    def register_function_segment_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_function_segment_bounded_colored_area)

    def _render_function_segment_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        try:
            from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
            renderable: FunctionSegmentAreaRenderable = FunctionSegmentAreaRenderable(area, coordinate_mapper)
            closed_area: Optional[Any] = renderable.build_screen_area(num_points=100)
            if not closed_area or not closed_area.forward_points or not closed_area.reverse_points:
                return
            d: str = f"M {closed_area.forward_points[0][0]},{closed_area.forward_points[0][1]}" + "".join(
                f" L {x},{y}" for x, y in closed_area.forward_points[1:]
            )
            d += "".join(f" L {x},{y}" for x, y in closed_area.reverse_points)
            d += " Z"
            fill_color: str = getattr(area, 'color', 'lightblue')
            fill_opacity: str = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return

    # ----------------------- Colored Areas: SegmentsBoundedColoredArea -----------------------
    def register_segments_bounded_colored_area(self, cls: type) -> None:
        self.register(cls, self._render_segments_bounded_colored_area)

    def _render_segments_bounded_colored_area(self, area: Any, coordinate_mapper: Any) -> None:
        try:
            if not area.segment2:
                x1: float
                y1: float
                x1, y1 = coordinate_mapper.math_to_screen(area.segment1.point1.x, area.segment1.point1.y)
                x2: float
                y2: float
                x2, y2 = coordinate_mapper.math_to_screen(area.segment1.point2.x, area.segment1.point2.y)
                p1: Tuple[float, float] = (x1, y1)
                p2: Tuple[float, float] = (x2, y2)
                origin_y: float = coordinate_mapper.math_to_screen(0, 0)[1]
                rev: List[Tuple[float, float]] = [(p2[0], origin_y), (p1[0], origin_y)]
                points: List[Tuple[float, float]] = [p1, p2]
            else:
                s1p1x: float
                s1p1y: float
                s1p1x, s1p1y = coordinate_mapper.math_to_screen(area.segment1.point1.x, area.segment1.point1.y)
                s1p2x: float
                s1p2y: float
                s1p2x, s1p2y = coordinate_mapper.math_to_screen(area.segment1.point2.x, area.segment1.point2.y)
                s2p1x: float
                s2p1y: float
                s2p1x, s2p1y = coordinate_mapper.math_to_screen(area.segment2.point1.x, area.segment2.point1.y)
                s2p2x: float
                s2p2y: float
                s2p2x, s2p2y = coordinate_mapper.math_to_screen(area.segment2.point2.x, area.segment2.point2.y)
                x1_min: float = min(s1p1x, s1p2x)
                x1_max: float = max(s1p1x, s1p2x)
                x2_min: float = min(s2p1x, s2p2x)
                x2_max: float = max(s2p1x, s2p2x)
                overlap_min: float = max(x1_min, x2_min)
                overlap_max: float = min(x1_max, x2_max)
                if overlap_max <= overlap_min:
                    return
                def get_y_at_x(segment: Any, x: float) -> float:
                    x1_seg: float
                    y1_seg: float
                    x1_seg, y1_seg = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
                    x2_seg: float
                    y2_seg: float
                    x2_seg, y2_seg = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
                    if x2_seg == x1_seg:
                        return y1_seg
                    t: float = (x - x1_seg) / (x2_seg - x1_seg)
                    return y1_seg + t * (y2_seg - y1_seg)
                y1_start: float = get_y_at_x(area.segment1, overlap_min)
                y1_end: float = get_y_at_x(area.segment1, overlap_max)
                y2_start: float = get_y_at_x(area.segment2, overlap_min)
                y2_end: float = get_y_at_x(area.segment2, overlap_max)
                points = [(overlap_min, y1_start), (overlap_max, y1_end)]
                rev = [(overlap_max, y2_end), (overlap_min, y2_start)]
            d: str = f"M {points[0][0]},{points[0][1]}" + "".join(f" L {x},{y}" for x,y in points[1:])
            d += "".join(f" L {x},{y}" for x,y in rev)
            d += " Z"
            fill_color: str = getattr(area, 'color', 'lightblue')
            fill_opacity: str = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return


