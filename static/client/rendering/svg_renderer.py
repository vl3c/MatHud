"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch: the renderer maintains a mapping from model classes
to handler methods. This keeps models strictly math-space and renderer-agnostic.

Initially only provides clear() and a minimal registry without handlers. Shapes
will be added incrementally (Point first), ensuring non-breaking integration.
"""

from browser import document, svg
from constants import default_color, default_point_size, point_label_font_size, DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR


class SvgRenderer:
    """SVG-based renderer.

    Parameters
    ----------
    style_config : dict | None
        Dict holding visual configuration such as colors, sizes, font sizes.
        Optional and can be extended per shape over time.
    """

    def __init__(self, style_config=None):
        self.style = style_config or {}
        # Handlers will be populated incrementally per shape
        # Example shape registrations will be added in later steps
        self._handlers_by_type = {}

    def register(self, cls, handler):
        """Register a handler for a given drawable class."""
        self._handlers_by_type[cls] = handler

    def clear(self):
        try:
            document["math-svg"].clear()
        except Exception:
            # In non-browser environments, silently ignore
            pass

    def render(self, drawable, coordinate_mapper):
        handler = self._handlers_by_type.get(type(drawable))
        if handler is None:
            return False
        handler(drawable, coordinate_mapper)
        return True

    # ----------------------- Point -----------------------
    def register_point(self, point_cls):
        self.register(point_cls, self._render_point)

    def _render_point(self, point, coordinate_mapper):
        # Visual params: prefer style overrides; fall back to model color for now (gradual migration),
        # and constants for sizes to match existing behavior.
        color = self.style.get('point_color', getattr(point, 'color', default_color))
        radius_val = self.style.get('point_radius', default_point_size)
        font_size_val = self.style.get('point_label_font_size', point_label_font_size)

        x, y = coordinate_mapper.math_to_screen(point.original_position.x, point.original_position.y)

        # Draw point
        circle_el = svg.circle(cx=str(x), cy=str(y), r=str(radius_val), fill=color)
        document["math-svg"] <= circle_el

        # Draw label including math coordinates, matching existing format
        label_text = point.name + f'({round(point.original_position.x, 3)}, {round(point.original_position.y, 3)})'
        label_offset = int(radius_val)
        text_el = svg.text(label_text, x=str(x + label_offset), y=str(y - label_offset), fill=color)
        text_el.style['user-select'] = 'none'
        text_el.style['-webkit-user-select'] = 'none'
        text_el.style['-moz-user-select'] = 'none'
        text_el.style['-ms-user-select'] = 'none'
        text_el.setAttribute('font-size', f'{int(font_size_val)}px')
        document["math-svg"] <= text_el

    # ----------------------- Segment -----------------------
    def register_segment(self, segment_cls):
        self.register(segment_cls, self._render_segment)

    def _render_segment(self, segment, coordinate_mapper):
        # Visual params: style override -> model color -> default
        color = self.style.get('segment_color', getattr(segment, 'color', default_color))

        x1, y1 = coordinate_mapper.math_to_screen(
            segment.point1.original_position.x, segment.point1.original_position.y)
        x2, y2 = coordinate_mapper.math_to_screen(
            segment.point2.original_position.x, segment.point2.original_position.y)

        line_el = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=color)
        document["math-svg"] <= line_el

    # ----------------------- Circle -----------------------
    def register_circle(self, circle_cls):
        self.register(circle_cls, self._render_circle)

    def _render_circle(self, circle, coordinate_mapper):
        color = self.style.get('circle_color', getattr(circle, 'color', default_color))
        cx, cy = coordinate_mapper.math_to_screen(
            circle.center.original_position.x, circle.center.original_position.y)
        r_screen = coordinate_mapper.scale_value(circle.radius)
        circle_el = svg.circle(cx=str(cx), cy=str(cy), r=str(r_screen), fill="none", stroke=color)
        document["math-svg"] <= circle_el

    # ----------------------- Ellipse -----------------------
    def register_ellipse(self, ellipse_cls):
        self.register(ellipse_cls, self._render_ellipse)

    def _render_ellipse(self, ellipse, coordinate_mapper):
        color = self.style.get('ellipse_color', getattr(ellipse, 'color', default_color))
        cx, cy = coordinate_mapper.math_to_screen(
            ellipse.center.original_position.x, ellipse.center.original_position.y)
        rx = coordinate_mapper.scale_value(ellipse.radius_x)
        ry = coordinate_mapper.scale_value(ellipse.radius_y)
        # Apply rotation around center if needed
        transform = None
        if getattr(ellipse, 'rotation_angle', 0) not in (0, None):
            transform = f"rotate({-ellipse.rotation_angle} {cx} {cy})"
        if transform:
            el = svg.ellipse(cx=str(cx), cy=str(cy), rx=str(rx), ry=str(ry), fill="none", stroke=color, transform=transform)
        else:
            el = svg.ellipse(cx=str(cx), cy=str(cy), rx=str(rx), ry=str(ry), fill="none", stroke=color)
        document["math-svg"] <= el

    # ----------------------- Vector -----------------------
    def register_vector(self, vector_cls):
        self.register(vector_cls, self._render_vector)

    def _render_vector(self, vector, coordinate_mapper):
        # Draw underlying segment
        seg = vector.segment
        seg_color = self.style.get('vector_color', getattr(vector, 'color', getattr(seg, 'color', default_color)))

        x1, y1 = coordinate_mapper.math_to_screen(
            seg.point1.original_position.x, seg.point1.original_position.y)
        x2, y2 = coordinate_mapper.math_to_screen(
            seg.point2.original_position.x, seg.point2.original_position.y)

        line_el = svg.line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), stroke=seg_color)
        document["math-svg"] <= line_el

        # Draw arrow tip at the end (near tip point)
        import math as _math
        dx = x2 - x1
        dy = y2 - y1
        angle = _math.atan2(dy, dx)
        side_length = self.style.get('vector_tip_size', default_point_size * 4)
        half_base = side_length / 2
        height = (_math.sqrt(side_length * side_length - half_base * half_base)
                  if side_length >= half_base else side_length)

        # Triangle points around (x2, y2)
        p1x, p1y = x2, y2
        p2x = x2 - height * _math.cos(angle) - half_base * _math.sin(angle)
        p2y = y2 - height * _math.sin(angle) + half_base * _math.cos(angle)
        p3x = x2 - height * _math.cos(angle) + half_base * _math.sin(angle)
        p3y = y2 - height * _math.sin(angle) - half_base * _math.cos(angle)

        points_str = f"{p1x},{p1y} {p2x},{p2y} {p3x},{p3y}"
        poly_el = svg.polygon(points=points_str, fill=seg_color, stroke=seg_color)
        document["math-svg"] <= poly_el

    # ----------------------- Angle -----------------------
    def register_angle(self, angle_cls):
        self.register(angle_cls, self._render_angle)

    def _render_angle(self, angle, coordinate_mapper):
        # Compute screen coordinates for vertex and arms
        vx, vy = coordinate_mapper.math_to_screen(angle.vertex_point.original_position.x, angle.vertex_point.original_position.y)
        p1x, p1y = coordinate_mapper.math_to_screen(angle.arm1_point.original_position.x, angle.arm1_point.original_position.y)
        p2x, p2y = coordinate_mapper.math_to_screen(angle.arm2_point.original_position.x, angle.arm2_point.original_position.y)

        # Use model's precise arc parameter computation for correct curvature and flags
        arc_params = angle._calculate_arc_parameters(vx, vy, p1x, p1y, p2x, p2y)
        if not arc_params:
            return

        color = getattr(angle, 'color', default_color)
        d = arc_params["path_d"]
        path_el = svg.path(d=d, stroke=color, **{'stroke-width': '1', 'fill': 'none', 'class': 'angle-arc'})
        document["math-svg"] <= path_el

        # Label placement mirroring model logic
        if getattr(angle, 'angle_degrees', None) is not None:
            import math as _math
            display_angle_for_text_rad = _math.radians(angle.angle_degrees)
            effective_text_mid_arc_delta_rad = display_angle_for_text_rad / 2.0
            if arc_params["final_sweep_flag"] == '0':  # CW
                effective_text_mid_arc_delta_rad = -effective_text_mid_arc_delta_rad
            text_angle_rad = arc_params["angle_v_p1_rad"] + effective_text_mid_arc_delta_rad
            text_r = arc_params["arc_radius_on_screen"] * DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR
            tx = vx + text_r * _math.cos(text_angle_rad)
            ty = vy + text_r * _math.sin(text_angle_rad)
            text = f"{angle.angle_degrees:.1f}°"
            text_el = svg.text(text, x=str(tx), y=str(ty), fill=color)
            text_el.setAttribute('font-size', str(int(point_label_font_size)))
            text_el.style['text-anchor'] = 'middle'
            text_el.style['dominant-baseline'] = 'middle'
            document["math-svg"] <= text_el

    # ----------------------- Function -----------------------
    def register_function(self, function_cls):
        self.register(function_cls, self._render_function)

    def _render_function(self, func, coordinate_mapper):
        # Build screen-space paths using validated generator to match original behavior
        try:
            screen_poly = func.build_screen_paths()
            screen_paths = screen_poly.paths if screen_poly else []
            if not screen_paths:
                return
            color = getattr(func, 'color', default_color)
            for sp in screen_paths:
                d = "M" + " L".join(f"{x},{y}" for x,y in sp)
                document["math-svg"] <= svg.path(d=d, stroke=color, fill="none")
            # Label at first point of first screen path
            from constants import point_label_font_size
            first = screen_paths[0]
            label_offset_x = (1 + len(func.name)) * point_label_font_size / 2
            label_x = first[0][0] - label_offset_x
            label_y = max(first[0][1], point_label_font_size)
            text_el = svg.text(func.name, x=str(label_x), y=str(label_y), fill=color)
            text_el.setAttribute('font-size', str(int(point_label_font_size)))
            document["math-svg"] <= text_el
        except Exception:
            return

    # ----------------------- Colored Areas: FunctionsBoundedColoredArea -----------------------
    def register_functions_bounded_colored_area(self, cls):
        self.register(cls, self._render_functions_bounded_colored_area)

    def _render_functions_bounded_colored_area(self, area, coordinate_mapper):
        try:
            left_bound, right_bound = area._get_bounds()
            num_points = area.num_sample_points
            dx = (right_bound - left_bound) / (num_points - 1)
            if dx <= 0:
                left_bound, right_bound = left_bound - 1, right_bound + 1
                dx = (right_bound - left_bound) / (num_points - 1)
            points = area._generate_path(area.func1, left_bound, right_bound, dx, num_points, reverse=False)
            reverse_points = area._generate_path(area.func2, left_bound, right_bound, dx, num_points, reverse=True)
            if not points or not reverse_points:
                return
            # Build closed path
            d = f"M {points[0][0]},{points[0][1]}" + "".join(f" L {x},{y}" for x,y in points[1:])
            d += "".join(f" L {x},{y}" for x,y in reverse_points)
            d += " Z"
            fill_color = getattr(area, 'color', 'lightblue')
            fill_opacity = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return

    # ----------------------- Colored Areas: FunctionSegmentBoundedColoredArea -----------------
    def register_function_segment_bounded_colored_area(self, cls):
        self.register(cls, self._render_function_segment_bounded_colored_area)

    def _render_function_segment_bounded_colored_area(self, area, coordinate_mapper):
        try:
            left_bound, right_bound = area._get_bounds()
            num_points = 100
            dx = (right_bound - left_bound) / (num_points - 1)
            fwd = area._generate_function_points(left_bound, right_bound, num_points, dx)
            rev = [(area.segment.point2.screen_x, area.segment.point2.screen_y),
                   (area.segment.point1.screen_x, area.segment.point1.screen_y)]
            if not fwd or not rev:
                return
            d = f"M {fwd[0][0]},{fwd[0][1]}" + "".join(f" L {x},{y}" for x,y in fwd[1:])
            d += "".join(f" L {x},{y}" for x,y in rev)
            d += " Z"
            fill_color = getattr(area, 'color', 'lightblue')
            fill_opacity = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return

    # ----------------------- Colored Areas: SegmentsBoundedColoredArea -----------------------
    def register_segments_bounded_colored_area(self, cls):
        self.register(cls, self._render_segments_bounded_colored_area)

    def _render_segments_bounded_colored_area(self, area, coordinate_mapper):
        try:
            if not area.segment2:
                p1 = (area.segment1.point1.screen_x, area.segment1.point1.screen_y)
                p2 = (area.segment1.point2.screen_x, area.segment1.point2.screen_y)
                origin_y = area.canvas.cartesian2axis.origin.y
                if area.segment1.point1.original_position.y > 0 and area.segment1.point2.original_position.y > 0:
                    rev = [(p2[0], origin_y), (p1[0], origin_y)]
                else:
                    rev = [(p2[0], origin_y), (p1[0], origin_y)]
                points = [p1, p2]
            else:
                x1_min = min(area.segment1.point1.screen_x, area.segment1.point2.screen_x)
                x1_max = max(area.segment1.point1.screen_x, area.segment1.point2.screen_x)
                x2_min = min(area.segment2.point1.screen_x, area.segment2.point2.screen_x)
                x2_max = max(area.segment2.point1.screen_x, area.segment2.point2.screen_x)
                overlap_min = max(x1_min, x2_min)
                overlap_max = min(x1_max, x2_max)
                if overlap_max <= overlap_min:
                    return
                def get_y_at_x(segment, x):
                    x1, y1 = segment.point1.screen_x, segment.point1.screen_y
                    x2, y2 = segment.point2.screen_x, segment.point2.screen_y
                    if x2 == x1:
                        return y1
                    t = (x - x1) / (x2 - x1)
                    return y1 + t * (y2 - y1)
                y1_start = get_y_at_x(area.segment1, overlap_min)
                y1_end = get_y_at_x(area.segment1, overlap_max)
                y2_start = get_y_at_x(area.segment2, overlap_min)
                y2_end = get_y_at_x(area.segment2, overlap_max)
                points = [(overlap_min, y1_start), (overlap_max, y1_end)]
                rev = [(overlap_max, y2_end), (overlap_min, y2_start)]
            d = f"M {points[0][0]},{points[0][1]}" + "".join(f" L {x},{y}" for x,y in points[1:])
            d += "".join(f" L {x},{y}" for x,y in rev)
            d += " Z"
            fill_color = getattr(area, 'color', 'lightblue')
            fill_opacity = str(getattr(area, 'opacity', 0.3))
            document["math-svg"] <= svg.path(d=d, stroke="none", fill=fill_color, **{"fill-opacity": fill_opacity})
        except Exception:
            return


