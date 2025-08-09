"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch: the renderer maintains a mapping from model classes
to handler methods. This keeps models strictly math-space and renderer-agnostic.

Initially only provides clear() and a minimal registry without handlers. Shapes
will be added incrementally (Point first), ensuring non-breaking integration.
"""

from browser import document, svg
from constants import default_color, default_point_size, point_label_font_size


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


