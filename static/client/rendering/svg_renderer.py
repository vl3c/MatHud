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


