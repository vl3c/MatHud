"""
SVG renderer for MatHud using Brython's browser.svg.

Registry-based dispatch: the renderer maintains a mapping from model classes
to handler methods. This keeps models strictly math-space and renderer-agnostic.

Initially only provides clear() and a minimal registry without handlers. Shapes
will be added incrementally (Point first), ensuring non-breaking integration.
"""

from browser import document, svg


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


