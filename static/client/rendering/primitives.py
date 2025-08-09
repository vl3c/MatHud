"""
Renderable primitives and intermediate representations for MatHud.

These are lightweight containers used to pass render-ready geometry between
math models and renderers, without embedding rendering concerns into models.
"""

class MathPolyline:
    """A polyline defined in math-space coordinates.

    Attributes:
        paths: list of paths, where each path is a list of (x, y) tuples
    """
    def __init__(self, paths):
        self.paths = paths or []


class ScreenPolyline:
    """A polyline defined in screen-space coordinates (pixels)."""
    def __init__(self, paths):
        self.paths = paths or []


class ClosedArea:
    """A closed area defined by a forward path and a reverse path.

    The paths can be in math-space or screen-space, depending on context.
    """
    def __init__(self, forward_points, reverse_points, is_screen=False):
        self.forward_points = forward_points or []
        self.reverse_points = reverse_points or []
        self.is_screen = is_screen


class Label:
    """A text label anchored at a math-space coordinate."""
    def __init__(self, text, x, y):
        self.text = text
        self.x = x
        self.y = y


