"""
Renderable primitives and intermediate representations for MatHud.

These are lightweight containers used to pass render-ready geometry between
math models and renderers, without embedding rendering concerns into models.
"""

from __future__ import annotations

from typing import List, Tuple


class MathPolyline:
    """A polyline defined in math-space coordinates.

    Attributes:
        paths: list of paths, where each path is a list of (x, y) tuples
    """
    def __init__(self, paths: List[List[Tuple[float, float]]]) -> None:
        self.paths: List[List[Tuple[float, float]]] = paths or []


class ScreenPolyline:
    """A polyline defined in screen-space coordinates (pixels)."""
    def __init__(self, paths: List[List[Tuple[float, float]]]) -> None:
        self.paths: List[List[Tuple[float, float]]] = paths or []


class ClosedArea:
    """A closed area defined by a forward path and a reverse path.
    
    The paths can be in math-space or screen-space, depending on context.
    """
    def __init__(
        self,
        forward_points: List[Tuple[float, float]],
        reverse_points: List[Tuple[float, float]],
        is_screen: bool = False,
        color: str | None = None,
        opacity: float | None = None,
    ) -> None:
        self.forward_points: List[Tuple[float, float]] = forward_points or []
        self.reverse_points: List[Tuple[float, float]] = reverse_points or []
        self.is_screen: bool = is_screen
        self.color: str | None = color
        self.opacity: float | None = opacity


class Label:
    """A text label anchored at a math-space coordinate."""
    def __init__(self, text: str, x: float, y: float) -> None:
        self.text: str = text
        self.x: float = x
        self.y: float = y


