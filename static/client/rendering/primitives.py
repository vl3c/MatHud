"""
Rendering primitives for MatHud.

Geometry containers and style objects used to pass render-ready data
between math models and renderers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
# Geometry containers
# ----------------------------------------------------------------------------


class MathPolyline:
    """A polyline defined in math-space coordinates."""

    def __init__(self, paths: List[List[Tuple[float, float]]]) -> None:
        self.paths: List[List[Tuple[float, float]]] = paths or []


class ScreenPolyline:
    """A polyline defined in screen-space coordinates (pixels)."""

    def __init__(self, paths: List[List[Tuple[float, float]]]) -> None:
        self.paths: List[List[Tuple[float, float]]] = paths or []


class ClosedArea:
    """A closed area defined by a forward path and a reverse path."""

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


# ----------------------------------------------------------------------------
# Style objects
# ----------------------------------------------------------------------------


class StrokeStyle:
    __slots__ = ("color", "width", "line_join", "line_cap")

    def __init__(self, color: str, width: float, line_join: Optional[str] = None, line_cap: Optional[str] = None, **kwargs: Any) -> None:
        self.color = str(color)
        self.width = float(width)
        self.line_join = line_join
        self.line_cap = line_cap


class FillStyle:
    __slots__ = ("color", "opacity")

    def __init__(self, color: str, opacity: Optional[float] = None, **kwargs: Any) -> None:
        self.color = str(color)
        self.opacity = None if opacity is None else float(opacity)


class FontStyle:
    __slots__ = ("family", "size", "weight")

    def __init__(self, family: str, size: Any, weight: Optional[str] = None) -> None:
        self.family = family
        try:
            size_float = float(size)
        except Exception:
            self.size = size
        else:
            if math.isfinite(size_float) and size_float.is_integer():
                self.size = int(size_float)
            else:
                self.size = size_float
        self.weight = weight


class TextAlignment:
    __slots__ = ("horizontal", "vertical")

    def __init__(self, horizontal: str = "left", vertical: str = "alphabetic") -> None:
        self.horizontal = horizontal
        self.vertical = vertical


# ----------------------------------------------------------------------------
# Renderer interface
# ----------------------------------------------------------------------------


class RendererPrimitives:
    """Backend-specific primitive surface consumed by shared helpers."""

    def stroke_line(self, start: Tuple[float, float], end: Tuple[float, float], stroke: StrokeStyle, *, include_width: bool = True) -> None:
        raise NotImplementedError

    def stroke_polyline(self, points: List[Tuple[float, float]], stroke: StrokeStyle) -> None:
        raise NotImplementedError

    def stroke_circle(self, center: Tuple[float, float], radius: float, stroke: StrokeStyle) -> None:
        raise NotImplementedError

    def fill_circle(self, center: Tuple[float, float], radius: float, fill: FillStyle, stroke: Optional[StrokeStyle] = None, *, screen_space: bool = False) -> None:
        raise NotImplementedError

    def stroke_ellipse(self, center: Tuple[float, float], radius_x: float, radius_y: float, rotation_rad: float, stroke: StrokeStyle) -> None:
        raise NotImplementedError

    def fill_polygon(
        self,
        points: List[Tuple[float, float]],
        fill: FillStyle,
        stroke: Optional[StrokeStyle] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def fill_joined_area(self, forward: List[Tuple[float, float]], reverse: List[Tuple[float, float]], fill: FillStyle) -> None:
        raise NotImplementedError

    def stroke_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        sweep_clockwise: bool,
        stroke: StrokeStyle,
        css_class: Optional[str] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def draw_text(
        self,
        text: str,
        position: Tuple[float, float],
        font: FontStyle,
        color: str,
        alignment: TextAlignment,
        style_overrides: Optional[Dict[str, Any]] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def clear_surface(self) -> None:
        raise NotImplementedError

    def resize_surface(self, width: float, height: float) -> None:
        raise NotImplementedError

    def begin_frame(self) -> None:
        """Hook invoked at the start of a full canvas render cycle."""
        return None

    def end_frame(self) -> None:
        """Hook invoked at the end of a full canvas render cycle."""
        return None

    def begin_batch(self, plan: Any = None) -> None:
        """Hook invoked before executing a batched render plan."""
        return None

    def end_batch(self, plan: Any = None) -> None:
        """Hook invoked after executing a batched render plan."""
        return None

    def execute_optimized(self, command: Any) -> None:
        """Fallback execution path for optimized commands."""
        handler = getattr(self, getattr(command, "op", ""), None)
        if callable(handler):
            handler(*getattr(command, "args", ()), **getattr(command, "kwargs", {}))
