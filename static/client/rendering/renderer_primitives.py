from __future__ import annotations

import math
from typing import Any, Dict, Optional


class StrokeStyle:
    __slots__ = ("color", "width", "line_join", "line_cap")

    def __init__(self, color, width, line_join=None, line_cap=None, **kwargs):
        self.color = str(color)
        self.width = float(width)
        self.line_join = line_join
        self.line_cap = line_cap


class FillStyle:
    __slots__ = ("color", "opacity")

    def __init__(self, color, opacity=None, **kwargs):
        self.color = str(color)
        self.opacity = None if opacity is None else float(opacity)


class FontStyle:
    __slots__ = ("family", "size", "weight")

    def __init__(self, family, size, weight=None):
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

    def __init__(self, horizontal="left", vertical="alphabetic"):
        self.horizontal = horizontal
        self.vertical = vertical


class RendererPrimitives:
    """Backend-specific primitive surface consumed by shared helpers."""

    def stroke_line(self, start, end, stroke, *, include_width=True):
        raise NotImplementedError

    def stroke_polyline(self, points, stroke):
        raise NotImplementedError

    def stroke_circle(self, center, radius, stroke):
        raise NotImplementedError

    def fill_circle(self, center, radius, fill, stroke=None, *, screen_space=False):
        raise NotImplementedError

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        raise NotImplementedError

    def fill_polygon(
        self,
        points,
        fill,
        stroke=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def fill_joined_area(self, forward, reverse, fill):
        raise NotImplementedError

    def stroke_arc(
        self,
        center,
        radius,
        start_angle_rad,
        end_angle_rad,
        sweep_clockwise,
        stroke,
        css_class=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def draw_text(
        self,
        text,
        position,
        font,
        color,
        alignment,
        style_overrides=None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def clear_surface(self):
        raise NotImplementedError

    def resize_surface(self, width, height):
        raise NotImplementedError

    def begin_frame(self):
        """Hook invoked at the start of a full canvas render cycle."""
        return None

    def end_frame(self):
        """Hook invoked at the end of a full canvas render cycle."""
        return None

    def begin_batch(self, plan=None):
        """Hook invoked before executing a batched render plan."""
        return None

    def end_batch(self, plan=None):
        """Hook invoked after executing a batched render plan."""
        return None

    def execute_optimized(self, command):
        """Fallback execution path for optimized commands."""
        handler = getattr(self, getattr(command, "op", ""), None)
        if callable(handler):
            handler(*getattr(command, "args", ()), **getattr(command, "kwargs", {}))

