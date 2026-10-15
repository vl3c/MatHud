"""
Renderable for FunctionSegmentBoundedColoredArea producing a math-space ClosedArea.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from rendering.primitives import ClosedArea

MathPoint = Tuple[float, float]


class FunctionSegmentAreaRenderable:
    def __init__(self, area_model: Any, coordinate_mapper: Any) -> None:
        self.area: Any = area_model
        self.mapper: Any = coordinate_mapper

    def _get_bounds(self) -> Tuple[float, float]:
        """The model's x-range: the segment's, within the function's and the area's bounds."""
        left: float
        right: float
        left, right = self.area._get_bounds()
        return left, right

    def _eval_function(self, x_math: float) -> Optional[float]:
        func: Any = self.area.func
        if func is None:
            return 0.0
        if isinstance(func, (int, float)):
            return float(func)
        if hasattr(func, "function"):
            try:
                result: Any = func.function(x_math)
            except Exception:
                return None
            if not isinstance(result, (int, float)) or not math.isfinite(result):
                return None
            return float(result)
        return None

    def _generate_function_points_math(self, left_bound: float, right_bound: float, num_points: int) -> List[MathPoint]:
        if num_points < 2:
            num_points = 2
        dx: float = (right_bound - left_bound) / (num_points - 1)
        pts: List[MathPoint] = []
        for i in range(num_points):
            x_m: float = right_bound if i == num_points - 1 else left_bound + i * dx
            y_m: Optional[float] = self._eval_function(x_m)
            if y_m is None:
                continue
            pts.append((x_m, y_m))
        return pts

    def _segment_reverse_points_math(self, left_bound: float, right_bound: float) -> Optional[List[MathPoint]]:
        """The segment's points at right_bound and left_bound, in that order (descending x).

        The forward path runs left to right along the function, so the segment closes the
        area from right to left whichever way the segment itself was drawn.
        """
        p1: Any = self.area.segment.point1
        p2: Any = self.area.segment.point2
        if p1 is None or p2 is None:
            return None
        if not hasattr(p1, "x") or not hasattr(p1, "y"):
            return None
        if not hasattr(p2, "x") or not hasattr(p2, "y"):
            return None
        x1, y1, x2, y2 = float(p1.x), float(p1.y), float(p2.x), float(p2.y)
        if x1 == x2:
            return None

        def y_at(x: float) -> float:
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        return [(right_bound, y_at(right_bound)), (left_bound, y_at(left_bound))]

    def build_screen_area(self, num_points: int = 100) -> Optional[ClosedArea]:
        left_bound: float
        right_bound: float
        left_bound, right_bound = self._get_bounds()
        if not left_bound < right_bound:
            return None
        forward: List[MathPoint] = self._generate_function_points_math(left_bound, right_bound, num_points)
        reverse_points: Optional[List[MathPoint]] = self._segment_reverse_points_math(left_bound, right_bound)
        if not forward or not reverse_points:
            return None
        return ClosedArea(
            forward,
            reverse_points,
            is_screen=False,
            color=getattr(self.area, "color", None),
            opacity=getattr(self.area, "opacity", None),
        )
