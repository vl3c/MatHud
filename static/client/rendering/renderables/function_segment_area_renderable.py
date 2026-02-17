"""
Renderable for FunctionSegmentBoundedColoredArea producing a math-space ClosedArea.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

from rendering.primitives import ClosedArea


class FunctionSegmentAreaRenderable:
    def __init__(self, area_model: Any, coordinate_mapper: Any) -> None:
        self.area: Any = area_model
        self.mapper: Any = coordinate_mapper

    def _get_bounds(self) -> Tuple[float, float]:
        seg_left: float
        seg_right: float
        seg_left, seg_right = self.area._get_segment_bounds()
        func: Any = self.area.func
        if hasattr(func, "left_bound") and hasattr(func, "right_bound"):
            return max(seg_left, func.left_bound), min(seg_right, func.right_bound)
        return seg_left, seg_right

    def _eval_function(self, x_math: float) -> Optional[float]:
        func: Any = self.area.func
        if func is None:
            return 0.0
        if isinstance(func, (int, float)):
            return float(func)
        if hasattr(func, "function"):
            try:
                result: Any = func.function(x_math)
                return cast(Optional[float], result)
            except Exception:
                return None
        return None

    def _generate_function_points_math(
        self, left_bound: float, right_bound: float, num_points: int
    ) -> List[Tuple[float, float]]:
        if num_points < 2:
            num_points = 2
        dx: float = (right_bound - left_bound) / (num_points - 1) if num_points > 1 else 1.0
        pts: List[Tuple[float, float]] = []
        for i in range(num_points):
            x_m: float = left_bound + i * dx
            y_m: Optional[float] = self._eval_function(x_m)
            if y_m is None:
                continue
            pts.append((x_m, y_m))
        return pts

    def _segment_reverse_points_math(self) -> Optional[List[Tuple[float, float]]]:
        p1: Any = self.area.segment.point1
        p2: Any = self.area.segment.point2
        if p1 is None or p2 is None:
            return None
        if not hasattr(p1, "x") or not hasattr(p1, "y"):
            return None
        if not hasattr(p2, "x") or not hasattr(p2, "y"):
            return None
        return [(p2.x, p2.y), (p1.x, p1.y)]

    def build_screen_area(self, num_points: int = 100) -> Optional[ClosedArea]:
        left_bound: float
        right_bound: float
        left_bound, right_bound = self._get_bounds()
        forward: List[Tuple[float, float]] = self._generate_function_points_math(left_bound, right_bound, num_points)
        reverse_points: Optional[List[Tuple[float, float]]] = self._segment_reverse_points_math()
        if not forward or not reverse_points:
            return None
        return ClosedArea(
            forward,
            reverse_points,
            is_screen=False,
            color=getattr(self.area, "color", None),
            opacity=getattr(self.area, "opacity", None),
        )
