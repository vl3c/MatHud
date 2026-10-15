"""
Renderable for FunctionsBoundedColoredArea producing screen-space ClosedAreas.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from rendering.primitives import ClosedArea

ScreenPoint = Tuple[float, float]


class FunctionsBoundedAreaRenderable:
    def __init__(self, area_model: Any, coordinate_mapper: Any) -> None:
        self.area: Any = area_model
        self.mapper: Any = coordinate_mapper

    def _is_function_like(self, f: Any) -> bool:
        return hasattr(f, "function")

    def _eval_y_math(self, f: Any, x_math: float) -> Optional[float]:
        if f is None:
            return 0.0
        if isinstance(f, (int, float)):
            return float(f)
        if self._is_function_like(f):
            try:
                y: Any = f.function(x_math)
            except Exception:
                return None
            if not isinstance(y, (int, float)) or not math.isfinite(y):
                return None
            return float(y)
        return None

    def _get_bounds(self) -> Tuple[float, float]:
        """The x-range to shade: the model's bounds clipped to the visible range.

        The model resolves the function and user bounds (None on a side that has none);
        an open side extends to the edge of the view. The range is empty (left >= right)
        when none of it is visible.
        """
        try:
            left_opt, right_opt = self.area._get_bounds()
        except Exception:
            left_opt, right_opt = None, None
        try:
            vis_left: float = self.mapper.get_visible_left_bound()
            vis_right: float = self.mapper.get_visible_right_bound()
        except Exception:
            vis_left, vis_right = -10.0, 10.0
        left = vis_left if left_opt is None else max(left_opt, vis_left)
        right = vis_right if right_opt is None else min(right_opt, vis_right)
        return left, right

    def _generate_pair_paths_screen(
        self, f1: Any, f2: Any, left: float, right: float, num_points: int
    ) -> List[Tuple[List[ScreenPoint], List[ScreenPoint]]]:
        """One (forward, reverse) pair per run of samples where both bounding curves are defined."""
        if num_points < 2:
            num_points = 2
        dx: float = (right - left) / (num_points - 1)
        chunks: List[Tuple[List[ScreenPoint], List[ScreenPoint]]] = []
        forward: List[ScreenPoint] = []
        reverse: List[ScreenPoint] = []
        for i in range(num_points):
            x_m: float = right if i == num_points - 1 else left + i * dx
            y1: Optional[float] = self._eval_y_math(f1, x_m)
            y2: Optional[float] = self._eval_y_math(f2, x_m)
            if y1 is None or y2 is None:
                self._close_chunk(chunks, forward, reverse)
                forward, reverse = [], []
                continue
            forward.append(self.mapper.math_to_screen(x_m, y1))
            reverse.append(self.mapper.math_to_screen(x_m, y2))
        self._close_chunk(chunks, forward, reverse)
        return chunks

    @staticmethod
    def _close_chunk(
        chunks: List[Tuple[List[ScreenPoint], List[ScreenPoint]]],
        forward: List[ScreenPoint],
        reverse: List[ScreenPoint],
    ) -> None:
        """Keep a run with at least two samples (a single sample encloses nothing)."""
        if len(forward) >= 2:
            chunks.append((forward, list(reversed(reverse))))

    def build_screen_areas(self, num_points: Optional[int] = None) -> List[ClosedArea]:
        """Screen-space areas, one for each part of the range where both curves are defined."""
        left, right = self._get_bounds()
        if not left < right:
            return []
        n: int = num_points if num_points is not None else getattr(self.area, "num_sample_points", 100)
        return [
            ClosedArea(
                fwd,
                rev,
                is_screen=True,
                color=getattr(self.area, "color", None),
                opacity=getattr(self.area, "opacity", None),
            )
            for fwd, rev in self._generate_pair_paths_screen(self.area.func1, self.area.func2, left, right, n)
        ]

    def build_screen_area(self, num_points: Optional[int] = None) -> Optional[ClosedArea]:
        """The first area of ``build_screen_areas`` (None when nothing is shaded)."""
        areas = self.build_screen_areas(num_points)
        return areas[0] if areas else None
