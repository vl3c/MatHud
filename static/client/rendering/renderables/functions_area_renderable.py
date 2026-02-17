"""
Renderable for FunctionsBoundedColoredArea producing a screen-space ClosedArea.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from rendering.primitives import ClosedArea


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
                if y is None:
                    return None
                if not isinstance(y, (int, float)):
                    return None
                if isinstance(y, float) and (y != y or abs(y) == float("inf")):
                    return None
                return y
            except Exception:
                return None
        return None

    def _get_bounds(self) -> Tuple[float, float]:
        try:
            left: float
            right: float
            left, right = self.area._get_bounds()
        except Exception:
            left, right = -10, 10
        for f in (self.area.func1, self.area.func2):
            if (
                hasattr(f, "left_bound")
                and hasattr(f, "right_bound")
                and f.left_bound is not None
                and f.right_bound is not None
            ):
                left = max(left, f.left_bound)
                right = min(right, f.right_bound)
        if getattr(self.area, "left_bound", None) is not None:
            left = max(left, self.area.left_bound)
        if getattr(self.area, "right_bound", None) is not None:
            right = min(right, self.area.right_bound)
        try:
            vis_left: float = self.mapper.get_visible_left_bound()
            vis_right: float = self.mapper.get_visible_right_bound()
            left = max(left, vis_left)
            right = min(right, vis_right)
        except Exception:
            pass
        if left >= right:
            c: float = (left + right) / 2.0
            left, right = c - 0.1, c + 0.1
        return left, right

    def _generate_pair_paths_screen(
        self, f1: Any, f2: Any, left: float, right: float, num_points: int
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        if num_points < 2:
            num_points = 2
        dx: float = (right - left) / (num_points - 1) if num_points > 1 else 1.0
        pairs: List[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]] = []
        for i in range(num_points):
            x_m: float = left + i * dx
            y1: Optional[float] = self._eval_y_math(f1, x_m)
            y2: Optional[float] = self._eval_y_math(f2, x_m)
            if y1 is None or y2 is None:
                pairs.append((None, None))
                continue
            s1: Tuple[float, float] = self.mapper.math_to_screen(x_m, y1)
            s2: Tuple[float, float] = self.mapper.math_to_screen(x_m, y2)
            pairs.append((s1, s2))
        if not pairs:
            return [], []

        def split_valid(seq: List[Optional[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
            chunks: List[List[Tuple[float, float]]] = []
            cur: List[Tuple[float, float]] = []
            for p in seq:
                if p is None:
                    if cur:
                        chunks.append(cur)
                        cur = []
                else:
                    cur.append(p)
            if cur:
                chunks.append(cur)
            return chunks

        f_seq: List[Optional[Tuple[float, float]]] = [p[0] if p[0] is not None else None for p in pairs]
        g_seq: List[Optional[Tuple[float, float]]] = [p[1] if p[1] is not None else None for p in pairs]
        f_chunks: List[List[Tuple[float, float]]] = split_valid(f_seq)
        g_chunks: List[List[Tuple[float, float]]] = split_valid(g_seq)
        if not f_chunks or not g_chunks:
            return [], []
        idx: int = max(range(min(len(f_chunks), len(g_chunks))), key=lambda i: min(len(f_chunks[i]), len(g_chunks[i])))
        forward: List[Tuple[float, float]] = f_chunks[idx]
        reverse: List[Tuple[float, float]] = list(reversed(g_chunks[idx]))
        return forward, reverse

    def build_screen_area(self, num_points: Optional[int] = None) -> Optional[ClosedArea]:
        left: float
        right: float
        left, right = self._get_bounds()
        n: int = num_points if num_points is not None else getattr(self.area, "num_sample_points", 100)
        fwd: List[Tuple[float, float]]
        rev: List[Tuple[float, float]]
        fwd, rev = self._generate_pair_paths_screen(self.area.func1, self.area.func2, left, right, n)
        if not fwd or not rev:
            return None
        return ClosedArea(
            fwd,
            rev,
            is_screen=True,
            color=getattr(self.area, "color", None),
            opacity=getattr(self.area, "opacity", None),
        )
