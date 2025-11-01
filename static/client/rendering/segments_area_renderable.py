"""
Renderable for SegmentsBoundedColoredArea producing a screen-space ClosedArea.

This keeps math models pure and provides renderer-agnostic geometry.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

from rendering.primitives import ClosedArea


class SegmentsBoundedAreaRenderable:
    def __init__(self, area_model: Any, coordinate_mapper: Any) -> None:
        self.area: Any = area_model
        self.mapper: Any = coordinate_mapper

    def _screen_xy(self, point: Any) -> Tuple[Optional[float], Optional[float]]:
        # Convert math coordinates to screen using the mapper exclusively
        if point is None or not hasattr(point, 'x') or not hasattr(point, 'y'):
            return None, None
        result: Any = self.mapper.math_to_screen(point.x, point.y)
        return cast(Tuple[Optional[float], Optional[float]], result)

    def _get_y_at_x_screen(self, segment: Any, x: float) -> Optional[float]:
        x1: Optional[float]
        y1: Optional[float]
        x1, y1 = self._screen_xy(segment.point1)
        x2: Optional[float]
        y2: Optional[float]
        x2, y2 = self._screen_xy(segment.point2)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return None
        if x2 == x1:
            return y1
        t: float = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def build_screen_area(self) -> Optional[ClosedArea]:
        if not getattr(self.area, 'segment2', None):
            p1: Tuple[Optional[float], Optional[float]] = self._screen_xy(self.area.segment1.point1)
            p2: Tuple[Optional[float], Optional[float]] = self._screen_xy(self.area.segment1.point2)
            if None in p1 or None in p2:
                return None
            # Use mapper to compute x-axis (y=0) at the same screen x positions
            assert p1[0] is not None and p1[1] is not None and p2[0] is not None and p2[1] is not None
            xaxis_screen_at_p2: float = self.mapper.math_to_screen(self.mapper.screen_to_math(p2[0], p2[1])[0], 0)[1]
            xaxis_screen_at_p1: float = self.mapper.math_to_screen(self.mapper.screen_to_math(p1[0], p1[1])[0], 0)[1]
            reverse_points: List[Tuple[float, float]] = [(p2[0], xaxis_screen_at_p2), (p1[0], xaxis_screen_at_p1)]
            return ClosedArea([(p1[0], p1[1]), (p2[0], p2[1])], reverse_points, is_screen=True)

        x11: Optional[float]
        x12: Optional[float]
        x11, _ = self._screen_xy(self.area.segment1.point1)
        x12, _ = self._screen_xy(self.area.segment1.point2)
        x21: Optional[float]
        x22: Optional[float]
        x21, _ = self._screen_xy(self.area.segment2.point1)
        x22, _ = self._screen_xy(self.area.segment2.point2)
        if None in (x11, x12, x21, x22):
            return None
        assert x11 is not None and x12 is not None and x21 is not None and x22 is not None
        x1_min: float = min(x11, x12)
        x1_max: float = max(x11, x12)
        x2_min: float = min(x21, x22)
        x2_max: float = max(x21, x22)
        overlap_min: float = max(x1_min, x2_min)
        overlap_max: float = min(x1_max, x2_max)
        if overlap_max <= overlap_min:
            return None
        y1_start: Optional[float] = self._get_y_at_x_screen(self.area.segment1, overlap_min)
        y1_end: Optional[float] = self._get_y_at_x_screen(self.area.segment1, overlap_max)
        y2_start: Optional[float] = self._get_y_at_x_screen(self.area.segment2, overlap_min)
        y2_end: Optional[float] = self._get_y_at_x_screen(self.area.segment2, overlap_max)
        if None in (y1_start, y1_end, y2_start, y2_end):
            return None
        assert y1_start is not None and y1_end is not None and y2_start is not None and y2_end is not None
        points: List[Tuple[float, float]] = [(overlap_min, y1_start), (overlap_max, y1_end)]
        reverse_points_seg: List[Tuple[float, float]] = [(overlap_max, y2_end), (overlap_min, y2_start)]
        return ClosedArea(points, reverse_points_seg, is_screen=True)


