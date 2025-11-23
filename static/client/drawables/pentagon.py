"""
MatHud Pentagon Geometric Object

Represents a five-sided polygon composed of five connected segments. Ensures the
segments form a closed loop and exposes mutually exclusive regular/irregular
type flags.

Dependencies:
    - drawables.polygon: Base polygon behaviors
    - utils.geometry_utils: Polygon classification helpers
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Set, cast

from constants import default_color
from drawables.point import Point
from drawables.polygon import Polygon
from drawables.segment import Segment
from utils.geometry_utils import GeometryUtils


class Pentagon(Polygon):
    """Represents a five-sided polygon with regular/irregular metadata."""

    def __init__(self, segments: List[Segment], *, color: str = default_color) -> None:
        if len(segments) != 5:
            raise ValueError("Pentagon requires exactly five segments")
        if not GeometryUtils.segments_form_polygon(segments):
            raise ValueError("Segments do not form a closed pentagon")

        ordered_points = GeometryUtils.order_segments_into_loop(segments)
        if ordered_points is None or len(ordered_points) != 5:
            raise ValueError("Unable to determine pentagon vertex order")

        name = "".join(point.name for point in ordered_points)

        self._segments: List[Segment] = list(segments)
        self._points: List[Point] = list(ordered_points)
        self._set_type_flags(GeometryUtils.polygon_flags(self._points))

        super().__init__(name=name, color=color, is_renderable=False)

    def get_class_name(self) -> str:
        return "Pentagon"

    def get_vertices(self) -> Set[Point]:
        return set(self._points)

    def get_type_flags(self) -> Dict[str, bool]:
        return super().get_type_flags()

    def is_regular(self) -> bool:
        return self.get_type_flags()["regular"]

    def is_irregular(self) -> bool:
        return self.get_type_flags()["irregular"]

    def get_state(self) -> Dict[str, Any]:
        args = {f"p{index + 1}": point.name for index, point in enumerate(self._points)}
        return {
            "name": self.name,
            "args": args,
            "types": self.get_type_flags(),
        }

    def get_segments(self) -> List[Segment]:
        return list(self._segments)

    def update_color(self, color: str) -> None:
        sanitized = str(color)
        self.color = sanitized
        for segment in self._segments:
            if hasattr(segment, "update_color") and callable(getattr(segment, "update_color")):
                segment.update_color(sanitized)
            else:
                segment.color = sanitized

    def __deepcopy__(self, memo: Dict[int, Any]) -> Pentagon:
        if id(self) in memo:
            return cast(Pentagon, memo[id(self)])
        new_segments = [deepcopy(segment, memo) for segment in self._segments]
        new_pentagon = Pentagon(new_segments, color=self.color)
        memo[id(self)] = new_pentagon
        return new_pentagon

