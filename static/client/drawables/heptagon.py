"""
MatHud Heptagon Geometric Object

Represents a seven-sided polygon composed of seven connected segments. Validates the
loop and tags the shape as regular or irregular.

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


class Heptagon(Polygon):
    """Represents a seven-sided polygon with regular/irregular metadata."""

    def __init__(self, segments: List[Segment], *, color: str = default_color) -> None:
        if len(segments) != 7:
            raise ValueError("Heptagon requires exactly seven segments")
        if not GeometryUtils.segments_form_polygon(segments):
            raise ValueError("Segments do not form a closed heptagon")

        ordered_points = GeometryUtils.order_segments_into_loop(segments)
        if ordered_points is None or len(ordered_points) != 7:
            raise ValueError("Unable to determine heptagon vertex order")

        name = "".join(point.name for point in ordered_points)

        self._segments: List[Segment] = list(segments)
        self._points: List[Point] = list(ordered_points)
        self._set_type_flags(GeometryUtils.polygon_flags(self._points))
        self._set_base_type_labels(["heptagon"])

        super().__init__(name=name, color=color, is_renderable=False)

    def get_class_name(self) -> str:
        return "Heptagon"

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
            "types": self.get_type_names(),
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

    def __deepcopy__(self, memo: Dict[int, Any]) -> Heptagon:
        if id(self) in memo:
            return cast(Heptagon, memo[id(self)])
        new_segments = [deepcopy(segment, memo) for segment in self._segments]
        new_heptagon = Heptagon(new_segments, color=self.color)
        memo[id(self)] = new_heptagon
        return new_heptagon

