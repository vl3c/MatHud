"""
MatHud Quadrilateral Geometric Object

Represents a four-sided polygon composed of four connected segments. Validates
that segments form a simple quadrilateral and caches classification flags
indicating whether the shape is a rectangle, square, rhombus, or irregular.

Dependencies:
    - drawables.polygon: Base polygon behaviors
    - utils.geometry_utils: Quadrilateral classification
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Set, cast

from constants import default_color
from drawables.point import Point
from drawables.polygon import Polygon
from drawables.segment import Segment
from utils.geometry_utils import GeometryUtils


class Quadrilateral(Polygon):
    """Represents a four-sided polygon with cached type metadata."""

    def __init__(
        self,
        segment1: Segment,
        segment2: Segment,
        segment3: Segment,
        segment4: Segment,
        *,
        color: str = default_color,
    ) -> None:
        segments = [segment1, segment2, segment3, segment4]
        if not GeometryUtils.segments_form_polygon(segments):
            raise ValueError("Segments do not form a closed quadrilateral")

        ordered_points = GeometryUtils.order_segments_into_loop(segments)
        if ordered_points is None or len(ordered_points) != 4:
            raise ValueError("Unable to determine quadrilateral vertex order")

        area_accumulator = 0.0
        for index, point in enumerate(ordered_points):
            next_point = ordered_points[(index + 1) % 4]
            area_accumulator += float(point.x) * float(next_point.y) - float(next_point.x) * float(point.y)
        if abs(area_accumulator) <= 1e-9:
            raise ValueError("Quadrilateral area must be non-zero")

        name = "".join(point.name for point in ordered_points)

        self.segment1 = segment1
        self.segment2 = segment2
        self.segment3 = segment3
        self.segment4 = segment4
        self._segments: List[Segment] = list(segments)
        self._points: List[Point] = list(ordered_points)
        self._set_type_flags(GeometryUtils.quadrilateral_type_flags(self._points))

        super().__init__(name=name, color=color)

    def get_class_name(self) -> str:
        return "Quadrilateral"

    def get_vertices(self) -> Set[Point]:
        return set(self._points)

    def get_segments(self) -> List[Segment]:
        return list(self._segments)

    def get_type_flags(self) -> Dict[str, bool]:
        return super().get_type_flags()

    def is_square(self) -> bool:
        return self.get_type_flags()["square"]

    def is_rectangle(self) -> bool:
        return self.get_type_flags()["rectangle"]

    def is_rhombus(self) -> bool:
        return self.get_type_flags()["rhombus"]

    def is_irregular(self) -> bool:
        return self.get_type_flags()["irregular"]

    def get_state(self) -> Dict[str, Any]:
        args = {f"p{index + 1}": point.name for index, point in enumerate(self._points)}
        return {
            "name": self.name,
            "args": args,
            "types": self.get_type_flags(),
        }

    def update_color(self, color: str) -> None:
        sanitized = str(color)
        self.color = sanitized
        for segment in self._segments:
            if hasattr(segment, "update_color") and callable(getattr(segment, "update_color")):
                segment.update_color(sanitized)
            else:
                segment.color = sanitized

    def __deepcopy__(self, memo: Dict[int, Any]) -> Quadrilateral:
        if id(self) in memo:
            return cast(Quadrilateral, memo[id(self)])

        new_segments = [deepcopy(segment, memo) for segment in self._segments]
        new_quad = Quadrilateral(
            new_segments[0],
            new_segments[1],
            new_segments[2],
            new_segments[3],
            color=self.color,
        )
        memo[id(self)] = new_quad
        return new_quad

