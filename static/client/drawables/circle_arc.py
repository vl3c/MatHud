"""
MatHud Circle Arc Drawable

Represents an arc along a circle defined by two boundary points. The arc can be
tied to an existing Circle drawable or operate independently using its own
center and radius snapshot. Rendering helpers consume the stored geometry to
draw the correct circular segment above the underlying circle.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from constants import DEFAULT_CIRCLE_ARC_COLOR
from drawables.drawable import Drawable
from drawables.point import Point
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from drawables.circle import Circle


def arc_start_and_sweep(
    center: Tuple[float, float],
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    use_major_arc: bool,
) -> Tuple[float, float]:
    """Where an arc between two points of a circle starts, and which way and how far it turns.

    Returns the angle of ``start_point`` about ``center`` (radians, math coordinates) and
    the signed sweep to ``end_point``: positive counter-clockwise, along the shorter arc,
    or the longer one when ``use_major_arc``. Points in the same direction from the centre
    give a sweep of 0. Every consumer of an arc's direction (renderers, path elements,
    intersections) uses this, so they all draw and measure the same arc.
    """
    cx, cy = center
    start = math.atan2(start_point[1] - cy, start_point[0] - cx)
    end = math.atan2(end_point[1] - cy, end_point[0] - cx)
    counter_clockwise = (end - start) % (2 * math.pi)
    if counter_clockwise == 0.0:
        return start, 0.0
    minor_is_counter_clockwise = counter_clockwise <= math.pi
    if minor_is_counter_clockwise != bool(use_major_arc):
        return start, counter_clockwise
    return start, -(2 * math.pi - counter_clockwise)


class CircleArc(Drawable):
    """
    Represents a circular arc between two points that lie on the same circle.

    Attributes:
        point1 (Point): Starting point of the arc sweep.
        point2 (Point): Ending point of the arc sweep.
        center_x (float): Circle center x-coordinate in math space.
        center_y (float): Circle center y-coordinate in math space.
        radius (float): Circle radius in math space units.
        circle (Optional[Circle]): Optional reference to the parent circle.
        use_major_arc (bool): Whether to draw the longer (major) arc.
    """

    def __init__(
        self,
        point1: Point,
        point2: Point,
        *,
        center_x: float,
        center_y: float,
        radius: float,
        circle: Optional["Circle"] = None,
        use_major_arc: bool = False,
        color: Optional[str] = None,
        name: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Build the arc; strict=False skips the check that both endpoints lie on the circle."""
        if radius is None or float(radius) <= 0:
            raise ValueError("CircleArc requires a positive radius.")
        if point1 is point2:
            raise ValueError("CircleArc requires two distinct boundary points.")

        self.point1: Point = point1
        self.point2: Point = point2
        self.center_x: float = float(center_x)
        self.center_y: float = float(center_y)
        self.radius: float = float(radius)
        self.circle: Optional["Circle"] = circle
        self.circle_name: Optional[str] = getattr(circle, "name", None)
        self.use_major_arc: bool = bool(use_major_arc)

        self._validate_points_on_circle(strict=strict)

        chosen_color = str(color) if color is not None else DEFAULT_CIRCLE_ARC_COLOR
        computed_name = (
            name if name else self._build_default_name(getattr(point1, "name", "P1"), getattr(point2, "name", "P2"))
        )

        super().__init__(name=computed_name, color=chosen_color)
        self._refresh_angles()

    def _build_default_name(self, start_name: str, end_name: str) -> str:
        sanitized_start = start_name if start_name else "P1"
        sanitized_end = end_name if end_name else "P2"
        suffix = "_major" if self.use_major_arc else ""
        return f"arc_{sanitized_start}_{sanitized_end}{suffix}"

    def _refresh_angles(self) -> None:
        self._point1_angle: float = self._angle_for_point(self.point1)
        self._point2_angle: float = self._angle_for_point(self.point2)

    def _angle_for_point(self, point: Point) -> float:
        return math.atan2(point.y - self.center_y, point.x - self.center_x)

    def sync_with_circle(self) -> None:
        """Refresh stored center/radius information when a parent circle exists."""
        if not self.circle:
            return
        try:
            self.center_x = float(self.circle.center.x)
            self.center_y = float(self.circle.center.y)
            self.radius = float(self.circle.radius)
        except Exception:
            return
        self._validate_points_on_circle(strict=False)
        self._refresh_angles()

    def get_class_name(self) -> str:
        return "CircleArc"

    def start_and_sweep(self) -> Tuple[float, float]:
        """Angle of point1 about the centre and the signed sweep to point2 (see arc_start_and_sweep)."""
        return arc_start_and_sweep(
            (self.center_x, self.center_y),
            (self.point1.x, self.point1.y),
            (self.point2.x, self.point2.y),
            self.use_major_arc,
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "circle_arc",
            "args": {
                "point1_name": getattr(self.point1, "name", ""),
                "point2_name": getattr(self.point2, "name", ""),
                "center_x": self.center_x,
                "center_y": self.center_y,
                "radius": self.radius,
                "circle_name": self.circle_name,
                "use_major_arc": self.use_major_arc,
                "color": self.color,
            },
        }

    def update_color(self, color: str) -> None:
        self.color = str(color)

    def set_use_major_arc(self, use_major_arc: bool) -> None:
        self.use_major_arc = bool(use_major_arc)
        self._refresh_angles()

    def update_points(
        self,
        *,
        point1: Optional[Point] = None,
        point2: Optional[Point] = None,
    ) -> None:
        """Rebind the arc to different boundary points and recalculate sweep angles."""
        if point1 is not None:
            self.point1 = point1
        if point2 is not None:
            self.point2 = point2

        self._validate_points_on_circle()
        self._refresh_angles()

    def _validate_points_on_circle(self, *, strict: bool = True) -> bool:
        first_valid = MathUtils.point_on_circle(
            self.point1,
            center_x=self.center_x,
            center_y=self.center_y,
            radius=self.radius,
            strict=strict,
        )
        second_valid = MathUtils.point_on_circle(
            self.point2,
            center_x=self.center_x,
            center_y=self.center_y,
            radius=self.radius,
            strict=strict,
        )

        tolerance = max(MathUtils.EPSILON * max(1.0, abs(self.radius)), 1e-6)
        points_overlap = math.isclose(self.point1.x, self.point2.x, abs_tol=tolerance) and math.isclose(
            self.point1.y, self.point2.y, abs_tol=tolerance
        )
        if points_overlap:
            if strict:
                raise ValueError("CircleArc endpoints must not coincide.")
            return False

        return first_valid and second_valid

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CircleArc":
        if id(self) in memo:
            return memo[id(self)]

        new_point1 = deepcopy(self.point1, memo)
        new_point2 = deepcopy(self.point2, memo)
        new_circle = deepcopy(self.circle, memo) if self.circle else None

        # A copy keeps the arc as it is (undo archives copy every arc), so it skips the
        # on-circle check that a newly built arc must pass.
        new_arc = CircleArc(
            new_point1,
            new_point2,
            center_x=self.center_x,
            center_y=self.center_y,
            radius=self.radius,
            circle=new_circle,
            use_major_arc=self.use_major_arc,
            color=self.color,
            name=self.name,
            strict=False,
        )
        memo[id(self)] = new_arc
        return new_arc

    def reset(self) -> None:
        """Refresh cached angles when undo/redo restores previous state."""
        self.sync_with_circle()
        self._refresh_angles()
