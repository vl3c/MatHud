"""
MatHud Elliptical Arc Path Element

Represents an interval of an ellipse defined by exact curve parameters.
This is a path element that can be combined with segments to form composite paths.

Key Features:
    - Stores exact curve definition (center, radii, rotation, angle interval)
    - Generates sampled points on demand for rendering
    - Supports clockwise and counter-clockwise traversal
    - Handles rotated ellipses
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

from .path_element import PathElement
from utils.math_utils import MathUtils


class EllipticalArc(PathElement):
    """Interval of an ellipse defined by exact curve parameters.

    The arc is defined by its center, radii, rotation, and angle interval.
    Points are generated only when sample() is called.

    Attributes:
        _center: Center point of the ellipse (x, y)
        _radius_x: Horizontal radius before rotation
        _radius_y: Vertical radius before rotation
        _rotation: Rotation angle in radians
        _start_angle: Starting parameter angle in radians
        _end_angle: Ending parameter angle in radians
        _clockwise: Direction of traversal
    """

    __slots__ = (
        "_center", "_radius_x", "_radius_y", "_rotation",
        "_start_angle", "_end_angle", "_clockwise"
    )

    def __init__(
        self,
        center: Tuple[float, float],
        radius_x: float,
        radius_y: float,
        start_angle: float,
        end_angle: float,
        *,
        rotation: float = 0.0,
        clockwise: bool = False,
    ) -> None:
        """Initialize an elliptical arc.

        Args:
            center: Center point of the ellipse (x, y)
            radius_x: Horizontal radius (must be positive)
            radius_y: Vertical radius (must be positive)
            start_angle: Starting parameter angle in radians
            end_angle: Ending parameter angle in radians
            rotation: Rotation of the ellipse in radians (default 0)
            clockwise: If True, arc goes clockwise from start to end
        """
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("Radii must be positive")
        self._center: Tuple[float, float] = (float(center[0]), float(center[1]))
        self._radius_x: float = float(radius_x)
        self._radius_y: float = float(radius_y)
        self._rotation: float = float(rotation)
        self._start_angle: float = float(start_angle)
        self._end_angle: float = float(end_angle)
        self._clockwise: bool = bool(clockwise)

    @classmethod
    def from_ellipse(cls, ellipse: Any) -> "EllipticalArc":
        """Create a full ellipse as an EllipticalArc.

        Args:
            ellipse: An Ellipse drawable with center, radius_x, radius_y,
                     and rotation_angle attributes.

        Returns:
            An EllipticalArc representing the full ellipse (0 to 2*pi).
        """
        center = (float(ellipse.center.x), float(ellipse.center.y))
        radius_x = float(ellipse.radius_x)
        radius_y = float(ellipse.radius_y)
        rotation = math.radians(float(getattr(ellipse, "rotation_angle", 0.0)))
        return cls(
            center, radius_x, radius_y, 0.0, 2 * math.pi,
            rotation=rotation, clockwise=False
        )

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center of the ellipse."""
        return self._center

    @property
    def radius_x(self) -> float:
        """Return the horizontal radius."""
        return self._radius_x

    @property
    def radius_y(self) -> float:
        """Return the vertical radius."""
        return self._radius_y

    @property
    def rotation(self) -> float:
        """Return the rotation angle in radians."""
        return self._rotation

    @property
    def start_angle(self) -> float:
        """Return the starting parameter angle in radians."""
        return self._start_angle

    @property
    def end_angle(self) -> float:
        """Return the ending parameter angle in radians."""
        return self._end_angle

    @property
    def clockwise(self) -> bool:
        """Return the traversal direction."""
        return self._clockwise

    def _point_at_angle(self, angle: float) -> Tuple[float, float]:
        """Calculate the point on the ellipse at a given parameter angle."""
        cos_r = math.cos(self._rotation)
        sin_r = math.sin(self._rotation)

        local_x = self._radius_x * math.cos(angle)
        local_y = self._radius_y * math.sin(angle)

        world_x = local_x * cos_r - local_y * sin_r + self._center[0]
        world_y = local_x * sin_r + local_y * cos_r + self._center[1]

        return (world_x, world_y)

    def start_point(self) -> Tuple[float, float]:
        """Return the starting point of this arc."""
        return self._point_at_angle(self._start_angle)

    def end_point(self) -> Tuple[float, float]:
        """Return the ending point of this arc."""
        return self._point_at_angle(self._end_angle)

    def sample(self, resolution: int = 32) -> List[Tuple[float, float]]:
        """Generate sampled points along this arc.

        Args:
            resolution: Number of sample points to generate.

        Returns:
            List of (x, y) coordinate tuples along the arc.
        """
        return MathUtils.sample_ellipse_arc(
            self._center[0],
            self._center[1],
            self._radius_x,
            self._radius_y,
            self._start_angle,
            self._end_angle,
            rotation_degrees=math.degrees(self._rotation),
            num_samples=max(2, resolution),
            clockwise=self._clockwise,
        )

    def reversed(self) -> EllipticalArc:
        """Return a new arc traversing the same path in reverse direction."""
        return EllipticalArc(
            self._center,
            self._radius_x,
            self._radius_y,
            self._end_angle,
            self._start_angle,
            rotation=self._rotation,
            clockwise=not self._clockwise,
        )

    def length(self) -> float:
        """Return an approximation of the arc length.

        Ellipse arc length has no closed-form solution, so this uses
        the Ramanujan approximation for a full ellipse scaled by the arc span.
        """
        angle_span = self._arc_angle_span()
        full_circumference = self._approximate_circumference()
        return full_circumference * (angle_span / (2 * math.pi))

    def _arc_angle_span(self) -> float:
        """Calculate the angular span of the arc in radians."""
        if self._clockwise:
            span = self._start_angle - self._end_angle
        else:
            span = self._end_angle - self._start_angle

        while span < 0:
            span += 2 * math.pi
        while span > 2 * math.pi:
            span -= 2 * math.pi

        if span == 0 and self._start_angle != self._end_angle:
            span = 2 * math.pi

        return span

    def _approximate_circumference(self) -> float:
        """Ramanujan approximation for ellipse circumference."""
        a = self._radius_x
        b = self._radius_y
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EllipticalArc):
            return NotImplemented
        return (
            self._center == other._center
            and self._radius_x == other._radius_x
            and self._radius_y == other._radius_y
            and self._rotation == other._rotation
            and self._start_angle == other._start_angle
            and self._end_angle == other._end_angle
            and self._clockwise == other._clockwise
        )

    def __hash__(self) -> int:
        return hash((
            self._center, self._radius_x, self._radius_y,
            self._rotation, self._start_angle, self._end_angle, self._clockwise
        ))

    def __repr__(self) -> str:
        direction = "CW" if self._clockwise else "CCW"
        return (
            f"EllipticalArc(center={self._center}, "
            f"rx={self._radius_x}, ry={self._radius_y}, "
            f"rot={math.degrees(self._rotation):.1f}deg, "
            f"{self._start_angle:.3f} to {self._end_angle:.3f} {direction})"
        )

