"""
MatHud Circular Arc Path Element

Represents an interval of a circle defined by exact curve parameters.
This is a path element that can be combined with segments to form composite paths.

Key Features:
    - Stores exact curve definition (center, radius, angle interval)
    - Generates sampled points on demand for rendering
    - Supports clockwise and counter-clockwise traversal
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

from .path_element import PathElement
from utils.math_utils import MathUtils


class CircularArc(PathElement):
    """Interval of a circle defined by exact curve parameters.
    
    The arc is defined by its center, radius, and angle interval.
    Points are generated only when sample() is called.
    
    Attributes:
        _center: Center point of the circle (x, y)
        _radius: Radius of the circle
        _start_angle: Starting angle in radians
        _end_angle: Ending angle in radians
        _clockwise: Direction of traversal
    """
    
    __slots__ = ("_center", "_radius", "_start_angle", "_end_angle", "_clockwise")
    
    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        *,
        clockwise: bool = False,
    ) -> None:
        """Initialize a circular arc.
        
        Args:
            center: Center point of the circle (x, y)
            radius: Radius of the circle (must be positive)
            start_angle: Starting angle in radians
            end_angle: Ending angle in radians
            clockwise: If True, arc goes clockwise from start to end
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self._center: Tuple[float, float] = (float(center[0]), float(center[1]))
        self._radius: float = float(radius)
        self._start_angle: float = float(start_angle)
        self._end_angle: float = float(end_angle)
        self._clockwise: bool = bool(clockwise)
    
    @classmethod
    def from_circle_arc(cls, arc: Any) -> "CircularArc":
        """Create a CircularArc from a CircleArc drawable.
        
        Args:
            arc: A CircleArc drawable with center_x, center_y, radius,
                 point1, point2, and use_major_arc attributes.
        
        Returns:
            A new CircularArc path element.
        """
        center = (float(arc.center_x), float(arc.center_y))
        radius = float(arc.radius)
        
        start_angle = math.atan2(
            arc.point1.y - arc.center_y,
            arc.point1.x - arc.center_x
        )
        end_angle = math.atan2(
            arc.point2.y - arc.center_y,
            arc.point2.x - arc.center_x
        )
        
        ccw_span = end_angle - start_angle
        if ccw_span < 0:
            ccw_span += 2 * math.pi
        
        minor_is_ccw = ccw_span <= math.pi
        
        if arc.use_major_arc:
            clockwise = minor_is_ccw
        else:
            clockwise = not minor_is_ccw
        
        return cls(center, radius, start_angle, end_angle, clockwise=clockwise)
    
    @classmethod
    def from_circle(cls, circle: Any) -> "CircularArc":
        """Create a full circle as a CircularArc.
        
        Args:
            circle: A Circle drawable with center and radius attributes.
        
        Returns:
            A CircularArc representing the full circle (0 to 2*pi).
        """
        center = (float(circle.center.x), float(circle.center.y))
        radius = float(circle.radius)
        return cls(center, radius, 0.0, 2 * math.pi, clockwise=False)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return the center of the circle."""
        return self._center
    
    @property
    def radius(self) -> float:
        """Return the radius of the circle."""
        return self._radius
    
    @property
    def start_angle(self) -> float:
        """Return the starting angle in radians."""
        return self._start_angle
    
    @property
    def end_angle(self) -> float:
        """Return the ending angle in radians."""
        return self._end_angle
    
    @property
    def clockwise(self) -> bool:
        """Return the traversal direction."""
        return self._clockwise
    
    def start_point(self) -> Tuple[float, float]:
        """Return the starting point of this arc."""
        x = self._center[0] + self._radius * math.cos(self._start_angle)
        y = self._center[1] + self._radius * math.sin(self._start_angle)
        return (x, y)
    
    def end_point(self) -> Tuple[float, float]:
        """Return the ending point of this arc."""
        x = self._center[0] + self._radius * math.cos(self._end_angle)
        y = self._center[1] + self._radius * math.sin(self._end_angle)
        return (x, y)
    
    def sample(self, resolution: int = 32) -> List[Tuple[float, float]]:
        """Generate sampled points along this arc.
        
        Args:
            resolution: Number of sample points to generate.
        
        Returns:
            List of (x, y) coordinate tuples along the arc.
        """
        return MathUtils.sample_circle_arc(
            self._center[0],
            self._center[1],
            self._radius,
            self._start_angle,
            self._end_angle,
            num_samples=max(2, resolution),
            clockwise=self._clockwise,
        )
    
    def reversed(self) -> CircularArc:
        """Return a new arc traversing the same path in reverse direction."""
        return CircularArc(
            self._center,
            self._radius,
            self._end_angle,
            self._start_angle,
            clockwise=not self._clockwise,
        )
    
    def length(self) -> float:
        """Return the arc length."""
        angle_span = self._arc_angle_span()
        return self._radius * angle_span
    
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
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircularArc):
            return NotImplemented
        return (
            self._center == other._center
            and self._radius == other._radius
            and self._start_angle == other._start_angle
            and self._end_angle == other._end_angle
            and self._clockwise == other._clockwise
        )
    
    def __hash__(self) -> int:
        return hash((self._center, self._radius, self._start_angle, self._end_angle, self._clockwise))
    
    def __repr__(self) -> str:
        direction = "CW" if self._clockwise else "CCW"
        return f"CircularArc(center={self._center}, r={self._radius}, {self._start_angle:.3f} to {self._end_angle:.3f} {direction})"

