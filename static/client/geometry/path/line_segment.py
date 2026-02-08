"""
MatHud Line Segment Path Element

Represents a straight line segment between two points.
This is a path element that can be combined with arcs to form composite paths.

Key Features:
    - Exact endpoint storage
    - Trivial sampling (returns just the two endpoints)
    - Can wrap existing Segment drawable references
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

from .path_element import PathElement


class LineSegment(PathElement):
    """Straight line segment between two points.

    Stores exact endpoint coordinates. For rendering, sampling simply
    returns the start and end points since a line needs no intermediate samples.

    Attributes:
        _start: Starting point coordinates (x, y)
        _end: Ending point coordinates (x, y)
    """

    __slots__ = ("_start", "_end")

    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> None:
        """Initialize a line segment.

        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
        """
        self._start: Tuple[float, float] = (float(start[0]), float(start[1]))
        self._end: Tuple[float, float] = (float(end[0]), float(end[1]))

    @classmethod
    def from_segment(cls, segment: Any) -> LineSegment:
        """Create a LineSegment from an existing Segment drawable.

        Args:
            segment: A Segment object with point1 and point2 attributes.

        Returns:
            A new LineSegment with the same endpoints.
        """
        start = (float(segment.point1.x), float(segment.point1.y))
        end = (float(segment.point2.x), float(segment.point2.y))
        return cls(start, end)

    def start_point(self) -> Tuple[float, float]:
        """Return the starting point of this segment."""
        return self._start

    def end_point(self) -> Tuple[float, float]:
        """Return the ending point of this segment."""
        return self._end

    def sample(self, resolution: int = 32) -> List[Tuple[float, float]]:
        """Return the two endpoints (resolution is ignored for line segments).

        A straight line requires no intermediate sampling.
        """
        return [self._start, self._end]

    def reversed(self) -> LineSegment:
        """Return a new segment with start and end swapped."""
        return LineSegment(self._end, self._start)

    def length(self) -> float:
        """Return the length of this segment."""
        dx = self._end[0] - self._start[0]
        dy = self._end[1] - self._start[1]
        return math.hypot(dx, dy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineSegment):
            return NotImplemented
        return self._start == other._start and self._end == other._end

    def __hash__(self) -> int:
        return hash((self._start, self._end))

    def __repr__(self) -> str:
        return f"LineSegment({self._start}, {self._end})"

