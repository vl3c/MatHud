"""
MatHud Composite Path

A path composed of connected path elements (segments and/or arcs).
Can be open (chain) or closed (boundary for a region).

Key Features:
    - Validates connectivity between elements
    - Supports mixed element types (segments and arcs)
    - Generates combined sample points for rendering
    - Factory methods for common construction patterns
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .path_element import PathElement
from .line_segment import LineSegment


class CompositePath:
    """A path composed of connected path elements.

    Elements are stored in order and must connect end-to-start.
    A closed path has its last element's end connecting back to the first element's start.

    Attributes:
        _elements: Ordered list of path elements
        _tolerance: Maximum distance between connected points
    """

    __slots__ = ("_elements", "_tolerance")

    def __init__(
        self,
        elements: Optional[List[PathElement]] = None,
        *,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize a composite path.

        Args:
            elements: List of connected path elements (optional, can add later)
            tolerance: Maximum distance to consider points as connected

        Raises:
            ValueError: If elements are not connected
        """
        self._tolerance: float = float(tolerance)
        self._elements: List[PathElement] = []

        if elements:
            for element in elements:
                self.append(element)

    def append(self, element: PathElement) -> None:
        """Add an element to the end of the path.

        Args:
            element: Path element to add

        Raises:
            ValueError: If element does not connect to the previous element
        """
        if self._elements:
            last = self._elements[-1]
            if not last.connects_to(element, self._tolerance):
                last_end = last.end_point()
                elem_start = element.start_point()
                raise ValueError(
                    f"Element does not connect: previous end {last_end} "
                    f"does not match new start {elem_start}"
                )
        self._elements.append(element)

    def prepend(self, element: PathElement) -> None:
        """Add an element to the start of the path.

        Args:
            element: Path element to add

        Raises:
            ValueError: If element does not connect to the first element
        """
        if self._elements:
            first = self._elements[0]
            if not element.connects_to(first, self._tolerance):
                elem_end = element.end_point()
                first_start = first.start_point()
                raise ValueError(
                    f"Element does not connect: new end {elem_end} "
                    f"does not match first start {first_start}"
                )
        self._elements.insert(0, element)

    @property
    def elements(self) -> List[PathElement]:
        """Return a copy of the elements list."""
        return list(self._elements)

    def __len__(self) -> int:
        """Return the number of elements in the path."""
        return len(self._elements)

    def __iter__(self):
        """Iterate over path elements."""
        return iter(self._elements)

    def is_empty(self) -> bool:
        """Check if the path has no elements."""
        return len(self._elements) == 0

    def is_closed(self) -> bool:
        """Check if the path forms a closed loop.

        Returns:
            True if the last element's end connects to the first element's start.
        """
        if len(self._elements) < 1:
            return False

        first_start = self._elements[0].start_point()
        last_end = self._elements[-1].end_point()

        dx = last_end[0] - first_start[0]
        dy = last_end[1] - first_start[1]
        return (dx * dx + dy * dy) <= self._tolerance * self._tolerance

    def start_point(self) -> Optional[Tuple[float, float]]:
        """Return the starting point of the path, or None if empty."""
        if not self._elements:
            return None
        return self._elements[0].start_point()

    def end_point(self) -> Optional[Tuple[float, float]]:
        """Return the ending point of the path, or None if empty."""
        if not self._elements:
            return None
        return self._elements[-1].end_point()

    def sample(self, resolution: int = 32) -> List[Tuple[float, float]]:
        """Generate sampled points along the entire path.

        Concatenates samples from all elements, avoiding duplicate points
        at element boundaries.

        Args:
            resolution: Number of sample points per curved element.

        Returns:
            List of (x, y) coordinate tuples.
        """
        if not self._elements:
            return []

        all_points: List[Tuple[float, float]] = []

        for i, element in enumerate(self._elements):
            points = element.sample(resolution)
            if i == 0:
                all_points.extend(points)
            else:
                all_points.extend(points[1:])

        return all_points

    def reversed(self) -> CompositePath:
        """Return a new path traversing the same route in reverse direction."""
        reversed_elements = [elem.reversed() for elem in reversed(self._elements)]
        return CompositePath(reversed_elements, tolerance=self._tolerance)

    def length(self) -> float:
        """Return the total length of the path."""
        return sum(element.length() for element in self._elements)

    @classmethod
    def from_segments(cls, segments: List[Any], *, tolerance: float = 1e-9) -> CompositePath:
        """Create a CompositePath from a list of Segment drawables.

        Args:
            segments: List of Segment objects with point1 and point2 attributes
            tolerance: Connection tolerance

        Returns:
            A new CompositePath containing LineSegment elements
        """
        elements: List[PathElement] = []
        for segment in segments:
            elements.append(LineSegment.from_segment(segment))
        return cls(elements, tolerance=tolerance)

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]], *, tolerance: float = 1e-9) -> CompositePath:
        """Create a CompositePath from a list of points (all line segments).

        Args:
            points: List of (x, y) coordinate tuples
            tolerance: Connection tolerance

        Returns:
            A new CompositePath with line segments connecting consecutive points
        """
        if len(points) < 2:
            return cls(tolerance=tolerance)

        elements: List[PathElement] = []
        for i in range(len(points) - 1):
            elements.append(LineSegment(points[i], points[i + 1]))
        return cls(elements, tolerance=tolerance)

    @classmethod
    def from_polygon(cls, polygon: Any, *, tolerance: float = 1e-9) -> CompositePath:
        """Create a closed CompositePath from a polygon drawable.

        Args:
            polygon: A polygon drawable with get_segments() method
                     (Triangle, Rectangle, Quadrilateral, etc.)
            tolerance: Connection tolerance

        Returns:
            A closed CompositePath representing the polygon boundary

        Raises:
            ValueError: If the polygon's segments don't form a closed path
        """
        segments = polygon.get_segments()
        path = cls.from_segments(segments, tolerance=tolerance)
        if not path.is_closed():
            raise ValueError("Polygon segments do not form a closed path")
        return path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositePath):
            return NotImplemented
        return self._elements == other._elements

    def __repr__(self) -> str:
        status = "closed" if self.is_closed() else "open"
        return f"CompositePath({len(self._elements)} elements, {status})"

