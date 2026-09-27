"""
MatHud Rectangle Geometric Object

Represents a rectangle formed by four connected line segments in 2D mathematical space.
Extends Polygon to provide rotation capabilities around the rectangle's center.

Key Features:
    - Four-segment rectangle validation and construction
    - Right angle and parallel side verification
    - Rotation around geometric center
    - Translation operations for all vertices
    - Segment connectivity and geometric validation

Geometric Properties:
    - Four segments forming a closed rectangle
    - Right angles at all vertices
    - Parallel opposite sides
    - Center-based rotation capabilities

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - drawables.polygon: Rotation capabilities
    - utils.math_utils: Rectangle validation and geometric calculations
"""

from __future__ import annotations

from typing import Any, Dict, Set

from constants import default_color
from drawables.point import Point
from drawables.quadrilateral import Quadrilateral
from drawables.segment import Segment
from utils.math_utils import MathUtils


class Rectangle(Quadrilateral):
    """Represents a rectangle formed by four connected line segments.

    Validates that four segments form a proper rectangle with right angles and
    provides rotation capabilities around the rectangle's geometric center.

    Attributes:
        segment1 (Segment): First side of the rectangle
        segment2 (Segment): Second side of the rectangle
        segment3 (Segment): Third side of the rectangle
        segment4 (Segment): Fourth side of the rectangle
    """

    def __init__(
        self, segment1: Segment, segment2: Segment, segment3: Segment, segment4: Segment, color: str = default_color
    ) -> None:
        """Initialize a rectangle from four connected line segments.

        Validates that the segments form a proper rectangle with right angles.

        Args:
            segment1 (Segment): First side of the rectangle
            segment2 (Segment): Second side of the rectangle
            segment3 (Segment): Third side of the rectangle
            segment4 (Segment): Fourth side of the rectangle
            color (str): CSS color value for rectangle visualization

        Raises:
            ValueError: If the segments do not form a valid rectangle
        """
        # The quadrilateral checks that the segments close a loop (in any direction) and orders the vertices.
        super().__init__(segment1, segment2, segment3, segment4, color=color)
        corners = [(point.x, point.y) for point in self._points]
        if not MathUtils.is_rectangle(*[coordinate for corner in corners for coordinate in corner]):
            raise ValueError("The quadrilateral formed by the segments is not a rectangle")
        # "rectangle" comes from the computed flags, so a sheared rectangle stops listing it.
        self._set_base_type_labels(["quadrilateral"])

    def get_class_name(self) -> str:
        return "Rectangle"

    def get_state(self) -> Dict[str, Any]:
        # Vertex names in cyclic order around the rectangle (older saves stored them sorted by name).
        state: Dict[str, Any] = {
            "name": self.name,
            "args": {f"p{index + 1}": point.name for index, point in enumerate(self._points)},
        }
        state["types"] = self.get_type_names()
        self._add_color_to_state(state)
        return state

    def get_vertices(self) -> Set[Point]:
        """Return the set of unique vertices of the rectangle"""
        return set(super().get_vertices())

    def update_color(self, color: str) -> None:
        """Update the rectangle and its edge colors."""
        sanitized = str(color)
        self.color = sanitized
        super().update_color(color)
