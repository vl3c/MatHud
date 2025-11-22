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

from copy import deepcopy
from typing import Any, Dict, Set, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.point import Point
from drawables.polygon import Polygon
from drawables.segment import Segment
from utils.math_utils import MathUtils

class Rectangle(Polygon):
    """Represents a rectangle formed by four connected line segments.
    
    Validates that four segments form a proper rectangle with right angles and
    provides rotation capabilities around the rectangle's geometric center.
    
    Attributes:
        segment1 (Segment): First side of the rectangle
        segment2 (Segment): Second side of the rectangle
        segment3 (Segment): Third side of the rectangle
        segment4 (Segment): Fourth side of the rectangle
    """
    def __init__(self, segment1: Segment, segment2: Segment, segment3: Segment, segment4: Segment, color: str = default_color) -> None:
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
        if not self._segments_form_rectangle(segment1, segment2, segment3, segment4):
            raise ValueError("The segments do not form a rectangle")
        if not MathUtils.is_rectangle(segment1.point1.x, segment1.point1.y, 
                                 segment2.point1.x, segment2.point1.y,
                                 segment3.point1.x, segment3.point1.y, 
                                 segment4.point1.x, segment4.point1.y):
            raise ValueError("The quadrilateral formed by the segments is not a rectangle")
        self.segment1: Segment = segment1
        self.segment2: Segment = segment2
        self.segment3: Segment = segment3
        self.segment4: Segment = segment4
        name: str = segment1.point1.name + segment1.point2.name + segment2.point2.name + segment3.point2.name
        super().__init__(name=name, color=color)

    def get_class_name(self) -> str:
        return 'Rectangle'

    def _segments_form_rectangle(self, s1: Segment, s2: Segment, s3: Segment, s4: Segment) -> bool:
        # Check if the end point of one segment is the start point of the next
        correct_connections: bool = (
            s1.point2 == s2.point1 and
            s2.point2 == s3.point1 and
            s3.point2 == s4.point1 and
            s4.point2 == s1.point1
        )
        return correct_connections

    def get_state(self) -> Dict[str, Any]:
        # Collect all point names into a list
        point_names: list[str] = [
            self.segment1.point1.name, self.segment1.point2.name,
            self.segment2.point1.name, self.segment2.point2.name,
            self.segment3.point1.name, self.segment3.point2.name,
            self.segment4.point1.name, self.segment4.point2.name
        ]
        # Convert the list into a set to remove duplicates, then convert it back to a list and sort it
        points_names: list[str] = sorted(list(set(point_names)))
        state: Dict[str, Any] = {"name": self.name, "args": {"p1": points_names[0], "p2": points_names[1], "p3": points_names[2], "p4": points_names[3]}}
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Check if the rectangle has already been deep copied
        if id(self) in memo:
            return cast(Rectangle, memo[id(self)])
        new_s1: Segment = deepcopy(self.segment1, memo)
        new_s2: Segment = deepcopy(self.segment2, memo)
        new_s3: Segment = deepcopy(self.segment3, memo)
        new_s4: Segment = deepcopy(self.segment4, memo)
        new_rectangle: Rectangle = Rectangle(new_s1, new_s2, new_s3, new_s4, color=self.color)
        memo[id(self)] = new_rectangle
        return new_rectangle

    def translate(self, x_offset: float, y_offset: float) -> None:
        # Translate each unique point only once
        unique_points: Set[Point] = self.get_vertices()
        
        for point in unique_points:
            point.translate(x_offset, y_offset)

    def get_vertices(self) -> Set[Point]:
        """Return the set of unique vertices of the rectangle"""
        return {
            self.segment1.point1, self.segment1.point2,
            self.segment2.point1, self.segment2.point2,
            self.segment3.point1, self.segment3.point2,
            self.segment4.point1, self.segment4.point2
        }

    def update_color(self, color: str) -> None:
        """Update the rectangle and its edge colors."""
        sanitized = str(color)
        self.color = sanitized
        for segment in (self.segment1, self.segment2, self.segment3, self.segment4):
            if hasattr(segment, "update_color") and callable(getattr(segment, "update_color")):
                segment.update_color(sanitized)
            else:
                segment.color = sanitized