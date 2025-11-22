"""
MatHud Triangle Geometric Object

Represents a triangle formed by three connected line segments in 2D mathematical space.
Extends Polygon to provide rotation capabilities around the triangle's center.

Key Features:
    - Three-segment triangle validation and construction
    - Automatic vertex naming from segment endpoints
    - Rotation around geometric center
    - Translation operations for all vertices
    - Segment connectivity validation

Geometric Properties:
    - Three segments forming a closed triangle
    - Unique vertex identification and naming
    - Center-based rotation capabilities
    - Vertex set management for transformations

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - drawables.polygon: Rotation capabilities
    - utils.geometry_utils: Type classification helpers
    - utils.math_utils: Geometric validation
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Set, cast

from constants import default_color
from drawables.point import Point
from drawables.polygon import Polygon
from drawables.segment import Segment
from utils.geometry_utils import GeometryUtils

class Triangle(Polygon):
    """Represents a triangle formed by three connected line segments.
    
    Validates that three segments form a proper triangle and provides rotation
    capabilities around the triangle's geometric center.
    
    Attributes:
        segment1 (Segment): First side of the triangle
        segment2 (Segment): Second side of the triangle  
        segment3 (Segment): Third side of the triangle
    """
    def __init__(self, segment1: Segment, segment2: Segment, segment3: Segment, color: str = default_color) -> None:
        """Initialize a triangle from three connected line segments.
        
        Validates that the segments form a proper triangle before construction.
        
        Args:
            segment1 (Segment): First side of the triangle
            segment2 (Segment): Second side of the triangle
            segment3 (Segment): Third side of the triangle
            color (str): CSS color value for triangle visualization
            
        Raises:
            ValueError: If the segments do not form a valid triangle
        """
        if not self._segments_form_triangle(segment1, segment2, segment3):
            raise ValueError("The segments do not form a triangle")
        self.segment1: Segment = segment1
        self.segment2: Segment = segment2
        self.segment3: Segment = segment3
        self._segments: list[Segment] = [self.segment1, self.segment2, self.segment3]
        self._set_type_flags(self._classify_triangle())
        name: str = self._set_name()
        super().__init__(name=name, color=color)

    def _set_name(self) -> str:
        # Get unique vertices using a set first, then sort
        vertices: Set[str] = {p.name for p in [self.segment1.point1, self.segment1.point2, 
                                   self.segment2.point1, self.segment2.point2, 
                                   self.segment3.point1, self.segment3.point2]}
        vertices_list: list[str] = sorted(vertices)  # Convert to sorted list
        return vertices_list[0] + vertices_list[1] + vertices_list[2]  # Now we're guaranteed three unique points 

    def get_class_name(self) -> str:
        return 'Triangle'

    def _segments_form_triangle(self, s1: Segment, s2: Segment, s3: Segment) -> bool:
        points: list[Point] = [s1.point1, s1.point2, s2.point1, s2.point2, s3.point1, s3.point2]
        for point in points:
            if points.count(point) != 2:
                return False
        return True
    
    def _classify_triangle(self) -> Dict[str, bool]:
        flags = GeometryUtils.triangle_type_flags_from_segments(self._segments)
        if flags is None:
            return {"equilateral": False, "isosceles": False, "scalene": False, "right": False}
        return flags

    def get_type_flags(self) -> Dict[str, bool]:
        """Return classification flags describing the triangle."""
        return super().get_type_flags()

    def is_equilateral(self) -> bool:
        return self.get_type_flags()["equilateral"]

    def is_isosceles(self) -> bool:
        return self.get_type_flags()["isosceles"]

    def is_scalene(self) -> bool:
        return self.get_type_flags()["scalene"]

    def is_right(self) -> bool:
        return self.get_type_flags()["right"]
    
    def get_state(self) -> Dict[str, Any]:
        # Collect all point names into a list
        point_names: list[str] = [
            self.segment1.point1.name, self.segment1.point2.name,
            self.segment2.point1.name, self.segment2.point2.name,
            self.segment3.point1.name, self.segment3.point2.name
        ]
        # Find the most frequent point
        most_frequent_point: str = max(set(point_names), key=point_names.count)
        # Convert the list into a set to remove duplicates, then convert it back to a list and sort it
        point_names = sorted(list(set(point_names)))
        # Ensure that the list has at least 3 points by appending the most frequent point
        while len(point_names) < 3:
            point_names.append(most_frequent_point)
        state: Dict[str, Any] = {
            "name": self.name,
            "args": {"p1": point_names[0], "p2": point_names[1], "p3": point_names[2]},
            "types": self.get_type_flags(),
        }
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Check if the triangle has already been deep copied
        if id(self) in memo:
            return cast(Triangle, memo[id(self)])
        new_s1: Segment = deepcopy(self.segment1, memo)
        new_s2: Segment = deepcopy(self.segment2, memo)
        new_s3: Segment = deepcopy(self.segment3, memo)
        new_triangle: Triangle = Triangle(new_s1, new_s2, new_s3, color=self.color)
        memo[id(self)] = new_triangle
        return new_triangle

    def get_vertices(self) -> Set[Point]:
        """Return the set of unique vertices of the triangle"""
        return {
            self.segment1.point1, self.segment1.point2,
            self.segment2.point1, self.segment2.point2,
            self.segment3.point1, self.segment3.point2
        }

    def update_color(self, color: str) -> None:
        """Update the triangle and its edge colors."""
        sanitized = str(color)
        self.color = sanitized
        for segment in (self.segment1, self.segment2, self.segment3):
            if hasattr(segment, "update_color") and callable(getattr(segment, "update_color")):
                segment.update_color(sanitized)
            else:
                segment.color = sanitized