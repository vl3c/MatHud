"""
MatHud Polygon Base Class

Abstract base class for polygonal shapes that support rotation around their geometric center.
Provides common rotation functionality for triangles, rectangles, and other multi-vertex shapes.

Key Features:
    - Center point calculation from vertex coordinates
    - Point rotation around arbitrary center using rotation matrices
    - Abstract vertex management interface
    - Mathematical coordinate preservation during transformations

Rotation Mathematics:
    - Uses standard 2D rotation matrix: [cos θ -sin θ; sin θ cos θ]
    - Rotates around calculated geometric center of shape
    - Preserves original mathematical coordinates during transformation

Dependencies:
    - drawables.drawable: Base class interface
    - math: Trigonometric functions for rotation calculations
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

from drawables.drawable import Drawable
from drawables.point import Point

class Polygon(Drawable):
    """Abstract base class for polygons that can be rotated around their geometric center.
    
    Provides rotation capabilities for multi-vertex shapes using center-based rotation
    with mathematical coordinate transformations.
    
    Subclasses must implement:
        get_vertices(): Returns set of vertex Point objects for the polygon
    """
    
    def _get_shape_center(self, points: Set[Point]) -> Tuple[float, float]:
        """Calculate center point of a shape given its vertices"""
        x_coords: list[float] = [p.x for p in points]
        y_coords: list[float] = [p.y for p in points]
        return (sum(x_coords) / len(x_coords), 
                sum(y_coords) / len(y_coords))

    def _rotate_point_around_center(self, point: Point, center_x: float, center_y: float, angle_rad: float) -> None:
        """Rotate a single point around a center by given angle in radians"""
        dx: float = point.x - center_x
        dy: float = point.y - center_y
        
        point.x = center_x + (dx * math.cos(angle_rad) - dy * math.sin(angle_rad))
        point.y = center_y + (dx * math.sin(angle_rad) + dy * math.cos(angle_rad))

    def get_vertices(self) -> Set[Point]:
        """Abstract method to be implemented by subclasses to return their vertices"""
        raise NotImplementedError("Subclasses must implement get_vertices()")

    def translate(self, x_offset: float, y_offset: float) -> None:
        """Translate polygon vertices by the provided offsets."""
        points = list(self.get_vertices())
        for point in points:
            point.translate(x_offset, y_offset)

    # ------------------------------------------------------------------
    # Type metadata caching
    # ------------------------------------------------------------------

    def _set_type_flags(self, flags: Dict[str, bool]) -> None:
        self._type_flags: Dict[str, bool] = dict(flags)

    def get_type_flags(self) -> Dict[str, bool]:
        return dict(getattr(self, "_type_flags", {}))

    def rotate(self, angle: float) -> Tuple[bool, Optional[str]]:
        """Rotate the polygon around its center by the given angle in degrees.
        Returns a tuple (should_proceed, message) where:
        - should_proceed is True if rotation can proceed, False if user should be asked
        - message is None or a message to show to the user"""
        
        # Math model no longer queries canvas; managers decide group-rotation policies
            
        points_to_rotate: Set[Point] = self.get_vertices()
        center_x: float
        center_y: float
        center_x, center_y = self._get_shape_center(points_to_rotate)
        angle_rad: float = math.radians(angle)
        
        for point in points_to_rotate:
            self._rotate_point_around_center(point, center_x, center_y, angle_rad)
            
        # No extra initialization needed in math-only model
        return True, None 