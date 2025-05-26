"""
MatHud Rotatable Polygon Base Class

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

from drawables.drawable import Drawable
import math

class RotatablePolygon(Drawable):
    """Abstract base class for polygons that can be rotated around their geometric center.
    
    Provides rotation capabilities for multi-vertex shapes using center-based rotation
    with mathematical coordinate transformations.
    
    Subclasses must implement:
        get_vertices(): Returns set of vertex Point objects for the polygon
    """
    
    def _get_shape_center(self, points):
        """Calculate center point of a shape given its vertices"""
        x_coords = [p.original_position.x for p in points]
        y_coords = [p.original_position.y for p in points]
        return (sum(x_coords) / len(x_coords), 
                sum(y_coords) / len(y_coords))

    def _rotate_point_around_center(self, point, center_x, center_y, angle_rad):
        """Rotate a single point around a center by given angle in radians"""
        dx = point.original_position.x - center_x
        dy = point.original_position.y - center_y
        
        point.original_position.x = center_x + (dx * math.cos(angle_rad) - dy * math.sin(angle_rad))
        point.original_position.y = center_y + (dx * math.sin(angle_rad) + dy * math.cos(angle_rad))

    def get_vertices(self):
        """Abstract method to be implemented by subclasses to return their vertices"""
        raise NotImplementedError("Subclasses must implement get_vertices()")

    def rotate(self, angle):
        """Rotate the polygon around its center by the given angle in degrees.
        Returns a tuple (should_proceed, message) where:
        - should_proceed is True if rotation can proceed, False if user should be asked
        - message is None or a message to show to the user"""
        
        # Check if this shape is part of a larger shape
        largest_shape, shape_type = self.canvas.find_largest_connected_shape(self)
        
        if largest_shape:
            return False, f"This {self.get_class_name().lower()} is part of a {shape_type.lower()}. Would you like to rotate the entire {shape_type.lower()} instead?"
            
        points_to_rotate = self.get_vertices()
        center_x, center_y = self._get_shape_center(points_to_rotate)
        angle_rad = math.radians(angle)
        
        for point in points_to_rotate:
            self._rotate_point_around_center(point, center_x, center_y, angle_rad)
            
        self._initialize()
        return True, None 