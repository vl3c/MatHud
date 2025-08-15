"""
MatHud Line Segment Geometric Object

Represents a line segment between two points in 2D mathematical space.
Provides line equation calculation, visibility detection, and rotation capabilities.

Key Features:
    - Two-endpoint line segment representation
    - Automatic line equation calculation (ax + by + c = 0)
    - Visibility detection based on canvas viewport intersection
    - Translation and rotation transformations
    - Midpoint-based rotation around segment center

Mathematical Properties:
    - line_formula: Algebraic line equation coefficients
    - Endpoint coordinate tracking through Point objects
    - Visibility optimization for performance

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - drawables.position: Coordinate calculations
    - utils.math_utils: Line equation and intersection calculations
"""

from constants import default_color
from copy import deepcopy
from drawables.drawable import Drawable
from drawables.position import Position
from utils.math_utils import MathUtils
import math

class Segment(Drawable):
    """Represents a line segment between two points with mathematical line properties.
    
    Maintains references to two Point objects and calculates line equation properties
    for mathematical operations and geometric intersections.
    
    Attributes:
        point1 (Point): First endpoint of the segment
        point2 (Point): Second endpoint of the segment
        line_formula (dict): Algebraic line equation coefficients (a, b, c for ax + by + c = 0)
    """
    def __init__(self, p1, p2, color=default_color):
        """Initialize a line segment between two points.
        
        Args:
            p1 (Point): First endpoint of the segment
            p2 (Point): Second endpoint of the segment
            canvas (Canvas): Parent canvas for coordinate transformations
            color (str): CSS color value for segment visualization
        """
        self.point1 = p1
        self.point2 = p2
        self.line_formula = self._calculate_line_algebraic_formula()
        name = self.point1.name + self.point2.name
        super().__init__(name=name, color=color)

    def get_class_name(self):
        return 'Segment'

    def _calculate_line_algebraic_formula(self):
        p1 = self.point1
        p2 = self.point2
        line_formula = MathUtils.get_line_formula(p1.x, p1.y, p2.x, p2.y)
        return line_formula

    def get_state(self):
        points_names = sorted([self.point1.name, self.point2.name])
        state = {"name": self.name, "args": {"p1": points_names[0], "p2": points_names[1], "line_formula": self.line_formula}}
        return state

    def __deepcopy__(self, memo):
        # Check if the segment has already been deep copied
        if id(self) in memo:
            return memo[id(self)]

        # Deepcopy points that define the segment
        new_p1 = deepcopy(self.point1, memo)
        new_p2 = deepcopy(self.point2, memo)
        # Create a new Segment instance with the copied points
        new_segment = Segment(new_p1, new_p2, color=self.color)
        memo[id(self)] = new_segment

        return new_segment

    def translate(self, x_offset, y_offset):
        self.point1.x += x_offset
        self.point1.y += y_offset
        self.point2.x += x_offset
        self.point2.y += y_offset

    def _get_midpoint(self):
        """Calculate the midpoint of the segment"""
        x = (self.point1.x + self.point2.x) / 2
        y = (self.point1.y + self.point2.y) / 2
        return (x, y)

    def _rotate_point_around_center(self, point, center_x, center_y, angle_rad):
        """Rotate a single point around a center by given angle in radians"""
        dx = point.x - center_x
        dy = point.y - center_y
        
        point.x = center_x + (dx * math.cos(angle_rad) - dy * math.sin(angle_rad))
        point.y = center_y + (dx * math.sin(angle_rad) + dy * math.cos(angle_rad))

    def rotate(self, angle):
        """Rotate the segment around its midpoint by the given angle in degrees"""
        # Get midpoint
        center_x, center_y = self._get_midpoint()
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Rotate both endpoints
        self._rotate_point_around_center(self.point1, center_x, center_y, angle_rad)
        self._rotate_point_around_center(self.point2, center_x, center_y, angle_rad)
        
        # Update line formula
        self.line_formula = self._calculate_line_algebraic_formula()
        
        # Return tuple (should_proceed, message) to match interface
        return True, None 

    def __eq__(self, other):
        """Checks if two segments are equal based on their endpoints."""
        if not isinstance(other, Segment):
            return NotImplemented
        # Use sets to ignore endpoint order
        # Assumes self.point1 and self.point2 are not None
        # Requires Point class to have proper __eq__ and __hash__ 
        if self.point1 is None or self.point2 is None or other.point1 is None or other.point2 is None:
            return False # Or handle appropriately if None points are possible during comparison
        points_self = {self.point1, self.point2}
        points_other = {other.point1, other.point2}
        return points_self == points_other

    def __hash__(self):
        """Computes hash based on a frozenset of the hashes of its endpoint Points."""
        # Hash is based on the IDs of the point objects, order-independent
        if self.point1 is None or self.point2 is None:
             # Consistent with __eq__ if points can be None
             return hash((None, None)) 
        # Use frozenset of point hashes to ensure hash is consistent regardless of point1/point2 order
        # and relies on Point.__hash__ which is value-based.
        return hash(frozenset([hash(self.point1), hash(self.point2)]))