"""
MatHud Triangle Geometric Object

Represents a triangle formed by three connected line segments in 2D mathematical space.
Extends RotatablePolygon to provide rotation capabilities around the triangle's center.

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
    - drawables.rotatable_polygon: Rotation capabilities
    - utils.math_utils: Geometric validation
"""

from constants import default_color
from copy import deepcopy
from drawables.drawable import Drawable
import utils.math_utils as math_utils
from drawables.rotatable_polygon import RotatablePolygon

class Triangle(RotatablePolygon):
    """Represents a triangle formed by three connected line segments.
    
    Validates that three segments form a proper triangle and provides rotation
    capabilities around the triangle's geometric center.
    
    Attributes:
        segment1 (Segment): First side of the triangle
        segment2 (Segment): Second side of the triangle  
        segment3 (Segment): Third side of the triangle
    """
    def __init__(self, segment1, segment2, segment3, canvas, color=default_color):
        """Initialize a triangle from three connected line segments.
        
        Validates that the segments form a proper triangle before construction.
        
        Args:
            segment1 (Segment): First side of the triangle
            segment2 (Segment): Second side of the triangle
            segment3 (Segment): Third side of the triangle
            canvas (Canvas): Parent canvas for coordinate transformations
            color (str): CSS color value for triangle visualization
            
        Raises:
            ValueError: If the segments do not form a valid triangle
        """
        if not self._segments_form_triangle(segment1, segment2, segment3):
            raise ValueError("The segments do not form a triangle")
        self.segment1 = segment1
        self.segment2 = segment2
        self.segment3 = segment3
        name = self._set_name()
        super().__init__(name=name, color=color, canvas=canvas)
        self._initialize()

    def _set_name(self):
        # Get unique vertices using a set first, then sort
        vertices = {p.name for p in [self.segment1.point1, self.segment1.point2, 
                                   self.segment2.point1, self.segment2.point2, 
                                   self.segment3.point1, self.segment3.point2]}
        vertices = sorted(vertices)  # Convert to sorted list
        return vertices[0] + vertices[1] + vertices[2]  # Now we're guaranteed three unique points

    @RotatablePolygon.canvas.setter
    def canvas(self, value):
        self._canvas = value
        self.segment1.canvas = value
        self.segment2.canvas = value
        self.segment3.canvas = value

    @canvas.getter
    def canvas(self):
        return self._canvas

    def get_class_name(self):
        return 'Triangle'

    def draw(self):
        pass   # Drawing is done by the canvas for the segments

    def _initialize(self):
        self.segment1._initialize()
        self.segment2._initialize()
        self.segment3._initialize()

    def _segments_form_triangle(self, s1, s2, s3):
        points = [s1.point1, s1.point2, s2.point1, s2.point2, s3.point1, s3.point2]
        for point in points:
            if points.count(point) != 2:
                return False
        return True
    
    def get_state(self):
        # Collect all point names into a list
        point_names = [
            self.segment1.point1.name, self.segment1.point2.name,
            self.segment2.point1.name, self.segment2.point2.name,
            self.segment3.point1.name, self.segment3.point2.name
        ]
        # Find the most frequent point
        most_frequent_point = max(set(point_names), key=point_names.count)
        # Convert the list into a set to remove duplicates, then convert it back to a list and sort it
        point_names = sorted(list(set(point_names)))
        # Ensure that the list has at least 3 points by appending the most frequent point
        while len(point_names) < 3:
            point_names.append(most_frequent_point)
        state = {"name": self.name, "args": {"p1": point_names[0], "p2": point_names[1], "p3": point_names[2]}}
        return state

    def __deepcopy__(self, memo):
        # Check if the triangle has already been deep copied
        if id(self) in memo:
            return memo[id(self)]
        new_s1 = deepcopy(self.segment1, memo)
        new_s2 = deepcopy(self.segment2, memo)
        new_s3 = deepcopy(self.segment3, memo)
        new_triangle = Triangle(new_s1, new_s2, new_s3, canvas=self.canvas, color=self.color)
        memo[id(self)] = new_triangle
        return new_triangle

    def translate(self, x_offset, y_offset):
        # Translate each unique point only once
        unique_points = {self.segment1.point1, self.segment1.point2, self.segment2.point2}
        
        for point in unique_points:
            point.translate(x_offset, y_offset)
        
        self._initialize()

    def get_vertices(self):
        """Return the set of unique vertices of the triangle"""
        return {
            self.segment1.point1, self.segment1.point2,
            self.segment2.point1, self.segment2.point2,
            self.segment3.point1, self.segment3.point2
        } 