"""
MatHud Vector Geometric Object

Represents a directed line segment (vector) with origin and tip points, displayed with an arrow tip.
Built on top of the Segment class with additional directional visualization.

Key Features:
    - Directed line segment with origin and tip designation
    - Automatic arrow head calculation and rendering
    - Translation and rotation operations maintaining direction
    - Integration with segment properties for mathematical operations

Visual Elements:
    - Line segment: Rendered using underlying Segment object
    - Arrow tip: Triangular polygon calculated from direction and size
    - Directional properties: Origin and tip point distinction

Dependencies:
    - constants: Default styling and arrow sizing
    - drawables.drawable: Base class interface
    - drawables.point: Endpoint objects
    - drawables.segment: Underlying line representation
    - utils.math_utils: Angle and geometric calculations
"""

from constants import default_color, default_point_size
from copy import deepcopy
from drawables.drawable import Drawable
from drawables.point import Point
from drawables.segment import Segment
import utils.math_utils as math_utils
import math

class Vector(Drawable):
    """Represents a directed line segment (vector) with origin, tip, and arrow head visualization.
    
    Extends the concept of a line segment to include directionality, displayed with
    an arrow head at the tip to indicate vector direction and magnitude.
    
    Attributes:
        segment (Segment): Underlying line segment providing mathematical properties
        origin (Point): Starting point of the vector (property access to segment.point1)
        tip (Point): Ending point of the vector (property access to segment.point2)
    """
    def __init__(self, origin, tip, canvas, color=default_color):
        """Initialize a vector with origin and tip points.
        
        Args:
            origin (Point): Starting point of the vector
            tip (Point): Ending point of the vector (where arrow head is drawn)
            canvas (Canvas): Parent canvas for coordinate transformations
            color (str): CSS color value for vector visualization
        """
        self.segment = Segment(origin, tip, canvas=canvas, color=color)
        name = self.segment.name
        super().__init__(name=name, color=color, canvas=canvas)
        self._initialize()
    
    @property
    def origin(self):
        """Get the origin point of the vector."""
        return self.segment.point1
        
    @property
    def tip(self):
        """Get the tip point of the vector."""
        return self.segment.point2
    
    def get_class_name(self):
        return 'Vector'

    def _initialize(self):
        # No-op: segment computes screen coords via mapper when needed
        pass

    def _draw_tip_triangle(self):
        # Rendering handled by renderer; no-op to preserve interface
        return None

    

    def get_state(self):
        origin = self.segment.point1.name
        tip = self.segment.point2.name
        state = {"name": self.name, "args": {"origin": origin, "tip": tip, "line_formula": self.segment.line_formula}}
        return state
    
    def __deepcopy__(self, memo):
        # Check if the vector has already been deep copied
        if id(self) in memo:
            return memo[id(self)]
        # Deepcopy the origin and tip points that define the vector
        new_origin = deepcopy(self.segment.point1, memo)
        new_tip = deepcopy(self.segment.point2, memo)
        # Create a new Vector instance with the deep-copied points
        new_vector = Vector(new_origin, new_tip, self.canvas, color=self.color)
        # Store the newly created vector in the memo dictionary
        memo[id(self)] = new_vector
        return new_vector

    def translate(self, x_offset, y_offset):
        self.segment.translate(x_offset, y_offset)
        self._initialize()

    def rotate(self, angle):
        """Rotate the vector around its origin by the given angle in degrees"""
        # Use segment's rotation method to rotate the line portion
        should_proceed, message = self.segment.rotate(angle)
        if not should_proceed:
            return False, message
            
        # Initialize to update any dependent properties (like the tip triangle)
        self._initialize()
        return True, None 