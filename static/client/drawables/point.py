"""
MatHud Point Geometric Object

Fundamental geometric building block representing a point in 2D mathematical space.
Provides coordinate tracking, labeling, and serves as endpoints for other geometric objects.

Key Features:
    - Math coordinate tracking
    - Automatic label display with coordinates
    - CoordinateMapper integration for zoom/pan transformations
    - Translation operations for object manipulation
    - Visibility checking based on canvas bounds

Coordinate Systems:
    - x, y: Mathematical coordinates

Dependencies:
    - constants: Point sizing and labeling configuration
    - drawables.drawable: Base class interface
    - drawables.position: Coordinate container
    - utils.math_utils: Mathematical operations
    - CoordinateMapper: Coordinate transformation service (via canvas)
"""

from constants import default_color, default_point_size, point_label_font_size
from drawables.drawable import Drawable
from drawables.position import Position
from utils.math_utils import MathUtils

class Point(Drawable):
    """Represents a point in 2D mathematical space with coordinate tracking and labeling.
    
    Fundamental building block for all geometric constructions, maintaining mathematical
    coordinates and calculating screen coordinates dynamically via CoordinateMapper.
    Uses Canvas CoordinateMapper for all coordinate transformations.
    
    Attributes:
        x, y (float): Mathematical coordinates (unaffected by zoom/pan)
    """
    def __init__(self, x, y, canvas, name="", color=default_color):
        """Initialize a point with mathematical coordinates and canvas integration.
        
        Args:
            x (float): Mathematical x-coordinate in the coordinate system
            y (float): Mathematical y-coordinate in the coordinate system
            canvas (Canvas): Parent canvas for coordinate transformations
            name (str): Unique identifier for the point
            color (str): CSS color value for point visualization
        """
        self._x = float(x)
        self._y = float(y)
        super().__init__(name=name, color=color, canvas=canvas)
    
    def get_class_name(self):
        return 'Point'


    def __str__(self):
        def fmt(v):
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        return f'{fmt(self.x)},{fmt(self.y)}'
    
    def get_state(self):
        state = {"name": self.name, "args": {"position": {"x": self.x, "y": self.y}}}
        return state
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        # For undo / redo / archive functionality
        # Create a new Point instance with the same coordinates and properties, but do not deep copy the canvas
        new_point = Point(self.x, self.y, self.canvas, name=self.name, color=self.color)
        memo[id(self)] = new_point
        return new_point

    def translate(self, x_offset, y_offset):
        self.x += x_offset
        self.y += y_offset

    def rotate(self, angle):
        pass

    def __eq__(self, other):
        """Checks if two points are equal based on coordinates within tolerance."""
        if not isinstance(other, Point):
            return NotImplemented
        # Use MathUtils for tolerance comparison
        x_match = abs(self.x - other.x) < MathUtils.EPSILON
        y_match = abs(self.y - other.y) < MathUtils.EPSILON
        return x_match and y_match

    def __hash__(self):
        """Computes hash based on rounded coordinates."""
        # Hash based on coordinates rounded to a few decimal places
        # Adjust precision as needed, should be coarser than EPSILON allows differences
        precision = 6 # e.g., 6 decimal places
        rounded_x = round(self.x, precision)
        rounded_y = round(self.y, precision)
        return hash((rounded_x, rounded_y))

    @property
    def x(self):
        """Math x-coordinate (will be flipped to math in this migration step)."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = float(value)

    @property  
    def y(self):
        """Math y-coordinate (will be flipped to math in this migration step)."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = float(value)

    def _initialize(self):
        """Empty method for backward compatibility.
        
        Screen coordinates are now calculated on-demand via x,y properties,
        so no initialization is needed.
        """
        pass

    def _translate(self, screen_offset):
        return

    def zoom(self):
        """Empty zoom method for backward compatibility.
        
        Zoom transformations are now handled centrally by CoordinateMapper
        when drawing, so individual drawable zoom() methods do nothing.
        """
        pass

    def pan(self):
        """Empty pan method for backward compatibility.
        
        Pan transformations are now handled centrally by CoordinateMapper
        when drawing, so individual drawable pan() methods do nothing.
        """
        pass