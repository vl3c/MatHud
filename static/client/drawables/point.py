"""
MatHud Point Geometric Object

Fundamental geometric building block representing a point in 2D mathematical space.
Provides coordinate tracking, labeling, and serves as endpoints for other geometric objects.

Key Features:
    - Original and screen coordinate tracking
    - Automatic label display with coordinates
    - CoordinateMapper integration for zoom/pan transformations
    - Translation operations for object manipulation
    - Visibility checking based on canvas bounds

Coordinate Systems:
    - original_position: Mathematical coordinates (unchanged by zoom/pan)
    - x, y: Screen coordinates (calculated via CoordinateMapper)

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
import math

class Point(Drawable):
    """Represents a point in 2D mathematical space with coordinate tracking and labeling.
    
    Fundamental building block for all geometric constructions, maintaining both original
    mathematical coordinates and transformed screen coordinates for proper rendering.
    Uses Canvas CoordinateMapper for all coordinate transformations.
    
    Attributes:
        original_position (Position): Mathematical coordinates (unaffected by zoom/pan)
        x (float): Current screen x-coordinate (calculated via CoordinateMapper)
        y (float): Current screen y-coordinate (calculated via CoordinateMapper)
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
        self.original_position = Position(x, y)
        super().__init__(name=name, color=color, canvas=canvas)
        self._initialize()
    
    @Drawable.canvas.setter
    def canvas(self, value):
        self._canvas = value
        self._initialize()

    @canvas.getter
    def canvas(self):
        return self._canvas
    
    def get_class_name(self):
        return 'Point'

    def draw(self):
        if not self.is_visible():
            return
        x, y = self.x, self.y
        # Draw point
        self.create_svg_element('circle', cx=str(x), cy=str(y), r=str(default_point_size), fill=self.color)
        # Draw label
        label_text = self.name + f'({round(self.original_position.x, 3)}, {round(self.original_position.y, 3)})'
        label_offset = default_point_size
        self.create_svg_element('text', x=str(x+label_offset), y=str(y-label_offset), fill=self.color, text_content=label_text, text_font_size=point_label_font_size)

    def __str__(self):
        x = self.x
        y = self.y
        return f'{x},{y}'
    
    def _initialize(self):
        """Convert mathematical coordinates to screen coordinates using CoordinateMapper."""
        self.x, self.y = self.canvas.coordinate_mapper.math_to_screen(
            self.original_position.x, self.original_position.y
        )

    def _translate(self, offset_point):
        """Translate point by screen offset."""
        self.x += offset_point.x
        self.y += offset_point.y

    def zoom(self):
        """Apply zoom transformation using CoordinateMapper."""
        current_screen_pos = Position(self.x, self.y)
        displacement = self.canvas.coordinate_mapper.get_zoom_towards_point_displacement(current_screen_pos)
        self._translate(displacement)

    def pan(self):
        """Apply pan transformation using CoordinateMapper."""
        self._translate(self.canvas.coordinate_mapper.offset)
        
    def get_state(self):
        pos = self.original_position
        state = {"name": self.name, "args": {"position": {"x": pos.x, "y": pos.y}}}
        return state
    
    def is_visible(self):
        """Check if point is visible within canvas bounds using CoordinateMapper."""
        return self.canvas.coordinate_mapper.is_point_visible(self.x, self.y)
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        # For undo / redo / archive functionality
        # Create a new Point instance with the same coordinates and properties, but do not deep copy the canvas
        new_point = Point(self.original_position.x, self.original_position.y, self.canvas, name=self.name, color=self.color)
        memo[id(self)] = new_point
        return new_point

    def translate(self, x_offset, y_offset):
        self.original_position.x += x_offset
        self.original_position.y += y_offset
        self._initialize()

    def rotate(self, angle):
        pass

    def __eq__(self, other):
        """Checks if two points are equal based on coordinates within tolerance."""
        if not isinstance(other, Point):
            return NotImplemented
        # Use MathUtils for tolerance comparison
        x_match = abs(self.original_position.x - other.original_position.x) < MathUtils.EPSILON
        y_match = abs(self.original_position.y - other.original_position.y) < MathUtils.EPSILON
        return x_match and y_match

    def __hash__(self):
        """Computes hash based on rounded coordinates."""
        # Hash based on coordinates rounded to a few decimal places
        # Adjust precision as needed, should be coarser than EPSILON allows differences
        precision = 6 # e.g., 6 decimal places
        rounded_x = round(self.original_position.x, precision)
        rounded_y = round(self.original_position.y, precision)
        return hash((rounded_x, rounded_y))