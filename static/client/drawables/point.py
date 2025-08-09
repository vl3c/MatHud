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

class Point(Drawable):
    """Represents a point in 2D mathematical space with coordinate tracking and labeling.
    
    Fundamental building block for all geometric constructions, maintaining mathematical
    coordinates and calculating screen coordinates dynamically via CoordinateMapper.
    Uses Canvas CoordinateMapper for all coordinate transformations.
    
    Attributes:
        original_position (Position): Mathematical coordinates (unaffected by zoom/pan)
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
    
    @Drawable.canvas.setter
    def canvas(self, value):
        self._canvas = value

    @canvas.getter
    def canvas(self):
        return self._canvas
    
    def get_class_name(self):
        return 'Point'

    def draw(self):
        # Rendering handled by renderer; no-op to preserve interface
        return None

    def __str__(self):
        return f'{self.original_position.x},{self.original_position.y}'
    
    def get_state(self):
        pos = self.original_position
        state = {"name": self.name, "args": {"position": {"x": pos.x, "y": pos.y}}}
        return state
    
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

    @property
    def x(self):
        """Math x-coordinate (will be flipped to math in this migration step)."""
        return self.original_position.x

    @property  
    def y(self):
        """Math y-coordinate (will be flipped to math in this migration step)."""
        return self.original_position.y

    @property
    def screen_x(self):
        """Explicit screen x-coordinate (alias for current behavior)."""
        screen_x, _ = self.canvas.coordinate_mapper.math_to_screen(
            self.original_position.x, self.original_position.y)
        return screen_x

    @property
    def screen_y(self):
        """Explicit screen y-coordinate (alias for current behavior)."""
        _, screen_y = self.canvas.coordinate_mapper.math_to_screen(
            self.original_position.x, self.original_position.y)
        return screen_y

    def _initialize(self):
        """Empty method for backward compatibility.
        
        Screen coordinates are now calculated on-demand via x,y properties,
        so no initialization is needed.
        """
        pass

    def _translate(self, screen_offset):
        """Translate point by screen coordinate offset for backward compatibility.
        
        Converts screen offset to mathematical offset and applies translation.
        This maintains compatibility with old code that used screen-space translation.
        
        Args:
            screen_offset (Position): Screen coordinate offset to apply
        """
        # Convert screen offset to mathematical offset
        math_dx = screen_offset.x / self.canvas.coordinate_mapper.scale_factor
        math_dy = -screen_offset.y / self.canvas.coordinate_mapper.scale_factor  # Y-axis flip
        
        # Apply mathematical translation
        self.translate(math_dx, math_dy)

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