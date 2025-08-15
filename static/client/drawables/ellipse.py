"""
MatHud Ellipse Geometric Object

Represents an ellipse defined by center point, radii, and rotation angle in 2D mathematical space.
Provides algebraic equation calculation and rotation transformation capabilities.

Key Features:
    - Center point and dual radius definition (rx, ry)
    - Rotation angle support for arbitrary ellipse orientation
    - Automatic ellipse equation calculation
    - Scale factor adaptation for zoom operations
    - SVG rotation transformation for rendering

Mathematical Properties:
    - ellipse_formula: Algebraic equation coefficients
    - Center point tracking through Point object
    - Dual radius scaling for viewport transformations
    - Rotation angle preservation and application

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - utils.math_utils: Ellipse equation calculations
"""

from constants import default_color
from copy import deepcopy
from drawables.drawable import Drawable
from utils.math_utils import MathUtils

class Ellipse(Drawable):
    """Represents an ellipse with center point, dual radii, and rotation angle.
    
    Maintains center Point object, x/y radius values, and rotation angle,
    calculating ellipse equation properties for mathematical operations.
    Screen radii are calculated dynamically using CoordinateMapper.
    
    Attributes:
        center (Point): Center point of the ellipse
        radius_x (float): Horizontal radius in mathematical coordinate units
        radius_y (float): Vertical radius in mathematical coordinate units
        rotation_angle (float): Rotation angle in degrees for ellipse orientation
        ellipse_formula (dict): Algebraic ellipse equation coefficients
    """
    def __init__(self, center_point, radius_x, radius_y, rotation_angle=0, color=default_color):
        """Initialize an ellipse with center point, radii, and rotation.
        
        Args:
            center_point (Point): Center point of the ellipse
            radius_x (float): Horizontal radius in mathematical coordinate units
            radius_y (float): Vertical radius in mathematical coordinate units
            canvas (Canvas): Parent canvas for coordinate transformations
            rotation_angle (float): Rotation angle in degrees (default: 0)
            color (str): CSS color value for ellipse visualization
        """
        self.center = center_point
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.rotation_angle = rotation_angle  # Initialize with provided angle
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        name = f"{self.center.name}({str(self.radius_x)}, {str(self.radius_y)})"
        super().__init__(name=name, color=color)

    def get_class_name(self):
        return 'Ellipse'
    
    def _calculate_ellipse_algebraic_formula(self):
        x = self.center.x
        y = self.center.y
        return MathUtils.get_ellipse_formula(x, y, self.radius_x, self.radius_y, self.rotation_angle)
        
    def get_state(self):
        """Return the state of the ellipse including rotation"""
        state = {
            "name": self.name, 
            "args": {
                "center": self.center.name, 
                "radius_x": self.radius_x, 
                "radius_y": self.radius_y, 
                "rotation_angle": self.rotation_angle,
                "ellipse_formula": self.ellipse_formula
            }
        }
        return state

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        # Deep copy the center point
        new_center = deepcopy(self.center, memo)
        # Create a new Ellipse instance with the copied center point and other properties
        new_ellipse = Ellipse(new_center, self.radius_x, self.radius_y, 
                             color=self.color, 
                             rotation_angle=self.rotation_angle)
        memo[id(self)] = new_ellipse
        return new_ellipse

    def translate(self, x_offset, y_offset):
        self.center.x += x_offset
        self.center.y += y_offset

    def rotate(self, angle):
        """Rotate the ellipse around its center by the given angle in degrees"""
        # Update rotation angle (keep it between 0 and 360 degrees)
        self.rotation_angle = (self.rotation_angle + angle) % 360
        
        # Update ellipse formula if needed
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        
        # Return tuple (should_proceed, message) to match interface
        return True, None 