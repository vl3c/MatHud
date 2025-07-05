"""
MatHud Circle Geometric Object

Represents a circle defined by a center point and radius in 2D mathematical space.
Provides algebraic equation calculation and scale-aware rendering.

Key Features:
    - Center point and radius definition
    - Automatic circle equation calculation ((x-h)² + (y-k)² = r²)
    - Scale factor adaptation for zoom operations
    - Mathematical formula generation for geometric operations

Mathematical Properties:
    - circle_formula: Algebraic equation coefficients
    - Center point tracking through Point object
    - Radius scaling for viewport transformations

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - utils.math_utils: Circle equation calculations
"""

from constants import default_color
from copy import deepcopy
from drawables.drawable import Drawable
from utils.math_utils import MathUtils


class Circle(Drawable):
    """Represents a circle with center point and radius, including mathematical properties.
    
    Maintains a center Point object and radius value, calculating circle equation
    properties for mathematical operations and geometric intersections.
    
    Attributes:
        center (Point): Center point of the circle
        radius (float): Radius in mathematical coordinate units
        circle_formula (dict): Algebraic circle equation coefficients
        drawn_radius (float): Current screen radius (affected by scale factor)
    """
    def __init__(self, center_point, radius, canvas, color=default_color):
        """Initialize a circle with center point and radius.
        
        Args:
            center_point (Point): Center point of the circle
            radius (float): Radius in mathematical coordinate units
            canvas (Canvas): Parent canvas for coordinate transformations
            color (str): CSS color value for circle visualization
        """
        self.center = center_point
        self.radius = radius
        self.circle_formula = self._calculate_circle_algebraic_formula()
        name = f"{self.center.name}({str(self.radius)})"
        super().__init__(name=name, color=color, canvas=canvas)
        self._initialize()

    @Drawable.canvas.setter
    def canvas(self, value):
        self._canvas = value
        self.center.canvas = value

    @canvas.getter
    def canvas(self):
        return self._canvas

    def get_class_name(self):
        return 'Circle'

    def draw(self):
        radius = self.drawn_radius
        x, y = self.center.x, self.center.y
        self.create_svg_element('circle', cx=str(x), cy=str(y), r=str(radius), fill="none", stroke=self.color)

    def _initialize(self):
        self.drawn_radius = self.radius * self.canvas.scale_factor
        self.center._initialize()

    def _calculate_circle_algebraic_formula(self):
        x = self.center.original_position.x
        y = self.center.original_position.y
        r = self.radius
        circle_formula = MathUtils.get_circle_formula(x, y, r)
        return circle_formula

    def zoom(self):
        self.drawn_radius = self.radius * self.canvas.scale_factor
    
    def pan(self):
        pass   # Panning is done by the canvas for the center point 
        
    def get_state(self):
        radius = self.radius
        center = self.center.name
        state = {"name": self.name, "args": {"center": center, "radius": radius, "circle_formula": self.circle_formula}}
        return state

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        # Deep copy the center point
        new_center = deepcopy(self.center, memo)
        # Create a new Circle instance with the copied center point and other properties
        new_circle = Circle(new_center, self.radius, canvas=self.canvas, color=self.color)
        memo[id(self)] = new_circle
        return new_circle

    def translate(self, x_offset, y_offset):
        self.center.original_position.x += x_offset
        self.center.original_position.y += y_offset
        self._initialize()

    def rotate(self, angle):
        pass 