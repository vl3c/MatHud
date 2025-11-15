"""
MatHud Circle Geometric Object

Represents a circle defined by a center point and radius in 2D mathematical space.
Provides algebraic equation calculation and scale-aware rendering.

Key Features:
    - Center point and radius definition
    - Automatic circle equation calculation ((x-h)² + (y-k)² = r²)
    - Pure math model; screen scaling handled by renderer
    - Mathematical formula generation for geometric operations

Mathematical Properties:
    - circle_formula: Algebraic equation coefficients
    - Center point tracking through Point object
    - Renderer-agnostic; no viewport scaling stored in model

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - utils.math_utils: Circle equation calculations
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.point import Point
from utils.math_utils import MathUtils


class Circle(Drawable):
    """Represents a circle with center point and radius, including mathematical properties.
    
    Maintains a center Point object and radius value, calculating circle equation
    properties for mathematical operations and geometric intersections.
    Screen radius is calculated dynamically using CoordinateMapper.
    
    Attributes:
        center (Point): Center point of the circle
        radius (float): Radius in mathematical coordinate units
        circle_formula (dict): Algebraic circle equation coefficients
    """
    def __init__(self, center_point: Point, radius: float, color: str = default_color) -> None:
        """Initialize a circle with center point and radius.
        
        Args:
            center_point (Point): Center point of the circle
            radius (float): Radius in mathematical coordinate units
            color (str): CSS color value for circle visualization
        """
        self.center: Point = center_point
        self.radius: float = radius
        self.circle_formula: Dict[str, float] = self._calculate_circle_algebraic_formula()
        super().__init__(name=self._generate_default_name(), color=color)

    def get_class_name(self) -> str:
        return 'Circle'
    
    def _calculate_circle_algebraic_formula(self) -> Dict[str, float]:
        x: float = self.center.x
        y: float = self.center.y
        r: float = self.radius
        circle_formula: Dict[str, float] = MathUtils.get_circle_formula(x, y, r)
        return circle_formula
        
    def get_state(self) -> Dict[str, Any]:
        radius: float = self.radius
        center: str = self.center.name
        state: Dict[str, Any] = {"name": self.name, "args": {"center": center, "radius": radius, "circle_formula": self.circle_formula}}
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(Circle, memo[id(self)])
        # Deep copy the center point
        new_center: Point = deepcopy(self.center, memo)
        # Create a new Circle instance with the copied center point and other properties
        new_circle: Circle = Circle(new_center, self.radius, color=self.color)
        memo[id(self)] = new_circle
        return new_circle

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.center.x += x_offset
        self.center.y += y_offset

    def rotate(self, angle: float) -> None:
        pass 

    def _generate_default_name(self) -> str:
        return f"{self.center.name}({self._format_radius_for_name(self.radius)})"

    def _format_radius_for_name(self, radius: float) -> str:
        if float(radius).is_integer():
            return str(int(radius))
        return str(radius)

    def regenerate_name(self) -> None:
        """Refresh the circle name based on its center name and radius."""
        self.name = self._generate_default_name()

    def update_color(self, color: str) -> None:
        """Update the circle color metadata."""
        self.color = str(color)

    def update_center_position(self, x: float, y: float) -> None:
        """Move the center point and refresh the cached circle formula."""
        self.center.update_position(x, y)
        self.circle_formula = self._calculate_circle_algebraic_formula()