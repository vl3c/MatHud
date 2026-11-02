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
    - circle_formula: Algebraic equation string, computed from the current centre and radius
    - Center point tracking through Point object
    - Renderer-agnostic; no viewport scaling stored in model

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - utils.math_utils: Circle equation calculations
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Collection, Dict, Optional, Tuple, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.point import Point
from name_generator.base import with_unique_suffix
from utils.math_utils import MathUtils
from utils.parameter_expressions import definition_parameters, evaluate_definition


class Circle(Drawable):
    """Represents a circle with center point and radius, including mathematical properties.

    Maintains a center Point object and radius value, calculating circle equation
    properties for mathematical operations and geometric intersections.
    Screen radius is calculated dynamically using CoordinateMapper.

    Attributes:
        center (Point): Center point of the circle
        radius (float): Radius in mathematical coordinate units
        circle_formula (str): Algebraic circle equation, computed from the current geometry
        has_custom_name (bool): True when the name was chosen rather than generated
        radius_expression (str or None): An expression in parameters that defines the
            radius (the circle follows the parameters); None for a plain number. The radius
            is its absolute value, as (x-h)^2 + (y-k)^2 = r^2 draws it. Scaling such a
            circle is refused.
        undefined (bool): The radius expression is not a finite real number, or is 0, at the
            current parameter values; the circle keeps its last radius and is not drawn.
    """

    def __init__(self, center_point: Point, radius: float, color: str = default_color, name: str = "") -> None:
        """Initialize a circle with center point and radius.

        Args:
            center_point (Point): Center point of the circle
            radius (float): Radius in mathematical coordinate units
            color (str): CSS color value for circle visualization
            name (str): Custom name kept through transforms; empty for the default name
        """
        self.center: Point = center_point
        self.radius: float = radius
        self.radius_expression: Optional[str] = None
        self.undefined: bool = False
        self.has_custom_name: bool = bool(name)
        super().__init__(name=name or self._generate_default_name(), color=color)

    def get_class_name(self) -> str:
        return "Circle"

    @property
    def circle_formula(self) -> str:
        """Algebraic circle equation, computed from the current centre and radius.

        Computed on each access so it follows the centre when another object
        (a polygon sharing the centre point, a point transform) moves it.
        """
        return self._calculate_circle_algebraic_formula()

    def _calculate_circle_algebraic_formula(self) -> str:
        x: float = self.center.x
        y: float = self.center.y
        r: float = self.radius
        return str(MathUtils.get_circle_formula(x, y, r))

    def get_state(self) -> Dict[str, Any]:
        radius: float = self.radius
        center: str = self.center.name
        state: Dict[str, Any] = {
            "name": self.name,
            "args": {"center": center, "radius": radius, "circle_formula": self.circle_formula},
        }
        if self.radius_expression is not None:
            state["args"]["radius_expression"] = self.radius_expression
        if self.undefined:
            state["args"]["undefined"] = True
        self._add_color_to_state(state)
        return state

    def parameter_names(self) -> Tuple[str, ...]:
        """The current parameters the radius definition reads, sorted (the centre's are its own)."""
        names: Tuple[str, ...] = definition_parameters(self.radius_expression)
        return names

    def set_radius_definition(self, radius_expression: Optional[str]) -> None:
        """Define the radius by an expression in parameters (already checked by read_definition)."""
        self.radius_expression = radius_expression
        self.refresh_definition()

    def refresh_definition(self) -> bool:
        """Set the radius to |definition| at the current values; True if it changed.

        Not a finite real number, or 0: the circle is marked undefined (not drawn) and keeps
        its radius.
        """
        if self.radius_expression is None:
            return False
        value = evaluate_definition(self.radius_expression)
        if value is None or value == 0:
            self.undefined = True
            return False
        self.undefined = False
        radius = abs(value)
        if radius == self.radius:
            return False
        self.radius = radius
        return True

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(Circle, memo[id(self)])
        # Deep copy the center point
        new_center: Point = deepcopy(self.center, memo)
        # Create a new Circle instance with the copied center point and other properties
        new_circle: Circle = Circle(new_center, self.radius, color=self.color)
        new_circle.radius_expression = self.radius_expression
        new_circle.undefined = self.undefined
        new_circle.name = self.name
        new_circle.has_custom_name = self.has_custom_name
        memo[id(self)] = new_circle
        return new_circle

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.center.x += x_offset
        self.center.y += y_offset
        self.regenerate_name()

    def reflect(self, axis: str, a: float = 0, b: float = 0, c: float = 0) -> None:
        """Reflect the circle across the specified axis (center moves, radius unchanged)."""
        self.center.reflect(axis, a, b, c)
        self.regenerate_name()

    def scale(self, sx: float, sy: float, cx: float, cy: float) -> None:
        """Scale the circle uniformly from center (cx, cy).

        Raises:
            ValueError: If scaling is non-uniform or zero.
        """
        if abs(sx) < 1e-18 or abs(sy) < 1e-18:
            raise ValueError("Scale factor must not be zero")
        if abs(sx - sy) > 1e-9:
            raise ValueError(
                "Non-uniform scaling of a circle is not supported; convert to an ellipse first or use equal sx and sy"
            )
        self.center.scale(sx, sy, cx, cy)
        self.radius = abs(self.radius * sx)
        self.regenerate_name()

    def shear(self, axis: str, factor: float, cx: float, cy: float) -> None:
        """Shearing a circle is not supported.

        Raises:
            ValueError: Always raised.
        """
        raise ValueError("Shearing a circle is not supported; convert to an ellipse first")

    def rotate_around(self, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate the circle center around an arbitrary point (cx, cy)."""
        self.center.rotate_around(angle_deg, cx, cy)
        self.regenerate_name()

    def rotate(self, angle: float) -> None:
        pass

    def _generate_default_name(self) -> str:
        """'<center>(<radius>)'; a radius defined by parameters is named by its expression: A(b)."""
        expression = getattr(self, "radius_expression", None)
        if expression is not None:
            return f"{self.center.name}({expression})"
        return f"{self.center.name}({self._format_radius_for_name(self.radius)})"

    def _format_radius_for_name(self, radius: float) -> str:
        if float(radius).is_integer():
            return str(int(radius))
        return str(radius)

    def is_default_name(self, name: str) -> bool:
        """True when name is the default name, with or without a '_<n>' suffix."""
        base = self._generate_default_name()
        suffix = name[len(base) + 1 :] if name.startswith(base + "_") else ""
        return name == base or suffix.isdigit()

    def regenerate_name(self, taken_names: Collection[str] = ()) -> None:
        """Refresh a default name from the current centre and radius; a custom name is kept.

        A default name that still fits the centre and radius and is not in taken_names is
        kept as it is, so a circle keeps its '_<n>' suffix through transforms. Otherwise
        the name becomes the default name, with the first free '_<n>' suffix when the
        default name is in taken_names.

        Args:
            taken_names: Names other drawables already use
        """
        if self.has_custom_name:
            return
        current_name = self.get_name()
        if current_name not in taken_names and self.is_default_name(current_name):
            return
        self.name = with_unique_suffix(self._generate_default_name(), taken_names)

    def update_color(self, color: str) -> None:
        """Update the circle color metadata."""
        self.color = str(color)

    def update_center_position(self, x: float, y: float) -> None:
        """Move the center point (the circle formula follows it)."""
        self.center.update_position(x, y)
