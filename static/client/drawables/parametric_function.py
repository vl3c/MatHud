"""
Parametric Function Drawable for MatHud

Represents a parametric curve defined by x(t) and y(t) expressions over a parameter range.
Parametric functions allow plotting curves that cannot be expressed as y = f(x), such as
circles, spirals, Lissajous curves, and other complex shapes.

Key Features:
    - Dual expression support: x(t) and y(t) evaluated at parameter t
    - Configurable parameter range with t_min and t_max
    - Full state serialization for workspace persistence
    - Integration with the rendering pipeline via ParametricFunctionRenderable
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple, cast

from constants import default_color
from drawables.drawable import Drawable
from expression_validator import ExpressionValidator


class ParametricFunction(Drawable):
    """
    A parametric curve defined by x(t) and y(t) expressions.

    Unlike regular functions that map x -> y, parametric functions define both
    coordinates as functions of an independent parameter t, enabling representation
    of curves that loop, spiral, or cross themselves.

    Attributes:
        x_expression: String expression for x(t), e.g., "cos(t)"
        y_expression: String expression for y(t), e.g., "sin(t)"
        t_min: Minimum value of parameter t (default: 0)
        t_max: Maximum value of parameter t (default: 2*pi)
        name: Identifier for the parametric curve
        color: Display color for rendering
    """

    def __init__(
        self,
        x_expression: str,
        y_expression: str,
        name: Optional[str] = None,
        t_min: float = 0.0,
        t_max: Optional[float] = None,
        color: str = default_color,
    ) -> None:
        """
        Initialize a parametric function with x(t) and y(t) expressions.

        Args:
            x_expression: Mathematical expression for x as a function of t
            y_expression: Mathematical expression for y as a function of t
            name: Optional name/label for the curve
            t_min: Starting value for parameter t (default: 0)
            t_max: Ending value for parameter t (default: 2*pi)
            color: Display color for the curve

        Raises:
            ValueError: If expressions cannot be parsed
        """
        self.t_min: float = float(t_min)
        self.t_max: float = float(t_max) if t_max is not None else 2 * math.pi

        try:
            self.x_expression: str = ExpressionValidator.fix_math_expression(x_expression)
            self.y_expression: str = ExpressionValidator.fix_math_expression(y_expression)
            self._x_function: Callable[[float], float] = ExpressionValidator.parse_parametric_expression(self.x_expression)
            self._y_function: Callable[[float], float] = ExpressionValidator.parse_parametric_expression(self.y_expression)
        except Exception as e:
            raise ValueError(
                f"Failed to parse parametric expressions x='{x_expression}', y='{y_expression}': {str(e)}"
            )

        super().__init__(name=name or "p", color=color)

    def evaluate(self, t: float) -> Tuple[float, float]:
        """
        Evaluate the parametric function at parameter t.

        Args:
            t: Parameter value at which to evaluate

        Returns:
            Tuple of (x, y) coordinates at the given t value
        """
        return (self._x_function(t), self._y_function(t))

    def evaluate_x(self, t: float) -> float:
        """Evaluate only the x component at parameter t."""
        return self._x_function(t)

    def evaluate_y(self, t: float) -> float:
        """Evaluate only the y component at parameter t."""
        return self._y_function(t)

    def get_class_name(self) -> str:
        """Return the class name for serialization and type identification."""
        return "ParametricFunction"

    def get_state(self) -> Dict[str, Any]:
        """
        Serialize the parametric function state for workspace persistence.

        Returns:
            Dictionary containing all state needed to reconstruct the curve
        """
        return {
            "name": self.name,
            "args": {
                "x_expression": self.x_expression,
                "y_expression": self.y_expression,
                "t_min": self.t_min,
                "t_max": self.t_max,
                "color": self.color,
            }
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ParametricFunction":
        """
        Create a deep copy of the parametric function.

        Args:
            memo: Dictionary of already-copied objects to handle circular refs

        Returns:
            New ParametricFunction instance with copied state
        """
        if id(self) in memo:
            return cast("ParametricFunction", memo[id(self)])

        new_func = ParametricFunction(
            x_expression=self.x_expression,
            y_expression=self.y_expression,
            name=self.name,
            t_min=self.t_min,
            t_max=self.t_max,
            color=self.color,
        )
        memo[id(self)] = new_func
        return new_func

    def translate(self, x_offset: float, y_offset: float) -> None:
        """
        Translate the parametric curve by the given offsets.

        Modifies the x and y expressions to include the translation.

        Args:
            x_offset: Amount to shift in the x direction
            y_offset: Amount to shift in the y direction
        """
        if x_offset == 0 and y_offset == 0:
            return

        try:
            if x_offset != 0:
                new_x_expression = f"({self.x_expression}) + {x_offset}"
                self.x_expression = ExpressionValidator.fix_math_expression(new_x_expression)
                self._x_function = ExpressionValidator.parse_parametric_expression(self.x_expression)

            if y_offset != 0:
                new_y_expression = f"({self.y_expression}) + {y_offset}"
                self.y_expression = ExpressionValidator.fix_math_expression(new_y_expression)
                self._y_function = ExpressionValidator.parse_parametric_expression(self.y_expression)
        except Exception as e:
            print(f"Warning: Could not translate parametric function: {str(e)}")

    def rotate(self, angle: float) -> None:
        """
        Rotate the parametric curve about the origin by the given angle.

        Args:
            angle: Rotation angle in radians
        """
        if angle == 0:
            return

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        try:
            # x' = x*cos(angle) - y*sin(angle)
            # y' = x*sin(angle) + y*cos(angle)
            old_x_expr = self.x_expression
            old_y_expr = self.y_expression

            new_x_expression = f"({old_x_expr})*{cos_a} - ({old_y_expr})*{sin_a}"
            new_y_expression = f"({old_x_expr})*{sin_a} + ({old_y_expr})*{cos_a}"

            self.x_expression = ExpressionValidator.fix_math_expression(new_x_expression)
            self.y_expression = ExpressionValidator.fix_math_expression(new_y_expression)
            self._x_function = ExpressionValidator.parse_parametric_expression(self.x_expression)
            self._y_function = ExpressionValidator.parse_parametric_expression(self.y_expression)
        except Exception as e:
            print(f"Warning: Could not rotate parametric function: {str(e)}")

    def update_color(self, color: str) -> None:
        """Update the display color of the curve."""
        self.color = str(color)

    def update_t_min(self, t_min: float) -> None:
        """Update the minimum parameter value."""
        self.t_min = float(t_min)

    def update_t_max(self, t_max: float) -> None:
        """Update the maximum parameter value."""
        self.t_max = float(t_max)
