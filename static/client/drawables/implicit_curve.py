"""
Implicit Curve Drawable for MatHud

Represents the curve of an equation in x and y, such as x^2 + y^2 = 25, x^2/9 + y^2/4 = 1,
y^2 = x^3 - x or x = 3: every conic, vertical lines and curves that are not functions of x.

Key Features:
    - One equation, read by utils.implicit_relations (standard syntax, no guessing)
    - Plotted over the visible view (utils.implicit_plotter), so it follows pans and zooms
    - No translate or rotate: draw a new equation instead
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, cast

from constants import default_color
from drawables.drawable import Drawable
from utils.implicit_relations import ImplicitEquation, parse_equation


class ImplicitCurve(Drawable):
    """The curve where an equation LHS = RHS in x and y holds.

    Attributes:
        equation: The stored form, "x^2 + y^2 = 25"
        name: Identifier for the curve
        color: Display color for rendering
    """

    # Plotting is slow: the renderer maps the last plot to the view while a zoom is under way
    reuses_plot_while_zooming = True

    def __init__(self, equation: str, name: Optional[str] = None, color: str = default_color) -> None:
        """
        Initialize an implicit curve.

        Args:
            equation: An equation in x and y, such as "x^2 + y^2 = 25"
            name: Optional name for the curve
            color: Display color

        Raises:
            ValueError: If the equation is refused (see implicit_relations.parse_equation)
        """
        self._relation: ImplicitEquation = parse_equation(equation)
        self.equation: str = self._relation.text
        super().__init__(name=name or "c", color=color)

    @property
    def g(self) -> Callable[[float, float], float]:
        """LHS - RHS, zero on the curve."""
        return self._relation.g

    @property
    def factors(self) -> Tuple[Callable[..., float], ...]:
        """Functions whose zero sets together are the curve (see ImplicitEquation)."""
        return self._relation.factors

    def value(self, x: float, y: float) -> float:
        """LHS - RHS at (x, y); NaN where it is undefined."""
        try:
            value = float(self._relation.g(x, y))
        except Exception:
            return float("nan")
        return value

    def get_class_name(self) -> str:
        """Return the class name for serialization and type identification."""
        return "ImplicitCurve"

    def get_state(self) -> Dict[str, Any]:
        """Serialize the curve for workspace persistence."""
        return {"name": self.name, "args": {"equation": self.equation, "color": self.color}}

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ImplicitCurve":
        """A copy rebuilt from the stored equation."""
        if id(self) in memo:
            return cast("ImplicitCurve", memo[id(self)])
        copy = ImplicitCurve(self.equation, name=self.name, color=self.color)
        memo[id(self)] = copy
        return copy

    def update_color(self, color: str) -> None:
        """Update the display color of the curve."""
        self.color = str(color)
