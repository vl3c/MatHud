"""
Inequality Colored Area for MatHud

The region of the plane where inequalities in x and y hold: y < x^2, x^2 + y^2 <= 4,
y > x^2 - 2 and y < 1, the feasible region x + y <= 10, x >= 0, y >= 0.

Key Features:
    - Comparisons joined by and/or (and commas), read by utils.implicit_relations
    - Shaded over the visible view, with its boundary dashed where strict (<, >) and solid
      where not (<=, >=), as graphing calculators draw it
    - A colored area: drawn below other objects, deleted and edited with the colored-area tools
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, cast

from constants import default_area_fill_color, default_area_opacity
from drawables.colored_area import ColoredArea
from utils.implicit_relations import InequalityTree, parse_inequality


class InequalityColoredArea(ColoredArea):
    """The region where inequalities in x and y hold.

    Attributes:
        inequality: The stored form, "y > x^2 - 2 and y < 1"
        color, opacity: The fill (the boundary is drawn in the same color, opaque)
    """

    # Plotting is slow: the renderer maps the last plot to the view while a zoom is under way
    reuses_plot_while_zooming = True

    def __init__(
        self,
        inequality: str,
        name: Optional[str] = None,
        color: str = default_area_fill_color,
        opacity: float = default_area_opacity,
        check_names: bool = True,
    ) -> None:
        """
        Initialize the region.

        Raises:
            ValueError: If the inequality is refused (see implicit_relations.parse_inequality)
        """
        self._relation: InequalityTree = parse_inequality(inequality, check_names=check_names)
        self.inequality: str = self._relation.text
        super().__init__(name=name or "region", color=color, opacity=opacity)

    @property
    def relation(self) -> InequalityTree:
        """The parsed inequalities."""
        return self._relation

    @property
    def value(self) -> Callable[[float, float], float]:
        """Negative exactly inside the region (see InequalityTree.value)."""
        return self._relation.value

    def parameter_names(self) -> Tuple[str, ...]:
        """The current parameters the inequalities read, sorted."""
        return self._relation.parameter_names()

    def reads_missing_names(self) -> bool:
        """Whether it reads a name that is not a parameter now (it is then shaded nowhere)."""
        return self._relation.reads_missing_names()

    def contains(self, x: float, y: float) -> bool:
        """Whether (x, y) satisfies the inequalities (a non-strict boundary point does)."""
        return self._relation.contains(x, y)

    def get_class_name(self) -> str:
        """Return the class name for serialization and type identification."""
        return "InequalityColoredArea"

    def get_state(self) -> Dict[str, Any]:
        """Serialize the region for workspace persistence."""
        return {
            "name": self.name,
            "args": {"inequality": self.inequality, "color": self.color, "opacity": self.opacity},
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "InequalityColoredArea":
        """A copy rebuilt from the stored inequality."""
        if id(self) in memo:
            return cast("InequalityColoredArea", memo[id(self)])
        copy = InequalityColoredArea(
            self.inequality, name=self.name, color=self.color, opacity=self.opacity, check_names=False
        )
        memo[id(self)] = copy
        return copy
