"""
Polar Function Manager for MatHud

Creates, finds, updates and deletes polar curves r = f(θ) (see drawables/polar_function.py),
with one undo entry per change. The AI tools bind to this manager directly (FunctionRegistry),
so it maps null tool arguments to their defaults itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from constants import default_color
from drawables.polar_function import PolarFunction
from managers.dependency_removal import remove_drawable_with_dependencies

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from name_generator.drawable import DrawableNameGenerator


class PolarFunctionManager:
    """Manages polar curves on the canvas.

    Attributes:
        canvas: The parent Canvas
        drawables: Container for all drawable objects
        name_generator: Generates names for unnamed curves
        dependency_manager: Tracks object dependencies
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
    ) -> None:
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager

    def _archive_for_undo(self) -> None:
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo:
            undo_redo.archive()

    def _redraw(self) -> None:
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

    @staticmethod
    def default_theta_range(theta_min: Optional[float], theta_max: Optional[float]) -> Tuple[float, float]:
        """The θ range with nulls filled: 0 to 2π, or θ_min to θ_min + 2π when only θ_min is given."""
        low = PolarFunction.DEFAULT_THETA_MIN if theta_min is None else theta_min
        if theta_max is not None:
            return low, theta_max
        if theta_min is None:
            return low, PolarFunction.DEFAULT_THETA_MAX
        try:
            return low, float(theta_min) + 2 * math.pi
        except (TypeError, ValueError):
            return low, PolarFunction.DEFAULT_THETA_MAX

    def draw_polar_function(
        self,
        r_expression: str,
        name: Optional[str] = None,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
        color: Optional[str] = None,
    ) -> PolarFunction:
        """Create a polar curve r = f(θ) and add it to the canvas (one undo entry).

        Args:
            r_expression: r in θ (or theta), such as "1 + cos(θ)"
            name: Optional name; a free requested name is kept, else one is generated
            theta_min: Start of the θ range in radians (null: 0)
            theta_max: End of the θ range in radians (null: θ_min + 2π)
            color: Display color (null: the default)

        Returns:
            The created PolarFunction

        Raises:
            ValueError: If the expression or range is refused (see PolarFunction); nothing
                is then changed and no undo entry is made
        """
        low, high = self.default_theta_range(theta_min, theta_max)
        curve = PolarFunction(
            r_expression=r_expression,
            name="r",
            theta_min=low,
            theta_max=high,
            color=color or default_color,
        )
        curve.name = (
            self.name_generator.generate_polar_function_name(None) if not name else self.drawables.unique_name(name)
        )
        self._archive_for_undo()
        self.drawables.add(curve)
        self._redraw()
        return curve

    def get_polar_function(self, name: str) -> Optional[PolarFunction]:
        """The polar curve with this name, or None."""
        if not name:
            return None
        for curve in self.drawables.PolarFunctions:
            if getattr(curve, "name", None) == name:
                return curve  # type: ignore[return-value]
        return None

    def delete_polar_function(self, name: str) -> bool:
        """Delete the polar curve with this name; False when there is none."""
        curve = self.get_polar_function(name)
        if curve is None:
            return False
        self._archive_for_undo()
        result = remove_drawable_with_dependencies(self.drawables, self.dependency_manager, curve)
        if result:
            self._redraw()
        return bool(result)

    def update_polar_function(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_theta_min: Optional[float] = None,
        new_theta_max: Optional[float] = None,
    ) -> bool:
        """Change a polar curve's color or θ range; null fields stay as they are.

        Returns:
            True if the curve exists (changed or already as asked), False when there is none

        Raises:
            ValueError: If the new range is refused (empty, too long, or r has no real value
                on it); the curve is then unchanged and no undo entry is made
        """
        curve = self.get_polar_function(name)
        if curve is None:
            return False
        color_changes = new_color is not None and new_color != curve.color
        range_changes = (new_theta_min is not None and new_theta_min != curve.theta_min) or (
            new_theta_max is not None and new_theta_max != curve.theta_max
        )
        if not (color_changes or range_changes):
            return True
        theta_range: Any = None
        if range_changes:
            low, high = PolarFunction.ordered_theta_range(
                curve.theta_min if new_theta_min is None else new_theta_min,
                curve.theta_max if new_theta_max is None else new_theta_max,
            )
            PolarFunction.check_has_real_value(curve.r_expression, curve._r_function, low, high)
            theta_range = (low, high)

        self._archive_for_undo()
        if color_changes:
            curve.update_color(str(new_color))
        if theta_range is not None:
            curve.update_theta_range(*theta_range)
            renderable = getattr(curve, "_renderable", None)
            if renderable:
                renderable.invalidate_cache()
        self._redraw()
        return True
