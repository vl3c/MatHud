"""
Parametric Function Manager for MatHud

Manages parametric function drawables, handling creation, retrieval, update, and deletion
operations with proper undo/redo archiving.

Parametric functions represent curves defined by x(t) and y(t) expressions over a
parameter range, enabling representation of complex curves like circles, spirals,
and Lissajous figures that cannot be expressed as y = f(x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from constants import default_color
from drawables.parametric_function import ParametricFunction
from managers.dependency_removal import remove_drawable_with_dependencies

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator


class ParametricFunctionManager:
    """
    Manages parametric function objects within the canvas system.

    Handles the lifecycle of parametric function drawables including creation,
    retrieval by name, property updates, and deletion with undo/redo support.

    Attributes:
        canvas: Reference to the parent Canvas instance
        drawables: Container for all drawable objects
        name_generator: Generates unique names for unnamed functions
        dependency_manager: Tracks object dependencies
        proxy: Manager proxy for inter-manager communication
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the ParametricFunctionManager.

        Args:
            canvas: Parent Canvas instance
            drawables: Container for storing drawables
            name_generator: Generator for unique drawable names
            dependency_manager: Manager for tracking dependencies
            proxy: Proxy for inter-manager communication
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.proxy: "DrawableManagerProxy" = proxy

    def _archive_for_undo(self) -> None:
        """Archive current state before making changes for undo support."""
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo:
            undo_redo.archive()

    def draw_parametric_function(
        self,
        x_expression: str,
        y_expression: str,
        name: Optional[str] = None,
        t_min: float = 0.0,
        t_max: Optional[float] = None,
        color: Optional[str] = None,
    ) -> ParametricFunction:
        """
        Create a new parametric function and add it to the canvas.

        Args:
            x_expression: Mathematical expression for x(t)
            y_expression: Mathematical expression for y(t)
            name: Optional name for the function (auto-generated if not provided)
            t_min: Minimum value of parameter t (default: 0)
            t_max: Maximum value of parameter t (default: 2*pi)
            color: Display color (default: default_color)

        Returns:
            The created ParametricFunction drawable

        Raises:
            ValueError: If expressions cannot be parsed
        """
        self._archive_for_undo()

        # Generate name if not provided
        if not name:
            name = self.name_generator.generate_parametric_function_name(None)

        # Use default color if not specified
        if color is None:
            color = default_color

        # Create the parametric function
        func = ParametricFunction(
            x_expression=x_expression,
            y_expression=y_expression,
            name=name,
            t_min=t_min,
            t_max=t_max,
            color=color,
        )

        # Add to container
        self.drawables.add(func)

        # Trigger render if enabled
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return func

    def get_parametric_function(self, name: str) -> Optional[ParametricFunction]:
        """
        Retrieve a parametric function by its name.

        Args:
            name: The name of the function to find

        Returns:
            The ParametricFunction if found, None otherwise
        """
        if not name:
            return None

        for func in self.drawables.ParametricFunctions:
            if getattr(func, "name", None) == name:
                return func  # type: ignore[return-value]

        return None

    def delete_parametric_function(self, name: str) -> bool:
        """
        Delete a parametric function by its name.

        Args:
            name: The name of the function to delete

        Returns:
            True if the function was found and deleted, False otherwise
        """
        func = self.get_parametric_function(name)
        if func is None:
            return False

        self._archive_for_undo()

        # Remove from container
        result = remove_drawable_with_dependencies(self.drawables, self.dependency_manager, func)

        # Trigger render if enabled
        if result and getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return result

    def update_parametric_function(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_t_min: Optional[float] = None,
        new_t_max: Optional[float] = None,
    ) -> bool:
        """
        Update properties of an existing parametric function.

        Args:
            name: The name of the function to update
            new_color: New display color (if provided)
            new_t_min: New minimum parameter value (if provided)
            new_t_max: New maximum parameter value (if provided)

        Returns:
            True if the function was found and updated, False otherwise
        """
        func = self.get_parametric_function(name)
        if func is None:
            return False

        # Check if any changes would be made
        changes_needed = False
        if new_color is not None and new_color != func.color:
            changes_needed = True
        if new_t_min is not None and new_t_min != func.t_min:
            changes_needed = True
        if new_t_max is not None and new_t_max != func.t_max:
            changes_needed = True

        if not changes_needed:
            return True

        self._archive_for_undo()

        # Apply updates
        if new_color is not None:
            func.update_color(new_color)
        if new_t_min is not None:
            func.update_t_min(new_t_min)
        if new_t_max is not None:
            func.update_t_max(new_t_max)

        # Invalidate renderable cache if bounds changed
        if new_t_min is not None or new_t_max is not None:
            renderable = getattr(func, "_renderable", None)
            if renderable:
                renderable.invalidate_cache()

        # Trigger render if enabled
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return True
