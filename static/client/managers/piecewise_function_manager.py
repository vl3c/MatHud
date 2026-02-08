"""
MatHud Piecewise Function Management System

Manages piecewise function creation, modification, and deletion for graph visualization.
Handles piecewise function plotting with multiple expression pieces and interval bounds.

Core Responsibilities:
    - Piecewise Function Creation: Creates piecewise function objects from piece definitions
    - Piecewise Function Modification: Updates existing piecewise function properties
    - Piecewise Function Deletion: Safe removal with cleanup of associated colored areas
    - Piece Validation: Ensures piece intervals are properly defined

Integration Points:
    - PiecewiseFunction: Mathematical piecewise function model
    - Canvas: Handles function plotting and visual updates
    - DrawableManager: Coordinates with other geometric objects

State Management:
    - Undo/Redo: Complete state archiving for piecewise function operations
    - Canvas Integration: Immediate visual updates after modifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.piecewise_function import PiecewiseFunction
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator


class PiecewiseFunctionManager:
    """Manages piecewise function drawables for a Canvas."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the PiecewiseFunctionManager.

        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.piecewise_function_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("PiecewiseFunction")

    def get_piecewise_function(self, name: str) -> Optional[PiecewiseFunction]:
        """
        Get a piecewise function by its name.

        Args:
            name: The name of the piecewise function to find

        Returns:
            The piecewise function with the matching name, or None if not found
        """
        piecewise_functions = self.drawables.PiecewiseFunctions
        for pf in piecewise_functions:
            if pf.name == name:
                return pf
        return None

    def draw_piecewise_function(
        self,
        pieces: List[Dict[str, Any]],
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> PiecewiseFunction:
        """
        Draw a piecewise function on the canvas.

        Creates a new piecewise function with the specified pieces.
        Archives the state for undo functionality.

        Args:
            pieces: List of piece definitions, each with:
                - expression: Mathematical expression string
                - left: Left interval bound (None for negative infinity)
                - right: Right interval bound (None for positive infinity)
                - left_inclusive: Whether left bound is included (default True)
                - right_inclusive: Whether right bound is included (default False)
            name: Optional name identifier for the function
            color: Optional color for the plotted function

        Returns:
            The newly created piecewise function object

        Raises:
            ValueError: If piece definitions are invalid
        """
        self.canvas.undo_redo_manager.archive()

        self._validate_pieces(pieces)

        generated_name = self.name_generator.generate_function_name(name)

        color_value = str(color).strip() if color is not None else ""
        function_kwargs: Dict[str, Any] = {
            "name": generated_name,
        }
        if color_value:
            function_kwargs["color"] = color_value

        new_function = PiecewiseFunction(pieces, **function_kwargs)

        self.drawables.add(new_function)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return new_function

    def _validate_pieces(self, pieces: List[Dict[str, Any]]) -> None:
        """Validate piece definitions."""
        if not pieces:
            raise ValueError("At least one piece is required for a piecewise function")

        for i, piece in enumerate(pieces):
            if "expression" not in piece or not piece["expression"]:
                raise ValueError(f"Piece {i + 1} must have an expression")

            left = piece.get("left")
            right = piece.get("right")

            if left is not None and right is not None:
                if left >= right:
                    raise ValueError(
                        f"Piece {i + 1}: left bound ({left}) must be less than right bound ({right})"
                    )

    def delete_piecewise_function(self, name: str) -> bool:
        """
        Delete a piecewise function by its name.

        Finds and removes the piecewise function with the specified name.
        Archives the state for undo functionality.

        Args:
            name: The name of the piecewise function to delete

        Returns:
            True if the function was found and deleted, False otherwise
        """
        pf = self.get_piecewise_function(name)
        if not pf:
            return False

        self.canvas.undo_redo_manager.archive()

        self.drawables.remove(pf)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def update_piecewise_function(
        self,
        function_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """
        Update properties of an existing piecewise function.

        Args:
            function_name: Name of the piecewise function to update
            new_color: Optional new color for the function

        Returns:
            True if the function was updated successfully

        Raises:
            ValueError: If function not found or no valid updates provided
        """
        pf = self.get_piecewise_function(function_name)
        if not pf:
            raise ValueError(f"Piecewise function '{function_name}' was not found.")

        pending_fields = self._collect_fields(new_color)
        self._validate_policy(list(pending_fields.keys()))
        self._validate_payload(new_color)

        self.canvas.undo_redo_manager.archive()
        self._apply_updates(pf, pending_fields, new_color)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _collect_fields(self, new_color: Optional[str]) -> Dict[str, str]:
        """Collect fields that are being updated."""
        pending_fields: Dict[str, str] = {}

        if new_color is not None:
            pending_fields["color"] = "color"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        """Validate that requested field updates are permitted."""
        if not self.piecewise_function_edit_policy:
            validated_rules: Dict[str, EditRule] = {}
            for field in requested_fields:
                if field == "color":
                    validated_rules[field] = EditRule(
                        field="color",
                        category="cosmetic",
                        description="Update the plotted function color.",
                    )
                else:
                    raise ValueError(f"Editing field '{field}' is not permitted for piecewise functions.")
            return validated_rules

        validated_rules = {}
        for field in requested_fields:
            rule = self.piecewise_function_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for piecewise functions.")
            validated_rules[field] = rule

        return validated_rules

    def _validate_payload(self, new_color: Optional[str]) -> None:
        """Validate update payload values."""
        if new_color is not None and not str(new_color).strip():
            raise ValueError("Piecewise function color cannot be empty.")

    def _apply_updates(
        self,
        pf: PiecewiseFunction,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        """Apply validated updates to the piecewise function."""
        if "color" in pending_fields and new_color is not None:
            pf.update_color(str(new_color))

