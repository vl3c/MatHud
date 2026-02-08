"""
MatHud Ellipse Management System

Manages ellipse creation, retrieval, and deletion operations for geometric visualization.
Handles ellipse operations with automatic center point management and dependency tracking.

Core Responsibilities:
    - Ellipse Creation: Creates ellipses from center coordinates, radii, and rotation angle
    - Ellipse Retrieval: Lookup by center/radii parameters or ellipse name
    - Ellipse Deletion: Safe removal with proper cleanup
    - Center Point Management: Automatic creation and tracking of ellipse centers

Manager Features:
    - Collision Detection: Checks for existing ellipses before creation (without rotation angle)
    - Dependency Tracking: Registers ellipse relationships with center points
    - State Archiving: Automatic undo/redo state capture before modifications
    - Extra Graphics: Optional creation of related geometric objects

Integration Points:
    - PointManager: Creates and manages ellipse center points
    - DependencyManager: Tracks ellipse relationships with center points
    - Canvas: Handles rendering and visual updates
    - DrawableManager: Coordinates with other geometric objects

State Management:
    - Undo/Redo: Complete state archiving for ellipse operations
    - Canvas Integration: Immediate visual updates after modifications
    - Drawables Container: Proper storage and retrieval through container system

Dependencies:
    - drawables.ellipse: Ellipse geometric object
    - utils.math_utils: Mathematical utilities (though MathUtils import is unused)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, cast

from drawables.ellipse import Ellipse
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy
from managers.dependency_removal import remove_drawable_with_dependencies

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator

class EllipseManager:
    """
    Manages ellipse drawables for a Canvas.

    This class is responsible for:
    - Creating ellipse objects with center points, radii, and rotation angles
    - Retrieving ellipse objects by coordinates, parameters, or name
    - Deleting ellipse objects with proper cleanup and redrawing
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the EllipseManager.

        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            point_manager: Manager for point drawables
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.point_manager: "PointManager" = point_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.ellipse_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Ellipse")

    def get_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float) -> Optional[Ellipse]:
        """
        Get an ellipse by its center coordinates and radii.

        Note: This method does not consider rotation angle when matching ellipses.

        Args:
            center_x (float): X-coordinate of the ellipse center
            center_y (float): Y-coordinate of the ellipse center
            radius_x (float): Horizontal radius of the ellipse
            radius_y (float): Vertical radius of the ellipse

        Returns:
            Ellipse: The matching ellipse object, or None if not found
        """
        ellipses = self.drawables.Ellipses
        for ellipse in ellipses:
            if (ellipse.center.x == center_x and
                ellipse.center.y == center_y and
                ellipse.radius_x == radius_x and
                ellipse.radius_y == radius_y):
                return ellipse
        return None

    def get_ellipse_by_name(self, name: str) -> Optional[Ellipse]:
        """
        Get an ellipse by its name.

        Args:
            name (str): The name of the ellipse

        Returns:
            Ellipse: The ellipse object with the given name, or None if not found
        """
        ellipses = self.drawables.Ellipses
        for ellipse in ellipses:
            if ellipse.name == name:
                return ellipse
        return None

    def create_ellipse(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        rotation_angle: float = 0,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Ellipse:
        """
        Create an ellipse with the specified center, radii, and rotation angle.

        Archives canvas state for undo functionality, creates center point if needed,
        and handles dependency registration and canvas redrawing.

        Args:
            center_x (float): X-coordinate of the ellipse center
            center_y (float): Y-coordinate of the ellipse center
            radius_x (float): Horizontal radius of the ellipse
            radius_y (float): Vertical radius of the ellipse
            rotation_angle (float): Rotation angle in degrees (default: 0)
            name (str): Optional name for the ellipse (default: "")
            color (str): Optional color for the ellipse
            extra_graphics (bool): Whether to create additional graphics (default: True)

        Returns:
            Ellipse: The newly created ellipse object, or existing ellipse if already present
        """
        # Archive before creation
        self.canvas.undo_redo_manager.archive()

        # Check if the ellipse already exists
        existing_ellipse = self.get_ellipse(center_x, center_y, radius_x, radius_y)
        if existing_ellipse:
            return existing_ellipse

        # Extract point name from ellipse name
        point_names: List[str] = self.name_generator.split_point_names(name, 1)

        # Create center point with the correct name
        center = self.point_manager.create_point(center_x, center_y, point_names[0], extra_graphics=False)

        # Create the ellipse (math-only)
        color_value = str(color).strip() if color is not None else ""
        if color_value:
            new_ellipse = Ellipse(
                center,
                radius_x,
                radius_y,
                rotation_angle=rotation_angle,
                color=color_value,
            )
        else:
            new_ellipse = Ellipse(center, radius_x, radius_y, rotation_angle=rotation_angle)

        # Add to drawables
        self.drawables.add(new_ellipse)

        # Register dependencies
        self.dependency_manager.analyze_drawable_for_dependencies(new_ellipse)

        # Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()

        # Draw the ellipse
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return new_ellipse

    def delete_ellipse(self, name: str) -> bool:
        """
        Delete an ellipse by its name.

        Archives canvas state for undo functionality, removes ellipse from drawables container,
        and triggers canvas redraw.

        Args:
            name (str): The name of the ellipse to delete

        Returns:
            bool: True if the ellipse was successfully deleted, False if not found
        """
        ellipse = self.get_ellipse_by_name(name)
        if not ellipse:
            return False

        # Archive before deletion
        self.canvas.undo_redo_manager.archive()

        # Delete any colored areas that depend on this ellipse (region areas, ellipse segments, etc.).
        if hasattr(self.drawable_manager, "delete_colored_areas_for_ellipse"):
            try:
                self.drawable_manager.delete_colored_areas_for_ellipse(ellipse, archive=False)
            except Exception:
                pass

        # Remove from drawables
        removed = remove_drawable_with_dependencies(
            self.drawables, self.dependency_manager, ellipse
        )

        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return bool(removed)

    def update_ellipse(
        self,
        ellipse_name: str,
        new_color: Optional[str] = None,
        new_radius_x: Optional[float] = None,
        new_radius_y: Optional[float] = None,
        new_rotation_angle: Optional[float] = None,
        new_center_x: Optional[float] = None,
        new_center_y: Optional[float] = None,
    ) -> bool:
        ellipse = self._get_ellipse_or_raise(ellipse_name)
        pending_fields = self._collect_ellipse_requested_fields(
            new_color, new_radius_x, new_radius_y, new_rotation_angle, new_center_x, new_center_y
        )
        rules = self._validate_ellipse_policy(list(pending_fields.keys()))
        self._enforce_ellipse_rules(ellipse, rules, pending_fields)
        self._validate_color_request(pending_fields, new_color)
        self._validate_radius_request("radius_x", pending_fields, new_radius_x)
        self._validate_radius_request("radius_y", pending_fields, new_radius_y)

        self.canvas.undo_redo_manager.archive()
        self._apply_ellipse_updates(
            ellipse,
            pending_fields,
            new_color,
            new_radius_x,
            new_radius_y,
            new_rotation_angle,
            new_center_x,
            new_center_y,
        )

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _get_ellipse_or_raise(self, ellipse_name: str) -> Ellipse:
        ellipse = self.get_ellipse_by_name(ellipse_name)
        if not ellipse:
            raise ValueError(f"Ellipse '{ellipse_name}' was not found.")
        return ellipse

    def _collect_ellipse_requested_fields(
        self,
        new_color: Optional[str],
        new_radius_x: Optional[float],
        new_radius_y: Optional[float],
        new_rotation_angle: Optional[float],
        new_center_x: Optional[float],
        new_center_y: Optional[float],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}

        if new_color is not None:
            pending_fields["color"] = str(new_color)

        if new_radius_x is not None:
            pending_fields["radius_x"] = "radius_x"

        if new_radius_y is not None:
            pending_fields["radius_y"] = "radius_y"

        if new_rotation_angle is not None:
            pending_fields["rotation_angle"] = "rotation_angle"

        if new_center_x is not None or new_center_y is not None:
            if new_center_x is None or new_center_y is None:
                raise ValueError("Updating an ellipse center requires both x and y coordinates.")
            pending_fields["center"] = "center"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_ellipse_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.ellipse_edit_policy:
            raise ValueError("Edit policy for ellipses is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.ellipse_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for ellipses.")
            validated_rules[field] = rule

        return validated_rules

    def _enforce_ellipse_rules(
        self,
        ellipse: Ellipse,
        rules: Dict[str, EditRule],
        pending_fields: Dict[str, str],
    ) -> None:
        if any(rule.requires_solitary for rule in rules.values()):
            if not self._is_ellipse_solitary(ellipse):
                raise ValueError(f"Ellipse '{ellipse.name}' is referenced by other drawables and cannot be edited in place.")

        if "center" in pending_fields:
            if not self._is_center_point_exclusive(ellipse):
                raise ValueError(
                    f"Ellipse '{ellipse.name}' cannot move its center because that point is referenced by other drawables."
                )

    def _is_ellipse_solitary(self, ellipse: Ellipse) -> bool:
        parents = set()
        children = set()
        if hasattr(self.dependency_manager, "get_parents"):
            raw_parents = self.dependency_manager.get_parents(ellipse)
            if raw_parents:
                parents = set(raw_parents)
        if hasattr(self.dependency_manager, "get_children"):
            raw_children = self.dependency_manager.get_children(ellipse)
            if raw_children:
                children = set(raw_children)

        parents.discard(ellipse.center)

        return not parents and not children

    def _is_center_point_exclusive(self, ellipse: Ellipse) -> bool:
        center = ellipse.center
        parents = set()
        children = set()
        if hasattr(self.dependency_manager, "get_parents"):
            raw_parents = self.dependency_manager.get_parents(center)
            if raw_parents:
                parents = set(raw_parents)
        if hasattr(self.dependency_manager, "get_children"):
            raw_children = self.dependency_manager.get_children(center)
            if raw_children:
                children = set(raw_children)

        parents.discard(ellipse)
        children.discard(ellipse)

        return not parents and not children

    def _validate_color_request(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Ellipse color cannot be empty.")

    def _validate_radius_request(
        self,
        field_name: str,
        pending_fields: Dict[str, str],
        new_value: Optional[float],
    ) -> None:
        if field_name not in pending_fields:
            return

        if new_value is None:
            raise ValueError(f"Ellipse {field_name} requires a numeric value.")

        if float(new_value) <= 0:
            raise ValueError(f"Ellipse {field_name} must be greater than zero.")

    def _apply_ellipse_updates(
        self,
        ellipse: Ellipse,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_radius_x: Optional[float],
        new_radius_y: Optional[float],
        new_rotation_angle: Optional[float],
        new_center_x: Optional[float],
        new_center_y: Optional[float],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            ellipse.update_color(str(new_color))

        if "radius_x" in pending_fields and new_radius_x is not None:
            ellipse.update_radius_x(float(new_radius_x))

        if "radius_y" in pending_fields and new_radius_y is not None:
            ellipse.update_radius_y(float(new_radius_y))

        if "rotation_angle" in pending_fields and new_rotation_angle is not None:
            ellipse.update_rotation_angle(float(new_rotation_angle))

        if "center" in pending_fields and new_center_x is not None and new_center_y is not None:
            ellipse.update_center_position(float(new_center_x), float(new_center_y))
