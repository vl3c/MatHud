"""
MatHud Circle Management System

Manages circle creation, retrieval, and deletion operations for geometric visualization.
Handles circle operations with automatic center point management and dependency tracking.

Core Responsibilities:
    - Circle Creation: Creates circles from center coordinates and radius values
    - Circle Retrieval: Lookup by center/radius parameters or circle name
    - Circle Deletion: Safe removal with proper cleanup
    - Center Point Management: Automatic creation and tracking of circle centers

Manager Features:
    - Collision Detection: Checks for existing circles before creation
    - Dependency Tracking: Registers circle relationships with center points
    - State Archiving: Automatic undo/redo state capture before modifications
    - Extra Graphics: Optional creation of related geometric objects

Integration Points:
    - PointManager: Creates and manages circle center points
    - DependencyManager: Tracks circle relationships with center points
    - Canvas: Handles rendering and visual updates
    - DrawableManager: Coordinates with other geometric objects

State Management:
    - Undo/Redo: Complete state archiving for circle operations
    - Canvas Integration: Immediate visual updates after modifications
    - Drawables Container: Proper storage and retrieval through container system

Dependencies:
    - drawables.circle: Circle geometric object
    - Dependency Tracking: Maintains relationships with center points
    - Name Generation: Systematic naming for mathematical clarity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, cast

from drawables.circle import Circle
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator

class CircleManager:
    """
    Manages circle drawables for a Canvas.

    This class is responsible for:
    - Creating circle objects with center points and radius values
    - Retrieving circle objects by coordinates, parameters, or name
    - Deleting circle objects with proper cleanup and redrawing
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
        Initialize the CircleManager.

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
        self.circle_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Circle")

    def get_circle(self, center_x: float, center_y: float, radius: float) -> Optional[Circle]:
        """
        Get a circle by its center coordinates and radius.

        Args:
            center_x (float): X-coordinate of the circle center
            center_y (float): Y-coordinate of the circle center
            radius (float): Radius of the circle

        Returns:
            Circle: The matching circle object, or None if not found
        """
        circles = self.drawables.Circles
        for circle in circles:
            if (circle.center.x == center_x and
                circle.center.y == center_y and
                circle.radius == radius):
                return circle
        return None

    def get_circle_by_name(self, name: str) -> Optional[Circle]:
        """
        Get a circle by its name.

        Args:
            name (str): The name of the circle

        Returns:
            Circle: The circle object with the given name, or None if not found
        """
        circles = self.drawables.Circles
        for circle in circles:
            if circle.name == name:
                return circle
        return None

    def create_circle(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Circle:
        """
        Create a circle with the specified center and radius.

        Archives canvas state for undo functionality, creates center point if needed,
        and handles dependency registration and canvas redrawing.

        Args:
            center_x (float): X-coordinate of the circle center
            center_y (float): Y-coordinate of the circle center
            radius (float): Radius of the circle
            name (str): Optional name for the circle (default: "")
            color (str): Optional color for the circle
            extra_graphics (bool): Whether to create additional graphics (default: True)

        Returns:
            Circle: The newly created circle object, or existing circle if already present
        """
        # Archive before creation
        self.canvas.undo_redo_manager.archive()

        # Check if the circle already exists
        existing_circle = self.get_circle(center_x, center_y, radius)
        if existing_circle:
            return existing_circle

        # Extract point name from circle name
        point_names: List[str] = self.name_generator.split_point_names(name, 1)

        # Create center point with the correct name
        center = self.point_manager.create_point(center_x, center_y, point_names[0], extra_graphics=False)

        # Create the circle (math-only)
        color_value = str(color).strip() if color is not None else ""
        if color_value:
            new_circle = Circle(center, radius, color=color_value)
        else:
            new_circle = Circle(center, radius)

        # Add to drawables
        self.drawables.add(new_circle)

        # Register dependencies
        self.dependency_manager.analyze_drawable_for_dependencies(new_circle)

        # Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()

        # Draw the circle
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return new_circle

    def delete_circle(self, name: str) -> bool:
        """
        Delete a circle by its name.

        Archives canvas state for undo functionality, removes circle from drawables container,
        and triggers canvas redraw.

        Args:
            name (str): The name of the circle to delete

        Returns:
            bool: True if the circle was successfully deleted, False if not found
        """
        circle = self.get_circle_by_name(name)
        if not circle:
            return False

        # Archive before deletion
        self.canvas.undo_redo_manager.archive()

        if hasattr(self.drawable_manager, "arc_manager") and self.drawable_manager.arc_manager:
            self.drawable_manager.arc_manager.handle_circle_removed(name)

        # Delete any colored areas that depend on this circle (region areas, circle segments, etc.).
        if hasattr(self.drawable_manager, "delete_colored_areas_for_circle"):
            try:
                self.drawable_manager.delete_colored_areas_for_circle(circle, archive=False)
            except Exception:
                pass

        # Remove from drawables
        self.drawables.remove(circle)

        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def update_circle(
        self,
        circle_name: str,
        new_color: Optional[str] = None,
        new_center_x: Optional[float] = None,
        new_center_y: Optional[float] = None,
    ) -> bool:
        circle = self._get_circle_or_raise(circle_name)
        pending_fields = self._collect_circle_requested_fields(new_color, new_center_x, new_center_y)
        rules = self._validate_circle_policy(list(pending_fields.keys()))
        self._enforce_circle_rules(circle, rules)
        self._validate_color_request(pending_fields, new_color)

        self.canvas.undo_redo_manager.archive()
        self._apply_circle_updates(circle, pending_fields, new_color, new_center_x, new_center_y)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _get_circle_or_raise(self, circle_name: str) -> Circle:
        circle = self.get_circle_by_name(circle_name)
        if not circle:
            raise ValueError(f"Circle '{circle_name}' was not found.")
        return circle

    def _collect_circle_requested_fields(
        self,
        new_color: Optional[str],
        new_center_x: Optional[float],
        new_center_y: Optional[float],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}

        if new_color is not None:
            pending_fields["color"] = str(new_color)

        if new_center_x is not None or new_center_y is not None:
            if new_center_x is None or new_center_y is None:
                raise ValueError("Updating a circle center requires both x and y coordinates.")
            pending_fields["center"] = "center"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_circle_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.circle_edit_policy:
            raise ValueError("Edit policy for circles is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.circle_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for circles.")
            validated_rules[field] = rule

        return validated_rules

    def _enforce_circle_rules(self, circle: Circle, rules: Dict[str, EditRule]) -> None:
        if "center" in rules and rules["center"].requires_solitary:
            if not self._is_center_point_exclusive(circle):
                raise ValueError(
                    f"Circle '{circle.name}' cannot move its center because that point is referenced by other drawables."
                )

    def _is_center_point_exclusive(self, circle: Circle) -> bool:
        center = circle.center
        parents: set = set()
        children: set = set()

        if hasattr(self.dependency_manager, "get_parents"):
            raw_parents = self.dependency_manager.get_parents(center)
            if raw_parents:
                parents = set(cast(List[object], raw_parents))

        if hasattr(self.dependency_manager, "get_children"):
            raw_children = self.dependency_manager.get_children(center)
            if raw_children:
                children = set(cast(List[object], raw_children))

        parents.discard(circle)
        children.discard(circle)

        return not parents and not children

    def _validate_color_request(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Circle color cannot be empty.")

    def _apply_circle_updates(
        self,
        circle: Circle,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_center_x: Optional[float],
        new_center_y: Optional[float],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            circle.update_color(str(new_color))

        if "center" in pending_fields and new_center_x is not None and new_center_y is not None:
            circle.update_center_position(float(new_center_x), float(new_center_y))
