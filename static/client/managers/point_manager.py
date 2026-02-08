"""
MatHud Point Management System

Manages point creation, retrieval, and deletion operations for geometric visualization.
Points serve as the fundamental building blocks for all other geometric objects.

Core Responsibilities:
    - Point Creation: Creates points at specified coordinates with collision detection
    - Point Retrieval: Efficient lookup by coordinates or name
    - Dependency Management: Handles cascading deletion of dependent objects
    - Extra Graphics: Automatically splits segments and creates new connections

Geometric Integration:
    - Segment Splitting: New points automatically split intersecting segments
    - Connection Creation: Generates appropriate segments between nearby points
    - Coordinate Validation: Ensures point placement follows mathematical constraints
    - Name Generation: Provides systematic naming for mathematical clarity

Dependency Hierarchy:
    - Points are used by: Segments, Vectors, Triangles, Rectangles, Circles, Ellipses
    - Deletion cascades to: All dependent geometric objects
    - Preservation logic: Maintains parent segments when deleting child points

Canvas Integration:
    - State Management: Automatic undo/redo archiving for all point operations
    - Visual Updates: Immediate canvas redrawing after point modifications
    - Extra Graphics: Optional geometric enhancement features

Error Handling:
    - Coordinate Validation: Ensures valid mathematical coordinates
    - Existence Checking: Prevents duplicate points at same location
    - Dependency Safety: Safe deletion with preservation of essential objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, cast

from drawables.point import Point
from drawables.segment import Segment
from utils.math_utils import MathUtils
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator

class PointManager:
    """
    Manages point drawables for a Canvas.

    This class is responsible for:
    - Creating point objects
    - Retrieving point objects by various criteria
    - Deleting point objects
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the PointManager.

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
        self.point_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Point")

    def get_point(self, x: float, y: float) -> Optional[Point]:
        """
        Get a point at the specified coordinates.

        Searches through all existing points to find one that matches the specified
        coordinates using mathematical tolerance for coordinate matching.

        Args:
            x (float): x-coordinate to search for
            y (float): y-coordinate to search for

        Returns:
            Point: The matching point object, or None if no match is found
        """
        for point in self.drawables.Points:
            if MathUtils.point_matches_coordinates(point, x, y):
                return point
        return None

    def get_point_by_name(self, name: str) -> Optional[Point]:
        """
        Get a point by its name.

        Searches through all existing points to find one with the specified name.

        Args:
            name (str): The name of the point to find

        Returns:
            Point: The point with the matching name, or None if not found
        """
        for point in self.drawables.Points:
            if point.name == name:
                return point
        return None

    def create_point(
        self,
        x: float,
        y: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Point:
        """
        Create a new point at the specified coordinates

        Args:
            x: x-coordinate
            y: y-coordinate
            name: Optional name for the point
            color: Optional color for the point
            extra_graphics: Whether to create additional graphics (e.g. split segments)

        Returns:
            Point: The newly created point
        """
        # Archive before creation for undo functionality
        self.canvas.undo_redo_manager.archive()

        # Check if a point already exists at these coordinates
        existing_point = self.get_point(x, y)
        if existing_point:
            return existing_point

        # Generate a name
        name = self.name_generator.generate_point_name(name)

        # Create the new point
        color_value = str(color).strip() if color is not None else ""
        if color_value:
            new_point = Point(x=x, y=y, name=name, color=color_value)
        else:
            new_point = Point(x=x, y=y, name=name)

        # Add to drawables
        self.drawables.add(new_point)

        # Handle extra graphics - splits segments and creates connections
        if extra_graphics:
            # Call the method on the SegmentManager via the proxy
            self.drawable_manager.segment_manager._split_segments_with_point(x, y)
            # Call the method on DrawableManager via the proxy
            self.drawable_manager.create_drawables_from_new_connections()

        # Draw the point if draw_enabled is True
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return new_point

    def delete_point(self, x: float, y: float) -> bool:
        """
        Delete a point at the specified coordinates.

        Finds and removes the point at the given coordinates, along with all
        dependent geometric objects. Archives the state for undo functionality
        and handles cascading deletion of dependent objects.

        Args:
            x (float): x-coordinate of the point to delete
            y (float): y-coordinate of the point to delete

        Returns:
            bool: True if the point was found and deleted, False otherwise
        """
        point = self.get_point(x, y)
        if not point:
            return False

        # Archive before deletion
        self.canvas.undo_redo_manager.archive()

        # Delete dependencies first
        self._delete_point_dependencies(x, y)

        # Now remove the point itself
        self.drawables.remove(point)

        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def delete_point_by_name(self, name: str) -> bool:
        """
        Delete a point by its name.

        Finds and removes the point with the specified name, along with all
        dependent geometric objects.

        Args:
            name (str): The name of the point to delete

        Returns:
            bool: True if the point was found and deleted, False otherwise
        """
        point = self.get_point_by_name(name)
        if not point:
            return False

        return self.delete_point(point.x, point.y)

    def _delete_point_dependencies(self, x: float, y: float) -> None:
        """
        Delete all geometric objects that depend on the specified point.

        Handles cascading deletion of dependent objects including segments, vectors,
        circles, ellipses, triangles, and rectangles that contain the point as an endpoint
        or center. Uses dependency manager to handle angle deletion properly.

        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point
        """
        # Get the point to be deleted for dependency checking
        point_to_delete = self.get_point(x, y)

        # Handle deletion of dependent objects using the dependency manager
        if point_to_delete:
            # Get all children (including angles and circle arcs) that depend on this point
            dependent_children = self.dependency_manager.get_children(point_to_delete)
            for child in cast(List["Drawable"], list(dependent_children)):
                if hasattr(child, 'get_class_name'):
                    class_name = child.get_class_name()
                else:
                    class_name = child.__class__.__name__

                if class_name == 'Angle':
                    print(f"PointManager: Point at ({x}, {y}) is being deleted. Removing dependent angle '{child.name}'.")
                    if hasattr(self.drawable_manager, 'angle_manager') and self.drawable_manager.angle_manager:
                        self.drawable_manager.angle_manager.delete_angle(child.name)
                if class_name == 'CircleArc':
                    print(f"PointManager: Point at ({x}, {y}) is being deleted. Removing dependent circle arc '{child.name}'.")
                    if hasattr(self.drawable_manager, 'arc_manager') and self.drawable_manager.arc_manager:
                        self.drawable_manager.arc_manager.delete_circle_arc(child.name)

        # Delete the segments that contain the point
        rectangles = self.drawables.Rectangles
        for rectangle in rectangles.copy():
            if any(MathUtils.segment_has_end_point(segment, x, y) for segment in [rectangle.segment1, rectangle.segment2, rectangle.segment3, rectangle.segment4]):
                self.drawables.remove(rectangle)

        # Delete the triangles that contain the point
        triangles = self.drawables.Triangles
        for triangle in triangles.copy():
            if any(MathUtils.segment_has_end_point(segment, x, y) for segment in [triangle.segment1, triangle.segment2, triangle.segment3]):
                self.drawables.remove(triangle)

        # Collect all segments that contain the point
        segments_to_delete: List[Segment] = []
        segments = self.drawables.Segments
        for segment in segments.copy():
            if MathUtils.segment_has_end_point(segment, x, y):
                segments_to_delete.append(segment)

        # Create a set of all parent segments that should be preserved
        segments_to_preserve: set[Segment] = set()
        for segment in segments_to_delete:
            # Get all parents of this segment using the manager's public method
            parents = self.dependency_manager.get_parents(segment)
            for parent in parents:
                if isinstance(parent, Segment) and not MathUtils.segment_has_end_point(parent, x, y):
                    segments_to_preserve.add(parent)

        # Delete segments that contain the point, but not if they're in the preserve list
        for segment in segments_to_delete:
            if segment not in segments_to_preserve:
                p1x = segment.point1.x
                p1y = segment.point1.y
                p2x = segment.point2.x
                p2y = segment.point2.y
                # Use the proxy to call delete_segment
                self.drawable_manager.delete_segment(p1x, p1y, p2x, p2y,
                                   delete_children=True, delete_parents=False)

        # Delete the vectors that contain the point
        vectors = self.drawables.Vectors
        for vector in vectors.copy():
            if MathUtils.segment_has_end_point(vector.segment, x, y):
                origin_x = vector.segment.point1.x
                origin_y = vector.segment.point1.y
                tip_x = vector.segment.point2.x
                tip_y = vector.segment.point2.y
                # Use the proxy to call delete_vector
                self.drawable_manager.delete_vector(origin_x, origin_y, tip_x, tip_y)

        # Delete the circles that contain the point
        circles = self.drawables.Circles
        for circle in circles.copy():
            if MathUtils.point_matches_coordinates(circle.center, x, y):
                # Use the proxy to call delete_circle
                self.drawable_manager.delete_circle(circle.name)

        # Delete the ellipses that contain the point
        ellipses = self.drawables.Ellipses
        for ellipse in ellipses.copy():
            if MathUtils.point_matches_coordinates(ellipse.center, x, y):
                # Use the proxy to call delete_ellipse
                self.drawable_manager.delete_ellipse(ellipse.name)

    def _is_point_solitary(self, point: Point) -> bool:
        """Return True when the point has no dependency relationships."""
        parents = set()
        children = set()

        if hasattr(self.dependency_manager, "get_parents"):
            parents = cast(set, self.dependency_manager.get_parents(point))
        if hasattr(self.dependency_manager, "get_children"):
            children = cast(set, self.dependency_manager.get_children(point))

        return not parents and not children

    def _can_bypass_solitary_rules(
        self,
        point: Point,
        rules: Dict[str, EditRule],
        pending_fields: Dict[str, str],
    ) -> bool:
        if set(pending_fields.keys()) != {"name"}:
            return False

        if not self._is_point_only_circle_center(point):
            return False

        return True

    def _is_point_only_circle_center(self, point: Point) -> bool:
        circles = [
            circle
            for circle in getattr(self.drawables, "Circles", [])
            if getattr(circle, "center", None) is point
        ]
        if not circles:
            return False

        allowed = set(circles)

        parents = set()
        children = set()
        if hasattr(self.dependency_manager, "get_parents"):
            parents = cast(set, self.dependency_manager.get_parents(point))
        if hasattr(self.dependency_manager, "get_children"):
            children = cast(set, self.dependency_manager.get_children(point))

        return parents.issubset(allowed) and children.issubset(allowed)

    def _rename_dependent_rotational_drawables(self, point: Point) -> None:
        for circle in getattr(self.drawables, "Circles", []):
            if getattr(circle, "center", None) is point and hasattr(circle, "regenerate_name"):
                circle.regenerate_name()
        for ellipse in getattr(self.drawables, "Ellipses", []):
            if getattr(ellipse, "center", None) is point and hasattr(ellipse, "regenerate_name"):
                ellipse.regenerate_name()

    def _validate_point_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        """Ensure every requested field is allowed by the policy definition."""
        if not self.point_edit_policy:
            raise ValueError("Edit policy for points is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.point_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for points.")
            validated_rules[field] = rule

        return validated_rules

    def update_point(
        self,
        point_name: str,
        new_name: Optional[str] = None,
        new_x: Optional[float] = None,
        new_y: Optional[float] = None,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update selectable properties of a solitary point."""
        point = self.get_point_by_name(point_name)
        if not point:
            raise ValueError(f"Point '{point_name}' was not found.")

        pending_fields: Dict[str, str] = {}
        if new_name is not None:
            pending_fields["name"] = new_name
        if new_color is not None:
            pending_fields["color"] = new_color
        if new_x is not None or new_y is not None:
            if new_x is None or new_y is None:
                raise ValueError("Updating a point position requires both x and y coordinates.")
            pending_fields["position"] = "position"

        if "position" in pending_fields and self._point_is_locked_center(point):
            raise ValueError(
                f"Point '{point_name}' is the center of a circle or ellipse and must be moved via the appropriate update command."
            )

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        rules = self._validate_point_policy(list(pending_fields.keys()))
        if any(rule.requires_solitary for rule in rules.values()):
            if not self._is_point_solitary(point):
                if not self._can_bypass_solitary_rules(point, rules, pending_fields):
                    raise ValueError(
                        f"Point '{point_name}' is referenced by other drawables and cannot be edited in place."
                    )

        filtered_name = self._compute_updated_name(point, pending_fields)
        new_coordinates = self._compute_updated_coordinates(point, pending_fields, new_x, new_y)
        self._validate_color_request(pending_fields, new_color)

        # Archive once the payload has been validated
        self.canvas.undo_redo_manager.archive()

        if filtered_name is not None:
            point.update_name(filtered_name)
            self._rename_dependent_rotational_drawables(point)

        if new_coordinates is not None:
            x_val, y_val = new_coordinates
            point.update_position(x_val, y_val)

        if "color" in pending_fields and new_color is not None:
            point.update_color(str(new_color))

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _point_is_locked_center(self, point: Point) -> bool:
        for circle in getattr(self.drawables, "Circles", []):
            if getattr(circle, "center", None) is point:
                return True
        for ellipse in getattr(self.drawables, "Ellipses", []):
            if getattr(ellipse, "center", None) is point:
                return True
        return False

    def _compute_updated_name(
        self, original_point: Point, pending_fields: Dict[str, str]
    ) -> Optional[str]:
        if "name" not in pending_fields:
            return None

        candidate = pending_fields["name"]
        filtered_candidate: str = str(
            self.name_generator.filter_string(candidate)
            if hasattr(self.name_generator, "filter_string")
            else candidate
        )
        filtered_candidate = filtered_candidate.strip()
        if not filtered_candidate:
            raise ValueError("Point name cannot be empty.")

        existing_point = self.get_point_by_name(filtered_candidate)
        if existing_point and existing_point is not original_point:
            raise ValueError(f"Another point named '{filtered_candidate}' already exists.")

        return filtered_candidate

    def _compute_updated_coordinates(
        self,
        original_point: Point,
        pending_fields: Dict[str, str],
        new_x: Optional[float],
        new_y: Optional[float],
    ) -> Optional[tuple[float, float]]:
        if "position" not in pending_fields:
            return None

        x_val = float(cast(float, new_x))
        y_val = float(cast(float, new_y))
        coordinate_conflict = self.get_point(x_val, y_val)
        if coordinate_conflict and coordinate_conflict is not original_point:
            raise ValueError(f"Another point already exists at ({x_val}, {y_val}).")

        return (x_val, y_val)

    def _validate_color_request(
        self, pending_fields: Dict[str, str], new_color: Optional[str]
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Point color cannot be empty.")
