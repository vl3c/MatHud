"""
MatHud Segment Management System

Manages line segment creation, retrieval, and deletion with comprehensive geometric integration.
Handles segment operations including point-to-point connections, intersections, and splitting.

Core Responsibilities:
    - Segment Creation: Creates line segments between points with endpoint management
    - Segment Retrieval: Lookup by coordinates, endpoints, or segment names
    - Segment Splitting: Automatic division when new points intersect existing segments
    - Dependency Tracking: Manages relationships with triangles, rectangles, and vectors

Geometric Operations:
    - Point Integration: Automatically creates missing endpoint points
    - Intersection Detection: Identifies and handles segment intersections
    - Splitting Logic: Divides segments while preserving dependent objects
    - Collinearity Handling: Manages segments that share the same line

Advanced Features:
    - Dependency Preservation: Maintains parent-child relationships during operations
    - Connection Finding: Identifies potential new segments from point arrangements
    - Validation: Ensures mathematical validity of segment configurations
    - Name Generation: Systematic naming based on endpoint coordinates

Integration Points:
    - PointManager: Automatic point creation for segment endpoints
    - PolygonManager: Supplies edges for polygon formation
    - VectorManager: Creates underlying segments for vector visualization

State Management:
    - Undo/Redo: Complete state archiving for all segment operations
    - Canvas Updates: Immediate visual feedback for segment modifications
    - Dependency Updates: Cascading updates to dependent geometric objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, cast

from drawables.label import Label
from drawables.segment import Segment
from utils.math_utils import MathUtils
from managers.dependency_removal import remove_drawable_with_dependencies
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from drawables.point import Point
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator

class SegmentManager:
    """
    Manages segment drawables for a Canvas.

    This class is responsible for:
    - Creating segment objects
    - Retrieving segment objects by various criteria
    - Deleting segment objects
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
        Initialize the SegmentManager.

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
        self.segment_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Segment")

    def get_segment_by_coordinates(self, x1: float, y1: float, x2: float, y2: float) -> Optional[Segment]:
        """
        Get a segment by its endpoint coordinates.

        Searches through all existing segments to find one that matches the specified
        endpoint coordinates using mathematical tolerance for coordinate matching.

        Args:
            x1 (float): x-coordinate of the first endpoint
            y1 (float): y-coordinate of the first endpoint
            x2 (float): x-coordinate of the second endpoint
            y2 (float): y-coordinate of the second endpoint

        Returns:
            Segment: The matching segment object, or None if no match is found
        """
        for segment in self.drawables.Segments:
            if MathUtils.segment_matches_coordinates(segment, x1, y1, x2, y2):
                return segment
        return None

    def get_segment_by_name(self, name: str) -> Optional[Segment]:
        """
        Get a segment by its name.

        Searches through all existing segments to find one with the specified name.

        Args:
            name (str): The name of the segment to find

        Returns:
            Segment: The segment with the matching name, or None if not found
        """
        for segment in self.drawables.Segments:
            if segment.name == name:
                return segment
        return None

    def get_segment_by_points(self, p1: "Point", p2: "Point") -> Optional[Segment]:
        """
        Get a segment by its endpoint points.

        Searches for a segment that connects the two specified point objects.

        Args:
            p1 (Point): The first endpoint point
            p2 (Point): The second endpoint point

        Returns:
            Segment: The segment connecting the two points, or None if not found
        """
        return self.get_segment_by_coordinates(p1.x, p1.y, p2.x, p2.y)

    def create_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
        label_text: Optional[str] = None,
        label_visible: Optional[bool] = None,
    ) -> Segment:
        """
        Create a new segment between the specified points

        Args:
            x1, y1: Coordinates of the first endpoint
            x2, y2: Coordinates of the second endpoint
            name: Optional name for the segment
            color: Optional color for the segment
            extra_graphics: Whether to create additional graphics
            label_text: Optional label text to attach to the segment
            label_visible: Optional visibility flag for the attached label

        Returns:
            Segment: The newly created segment
        """
        # Archive before creation
        self.canvas.undo_redo_manager.archive()

        # Check if the segment already exists
        existing_segment = self.get_segment_by_coordinates(x1, y1, x2, y2)
        if existing_segment:
            return existing_segment

        # Handle point names from segment name, if provided
        point_names: List[str] = ["", ""]
        if name:
            point_names = self.name_generator.split_point_names(name, 2)

        # Create or get the endpoints with proper names
        p1 = self.point_manager.create_point(x1, y1, name=point_names[0], extra_graphics=False)
        p2 = self.point_manager.create_point(x2, y2, name=point_names[1], extra_graphics=False)

        # Create the segment (math-only; renderer uses canvas as needed)
        color_value = str(color).strip() if color is not None else ""
        sanitized_label_text = Label.validate_text(label_text or "") if label_text is not None else ""
        label_visibility = bool(label_visible) if label_visible is not None else False
        if color_value:
            segment = Segment(
                p1,
                p2,
                color=color_value,
                label_text=sanitized_label_text,
                label_visible=label_visibility,
            )
        else:
            segment = Segment(
                p1,
                p2,
                label_text=sanitized_label_text,
                label_visible=label_visibility,
            )

        # Add to drawables
        self.drawables.add(segment)

        # Register with dependency manager
        self.dependency_manager.analyze_drawable_for_dependencies(segment)

        # Handle extra graphics
        if extra_graphics:
            # Call the method on DrawableManager via the proxy
            self.drawable_manager.create_drawables_from_new_connections()

        # Draw the segment
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return segment

    def create_segment_from_points(
        self,
        p1: "Point",
        p2: "Point",
        name: str = "",
        color: Optional[str] = None,
        label_text: Optional[str] = None,
        label_visible: Optional[bool] = None,
    ) -> Segment:
        """Create a segment from existing Point objects.

        Unlike create_segment which takes coordinates, this method directly uses
        the provided Point objects, ensuring the segment references the exact
        points without any floating-point lookup issues.

        Args:
            p1: The first endpoint Point object
            p2: The second endpoint Point object
            name: Optional name for the segment
            color: Optional color for the segment
            label_text: Optional label text to attach to the segment
            label_visible: Optional visibility flag for the attached label

        Returns:
            Segment: The newly created segment object
        """
        existing_segment = self.get_segment_by_coordinates(p1.x, p1.y, p2.x, p2.y)
        if existing_segment:
            if existing_segment.point1 is p1 and existing_segment.point2 is p2:
                return existing_segment
            if existing_segment.point1 is p2 and existing_segment.point2 is p1:
                return existing_segment
            # Existing segment has same coords but different Point objects.
            # Remove stale segment to avoid duplicate rendering.
            self.drawables.remove(existing_segment)
            self.dependency_manager.remove_drawable(existing_segment)

        color_value = str(color).strip() if color is not None else ""
        sanitized_label_text = Label.validate_text(label_text or "") if label_text is not None else ""
        label_visibility = bool(label_visible) if label_visible is not None else False

        if color_value:
            segment = Segment(p1, p2, color=color_value, label_text=sanitized_label_text, label_visible=label_visibility)
        else:
            segment = Segment(p1, p2, label_text=sanitized_label_text, label_visible=label_visibility)

        self.drawables.add(segment)
        self.dependency_manager.analyze_drawable_for_dependencies(segment)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return segment

    def delete_segment(self, x1: float, y1: float, x2: float, y2: float, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """
        Delete a segment by its endpoint coordinates

        Args:
            x1, y1: Coordinates of the first endpoint
            x2, y2: Coordinates of the second endpoint
            delete_children: Whether to delete child objects
            delete_parents: Whether to delete parent objects

        Returns:
            bool: True if the segment was deleted, False otherwise
        """
        segment = self.get_segment_by_coordinates(x1, y1, x2, y2)
        if not segment:
            return False

        # Archive before deletion
        self.canvas.undo_redo_manager.archive()

        # Delete any colored areas that depend on this segment.
        # This keeps region/area drawables from becoming orphaned when a boundary segment is removed.
        if hasattr(self.drawable_manager, "delete_colored_areas_for_segment"):
            try:
                self.drawable_manager.delete_colored_areas_for_segment(segment, archive=False)
            except Exception:
                pass

        # Handle deletion of dependent angles using the dependency manager
        if segment:
            # Get all children (including angles) that depend on this segment
            dependent_children = self.dependency_manager.get_children(segment)
            for child in cast(List["Drawable"], list(dependent_children)):
                if hasattr(child, 'get_class_name') and child.get_class_name() == 'Angle':
                    print(f"SegmentManager: Segment '{segment.name}' is being deleted. Removing dependent angle '{child.name}'.")
                    if hasattr(self.drawable_manager, 'angle_manager') and self.drawable_manager.angle_manager:
                        self.drawable_manager.angle_manager.delete_angle(child.name)

        # Also notify AngleManager if a segment is about to be removed (for backward compatibility)
        if hasattr(self.drawable_manager, 'angle_manager') and self.drawable_manager.angle_manager:
            self.drawable_manager.angle_manager.handle_segment_removed(segment.name)

        # Handle dependencies by calling the internal method
        self._delete_segment_dependencies(x1, y1, x2, y2, delete_children, delete_parents)

        # Now remove the segment itself
        remove_drawable_with_dependencies(
            self.drawables, self.dependency_manager, segment
        )

        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def delete_segment_by_name(self, name: str, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """
        Delete a segment by its name.

        Finds and removes the segment with the specified name, along with
        dependent objects based on the deletion flags.

        Args:
            name (str): The name of the segment to delete
            delete_children (bool): Whether to recursively delete child segments
            delete_parents (bool): Whether to recursively delete parent segments

        Returns:
            bool: True if the segment was found and deleted, False otherwise
        """
        segment = self.get_segment_by_name(name)
        if not segment:
            return False

        x1 = segment.point1.x
        y1 = segment.point1.y
        x2 = segment.point2.x
        y2 = segment.point2.y

        result = self.delete_segment(x1, y1, x2, y2, delete_children, delete_parents)
        return result

    def update_segment(
        self,
        segment_name: str,
        new_color: Optional[str] = None,
        new_label_text: Optional[str] = None,
        new_label_visible: Optional[bool] = None,
    ) -> bool:
        segment = self._get_segment_or_raise(segment_name)
        pending_fields = self._collect_segment_requested_fields(new_color, new_label_text, new_label_visible)
        self._validate_segment_policy(list(pending_fields.keys()))

        new_color_value = self._normalize_color_request(pending_fields, new_color)
        normalized_label_text = self._normalize_label_text(pending_fields, new_label_text)
        normalized_label_visibility = self._normalize_label_visibility(pending_fields, new_label_visible)

        self.canvas.undo_redo_manager.archive()

        if new_color_value is not None:
            segment.update_color(new_color_value)
        if normalized_label_text is not None:
            segment.update_label_text(normalized_label_text)
        if normalized_label_visibility is not None:
            segment.set_label_visibility(normalized_label_visibility)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _get_segment_or_raise(self, segment_name: str) -> Segment:
        segment = self.get_segment_by_name(segment_name)
        if not segment:
            raise ValueError(f"Segment '{segment_name}' was not found.")
        return segment

    def _collect_segment_requested_fields(
        self,
        new_color: Optional[str],
        new_label_text: Optional[str],
        new_label_visible: Optional[bool],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}
        if new_color is not None:
            pending_fields["color"] = "color"
        if new_label_text is not None:
            pending_fields["label_text"] = "label_text"
        if new_label_visible is not None:
            pending_fields["label_visible"] = "label_visible"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")
        return pending_fields

    def _validate_segment_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.segment_edit_policy:
            raise ValueError("Edit policy for segments is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.segment_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for segments.")
            validated_rules[field] = rule
        return validated_rules

    def _normalize_color_request(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> Optional[str]:
        if "color" not in pending_fields:
            return None
        sanitized = str(new_color).strip() if new_color is not None else ""
        if not sanitized:
            raise ValueError("Segment color cannot be empty.")
        return sanitized

    def _normalize_label_text(
        self,
        pending_fields: Dict[str, str],
        new_label_text: Optional[str],
    ) -> Optional[str]:
        if "label_text" not in pending_fields:
            return None
        result = Label.validate_text("" if new_label_text is None else new_label_text)
        return str(result) if result is not None else None

    def _normalize_label_visibility(
        self,
        pending_fields: Dict[str, str],
        new_label_visible: Optional[bool],
    ) -> Optional[bool]:
        if "label_visible" not in pending_fields:
            return None
        return bool(new_label_visible)


    def _delete_segment_dependencies(self, x1: float, y1: float, x2: float, y2: float, delete_children: bool = True, delete_parents: bool = False) -> None:
        """
        Delete all geometric objects that depend on the specified segment.

        Handles cascading deletion of dependent objects including child segments,
        parent segments, vectors, triangles, and rectangles that contain the segment.
        Uses dependency manager for proper relationship tracking.

        Args:
            x1, y1 (float): Coordinates of the first endpoint
            x2, y2 (float): Coordinates of the second endpoint
            delete_children (bool): Whether to recursively delete child segments
            delete_parents (bool): Whether to recursively delete parent segments
        """
        segment = self.get_segment_by_coordinates(x1, y1, x2, y2)
        if not segment:
            return

        # Handle recursive deletion of children if requested
        if delete_children:
            children = self.dependency_manager.get_all_children(segment)
            # print(f"Handling deletion of {len(children)} children for segment {segment.name}") # Keep commented unless debugging
            for child in cast(List["Drawable"], list(children)): # Iterate over a copy
                if hasattr(child, 'point1') and hasattr(child, 'point2'): # Check if child is segment-like
                    # Unlink child from the current segment being processed
                    self.dependency_manager.unregister_dependency(child=child, parent=segment)

                    # Check if the child still has any other SEGMENT parents remaining
                    parents_of_child = self.dependency_manager.get_parents(child)
                    has_segment_parent = any(
                        hasattr(p, 'get_class_name') and p.get_class_name() == 'Segment'
                        for p in parents_of_child
                    )

                    # If the child no longer has any segment parents, delete it recursively
                    if not has_segment_parent:
                        self.delete_segment(
                            child.point1.x, child.point1.y,
                            child.point2.x, child.point2.y,
                            delete_children=True, delete_parents=False
                        )
                else:
                    # Handle non-segment children.
                    #
                    # Graph drawables are intentionally registered as children of segments so they can
                    # remove internal references when a segment is deleted (via DrawableDependencyManager.remove_drawable).
                    # They are not part of the segment-splitting child-segment recursion and should be ignored here.
                    child_class = child.get_class_name() if hasattr(child, "get_class_name") else ""
                    if child_class in ("Graph", "DirectedGraph", "UndirectedGraph", "Tree"):
                        continue

                    # Keep this silent for other non-segment children; they are handled by the rest of the deletion
                    # logic (vectors, triangles, rectangles, etc.).
                    continue

        # Handle recursive deletion of parents if requested
        if delete_parents:
            parents_to_delete = self.dependency_manager.get_all_parents(segment)
            print(f"Handling deletion of {len(parents_to_delete)} parents for segment {segment.name}")
            for parent in cast(List["Drawable"], list(parents_to_delete)):
                if hasattr(parent, 'point1') and hasattr(parent, 'point2'):
                        self.delete_segment(parent.point1.x, parent.point1.y,
                                          parent.point2.x, parent.point2.y,
                                          delete_children=True, delete_parents=False)
                else:
                    print(f"Warning: Parent {parent} of {segment.name} is not a segment, cannot recursively delete.")

        # Delete the segment's vectors using the proxy
        self.drawable_manager.delete_vector(x1, y1, x2, y2)
        self.drawable_manager.delete_vector(x2, y2, x1, y1)

        # Delete the rectangles that contain the segment
        rectangles = self.drawables.Rectangles
        for rectangle in rectangles.copy():
            if any(MathUtils.segment_matches_coordinates(s, x1, y1, x2, y2) for s in [rectangle.segment1, rectangle.segment2, rectangle.segment3, rectangle.segment4]):
                remove_drawable_with_dependencies(
                    self.drawables, self.dependency_manager, rectangle
                )

        # Delete the triangles that contain the segment
        triangles = self.drawables.Triangles
        for triangle in triangles.copy():
            if any(MathUtils.segment_matches_coordinates(s, x1, y1, x2, y2) for s in [triangle.segment1, triangle.segment2, triangle.segment3]):
                remove_drawable_with_dependencies(
                    self.drawables, self.dependency_manager, triangle
                )

    def _split_segments_with_point(self, x: float, y: float) -> None:
        """
        Split existing segments that pass through the specified point.

        When a new point is created, this method finds all existing segments that
        pass through the point's coordinates and splits them into two new segments.
        The original segment becomes a parent to the two new child segments through
        the dependency manager.

        Args:
            x (float): x-coordinate of the splitting point
            y (float): y-coordinate of the splitting point
        """
        segments = self.drawables.Segments
        for segment in segments.copy():
            sp1x, sp1y = segment.point1.x, segment.point1.y
            sp2x, sp2y = segment.point2.x, segment.point2.y
            # If the new point is either of the segment's endpoints, we don't need to create new segments
            if (x, y) == (sp1x, sp1y) or (x, y) == (sp2x, sp2y):
                continue
            if MathUtils.is_point_on_segment(x, y, sp1x, sp1y, sp2x, sp2y):
                # Create new segments
                segment1 = self.create_segment(x, y, sp1x, sp1y, extra_graphics=False)
                segment2 = self.create_segment(x, y, sp2x, sp2y, extra_graphics=False)

                # Register the new segments as children of the original segment that was split
                if segment1:
                    # The original segment that was split is a direct parent of the new segment
                    self.dependency_manager.register_dependency(child=segment1, parent=segment)
                    # Propagate dependency to SEGMENT ancestors of the original segment
                    for ancestor in self.dependency_manager.get_all_parents(segment):
                        if hasattr(ancestor, 'get_class_name') and ancestor.get_class_name() == 'Segment':
                            self.dependency_manager.register_dependency(child=segment1, parent=ancestor)
                if segment2:
                    # The original segment that was split is a direct parent of the new segment
                    self.dependency_manager.register_dependency(child=segment2, parent=segment)
                    # Propagate dependency to SEGMENT ancestors of the original segment
                    for ancestor in self.dependency_manager.get_all_parents(segment):
                        if hasattr(ancestor, 'get_class_name') and ancestor.get_class_name() == 'Segment':
                            self.dependency_manager.register_dependency(child=segment2, parent=ancestor)
