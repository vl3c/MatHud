"""
MatHud Rectangle Management System

Manages rectangle creation, retrieval, and deletion with diagonal-based construction.
Handles rectangle operations through four-corner coordinate systems and segment management.

Core Responsibilities:
    - Rectangle Creation: Constructs rectangles from diagonal point coordinates
    - Rectangle Retrieval: Lookup by diagonal points or rectangle name
    - Rectangle Deletion: Safe removal with cleanup of all four edge segments
    - Dependency Management: Tracks relationships with constituent points and segments

Geometric Construction:
    - Diagonal-Based Creation: Uses two opposite corners to define complete rectangle
    - Automatic Corner Calculation: Computes all four vertices from diagonal points
    - Edge Segment Creation: Creates all four boundary segments automatically
    - Point Management: Generates corner points with systematic naming

Advanced Features:
    - Collision Detection: Prevents creation of duplicate rectangles
    - Name Parsing: Extracts vertex names from rectangle identifiers
    - Coordinate Validation: Ensures mathematically valid rectangle construction
    - Extra Graphics: Optional creation of related geometric objects

Integration Points:
    - PointManager: Creates and manages rectangle corner points
    - SegmentManager: Creates and manages rectangle edge segments
    - DependencyManager: Tracks rectangle relationships with constituent elements
    - Canvas: Handles rendering and visual updates

Mathematical Properties:
    - Area Calculation: Supports rectangle area computation from diagonal points
    - Coordinate System: Works within canvas coordinate space
    - Precision Handling: Uses mathematical tolerance for coordinate operations
    - Geometric Validation: Ensures rectangles maintain proper geometric properties

State Management:
    - Undo/Redo: Complete state archiving for rectangle operations
    - Canvas Integration: Immediate visual updates after modifications
    - Dependency Tracking: Maintains relationships during operations
    - Cleanup Logic: Intelligent removal of constituent segments during deletion
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from drawables.rectangle import Rectangle
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from name_generator.drawable import DrawableNameGenerator

class RectangleManager:
    """
    Manages rectangle drawables for a Canvas.
    
    This class is responsible for:
    - Creating rectangle objects
    - Retrieving rectangle objects by various criteria
    - Deleting rectangle objects
    """
    
    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        segment_manager: "SegmentManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the RectangleManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            point_manager: Manager for point drawables
            segment_manager: Manager for segment drawables
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.point_manager: "PointManager" = point_manager
        self.segment_manager: "SegmentManager" = segment_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.rectangle_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Rectangle")
        
    def get_rectangle_by_diagonal_points(self, px: float, py: float, opposite_px: float, opposite_py: float) -> Optional[Rectangle]:
        """
        Get a rectangle by two diagonal points.
        
        Searches through all existing rectangles to find one that matches the specified
        diagonal corner coordinates. Calculates all four corners from the diagonal
        and verifies complete match.
        
        Args:
            px (float): x-coordinate of the first diagonal corner
            py (float): y-coordinate of the first diagonal corner
            opposite_px (float): x-coordinate of the opposite diagonal corner
            opposite_py (float): y-coordinate of the opposite diagonal corner
            
        Returns:
            Rectangle: The matching rectangle object, or None if no match is found
        """
        rectangles = self.drawables.Rectangles
        
        # Calculate the coordinates of the other two corners based on the diagonal points
        corner1: tuple[float, float] = (px, py)
        corner2: tuple[float, float] = (opposite_px, py)
        corner3: tuple[float, float] = (opposite_px, opposite_py)
        corner4: tuple[float, float] = (px, opposite_py)
        
        # Iterate over all rectangles
        for rectangle in rectangles:
            segments: List[Any] = [rectangle.segment1, rectangle.segment2, rectangle.segment3, rectangle.segment4]
            rectangle_corners: List[Tuple[float, float]] = [(segment.point1.x, segment.point1.y) for segment in segments]
            
            # Ensuring all corners are matched, considering rectangles could be defined in any direction
            if all(corner in rectangle_corners for corner in [corner1, corner2, corner3, corner4]):
                return rectangle
                
        return None
        
    def get_rectangle_by_name(self, name: str) -> Optional[Rectangle]:
        """
        Get a rectangle by its name.
        
        Searches through all existing rectangles to find one with the specified name.
        
        Args:
            name (str): The name of the rectangle to find
            
        Returns:
            Rectangle: The rectangle with the matching name, or None if not found
        """
        rectangles = self.drawables.Rectangles
        for rectangle in rectangles:
            if rectangle.name == name:
                return rectangle
        return None
        
    def create_rectangle(
        self,
        px: float,
        py: float,
        opposite_px: float,
        opposite_py: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Rectangle:
        """
        Create a rectangle with the specified diagonal points.
        
        Creates a new rectangle object from two diagonal corner coordinates.
        Automatically creates all four corner points and edge segments.
        Archives the state for undo functionality.
        
        Args:
            px (float): x-coordinate of the first diagonal corner
            py (float): y-coordinate of the first diagonal corner
            opposite_px (float): x-coordinate of the opposite diagonal corner
            opposite_py (float): y-coordinate of the opposite diagonal corner
            name (str): Optional name for the rectangle
            color (str): Optional color for the rectangle
            extra_graphics (bool): Whether to create additional related graphics
            
        Returns:
            Rectangle: The newly created or existing rectangle object
        """
        # Archive before creation
        self.canvas.undo_redo_manager.archive()
        
        # Check if the rectangle already exists
        existing_rectangle = self.get_rectangle_by_diagonal_points(px, py, opposite_px, opposite_py)
        if existing_rectangle:
            return existing_rectangle
            
        # Extract point names from rectangle name
        point_names: List[str] = ["", "", "", ""]
        if name:
            point_names = self.name_generator.split_point_names(name, 4)
        
        # Create points first with the correct names
        p1 = self.point_manager.create_point(px, py, point_names[0], extra_graphics=False)
        p2 = self.point_manager.create_point(opposite_px, py, point_names[1], extra_graphics=False)
        p3 = self.point_manager.create_point(opposite_px, opposite_py, point_names[2], extra_graphics=False)
        p4 = self.point_manager.create_point(px, opposite_py, point_names[3], extra_graphics=False)
        
        # Create segments using the points
        segment_color_kwargs: Dict[str, Any] = {}
        color_value = str(color).strip() if color is not None else ""
        if color_value:
            segment_color_kwargs["color"] = color_value
        s1 = self.segment_manager.create_segment(
            p1.x,
            p1.y,
            p2.x,
            p2.y,
            extra_graphics=False,
            **segment_color_kwargs,
        )
        s2 = self.segment_manager.create_segment(
            p2.x,
            p2.y,
            p3.x,
            p3.y,
            extra_graphics=False,
            **segment_color_kwargs,
        )
        s3 = self.segment_manager.create_segment(
            p3.x,
            p3.y,
            p4.x,
            p4.y,
            extra_graphics=False,
            **segment_color_kwargs,
        )
        s4 = self.segment_manager.create_segment(
            p4.x,
            p4.y,
            p1.x,
            p1.y,
            extra_graphics=False,
            **segment_color_kwargs,
        )
        
        # Create the rectangle
        if color_value:
            new_rectangle = Rectangle(s1, s2, s3, s4, color=color_value)
            new_rectangle.update_color(color_value)
        else:
            new_rectangle = Rectangle(s1, s2, s3, s4)
        
        # Add to drawables
        self.drawables.add(new_rectangle)
        
        # Register dependencies
        self.dependency_manager.analyze_drawable_for_dependencies(new_rectangle)
        
        # Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()
        
        # Draw the rectangle
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return new_rectangle
        
    def delete_rectangle(self, name: str) -> bool:
        """
        Delete a rectangle by its name.
        
        Finds and removes the rectangle with the specified name, along with
        all four constituent edge segments. Archives the state for undo functionality.
        
        Args:
            name (str): The name of the rectangle to delete
            
        Returns:
            bool: True if the rectangle was found and deleted, False otherwise
        """
        rectangle = self.get_rectangle_by_name(name)
        if not rectangle:
            return False
            
        # Archive before deletion
        self.canvas.undo_redo_manager.archive()
        
        # Remove from drawables
        self.drawables.remove(rectangle)
        
        # Delete all 4 segments
        self.segment_manager.delete_segment(rectangle.segment1.point1.x, rectangle.segment1.point1.y, 
                                          rectangle.segment1.point2.x, rectangle.segment1.point2.y)
        self.segment_manager.delete_segment(rectangle.segment2.point1.x, rectangle.segment2.point1.y, 
                                          rectangle.segment2.point2.x, rectangle.segment2.point2.y)
        self.segment_manager.delete_segment(rectangle.segment3.point1.x, rectangle.segment3.point1.y, 
                                          rectangle.segment3.point2.x, rectangle.segment3.point2.y)
        self.segment_manager.delete_segment(rectangle.segment4.point1.x, rectangle.segment4.point1.y, 
                                          rectangle.segment4.point2.x, rectangle.segment4.point2.y)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 

    def update_rectangle(
        self,
        rectangle_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        rectangle = self._get_rectangle_or_raise(rectangle_name)
        pending_fields = self._collect_rectangle_requested_fields(new_color)
        self._validate_rectangle_policy(list(pending_fields.keys()))
        self._validate_color_request(pending_fields, new_color)

        self.canvas.undo_redo_manager.archive()
        self._apply_rectangle_updates(rectangle, pending_fields, new_color)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _get_rectangle_or_raise(self, rectangle_name: str) -> Rectangle:
        rectangle = self.get_rectangle_by_name(rectangle_name)
        if not rectangle:
            raise ValueError(f"Rectangle '{rectangle_name}' was not found.")
        return rectangle

    def _collect_rectangle_requested_fields(
        self,
        new_color: Optional[str],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}
        if new_color is not None:
            pending_fields["color"] = str(new_color)

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_rectangle_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.rectangle_edit_policy:
            raise ValueError("Edit policy for rectangles is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.rectangle_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for rectangles.")
            validated_rules[field] = rule

        return validated_rules

    def _validate_color_request(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not str(new_color).strip()):
            raise ValueError("Rectangle color cannot be empty.")

    def _apply_rectangle_updates(
        self,
        rectangle: Rectangle,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            if hasattr(rectangle, "update_color") and callable(getattr(rectangle, "update_color")):
                rectangle.update_color(str(new_color))
            else:
                rectangle.color = str(new_color)