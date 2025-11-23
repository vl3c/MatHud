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
    - Canonicalized Geometry: All input is normalized via polygon_canonicalizer to produce an ideal rectangle
    - Diagonal-Based Creation: Uses two opposite corners to define complete rectangle when vertices are omitted
    - Vertex Construction: Supports rotated rectangles by best-fitting supplied vertices
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

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, cast

from drawables.rectangle import Rectangle
from managers.polygon_type import PolygonType
from utils.polygon_canonicalizer import (
    PolygonCanonicalizationError,
    PointLike,
    canonicalize_rectangle,
)
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
        
        try:
            target_vertices = canonicalize_rectangle(
                [(px, py), (opposite_px, opposite_py)],
                construction_mode="diagonal",
            )
        except PolygonCanonicalizationError:
            return None

        target_signature = self._vertex_signature(target_vertices)
        
        # Iterate over all rectangles
        for rectangle in rectangles:
            if self._rectangle_signature(rectangle) == target_signature:
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
        *,
        vertices: Optional[Sequence[PointLike]] = None,
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
        try:
            if vertices is not None:
                canonical_vertices = canonicalize_rectangle(
                    vertices,
                    construction_mode="vertices",
                )
            else:
                canonical_vertices = canonicalize_rectangle(
                    [(px, py), (opposite_px, opposite_py)],
                    construction_mode="diagonal",
                )
        except PolygonCanonicalizationError as exc:
            raise ValueError(str(exc)) from exc

        polygon = self.drawable_manager.polygon_manager.create_polygon(
            canonical_vertices,
            polygon_type=PolygonType.RECTANGLE,
            name=name,
            color=color,
            extra_graphics=extra_graphics,
        )
        return cast(Rectangle, polygon)
        
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
        return bool(
            self.drawable_manager.polygon_manager.delete_polygon(
                polygon_type=PolygonType.RECTANGLE,
                name=name,
            )
        )

    def update_rectangle(
        self,
        rectangle_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        return bool(
            self.drawable_manager.polygon_manager.update_polygon(
                rectangle_name,
                polygon_type=PolygonType.RECTANGLE,
                new_color=new_color,
            )
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vertex_signature(vertices: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
        """Normalize vertex coordinates for comparison with tolerance handling."""
        rounded = {(round(x, 6), round(y, 6)) for x, y in vertices}
        return tuple(sorted(rounded))

    def _rectangle_signature(self, rectangle: Rectangle) -> Tuple[Tuple[float, float], ...]:
        segments: Sequence[Any] = [
            rectangle.segment1,
            rectangle.segment2,
            rectangle.segment3,
            rectangle.segment4,
        ]
        vertices = [(segment.point1.x, segment.point1.y) for segment in segments]
        return self._vertex_signature(vertices)