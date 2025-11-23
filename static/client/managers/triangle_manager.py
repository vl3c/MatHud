"""
MatHud Triangle Management System

Manages triangle creation, retrieval, and deletion with comprehensive geometric validation.
Handles triangle construction from three vertices with automatic segment and point management.

Core Responsibilities:
    - Triangle Creation: Constructs triangles from three vertex coordinates
    - Triangle Retrieval: Efficient lookup by vertex coordinates
    - Triangle Deletion: Safe removal with cleanup of constituent segments
    - Dependency Management: Tracks relationships with points and segments

Geometric Validation:
    - Collinearity Detection: Prevents creation of degenerate triangles
    - Coordinate Precision: Uses mathematical tolerance for vertex matching
    - Connectivity Analysis: Automatically detects triangles from segment arrangements
    - Graph Theory Integration: Leverages connectivity algorithms for triangle identification

Advanced Features:
    - Automatic Triangle Detection: Identifies triangles from connected segments
    - Smart Construction: Creates missing points and segments as needed
    - Name Parsing: Extracts vertex names from triangle identifiers
    - Extra Graphics: Optional creation of related geometric objects

Integration Points:
    - PointManager: Creates and manages triangle vertices
    - SegmentManager: Creates and manages triangle edges
    - GeometryUtils: Graph connectivity and geometric analysis
    - DependencyManager: Tracks triangle relationships with constituent elements

Mathematical Properties:
    - Area Calculation: Supports triangle area computation
    - Centroid Finding: Enables geometric center calculations
    - Orientation Testing: Determines vertex ordering (clockwise/counterclockwise)
    - Validity Checking: Ensures triangles meet geometric requirements

State Management:
    - Undo/Redo: Complete state archiving for triangle operations
    - Canvas Integration: Immediate visual updates after modifications
    - Dependency Tracking: Maintains relationships during operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, cast

from drawables.triangle import Triangle
from managers.polygon_type import PolygonType
from utils.math_utils import MathUtils
from utils.geometry_utils import GeometryUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from name_generator.drawable import DrawableNameGenerator

class TriangleManager:
    """
    Manages triangle drawables for a Canvas.
    
    This class is responsible for:
    - Creating triangle objects
    - Retrieving triangle objects by various criteria
    - Deleting triangle objects
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
        Initialize the TriangleManager.
        
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
        
    def get_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional[Triangle]:
        """
        Get a triangle by its vertex coordinates.
        
        Searches through all existing triangles to find one that matches the specified
        vertex coordinates using mathematical tolerance for coordinate matching.
        
        Args:
            x1 (float): x-coordinate of the first vertex
            y1 (float): y-coordinate of the first vertex
            x2 (float): x-coordinate of the second vertex
            y2 (float): y-coordinate of the second vertex
            x3 (float): x-coordinate of the third vertex
            y3 (float): y-coordinate of the third vertex
            
        Returns:
            Triangle: The matching triangle object, or None if no match is found
        """
        polygon = self.drawable_manager.polygon_manager.get_polygon_by_vertices(
            [(x1, y1), (x2, y2), (x3, y3)],
            polygon_type=PolygonType.TRIANGLE,
        )
        return cast(Optional[Triangle], polygon)

    def get_triangle_by_name(self, name: str) -> Optional[Triangle]:
        if not name:
            return None
        polygon = self.drawable_manager.polygon_manager.get_polygon_by_name(
            name,
            polygon_type=PolygonType.TRIANGLE,
        )
        return cast(Optional[Triangle], polygon)
        
    def create_triangle(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Triangle:
        """
        Create a triangle from three vertex coordinates.
        
        Creates a new triangle object from the specified vertex coordinates.
        Automatically creates the necessary point and segment objects if they don't exist.
        Validates that the three points are not collinear to ensure a valid triangle.
        
        Args:
            x1 (float): x-coordinate of the first vertex
            y1 (float): y-coordinate of the first vertex
            x2 (float): x-coordinate of the second vertex
            y2 (float): y-coordinate of the second vertex
            x3 (float): x-coordinate of the third vertex
            y3 (float): y-coordinate of the third vertex
            name (str): Optional name for the triangle
            color (str): Optional color for the triangle
            extra_graphics (bool): Whether to create additional related graphics
            
        Returns:
            Triangle: The newly created or existing triangle object
        """
        polygon = self.drawable_manager.polygon_manager.create_polygon(
            [(x1, y1), (x2, y2), (x3, y3)],
            polygon_type=PolygonType.TRIANGLE,
            name=name,
            color=color,
            extra_graphics=extra_graphics,
        )
        return cast(Triangle, polygon)
        
    def delete_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
        """
        Delete a triangle by its vertex coordinates.
        
        Finds and removes the triangle with the specified vertex coordinates.
        Also removes the constituent segments that form the triangle edges.
        Archives the state for undo functionality.
        
        Args:
            x1 (float): x-coordinate of the first vertex
            y1 (float): y-coordinate of the first vertex
            x2 (float): x-coordinate of the second vertex
            y2 (float): y-coordinate of the second vertex
            x3 (float): x-coordinate of the third vertex
            y3 (float): y-coordinate of the third vertex
            
        Returns:
            bool: True if the triangle was found and deleted, False otherwise
        """
        return bool(
            self.drawable_manager.polygon_manager.delete_polygon(
                polygon_type=PolygonType.TRIANGLE,
                vertices=[(x1, y1), (x2, y2), (x3, y3)],
            )
        )

    def update_triangle(
        self,
        triangle_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        return bool(
            self.drawable_manager.polygon_manager.update_polygon(
                triangle_name,
                polygon_type=PolygonType.TRIANGLE,
                new_color=new_color,
            )
        )

    def create_new_triangles_from_connected_segments(self) -> None:
        """
        Automatically detect and create triangles from existing connected segments.
        
        Analyzes all existing segments to identify sets of three segments that form
        closed triangular paths. Uses graph theory to verify full connectivity and
        creates triangle objects for valid configurations. Skips degenerate triangles
        where points are collinear.
        
        The method examines all possible combinations of three segments and:
        1. Extracts unique endpoint names from each trio
        2. Verifies the three segments form a fully connected graph
        3. Checks for collinearity to avoid degenerate triangles
        4. Creates triangle objects for valid configurations
        
        Note: This method is typically called during extra graphics operations
        to automatically detect triangular structures that emerge from point
        and segment creation.
        """
        segments: List[Any] = cast(List[Any], list(self.drawables.Segments))
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                for k in range(j+1, len(segments)):
                    s1, s2, s3 = segments[i], segments[j], segments[k]
                    # Get the unique points in the three segments
                    # Assuming GeometryUtils is imported
                    unique_points = GeometryUtils.get_unique_point_names_from_segments([s1, s2, s3])
                    # If the segments share exactly three unique points, check if they form a triangle
                    if len(unique_points) == 3 and GeometryUtils.is_fully_connected_graph(unique_points, self.drawables.Segments):
                        # Use point_manager to get points
                        p1 = self.point_manager.get_point_by_name(unique_points[0])
                        p2 = self.point_manager.get_point_by_name(unique_points[1])
                        p3 = self.point_manager.get_point_by_name(unique_points[2])
                        # Call self.create_triangle (already in TriangleManager)
                        if not all([p1, p2, p3]):
                            print(f"Warning: Could not find all points {unique_points} to create triangle.")
                            continue
                            
                        # Skip if points are collinear (degenerate triangle)
                        if MathUtils.points_orientation(p1.x, p1.y,
                                                      p2.x, p2.y,
                                                      p3.x, p3.y) == 0:
                            continue
                            
                        self.create_triangle(p1.x, p1.y,
                                           p2.x, p2.y,
                                           p3.x, p3.y,
                                           extra_graphics=False)