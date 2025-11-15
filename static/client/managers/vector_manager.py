"""
MatHud Vector Management System

Manages vector creation, retrieval, and deletion for directed line segment visualization.
Handles vector operations with automatic point and segment management.

Core Responsibilities:
    - Vector Creation: Creates directed line segments from origin to tip points
    - Vector Retrieval: Lookup by origin/tip coordinates with mathematical precision
    - Vector Deletion: Safe removal with underlying segment cleanup
    - Point Integration: Automatic creation and management of endpoint points

Vector Properties:
    - Direction: Maintains explicit origin and tip point relationships
    - Visualization: Renders with directional arrows and proper styling
    - Mathematical Accuracy: Preserves vector mathematics with coordinate precision
    - Naming: Systematic naming based on endpoint coordinates or custom names

Integration Points:
    - PointManager: Creates and manages origin and tip points
    - SegmentManager: Creates underlying line segments for vector visualization
    - DrawableManager: Coordinates with other geometric objects
    - Canvas: Handles rendering and visual updates

Advanced Features:
    - Extra Graphics: Optional creation of related geometric objects
    - Dependency Management: Tracks relationships with underlying segments
    - State Preservation: Maintains vector integrity during operations
    - Cleanup Logic: Intelligent removal of unused segments during deletion

Mathematical Context:
    - Vector Mathematics: Supports standard vector operations and properties
    - Coordinate System: Works within canvas coordinate space
    - Precision Handling: Uses mathematical tolerance for coordinate matching
    - Geometric Relationships: Integrates with other geometric constructions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from drawables.vector import Vector
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator

class VectorManager:
    """
    Manages vector drawables for a Canvas.
    
    This class is responsible for:
    - Creating vector objects
    - Retrieving vector objects by various criteria
    - Deleting vector objects
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
        Initialize the VectorManager.
        
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
        
    def get_vector(self, x1: float, y1: float, x2: float, y2: float) -> Optional[Vector]:
        """
        Get a vector by its origin and tip coordinates.
        
        Searches through all existing vectors to find one that matches the specified
        origin and tip coordinates using mathematical tolerance for coordinate matching.
        
        Args:
            x1 (float): x-coordinate of the vector origin
            y1 (float): y-coordinate of the vector origin
            x2 (float): x-coordinate of the vector tip
            y2 (float): y-coordinate of the vector tip
            
        Returns:
            Vector: The matching vector object, or None if no match is found
        """
        vectors = self.drawables.Vectors
        for vector in vectors:
            if (MathUtils.point_matches_coordinates(vector.origin, x1, y1) and 
                MathUtils.point_matches_coordinates(vector.tip, x2, y2)):
                return vector
        return None

    def get_vector_by_name(self, name: str) -> Optional[Vector]:
        if not name:
            return None
        for vector in self.drawables.Vectors:
            if vector.name == name:
                return vector
        return None
        
    def create_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float, name: str = "", extra_graphics: bool = True) -> Vector:
        """
        Create a vector from origin to tip coordinates.
        
        Creates a new vector object between the specified origin and tip points.
        Automatically creates the necessary point objects if they don't exist,
        and generates a proper name if not provided.
        
        Args:
            origin_x (float): x-coordinate of the vector origin
            origin_y (float): y-coordinate of the vector origin  
            tip_x (float): x-coordinate of the vector tip
            tip_y (float): y-coordinate of the vector tip
            name (str): Optional name for the vector
            extra_graphics (bool): Whether to create additional related graphics
            
        Returns:
            Vector: The newly created or existing vector object
        """
        # Check if the vector already exists
        existing_vector = self.get_vector(origin_x, origin_y, tip_x, tip_y)
        if existing_vector:
            return existing_vector
            
        # Extract point names from vector name
        point_names: List[str] = ["", ""]
        if name:
            point_names = self.name_generator.split_point_names(name, 2)
        
        # Create or get the origin and tip points
        origin = self.point_manager.create_point(origin_x, origin_y, name=point_names[0], extra_graphics=False)
        tip = self.point_manager.create_point(tip_x, tip_y, name=point_names[1], extra_graphics=False)
        
        # Create the new vector
        new_vector = Vector(origin, tip)
        
        # Add to drawables
        self.drawables.add(new_vector)
        
        # Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()
        
        # Draw the vector
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return new_vector
        
    def delete_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> bool:
        """
        Delete a vector by its origin and tip coordinates.
        
        Finds and removes the vector that matches the specified coordinates.
        Also handles cleanup of the underlying segment if it's not used by
        other objects. Archives the state for undo functionality.
        
        Args:
            origin_x (float): x-coordinate of the vector origin
            origin_y (float): y-coordinate of the vector origin
            tip_x (float): x-coordinate of the vector tip
            tip_y (float): y-coordinate of the vector tip
            
        Returns:
            bool: True if the vector was found and deleted, False otherwise
        """
        # Find the vector that matches these coordinates
        vectors = self.drawables.Vectors
        for vector in vectors.copy():
            if (MathUtils.point_matches_coordinates(vector.origin, origin_x, origin_y) and
                MathUtils.point_matches_coordinates(vector.tip, tip_x, tip_y)):
                # Archive before deletion
                self.canvas.undo_redo_manager.archive()
                
                # Remove the vector's segment if it's not used by other objects
                if hasattr(vector, 'segment'):
                    segment = vector.segment
                    p1x = segment.point1.x
                    p1y = segment.point1.y
                    p2x = segment.point2.x
                    p2y = segment.point2.y
                    self.canvas.drawable_manager.delete_segment(p1x, p1y, p2x, p2y)
                
                # Remove the vector
                self.drawables.remove(vector)
                
                # Redraw
                if self.canvas.draw_enabled:
                    self.canvas.draw()
                    
                return True
        return False 

    def update_vector(
        self,
        vector_name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        vector = self.get_vector_by_name(vector_name)
        if not vector:
            raise ValueError(f"Vector '{vector_name}' was not found.")

        if new_color is None:
            raise ValueError("Provide at least one property to update.")

        sanitized_color = str(new_color).strip()
        if not sanitized_color:
            raise ValueError("Vector color cannot be empty.")

        self.canvas.undo_redo_manager.archive()
        vector.update_color(sanitized_color)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True