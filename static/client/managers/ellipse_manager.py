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

from typing import TYPE_CHECKING, List, Optional

from drawables.ellipse import Ellipse
from utils.math_utils import MathUtils

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
        
    def create_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float, rotation_angle: float = 0, name: str = "", extra_graphics: bool = True) -> Ellipse:
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
        
        # Remove from drawables
        self.drawables.remove(ellipse)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 