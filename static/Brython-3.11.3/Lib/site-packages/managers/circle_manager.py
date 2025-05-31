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

from drawables.circle import Circle
from utils.math_utils import MathUtils

class CircleManager:
    """
    Manages circle drawables for a Canvas.
    
    This class is responsible for:
    - Creating circle objects with center points and radius values
    - Retrieving circle objects by coordinates, parameters, or name
    - Deleting circle objects with proper cleanup and redrawing
    """
    
    def __init__(self, canvas, drawables_container, name_generator, dependency_manager, point_manager, drawable_manager_proxy):
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
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.drawable_manager = drawable_manager_proxy
        
    def get_circle(self, center_x, center_y, radius):
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
            if (circle.center.original_position.x == center_x and 
                circle.center.original_position.y == center_y and 
                circle.radius == radius):
                return circle
        return None
        
    def get_circle_by_name(self, name):
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
        
    def create_circle(self, center_x, center_y, radius, name="", extra_graphics=True):
        """
        Create a circle with the specified center and radius.
        
        Archives canvas state for undo functionality, creates center point if needed,
        and handles dependency registration and canvas redrawing.
        
        Args:
            center_x (float): X-coordinate of the circle center
            center_y (float): Y-coordinate of the circle center
            radius (float): Radius of the circle
            name (str): Optional name for the circle (default: "")
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
        point_names = self.name_generator.split_point_names(name, 1)
        
        # Create center point with the correct name
        center = self.point_manager.create_point(center_x, center_y, point_names[0], extra_graphics=False)
            
        # Create the circle
        new_circle = Circle(center, radius, self.canvas)
        
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
        
    def delete_circle(self, name):
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
        
        # Remove from drawables
        self.drawables.remove(circle)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 