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

from drawables.point import Point
from drawables.segment import Segment
from utils.math_utils import MathUtils

class PointManager:
    """
    Manages point drawables for a Canvas.
    
    This class is responsible for:
    - Creating point objects
    - Retrieving point objects by various criteria
    - Deleting point objects
    """
    
    def __init__(self, canvas, drawables_container, name_generator, dependency_manager, drawable_manager_proxy):
        """
        Initialize the PointManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.drawable_manager = drawable_manager_proxy
        
    def get_point(self, x, y):
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
        
    def get_point_by_name(self, name):
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
        
    def create_point(self, x, y, name="", extra_graphics=True):
        """
        Create a new point at the specified coordinates
        
        Args:
            x: x-coordinate
            y: y-coordinate
            name: Optional name for the point
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
        new_point = Point(x=x, y=y, canvas=self.canvas, name=name)
        
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
        
    def delete_point(self, x, y):
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
        
    def delete_point_by_name(self, name):
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
            
        return self.delete_point(point.original_position.x, point.original_position.y)

    def _delete_point_dependencies(self, x, y):
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
            # Get all children (including angles) that depend on this point
            dependent_children = self.dependency_manager.get_children(point_to_delete)
            for child in list(dependent_children):
                if hasattr(child, 'get_class_name') and child.get_class_name() == 'Angle':
                    print(f"PointManager: Point at ({x}, {y}) is being deleted. Removing dependent angle '{child.name}'.")
                    if hasattr(self.drawable_manager, 'angle_manager') and self.drawable_manager.angle_manager:
                        self.drawable_manager.angle_manager.delete_angle(child.name)
        
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
        segments_to_delete = []
        segments = self.drawables.Segments
        for segment in segments.copy():
            if MathUtils.segment_has_end_point(segment, x, y):
                segments_to_delete.append(segment)
        
        # Create a set of all parent segments that should be preserved
        segments_to_preserve = set()
        for segment in segments_to_delete:
            # Get all parents of this segment using the manager's public method
            parents = self.dependency_manager.get_parents(segment)
            for parent in parents:
                if isinstance(parent, Segment) and not MathUtils.segment_has_end_point(parent, x, y):
                    segments_to_preserve.add(parent)
        
        # Delete segments that contain the point, but not if they're in the preserve list
        for segment in segments_to_delete:
            if segment not in segments_to_preserve:
                p1x = segment.point1.original_position.x
                p1y = segment.point1.original_position.y
                p2x = segment.point2.original_position.x
                p2y = segment.point2.original_position.y
                # Use the proxy to call delete_segment
                self.drawable_manager.delete_segment(p1x, p1y, p2x, p2y, 
                                   delete_children=True, delete_parents=False)
        
        # Delete the vectors that contain the point
        vectors = self.drawables.Vectors
        for vector in vectors.copy():
            if MathUtils.segment_has_end_point(vector.segment, x, y):
                origin_x = vector.segment.point1.original_position.x
                origin_y = vector.segment.point1.original_position.y
                tip_x = vector.segment.point2.original_position.x
                tip_y = vector.segment.point2.original_position.y
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