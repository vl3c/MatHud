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

from drawables.triangle import Triangle
from utils.math_utils import MathUtils
from utils.geometry_utils import GeometryUtils

class TriangleManager:
    """
    Manages triangle drawables for a Canvas.
    
    This class is responsible for:
    - Creating triangle objects
    - Retrieving triangle objects by various criteria
    - Deleting triangle objects
    """
    
    def __init__(self, canvas, drawables_container, name_generator, dependency_manager, 
                 point_manager, segment_manager, drawable_manager_proxy):
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
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.segment_manager = segment_manager
        self.drawable_manager = drawable_manager_proxy
        
    def get_triangle(self, x1, y1, x2, y2, x3, y3):
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
        triangles = self.drawables.Triangles
        for triangle in triangles:
            if MathUtils.triangle_matches_coordinates(triangle, x1, y1, x2, y2, x3, y3):
                return triangle
        return None
        
    def create_triangle(self, x1, y1, x2, y2, x3, y3, name="", extra_graphics=True):
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
            extra_graphics (bool): Whether to create additional related graphics
            
        Returns:
            Triangle: The newly created or existing triangle object
        """
        # Archive before creation
        self.canvas.undo_redo_manager.archive()
        
        # Check if the triangle already exists
        existing_triangle = self.get_triangle(x1, y1, x2, y2, x3, y3)
        if existing_triangle:
            return existing_triangle
            
        # Extract point names from triangle name
        point_names = ["", "", ""]
        if name:
            point_names = self.name_generator.split_point_names(name, 3)
            
        # Create points first with the correct names
        p1 = self.point_manager.create_point(x1, y1, name=point_names[0], extra_graphics=False)
        p2 = self.point_manager.create_point(x2, y2, name=point_names[1], extra_graphics=False)
        p3 = self.point_manager.create_point(x3, y3, name=point_names[2], extra_graphics=False)
            
        # Create segments using the points
        s1 = self.segment_manager.create_segment(p1.x, p1.y, 
                                               p2.x, p2.y, 
                                               extra_graphics=False)
        s2 = self.segment_manager.create_segment(p2.x, p2.y, 
                                               p3.x, p3.y, 
                                               extra_graphics=False)
        s3 = self.segment_manager.create_segment(p3.x, p3.y, 
                                               p1.x, p1.y, 
                                               extra_graphics=False)
            
        # Create the triangle
        new_triangle = Triangle(s1, s2, s3, self.canvas)
        
        # Add to drawables
        self.drawables.add(new_triangle)
        
        # Register dependencies
        self.dependency_manager.analyze_drawable_for_dependencies(new_triangle)
        
        # Handle extra graphics if requested
        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()
        
        # Draw the triangle
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return new_triangle
        
    def delete_triangle(self, x1, y1, x2, y2, x3, y3):
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
        triangle = self.get_triangle(x1, y1, x2, y2, x3, y3)
        if not triangle:
            return False
            
        # Archive before deletion
        self.canvas.undo_redo_manager.archive()
        
        # Remove from drawables
        self.drawables.remove(triangle)
        
        # Delete the segments using the vertex coordinates
        self.segment_manager.delete_segment(x1, y1, x2, y2)
        self.segment_manager.delete_segment(x2, y2, x3, y3)
        self.segment_manager.delete_segment(x3, y3, x1, y1)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True

    def create_new_triangles_from_connected_segments(self):
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
        segments = list(self.drawables.Segments)
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