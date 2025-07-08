"""
MatHud Mathematical Canvas System

Core SVG-based canvas for interactive mathematical visualization and geometric construction.
Serves as the central coordinator for all drawable objects, coordinate systems, and user interactions.

Key Features:
    - SVG viewport management with zoom/pan capabilities
    - Geometric object creation (points, segments, vectors, shapes, functions)
    - Coordinate system with Cartesian grid visualization
    - Undo/redo state management for user actions
    - Mathematical computation history tracking
    - Colored area visualization between objects
    - Angle measurement and display
    - Object transformations (translate, rotate)

Architecture:
    - Canvas: Central coordinator and state manager
    - DrawableManager: Handles all geometric object lifecycle
    - UndoRedoManager: Provides state archiving and restoration
    - TransformationsManager: Manages object positioning and rotation
    - Cartesian2Axis: Coordinate system visualization

Dependencies:
    - browser.document: DOM manipulation for SVG rendering
    - geometry: Geometric object definitions (Point, Position, etc.)
    - cartesian_system_2axis: Coordinate grid system
    - coordinate_mapper: Coordinate transformation management
    - utils.*: Mathematical, styling, and geometry utilities
    - managers.*: Specialized management components
"""

from browser import document
from geometry import Point
from cartesian_system_2axis import Cartesian2Axis
from coordinate_mapper import CoordinateMapper
from utils.math_utils import MathUtils
from utils.style_utils import StyleUtils
from utils.geometry_utils import GeometryUtils
from utils.computation_utils import ComputationUtils
from managers.undo_redo_manager import UndoRedoManager
from managers.drawable_manager import DrawableManager
from managers.transformations_manager import TransformationsManager


class Canvas:
    """Central mathematical visualization canvas coordinating all drawable objects and interactions.
    
    Manages the SVG viewport, coordinate system, geometric objects, and user interactions.
    Provides the main interface for creating, manipulating, and visualizing mathematical content.
    
    Attributes:
        width (float): Canvas viewport width in pixels
        height (float): Canvas viewport height in pixels
        coordinate_mapper (CoordinateMapper): Centralized coordinate transformation service
        computations (list): History of mathematical computations performed
        cartesian2axis (Cartesian2Axis): Coordinate grid system
        drawable_manager (DrawableManager): Manages all geometric objects
        undo_redo_manager (UndoRedoManager): Handles state archiving/restoration
        transformations_manager (TransformationsManager): Manages object transformations
        
    Legacy Properties (delegated to coordinate_mapper):
        center (Position): Current viewport center point
        scale_factor (float): Current zoom level (1.0 = normal)
        offset (Position): Current pan offset
        zoom_point (Position): Center point for zoom operations
        zoom_direction (int): Current zoom direction (-1=in, 1=out, 0=none)
        zoom_step (float): Zoom increment per step
    """
    def __init__(self, width, height, draw_enabled=True):
        """Initialize the mathematical canvas with specified dimensions.
        
        Sets up the coordinate system, managers, and initial state for mathematical visualization.
        
        Args:
            width (float): Canvas viewport width in pixels
            height (float): Canvas viewport height in pixels
            draw_enabled (bool): Whether to enable immediate drawing operations
        """
        self.computations = []  # Store computation history
        self.width = width
        self.height = height
        
        # Initialize CoordinateMapper for coordinate transformation management
        self.coordinate_mapper = CoordinateMapper(width, height)
        
        # Legacy properties for backward compatibility - delegate to coordinate_mapper
        self.dragging = False
        self.draw_enabled = draw_enabled
        
        # Initialize coordinate system and managers
        self.cartesian2axis = Cartesian2Axis(self)
        self.cartesian2axis.origin = Point(x=0, y=0, canvas=self, name="o") # initializing a point requires a cartesian2axis object
        
        # Add managers
        self.undo_redo_manager = UndoRedoManager(self)
        self.drawable_manager = DrawableManager(self)
        self.transformations_manager = TransformationsManager(self)
        
        if self.draw_enabled:
            self.cartesian2axis.draw()

    def add_drawable(self, drawable):
        drawable.canvas = self  # Set the drawable's canvas reference
        self.drawable_manager.drawables.add(drawable)

    def draw(self, apply_zoom=False):
        if not self.draw_enabled:
            return
        svg_container = document["math-svg"]
        svg_container.clear()
        self._draw_cartesian(apply_zoom)
        
        # Draw all drawables
        for drawable in self.drawable_manager.get_drawables():
            drawable.pan()
            if apply_zoom:
                drawable.zoom()
            drawable.draw()  # Draw each drawable
        
        # Reset pan offset after drawing using CoordinateMapper
        self.coordinate_mapper.reset_pan()

    def _draw_cartesian(self, apply_zoom=False):
        self.cartesian2axis.pan()
        if apply_zoom:
            self.cartesian2axis.zoom()
        self.cartesian2axis.draw()

    def _fix_drawable_canvas_references(self):
        for drawable in self.drawable_manager.get_drawables():
            drawable.canvas = self

    def clear(self):
        """Clear all drawables"""
        self.archive()
        self.drawable_manager.drawables.clear()
        
        # Reset the name generator state
        if hasattr(self.drawable_manager, 'name_generator') and hasattr(self.drawable_manager.name_generator, 'reset_state'):
            self.drawable_manager.name_generator.reset_state()
        
        self.reset()

    def reset(self):
        """Reset the canvas to its initial state"""
        # Reset coordinate transformations using CoordinateMapper
        self.coordinate_mapper.reset_transformations()
        
        # Reset other canvas state
        self.dragging = False
        
        # Reset cartesian system and drawables
        self.cartesian2axis.reset()
        for drawable in self.get_drawables():
            drawable.reset()
        self.draw()

    def archive(self):
        """Archive the current state for undo functionality"""
        self.undo_redo_manager.archive()

    def undo(self):
        """Restores the last archived state from the undo stack"""
        return self.undo_redo_manager.undo()

    def redo(self):
        """Restores the last undone state from the redo stack"""
        return self.undo_redo_manager.redo()

    def get_drawables(self):
        """Get all drawables as a flat list"""
        return self.drawable_manager.get_drawables()

    def get_drawables_state(self):
        return self.drawable_manager.drawables.get_state()

    def get_drawables_by_class_name(self, class_name):
        """Get drawables of a specific class name"""
        return self.drawable_manager.drawables.get_by_class_name(class_name)

    def get_cartesian2axis_state(self):
        return self.cartesian2axis.get_state()

    def get_canvas_state(self):
        state = self.get_drawables_state()
        cartesian_state = self.get_cartesian2axis_state()
        if cartesian_state is not None:
            state.update(cartesian_state)
        if self.computations:  # Add computations to state if they exist
            state["computations"] = self.computations
        return state

    def get_point(self, x, y):
        """Get a point at the specified coordinates"""
        return self.drawable_manager.get_point(x, y)

    def get_point_by_name(self, name):
        """Get a point by its name"""
        return self.drawable_manager.get_point_by_name(name)

    def create_point(self, x, y, name="", extra_graphics=True):
        """Create a point at the specified coordinates"""
        return self.drawable_manager.create_point(x, y, name, extra_graphics)

    def delete_point(self, x, y):
        """Delete a point at the specified coordinates"""
        return self.drawable_manager.delete_point(x, y)

    def delete_point_by_name(self, name):
        """Delete a point by its name"""
        return self.drawable_manager.delete_point_by_name(name)

    def is_point_within_canvas_visible_area(self, x, y):
        """Check if a point is within the visible area of the canvas"""
        return (0 <= x <= self.width) and (0 <= y <= self.height)

    def get_segment_by_coordinates(self, x1, y1, x2, y2):
        """Get a segment by its endpoint coordinates"""
        return self.drawable_manager.get_segment_by_coordinates(x1, y1, x2, y2)

    def get_segment_by_name(self, name):
        """Get a segment by its name"""
        return self.drawable_manager.get_segment_by_name(name)

    def get_segment_by_points(self, p1, p2):
        """Get a segment by its endpoint points"""
        return self.drawable_manager.get_segment_by_points(p1, p2)

    def create_segment(self, x1, y1, x2, y2, name="", extra_graphics=True):
        """Create a segment between two points"""
        return self.drawable_manager.create_segment(x1, y1, x2, y2, name, extra_graphics)

    def delete_segment(self, x1, y1, x2, y2, delete_children=True, delete_parents=False):
        """Delete a segment by its endpoint coordinates"""
        return self.drawable_manager.delete_segment(x1, y1, x2, y2, delete_children, delete_parents)

    def delete_segment_by_name(self, name, delete_children=True, delete_parents=False):
        """Delete a segment by its name"""
        return self.drawable_manager.delete_segment_by_name(name, delete_children, delete_parents)

    def any_segment_part_visible_in_canvas_area(self, x1, y1, x2, y2):
        """Check if any part of a segment is visible in the canvas area"""
        intersect_top = MathUtils.segments_intersect(x1, y1, x2, y2, 0, 0, self.width, 0)
        intersect_right = MathUtils.segments_intersect(x1, y1, x2, y2, self.width, 0, self.width, self.height)
        intersect_bottom = MathUtils.segments_intersect(x1, y1, x2, y2, self.width, self.height, 0, self.height)
        intersect_left = MathUtils.segments_intersect(x1, y1, x2, y2, 0, self.height, 0, 0)
        return intersect_top or intersect_right or intersect_bottom or intersect_left

    def get_vector(self, x1, y1, x2, y2):
        """Get a vector by its origin and tip coordinates"""
        return self.drawable_manager.get_vector(x1, y1, x2, y2)

    def create_vector(self, origin_x, origin_y, tip_x, tip_y, name="", extra_graphics=True):
        """Create a vector from origin to tip"""
        return self.drawable_manager.create_vector(origin_x, origin_y, tip_x, tip_y, name, extra_graphics)

    def delete_vector(self, origin_x, origin_y, tip_x, tip_y):
        """Delete a vector by its origin and tip coordinates"""
        return self.drawable_manager.delete_vector(origin_x, origin_y, tip_x, tip_y)

    def get_triangle(self, x1, y1, x2, y2, x3, y3):
        """Get a triangle by its vertex coordinates"""
        return self.drawable_manager.get_triangle(x1, y1, x2, y2, x3, y3)

    def create_triangle(self, x1, y1, x2, y2, x3, y3, name="", extra_graphics=True):
        """Create a triangle with the specified vertices"""
        return self.drawable_manager.create_triangle(x1, y1, x2, y2, x3, y3, name, extra_graphics)

    def delete_triangle(self, x1, y1, x2, y2, x3, y3):
        """Delete a triangle by its vertex coordinates"""
        return self.drawable_manager.delete_triangle(x1, y1, x2, y2, x3, y3)

    def get_rectangle_by_diagonal_points(self, px, py, opposite_px, opposite_py):
        """Get a rectangle by its diagonal points"""
        return self.drawable_manager.get_rectangle_by_diagonal_points(px, py, opposite_px, opposite_py)

    def get_rectangle_by_name(self, name):
        """Get a rectangle by its name"""
        return self.drawable_manager.get_rectangle_by_name(name)

    def create_rectangle(self, px, py, opposite_px, opposite_py, name="", extra_graphics=True):
        """Create a rectangle with the specified diagonal points"""
        return self.drawable_manager.create_rectangle(px, py, opposite_px, opposite_py, name, extra_graphics)

    def delete_rectangle(self, name):
        """Delete a rectangle by its name"""
        return self.drawable_manager.delete_rectangle(name)

    def get_circle(self, center_x, center_y, radius):
        """Get a circle by its center coordinates and radius"""
        return self.drawable_manager.get_circle(center_x, center_y, radius)

    def get_circle_by_name(self, name):
        """Get a circle by its name"""
        return self.drawable_manager.get_circle_by_name(name)

    def create_circle(self, center_x, center_y, radius, name="", extra_graphics=True):
        """Create a circle with the specified center and radius"""
        return self.drawable_manager.create_circle(center_x, center_y, radius, name, extra_graphics)

    def delete_circle(self, name):
        """Delete a circle by its name"""
        return self.drawable_manager.delete_circle(name)

    def get_ellipse(self, center_x, center_y, radius_x, radius_y):
        """Get an ellipse by its center coordinates and radii"""
        return self.drawable_manager.get_ellipse(center_x, center_y, radius_x, radius_y)

    def get_ellipse_by_name(self, name):
        """Get an ellipse by its name"""
        return self.drawable_manager.get_ellipse_by_name(name)

    def create_ellipse(self, center_x, center_y, radius_x, radius_y, rotation_angle=0, name="", extra_graphics=True):
        """Create an ellipse with the specified center, radii, and rotation"""
        return self.drawable_manager.create_ellipse(center_x, center_y, radius_x, radius_y, rotation_angle, name, extra_graphics)

    def delete_ellipse(self, name):
        """Delete an ellipse by its name"""
        return self.drawable_manager.delete_ellipse(name)

    def get_function(self, name):
        """Get a function by its name"""
        return self.drawable_manager.get_function(name)

    def draw_function(self, function_string, name, left_bound=None, right_bound=None):
        """Draw a function on the canvas"""
        return self.drawable_manager.draw_function(function_string, name, left_bound, right_bound)

    def delete_function(self, name):
        """Delete a function by its name"""
        return self.drawable_manager.delete_function(name)

    def translate_object(self, name, x_offset, y_offset):
        """Translates a drawable object by the specified offset"""
        return self.transformations_manager.translate_object(name, x_offset, y_offset)
        
    def rotate_object(self, name, angle):
        """Rotates a drawable object by the specified angle"""
        return self.transformations_manager.rotate_object(name, angle)

    def has_computation(self, expression):
        """Check if a computation with the given expression already exists."""
        return ComputationUtils.has_computation(self.computations, expression)

    def add_computation(self, expression, result):
        """Add a computation to the history if it doesn't already exist."""
        self.computations = ComputationUtils.add_computation(self.computations, expression, result)

    def find_largest_connected_shape(self, shape):
        """Find the largest shape that shares segments with the given shape.
        Returns a tuple (largest_parent_shape, shape_type) where shape_type is the class name
        or None if no larger shape is found."""
        if not shape:
            return None, None

        # Get all shapes from the canvas
        rectangles = self.drawable_manager.drawables.Rectangles
        triangles = self.drawable_manager.drawables.Triangles

        # If the shape is a rectangle, don't check for parent shapes
        if shape.get_class_name() == 'Rectangle':
            return None, None

        # Check rectangles first as they are larger
        for rect in rectangles:
            if rect != shape:  # Don't compare with itself
                shared_segs = self.get_shared_segments(shape, rect)
                if shared_segs:  # If any segments are shared with a rectangle, return it
                    return rect, rect.get_class_name()

        # Only check triangles if no rectangle was found and the shape isn't a triangle
        if shape.get_class_name() == 'Triangle':
            return None, None

        largest_parent_shape = None
        max_segments = 0

        for tri in triangles:
            if tri != shape:  # Don't compare with itself
                shared_segs = self.get_shared_segments(shape, tri)
                if shared_segs and len(shared_segs) > max_segments:
                    largest_parent_shape = tri
                    max_segments = len(shared_segs)

        return largest_parent_shape, largest_parent_shape.get_class_name() if largest_parent_shape else None

    def get_shared_segments(self, shape1, shape2):
        """Check if two shapes share any segments.
        Returns a list of shared segments."""
        shape1_segments = []
        shape2_segments = []

        # Get segments from shape1
        if hasattr(shape1, 'segment1'):
            shape1_segments.append(shape1.segment1)
        if hasattr(shape1, 'segment2'):
            shape1_segments.append(shape1.segment2)
        if hasattr(shape1, 'segment3'):
            shape1_segments.append(shape1.segment3)
        if hasattr(shape1, 'segment4'):
            shape1_segments.append(shape1.segment4)

        # Get segments from shape2
        if hasattr(shape2, 'segment1'):
            shape2_segments.append(shape2.segment1)
        if hasattr(shape2, 'segment2'):
            shape2_segments.append(shape2.segment2)
        if hasattr(shape2, 'segment3'):
            shape2_segments.append(shape2.segment3)
        if hasattr(shape2, 'segment4'):
            shape2_segments.append(shape2.segment4)

        # Find shared segments
        shared_segments = []
        for s1 in shape1_segments:
            for s2 in shape2_segments:
                if s1 == s2:
                    shared_segments.append(s1)

        return shared_segments

    def create_colored_area(self, drawable1_name, drawable2_name=None, left_bound=None, right_bound=None, color="lightblue", opacity=0.3):
        """Creates a colored area between two functions, two segments, or a function and a segment"""
        return self.drawable_manager.create_colored_area(drawable1_name, drawable2_name, left_bound, right_bound, color, opacity)
        
    def delete_colored_area(self, name):
        """Deletes a colored area with the given name"""
        return self.drawable_manager.delete_colored_area(name)
        
    def delete_colored_areas_for_function(self, func):
        """Deletes all colored areas associated with a function"""
        return self.drawable_manager.delete_colored_areas_for_function(func)
        
    def delete_colored_areas_for_segment(self, segment):
        """Deletes all colored areas associated with a segment"""
        return self.drawable_manager.delete_colored_areas_for_segment(segment)
        
    def get_colored_areas_for_drawable(self, drawable):
        """Gets all colored areas associated with a drawable (function or segment)"""
        return self.drawable_manager.get_colored_areas_for_drawable(drawable)
        
    def update_colored_area_style(self, name, color=None, opacity=None):
        """Updates the color and/or opacity of a colored area"""
        return self.drawable_manager.update_colored_area_style(name, color, opacity)

    def _validate_color_and_opacity(self, color, opacity):
        """Validates both color and opacity values"""
        return StyleUtils.validate_color_and_opacity(color, opacity)

    def _is_valid_css_color(self, color):
        """Validates if a string is a valid CSS color."""
        return StyleUtils.is_valid_css_color(color)

    def _validate_opacity(self, opacity):
        """Validates if an opacity value is between 0 and 1"""
        return StyleUtils.validate_opacity(opacity)

    @property
    def name_generator(self):
        """Property to access the name generator from the drawable manager"""
        return self.drawable_manager.name_generator

    # ------------------- Angle Methods -------------------

    def create_angle(self, vx, vy, p1x, p1y, p2x, p2y, color=None, angle_name=None, is_reflex: bool = False, extra_graphics=True):
        """Create an angle defined by three points via AngleManager."""
        if self.drawable_manager.angle_manager:
            return self.drawable_manager.angle_manager.create_angle(
                vx, vy, p1x, p1y, p2x, p2y, 
                color=color, 
                angle_name=angle_name, 
                is_reflex=is_reflex,
                extra_graphics=extra_graphics
            )
        return None

    def delete_angle(self, name):
        """Remove an angle by its name via AngleManager."""
        if self.drawable_manager.angle_manager:
            return self.drawable_manager.angle_manager.delete_angle(name)
        return False

    def update_angle_properties(self, name, new_color=None):
        """Update properties of an angle via AngleManager."""
        if self.drawable_manager.angle_manager:
            return self.drawable_manager.angle_manager.update_angle_properties(
                name, new_color=new_color
            )
        return False

    # Property delegations to CoordinateMapper for backward compatibility
    @property
    def center(self):
        """Current viewport center point - delegates to coordinate_mapper.origin"""
        return self.coordinate_mapper.origin
    
    @center.setter
    def center(self, value):
        """Set viewport center point - delegates to coordinate_mapper.origin"""
        self.coordinate_mapper.origin = value
    
    @property
    def scale_factor(self):
        """Current zoom level - delegates to coordinate_mapper.scale_factor"""
        return self.coordinate_mapper.scale_factor
    
    @scale_factor.setter
    def scale_factor(self, value):
        """Set zoom level - delegates to coordinate_mapper.scale_factor"""
        self.coordinate_mapper.scale_factor = value
    
    @property
    def offset(self):
        """Current pan offset - delegates to coordinate_mapper.offset"""
        return self.coordinate_mapper.offset
    
    @offset.setter
    def offset(self, value):
        """Set pan offset - delegates to coordinate_mapper.offset"""
        self.coordinate_mapper.offset = value
    
    @property
    def zoom_point(self):
        """Current zoom center point - delegates to coordinate_mapper.zoom_point"""
        return self.coordinate_mapper.zoom_point
    
    @zoom_point.setter
    def zoom_point(self, value):
        """Set zoom center point - delegates to coordinate_mapper.zoom_point"""
        self.coordinate_mapper.zoom_point = value
    
    @property
    def zoom_direction(self):
        """Current zoom direction - delegates to coordinate_mapper.zoom_direction"""
        return self.coordinate_mapper.zoom_direction
    
    @zoom_direction.setter
    def zoom_direction(self, value):
        """Set zoom direction - delegates to coordinate_mapper.zoom_direction"""
        self.coordinate_mapper.zoom_direction = value
    
    @property
    def zoom_step(self):
        """Zoom step size - delegates to coordinate_mapper.zoom_step"""
        return self.coordinate_mapper.zoom_step
    
    @zoom_step.setter
    def zoom_step(self, value):
        """Set zoom step size - delegates to coordinate_mapper.zoom_step"""
        self.coordinate_mapper.zoom_step = value

