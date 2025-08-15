"""
MatHud Colored Area Management System

Manages colored area creation, deletion, and style management for mathematical region visualization.
Handles areas bounded by functions, segments, or mixed boundaries with automatic type detection.

Core Responsibilities:
    - Area Creation: Creates colored regions between mathematical boundaries
    - Area Deletion: Safe removal with cleanup of dependencies
    - Type Detection: Automatically determines area type based on boundary objects
    - Style Management: Handles color and opacity customization

Supported Area Types:
    - Function-Function Areas: Regions between two mathematical functions
    - Segment-Segment Areas: Regions between line segments and axes
    - Function-Segment Areas: Mixed regions between functions and segments
    - Axis Integration: Areas between objects and coordinate axes

Advanced Features:
    - Automatic Type Detection: Determines appropriate area class based on boundaries
    - Smart Boundary Handling: Manages x-axis references and null boundaries
    - Intersection Calculation: Creates boundary points for accurate area representation
    - Color Validation: Ensures valid color and opacity values

Integration Points:
    - FunctionManager: Retrieves function objects for boundary definition
    - SegmentManager: Retrieves segment objects and creates intersection points
    - StyleUtils: Validates color and opacity parameters
    - Canvas: Handles area rendering and visual updates

Mathematical Context:
    - Integration Visualization: Represents definite integrals as colored areas
    - Geometric Analysis: Shows relationships between mathematical objects
    - Boundary Calculations: Handles complex boundary intersections
    - Domain Management: Respects function domains and segment ranges

State Management:
    - Undo/Redo: Complete state archiving for area operations
    - Dependency Tracking: Maintains relationships with boundary objects
    - Canvas Integration: Immediate visual updates after modifications
    - Cleanup Logic: Intelligent removal of areas when boundaries are deleted
"""

from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from drawables.segment import Segment
from drawables.function import Function
from utils.style_utils import StyleUtils

class ColoredAreaManager:
    """
    Manages colored area drawables for a Canvas.
    
    This class is responsible for:
    - Creating colored area objects
    - Retrieving colored area objects
    - Deleting colored area objects
    """
    
    def __init__(self, canvas, drawables_container, name_generator, dependency_manager, drawable_manager_proxy):
        """
        Initialize the ColoredAreaManager.
        
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
        
    def create_colored_area(self, drawable1_name, drawable2_name=None, left_bound=None, right_bound=None, color="lightblue", opacity=0.3):
        """
        Creates a colored area between two functions, two segments, or a function and a segment.
        Automatically determines the type of colored area based on the inputs.
        
        Args:
            drawable1_name: Name of first function/segment (or None for x-axis)
            drawable2_name: Name of second function/segment (or None for x-axis)
            left_bound: Optional left bound for function areas
            right_bound: Optional right bound for function areas
            color: Color of the area (default: lightblue)
            opacity: Opacity of the area (default: 0.3)
            
        Returns:
            The created colored area object
            
        Raises:
            ValueError: If color or opacity values are invalid or if drawables not found
        """
        # Validate color and opacity before proceeding
        self.canvas._validate_color_and_opacity(color, opacity)
        
        # Archive for undo
        self.canvas.undo_redo_manager.archive()
        
        # Get the first drawable
        drawable1 = None
        if drawable1_name is not None and drawable1_name != "x_axis":
            drawable1 = self.drawable_manager.get_segment_by_name(drawable1_name)
            if drawable1 is None:
                drawable1 = self.drawable_manager.get_function(drawable1_name)
            if drawable1 is None:
                raise ValueError(f"Could not find drawable with name {drawable1_name}")
        
        # Get the second drawable if provided
        drawable2 = None
        if drawable2_name is not None and drawable2_name != "x_axis":
            drawable2 = self.drawable_manager.get_segment_by_name(drawable2_name)
            if drawable2 is None:
                drawable2 = self.drawable_manager.get_function(drawable2_name)
            if drawable2 is None:
                raise ValueError(f"Could not find drawable with name {drawable2_name}")
        
        if isinstance(drawable1, Segment):
            if drawable2 is not None and not isinstance(drawable2, Segment):
                if isinstance(drawable2, (Function, type(None))) or isinstance(drawable2, (int, float)):
                    # If drawable1 is a segment and drawable2 is a function/None/number, swap them
                    drawable1, drawable2 = drawable2, drawable1
                else:
                    raise ValueError("Invalid combination of arguments")

        if isinstance(drawable1, Segment) and (drawable2 is None or isinstance(drawable2, Segment)):
            # Segment-segment or segment-xaxis case
            if drawable2:  # Segment-segment case
                # Create points at overlap boundaries
                def get_y_at_x(segment, x):
                    # Linear interpolation to find y value at x using math coordinates
                    x1, y1 = segment.point1.x, segment.point1.y
                    x2, y2 = segment.point2.x, segment.point2.y
                    if x2 == x1:
                        return y1  # Vertical segment
                    t = (x - x1) / (x2 - x1)
                    return y1 + t * (y2 - y1)

                # Get x-ranges of both segments using math coordinates
                x1_min = min(drawable1.point1.x, drawable1.point2.x)
                x1_max = max(drawable1.point1.x, drawable1.point2.x)
                x2_min = min(drawable2.point1.x, drawable2.point2.x)
                x2_max = max(drawable2.point1.x, drawable2.point2.x)

                # Check if segment1 endpoints create points on segment2
                if x1_min >= x2_min and x1_min <= x2_max:
                    y = get_y_at_x(drawable2, x1_min)
                    self.drawable_manager.create_point(x1_min, y)
                if x1_max >= x2_min and x1_max <= x2_max:
                    y = get_y_at_x(drawable2, x1_max)
                    self.drawable_manager.create_point(x1_max, y)

                # Check if segment2 endpoints create points on segment1
                if x2_min >= x1_min and x2_min <= x1_max:
                    y = get_y_at_x(drawable1, x2_min)
                    self.drawable_manager.create_point(x2_min, y)
                if x2_max >= x1_min and x2_max <= x1_max:
                    y = get_y_at_x(drawable1, x2_max)
                    self.drawable_manager.create_point(x2_max, y)

            colored_area = SegmentsBoundedColoredArea(drawable1, drawable2, color=color, opacity=opacity)
        elif isinstance(drawable2, Segment):
            # Function-segment case (we know drawable1 is not a segment due to the swap above)
            colored_area = FunctionSegmentBoundedColoredArea(drawable1, drawable2, color=color, opacity=opacity)
        else:
            # Function-function case
            colored_area = FunctionsBoundedColoredArea(drawable1, drawable2, 
                                                     left_bound=left_bound, right_bound=right_bound,
                                                     color=color, opacity=opacity)

        # Add to drawables
        self.drawables.add(colored_area)
        
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return colored_area
        
    def delete_colored_area(self, name):
        """
        Delete a colored area by its name.
        
        Searches through all colored area categories to find and remove the area
        with the specified name. Archives the state for undo functionality.
        
        Args:
            name (str): The name of the colored area to delete
            
        Returns:
            bool: True if the colored area was found and deleted, False otherwise
        """
        # Find the colored area in all categories
        colored_area = None
        for category_property in [self.drawables.FunctionsBoundedColoredAreas, 
                                 self.drawables.SegmentsBoundedColoredAreas,
                                 self.drawables.FunctionSegmentBoundedColoredAreas]:
            for area in category_property:
                if area.name == name:
                    colored_area = area
                    break
            if colored_area:
                break
                
        if not colored_area:
            return False
            
        # Archive before deletion
        self.canvas.undo_redo_manager.archive()
        
        # Remove the colored area
        self.drawables.remove(colored_area)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True
        
    def delete_colored_areas_for_function(self, func):
        """
        Deletes all colored areas associated with a function
        
        Args:
            func: The function whose colored areas should be deleted
            
        Returns:
            bool: True if any areas were deleted, False otherwise
        """
        # Check if function is a string (name) or object
        if isinstance(func, str):
            func = self.drawable_manager.get_function(func)
            
        if not func:
            return False
            
        # First check if there are any areas to delete
        areas_to_delete = []
        
        # Check FunctionsBoundedColoredArea
        for area in self.drawables.FunctionsBoundedColoredAreas:
            if area.func1 == func or area.func2 == func:
                areas_to_delete.append(area)

        # Check FunctionSegmentBoundedColoredArea
        for area in self.drawables.FunctionSegmentBoundedColoredAreas:
            if area.func == func:
                areas_to_delete.append(area)
        
        if areas_to_delete:
            # Archive for undo
            self.canvas.undo_redo_manager.archive()
            
            # Now delete the areas
            for area in areas_to_delete:
                self.drawables.remove(area)
                
            if self.canvas.draw_enabled:
                self.canvas.draw()
                
            return True
            
        return False
        
    def delete_colored_areas_for_segment(self, segment):
        """
        Deletes all colored areas associated with a segment
        
        Args:
            segment: The segment whose colored areas should be deleted
            
        Returns:
            bool: True if any areas were deleted, False otherwise
        """
        # Check if segment is a string (name) or object
        if isinstance(segment, str):
            segment = self.drawable_manager.get_segment_by_name(segment)
            
        if not segment:
            return False
            
        # First check if there are any areas to delete
        areas_to_delete = []
        
        # Check SegmentsBoundedColoredArea
        for area in self.drawables.SegmentsBoundedColoredAreas:
            if area.uses_segment(segment):
                areas_to_delete.append(area)
                
        # Check FunctionSegmentBoundedColoredArea
        for area in self.drawables.FunctionSegmentBoundedColoredAreas:
            if area.uses_segment(segment):
                areas_to_delete.append(area)
        
        if areas_to_delete:
            # Archive for undo
            self.canvas.undo_redo_manager.archive()
            
            # Now delete the areas
            for area in areas_to_delete:
                self.drawables.remove(area)
                
            if self.canvas.draw_enabled:
                self.canvas.draw()
                
            return True
            
        return False
        
    def get_colored_areas_for_drawable(self, drawable):
        """
        Gets all colored areas associated with a drawable (function or segment)
        
        Args:
            drawable: The function or segment to find colored areas for
            
        Returns:
            list: List of colored areas that use the drawable
        """
        areas = []
        
        if isinstance(drawable, Function):
            # Check FunctionsBoundedColoredArea
            for area in self.drawables.FunctionsBoundedColoredAreas:
                if area.func1 == drawable or area.func2 == drawable:
                    areas.append(area)

            # Check FunctionSegmentBoundedColoredArea
            for area in self.drawables.FunctionSegmentBoundedColoredAreas:
                if area.func == drawable:
                    areas.append(area)
                    
        elif isinstance(drawable, Segment):
            # Check SegmentsBoundedColoredArea
            for area in self.drawables.SegmentsBoundedColoredAreas:
                if area.uses_segment(drawable):
                    areas.append(area)

            # Check FunctionSegmentBoundedColoredArea
            for area in self.drawables.FunctionSegmentBoundedColoredAreas:
                if area.uses_segment(drawable):
                    areas.append(area)
                    
        return areas
        
    def update_colored_area_style(self, name, color=None, opacity=None):
        """
        Updates the color and/or opacity of a colored area
        
        Args:
            name: Name of the colored area to update
            color: New color (optional)
            opacity: New opacity (optional)
            
        Returns:
            bool: True if the area was updated, False otherwise
            
        Raises:
            ValueError: If the area doesn't exist or if color/opacity values are invalid
        """
        # First find the area
        area = None
        
        # Check in all colored area collections
        collections = [
            self.drawables.ColoredAreas,
            self.drawables.FunctionsBoundedColoredAreas,
            self.drawables.SegmentsBoundedColoredAreas,
            self.drawables.FunctionSegmentBoundedColoredAreas
        ]
        
        for collection in collections:
            for a in collection:
                if a.name == name:
                    area = a
                    break
            if area:
                break
                
        if not area:
            raise ValueError(f"Colored area '{name}' not found")
        
        # Archive before update
        self.canvas.undo_redo_manager.archive()
            
        # Validate color and opacity if provided
        if color is not None:
            if not StyleUtils.is_valid_css_color(color):
                raise ValueError(f"Invalid CSS color: {color}")
            area.color = color
            
        if opacity is not None:
            if not StyleUtils.validate_opacity(opacity):
                raise ValueError(f"Invalid opacity value: {opacity}. Must be between 0 and 1")
            area.opacity = opacity
            
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 