"""
MatHud Two-Axis Cartesian Coordinate System

Implements a complete Cartesian coordinate system with grid visualization, axis rendering,
and coordinate transformations. Provides the mathematical foundation for geometric object
positioning and user interaction coordinate mapping.

Key Features:
    - Dynamic grid scaling with zoom-adaptive tick spacing
    - Axis rendering with numerical labels and origin marking
    - Coordinate transformation between screen and mathematical space
    - Viewport boundary calculations for efficient rendering
    - Grid line visualization with customizable spacing
    - Origin point management and positioning

Visual Components:
    - X and Y axes with customizable colors and thickness
    - Tick marks with automatic spacing calculation
    - Numerical labels with mathematical formatting
    - Grid lines for visual coordinate reference
    - Origin marker ('O') at coordinate system center

Coordinate Transformations:
    - Screen to mathematical coordinate conversion
    - Mathematical to screen coordinate conversion
    - Zoom and pan transformation support
    - Viewport boundary calculations
    - Visible area determination for rendering optimization

Dependencies:
    - browser.document: DOM manipulation for SVG rendering
    - geometry: Drawable base class and Position utilities
    - constants: Default styling and configuration values
    - utils.math_utils: Mathematical calculations and number formatting
"""

from browser import document
from constants import default_color
from geometry import Drawable, Position
from utils.math_utils import MathUtils
import math

class Cartesian2Axis(Drawable):
    """
    Two-axis Cartesian coordinate system with dynamic scaling and grid visualization.

    Inherits from Drawable to provide complete coordinate system rendering with
    automatic scaling, tick spacing calculation, and viewport management.

    Attributes:
        width (float): Canvas width for coordinate system bounds
        height (float): Canvas height for coordinate system bounds
        origin (Position): Current origin position in screen coordinates
        default_tick_spacing (float): Base tick spacing for coordinate labels
        current_tick_spacing (float): Current tick spacing adjusted for zoom level
        max_ticks (int): Maximum number of ticks to display
        tick_size (int): Visual size of tick marks in pixels
        tick_color (str): Color for axis lines and tick marks
        tick_label_color (str): Color for numerical coordinate labels
        tick_label_font_size (int): Font size for coordinate labels
        grid_color (str): Color for grid lines
    """
    
    def __init__(self, canvas, color=default_color):
        """Initialize Cartesian coordinate system with canvas and color."""
        self.name = "cartesian-2axis-system"
        self.width = canvas.width
        self.height = canvas.height
        self.origin = Position(canvas.center.x, canvas.center.y)  # initial placement on canvas
        self.default_tick_spacing = 100
        self.current_tick_spacing = 100  # Track the previous tick spacing to determine zoom level
        self.max_ticks = 10
        self.tick_size = 3
        self.tick_color = color
        self.tick_label_color = "grey"
        self.tick_label_font_size = 8
        self.grid_color = "lightgrey"
        super().__init__(name=self.name, color=color, canvas=canvas)

    @Drawable.canvas.setter
    def canvas(self, value):
        self._canvas = value

    @canvas.getter
    def canvas(self):
        return self._canvas
    
    def _initialize(self):
        pass   # Cartesian2Axis is not initialized
    
    def reset(self):
        """Reset coordinate system to initial state with centered origin."""
        self.origin.x = self.canvas.center.x
        self.origin.y = self.canvas.center.y
        self.current_tick_spacing = self.default_tick_spacing

    def get_class_name(self):
        """Return the class name 'Cartesian2Axis'."""
        return 'Cartesian2Axis'
    
    def get_visible_left_bound(self):
        """Calculate visible left boundary in mathematical coordinates."""
        # Calculate the visible left bound system-axis absolute value based on the origin and scale factor
        return -self.origin.x / self.canvas.scale_factor

    def get_visible_right_bound(self):
        """Calculate visible right boundary in mathematical coordinates."""
        # Calculate the visible right bound system-axis absolute value based on the origin, width of the canvas, and scale factor
        return (self.width - self.origin.x) / self.canvas.scale_factor

    def get_visible_top_bound(self):
        """Calculate visible top boundary in mathematical coordinates."""
        # Calculate the visible top bound system-axis absolute value based on the origin and scale factor
        return self.origin.y / self.canvas.scale_factor

    def get_visible_bottom_bound(self):
        """Calculate visible bottom boundary in mathematical coordinates."""
        # Calculate the visible bottom bound system-axis absolute value based on the origin, height of the canvas, and scale factor
        return (self.origin.y - self.height) / self.canvas.scale_factor

    def get_relative_width(self):
        """Get canvas width adjusted for current scale factor."""
        return self.width / self.canvas.scale_factor
    
    def get_relative_height(self):
        """Get canvas height adjusted for current scale factor."""
        return self.height / self.canvas.scale_factor

    def draw(self):
        """Render complete coordinate system including axes, ticks, labels, and grid."""
        # Draw axes
        self._draw_axes()
        # Add spaced ticks on the axes
        display_tick_spacing = self.current_tick_spacing * self.canvas.scale_factor
        # Draw ticks on axes
        self._draw_ticks(display_tick_spacing)
        # Draw grid
        self._draw_grid(display_tick_spacing / 2)

    def _draw_axes(self):
        # Draw x-axis
        self.create_svg_element('line', 
                              x1=str(0), 
                              y1=str(self.origin.y), 
                              x2=str(self.width), 
                              y2=str(self.origin.y), 
                              stroke=self.color)
        # Draw y-axis
        self.create_svg_element('line', 
                              x1=str(self.origin.x), 
                              y1=str(0), 
                              x2=str(self.origin.x), 
                              y2=str(self.height), 
                              stroke=self.color)

    def _draw_ticks(self, step):
        for axis in ['x', 'y']:
            for direction in [-1, 1]:
                self._draw_axis_ticks(axis, step, direction)
    
    def _draw_axis_ticks(self, axis, step, direction):
        tick_mark = self._get_axis_origin(axis)
        boundary = self._get_axis_boundary(axis)
        tick_size = self.tick_size
        
        while self._should_continue_drawing(tick_mark, boundary, direction):
            if axis == 'x':
                self._draw_x_axis_tick(tick_mark, tick_size)
                self._draw_x_axis_label(tick_mark)
            else:
                self._draw_y_axis_tick(tick_mark, tick_size)
                self._draw_y_axis_label(tick_mark)
            
            tick_mark += step * direction
    
    def _should_continue_drawing(self, position, boundary, direction):
        return (direction == 1 and position < boundary) or (direction == -1 and position > 0)
    
    def _draw_x_axis_tick(self, tick_mark, tick_size):
        self.create_svg_element('line', 
                              x1=str(tick_mark), 
                              y1=str(self.origin.y - tick_size), 
                              x2=str(tick_mark), 
                              y2=str(self.origin.y + tick_size), 
                              stroke=self.color)
    
    def _draw_x_axis_label(self, tick_mark):
        if tick_mark == self.origin.x:
            # Draw 'O' at origin
            self._draw_origin_label(tick_mark)
        else:
            # Draw number labels for non-origin ticks
            tick_label_value = (tick_mark - self.origin.x) / self.canvas.scale_factor
            self._draw_tick_label(tick_mark + 2, 
                                self.origin.y + self.tick_size + self.tick_label_font_size,
                                tick_label_value)
    
    def _draw_y_axis_tick(self, tick_mark, tick_size):
        self.create_svg_element('line', 
                              x1=str(self.origin.x - tick_size), 
                              y1=str(tick_mark), 
                              x2=str(self.origin.x + tick_size), 
                              y2=str(tick_mark), 
                              stroke=self.color)
    
    def _draw_y_axis_label(self, tick_mark):
        if tick_mark != self.origin.y:
            # Draw number labels for non-origin ticks
            tick_label_value = (self.origin.y - tick_mark) / self.canvas.scale_factor
            self._draw_tick_label(self.origin.x + self.tick_size, 
                                tick_mark - self.tick_size,
                                tick_label_value)
    
    def _draw_origin_label(self, tick_mark):
        self.create_svg_element('text', 
                              x=str(tick_mark + 2), 
                              y=str(self.origin.y + self.tick_size + self.tick_label_font_size), 
                              fill=self.tick_label_color, 
                              text_content='O', 
                              text_font_size=self.tick_label_font_size)
    
    def _draw_tick_label(self, x, y, value):
        tick_label_text = MathUtils.format_number_for_cartesian(value)
        self.create_svg_element('text', 
                              x=str(x), 
                              y=str(y), 
                              fill=self.tick_label_color, 
                              text_content=tick_label_text, 
                              text_font_size=self.tick_label_font_size)

    def _draw_grid(self, step):
        for axis in ['x', 'y']:
            for direction in [-1, 1]:
                self._draw_grid_axis_lines(axis, step, direction)
    
    def _draw_grid_axis_lines(self, axis, step, direction):
        grid_mark = self._get_axis_origin(axis)
        boundary = self._get_axis_boundary(axis)
        
        while self._should_continue_drawing(grid_mark, boundary, direction):
            self._draw_grid_line(axis, grid_mark)
            grid_mark += step * direction
    
    def _draw_grid_line(self, axis, grid_mark):
        if axis == 'y':
            start_point = Position(0, grid_mark)
            end_point = Position(self.width, grid_mark)
        else:  # axis == 'x'
            start_point = Position(grid_mark, 0)
            end_point = Position(grid_mark, self.height)
            
        self.create_svg_element('line', 
                              x1=str(start_point.x), 
                              y1=str(start_point.y), 
                              x2=str(end_point.x), 
                              y2=str(end_point.y), 
                              stroke=self.grid_color)

    def _translate(self, offset):
        self.origin._translate(offset)
   
    def _calculate_tick_spacing(self):
        ideal_spacing = self._calculate_ideal_tick_spacing()
        return self._find_appropriate_spacing(ideal_spacing)
    
    def _calculate_ideal_tick_spacing(self):
        relative_width = self.get_relative_width()  # Width of the visible cartesian system in units
        return relative_width / self.max_ticks  # Ideal width of a tick spacing
    
    def _find_appropriate_spacing(self, ideal_spacing):
        # Find the order of magnitude of the ideal_tick_spacing
        magnitude = 10 ** math.floor(math.log10(ideal_spacing))
        possible_spacings = [magnitude * i for i in [1, 2.5, 5, 10]]      
        
        # Find the closest spacing to ideal_tick_spacing that is larger or equal to it
        for spacing in possible_spacings:
            if spacing >= ideal_spacing:
                return spacing
                
        # If none of the spacings fit (very unlikely), fallback to the smallest spacing
        return possible_spacings[0]

    def zoom(self):
        """Update coordinate system for zoom operations with dynamic tick spacing."""
        # Calculate the new display tick spacing based on the current zoom level and scale factor
        proposed_tick_spacing = self._calculate_tick_spacing()
        # Determine whether to zoom in or out based on zoom_direction
        zoom_in = self.canvas.zoom_direction == -1
        if zoom_in:
            # Zooming in: Decrease the tick spacing if it's less than twice the current tick spacing
            if proposed_tick_spacing < self.current_tick_spacing / 0.5:
                self.current_tick_spacing = proposed_tick_spacing
        else:
            # Zooming out: Increase the tick spacing if it's greater than half the current tick spacing
            if proposed_tick_spacing > self.current_tick_spacing * 0.5:
                self.current_tick_spacing = proposed_tick_spacing
        self.origin.zoom()

    def pan(self):
        """Update coordinate system for pan operations."""
        self._translate(self.canvas.offset)

    def get_state(self):
        """Serialize coordinate system state for persistence."""
        state = {"Cartesian_System_Visibility": {"left_bound": int(self.get_visible_left_bound()), "right_bound": int(self.get_visible_right_bound()), "top_bound": int(self.get_visible_top_bound()), "bottom_bound": int(self.get_visible_bottom_bound())}}
        return state

    def _get_axis_origin(self, axis):
        """Get the origin position for the specified axis"""
        return self.origin.x if axis == 'x' else self.origin.y
    
    def _get_axis_boundary(self, axis):
        """Get the boundary (width/height) for the specified axis"""
        return self.width if axis == 'x' else self.height