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
    - geometry: Drawable base class and Position utilities
    - constants: Default styling and configuration values
    - utils.math_utils: Mathematical calculations and number formatting
"""

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
        self.default_tick_spacing = 100
        self.current_tick_spacing = 100  # Track the previous tick spacing to determine zoom level
        # Bias factor to make intermediate tick spacing appear sooner (tuneable)
        self.tick_spacing_bias = 0.5
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
        self.current_tick_spacing = self.default_tick_spacing

    def get_class_name(self):
        """Return the class name 'Cartesian2Axis'."""
        return 'Cartesian2Axis'
    
    @property
    def origin(self):
        """Get the screen coordinates of the mathematical origin (0,0) using CoordinateMapper."""
        origin_x, origin_y = self.canvas.coordinate_mapper.math_to_screen(0, 0)
        return Position(origin_x, origin_y)
    
    def get_visible_left_bound(self):
        """Calculate visible left boundary in mathematical coordinates."""
        return self.canvas.coordinate_mapper.get_visible_left_bound()

    def get_visible_right_bound(self):
        """Calculate visible right boundary in mathematical coordinates."""
        return self.canvas.coordinate_mapper.get_visible_right_bound()

    def get_visible_top_bound(self):
        """Calculate visible top boundary in mathematical coordinates."""
        return self.canvas.coordinate_mapper.get_visible_top_bound()

    def get_visible_bottom_bound(self):
        """Calculate visible bottom boundary in mathematical coordinates."""
        return self.canvas.coordinate_mapper.get_visible_bottom_bound()

    def get_relative_width(self):
        """Get canvas width adjusted for current scale factor."""
        return self.width / self.canvas.scale_factor
    
    def get_relative_height(self):
        """Get canvas height adjusted for current scale factor."""
        return self.height / self.canvas.scale_factor

    def draw(self):
        """No-op: rendering handled via renderer."""
        return

    def _draw_axes(self):
        return

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
        return
    
    def _draw_x_axis_label(self, tick_mark):
        return
    
    def _draw_y_axis_tick(self, tick_mark, tick_size):
        return
    
    def _draw_y_axis_label(self, tick_mark):
        return
    
    def _draw_origin_label(self, tick_mark):
        return
    
    def _draw_tick_label(self, x, y, value):
        return

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
        return


   
    def _calculate_tick_spacing(self):
        ideal_spacing = self._calculate_ideal_tick_spacing()
        return self._find_appropriate_spacing(ideal_spacing)
    
    def _calculate_ideal_tick_spacing(self):
        relative_width = self.get_relative_width()  # Width of the visible cartesian system in units
        ideal = relative_width / self.max_ticks  # Ideal width of a tick spacing
        return ideal
    
    def _find_appropriate_spacing(self, ideal_spacing):
        # Bias to densify sooner and coarsen later
        effective_ideal = ideal_spacing * self.tick_spacing_bias
        # Find the order of magnitude of the (biased) ideal spacing
        magnitude = 10 ** math.floor(math.log10(effective_ideal))
        # Standard nice steps (denser than 2.5): 1, 2, 5, 10
        possible_spacings = [magnitude * i for i in [1, 2, 5, 10]]
        # Pick the smallest spacing not less than effective_ideal,
        # but avoid collapsing to the same spacing if viewport changed significantly.
        for spacing in possible_spacings:
            if spacing >= effective_ideal:
                return spacing

        # Fallback
        return possible_spacings[0]

    def _invalidate_cache_on_zoom(self):
        """Update tick spacing for zoom operations with dynamic spacing calculation."""
        # Recompute ideal spacing in math units based on current scale and viewport
        proposed_tick_spacing = self._calculate_tick_spacing()
        # Apply directly to keep ticks consistent with zoom (no dependency on zoom_direction)
        if proposed_tick_spacing and proposed_tick_spacing > 0:
            self.current_tick_spacing = proposed_tick_spacing

    def get_state(self):
        """Serialize coordinate system state for persistence."""
        state = {"Cartesian_System_Visibility": {"left_bound": int(self.get_visible_left_bound()), "right_bound": int(self.get_visible_right_bound()), "top_bound": int(self.get_visible_top_bound()), "bottom_bound": int(self.get_visible_bottom_bound())}}
        return state

    def _get_axis_origin(self, axis):
        """Get the origin position for the specified axis"""
        origin = self.origin
        return origin.x if axis == 'x' else origin.y
    
    def _get_axis_boundary(self, axis):
        """Get the boundary (width/height) for the specified axis"""
        return self.width if axis == 'x' else self.height