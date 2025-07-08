"""
MatHud Coordinate Mapping Service

Centralized coordinate transformation service that converts between mathematical coordinates
and screen pixel coordinates. Manages zoom, pan, and scale transformations in one place
to eliminate scattered transformation logic across drawable classes.

Key Features:
    - Math-to-screen and screen-to-math coordinate conversion
    - Scale factor management for zoom operations
    - Pan offset handling for viewport translation
    - Visible bounds calculation for optimization
    - Y-axis flipping for mathematical coordinate system

Dependencies:
    - drawables.position: Position coordinate container
    - math: Standard library mathematical operations
"""

from drawables.position import Position
import math


class CoordinateMapper:
    """Centralized coordinate transformation service for Canvas and Drawable objects.
    
    Manages all coordinate conversions between mathematical space and screen pixels,
    eliminating the need for individual drawable classes to handle transformations.
    
    Attributes:
        canvas_width (float): Canvas viewport width in pixels
        canvas_height (float): Canvas viewport height in pixels
        scale_factor (float): Current zoom level (1.0 = normal scale)
        offset (Position): Current pan offset in screen pixels
        origin (Position): Canvas center point in screen coordinates
        zoom_point (Position): Center point for zoom operations
        zoom_direction (int): Zoom direction (-1 = zoom in, 1 = zoom out)
        zoom_step (float): Zoom increment per step (default 0.1 = 10%)
    """
    
    def __init__(self, canvas_width, canvas_height):
        """Initialize coordinate mapper with canvas dimensions.
        
        Args:
            canvas_width (float): Canvas viewport width in pixels
            canvas_height (float): Canvas viewport height in pixels
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.scale_factor = 1.0
        self.offset = Position(0, 0)
        self.origin = Position(canvas_width / 2, canvas_height / 2)
        
        # Zoom state management
        self.zoom_point = Position(0, 0)
        self.zoom_direction = 0
        self.zoom_step = 0.1
    
    @classmethod
    def from_canvas(cls, canvas):
        """Create a CoordinateMapper from an existing Canvas object.
        
        This factory method extracts coordinate transformation state from a Canvas
        to create a properly initialized CoordinateMapper.
        
        Args:
            canvas: Canvas object with coordinate transformation state
            
        Returns:
            CoordinateMapper: Initialized with Canvas state
        """
        # Create mapper with canvas dimensions
        mapper = cls(canvas.width, canvas.height)
        
        # Sync with canvas state
        mapper.sync_from_canvas(canvas)
        
        return mapper
    
    def math_to_screen(self, math_x, math_y):
        """Convert mathematical coordinates to screen pixel coordinates.
        
        Applies scale factor, origin translation, and pan offset with Y-axis flipping
        to match mathematical coordinate system conventions.
        
        Args:
            math_x (float): Mathematical x-coordinate
            math_y (float): Mathematical y-coordinate
            
        Returns:
            tuple: (screen_x, screen_y) in pixel coordinates
        """
        screen_x = self.origin.x + (math_x * self.scale_factor) + self.offset.x
        screen_y = self.origin.y - (math_y * self.scale_factor) + self.offset.y
        return screen_x, screen_y
    
    def screen_to_math(self, screen_x, screen_y):
        """Convert screen pixel coordinates to mathematical coordinates.
        
        Reverses scale factor, origin translation, and pan offset with Y-axis flipping
        to get original mathematical values.
        
        Args:
            screen_x (float): Screen x-coordinate in pixels
            screen_y (float): Screen y-coordinate in pixels
            
        Returns:
            tuple: (math_x, math_y) in mathematical coordinates
        """
        math_x = (screen_x - self.offset.x - self.origin.x) / self.scale_factor
        math_y = (self.origin.y + self.offset.y - screen_y) / self.scale_factor
        return math_x, math_y
    
    def scale_value(self, math_value):
        """Scale a mathematical value to screen units.
        
        Used for scaling mathematical properties like radius, distance, 
        or any other measurement that needs to scale with zoom level.
        
        Args:
            math_value (float): Mathematical value to scale
            
        Returns:
            float: Scaled value in screen units
        """
        return math_value * self.scale_factor
    
    def unscale_value(self, screen_value):
        """Convert a screen value back to mathematical units.
        
        Inverse of scale_value, useful for converting measurements
        back to mathematical space.
        
        Args:
            screen_value (float): Screen value to unscale
            
        Returns:
            float: Value in mathematical units
        """
        return screen_value / self.scale_factor
    
    def apply_zoom(self, zoom_factor, zoom_center_screen=None):
        """Apply zoom transformation with optional center point.
        
        Updates scale factor and adjusts origin if zoom center is specified
        to zoom towards the specified point.
        
        Args:
            zoom_factor (float): Zoom multiplier (>1 = zoom in, <1 = zoom out)
            zoom_center_screen (Position, optional): Screen point to zoom towards
        """
        if zoom_center_screen is not None:
            # Store zoom state for complex zoom-towards-point operations
            self.zoom_point = zoom_center_screen
            self.zoom_direction = -1 if zoom_factor > 1 else 1
        
        self.scale_factor *= zoom_factor
        
        # Ensure scale factor stays within reasonable bounds
        self.scale_factor = max(0.01, min(100.0, self.scale_factor))
    
    def apply_zoom_step(self, direction, zoom_center_screen=None):
        """Apply a standard zoom step in the specified direction.
        
        Uses the configured zoom_step to zoom in or out by a consistent amount.
        
        Args:
            direction (int): Zoom direction (-1 = zoom in, 1 = zoom out)
            zoom_center_screen (Position, optional): Screen point to zoom towards
        """
        zoom_factor = (1 + self.zoom_step) if direction == -1 else (1 - self.zoom_step)
        self.apply_zoom(zoom_factor, zoom_center_screen)
    
    def apply_pan(self, dx, dy):
        """Apply pan offset to the coordinate system.
        
        Adds the specified offset to the current pan state.
        
        Args:
            dx (float): Horizontal pan offset in screen pixels
            dy (float): Vertical pan offset in screen pixels
        """
        self.offset.x += dx
        self.offset.y += dy
    
    def reset_pan(self):
        """Reset pan offset to zero."""
        self.offset = Position(0, 0)
    
    def reset_transformations(self):
        """Reset all transformations to default state."""
        self.scale_factor = 1.0
        self.offset = Position(0, 0)
        self.origin = Position(self.canvas_width / 2, self.canvas_height / 2)
        self.zoom_point = Position(0, 0)
        self.zoom_direction = 0
    
    def get_visible_bounds(self):
        """Get mathematical bounds of the currently visible area.
        
        Calculates the mathematical coordinate range that is currently
        visible on the canvas, useful for optimization and clipping.
        
        Returns:
            dict: Bounds with 'left', 'right', 'top', 'bottom' keys
        """
        # Calculate bounds using screen corners
        left_bound, top_bound = self.screen_to_math(0, 0)
        right_bound, bottom_bound = self.screen_to_math(self.canvas_width, self.canvas_height)
        
        return {
            'left': left_bound,
            'right': right_bound,
            'top': top_bound,
            'bottom': bottom_bound
        }
    
    def get_visible_width(self):
        """Get mathematical width of visible area."""
        bounds = self.get_visible_bounds()
        return bounds['right'] - bounds['left']
    
    def get_visible_height(self):
        """Get mathematical height of visible area."""
        bounds = self.get_visible_bounds()
        return bounds['top'] - bounds['bottom']
    
    def get_visible_left_bound(self):
        """Get mathematical left boundary of visible area.
        
        Matches cartesian2axis.get_visible_left_bound() pattern.
        """
        return -(self.origin.x + self.offset.x) / self.scale_factor
    
    def get_visible_right_bound(self):
        """Get mathematical right boundary of visible area.
        
        Matches cartesian2axis.get_visible_right_bound() pattern.
        """
        return (self.canvas_width - self.origin.x - self.offset.x) / self.scale_factor
    
    def get_visible_top_bound(self):
        """Get mathematical top boundary of visible area.
        
        Matches cartesian2axis.get_visible_top_bound() pattern.
        """
        return (self.origin.y + self.offset.y) / self.scale_factor
    
    def get_visible_bottom_bound(self):
        """Get mathematical bottom boundary of visible area.
        
        Matches cartesian2axis.get_visible_bottom_bound() pattern.
        """
        return (self.origin.y + self.offset.y - self.canvas_height) / self.scale_factor
    
    def convert_canvas_x_to_math(self, canvas_x):
        """Convert canvas x-coordinate to mathematical x-coordinate.
        
        Matches _canvas_to_original_x() pattern found in colored areas.
        """
        return (canvas_x - self.origin.x - self.offset.x) / self.scale_factor
    
    def convert_math_y_to_canvas(self, math_y):
        """Convert mathematical y-coordinate to canvas y-coordinate.
        
        Matches _original_to_canvas_y() pattern found in colored areas.
        """
        return self.origin.y - math_y * self.scale_factor + self.offset.y
    
    def convert_math_x_to_canvas(self, math_x):
        """Convert mathematical x-coordinate to canvas x-coordinate.
        
        Matches coordinate conversion patterns found in functions.
        """
        return self.origin.x + math_x * self.scale_factor + self.offset.x
    
    def is_point_visible(self, screen_x, screen_y):
        """Check if a screen point is within canvas bounds.
        
        Args:
            screen_x (float): Screen x-coordinate in pixels
            screen_y (float): Screen y-coordinate in pixels
            
        Returns:
            bool: True if point is visible within canvas bounds
        """
        return (0 <= screen_x <= self.canvas_width) and (0 <= screen_y <= self.canvas_height)
    
    def is_math_point_visible(self, math_x, math_y):
        """Check if a mathematical point is visible in current viewport.
        
        Args:
            math_x (float): Mathematical x-coordinate
            math_y (float): Mathematical y-coordinate
            
        Returns:
            bool: True if point is visible in current viewport
        """
        screen_x, screen_y = self.math_to_screen(math_x, math_y)
        return self.is_point_visible(screen_x, screen_y)
    
    def update_canvas_size(self, width, height):
        """Update canvas dimensions and recalculate origin.
        
        Args:
            width (float): New canvas width in pixels
            height (float): New canvas height in pixels
        """
        self.canvas_width = width
        self.canvas_height = height
        self.origin = Position(width / 2, height / 2)
    
    def get_zoom_towards_point_displacement(self, target_point_screen):
        """Calculate displacement for zoom-towards-point operation.
        
        This implements the complex zoom logic that was scattered across
        individual drawable classes.
        
        Args:
            target_point_screen (Position): Current screen position of target
            
        Returns:
            Position: Displacement offset for the target point
        """
        if self.zoom_direction == 0:
            return Position(0, 0)
        
        # Calculate distance using standard Euclidean formula
        dx = self.zoom_point.x - target_point_screen.x
        dy = self.zoom_point.y - target_point_screen.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        displacement_magnitude = distance * self.zoom_step * self.zoom_direction
        
        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance
            return Position(displacement_magnitude * dx, displacement_magnitude * dy)
        
        return Position(0, 0)
    
    def get_state(self):
        """Get current transformation state for serialization.
        
        Returns:
            dict: Current coordinate mapper state
        """
        return {
            'canvas_width': self.canvas_width,
            'canvas_height': self.canvas_height,
            'scale_factor': self.scale_factor,
            'offset': {'x': self.offset.x, 'y': self.offset.y},
            'origin': {'x': self.origin.x, 'y': self.origin.y},
            'zoom_point': {'x': self.zoom_point.x, 'y': self.zoom_point.y},
            'zoom_direction': self.zoom_direction,
            'zoom_step': self.zoom_step
        }
    
    def set_state(self, state):
        """Set coordinate mapper state from dictionary.
        
        Args:
            state (dict): State dictionary with mapper properties
        """
        # Update canvas dimensions if provided
        self.canvas_width = state.get('canvas_width', self.canvas_width)
        self.canvas_height = state.get('canvas_height', self.canvas_height)
        
        self.scale_factor = state.get('scale_factor', 1.0)
        offset_data = state.get('offset', {'x': 0, 'y': 0})
        self.offset = Position(offset_data['x'], offset_data['y'])
        origin_data = state.get('origin', {'x': self.canvas_width / 2, 'y': self.canvas_height / 2})
        self.origin = Position(origin_data['x'], origin_data['y'])
        
        # Zoom state
        zoom_point_data = state.get('zoom_point', {'x': 0, 'y': 0})
        self.zoom_point = Position(zoom_point_data['x'], zoom_point_data['y'])
        self.zoom_direction = state.get('zoom_direction', 0)
        self.zoom_step = state.get('zoom_step', 0.1)
        
    def sync_from_canvas(self, canvas):
        """Synchronize coordinate mapper state with Canvas object.
        
        This method extracts coordinate transformation state from a Canvas
        object to ensure the CoordinateMapper is using the same values.
        
        Args:
            canvas: Canvas object with scale_factor, offset, center, etc.
        """
        # Update basic transformation parameters
        self.scale_factor = getattr(canvas, 'scale_factor', 1.0)
        
        # Handle offset - Canvas uses Position objects
        canvas_offset = getattr(canvas, 'offset', None)
        if canvas_offset:
            self.offset = Position(canvas_offset.x, canvas_offset.y)
        else:
            self.offset = Position(0, 0)
            
        # Update canvas dimensions first if they've changed
        if hasattr(canvas, 'width') and hasattr(canvas, 'height'):
            self.canvas_width = canvas.width
            self.canvas_height = canvas.height
            
        # Handle origin - Canvas may use cartesian2axis.origin or center
        if hasattr(canvas, 'cartesian2axis') and hasattr(canvas.cartesian2axis, 'origin'):
            cartesian_origin = canvas.cartesian2axis.origin
            self.origin = Position(cartesian_origin.x, cartesian_origin.y)
        else:
            canvas_center = getattr(canvas, 'center', None)
            if canvas_center:
                self.origin = Position(canvas_center.x, canvas_center.y)
            else:
                self.origin = Position(self.canvas_width / 2, self.canvas_height / 2)
                
        # Handle zoom state if available
        if hasattr(canvas, 'zoom_point'):
            zoom_point = canvas.zoom_point
            self.zoom_point = Position(zoom_point.x, zoom_point.y)
        if hasattr(canvas, 'zoom_direction'):
            self.zoom_direction = canvas.zoom_direction
        if hasattr(canvas, 'zoom_step'):
            self.zoom_step = canvas.zoom_step 