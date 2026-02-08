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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import math

from drawables.position import Position

if TYPE_CHECKING:
    from canvas import Canvas


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

    def __init__(self, canvas_width: float, canvas_height: float) -> None:
        """Initialize coordinate mapper with canvas dimensions.

        Args:
            canvas_width (float): Canvas viewport width in pixels
            canvas_height (float): Canvas viewport height in pixels
        """
        self.canvas_width: float = canvas_width
        self.canvas_height: float = canvas_height
        self.scale_factor: float = 1.0
        self.offset: Position = Position(0, 0)
        self.origin: Position = Position(canvas_width / 2, canvas_height / 2)

        # Zoom state management
        self.zoom_point: Position = Position(0, 0)
        self.zoom_direction: int = 0
        self.zoom_step: float = 0.1

    @classmethod
    def from_canvas(cls: type["CoordinateMapper"], canvas: "Canvas") -> "CoordinateMapper":
        """Create a CoordinateMapper from an existing Canvas object.

        This factory method extracts coordinate transformation state from a Canvas
        to create a properly initialized CoordinateMapper.

        Args:
            canvas: Canvas object with coordinate transformation state

        Returns:
            CoordinateMapper: Initialized with Canvas state
        """
        # Create mapper with canvas dimensions
        mapper: "CoordinateMapper" = cls(canvas.width, canvas.height)

        # Sync with canvas state
        mapper.sync_from_canvas(canvas)

        return mapper

    def math_to_screen(self, math_x: float, math_y: float) -> Tuple[float, float]:
        """Convert mathematical coordinates to screen pixel coordinates.

        Applies scale factor, origin translation, and pan offset with Y-axis flipping
        to match mathematical coordinate system conventions.

        Args:
            math_x (float): Mathematical x-coordinate
            math_y (float): Mathematical y-coordinate

        Returns:
            tuple: (screen_x, screen_y) in pixel coordinates
        """
        screen_x: float = self.origin.x + (math_x * self.scale_factor) + self.offset.x
        screen_y: float = self.origin.y - (math_y * self.scale_factor) + self.offset.y
        return screen_x, screen_y

    def screen_to_math(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen pixel coordinates to mathematical coordinates.

        Reverses scale factor, origin translation, and pan offset with Y-axis flipping
        to get original mathematical values.

        Args:
            screen_x (float): Screen x-coordinate in pixels
            screen_y (float): Screen y-coordinate in pixels

        Returns:
            tuple: (math_x, math_y) in mathematical coordinates
        """
        math_x: float = (screen_x - self.offset.x - self.origin.x) / self.scale_factor
        math_y: float = (self.origin.y + self.offset.y - screen_y) / self.scale_factor
        return math_x, math_y

    def scale_value(self, math_value: float) -> float:
        """Scale a mathematical value to screen units.

        Used for scaling mathematical properties like radius, distance,
        or any other measurement that needs to scale with zoom level.

        Args:
            math_value (float): Mathematical value to scale

        Returns:
            float: Scaled value in screen units
        """
        return math_value * self.scale_factor

    def unscale_value(self, screen_value: float) -> float:
        """Convert a screen value back to mathematical units.

        Inverse of scale_value, useful for converting measurements
        back to mathematical space.

        Args:
            screen_value (float): Screen value to unscale

        Returns:
            float: Value in mathematical units
        """
        return screen_value / self.scale_factor

    def apply_zoom(self, zoom_factor: float, zoom_center_screen: Optional[Position] = None) -> None:
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
        # Ensure scale factor stays above a minimal positive value; no arbitrary upper cap
        self.scale_factor = max(0.01, self.scale_factor)

    def apply_zoom_step(self, direction: int, zoom_center_screen: Optional[Position] = None) -> None:
        """Apply a standard zoom step in the specified direction.

        Uses the configured zoom_step to zoom in or out by a consistent amount.

        Args:
            direction (int): Zoom direction (-1 = zoom in, 1 = zoom out)
            zoom_center_screen (Position, optional): Screen point to zoom towards
        """
        zoom_factor: float = (1 + self.zoom_step) if direction == -1 else (1 - self.zoom_step)
        self.apply_zoom(zoom_factor, zoom_center_screen)

    def apply_pan(self, dx: float, dy: float) -> None:
        """Apply pan offset to the coordinate system.

        Adds the specified offset to the current pan state.

        Args:
            dx (float): Horizontal pan offset in screen pixels
            dy (float): Vertical pan offset in screen pixels
        """
        self.offset.x += dx
        self.offset.y += dy

    def reset_pan(self) -> None:
        """Reset pan offset to zero."""
        self.offset = Position(0, 0)

    def reset_transformations(self) -> None:
        """Reset all transformations to default state."""
        self.scale_factor = 1.0
        self.offset = Position(0, 0)
        self.origin = Position(self.canvas_width / 2, self.canvas_height / 2)
        self.zoom_point = Position(0, 0)
        self.zoom_direction = 0

    def set_visible_bounds(self, left_bound: float, right_bound: float, top_bound: float, bottom_bound: float) -> None:
        """Fit the viewport to the requested math bounds while preserving aspect ratio.

        The new scale factor is chosen so that the limiting axis (horizontal or vertical)
        exactly matches the requested span, guaranteeing the entire rectangle remains
        visible. Offsets are recalculated so the bounds are centered in view and any
        lingering zoom-towards state is cleared.
        """
        try:
            left: float = float(left_bound)
            right: float = float(right_bound)
            top: float = float(top_bound)
            bottom: float = float(bottom_bound)
        except (TypeError, ValueError):
            raise ValueError("Bounds must be numeric values")

        if not (left < right and bottom < top):
            raise ValueError("Bounds must satisfy left < right and bottom < top")

        width: float = right - left
        height: float = top - bottom

        if width <= 0 or height <= 0:
            raise ValueError("Bounds must define a positive area")

        scale_x: float = self.canvas_width / width
        scale_y: float = self.canvas_height / height
        new_scale: float = min(scale_x, scale_y)
        self.scale_factor = max(new_scale, 1e-9)

        center_x: float = (left + right) / 2.0
        center_y: float = (top + bottom) / 2.0

        self.offset.x = -center_x * self.scale_factor
        self.offset.y = center_y * self.scale_factor
        self.zoom_point = Position(0, 0)
        self.zoom_direction = 0

    def get_visible_bounds(self) -> Dict[str, float]:
        """Get mathematical bounds of the currently visible area.

        Calculates the mathematical coordinate range that is currently
        visible on the canvas, useful for optimization and clipping.

        Returns:
            dict: Bounds with 'left', 'right', 'top', 'bottom' keys
        """
        # Calculate bounds using screen corners
        left_bound: float
        top_bound: float
        right_bound: float
        bottom_bound: float
        left_bound, top_bound = self.screen_to_math(0, 0)
        right_bound, bottom_bound = self.screen_to_math(self.canvas_width, self.canvas_height)

        return {
            'left': left_bound,
            'right': right_bound,
            'top': top_bound,
            'bottom': bottom_bound
        }

    def get_visible_width(self) -> float:
        """Get mathematical width of visible area."""
        bounds: Dict[str, float] = self.get_visible_bounds()
        return bounds['right'] - bounds['left']

    def get_visible_height(self) -> float:
        """Get mathematical height of visible area."""
        bounds: Dict[str, float] = self.get_visible_bounds()
        return bounds['top'] - bounds['bottom']

    def get_visible_left_bound(self) -> float:
        """Get mathematical left boundary of visible area.

        Matches cartesian2axis.get_visible_left_bound() pattern.
        """
        return cast(float, -(self.origin.x + self.offset.x) / self.scale_factor)

    def get_visible_right_bound(self) -> float:
        """Get mathematical right boundary of visible area.

        Matches cartesian2axis.get_visible_right_bound() pattern.
        """
        return cast(float, (self.canvas_width - self.origin.x - self.offset.x) / self.scale_factor)

    def get_visible_top_bound(self) -> float:
        """Get mathematical top boundary of visible area.

        Matches cartesian2axis.get_visible_top_bound() pattern.
        """
        return cast(float, (self.origin.y + self.offset.y) / self.scale_factor)

    def get_visible_bottom_bound(self) -> float:
        """Get mathematical bottom boundary of visible area.

        Matches cartesian2axis.get_visible_bottom_bound() pattern.
        """
        return cast(float, (self.origin.y + self.offset.y - self.canvas_height) / self.scale_factor)

    def convert_canvas_x_to_math(self, canvas_x: float) -> float:
        """Convert canvas x-coordinate to mathematical x-coordinate.

        Matches _canvas_to_original_x() pattern found in colored areas.
        """
        return cast(float, (canvas_x - self.origin.x - self.offset.x) / self.scale_factor)

    def convert_math_y_to_canvas(self, math_y: float) -> float:
        """Convert mathematical y-coordinate to canvas y-coordinate.

        Matches _original_to_canvas_y() pattern found in colored areas.
        """
        return cast(float, self.origin.y - math_y * self.scale_factor + self.offset.y)

    def convert_math_x_to_canvas(self, math_x: float) -> float:
        """Convert mathematical x-coordinate to canvas x-coordinate.

        Matches coordinate conversion patterns found in functions.
        """
        return cast(float, self.origin.x + math_x * self.scale_factor + self.offset.x)

    def is_point_visible(self, screen_x: float, screen_y: float) -> bool:
        """Check if a screen point is within canvas bounds.

        Args:
            screen_x (float): Screen x-coordinate in pixels
            screen_y (float): Screen y-coordinate in pixels

        Returns:
            bool: True if point is visible within canvas bounds
        """
        return (0 <= screen_x <= self.canvas_width) and (0 <= screen_y <= self.canvas_height)

    def is_math_point_visible(self, math_x: float, math_y: float) -> bool:
        """Check if a mathematical point is visible in current viewport.

        Args:
            math_x (float): Mathematical x-coordinate
            math_y (float): Mathematical y-coordinate

        Returns:
            bool: True if point is visible in current viewport
        """
        screen_x: float
        screen_y: float
        screen_x, screen_y = self.math_to_screen(math_x, math_y)
        return self.is_point_visible(screen_x, screen_y)

    def update_canvas_size(self, width: float, height: float) -> None:
        """Update canvas dimensions and recalculate origin.

        Args:
            width (float): New canvas width in pixels
            height (float): New canvas height in pixels
        """
        self.canvas_width = width
        self.canvas_height = height
        self.origin = Position(width / 2, height / 2)

    def get_zoom_towards_point_displacement(self, target_point_screen: Position) -> Position:
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
        dx: float = self.zoom_point.x - target_point_screen.x
        dy: float = self.zoom_point.y - target_point_screen.y
        distance: float = math.sqrt(dx * dx + dy * dy)

        displacement_magnitude: float = distance * self.zoom_step * self.zoom_direction

        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance
            return Position(displacement_magnitude * dx, displacement_magnitude * dy)

        return Position(0, 0)

    def get_state(self) -> Dict[str, Any]:
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

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set coordinate mapper state from dictionary.

        Args:
            state (dict): State dictionary with mapper properties
        """
        # Update canvas dimensions if provided
        self.canvas_width = state.get('canvas_width', self.canvas_width)
        self.canvas_height = state.get('canvas_height', self.canvas_height)

        self.scale_factor = state.get('scale_factor', 1.0)
        offset_data: Dict[str, float] = state.get('offset', {'x': 0, 'y': 0})
        self.offset = Position(offset_data['x'], offset_data['y'])
        origin_data: Dict[str, float] = state.get('origin', {'x': self.canvas_width / 2, 'y': self.canvas_height / 2})
        self.origin = Position(origin_data['x'], origin_data['y'])

        # Zoom state
        zoom_point_data: Dict[str, float] = state.get('zoom_point', {'x': 0, 'y': 0})
        self.zoom_point = Position(zoom_point_data['x'], zoom_point_data['y'])
        self.zoom_direction = state.get('zoom_direction', 0)
        self.zoom_step = state.get('zoom_step', 0.1)

    def sync_from_canvas(self, canvas: "Canvas") -> None:
        """Synchronize coordinate mapper state with Canvas object.

        This method extracts coordinate transformation state from a Canvas
        object to ensure the CoordinateMapper is using the same values.

        Args:
            canvas: Canvas object with scale_factor, offset, center, etc.
        """
        # Update basic transformation parameters
        self.scale_factor = getattr(canvas, 'scale_factor', 1.0)

        # Handle offset - Canvas uses Position objects
        canvas_offset: Any = getattr(canvas, 'offset', None)
        if canvas_offset:
            self.offset = Position(canvas_offset.x, canvas_offset.y)
        else:
            self.offset = Position(0, 0)

        # Update canvas dimensions first if they've changed
        if hasattr(canvas, 'width') and hasattr(canvas, 'height'):
            self.canvas_width = canvas.width
            self.canvas_height = canvas.height

        # Handle origin - Use canvas.center as the base origin (before offset)
        # Note: cartesian2axis.origin already includes offset via math_to_screen,
        # so we must use canvas.center to avoid double-counting offset
        canvas_center: Any = getattr(canvas, 'center', None)
        if canvas_center:
            self.origin = Position(canvas_center.x, canvas_center.y)
        else:
            self.origin = Position(self.canvas_width / 2, self.canvas_height / 2)

        # Handle zoom state if available
        if hasattr(canvas, 'zoom_point'):
            zoom_point: Any = canvas.zoom_point
            self.zoom_point = Position(zoom_point.x, zoom_point.y)
        if hasattr(canvas, 'zoom_direction'):
            self.zoom_direction = canvas.zoom_direction
        if hasattr(canvas, 'zoom_step'):
            self.zoom_step = canvas.zoom_step
