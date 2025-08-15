"""
MatHud Canvas Event Management System

Handles all user interactions with the mathematical canvas including mouse events, keyboard input,
and coordinate system navigation. Provides smooth zoom, pan, and click detection capabilities.

Key Features:
    - Mouse wheel zooming with dynamic zoom point tracking
    - Canvas panning via mouse drag operations
    - Double-click coordinate capture for precise input
    - Throttled mouse movement for performance optimization
    - Chat interface keyboard shortcuts (Enter key)
    - Error handling for robust user experience

Event Types:
    - Wheel: Zoom in/out with scale factor adjustments
    - Mouse down/up: Drag initialization and termination
    - Mouse move: Canvas panning and coordinate tracking
    - Key press: Chat input shortcuts and navigation
    - Double-click: Coordinate capture for mathematical input

Dependencies:
    - browser: DOM event handling and element access
    - constants: Timing thresholds and scaling factors
    - geometry: Position calculations for coordinate systems
"""

from browser import document, window
from constants import (
    double_click_threshold_s,
    zoom_in_scale_factor,
    zoom_out_scale_factor,
    mousemove_throttle_ms
)
from geometry import Position
import time


def throttle(wait_ms):
    """
    Decorator factory that creates a throttle decorator with specified wait time.
    Throttling ensures the function is called at a regular interval, unlike
    debouncing which waits for a pause in calls.
    
    Args:
        wait_ms: The minimum time between function calls in milliseconds
    
    Returns:
        A decorator function that will throttle the decorated function
    """
    def decorator(func):
        last_call = None
        queued = None
        
        def throttled(*args, **kwargs):
            nonlocal last_call, queued
            current_time = window.performance.now()  # Get high-resolution timestamp in ms
            
            try:
                if queued is not None:
                    window.clearTimeout(queued)
                    queued = None

                if last_call is None:
                    # First call executes immediately
                    last_call = current_time
                    func(*args, **kwargs)
                else:
                    elapsed = current_time - last_call
                    if elapsed >= wait_ms:
                        # If enough time has passed, execute immediately
                        last_call = current_time
                        func(*args, **kwargs)
                    else:
                        # Schedule to run at next interval
                        remaining_time = wait_ms - elapsed
                        queued = window.setTimeout(
                            lambda: throttled(*args, **kwargs),
                            remaining_time
                        )
            except Exception as e:
                print(f"Error in throttle: {str(e)}")
        
        return throttled
    
    return decorator


class CanvasEventHandler:
    """Manages all user interaction events for the mathematical canvas interface.
    
    Coordinates mouse and keyboard events to provide intuitive navigation, input capture,
    and canvas manipulation capabilities. Implements performance optimizations through
    event throttling and efficient coordinate calculations.
    
    Attributes:
        canvas (Canvas): Mathematical canvas for visualization and state updates
        ai_interface (AIInterface): Communication interface for user input processing
        last_click_timestamp (float): Timestamp of last click for double-click detection
        current_mouse_position (Position): Current mouse coordinates for drag calculations
        touch_start_positions (list): List of touch start positions for multi-touch gestures
        initial_pinch_distance (float): Initial distance between two fingers for pinch-to-zoom
        last_pinch_distance (float): Last recorded distance for pinch gesture
    """
    def __init__(self, canvas, ai_interface):
        """Initialize event handler with canvas and AI interface integration.
        
        Sets up event bindings for all supported user interactions including mouse
        events, keyboard shortcuts, touch events, and coordinate system navigation.
        
        Args:
            canvas (Canvas): The mathematical canvas to handle events for
            ai_interface (AIInterface): Interface for processing user input and interactions
        """
        self.canvas = canvas
        self.ai_interface = ai_interface
        self.last_click_timestamp = None
        self.current_mouse_position = None
        # Touch support attributes
        self.touch_start_positions = []
        self.initial_pinch_distance = None
        self.last_pinch_distance = None
        self.bind_events()
    
    def bind_events(self):
        """Bind all event handlers with error handling."""
        try:
            document["send-button"].bind("click", self.ai_interface.interact_with_ai)
            document["chat-input"].bind("keypress", self.check_enter)
            document["math-svg"].bind("wheel", self.handle_wheel)
            document["math-svg"].bind("mousedown", self.handle_mousedown)
            document["math-svg"].bind("mouseup", self.handle_mouseup)
            document["math-svg"].bind("mousemove", self.handle_mousemove)
            document["new-conversation-button"].bind("click", self.ai_interface.start_new_conversation)
            
            # Mobile touch events
            document["math-svg"].bind("touchstart", self.handle_touchstart)
            document["math-svg"].bind("touchend", self.handle_touchend)
            document["math-svg"].bind("touchmove", self.handle_touchmove)
            document["math-svg"].bind("touchcancel", self.handle_touchcancel)
            
            # Prevent default touch behaviors that interfere with canvas interaction
            document["math-svg"].style.touchAction = "none"
            
        except Exception as e:
            print(f"Error binding events: {str(e)}")

    def check_enter(self, event):
        """Handle enter key press in chat input."""
        try:
            if event.keyCode == 13:  # 13 is the key code for Enter
                self.ai_interface.interact_with_ai(event)
        except Exception as e:
            print(f"Error handling enter key: {str(e)}")

    def handle_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        try:
            self._update_zoom_point(event)
            self._adjust_scale_factor(event.deltaY)
            self.canvas.draw(True)
        except Exception as e:
            print(f"Error handling wheel event: {str(e)}")
    
    def _update_zoom_point(self, event):
        """Update the zoom point based on mouse position."""
        try:
            svg_canvas = document['math-svg']
            rect = svg_canvas.getBoundingClientRect()
            # Save the current zoom point and update it to the mouse position
            self.canvas.zoom_point = Position(event.clientX - rect.left, event.clientY - rect.top)
        except Exception as e:
            print(f"Error updating zoom point: {str(e)}")
    
    def _adjust_scale_factor(self, delta_y):
        """Adjust scale factor based on wheel movement direction."""
        try:
            if delta_y < 0:
                self._zoom_in()
            else:
                self._zoom_out()
        except Exception as e:
            print(f"Error adjusting scale factor: {str(e)}")
    
    def _zoom_in(self):
        """Apply zoom in with cursor anchoring using math/screen mapping."""
        try:
            self._apply_zoom_with_anchor(zoom_in_scale_factor)
        except Exception as e:
            print(f"Error zooming in: {str(e)}")
    
    def _zoom_out(self):
        """Apply zoom out with cursor anchoring using math/screen mapping."""
        try:
            self._apply_zoom_with_anchor(zoom_out_scale_factor)
        except Exception as e:
            print(f"Error zooming out: {str(e)}")

    def _apply_zoom_with_anchor(self, zoom_factor):
        """Zoom keeping the current mouse position fixed on screen."""
        cm = self.canvas.coordinate_mapper
        # Use last updated zoom point (screen coords)
        zp = self.canvas.zoom_point
        if zp is None:
            # Fallback: zoom about canvas center
            from geometry import Position
            zp = Position(self.canvas.width / 2, self.canvas.height / 2)
            self.canvas.zoom_point = zp

        # Compute math coords under cursor before scaling
        pre_mx, pre_my = cm.screen_to_math(zp.x, zp.y)

        # Update scale with clamping
        new_scale = cm.scale_factor * zoom_factor
        new_scale = max(0.01, min(100.0, new_scale))
        cm.scale_factor = new_scale

        # Adjust offset to keep the same math point under cursor
        cm.offset.x = zp.x - cm.origin.x - pre_mx * cm.scale_factor
        cm.offset.y = zp.y - cm.origin.y + pre_my * cm.scale_factor

        # Ensure no legacy point displacement runs
        self.canvas.zoom_direction = 0

    def get_decimal_places(self, value):
        """Calculate appropriate decimal places for coordinate display."""
        try:
            abs_val = abs(value)
            
            if 0 < abs_val < 1:
                return self._get_decimal_places_for_fraction(abs_val)
            else:
                return self._get_decimal_places_for_integer(abs_val)
        except Exception as e:
            print(f"Error calculating decimal places: {str(e)}")
            return 2  # Default to 2 decimal places on error
    
    def _get_decimal_places_for_fraction(self, value):
        """Calculate decimal places for fractional values."""
        try:
            decimal_part = format(value, ".10f").split(".")[1]
            leading_zeros = len(decimal_part) - len(decimal_part.lstrip('0'))
            return leading_zeros + 2
        except Exception as e:
            print(f"Error calculating decimal places for fraction: {str(e)}")
            return 2
    
    def _get_decimal_places_for_integer(self, value):
        """Calculate decimal places for integer or larger values."""
        try:
            if value < 10:
                return 2
            elif value < 100:
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error calculating decimal places for integer: {str(e)}")
            return 0

    def handle_mousedown(self, event):
        """Handle mouse down events for panning and coordinate capture."""
        try:
            current_timestamp = time.time()
            
            if self._is_double_click(current_timestamp):
                self._handle_double_click(event)
                
            self.last_click_timestamp = current_timestamp
            self._initialize_dragging(event)
        except Exception as e:
            print(f"Error handling mousedown: {str(e)}")
            self.canvas.dragging = False
    
    def _is_double_click(self, current_timestamp):
        """Determine if this is a double click based on timing."""
        try:
            return self.last_click_timestamp and (current_timestamp - self.last_click_timestamp) < double_click_threshold_s
        except Exception as e:
            print(f"Error detecting double click: {str(e)}")
            return False
    
    def _handle_double_click(self, event):
        """Handle double click action - capture coordinates."""
        try:
            coordinates = self._calculate_click_coordinates(event)
            self._add_coordinates_to_chat(coordinates)
        except Exception as e:
            print(f"Error handling double click: {str(e)}")
    
    def _calculate_click_coordinates(self, event):
        """Calculate cartesian coordinates from click position."""
        try:
            rect = document["math-svg"].getBoundingClientRect()
            canvas_x = event.clientX - rect.left
            canvas_y = event.clientY - rect.top
            scale_factor = self.canvas.scale_factor
            origin = self.canvas.cartesian2axis.origin
            
            x = (canvas_x - origin.x) * 1/scale_factor
            y = (origin.y - canvas_y) * 1/scale_factor
            
            decimal_places_x = self.get_decimal_places(x)
            decimal_places_y = self.get_decimal_places(y)
            
            x = round(x, decimal_places_x)
            y = round(y, decimal_places_y)
            
            return f"({x}, {y}) "
        except Exception as e:
            print(f"Error calculating coordinates: {str(e)}")
            return "(error) "
    
    def _add_coordinates_to_chat(self, coordinates):
        """Add the coordinates to the chat input field."""
        try:
            document["chat-input"].value += coordinates
        except Exception as e:
            print(f"Error adding coordinates to chat: {str(e)}")
    
    def _initialize_dragging(self, event):
        """Initialize dragging state with current mouse position."""
        try:
            self.canvas.dragging = True
            self.current_mouse_position = Position(event.clientX, event.clientY)
            self.canvas.last_mouse_position = self.current_mouse_position
        except Exception as e:
            print(f"Error initializing dragging: {str(e)}")
            self.canvas.dragging = False

    def handle_mouseup(self, event):
        """Handle mouse up events."""
        try:
            self.canvas.dragging = False
            self.current_mouse_position = None
        except Exception as e:
            print(f"Error handling mouseup: {str(e)}")

    def handle_mousemove(self, event):
        """Handle mouse movement for canvas panning."""
        try:
            if not self.canvas.dragging:
                return
                
            self._update_mouse_position(event)
            self._update_canvas_position(event)
        except Exception as e:
            print(f"Error handling mousemove: {str(e)}")
    
    def _update_mouse_position(self, event):
        """Update the tracked mouse position."""
        try:
            self.current_mouse_position = Position(event.clientX, event.clientY)
        except Exception as e:
            print(f"Error updating mouse position: {str(e)}")
    
    @throttle(mousemove_throttle_ms)
    def _update_canvas_position(self, event):
        """Update canvas position with throttling for smooth performance."""
        try:
            if self.current_mouse_position and self.canvas.last_mouse_position:
                offset = self._calculate_drag_offset()
                self._apply_offset_to_canvas(offset)
                self._update_last_mouse_position()
                self.canvas.draw(False)
        except Exception as e:
            print(f"Error updating canvas position: {str(e)}")
    
    def _calculate_drag_offset(self):
        """Calculate the drag offset based on mouse movement."""
        try:
            dx = self.current_mouse_position.x - self.canvas.last_mouse_position.x
            dy = self.current_mouse_position.y - self.canvas.last_mouse_position.y
            return Position(dx, dy)
        except Exception as e:
            print(f"Error calculating drag offset: {str(e)}")
            return Position(0, 0)
    
    def _apply_offset_to_canvas(self, offset):
        """Apply the calculated offset to the canvas."""
        try:
            self.canvas.offset.x += offset.x
            self.canvas.offset.y += offset.y
        except Exception as e:
            print(f"Error applying offset to canvas: {str(e)}")
    
    def _update_last_mouse_position(self):
        """Update the last mouse position to the current position."""
        try:
            self.canvas.last_mouse_position = self.current_mouse_position
        except Exception as e:
            print(f"Error updating last mouse position: {str(e)}")

    def handle_touchstart(self, event):
        """Handle touch start events for mobile panning and zooming."""
        try:
            event.preventDefault()  # Prevent default scrolling/zooming
            
            touches = event.touches
            if len(touches) == 1:
                # Single touch - start panning (similar to mousedown)
                touch = touches[0]
                self._handle_single_touch_start(touch)
            elif len(touches) == 2:
                # Two fingers - start pinch-to-zoom
                self._handle_pinch_start(touches)
                
        except Exception as e:
            print(f"Error handling touchstart: {str(e)}")
    
    def _handle_single_touch_start(self, touch):
        """Handle single touch start for panning."""
        try:
            current_timestamp = time.time()
            
            # Check for double tap (similar to double click)
            if self._is_double_click(current_timestamp):
                self._handle_double_tap(touch)
                
            self.last_click_timestamp = current_timestamp
            self._initialize_touch_dragging(touch)
        except Exception as e:
            print(f"Error handling single touch start: {str(e)}")
    
    def _handle_pinch_start(self, touches):
        """Initialize pinch-to-zoom gesture."""
        try:
            touch1, touch2 = touches[0], touches[1]
            
            # Calculate initial distance between fingers
            self.initial_pinch_distance = self._calculate_touch_distance(touch1, touch2)
            self.last_pinch_distance = self.initial_pinch_distance
            
            # Set zoom point to center between the two touches
            center_x = (touch1.clientX + touch2.clientX) / 2
            center_y = (touch1.clientY + touch2.clientY) / 2
            
            rect = document["math-svg"].getBoundingClientRect()
            self.canvas.zoom_point = Position(center_x - rect.left, center_y - rect.top)
            
        except Exception as e:
            print(f"Error handling pinch start: {str(e)}")
    
    def _calculate_touch_distance(self, touch1, touch2):
        """Calculate distance between two touch points."""
        try:
            dx = touch2.clientX - touch1.clientX
            dy = touch2.clientY - touch1.clientY
            return (dx * dx + dy * dy) ** 0.5
        except Exception as e:
            print(f"Error calculating touch distance: {str(e)}")
            return 0

    def handle_touchmove(self, event):
        """Handle touch move events for panning and pinch-to-zoom."""
        try:
            event.preventDefault()  # Prevent default scrolling
            
            touches = event.touches
            if len(touches) == 1 and self.canvas.dragging:
                # Single finger - panning
                self._handle_touch_pan(touches[0])
            elif len(touches) == 2:
                # Two fingers - pinch-to-zoom
                self._handle_pinch_zoom(touches)
                
        except Exception as e:
            print(f"Error handling touchmove: {str(e)}")
    
    def _handle_touch_pan(self, touch):
        """Handle single finger panning."""
        try:
            # Create a mock event object similar to mouse event
            mock_event = type('obj', (object,), {
                'clientX': touch.clientX,
                'clientY': touch.clientY
            })
            
            self._update_mouse_position(mock_event)
            self._update_canvas_position(mock_event)
        except Exception as e:
            print(f"Error handling touch pan: {str(e)}")
    
    def _handle_pinch_zoom(self, touches):
        """Handle two-finger pinch-to-zoom."""
        try:
            touch1, touch2 = touches[0], touches[1]
            current_distance = self._calculate_touch_distance(touch1, touch2)
            
            if self.last_pinch_distance and current_distance > 0:
                # Calculate scale change
                scale_change = current_distance / self.last_pinch_distance
                
                # Apply zoom with sensitivity adjustment
                if scale_change > 1.02:  # Zoom in threshold
                    self.canvas.scale_factor *= 1.02
                    self.canvas.zoom_direction = -1
                    self.canvas.draw(True)
                elif scale_change < 0.98:  # Zoom out threshold
                    self.canvas.scale_factor *= 0.98
                    self.canvas.zoom_direction = 1
                    self.canvas.draw(True)
                    
            self.last_pinch_distance = current_distance
            
        except Exception as e:
            print(f"Error handling pinch zoom: {str(e)}")

    def handle_touchend(self, event):
        """Handle touch end events."""
        try:
            event.preventDefault()
            
            # Reset dragging state
            self.canvas.dragging = False
            self.current_mouse_position = None
            
            # Reset pinch state
            self.initial_pinch_distance = None
            self.last_pinch_distance = None
            
        except Exception as e:
            print(f"Error handling touchend: {str(e)}")

    def handle_touchcancel(self, event):
        """Handle touch cancel events (same as touch end)."""
        try:
            self.handle_touchend(event)
        except Exception as e:
            print(f"Error handling touchcancel: {str(e)}")
    
    def _handle_double_tap(self, touch):
        """Handle double tap action - capture coordinates."""
        try:
            # Create a mock event object similar to mouse event
            mock_event = type('obj', (object,), {
                'clientX': touch.clientX,
                'clientY': touch.clientY
            })
            
            coordinates = self._calculate_click_coordinates(mock_event)
            self._add_coordinates_to_chat(coordinates)
        except Exception as e:
            print(f"Error handling double tap: {str(e)}")
    
    def _initialize_touch_dragging(self, touch):
        """Initialize dragging state with current touch position."""
        try:
            self.canvas.dragging = True
            self.current_mouse_position = Position(touch.clientX, touch.clientY)
            self.canvas.last_mouse_position = self.current_mouse_position
        except Exception as e:
            print(f"Error initializing touch dragging: {str(e)}")
            self.canvas.dragging = False 