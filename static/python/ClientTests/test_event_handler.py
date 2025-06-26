import unittest
from canvas_event_handler import CanvasEventHandler
from .simple_mock import SimpleMock
from browser import window as browser_window
from geometry import Position
from canvas import Canvas
import time


class TestCanvasEventHandlerTouch(unittest.TestCase):
    """Test the touch event handling methods added for mobile support."""
    
    def setUp(self):
        """Set up test fixtures with mock canvas, AI interface, and document elements."""
        # Create mock canvas
        self.mock_canvas = Canvas(500, 500, draw_enabled=False)
        self.mock_canvas.dragging = False
        self.mock_canvas.scale_factor = 1.0
        self.mock_canvas.zoom_direction = 0
        self.mock_canvas.zoom_point = Position(0, 0)
        self.mock_canvas.last_mouse_position = None
        
        # Mock canvas methods
        self.mock_canvas.draw = SimpleMock()
        
        # Create mock AI interface
        self.mock_ai_interface = SimpleMock()
        self.mock_ai_interface.interact_with_ai = SimpleMock()
        self.mock_ai_interface.start_new_conversation = SimpleMock()
        
        # Mock document elements
        self.mock_svg_element = SimpleMock()
        self.mock_svg_element.getBoundingClientRect = SimpleMock(return_value=SimpleMock(left=10, top=20))
        self.mock_svg_element.style = SimpleMock()
        self.mock_svg_element.bind = SimpleMock()
        
        self.mock_chat_input = SimpleMock(value="")
        self.mock_chat_input.bind = SimpleMock()
        self.mock_send_button = SimpleMock()
        self.mock_send_button.bind = SimpleMock()
        self.mock_new_conversation_button = SimpleMock()
        self.mock_new_conversation_button.bind = SimpleMock()
        
        # Mock document
        self.mock_document = {
            "math-svg": self.mock_svg_element,
            "chat-input": self.mock_chat_input,
            "send-button": self.mock_send_button,
            "new-conversation-button": self.mock_new_conversation_button
        }
        
        # Replace the actual document with our mock
        import canvas_event_handler
        self.original_document = canvas_event_handler.document
        canvas_event_handler.document = self.mock_document
        
        # Create event handler
        self.event_handler = CanvasEventHandler(self.mock_canvas, self.mock_ai_interface)

    def tearDown(self):
        """Restore original document."""
        import canvas_event_handler
        canvas_event_handler.document = self.original_document

    def _create_mock_touch(self, client_x, client_y):
        """Helper to create a mock touch object."""
        return SimpleMock(clientX=client_x, clientY=client_y)

    def _create_mock_touch_event(self, touches, event_type="touchstart"):
        """Helper to create a mock touch event."""
        mock_event = SimpleMock()
        mock_event.type = event_type
        mock_event.touches = touches
        mock_event.preventDefault = SimpleMock()
        return mock_event

    def test_handle_touchstart_single_touch(self):
        """Test handling single finger touch start for panning."""
        # Create single touch event
        touch = self._create_mock_touch(100, 150)
        event = self._create_mock_touch_event([touch])
        
        # Handle touchstart
        self.event_handler.handle_touchstart(event)
        
        # Verify dragging was initialized
        self.assertTrue(self.mock_canvas.dragging)
        self.assertIsNotNone(self.event_handler.current_mouse_position)
        
        # Verify preventDefault was called
        self.assertEqual(len(event.preventDefault.calls), 1)

    def test_handle_touchstart_two_fingers_pinch(self):
        """Test handling two finger touch start for pinch-to-zoom."""
        # Create two touch event
        touch1 = self._create_mock_touch(100, 150)
        touch2 = self._create_mock_touch(200, 250)
        event = self._create_mock_touch_event([touch1, touch2])
        
        # Handle touchstart
        self.event_handler.handle_touchstart(event)
        
        # Verify pinch state was initialized
        self.assertIsNotNone(self.event_handler.initial_pinch_distance)
        self.assertIsNotNone(self.event_handler.last_pinch_distance)
        
        # Verify zoom point was set
        self.assertIsNotNone(self.mock_canvas.zoom_point)

    def test_calculate_touch_distance(self):
        """Test calculation of distance between two touch points."""
        touch1 = self._create_mock_touch(0, 0)
        touch2 = self._create_mock_touch(30, 40)
        
        # Use the actual method name with underscore
        distance = self.event_handler._calculate_touch_distance(touch1, touch2)
        
        # Distance should be 50 (3-4-5 triangle)
        self.assertEqual(distance, 50)

    def test_handle_touchmove_single_finger_panning(self):
        """Test handling single finger movement for panning."""
        # Setup initial dragging state
        self.mock_canvas.dragging = True
        self.event_handler.current_mouse_position = Position(90, 130)
        
        # Create move event
        touch = self._create_mock_touch(120, 170)
        event = self._create_mock_touch_event([touch], "touchmove")
        
        # Handle touchmove
        self.event_handler.handle_touchmove(event)
        
        # Verify preventDefault was called
        self.assertEqual(len(event.preventDefault.calls), 1)

    def test_handle_touchmove_pinch_zoom_in(self):
        """Test handling pinch gesture for zooming in."""
        # Setup initial pinch state
        self.event_handler.last_pinch_distance = 50
        self.event_handler.initial_pinch_distance = 50
        
        # Create move event with fingers further apart (zoom in)
        touch1 = self._create_mock_touch(80, 120)  # Moved further from center
        touch2 = self._create_mock_touch(220, 260)  # Moved further from center
        event = self._create_mock_touch_event([touch1, touch2], "touchmove")
        
        # Handle touchmove
        self.event_handler.handle_touchmove(event)
        
        # Verify preventDefault was called
        self.assertEqual(len(event.preventDefault.calls), 1)
        
        # Verify last_pinch_distance was updated
        self.assertIsNotNone(self.event_handler.last_pinch_distance)

    def test_handle_touchmove_pinch_zoom_out(self):
        """Test handling pinch gesture for zooming out."""
        # Setup initial pinch state
        self.event_handler.last_pinch_distance = 100
        self.event_handler.initial_pinch_distance = 100
        
        # Create move event with fingers closer together (zoom out)
        touch1 = self._create_mock_touch(110, 140)  # Moved closer to center
        touch2 = self._create_mock_touch(170, 220)  # Moved closer to center
        event = self._create_mock_touch_event([touch1, touch2], "touchmove")
        
        # Handle touchmove
        self.event_handler.handle_touchmove(event)
        
        # Verify preventDefault was called
        self.assertEqual(len(event.preventDefault.calls), 1)
        
        # Verify last_pinch_distance was updated
        self.assertIsNotNone(self.event_handler.last_pinch_distance)

    def test_handle_touchend(self):
        """Test handling touch end events."""
        # Setup touch dragging state
        self.mock_canvas.dragging = True
        self.event_handler.current_mouse_position = Position(90, 130)
        self.event_handler.initial_pinch_distance = 100
        self.event_handler.last_pinch_distance = 100
        
        # Create touchend event
        event = self._create_mock_touch_event([], "touchend")
        
        # Handle touchend
        self.event_handler.handle_touchend(event)
        
        # Verify state was reset
        self.assertFalse(self.mock_canvas.dragging)
        self.assertIsNone(self.event_handler.current_mouse_position)
        self.assertIsNone(self.event_handler.initial_pinch_distance)
        self.assertIsNone(self.event_handler.last_pinch_distance)

    def test_handle_touchcancel(self):
        """Test handling touch cancel events."""
        # Setup touch dragging state
        self.mock_canvas.dragging = True
        self.event_handler.current_mouse_position = Position(90, 130)
        
        # Create touchcancel event
        event = self._create_mock_touch_event([], "touchcancel")
        
        # Handle touchcancel
        self.event_handler.handle_touchcancel(event)
        
        # Verify state was reset (same as touchend)
        self.assertFalse(self.mock_canvas.dragging)
        self.assertIsNone(self.event_handler.current_mouse_position)

    def test_handle_double_tap(self):
        """Test handling double tap for coordinate capture."""
        # Create first tap
        touch = self._create_mock_touch(100, 150)
        
        # Set timestamp for first tap
        self.event_handler.last_click_timestamp = 1000
        
        # Mock time.time to return time within double-tap threshold
        original_time = time.time
        time.time = lambda: 1000.2  # 200ms later
        
        try:
            # Handle double tap directly
            self.event_handler._handle_double_tap(touch)
            
            # Verify chat input was updated
            self.assertTrue(len(self.mock_chat_input.value) > 0)
        finally:
            # Restore original time
            time.time = original_time

    def test_initialize_touch_dragging(self):
        """Test initialization of touch dragging state."""
        touch = self._create_mock_touch(100, 150)
        
        # Call initialize method
        self.event_handler._initialize_touch_dragging(touch)
        
        # Verify state was set correctly
        self.assertTrue(self.mock_canvas.dragging)
        self.assertIsNotNone(self.event_handler.current_mouse_position)
        self.assertEqual(self.event_handler.current_mouse_position.x, 100)
        self.assertEqual(self.event_handler.current_mouse_position.y, 150)

    def test_touch_action_none_set_in_bind_events(self):
        """Test that touch-action: none is set when binding events."""
        # This test verifies that the CSS property is set to prevent default touch behaviors
        # The actual binding happens in bind_events which is called in __init__
        self.assertEqual(self.mock_svg_element.style.touchAction, "none")

    def test_error_handling_in_touch_methods(self):
        """Test that touch methods handle errors gracefully."""
        # Create an event that might cause errors
        malformed_event = SimpleMock()
        malformed_event.touches = None  # This should cause an error
        malformed_event.preventDefault = SimpleMock()
        
        # These should not raise exceptions, but handle errors gracefully
        try:
            self.event_handler.handle_touchstart(malformed_event)
            self.event_handler.handle_touchmove(malformed_event)
            self.event_handler.handle_touchend(malformed_event)
            self.event_handler.handle_touchcancel(malformed_event)
        except Exception as e:
            self.fail(f"Touch methods should handle errors gracefully, but got: {e}") 