"""
Unit tests for CoordinateMapper class

Tests all coordinate transformation functionality including:
- Math-to-screen and screen-to-math conversions
- Scale value transformations
- Zoom and pan operations
- Visible bounds calculations
- State management
- Canvas integration and synchronization
"""

from __future__ import annotations

import unittest
from coordinate_mapper import CoordinateMapper
from drawables.position import Position
from client_tests.simple_mock import SimpleMock


class TestCoordinateMapper(unittest.TestCase):
    
    def setUp(self) -> None:
        """Set up test fixtures with standard canvas size."""
        self.canvas_width = 800
        self.canvas_height = 600
        self.mapper = CoordinateMapper(self.canvas_width, self.canvas_height)
    
    def test_initialization(self) -> None:
        """Test CoordinateMapper initialization with correct default values."""
        self.assertEqual(self.mapper.canvas_width, 800)
        self.assertEqual(self.mapper.canvas_height, 600)
        self.assertEqual(self.mapper.scale_factor, 1.0)
        self.assertEqual(self.mapper.offset.x, 0)
        self.assertEqual(self.mapper.offset.y, 0)
        self.assertEqual(self.mapper.origin.x, 400)  # width / 2
        self.assertEqual(self.mapper.origin.y, 300)  # height / 2
        self.assertEqual(self.mapper.zoom_direction, 0)
        self.assertEqual(self.mapper.zoom_step, 0.1)
    
    def test_math_to_screen_basic(self) -> None:
        """Test basic math-to-screen coordinate conversion."""
        # Origin should map to canvas center
        screen_x, screen_y = self.mapper.math_to_screen(0, 0)
        self.assertEqual(screen_x, 400)  # canvas center x
        self.assertEqual(screen_y, 300)  # canvas center y
        
        # Positive x should move right, positive y should move up (screen y decreases)
        screen_x, screen_y = self.mapper.math_to_screen(100, 50)
        self.assertEqual(screen_x, 500)  # 400 + 100
        self.assertEqual(screen_y, 250)  # 300 - 50 (Y flipped)
        
        # Negative coordinates
        screen_x, screen_y = self.mapper.math_to_screen(-100, -50)
        self.assertEqual(screen_x, 300)  # 400 - 100
        self.assertEqual(screen_y, 350)  # 300 + 50 (Y flipped)
    
    def test_screen_to_math_basic(self) -> None:
        """Test basic screen-to-math coordinate conversion."""
        # Canvas center should map to origin
        math_x, math_y = self.mapper.screen_to_math(400, 300)
        self.assertAlmostEqual(math_x, 0, places=6)
        self.assertAlmostEqual(math_y, 0, places=6)
        
        # Screen coordinates should convert back correctly
        math_x, math_y = self.mapper.screen_to_math(500, 250)
        self.assertAlmostEqual(math_x, 100, places=6)
        self.assertAlmostEqual(math_y, 50, places=6)
        
        # Negative coordinates
        math_x, math_y = self.mapper.screen_to_math(300, 350)
        self.assertAlmostEqual(math_x, -100, places=6)
        self.assertAlmostEqual(math_y, -50, places=6)
    
    def test_coordinate_conversion_roundtrip(self) -> None:
        """Test that math to screen to math conversion preserves values."""
        test_cases = [
            (0, 0),
            (100, 50),
            (-100, -50),
            (3.14159, -2.71828),
            (1000, -500)
        ]
        
        for orig_x, orig_y in test_cases:
            with self.subTest(x=orig_x, y=orig_y):
                # Convert to screen and back
                screen_x, screen_y = self.mapper.math_to_screen(orig_x, orig_y)
                math_x, math_y = self.mapper.screen_to_math(screen_x, screen_y)
                
                # Should match original values
                self.assertAlmostEqual(math_x, orig_x, places=6)
                self.assertAlmostEqual(math_y, orig_y, places=6)
    
    def test_scale_value(self) -> None:
        """Test mathematical value scaling."""
        # Default scale factor of 1.0
        self.assertEqual(self.mapper.scale_value(100), 100)
        self.assertEqual(self.mapper.scale_value(0), 0)
        self.assertEqual(self.mapper.scale_value(-50), -50)
        
        # Change scale factor
        self.mapper.scale_factor = 2.0
        self.assertEqual(self.mapper.scale_value(100), 200)
        self.assertEqual(self.mapper.scale_value(50), 100)
        
        # Fractional scale factor
        self.mapper.scale_factor = 0.5
        self.assertEqual(self.mapper.scale_value(100), 50)
        self.assertEqual(self.mapper.scale_value(200), 100)
    
    def test_unscale_value(self) -> None:
        """Test screen value unscaling."""
        # Default scale factor of 1.0
        self.assertEqual(self.mapper.unscale_value(100), 100)
        
        # Scale factor of 2.0
        self.mapper.scale_factor = 2.0
        self.assertEqual(self.mapper.unscale_value(200), 100)
        self.assertEqual(self.mapper.unscale_value(100), 50)
        
        # Scale value round trip
        original = 123.456
        scaled = self.mapper.scale_value(original)
        unscaled = self.mapper.unscale_value(scaled)
        self.assertAlmostEqual(unscaled, original, places=6)
    
    def test_apply_zoom(self) -> None:
        """Test zoom application."""
        original_scale = self.mapper.scale_factor
        
        # Zoom in (factor > 1)
        self.mapper.apply_zoom(2.0)
        self.assertEqual(self.mapper.scale_factor, 2.0)
        
        # Zoom out (factor < 1)
        self.mapper.apply_zoom(0.5)
        self.assertEqual(self.mapper.scale_factor, 1.0)  # 2.0 * 0.5
        
        # Zoom with center point
        zoom_center = Position(100, 100)
        self.mapper.apply_zoom(2.0, zoom_center)
        self.assertEqual(self.mapper.zoom_point, zoom_center)
        self.assertEqual(self.mapper.zoom_direction, -1)  # zoom in
    
    def test_zoom_bounds(self) -> None:
        """Test zoom scale factor lower bound; no arbitrary upper bound now."""
        # Zoom way out: should clamp to minimal positive value (0.01)
        self.mapper.apply_zoom(0.001)
        self.assertGreaterEqual(self.mapper.scale_factor, 0.01)
        # Zoom way in: no upper cap enforced (only lower bound exists)
        self.mapper.scale_factor = 1.0
        self.mapper.apply_zoom(200.0)
        self.assertEqual(self.mapper.scale_factor, 200.0)
    
    def test_apply_zoom_step(self) -> None:
        """Test standardized zoom step operations."""
        original_scale = self.mapper.scale_factor
        
        # Zoom in step
        self.mapper.apply_zoom_step(-1)
        expected_scale = original_scale * (1 + self.mapper.zoom_step)
        self.assertAlmostEqual(self.mapper.scale_factor, expected_scale, places=6)
        
        # Zoom out step
        self.mapper.scale_factor = 1.0  # Reset
        self.mapper.apply_zoom_step(1)
        expected_scale = 1.0 * (1 - self.mapper.zoom_step)
        self.assertAlmostEqual(self.mapper.scale_factor, expected_scale, places=6)
    
    def test_apply_pan(self) -> None:
        """Test pan offset application."""
        # Initial offset should be zero
        self.assertEqual(self.mapper.offset.x, 0)
        self.assertEqual(self.mapper.offset.y, 0)
        
        # Apply pan
        self.mapper.apply_pan(50, -30)
        self.assertEqual(self.mapper.offset.x, 50)
        self.assertEqual(self.mapper.offset.y, -30)
        
        # Apply additional pan (should accumulate)
        self.mapper.apply_pan(25, 10)
        self.assertEqual(self.mapper.offset.x, 75)
        self.assertEqual(self.mapper.offset.y, -20)
    
    def test_reset_pan(self) -> None:
        """Test pan offset reset."""
        self.mapper.apply_pan(100, 200)
        self.mapper.reset_pan()
        self.assertEqual(self.mapper.offset.x, 0)
        self.assertEqual(self.mapper.offset.y, 0)
    
    def test_reset_transformations(self) -> None:
        """Test complete transformation reset."""
        # Apply various transformations
        self.mapper.apply_zoom(2.0)
        self.mapper.apply_pan(100, 200)
        self.mapper.zoom_direction = -1
        
        # Reset everything
        self.mapper.reset_transformations()
        
        # Check all values are back to defaults
        self.assertEqual(self.mapper.scale_factor, 1.0)
        self.assertEqual(self.mapper.offset.x, 0)
        self.assertEqual(self.mapper.offset.y, 0)
        self.assertEqual(self.mapper.origin.x, self.canvas_width / 2)
        self.assertEqual(self.mapper.origin.y, self.canvas_height / 2)
        self.assertEqual(self.mapper.zoom_direction, 0)
    
    def test_get_visible_bounds(self) -> None:
        """Test visible bounds calculation."""
        bounds = self.mapper.get_visible_bounds()
        
        # With default settings, bounds should cover canvas converted to math coords
        # Canvas corners: (0,0) and (800,600)
        # Math corners: (-400,300) and (400,-300)
        self.assertAlmostEqual(bounds['left'], -400, places=1)
        self.assertAlmostEqual(bounds['right'], 400, places=1)
        self.assertAlmostEqual(bounds['top'], 300, places=1)
        self.assertAlmostEqual(bounds['bottom'], -300, places=1)
    
    def test_get_visible_bounds_with_zoom(self) -> None:
        """Test visible bounds calculation with zoom."""
        # Zoom in 2x - visible area should be half the size
        self.mapper.apply_zoom(2.0)
        bounds = self.mapper.get_visible_bounds()
        
        # Bounds should be halved
        self.assertAlmostEqual(bounds['left'], -200, places=1)
        self.assertAlmostEqual(bounds['right'], 200, places=1)
        self.assertAlmostEqual(bounds['top'], 150, places=1)
        self.assertAlmostEqual(bounds['bottom'], -150, places=1)
    
    def test_get_visible_width_height(self) -> None:
        """Test visible width and height calculation."""
        width = self.mapper.get_visible_width()
        height = self.mapper.get_visible_height()
        
        # Default: 800-unit wide canvas should be 800 math units wide
        self.assertAlmostEqual(width, 800, places=1)
        self.assertAlmostEqual(height, 600, places=1)
        
        # After 2x zoom, should be half the size
        self.mapper.apply_zoom(2.0)
        width = self.mapper.get_visible_width()
        height = self.mapper.get_visible_height()
        self.assertAlmostEqual(width, 400, places=1)
        self.assertAlmostEqual(height, 300, places=1)
    
    def test_set_visible_bounds_horizontal_limited(self) -> None:
        """Requested bounds should be visible when width is limiting axis."""
        self.mapper.set_visible_bounds(-20, 20, 5, -5)

        self.assertAlmostEqual(self.mapper.get_visible_left_bound(), -20, places=6)
        self.assertAlmostEqual(self.mapper.get_visible_right_bound(), 20, places=6)
        self.assertGreaterEqual(self.mapper.get_visible_top_bound(), 5)
        self.assertLessEqual(self.mapper.get_visible_bottom_bound(), -5)

    def test_set_visible_bounds_vertical_limited(self) -> None:
        """Requested bounds should be visible when height is limiting axis."""
        self.mapper.set_visible_bounds(-5, 5, 12, -12)

        self.assertLessEqual(self.mapper.get_visible_left_bound(), -5)
        self.assertGreaterEqual(self.mapper.get_visible_right_bound(), 5)
        self.assertAlmostEqual(self.mapper.get_visible_top_bound(), 12, places=6)
        self.assertAlmostEqual(self.mapper.get_visible_bottom_bound(), -12, places=6)

    def test_set_visible_bounds_invalid(self) -> None:
        """Invalid bounds should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.set_visible_bounds(5, 5, 1, -1)
        with self.assertRaises(ValueError):
            self.mapper.set_visible_bounds(-1, 1, -2, 4)
        with self.assertRaises(ValueError):
            self.mapper.set_visible_bounds("a", 1, 2, -2)

    def test_is_point_visible(self) -> None:
        """Test screen point visibility checking."""
        # Points within canvas should be visible
        self.assertTrue(self.mapper.is_point_visible(400, 300))  # center
        self.assertTrue(self.mapper.is_point_visible(0, 0))      # top-left
        self.assertTrue(self.mapper.is_point_visible(799, 599))  # bottom-right
        
        # Points outside canvas should not be visible
        self.assertFalse(self.mapper.is_point_visible(-1, 300))   # left edge
        self.assertFalse(self.mapper.is_point_visible(400, -1))   # top edge
        self.assertFalse(self.mapper.is_point_visible(801, 300))  # right edge
        self.assertFalse(self.mapper.is_point_visible(400, 601))  # bottom edge
    
    def test_is_math_point_visible(self) -> None:
        """Test mathematical point visibility checking."""
        # Origin should be visible (maps to canvas center)
        self.assertTrue(self.mapper.is_math_point_visible(0, 0))
        
        # Points within visible math bounds should be visible
        self.assertTrue(self.mapper.is_math_point_visible(300, 200))
        
        # Points far outside should not be visible
        self.assertFalse(self.mapper.is_math_point_visible(1000, 1000))
        self.assertFalse(self.mapper.is_math_point_visible(-1000, -1000))
    
    def test_update_canvas_size(self) -> None:
        """Test canvas size update and origin recalculation."""
        # Update canvas size
        new_width, new_height = 1000, 800
        self.mapper.update_canvas_size(new_width, new_height)
        
        # Check new dimensions
        self.assertEqual(self.mapper.canvas_width, new_width)
        self.assertEqual(self.mapper.canvas_height, new_height)
        
        # Origin should be recalculated
        self.assertEqual(self.mapper.origin.x, new_width / 2)
        self.assertEqual(self.mapper.origin.y, new_height / 2)
    
    def test_coordinate_conversion_with_transformations(self) -> None:
        """Test coordinate conversion with zoom and pan applied."""
        # Apply zoom and pan
        self.mapper.apply_zoom(2.0)
        self.mapper.apply_pan(50, -30)
        
        # Test conversion with transformations
        screen_x, screen_y = self.mapper.math_to_screen(0, 0)
        # Origin (0,0) should map to canvas center + pan offset
        self.assertEqual(screen_x, 450)  # 400 + 50
        self.assertEqual(screen_y, 270)  # 300 - 30
        
        # Convert back
        math_x, math_y = self.mapper.screen_to_math(screen_x, screen_y)
        self.assertAlmostEqual(math_x, 0, places=6)
        self.assertAlmostEqual(math_y, 0, places=6)
    
    def test_get_zoom_towards_point_displacement(self) -> None:
        """Test zoom-towards-point displacement calculation."""
        # Set up zoom state
        self.mapper.zoom_point = Position(500, 200)  # Zoom center
        self.mapper.zoom_direction = -1  # Zoom in
        self.mapper.zoom_step = 0.1
        
        # Test point that should move towards zoom center
        target_point = Position(400, 300)
        displacement = self.mapper.get_zoom_towards_point_displacement(target_point)
        
        # Displacement should not be zero
        self.assertNotEqual(displacement.x, 0)
        self.assertNotEqual(displacement.y, 0)
        
        # When zooming in, objects should move AWAY from zoom point to maintain relative position
        # Target (400,300) is left/below zoom point (500,200), so displacement should be left/up
        self.assertLess(displacement.x, 0)     # Move left (away from zoom point)
        self.assertGreater(displacement.y, 0)  # Move up (away from zoom point)
    
    def test_state_management(self) -> None:
        """Test state serialization and restoration."""
        # Apply transformations
        self.mapper.apply_zoom(1.5)
        self.mapper.apply_pan(75, -40)
        self.mapper.zoom_direction = -1
        
        # Get state
        state = self.mapper.get_state()
        
        # Verify state contains expected keys
        expected_keys = ['canvas_width', 'canvas_height', 'scale_factor', 'offset', 
                        'origin', 'zoom_point', 'zoom_direction', 'zoom_step']
        for key in expected_keys:
            self.assertIn(key, state)
        
        # Create new mapper and restore state
        new_mapper = CoordinateMapper(1000, 1000)  # Different initial size
        new_mapper.set_state(state)
        
        # Should match original mapper
        self.assertEqual(new_mapper.canvas_width, self.mapper.canvas_width)
        self.assertEqual(new_mapper.scale_factor, self.mapper.scale_factor)
        self.assertEqual(new_mapper.offset.x, self.mapper.offset.x)
        self.assertEqual(new_mapper.offset.y, self.mapper.offset.y)
        self.assertEqual(new_mapper.zoom_direction, self.mapper.zoom_direction)
    
    def test_individual_boundary_methods(self) -> None:
        """Test individual boundary calculation methods."""
        # Test with default settings
        left = self.mapper.get_visible_left_bound()
        right = self.mapper.get_visible_right_bound()
        top = self.mapper.get_visible_top_bound()
        bottom = self.mapper.get_visible_bottom_bound()
        
        # Should match get_visible_bounds() results
        bounds = self.mapper.get_visible_bounds()
        self.assertAlmostEqual(left, bounds['left'], places=6)
        self.assertAlmostEqual(right, bounds['right'], places=6)
        self.assertAlmostEqual(top, bounds['top'], places=6)
        self.assertAlmostEqual(bottom, bounds['bottom'], places=6)
        
        # Test with transformations
        self.mapper.apply_zoom(2.0)
        self.mapper.apply_pan(100, -50)
        
        left_zoom = self.mapper.get_visible_left_bound()
        right_zoom = self.mapper.get_visible_right_bound()
        top_zoom = self.mapper.get_visible_top_bound()
        bottom_zoom = self.mapper.get_visible_bottom_bound()
        
        # Bounds should be affected by zoom and pan
        self.assertNotEqual(left_zoom, left)
        self.assertNotEqual(right_zoom, right)
        self.assertNotEqual(top_zoom, top)
        self.assertNotEqual(bottom_zoom, bottom)
    
    def test_legacy_pattern_support_methods(self) -> None:
        """Test methods that support legacy coordinate conversion patterns."""
        # Test convert_canvas_x_to_math
        canvas_x = 500  # 100 units right of center
        math_x = self.mapper.convert_canvas_x_to_math(canvas_x)
        self.assertAlmostEqual(math_x, 100, places=6)
        
        # Test convert_math_y_to_canvas
        math_y = 75
        canvas_y = self.mapper.convert_math_y_to_canvas(math_y)
        self.assertAlmostEqual(canvas_y, 225, places=6)  # 300 - 75
        
        # Test convert_math_x_to_canvas
        math_x = -50
        canvas_x = self.mapper.convert_math_x_to_canvas(math_x)
        self.assertAlmostEqual(canvas_x, 350, places=6)  # 400 - 50
        
        # Test with transformations
        self.mapper.apply_zoom(2.0)
        self.mapper.apply_pan(25, -15)
        
        # Canvas to math with transformations
        canvas_x = 500
        math_x = self.mapper.convert_canvas_x_to_math(canvas_x)
        expected_math_x = (500 - 25 - 400) / 2.0  # (canvas_x - offset.x - origin.x) / scale_factor
        self.assertAlmostEqual(math_x, expected_math_x, places=6)
        
        # Math to canvas with transformations
        math_y = 50
        canvas_y = self.mapper.convert_math_y_to_canvas(math_y)
        expected_canvas_y = 300 - 50 * 2.0 + (-15)  # origin.y - math_y * scale_factor + offset.y
        self.assertAlmostEqual(canvas_y, expected_canvas_y, places=6)
    
    def test_legacy_methods_consistency(self) -> None:
        """Test that legacy methods are consistent with core methods."""
        test_cases = [
            (0, 0),
            (100, -50),
            (-75, 125),
            (3.14159, -2.71828)
        ]
        
        for math_x, math_y in test_cases:
            with self.subTest(x=math_x, y=math_y):
                # Core methods
                screen_x, screen_y = self.mapper.math_to_screen(math_x, math_y)
                
                # Legacy methods should produce same results
                legacy_screen_x = self.mapper.convert_math_x_to_canvas(math_x)
                legacy_screen_y = self.mapper.convert_math_y_to_canvas(math_y)
                
                self.assertAlmostEqual(screen_x, legacy_screen_x, places=6)
                self.assertAlmostEqual(screen_y, legacy_screen_y, places=6)
                
                # Reverse conversion
                reverse_math_x = self.mapper.convert_canvas_x_to_math(screen_x)
                self.assertAlmostEqual(math_x, reverse_math_x, places=6)
                
    def test_sync_from_canvas_mock(self) -> None:
        """Test synchronization with a mock Canvas object."""
        # Create a mock canvas object with coordinate properties
        class MockCanvas:
            def __init__(self) -> None:
                self.width = 1000
                self.height = 800
                self.scale_factor = 1.5
                self.offset = Position(75, -25)
                self.center = Position(500, 400)
                self.zoom_point = Position(600, 300)
                self.zoom_direction = -1
                self.zoom_step = 0.15
                
                # Mock cartesian2axis with origin
                class MockCartesian:
                    def __init__(self) -> None:
                        self.origin = Position(520, 380)  # Slightly off center
                        
                self.cartesian2axis = MockCartesian()
        
        mock_canvas = MockCanvas()
        
        # Sync mapper with mock canvas
        self.mapper.sync_from_canvas(mock_canvas)
        
        # Check that mapper was updated with canvas values
        self.assertEqual(self.mapper.canvas_width, 1000)
        self.assertEqual(self.mapper.canvas_height, 800)
        self.assertEqual(self.mapper.scale_factor, 1.5)
        self.assertEqual(self.mapper.offset.x, 75)
        self.assertEqual(self.mapper.offset.y, -25)
        self.assertEqual(self.mapper.origin.x, 520)  # From cartesian2axis.origin
        self.assertEqual(self.mapper.origin.y, 380)
        self.assertEqual(self.mapper.zoom_point.x, 600)
        self.assertEqual(self.mapper.zoom_point.y, 300)
        self.assertEqual(self.mapper.zoom_direction, -1)
        self.assertEqual(self.mapper.zoom_step, 0.15)
    
    def test_sync_from_canvas_minimal(self) -> None:
        """Test sync with minimal Canvas object (missing some properties)."""
        class MinimalCanvas:
            def __init__(self) -> None:
                self.width = 600
                self.height = 400
                self.scale_factor = 0.8
                # Missing offset, cartesian2axis, zoom properties
        
        minimal_canvas = MinimalCanvas()
        original_offset_x = self.mapper.offset.x
        original_offset_y = self.mapper.offset.y
        
        # Sync should handle missing properties gracefully
        self.mapper.sync_from_canvas(minimal_canvas)
        
        # Should update available properties
        self.assertEqual(self.mapper.canvas_width, 600)
        self.assertEqual(self.mapper.canvas_height, 400)
        self.assertEqual(self.mapper.scale_factor, 0.8)
        
        # Should set default values for missing properties
        self.assertEqual(self.mapper.offset.x, 0)  # Default when offset missing
        self.assertEqual(self.mapper.offset.y, 0)
        self.assertEqual(self.mapper.origin.x, 300)  # width / 2
        self.assertEqual(self.mapper.origin.y, 200)  # height / 2
    
    def test_from_canvas_factory_method(self) -> None:
        """Test factory method to create CoordinateMapper from Canvas."""
        # Create mock canvas
        class MockCanvas:
            def __init__(self) -> None:
                self.width = 1200
                self.height = 900
                self.scale_factor = 2.5
                self.offset = Position(150, -100)
                
                class MockCartesian:
                    def __init__(self) -> None:
                        self.origin = Position(600, 450)
                        
                self.cartesian2axis = MockCartesian()
        
        mock_canvas = MockCanvas()
        
        # Create mapper using factory method
        mapper = CoordinateMapper.from_canvas(mock_canvas)
        
        # Should have canvas properties
        self.assertEqual(mapper.canvas_width, 1200)
        self.assertEqual(mapper.canvas_height, 900)
        self.assertEqual(mapper.scale_factor, 2.5)
        self.assertEqual(mapper.offset.x, 150)
        self.assertEqual(mapper.offset.y, -100)
        self.assertEqual(mapper.origin.x, 600)
        self.assertEqual(mapper.origin.y, 450)
    
    def test_coordinate_conversion_with_offset(self) -> None:
        """Test coordinate conversion when pan offset is applied."""
        # Apply pan offset
        self.mapper.apply_pan(50, -30)
        
        # Math origin should now map to canvas center + offset
        screen_x, screen_y = self.mapper.math_to_screen(0, 0)
        self.assertEqual(screen_x, 450)  # 400 + 50
        self.assertEqual(screen_y, 270)  # 300 - 30
        
        # Screen center should map to different math coordinates
        math_x, math_y = self.mapper.screen_to_math(400, 300)
        self.assertAlmostEqual(math_x, -50, places=6)   # Shifted by offset
        self.assertAlmostEqual(math_y, -30, places=6)    # Shifted by offset (negative)
        
        # Test individual conversion methods with offset
        canvas_x = 400  # Canvas center x
        math_x = self.mapper.convert_canvas_x_to_math(canvas_x)
        self.assertAlmostEqual(math_x, -50, places=6)   # (400 - 50 - 400) / 1.0
        
        math_y = 0
        canvas_y = self.mapper.convert_math_y_to_canvas(math_y)
        self.assertAlmostEqual(canvas_y, 270, places=6)  # 300 - 0 * 1.0 + (-30)

    def test_from_canvas_with_simple_mock_full_featured(self) -> None:
        """Test from_canvas factory method with SimpleMock having all canvas features."""
        # Create a full-featured mock canvas using SimpleMock
        cartesian_mock = SimpleMock(origin=Position(600, 400))
        canvas_mock = SimpleMock(
            width=1200,
            height=800,
            scale_factor=2.0,
            offset=Position(100, -50),
            cartesian2axis=cartesian_mock,
            zoom_point=Position(700, 350),
            zoom_direction=-1,
            zoom_step=0.15
        )
        
        # Create CoordinateMapper using from_canvas factory method
        mapper = CoordinateMapper.from_canvas(canvas_mock)
        
        # Verify all properties were correctly extracted
        self.assertEqual(mapper.canvas_width, 1200)
        self.assertEqual(mapper.canvas_height, 800)
        self.assertEqual(mapper.scale_factor, 2.0)
        self.assertEqual(mapper.offset.x, 100)
        self.assertEqual(mapper.offset.y, -50)
        self.assertEqual(mapper.origin.x, 600)   # From cartesian2axis.origin
        self.assertEqual(mapper.origin.y, 400)
        self.assertEqual(mapper.zoom_point.x, 700)
        self.assertEqual(mapper.zoom_point.y, 350)
        self.assertEqual(mapper.zoom_direction, -1)
        self.assertEqual(mapper.zoom_step, 0.15)
    
    def test_from_canvas_with_simple_mock_minimal(self) -> None:
        """Test from_canvas with minimal SimpleMock canvas (missing optional properties)."""
        # Create minimal mock canvas with only required properties
        canvas_mock = SimpleMock(
            width=800,
            height=600,
            scale_factor=1.5
        )
        
        # Should handle missing properties gracefully
        mapper = CoordinateMapper.from_canvas(canvas_mock)
        
        # Check basic properties
        self.assertEqual(mapper.canvas_width, 800)
        self.assertEqual(mapper.canvas_height, 600)
        self.assertEqual(mapper.scale_factor, 1.5)
        
        # Should use defaults for missing properties
        self.assertEqual(mapper.offset.x, 0)
        self.assertEqual(mapper.offset.y, 0)
        self.assertEqual(mapper.origin.x, 400)   # width / 2
        self.assertEqual(mapper.origin.y, 300)   # height / 2
        self.assertEqual(mapper.zoom_direction, 0)
        self.assertEqual(mapper.zoom_step, 0.1)
    
    def test_from_canvas_with_center_instead_of_cartesian(self) -> None:
        """Test from_canvas when canvas uses center instead of cartesian2axis.origin."""
        # Mock canvas that uses center property instead of cartesian2axis
        canvas_mock = SimpleMock(
            width=1000,
            height=800,
            scale_factor=1.2,
            offset=Position(30, 20),
            center=Position(520, 380),  # Using center instead of cartesian2axis
            zoom_point=Position(500, 400)
        )
        
        mapper = CoordinateMapper.from_canvas(canvas_mock)
        
        # Should use center for origin
        self.assertEqual(mapper.origin.x, 520)
        self.assertEqual(mapper.origin.y, 380)
        self.assertEqual(mapper.scale_factor, 1.2)
        self.assertEqual(mapper.offset.x, 30)
        self.assertEqual(mapper.offset.y, 20)
    
    def test_sync_from_canvas_with_simple_mock_updates_existing(self) -> None:
        """Test sync_from_canvas updates existing CoordinateMapper with SimpleMock."""
        # Start with a mapper with some initial values
        mapper = CoordinateMapper(800, 600)
        mapper.apply_zoom(3.0)
        mapper.apply_pan(200, -100)
        
        # Create mock canvas with different values
        cartesian_mock = SimpleMock(origin=Position(450, 350))
        canvas_mock = SimpleMock(
            width=1400,
            height=1000,
            scale_factor=0.8,
            offset=Position(-75, 40),
            cartesian2axis=cartesian_mock,
            zoom_direction=1,
            zoom_step=0.2
        )
        
        # Sync with canvas
        mapper.sync_from_canvas(canvas_mock)
        
        # All values should be updated to match canvas
        self.assertEqual(mapper.canvas_width, 1400)
        self.assertEqual(mapper.canvas_height, 1000)
        self.assertEqual(mapper.scale_factor, 0.8)
        self.assertEqual(mapper.offset.x, -75)
        self.assertEqual(mapper.offset.y, 40)
        self.assertEqual(mapper.origin.x, 450)
        self.assertEqual(mapper.origin.y, 350)
        self.assertEqual(mapper.zoom_direction, 1)
        self.assertEqual(mapper.zoom_step, 0.2)
    
    def test_sync_from_canvas_partial_properties(self) -> None:
        """Test sync_from_canvas with SimpleMock having only some properties."""
        mapper = CoordinateMapper(600, 400)
        original_zoom_step = mapper.zoom_step
        
        # Mock canvas with only some properties
        canvas_mock = SimpleMock(
            width=900,
            height=700,
            scale_factor=2.5
            # Missing: offset, cartesian2axis, center, zoom properties
        )
        
        mapper.sync_from_canvas(canvas_mock)
        
        # Should update available properties
        self.assertEqual(mapper.canvas_width, 900)
        self.assertEqual(mapper.canvas_height, 700) 
        self.assertEqual(mapper.scale_factor, 2.5)
        
        # Should set defaults for missing properties
        self.assertEqual(mapper.offset.x, 0)
        self.assertEqual(mapper.offset.y, 0)
        self.assertEqual(mapper.origin.x, 450)  # width / 2
        self.assertEqual(mapper.origin.y, 350)  # height / 2
        
        # Should preserve existing values for properties not in canvas
        self.assertEqual(mapper.zoom_step, original_zoom_step)
    
    def test_sync_from_canvas_coordinate_transformations_work(self) -> None:
        """Test that coordinate transformations work correctly after sync_from_canvas."""
        # Create mock canvas with specific transformation state
        cartesian_mock = SimpleMock(origin=Position(500, 300))
        canvas_mock = SimpleMock(
            width=1000,
            height=600,
            scale_factor=2.0,
            offset=Position(100, -50),
            cartesian2axis=cartesian_mock
        )
        
        mapper = CoordinateMapper(800, 600)  # Different initial dimensions
        mapper.sync_from_canvas(canvas_mock)
        
        # Test that coordinate transformations use synced values
        screen_x, screen_y = mapper.math_to_screen(0, 0)
        expected_x = 500 + 0 * 2.0 + 100  # origin.x + math_x * scale + offset.x
        expected_y = 300 - 0 * 2.0 + (-50)  # origin.y - math_y * scale + offset.y
        
        self.assertEqual(screen_x, expected_x)
        self.assertEqual(screen_y, expected_y)
        
        # Test reverse conversion
        math_x, math_y = mapper.screen_to_math(screen_x, screen_y)
        self.assertAlmostEqual(math_x, 0, places=6)
        self.assertAlmostEqual(math_y, 0, places=6)
    
    def test_sync_from_canvas_visible_bounds_consistency(self) -> None:
        """Test that visible bounds calculations are consistent after sync_from_canvas."""
        # Mock canvas with transformations
        canvas_mock = SimpleMock(
            width=800,
            height=600,
            scale_factor=1.5,
            offset=Position(50, 25),
            center=Position(400, 300)  # Using center instead of cartesian2axis
        )
        
        mapper = CoordinateMapper(1000, 800)  # Different initial size
        mapper.sync_from_canvas(canvas_mock)
        
        # Calculate bounds using different methods
        bounds = mapper.get_visible_bounds()
        left = mapper.get_visible_left_bound()
        right = mapper.get_visible_right_bound()
        top = mapper.get_visible_top_bound()
        bottom = mapper.get_visible_bottom_bound()
        
        # Individual methods should match bounds dictionary
        self.assertAlmostEqual(left, bounds['left'], places=6)
        self.assertAlmostEqual(right, bounds['right'], places=6)
        self.assertAlmostEqual(top, bounds['top'], places=6)
        self.assertAlmostEqual(bottom, bounds['bottom'], places=6)
        
        # Width and height should be consistent
        width = mapper.get_visible_width()
        height = mapper.get_visible_height()
        self.assertAlmostEqual(width, bounds['right'] - bounds['left'], places=6)
        self.assertAlmostEqual(height, bounds['top'] - bounds['bottom'], places=6)
    
    def test_canvas_mock_attribute_error_handling(self) -> None:
        """Test handling of missing attributes in canvas mock gracefully."""
        # Mock with minimal attributes
        canvas_mock = SimpleMock(
            width=640,
            height=480
            # Missing scale_factor and other properties
        )
        
        mapper = CoordinateMapper(800, 600)
        
        # Should not raise exceptions
        mapper.sync_from_canvas(canvas_mock)
        
        # Should use defaults for missing properties
        self.assertEqual(mapper.scale_factor, 1.0)  # Default value
        self.assertEqual(mapper.offset.x, 0)
        self.assertEqual(mapper.offset.y, 0)
        self.assertEqual(mapper.canvas_width, 640)
        self.assertEqual(mapper.canvas_height, 480)
    
    def test_factory_method_vs_sync_consistency(self) -> None:
        """Test that from_canvas factory method and sync_from_canvas produce same results."""
        # Create complex mock canvas
        cartesian_mock = SimpleMock(origin=Position(550, 325))
        canvas_mock = SimpleMock(
            width=1100,
            height=650,
            scale_factor=1.8,
            offset=Position(80, -35),
            cartesian2axis=cartesian_mock,
            zoom_point=Position(600, 400),
            zoom_direction=-1,
            zoom_step=0.12
        )
        
        # Method 1: Use factory method
        mapper1 = CoordinateMapper.from_canvas(canvas_mock)
        
        # Method 2: Create and sync
        mapper2 = CoordinateMapper(1100, 650)  # Same dimensions
        mapper2.sync_from_canvas(canvas_mock)
        
        # Both should have identical state
        self.assertEqual(mapper1.canvas_width, mapper2.canvas_width)
        self.assertEqual(mapper1.canvas_height, mapper2.canvas_height)
        self.assertEqual(mapper1.scale_factor, mapper2.scale_factor)
        self.assertEqual(mapper1.offset.x, mapper2.offset.x)
        self.assertEqual(mapper1.offset.y, mapper2.offset.y)
        self.assertEqual(mapper1.origin.x, mapper2.origin.x)
        self.assertEqual(mapper1.origin.y, mapper2.origin.y)
        self.assertEqual(mapper1.zoom_point.x, mapper2.zoom_point.x)
        self.assertEqual(mapper1.zoom_point.y, mapper2.zoom_point.y)
        self.assertEqual(mapper1.zoom_direction, mapper2.zoom_direction)
        self.assertEqual(mapper1.zoom_step, mapper2.zoom_step)
        
        # Coordinate conversions should also be identical
        test_math_x, test_math_y = 123.45, -67.89
        screen1_x, screen1_y = mapper1.math_to_screen(test_math_x, test_math_y)
        screen2_x, screen2_y = mapper2.math_to_screen(test_math_x, test_math_y)
        
        self.assertAlmostEqual(screen1_x, screen2_x, places=6)
        self.assertAlmostEqual(screen1_y, screen2_y, places=6)


if __name__ == '__main__':
    unittest.main() 