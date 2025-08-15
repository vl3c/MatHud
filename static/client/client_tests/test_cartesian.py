import unittest
from geometry import Point, Position
from cartesian_system_2axis import Cartesian2Axis
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestCartesian2Axis(unittest.TestCase):
    def setUp(self):
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(800, 600)  # 800x600 canvas
        
        # Create canvas mock with all properties that CoordinateMapper needs
        self.canvas = SimpleMock(
            width=800,  # Required by sync_from_canvas
            height=600,  # Required by sync_from_canvas
            scale_factor=1, 
            center=Position(400, 300),  # Canvas center
            cartesian2axis=None,  # Will be set after creating cartesian system
            coordinate_mapper=self.coordinate_mapper,
            zoom_direction=0, 
            offset=Position(0, 0),  # Set to (0,0) for simpler tests
            zoom_point=Position(0, 0), 
            zoom_step=0.1
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        self.cartesian_system = Cartesian2Axis(coordinate_mapper=self.coordinate_mapper)
        self.canvas.cartesian2axis = self.cartesian_system

    def test_init(self):
        self.assertEqual(self.cartesian_system.name, "cartesian-2axis-system")
        self.assertEqual(self.cartesian_system.width, self.canvas.width)
        self.assertEqual(self.cartesian_system.height, self.canvas.height)
        # With real coordinate mapper, origin Point(0,0) in math space -> (400,300) in screen space
        self.assertEqual(self.cartesian_system.origin.x, 400)
        self.assertEqual(self.cartesian_system.origin.y, 300)
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)

    def test_get_visible_bounds(self):
        # Test initial bounds - now calculated dynamically via CoordinateMapper
        left_bound = self.cartesian_system.get_visible_left_bound()
        right_bound = self.cartesian_system.get_visible_right_bound()
        top_bound = self.cartesian_system.get_visible_top_bound()
        bottom_bound = self.cartesian_system.get_visible_bottom_bound()
        
        self.assertEqual(left_bound, -400.0)  # -self.origin.x / self.canvas.scale_factor
        self.assertEqual(right_bound, 400.0)  # (self.width - self.origin.x) / self.canvas.scale_factor
        self.assertEqual(top_bound, 300.0)    # self.origin.y / self.canvas.scale_factor
        self.assertEqual(bottom_bound, -300.0)  # (self.origin.y - self.height) / self.canvas.scale_factor

        # Test bounds with different scale factor - bounds are now dynamic via CoordinateMapper
        self.canvas.scale_factor = 2
        self.coordinate_mapper.sync_from_canvas(self.canvas)  # Update coordinate mapper
        
        left_bound = self.cartesian_system.get_visible_left_bound()
        right_bound = self.cartesian_system.get_visible_right_bound()
        top_bound = self.cartesian_system.get_visible_top_bound()
        bottom_bound = self.cartesian_system.get_visible_bottom_bound()
        
        self.assertEqual(left_bound, -200.0)  # Bounds should be halved when scale_factor doubles
        self.assertEqual(right_bound, 200.0)
        self.assertEqual(top_bound, 150.0)
        self.assertEqual(bottom_bound, -150.0)

    def test_reset(self):
        # Changing the state of the Cartesian2Axis instance
        self.cartesian_system.current_tick_spacing = 50
        self.cartesian_system.reset()
        # Testing reset functionality
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)
        # Reset should place origin back at canvas center (400, 300)
        self.assertEqual(self.cartesian_system.origin.x, 400)
        self.assertEqual(self.cartesian_system.origin.y, 300)

    def test_calculate_tick_spacing(self):
        # Set up a Cartesian2Axis object with a specific width, max_ticks, and canvas scale factor
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.coordinate_mapper.scale_factor = 2  # Add this line
        # Calculate the tick spacing
        tick_spacing = self.cartesian_system._calculate_tick_spacing()
        # The relative width is self.width / self.canvas.scale_factor: 1000 / 2 = 500.
        # The ideal tick spacing is relative_width / max_ticks: 500 / 10 = 50.
        # The order of magnitude of the ideal tick spacing is 10 ** math.floor(math.log10(ideal_tick_spacing)): 10^1 = 10.
        # The possible spacings are [10, 25, 50, 100].
        # The closest spacing to the ideal tick spacing that is larger or equal to it is 50.
        self.assertEqual(tick_spacing, 50)

    def test_zoom_via_cache_invalidation(self):
        # Test zoom handling via new _invalidate_cache_on_zoom mechanism
        # Set up a Cartesian2Axis object with specific properties
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.canvas.scale_factor = 2
        self.cartesian_system.tick_spacing_bias = 1.0
        self.cartesian_system.current_tick_spacing = 100
        self.canvas.zoom_direction = -1  # Zoom in
        
        # Update coordinate mapper to reflect new scale
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        # Call the new zoom invalidation method
        self.cartesian_system._invalidate_cache_on_zoom()
        
        # The relative width is 500, so the ideal tick spacing is 50.
        # The proposed tick spacing is also 50, which is less than twice the current tick spacing (200).
        # Therefore, the current tick spacing should be updated to the proposed tick spacing.
        self.assertEqual(self.cartesian_system.current_tick_spacing, 50)
        
        # Change the zoom direction to zoom out
        self.canvas.zoom_direction = 1
        self.canvas.scale_factor = 0.5  # Decrease scale factor to simulate zooming out
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        # Call the cache invalidation method again
        self.cartesian_system._invalidate_cache_on_zoom()
        
        # Now, the relative width is 2000, so the ideal tick spacing is 200.
        # With steps [1,2,5,10] the possible spacings are [100, 200, 500, 1000].
        # The closest spacing >= 200 is 200.
        # Therefore, the current tick spacing should be updated to 200.
        self.assertEqual(self.cartesian_system.current_tick_spacing, 200)

    def test_dynamic_origin_calculation(self):
        # Test that origin is calculated dynamically from CoordinateMapper
        original_origin = Position(self.cartesian_system.origin.x, self.cartesian_system.origin.y)
        
        # Set a non-zero offset to test dynamic origin calculation
        self.canvas.offset = Position(10, 5)
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        # Origin should now be different due to dynamic calculation
        new_origin = self.cartesian_system.origin
        self.assertNotEqual(new_origin.x, original_origin.x)
        self.assertNotEqual(new_origin.y, original_origin.y)
        
        # Verify origin is calculated dynamically from CoordinateMapper
        expected_x, expected_y = self.coordinate_mapper.math_to_screen(0, 0)
        self.assertEqual(new_origin.x, expected_x)
        self.assertEqual(new_origin.y, expected_y)

    def test_state_retrieval(self):
        state = self.cartesian_system.get_state()
        expected_keys = ["Cartesian_System_Visibility"]
        # Test the state retrieval
        self.assertTrue(all(key in state for key in expected_keys))
        self.assertTrue(isinstance(state["Cartesian_System_Visibility"], dict))

    def test_get_axis_helpers(self):
        # Test the axis helper methods we added during refactoring
        self.assertEqual(self.cartesian_system._get_axis_origin('x'), self.cartesian_system.origin.x)
        self.assertEqual(self.cartesian_system._get_axis_origin('y'), self.cartesian_system.origin.y)
        
        self.assertEqual(self.cartesian_system._get_axis_boundary('x'), self.cartesian_system.width)
        self.assertEqual(self.cartesian_system._get_axis_boundary('y'), self.cartesian_system.height)
    
    def test_should_continue_drawing(self):
        # Test the boundary condition method for drawing
        # Position within boundary, direction positive
        self.assertTrue(self.cartesian_system._should_continue_drawing(50, 100, 1))
        # Position at boundary, direction positive
        self.assertFalse(self.cartesian_system._should_continue_drawing(100, 100, 1))
        # Position beyond boundary, direction positive
        self.assertFalse(self.cartesian_system._should_continue_drawing(150, 100, 1))
        
        # Position within boundary, direction negative
        self.assertTrue(self.cartesian_system._should_continue_drawing(50, 100, -1))
        # Position at zero, direction negative
        self.assertFalse(self.cartesian_system._should_continue_drawing(0, 100, -1))
        # Position below zero, direction negative
        self.assertFalse(self.cartesian_system._should_continue_drawing(-10, 100, -1))
    
    def test_calculate_ideal_tick_spacing(self):
        # Test the refactored tick spacing calculation
        self.cartesian_system.width = 1000
        self.cartesian_system.max_ticks = 10
        self.coordinate_mapper.scale_factor = 2
        
        ideal_spacing = self.cartesian_system._calculate_ideal_tick_spacing()
        self.assertEqual(ideal_spacing, 50)  # 1000/2/10 = 50
        
        # Test with different values
        self.cartesian_system.width = 800
        self.cartesian_system.max_ticks = 8
        self.coordinate_mapper.scale_factor = 1
        
        ideal_spacing = self.cartesian_system._calculate_ideal_tick_spacing()
        self.assertEqual(ideal_spacing, 100)  # 800/1/8 = 100
    
    def test_find_appropriate_spacing(self):
        # Test the spacing selection logic
        self.cartesian_system.tick_spacing_bias = 1.0
        # When ideal spacing is exactly a standard spacing value
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(10), 10)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(25), 50)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(50), 50)
        
        # When ideal spacing falls between standard values
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(15), 20)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(30), 50)
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(6), 10)
        
        # When ideal spacing is very small
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(0.15), 0.2)
        
        # When ideal spacing is very large
        self.assertEqual(self.cartesian_system._find_appropriate_spacing(750), 1000)

    def test_zoom_in_allows_fractional_tick_spacing(self):
        # Simulate high zoom-in so relative width is very small
        self.canvas.scale_factor = 200
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        # Bias encourages splitting sooner
        self.cartesian_system.tick_spacing_bias = 0.5
        # Recompute spacing via invalidate
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing = self.cartesian_system.current_tick_spacing
        # With width 800, scale 200 -> visible width 4.0; ideal = 4/10 = 0.4
        # magnitude = 10**-1 = 0.1; candidates [0.1, 0.2, 0.5, 1.0]
        # bias 0.5 -> effective_ideal 0.2; choose 0.2
        self.assertLess(spacing, 1.0)
        self.assertAlmostEqual(spacing, 0.2, places=6)

    def test_zoom_out_allows_large_tick_spacing(self):
        # Simulate extreme zoom-out so relative width is very large
        self.canvas.scale_factor = 0.01
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system.tick_spacing_bias = 1.0
        # Recompute spacing via invalidate
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing = self.cartesian_system.current_tick_spacing
        # With width 800, scale 0.01 -> visible width 80000; ideal = 80000/10 = 8000
        # magnitude = 1000; candidates [1000, 2000, 5000, 10000] -> pick 10000
        self.assertGreater(spacing, 10000 - 1e-6)

    def test_zoom_in_does_not_get_stuck_at_one(self):
        # Start around spacing ~1 and verify further zoom-in drops below 1
        # Configure such that visible_width yields ideal a bit under 2
        self.canvas.scale_factor = 40
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system.tick_spacing_bias = 0.25  # force earlier split
        # First invalidate to get close to 1
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing1 = self.cartesian_system.current_tick_spacing
        # Increase scale significantly to shrink visible width further
        self.canvas.scale_factor = 400
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing2 = self.cartesian_system.current_tick_spacing
        self.assertLessEqual(spacing1, 1.0)
        self.assertLess(spacing2, 1.0)

    def test_zoom_out_does_not_get_stuck_at_10000(self):
        # Start at spacing ~10000 and verify further zoom-out increases it beyond 10000
        self.canvas.scale_factor = 0.01  # very large visible width -> ideal ~8000 -> 10000 chosen
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system.tick_spacing_bias = 1.0
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing1 = self.cartesian_system.current_tick_spacing
        # Zoom out further (smaller scale)
        self.canvas.scale_factor = 0.002
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing2 = self.cartesian_system.current_tick_spacing
        self.assertGreaterEqual(spacing1, 10000 - 1e-6)
        self.assertGreater(spacing2, spacing1)
    
    def test_relative_dimensions(self):
        # Test the relative width and height calculations
        self.cartesian_system.width = 800
        self.cartesian_system.height = 600
        self.coordinate_mapper.scale_factor = 2
        
        self.assertEqual(self.cartesian_system.get_relative_width(), 400)
        self.assertEqual(self.cartesian_system.get_relative_height(), 300)
        
        # Test with different scale factor
        self.coordinate_mapper.scale_factor = 0.5
        self.assertEqual(self.cartesian_system.get_relative_width(), 1600)
        self.assertEqual(self.cartesian_system.get_relative_height(), 1200)
