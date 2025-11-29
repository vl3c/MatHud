import unittest
from geometry import Point, Position
from cartesian_system_2axis import Cartesian2Axis
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestCartesian2Axis(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_init(self) -> None:
        self.assertEqual(self.cartesian_system.name, "cartesian-2axis-system")
        self.assertEqual(self.cartesian_system.width, self.canvas.width)
        self.assertEqual(self.cartesian_system.height, self.canvas.height)
        # With real coordinate mapper, origin Point(0,0) in math space -> (400,300) in screen space
        self.assertEqual(self.cartesian_system.origin.x, 400)
        self.assertEqual(self.cartesian_system.origin.y, 300)
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)

    def test_get_visible_bounds(self) -> None:
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

    def test_reset(self) -> None:
        # Changing the state of the Cartesian2Axis instance
        self.cartesian_system.current_tick_spacing = 50
        self.cartesian_system.reset()
        # Testing reset functionality
        self.assertEqual(self.cartesian_system.current_tick_spacing, self.cartesian_system.default_tick_spacing)
        # Reset should place origin back at canvas center (400, 300)
        self.assertEqual(self.cartesian_system.origin.x, 400)
        self.assertEqual(self.cartesian_system.origin.y, 300)

    def test_calculate_tick_spacing(self) -> None:
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

    def test_zoom_via_cache_invalidation(self) -> None:
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

    def test_dynamic_origin_calculation(self) -> None:
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

    def test_state_retrieval(self) -> None:
        state = self.cartesian_system.get_state()
        expected_keys = {
            "Cartesian_System_Visibility",
            "current_tick_spacing",
            "default_tick_spacing",
            "current_tick_spacing_repr",
            "min_tick_spacing",
        }
        missing = expected_keys.difference(state.keys())
        self.assertTrue(
            not missing,
            msg=f"Missing keys: {sorted(missing)}; state keys: {sorted(state.keys())}; state: {state}",
        )
        self.assertTrue(isinstance(state["Cartesian_System_Visibility"], dict))
        self.assertTrue(isinstance(state["current_tick_spacing"], float))
        self.assertTrue(isinstance(state["default_tick_spacing"], float))
        self.assertTrue(isinstance(state["current_tick_spacing_repr"], str))
        expected_repr = format(self.cartesian_system.current_tick_spacing, ".12g")
        self.assertEqual(state["current_tick_spacing_repr"], expected_repr)
        self.assertTrue(isinstance(state["min_tick_spacing"], float))

    def test_get_axis_helpers(self) -> None:
        # Test the axis helper methods we added during refactoring
        self.assertEqual(self.cartesian_system._get_axis_origin('x'), self.cartesian_system.origin.x)
        self.assertEqual(self.cartesian_system._get_axis_origin('y'), self.cartesian_system.origin.y)
        
        self.assertEqual(self.cartesian_system._get_axis_boundary('x'), self.cartesian_system.width)
        self.assertEqual(self.cartesian_system._get_axis_boundary('y'), self.cartesian_system.height)
    
    def test_should_continue_drawing(self) -> None:
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
    
    def test_calculate_ideal_tick_spacing(self) -> None:
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
    
    def test_find_appropriate_spacing(self) -> None:
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

    def test_zoom_in_allows_fractional_tick_spacing(self) -> None:
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

    def test_zoom_out_allows_large_tick_spacing(self) -> None:
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

    def test_zoom_in_does_not_get_stuck_at_one(self) -> None:
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

    def test_zoom_in_progresses_by_halving(self) -> None:
        # Once spacing reaches the small-scale regime, it should halve stepwise as zoom increases
        self.cartesian_system.current_tick_spacing = 1.0
        for scale, expected in ((100, 0.5), (200, 0.25), (400, 0.125)):
            self.canvas.scale_factor = scale
            self.coordinate_mapper.sync_from_canvas(self.canvas)
            self.cartesian_system._invalidate_cache_on_zoom()
            self.assertAlmostEqual(self.cartesian_system.current_tick_spacing, expected, places=6)

    def test_zoom_in_halving_triggers_on_equal_ratio(self) -> None:
        # When proposed spacing equals the trigger threshold we should still halve
        self.cartesian_system.current_tick_spacing = 0.125
        self.canvas.scale_factor = 400  # produces proposed spacing of 0.1
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system._invalidate_cache_on_zoom()
        self.assertAlmostEqual(self.cartesian_system.current_tick_spacing, 0.0625, places=6)

    def test_zoom_in_refines_below_point_one(self) -> None:
        # Ensure that once spacing reaches 0.1, additional zoom halves it as needed
        self.canvas.scale_factor = 400
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing_initial = self.cartesian_system.current_tick_spacing
        self.assertAlmostEqual(spacing_initial, 0.1, places=6)

        self.canvas.scale_factor = 900
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.cartesian_system._invalidate_cache_on_zoom()
        spacing_refined = self.cartesian_system.current_tick_spacing
        self.assertLess(spacing_refined, spacing_initial)
        self.assertAlmostEqual(spacing_refined, 0.05, places=6)

    def test_zoom_in_handles_extremely_small_spacing(self) -> None:
        self.cartesian_system.current_tick_spacing = 1e-2
        for scale, expected in ((8_000, 5e-3), (32_000, 2.5e-3), (64_000, 1.25e-3)):
            self.canvas.scale_factor = scale
            self.coordinate_mapper.sync_from_canvas(self.canvas)
            self.cartesian_system._invalidate_cache_on_zoom()
            self.assertAlmostEqual(self.cartesian_system.current_tick_spacing, expected, places=9)

    def test_can_zoom_in_further_respects_min_spacing(self) -> None:
        min_spacing = self.cartesian_system.min_tick_spacing
        self.cartesian_system.current_tick_spacing = min_spacing * 1.5
        self.assertTrue(self.cartesian_system.can_zoom_in_further())
        self.cartesian_system.current_tick_spacing = min_spacing
        self.assertFalse(self.cartesian_system.can_zoom_in_further())

    def test_zoom_out_handles_extremely_large_spacing(self) -> None:
        self.cartesian_system.current_tick_spacing = 1e3
        for scale, expected in ((0.02, 2e3), (0.01, 5e3), (0.005, 1e4)):
            self.canvas.scale_factor = scale
            self.coordinate_mapper.sync_from_canvas(self.canvas)
            self.cartesian_system._invalidate_cache_on_zoom()
            self.assertAlmostEqual(self.cartesian_system.current_tick_spacing, expected, places=6)

    def test_zoom_out_does_not_get_stuck_at_10000(self) -> None:
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
    
    def test_relative_dimensions(self) -> None:
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

    def _count_grid_lines(self, origin, dimension_px, display_tick):
        import math
        if display_tick <= 0:
            return 0
        start_n = int(math.ceil(-origin / display_tick))
        end_n = int(math.floor((dimension_px - origin) / display_tick))
        count = 0
        for n in range(start_n, end_n + 1):
            pos = origin + n * display_tick
            if 0 <= pos <= dimension_px:
                count += 1
        return count

    def _calculate_adaptive_tick_spacing(self, width, scale_factor, max_ticks=10):
        import math
        relative_width = width / scale_factor
        ideal_spacing = relative_width / max_ticks
        if ideal_spacing <= 0:
            return 1.0
        magnitude = 10 ** math.floor(math.log10(ideal_spacing))
        candidates = [magnitude * m for m in [1, 2, 5, 10]]
        for spacing in candidates:
            if spacing >= ideal_spacing:
                return spacing
        return candidates[-1]

    def test_grid_line_count_constant_at_various_origins(self):
        width_px = 800
        height_px = 600
        display_tick = 50
        
        base_count_x = self._count_grid_lines(400, width_px, display_tick)
        base_count_y = self._count_grid_lines(300, height_px, display_tick)
        
        offsets = [
            (0, 0),
            (100, 100),
            (-500, -500),
            (1000, 1000),
            (-10000, -10000),
            (50000, 50000),
            (-123456, -654321),
        ]
        
        for ox_offset, oy_offset in offsets:
            ox = 400 + ox_offset
            oy = 300 + oy_offset
            count_x = self._count_grid_lines(ox, width_px, display_tick)
            count_y = self._count_grid_lines(oy, height_px, display_tick)
            
            self.assertAlmostEqual(
                count_x, base_count_x, delta=2,
                msg=f"X grid line count {count_x} differs from base {base_count_x} at ox={ox}"
            )
            self.assertAlmostEqual(
                count_y, base_count_y, delta=2,
                msg=f"Y grid line count {count_y} differs from base {base_count_y} at oy={oy}"
            )

    def test_grid_line_count_bounded_across_zoom_levels(self):
        width_px = 800
        height_px = 600
        
        zoom_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        
        for scale_factor in zoom_levels:
            tick_spacing = self._calculate_adaptive_tick_spacing(width_px, scale_factor)
            display_tick = tick_spacing * scale_factor
            
            ox = width_px / 2
            oy = height_px / 2
            count_x = self._count_grid_lines(ox, width_px, display_tick)
            count_y = self._count_grid_lines(oy, height_px, display_tick)
            
            self.assertGreaterEqual(
                count_x, 4,
                msg=f"Too few X grid lines ({count_x}) at scale {scale_factor}"
            )
            self.assertLessEqual(
                count_x, 25,
                msg=f"Too many X grid lines ({count_x}) at scale {scale_factor}"
            )
            self.assertGreaterEqual(
                count_y, 3,
                msg=f"Too few Y grid lines ({count_y}) at scale {scale_factor}"
            )
            self.assertLessEqual(
                count_y, 20,
                msg=f"Too many Y grid lines ({count_y}) at scale {scale_factor}"
            )

    def test_grid_line_count_constant_at_extreme_distances(self):
        width_px = 800
        height_px = 600
        
        scale_factors = [0.01, 1.0, 100.0]
        
        for scale_factor in scale_factors:
            tick_spacing = self._calculate_adaptive_tick_spacing(width_px, scale_factor)
            display_tick = tick_spacing * scale_factor
            
            base_ox = width_px / 2
            base_oy = height_px / 2
            base_count_x = self._count_grid_lines(base_ox, width_px, display_tick)
            base_count_y = self._count_grid_lines(base_oy, height_px, display_tick)
            
            extreme_offsets = [
                1e6, -1e6,
                1e9, -1e9,
                1e12, -1e12,
            ]
            
            for offset in extreme_offsets:
                ox = base_ox + offset
                oy = base_oy + offset
                count_x = self._count_grid_lines(ox, width_px, display_tick)
                count_y = self._count_grid_lines(oy, height_px, display_tick)
                
                self.assertAlmostEqual(
                    count_x, base_count_x, delta=2,
                    msg=f"scale={scale_factor}, offset={offset}: X count {count_x} vs base {base_count_x}"
                )
                self.assertAlmostEqual(
                    count_y, base_count_y, delta=2,
                    msg=f"scale={scale_factor}, offset={offset}: Y count {count_y} vs base {base_count_y}"
                )

    def test_grid_line_count_combined_zoom_and_distance(self):
        width_px = 800
        height_px = 600
        
        test_cases = [
            (0.1, 0),
            (0.1, 1e6),
            (0.1, -1e6),
            (1.0, 0),
            (1.0, 1e6),
            (1.0, -1e6),
            (10.0, 0),
            (10.0, 1e6),
            (10.0, -1e6),
            (100.0, 0),
            (100.0, 1e9),
            (100.0, -1e9),
        ]
        
        results = {}
        for scale_factor, offset in test_cases:
            if scale_factor not in results:
                results[scale_factor] = []
            
            tick_spacing = self._calculate_adaptive_tick_spacing(width_px, scale_factor)
            display_tick = tick_spacing * scale_factor
            
            ox = width_px / 2 + offset
            oy = height_px / 2 + offset
            count_x = self._count_grid_lines(ox, width_px, display_tick)
            count_y = self._count_grid_lines(oy, height_px, display_tick)
            results[scale_factor].append((count_x, count_y, offset))
        
        for scale_factor, counts in results.items():
            x_counts = [c[0] for c in counts]
            y_counts = [c[1] for c in counts]
            
            x_range = max(x_counts) - min(x_counts)
            y_range = max(y_counts) - min(y_counts)
            
            self.assertLessEqual(
                x_range, 2,
                msg=f"At scale {scale_factor}, X counts vary too much: {counts}"
            )
            self.assertLessEqual(
                y_range, 2,
                msg=f"At scale {scale_factor}, Y counts vary too much: {counts}"
            )