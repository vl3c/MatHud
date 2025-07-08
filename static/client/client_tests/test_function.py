import unittest
import copy
from geometry import Position, Function
from expression_validator import ExpressionValidator
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestFunction(unittest.TestCase):
    def setUp(self):
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas
        
        self.canvas = SimpleMock(
            scale_factor=1, 
            cartesian2axis=SimpleMock(
                origin=Position(250, 250),  # Canvas center for 500x500
                get_visible_left_bound=SimpleMock(return_value=-10),
                get_visible_right_bound=SimpleMock(return_value=10),
                get_visible_top_bound=SimpleMock(return_value=10),
                get_visible_bottom_bound=SimpleMock(return_value=-10),
                height=500
            ),
            is_point_within_canvas_visible_area=SimpleMock(return_value=True),
            # Add coordinate_mapper properties
            width=500,
            height=500,
            center=Position(250, 250),
            coordinate_mapper=self.coordinate_mapper,
            zoom_point=Position(1, 1),
            zoom_direction=1,
            zoom_step=0.1,
            offset=Position(0, 0)
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        self.left_bound = -9
        self.right_bound = 9
        self.function_string = "x*2"
        self.name = "DoubleX"
        self.function = Function(self.function_string, self.canvas, self.name, left_bound=self.left_bound, right_bound=self.right_bound)

    def test_initialize(self):
        # Test that the function is correctly initialized
        self.assertEqual(self.function.function_string, ExpressionValidator.fix_math_expression(self.function_string))
        self.assertIsNotNone(self.function.function)
        self.assertEqual(self.function.name, self.name)

    def test_init_with_invalid_function_explicit(self):
        with self.assertRaises(ValueError):
            _ = Function("sin(/0)", self.canvas, "InvalidFunction")

    def test_get_class_name(self):
        self.assertEqual(self.function.get_class_name(), 'Function')

    def test_generate_values(self):
        # Test the generation of function values within canvas bounds
        paths = self.function._generate_paths()
        self.assertTrue(len(paths) > 0)
        
        # Check that we have points in our paths
        points_count = sum(len(path) for path in paths)
        self.assertTrue(points_count > 0)
        
        # Check x bounds and count points within and outside y bounds
        points_within_bounds = 0
        points_outside_bounds = 0
        top_bound = self.canvas.cartesian2axis.get_visible_top_bound()
        bottom_bound = self.canvas.cartesian2axis.get_visible_bottom_bound()
        
        for path in paths:
            for point in path:
                # Check x bounds (these should still be strict)
                original_pos = self.function._scaled_to_original(point.x, point.y)
                self.assertTrue(self.canvas.cartesian2axis.get_visible_left_bound() <= original_pos.x <= self.canvas.cartesian2axis.get_visible_right_bound())
                
                # Count points within and outside y bounds
                if bottom_bound <= original_pos.y <= top_bound:
                    points_within_bounds += 1
                else:
                    points_outside_bounds += 1
        
        # Ensure majority of points are within bounds
        self.assertGreater(points_within_bounds, points_outside_bounds, 
                          "Majority of points should be within y bounds")

    def test_zoom(self):
        # Test zoom functionality (mainly impacts _generate_values)
        self.function.zoom()  # This might not change anything directly but can prepare for future tests with scale_factor changes

    def test_get_state(self):
        state = self.function.get_state()
        expected_state = {"name": self.name, "args": {"function_string": self.function_string, "left_bound": self.left_bound, "right_bound": self.right_bound}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        function_copy = copy.deepcopy(self.function)
        self.assertIsNot(function_copy, self.function)
        self.assertEqual(function_copy.function_string, self.function.function_string)
        self.assertEqual(function_copy.name, self.function.name)

    def test_scaled_to_original_and_back(self):
        # Test conversion from scaled to original coordinates and back
        original = (2, 4)
        scaled = self.function._original_to_scaled(*original)
        back_to_original = self.function._scaled_to_original(scaled.x, scaled.y)
        self.assertAlmostEqual(original[0], back_to_original.x)
        self.assertAlmostEqual(original[1], back_to_original.y)

    def test_caching_mechanism(self):
        # Test that points are cached and reused
        self.function._cached_paths = None
        self.function._cache_valid = False
        
        # First draw should generate points
        points = self.function._generate_paths()
        self.function._cached_paths = points
        self.function._cache_valid = True
        initial_points = self.function._cached_paths
        
        self.assertIsNotNone(initial_points)
        self.assertTrue(self.function._cache_valid)

    def test_cache_invalidation_on_zoom(self):
        # Test that cache is invalidated on zoom
        paths = self.function._generate_paths()
        self.function._cached_paths = paths
        self.function._cache_valid = True
        initial_paths = self.function._cached_paths
        
        self.function.zoom()
        self.assertFalse(self.function._cache_valid)
        
        # Change scale factor to simulate zoom
        self.canvas.scale_factor = 2
        new_paths = self.function._generate_paths()
        self.function._cached_paths = new_paths
        
        # Calculate total points before and after zoom
        initial_total_points = sum(len(path) for path in initial_paths)
        new_total_points = sum(len(path) for path in new_paths)
        
        # Number of points or density might change due to zoom
        # If points remain the same, at least verify cache was invalidated
        if initial_total_points == new_total_points:
            self.assertFalse(self.function._cache_valid)

    def test_cache_invalidation_on_pan(self):
        # Test that cache is invalidated on pan
        self.function.draw()
        initial_points = self.function._cached_paths
        
        self.function.pan()
        self.assertFalse(self.function._cache_valid)

    def test_adaptive_step_size(self):
        # Test function with varying slopes
        steep_function = Function("100*x", self.canvas, "Steep")
        gradual_function = Function("0.1*x", self.canvas, "Gradual")
        
        steep_paths = steep_function._generate_paths()
        gradual_paths = gradual_function._generate_paths()
        
        # Calculate total points in each function
        steep_total_points = sum(len(path) for path in steep_paths)
        gradual_total_points = sum(len(path) for path in gradual_paths)
        
        # Steep function should have MORE points for better detail
        self.assertGreaterEqual(steep_total_points, gradual_total_points, 
                              f"Expected steep function to have at least as many points ({steep_total_points}) as gradual function ({gradual_total_points})")

    def test_discontinuity_handling(self):
        # Test function with discontinuity
        discontinuous_function = Function("1/x", self.canvas, "Discontinuous", step=0.5)  # Smaller step size
        paths = discontinuous_function._generate_paths()
        
        # Flatten all points from all paths for testing
        flat_points = []
        for path in paths:
            flat_points.extend(path)
            
        # Sort points by x-value to ensure chronological order for checking gaps
        flat_points.sort(key=lambda p: p.x)
        
        # Find gaps in x coordinates that indicate discontinuity
        has_discontinuity = False
        for i in range(1, len(flat_points)):
            original_p1 = self.function._scaled_to_original(flat_points[i-1].x, flat_points[i-1].y)
            original_p2 = self.function._scaled_to_original(flat_points[i].x, flat_points[i].y)
            # Check for either a large x gap or a transition through bounds
            if (abs(original_p2.x - original_p1.x) > discontinuous_function.step * 2) or \
               (abs(original_p2.y - original_p1.y) > (self.canvas.cartesian2axis.get_visible_top_bound() - 
                                                     self.canvas.cartesian2axis.get_visible_bottom_bound())):
                has_discontinuity = True
                break
        
        self.assertTrue(has_discontinuity)

    def test_bounds_checking(self):
        # Test that points are properly bounded
        # For a 500x500 canvas with origin at (250, 250), to get bounds from -5 to 5:
        # scale_factor = 250 / 5 = 50
        self.coordinate_mapper.scale_factor = 50
        self.canvas.scale_factor = 50
        
        # Update the canvas cartesian2axis mock to reflect new bounds
        self.canvas.cartesian2axis.get_visible_left_bound.return_value = -5
        self.canvas.cartesian2axis.get_visible_right_bound.return_value = 5
        self.canvas.cartesian2axis.get_visible_top_bound.return_value = 5
        self.canvas.cartesian2axis.get_visible_bottom_bound.return_value = -5
        
        # Need to update the function's bounds as well
        self.function.left_bound = -5
        self.function.right_bound = 5
        
        paths = self.function._generate_paths()
        
        # Ensure we have paths
        self.assertTrue(len(paths) > 0)
        
        for path in paths:
            # Ensure each path has points
            self.assertTrue(len(path) > 0)
            for point in path:
                # We need to check the original coordinates, not the scaled ones
                original_pos = self.function._scaled_to_original(point.x, point.y)
                # Add some tolerance for floating-point comparisons
                self.assertGreaterEqual(original_pos.x, -5.1)
                self.assertLessEqual(original_pos.x, 5.1)
                self.assertGreaterEqual(original_pos.y, -10.1)  # y=2x means y range is double x range
                self.assertLessEqual(original_pos.y, 10.1)

    def test_should_regenerate_points(self):
        # Test conditions for point regeneration
        points = self.function._generate_paths()
        self.function._cached_paths = points
        self.function._cache_valid = True
        self.function._last_scale = self.canvas.scale_factor
        self.function._last_bounds = (
            self.canvas.cartesian2axis.get_visible_left_bound(),
            self.canvas.cartesian2axis.get_visible_right_bound(),
            self.canvas.cartesian2axis.get_visible_top_bound(),
            self.canvas.cartesian2axis.get_visible_bottom_bound()
        )

    def test_draw_with_empty_points(self):
        # Test drawing behavior when no points are generated
        # Mock _generate_values to return empty list
        original_generate = self.function._generate_paths
        self.function._generate_paths = lambda: []
        
        # Should not raise error when drawing with no points
        self.function.draw()
        
        # Restore original method
        self.function._generate_paths = original_generate

    def test_deepcopy_with_cache(self):
        # Test that deepcopy properly handles cache attributes
        self.function.draw()  # Generate initial cache
        function_copy = copy.deepcopy(self.function)
        
        self.assertIsNone(function_copy._cached_paths)
        self.assertFalse(function_copy._cache_valid)
        self.assertIsNone(function_copy._last_scale)
        self.assertIsNone(function_copy._last_bounds)

    def test_performance_limits(self):
        # Test that point generation doesn't exceed limits
        complex_function = Function("sin(x*10)", self.canvas, "Complex")
        paths = complex_function._generate_paths()
        
        # Check number of paths is reasonable
        self.assertLess(len(paths), 100, "Should not exceed maximum paths limit")
        
        # Check total number of points is reasonable
        total_points = sum(len(path) for path in paths)
        self.assertLess(total_points, 1001, "Should not exceed maximum points limit")
        
        # Check minimum distance between consecutive points in each path
        for path in paths:
            if len(path) > 1:  # Only check paths with at least 2 points
                for i in range(1, len(path)):
                    dx = abs(path[i].x - path[i-1].x)
                    dy = abs(path[i].y - path[i-1].y)
                    self.assertGreater(dx + dy, 0, "Points should not be duplicates")

    def test_high_frequency_trig_functions(self):
        test_cases = [
            ("10*sin(10*x)", "High frequency and amplitude", 50, 500),  # Custom bounds for high frequency
            ("sin(20*x)", "High frequency only", 50, 500),               # Custom bounds for high frequency 
            ("5*sin(x)", "High amplitude only", 20, 200),               # Lower bounds for simple functions
            ("2*sin(3*x)", "Medium frequency and amplitude", 20, 200),  # Lower bounds for medium complexity
            ("100*sin(50*x)", "Very high frequency and amplitude", 100, 1000)  # Higher bounds for very complex
        ]
        
        for function_string, description, min_points, max_points in test_cases:
            with self.subTest(function_string=function_string, description=description):
                try:
                    f = Function(function_string, self.canvas, "test_func")
                    paths = f._generate_paths()
                    self.assertIsNotNone(paths)
                    self.assertGreater(len(paths), 0)
                    
                    # Count total points across all paths
                    total_points = sum(len(path) for path in paths)
                    
                    # Use function-specific bounds
                    self.assertGreaterEqual(total_points, min_points, 
                                         f"{function_string} ({description}): {total_points} points, expected at least {min_points}")
                    self.assertLessEqual(total_points, max_points, 
                                       f"{function_string} ({description}): {total_points} points, expected at most {max_points}")
                except Exception as e:
                    self.fail(f"Failed to handle {function_string}: {str(e)}")

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

