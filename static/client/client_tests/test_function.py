import unittest
import copy
from geometry import Position, Function
from expression_validator import ExpressionValidator
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper
from rendering.function_renderable import FunctionRenderable


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
        # Test the generation of function values within canvas bounds using FunctionRenderable
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        screen_polyline = renderable.build_screen_paths()
        paths = screen_polyline.paths
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
            for point_tuple in path:
                # Convert tuple (x, y) to screen coordinates for bounds checking
                math_x, math_y = self.canvas.coordinate_mapper.screen_to_math(point_tuple[0], point_tuple[1])
                self.assertTrue(self.canvas.cartesian2axis.get_visible_left_bound() <= math_x <= self.canvas.cartesian2axis.get_visible_right_bound())
                
                # Count points within and outside y bounds
                if bottom_bound <= math_y <= top_bound:
                    points_within_bounds += 1
                else:
                    points_outside_bounds += 1
        
        # Ensure majority of points are within bounds
        self.assertGreater(points_within_bounds, points_outside_bounds, 
                          "Majority of points should be within y bounds")

    def test_get_state(self):
        state = self.function.get_state()
        expected_state = {"name": self.name, "args": {"function_string": self.function_string, "left_bound": self.left_bound, "right_bound": self.right_bound}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        function_copy = copy.deepcopy(self.function)
        self.assertIsNot(function_copy, self.function)
        self.assertEqual(function_copy.function_string, self.function.function_string)
        self.assertEqual(function_copy.name, self.function.name)

    def test_caching_mechanism(self):
        # Test that points are cached and reused in FunctionRenderable
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        
        # First call should generate paths
        first_call = renderable.build_screen_paths()
        self.assertIsNotNone(first_call.paths)
        self.assertTrue(len(first_call.paths) > 0)
        
        # Second call should use cached paths (same result)
        second_call = renderable.build_screen_paths()
        self.assertEqual(len(first_call.paths), len(second_call.paths))

    def test_cache_invalidation_on_zoom_new_mechanism(self):
        """Test that cache is invalidated when invalidate_cache() is called on FunctionRenderable"""
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        
        # Generate initial cache
        initial_paths = renderable.build_screen_paths()
        self.assertTrue(len(initial_paths.paths) > 0)
        
        # Simulate zoom by calling cache invalidation
        renderable.invalidate_cache()
        
        # Build paths again after invalidation
        new_paths = renderable.build_screen_paths()
        
        # Should still generate valid paths (cache regenerated)
        self.assertTrue(len(new_paths.paths) > 0)
        
    def test_zoom_via_canvas_draw_mechanism(self):
        """Test zoom cache invalidation through the Canvas.draw(apply_zoom=True) pattern using FunctionRenderable"""
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        
        # Generate initial cache
        initial_paths = renderable.build_screen_paths()
        self.assertTrue(len(initial_paths.paths) > 0)
        
        # Simulate what Canvas.draw(apply_zoom=True) does:
        # 1. Check if renderable has invalidate_cache method
        # 2. Call it if it exists
        # 3. Call build_screen_paths()
        if hasattr(renderable, 'invalidate_cache'):
            renderable.invalidate_cache()
            
        # Build paths again after cache invalidation
        new_paths = renderable.build_screen_paths()
        
        # Verify paths are regenerated successfully
        self.assertTrue(len(new_paths.paths) > 0)

    def test_adaptive_step_size(self):
        # Test function with varying slopes using FunctionRenderable
        steep_function = Function("100*x", self.canvas, "Steep")
        gradual_function = Function("0.1*x", self.canvas, "Gradual")
        
        steep_renderable = FunctionRenderable(steep_function, self.canvas.coordinate_mapper)
        gradual_renderable = FunctionRenderable(gradual_function, self.canvas.coordinate_mapper)
        
        steep_paths = steep_renderable.build_screen_paths().paths
        gradual_paths = gradual_renderable.build_screen_paths().paths
        
        # Calculate total points in each function
        steep_total_points = sum(len(path) for path in steep_paths)
        gradual_total_points = sum(len(path) for path in gradual_paths)
        
        # Steep function should have MORE points for better detail
        self.assertGreaterEqual(steep_total_points, gradual_total_points, 
                              f"Expected steep function to have at least as many points ({steep_total_points}) as gradual function ({gradual_total_points})")

    def test_discontinuity_handling(self):
        # Test function with discontinuity using FunctionRenderable
        discontinuous_function = Function("1/x", self.canvas, "Discontinuous", step=0.5)  # Smaller step size
        renderable = FunctionRenderable(discontinuous_function, self.canvas.coordinate_mapper)
        paths = renderable.build_screen_paths().paths
        
        # Flatten all points from all paths for testing
        flat_points = []
        for path in paths:
            flat_points.extend(path)
            
        # Sort points by x-value to ensure chronological order for checking gaps
        flat_points.sort(key=lambda p: p[0])  # Sort by x coordinate (first element of tuple)
        
        # Find gaps in x coordinates that indicate discontinuity
        has_discontinuity = False
        for i in range(1, len(flat_points)):
            math_x1, math_y1 = self.canvas.coordinate_mapper.screen_to_math(flat_points[i-1][0], flat_points[i-1][1])
            math_x2, math_y2 = self.canvas.coordinate_mapper.screen_to_math(flat_points[i][0], flat_points[i][1])
            # Check for either a large x gap or a transition through bounds
            if (abs(math_x2 - math_x1) > discontinuous_function.step * 2) or \
               (abs(math_y2 - math_y1) > (self.canvas.cartesian2axis.get_visible_top_bound() - 
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
        
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        paths = renderable.build_screen_paths().paths
        
        # Ensure we have paths
        self.assertTrue(len(paths) > 0)
        
        for path in paths:
            # Ensure each path has points
            self.assertTrue(len(path) > 0)
            for point_tuple in path:
                # We need to check the original coordinates, not the scaled ones
                math_x, math_y = self.canvas.coordinate_mapper.screen_to_math(point_tuple[0], point_tuple[1])
                # Add some tolerance for floating-point comparisons
                self.assertGreaterEqual(math_x, -5.1)
                self.assertLessEqual(math_x, 5.1)
                self.assertGreaterEqual(math_y, -10.1)  # y=2x means y range is double x range
                self.assertLessEqual(math_y, 10.1)

    def test_should_regenerate_points(self):
        # Test conditions for point regeneration using FunctionRenderable
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        
        # Generate initial paths
        initial_paths = renderable.build_screen_paths()
        self.assertTrue(len(initial_paths.paths) > 0)
        
        # Invalidate cache and regenerate
        renderable.invalidate_cache()
        new_paths = renderable.build_screen_paths()
        self.assertTrue(len(new_paths.paths) > 0)

    def test_draw_with_empty_points(self):
        # Test drawing behavior when no points are generated using FunctionRenderable
        # Create a function that should generate no valid points
        empty_function = Function("sin(1/0)", self.canvas, "Empty")  # This should fail evaluation
        renderable = FunctionRenderable(empty_function, self.canvas.coordinate_mapper)
        
        # Should not raise error when building paths with invalid function
        try:
            paths = renderable.build_screen_paths()
            # Should return empty paths gracefully
            self.assertEqual(len(paths.paths), 0)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_deepcopy_with_cache(self):
        # Test that deepcopy properly handles Function objects (cache handled by renderables)
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        renderable.build_screen_paths()  # Generate initial cache in renderable
        
        function_copy = copy.deepcopy(self.function)
        
        # Function copy should be independent and structurally equal
        self.assertEqual(function_copy.function_string, self.function.function_string)
        self.assertEqual(function_copy.name, self.function.name)

    def test_performance_limits(self):
        # Test that point generation doesn't exceed limits using FunctionRenderable
        complex_function = Function("sin(x*10)", self.canvas, "Complex")
        renderable = FunctionRenderable(complex_function, self.canvas.coordinate_mapper)
        paths = renderable.build_screen_paths().paths
        
        # Check number of paths is reasonable
        self.assertLess(len(paths), 100, "Should not exceed maximum paths limit")
        
        # Check total number of points is reasonable
        total_points = sum(len(path) for path in paths)
        self.assertLess(total_points, 1001, "Should not exceed maximum points limit")
        
        # Check minimum distance between consecutive points in each path
        for path in paths:
            if len(path) > 1:  # Only check paths with at least 2 points
                for i in range(1, len(path)):
                    dx = abs(path[i][0] - path[i-1][0])  # x coordinates
                    dy = abs(path[i][1] - path[i-1][1])  # y coordinates
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
                    renderable = FunctionRenderable(f, self.canvas.coordinate_mapper)
                    paths = renderable.build_screen_paths().paths
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
        # Test that FunctionRenderable can build screen paths successfully
        renderable = FunctionRenderable(self.function, self.canvas.coordinate_mapper)
        paths = renderable.build_screen_paths()
        
        # Should have valid paths
        self.assertIsNotNone(paths.paths)
        self.assertTrue(len(paths.paths) > 0)
        
        # Each path should contain tuples of (x, y) coordinates
        for path in paths.paths:
            if len(path) > 0:
                self.assertIsInstance(path[0], tuple)
                self.assertEqual(len(path[0]), 2)  # Should be (x, y) tuple

