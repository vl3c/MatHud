import unittest
import copy
from geometry import Position
from coordinate_mapper import CoordinateMapper
from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from drawables.function import Function
from .simple_mock import SimpleMock


class TestFunctionsBoundedColoredArea(unittest.TestCase):
    def setUp(self):
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas
        
        # Create canvas mock with all properties that CoordinateMapper needs
        self.canvas = SimpleMock(
            width=500,  # Required by sync_from_canvas
            height=500,  # Required by sync_from_canvas
            scale_factor=1, 
            center=Position(250, 250),  # Canvas center
            cartesian2axis=SimpleMock(origin=Position(250, 250)),  # Coordinate system origin
            coordinate_mapper=self.coordinate_mapper,
            is_point_within_canvas_visible_area=SimpleMock(return_value=True),
            any_segment_part_visible_in_canvas_area=SimpleMock(return_value=True),
            zoom_point=Position(1, 1), 
            zoom_direction=1, 
            zoom_step=0.1, 
            offset=Position(0, 0)  # Set to (0,0) for simpler tests
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        # Create mock functions
        self.func1 = SimpleMock(
            name="f1",
            function=lambda x: x,  # Linear function y = x
            left_bound=-5,
            right_bound=5
        )
        self.func2 = SimpleMock(
            name="f2", 
            function=lambda x: x**2,  # Quadratic function y = x^2
            left_bound=-3,
            right_bound=3
        )

    def test_init(self):
        """Test initialization of FunctionsBoundedColoredArea."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        self.assertEqual(area.func1, self.func1)
        self.assertEqual(area.func2, self.func2)
        self.assertEqual(area.canvas, self.canvas)
        self.assertEqual(area.color, "lightblue")
        self.assertEqual(area.opacity, 0.3)

    def test_get_class_name(self):
        """Test class name retrieval."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        self.assertEqual(area.get_class_name(), 'FunctionsBoundedColoredArea')

    def test_get_bounds_model_default_then_clipped_in_renderer(self):
        """Model provides math-space defaults; renderer handles viewport clipping."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        left, right = area._get_bounds()
        # Defaults start at [-10,10] but functions apply tighter bounds [-3,3]
        self.assertEqual((left, right), (-3, 3))

    def test_get_function_y_at_x_with_function_uses_coordinate_mapper(self):
        """Test that _get_function_y_at_x with Function uses coordinate_mapper."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper methods
        self.coordinate_mapper.math_to_screen = SimpleMock(return_value=(100, 200))
        
        # Test with Function object
        result = area._get_function_y_at_x(self.func1, 1.0)
        
        # Verify coordinate_mapper.math_to_screen was called
        self.coordinate_mapper.math_to_screen.assert_called_once()
        
        # Should return the y coordinate from math_to_screen
        self.assertEqual(result, 200)

    def test_get_function_y_at_x_with_constant_uses_coordinate_mapper(self):
        """Test that _get_function_y_at_x with constant uses coordinate_mapper."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper methods
        self.coordinate_mapper.math_to_screen = SimpleMock(return_value=(100, 150))
        
        # Test with constant function
        result = area._get_function_y_at_x(5, 1.0)
        
        # Verify coordinate_mapper.math_to_screen was called with correct arguments
        self.coordinate_mapper.math_to_screen.assert_called_once_with(1.0, 5.0)
        
        # Should return the y coordinate from math_to_screen
        self.assertEqual(result, 150)

    def test_get_function_y_at_x_with_none_uses_coordinate_mapper(self):
        """Test that _get_function_y_at_x with None (x-axis) uses coordinate_mapper."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper methods
        self.coordinate_mapper.math_to_screen = SimpleMock(return_value=(100, 250))
        
        # Test with None (x-axis)
        result = area._get_function_y_at_x(None, 1.0)
        
        # Verify coordinate_mapper.math_to_screen was called with y=0 for x-axis
        self.coordinate_mapper.math_to_screen.assert_called_once_with(1.0, 0)
        
        # Should return the y coordinate from math_to_screen
        self.assertEqual(result, 250)

    def test_get_bounds_applies_coordinate_mapper_bounds_in_renderable(self):
        """Viewport clipping is performed in the renderable, not the model _get_bounds."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper bounds
        self.coordinate_mapper.get_visible_left_bound = SimpleMock(return_value=-8)
        self.coordinate_mapper.get_visible_right_bound = SimpleMock(return_value=8)
        
        renderable = FunctionsBoundedAreaRenderable(area, self.coordinate_mapper)
        left_bound, right_bound = renderable._get_bounds()
        
        # Verify coordinate_mapper methods were called by the renderable
        self.coordinate_mapper.get_visible_left_bound.assert_called_once()
        self.coordinate_mapper.get_visible_right_bound.assert_called_once()
        
        # Bounds should be intersection of model (-3,3) and visible (-8,8) -> (-3,3)
        self.assertEqual(left_bound, -3)
        self.assertEqual(right_bound, 3)

    def test_get_state(self):
        """Test state serialization."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, left_bound=-2, right_bound=2)
        state = area.get_state()
        
        expected_args = {
            "func1": "f1",
            "func2": "f2", 
            "left_bound": -2,
            "right_bound": 2,
            "color": "lightblue",
            "opacity": 0.3,
            "num_sample_points": 100
        }
        self.assertEqual(state["args"], expected_args)

    def test_deepcopy(self):
        """Test deep copy functionality."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        area_copy = copy.deepcopy(area)
        
        self.assertIsNot(area_copy, area)
        self.assertEqual(area_copy.color, area.color)
        self.assertEqual(area_copy.opacity, area.opacity)
        self.assertEqual(area_copy.canvas, area.canvas)  # Canvas reference should be same 

    def test_asymptote_detection_with_tangent_function(self):
        """Test asymptote detection for tangent function."""
        # Create a tangent function with known asymptotes
        import math
        tangent_func = SimpleMock(
            name="f3",  # Special name that triggers asymptote detection
            function=lambda x: math.tan(x/100),
            left_bound=-500,
            right_bound=500
        )
        
        area = FunctionsBoundedColoredArea(tangent_func, self.func2, self.canvas)
        
        # Test asymptote detection at known asymptote positions
        asym_x = 100 * (math.pi/2)  # First asymptote
        dx = 1.0
        
        # Should detect asymptote when very close
        has_asymptote = area._has_asymptote_at(tangent_func, asym_x, dx)
        self.assertTrue(has_asymptote, "Should detect asymptote at π/2 * 100")
        
        # Should not detect asymptote when far away
        has_asymptote = area._has_asymptote_at(tangent_func, 0, dx)
        self.assertFalse(has_asymptote, "Should not detect asymptote at x=0")

    def test_asymptote_handling_during_path_generation(self):
        """Test that asymptotes are properly handled during path generation."""
        # Create function with division by zero at x=0
        asymptote_func = SimpleMock(
            name="asymptote_func",
            function=lambda x: 1/x if x != 0 else float('inf'),
            left_bound=-5,
            right_bound=5
        )
        
        area = FunctionsBoundedColoredArea(asymptote_func, None, self.canvas)
        
        # Use renderable; should build a ClosedArea or None but must not crash
        renderable = FunctionsBoundedAreaRenderable(area, self.coordinate_mapper)
        _ = renderable.build_screen_area(num_points=50)

    def test_bounds_validation_prevents_invalid_bounds(self):
        """Test that constructor validation prevents invalid bounds."""
        # Test case 1: Constructor should reject inverted bounds
        with self.assertRaises(ValueError) as context:
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, 
                                       left_bound=5, right_bound=-5)
        self.assertIn("left_bound must be less than right_bound", str(context.exception))

        # Test case 2: Constructor should reject equal bounds
        with self.assertRaises(ValueError) as context:
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas,
                                      left_bound=3, right_bound=3)
        self.assertIn("left_bound must be less than right_bound", str(context.exception))

    def test_bounds_calculation_internal_correction(self):
        """When function bounds do not overlap, _get_bounds corrects to a small range."""
        # func1 bounds to the left, func2 bounds to the right -> inverted after merging
        func_left = SimpleMock(name="fL", function=lambda x: x, left_bound=-5, right_bound=-3)
        func_right = SimpleMock(name="fR", function=lambda x: x, left_bound=3, right_bound=5)
        area = FunctionsBoundedColoredArea(func_left, func_right, self.canvas)
        left, right = area._get_bounds()
        self.assertLess(left, right, "Internal bounds correction should work")
        self.assertAlmostEqual(right - left, 0.2, places=1)

    def test_function_evaluation_with_nan_and_infinity(self):
        """Test function evaluation with NaN and infinity values."""
        # Function that returns various problematic values
        problematic_func = SimpleMock(
            name="problematic",
            function=lambda x: {
                2: None,
                3: "not_a_number"
            }.get(x, x)  # Return x for other values, None and string for specific cases
        )
        
        area = FunctionsBoundedColoredArea(problematic_func, self.func2, self.canvas)
        
        # Test cases that should return None
        test_cases = [
            (2, None),   # None -> None
            (3, None),   # String -> None (not int/float)
        ]
        
        for x_input, expected in test_cases:
            result = area._get_function_y_at_x(problematic_func, x_input)
            self.assertIsNone(result, f"Expected None for x={x_input}, got {result}")
        
        # Test that normal values work
        result = area._get_function_y_at_x(problematic_func, 5)
        self.assertIsNotNone(result, "Normal values should work")

    def test_coordinate_conversion_integration(self):
        """Test coordinate conversion integration by checking that functions call coordinate mapping."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Test constant function evaluation - should work and return a value
        result = area._get_function_y_at_x(5, 2.0)  # Constant function y = 5
        self.assertIsNotNone(result, "Constant function should return a result")
        
        # Test None function (x-axis) evaluation - should work and return a value
        result_none = area._get_function_y_at_x(None, 2.0)  # x-axis
        self.assertIsNotNone(result_none, "None function (x-axis) should return a result")
        
        # Test that different functions return different results
        result1 = area._get_function_y_at_x(5, 2.0)   # y = 5
        result2 = area._get_function_y_at_x(10, 2.0)  # y = 10
        self.assertNotEqual(result1, result2, "Different constant functions should return different canvas coordinates")

    def test_parameter_validation_comprehensive(self):
        """Test comprehensive parameter validation."""
        # Test invalid func1 types
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea("invalid_func", self.func2, self.canvas)
        
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea([], self.func2, self.canvas)
        
        # Test invalid func2 types  
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea(self.func1, "invalid_func", self.canvas)
        
        # Test invalid bounds types
        with self.assertRaises(TypeError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, left_bound="invalid")
        
        with self.assertRaises(TypeError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, right_bound="invalid")
        
        # Test invalid bounds values
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, left_bound=5, right_bound=3)
        
        # Test invalid num_sample_points
        with self.assertRaises(TypeError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, num_sample_points="invalid")
        
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, num_sample_points=0)
        
        with self.assertRaises(ValueError):
            FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas, num_sample_points=-5)

    def test_canvas_bounds_fallback(self):
        """Test fallback when canvas bounds cannot be determined."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper methods to raise exceptions
        def raise_exception():
            raise Exception("Canvas error")
        
        area.canvas.coordinate_mapper.get_visible_left_bound = raise_exception
        area.canvas.coordinate_mapper.get_visible_right_bound = raise_exception
        
        # _get_bounds should still be stable independent of canvas visibility
        left, right = area._get_bounds()
        self.assertEqual((left, right), (-3, 3))

    def test_path_generation_with_large_values(self):
        """Test path generation with very large function values."""
        # Function that returns very large values
        large_value_func = SimpleMock(
            name="large_func",
            function=lambda x: x**10,  # Very large values for |x| > 1
            left_bound=-2,
            right_bound=2
        )
        
        area = FunctionsBoundedColoredArea(large_value_func, None, self.canvas)
        
        # Mock canvas properties needed for asymptote handling
        area.canvas.height = 500
        area.canvas.cartesian2axis.origin.y = 250
        area.canvas.scale_factor = 1
        
        # Should handle large values without crashing
        result = area._get_function_y_at_x_with_asymptote_handling(
            large_value_func, 100, 1.5, 0.1, 0, 500, 50
        )
        
        # Should return a clipped value, not crash
        self.assertIsNotNone(result)

    def test_function_name_generation_edge_cases(self):
        """Test function name generation with edge cases."""
        # Test with negative constant
        area1 = FunctionsBoundedColoredArea(-2.5, self.func1, self.canvas)
        self.assertEqual(area1.name, "area_between_y_-2.5_and_f1")
        
        # Test with zero constant
        area2 = FunctionsBoundedColoredArea(0, None, self.canvas) 
        self.assertEqual(area2.name, "area_between_y_0_and_x_axis")
        
        # Test with very large number
        area3 = FunctionsBoundedColoredArea(1e6, self.func2, self.canvas)
        self.assertEqual(area3.name, "area_between_y_1000000.0_and_f2")

    def test_draw_method_with_no_valid_points(self):
        """Test draw method when no valid points can be generated."""
        # Function that always returns None
        invalid_func = SimpleMock(
            name="invalid_func",
            function=lambda x: float('nan'),  # Always invalid
            left_bound=-1,
            right_bound=1
        )
        
        area = FunctionsBoundedColoredArea(invalid_func, None, self.canvas)
        
        # Should handle gracefully without crashing using renderable
        try:
            renderable = FunctionsBoundedAreaRenderable(area, self.coordinate_mapper)
            _ = renderable.build_screen_area(num_points=50)
        except Exception as e:
            self.fail(f"Renderable should handle invalid functions gracefully, but raised: {e}")

    def test_reverse_path_generation(self):
        """Test reverse path generation functionality."""
        area = FunctionsBoundedColoredArea(self.func1, self.func2, self.canvas)
        
        # Mock coordinate_mapper for predictable results
        call_count = [0]  # Use list to allow modification in nested function
        def mock_math_to_screen(x, y):
            call_count[0] += 1
            return (x * 10 + 250, 250 - y * 10)  # Simple linear transformation
        
        self.coordinate_mapper.math_to_screen = mock_math_to_screen
        
        # Mock canvas properties to avoid any potential issues
        self.canvas.height = 500
        
        # Generate reverse path with a constant function for predictability
        reverse_points = area._generate_path(5, -1, 1, 0.5, 5, reverse=True)  # Constant function y=5
        
        # Should generate points
        self.assertGreater(len(reverse_points), 0, "Should generate reverse points")
        
        # Verify coordinate conversions were called
        self.assertGreater(call_count[0], 0, "Should call coordinate conversion") 