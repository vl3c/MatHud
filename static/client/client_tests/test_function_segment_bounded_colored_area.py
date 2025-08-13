import unittest
import copy
from geometry import Position
from coordinate_mapper import CoordinateMapper
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from drawables.function import Function
from .simple_mock import SimpleMock


class TestFunctionSegmentBoundedColoredArea(unittest.TestCase):
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
        
        # Create mock function
        self.func = SimpleMock(
            name="f1",
            function=lambda x: x**2,  # Quadratic function y = x^2
            left_bound=-5,
            right_bound=5
        )
        
        # Create mock segment
        self.segment = SimpleMock(
            name="AB",
            point1=SimpleMock(
                x=100, y=200,  # Canvas coordinates
                original_position=SimpleMock(x=-150, y=50)  # Math coordinates
            ),
            point2=SimpleMock(
                x=400, y=300,  # Canvas coordinates  
                original_position=SimpleMock(x=150, y=-50)  # Math coordinates
            )
        )

    def test_init(self):
        """Test initialization of FunctionSegmentBoundedColoredArea."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        self.assertEqual(area.func, self.func)
        self.assertEqual(area.segment, self.segment)
        self.assertEqual(area.canvas, self.canvas)
        self.assertEqual(area.color, "lightblue")
        self.assertEqual(area.opacity, 0.3)

    def test_get_class_name(self):
        """Test class name retrieval."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        self.assertEqual(area.get_class_name(), 'FunctionSegmentBoundedColoredArea')

    def test_generate_name(self):
        """Test name generation."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        expected_name = "area_between_f1_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_generate_name_with_none_function(self):
        """Test name generation with None function (x-axis)."""
        area = FunctionSegmentBoundedColoredArea(None, self.segment, self.canvas)
        expected_name = "area_between_x_axis_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_generate_name_with_constant_function(self):
        """Test name generation with constant function."""
        area = FunctionSegmentBoundedColoredArea(5, self.segment, self.canvas)
        expected_name = "area_between_y_5_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_get_function_y_at_x_with_none(self):
        """Test _get_function_y_at_x with None (x-axis)."""
        area = FunctionSegmentBoundedColoredArea(None, self.segment, self.canvas)
        result = area._get_function_y_at_x(100)
        self.assertEqual(result, 0)

    def test_get_function_y_at_x_with_constant(self):
        """Test _get_function_y_at_x with constant function."""
        area = FunctionSegmentBoundedColoredArea(3.5, self.segment, self.canvas)
        result = area._get_function_y_at_x(100)
        self.assertEqual(result, 3.5)

    def test_calculate_function_y_value_with_math_coordinates(self):
        """Test that _calculate_function_y_value works with math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        # Test with math x coordinate (function is y = x^2)
        result = area._calculate_function_y_value(2.0)  # Math x coordinate
        
        # Should return the math y coordinate: 2^2 = 4
        self.assertEqual(result, 4.0)

    def test_calculate_function_y_value_handles_exceptions(self):
        """Test that _calculate_function_y_value handles exceptions gracefully."""
        # Create a function that throws an exception
        bad_func = SimpleMock(
            name="bad_func",
            function=lambda x: 1/0  # This will cause ZeroDivisionError
        )
        
        area = FunctionSegmentBoundedColoredArea(bad_func, self.segment, self.canvas)
        
        result = area._calculate_function_y_value(2.0)  # Math coordinate
        
        # Should return None when exception occurs
        self.assertIsNone(result)

    def test_get_segment_bounds(self):
        """Test _get_segment_bounds returns correct min/max in math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        left_bound, right_bound = area._get_segment_bounds()
        
        # Should return min and max of segment point original_position.x coordinates
        self.assertEqual(left_bound, -150)  # min(-150, 150)
        self.assertEqual(right_bound, 150)  # max(-150, 150)

    def test_get_intersection_bounds(self):
        """Test _get_intersection_bounds calculates proper intersection in math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        # Test with segment bounds [-150, 150] (math coords) and function bounds [-5, 5]
        left_bound, right_bound = area._get_intersection_bounds(-150, 150)
        
        # Should return intersection: max(-150, -5) to min(150, 5)
        self.assertEqual(left_bound, -5)
        self.assertEqual(right_bound, 5)

    def test_uses_segment(self):
        """Test uses_segment method correctly identifies segment usage."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        # Create matching segment
        matching_segment = SimpleMock(
            point1=SimpleMock(x=100, y=200),
            point2=SimpleMock(x=400, y=300)
        )
        
        # Create non-matching segment
        different_segment = SimpleMock(
            point1=SimpleMock(x=150, y=250),
            point2=SimpleMock(x=450, y=350)
        )
        
        self.assertTrue(area.uses_segment(matching_segment))
        self.assertFalse(area.uses_segment(different_segment))

    def test_get_state(self):
        """Test state serialization."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        state = area.get_state()
        
        expected_args = {
            "func": "f1",
            "segment": "AB"
        }
        # Check that the args contain the expected function and segment names
        self.assertEqual(state["args"]["func"], expected_args["func"])
        self.assertEqual(state["args"]["segment"], expected_args["segment"])

    def test_deepcopy(self):
        """Test deep copy functionality."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        area_copy = copy.deepcopy(area)
        
        self.assertIsNot(area_copy, area)
        self.assertEqual(area_copy.func, area.func)
        self.assertEqual(area_copy.segment, area.segment)
        self.assertEqual(area_copy.color, area.color)
        self.assertEqual(area_copy.opacity, area.opacity)
        self.assertEqual(area_copy.canvas, area.canvas)  # Canvas reference should be same 

    def test_coordinate_conversion_accuracy(self):
        """Test that coordinate conversion is accurate between math and canvas coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        # Test specific coordinate conversions
        test_coords = [(-150, 22500), (0, 0), (150, 22500)]  # (x, x^2) pairs
        
        for x_math, expected_y_math in test_coords:
            y_math = area._calculate_function_y_value(x_math)
            self.assertEqual(y_math, expected_y_math, f"Math calculation failed for x={x_math}")

    def test_function_with_domain_restrictions(self):
        """Test function with restricted domain bounds."""
        restricted_func = SimpleMock(
            name="restricted",
            function=lambda x: x**2,
            left_bound=-2,   # Restricted domain
            right_bound=2
        )
        
        area = FunctionSegmentBoundedColoredArea(restricted_func, self.segment, self.canvas)
        
        # Test intersection bounds calculation
        left_bound, right_bound = area._get_intersection_bounds(-150, 150)
        
        # Should intersect with function bounds: max(-150, -2) to min(150, 2)
        self.assertEqual(left_bound, -2)
        self.assertEqual(right_bound, 2)

    def test_function_evaluation_error_handling(self):
        """Test graceful handling of function evaluation errors."""
        # Create function that throws ZeroDivisionError 
        error_func = SimpleMock(
            name="error_func",
            function=lambda x: 1/0 if x == 0 else 1/x  # Division by zero exception
        )
        
        area = FunctionSegmentBoundedColoredArea(error_func, self.segment, self.canvas)
        
        # Test that ZeroDivisionError is caught and returns None
        result = area._calculate_function_y_value(0)
        self.assertIsNone(result, "ZeroDivisionError should be caught and return None")
        
        # Test that normal values work  
        result = area._calculate_function_y_value(2)
        self.assertAlmostEqual(result, 0.5, places=5)  # 1/2 = 0.5

    def test_segment_bounds_with_swapped_points(self):
        """Test segment bounds calculation when points are in reverse order."""
        # Create segment with points swapped (larger x first)
        swapped_segment = SimpleMock(
            name="BA",  # Reverse order
            point1=SimpleMock(
                x=400, y=300,  # Canvas coordinates
                original_position=SimpleMock(x=150, y=-50)  # Math coordinates (larger x first)
            ),
            point2=SimpleMock(
                x=100, y=200,  # Canvas coordinates
                original_position=SimpleMock(x=-150, y=50)  # Math coordinates (smaller x second)
            )
        )
        
        area = FunctionSegmentBoundedColoredArea(self.func, swapped_segment, self.canvas)
        
        left_bound, right_bound = area._get_segment_bounds()
        
        # Should still return min, max correctly regardless of point order
        self.assertEqual(left_bound, -150)  # min(-150, 150)
        self.assertEqual(right_bound, 150)  # max(-150, 150)

    def test_generate_function_points_coordinate_conversion(self):
        """Test that _generate_function_points properly converts coordinates and generates points."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        
        # Test the method directly - generate 3 points from x=-1 to x=1
        # For y=x^2: x=-1 gives y=1, x=0 gives y=0, x=1 gives y=1
        left_bound, right_bound, num_points = -1, 1, 3
        dx = (right_bound - left_bound) / (num_points - 1)  # dx = 1.0
        points = area._generate_function_points(left_bound, right_bound, num_points, dx)
        
        # Verify that points were generated
        self.assertGreater(len(points), 0, "Should generate at least some points")
        self.assertEqual(len(points), 3, "Should generate exactly 3 points")
        
        # Verify that all points are tuples with two elements (x, y)
        for i, point in enumerate(points):
            self.assertIsInstance(point, tuple, f"Point {i} should be a tuple")
            self.assertEqual(len(point), 2, f"Point {i} should have 2 coordinates")
            self.assertIsInstance(point[0], (int, float), f"Point {i} x-coordinate should be numeric")
            self.assertIsInstance(point[1], (int, float), f"Point {i} y-coordinate should be numeric")
        
        # Test that function evaluation works directly
        y_val = area._get_function_y_at_x(2)  # Should return 4 for x^2
        self.assertEqual(y_val, 4, "Function evaluation should work")
        
        # Test edge cases
        y_val_zero = area._get_function_y_at_x(0)  # Should return 0 for x^2
        self.assertEqual(y_val_zero, 0, "Function evaluation at x=0 should work")
        
        y_val_negative = area._get_function_y_at_x(-3)  # Should return 9 for (-3)^2
        self.assertEqual(y_val_negative, 9, "Function evaluation with negative x should work")

    def test_draw_method_integration(self):
        """Integration via renderable: build screen area paths."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment, self.canvas)
        renderable = FunctionSegmentAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area(num_points=50)
        self.assertIsNotNone(closed_area, "ClosedArea should be produced")
        self.assertGreater(len(closed_area.forward_points), 0)
        self.assertEqual(len(closed_area.reverse_points), 2)
        expected_reverse = [(400, 300), (100, 200)]
        self.assertEqual(closed_area.reverse_points, expected_reverse)

    def test_edge_case_single_point_segment(self):
        """Test edge case where segment endpoints are the same."""
        # Create segment where both points are identical
        single_point_segment = SimpleMock(
            name="AA",
            point1=SimpleMock(
                x=250, y=250,  # Same canvas coordinates
                original_position=SimpleMock(x=0, y=0)  # Same math coordinates
            ),
            point2=SimpleMock(
                x=250, y=250,  # Same canvas coordinates
                original_position=SimpleMock(x=0, y=0)  # Same math coordinates  
            )
        )
        
        area = FunctionSegmentBoundedColoredArea(self.func, single_point_segment, self.canvas)
        
        # Should still work without errors
        left_bound, right_bound = area._get_segment_bounds()
        self.assertEqual(left_bound, 0)
        self.assertEqual(right_bound, 0)
        
        # Should handle intersection bounds
        left_bound, right_bound = area._get_intersection_bounds(0, 0)
        self.assertEqual(left_bound, 0)  # max(0, -5)
        self.assertEqual(right_bound, 0)  # min(0, 5) 