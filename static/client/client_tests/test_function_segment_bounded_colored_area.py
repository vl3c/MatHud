import unittest
import copy
from geometry import Position
from coordinate_mapper import CoordinateMapper
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from drawables.function import Function
from .simple_mock import SimpleMock


class TestFunctionSegmentBoundedColoredArea(unittest.TestCase):
    def setUp(self) -> None:
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
        
        # Create mock segment (math coordinates on x/y)
        self.segment = SimpleMock(
            name="AB",
            point1=SimpleMock(
                x=-150, y=50
            ),
            point2=SimpleMock(
                x=150, y=-50
            )
        )

    def test_init(self) -> None:
        """Test initialization of FunctionSegmentBoundedColoredArea."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        self.assertEqual(area.func, self.func)
        self.assertEqual(area.segment, self.segment)
        self.assertEqual(area.color, "lightblue")
        self.assertEqual(area.opacity, 0.3)

    def test_get_class_name(self) -> None:
        """Test class name retrieval."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        self.assertEqual(area.get_class_name(), 'FunctionSegmentBoundedColoredArea')

    def test_generate_name(self) -> None:
        """Test name generation."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        expected_name = "area_between_f1_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_generate_name_with_none_function(self) -> None:
        """Test name generation with None function (x-axis)."""
        area = FunctionSegmentBoundedColoredArea(None, self.segment)
        expected_name = "area_between_x_axis_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_generate_name_with_constant_function(self) -> None:
        """Test name generation with constant function."""
        area = FunctionSegmentBoundedColoredArea(5, self.segment)
        expected_name = "area_between_y_5_and_AB"
        self.assertEqual(area.name, expected_name)

    def test_get_function_y_at_x_with_none(self) -> None:
        """Test _get_function_y_at_x with None (x-axis)."""
        area = FunctionSegmentBoundedColoredArea(None, self.segment)
        result = area._get_function_y_at_x(100)
        self.assertEqual(result, 0)

    def test_get_function_y_at_x_with_constant(self) -> None:
        """Test _get_function_y_at_x with constant function."""
        area = FunctionSegmentBoundedColoredArea(3.5, self.segment)
        result = area._get_function_y_at_x(100)
        self.assertEqual(result, 3.5)

    def test_calculate_function_y_value_with_math_coordinates(self) -> None:
        """Test that _calculate_function_y_value works with math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
        # Test with math x coordinate (function is y = x^2)
        result = area._calculate_function_y_value(2.0)  # Math x coordinate
        
        # Should return the math y coordinate: 2^2 = 4
        self.assertEqual(result, 4.0)

    def test_calculate_function_y_value_handles_exceptions(self) -> None:
        """Test that _calculate_function_y_value handles exceptions gracefully."""
        # Create a function that throws an exception
        bad_func = SimpleMock(
            name="bad_func",
            function=lambda x: 1/0  # This will cause ZeroDivisionError
        )
        
        area = FunctionSegmentBoundedColoredArea(bad_func, self.segment)
        
        result = area._calculate_function_y_value(2.0)  # Math coordinate
        
        # Should return None when exception occurs
        self.assertIsNone(result)

    def test_get_segment_bounds(self) -> None:
        """Test _get_segment_bounds returns correct min/max in math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
        left_bound, right_bound = area._get_segment_bounds()
        
        # Should return min and max of segment point x coordinates (math)
        self.assertEqual(left_bound, -150)  # min(-150, 150)
        self.assertEqual(right_bound, 150)  # max(-150, 150)

    def test_get_intersection_bounds(self) -> None:
        """Test _get_intersection_bounds calculates proper intersection in math coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
        # Test with segment bounds [-150, 150] (math coords) and function bounds [-5, 5]
        left_bound, right_bound = area._get_intersection_bounds(-150, 150)
        
        # Should return intersection: max(-150, -5) to min(150, 5)
        self.assertEqual(left_bound, -5)
        self.assertEqual(right_bound, 5)

    def test_uses_segment(self) -> None:
        """Test uses_segment method correctly identifies segment usage."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
        # Create matching segment
        matching_segment = SimpleMock(
            point1=SimpleMock(x=-150, y=50),
            point2=SimpleMock(x=150, y=-50)
        )
        
        # Create non-matching segment
        different_segment = SimpleMock(
            point1=SimpleMock(x=150, y=250),
            point2=SimpleMock(x=450, y=350)
        )
        
        self.assertTrue(area.uses_segment(matching_segment))
        self.assertFalse(area.uses_segment(different_segment))

    def test_get_state(self) -> None:
        """Test state serialization."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        state = area.get_state()
        
        expected_args = {
            "func": "f1",
            "segment": "AB"
        }
        # Check that the args contain the expected function and segment names
        self.assertEqual(state["args"]["func"], expected_args["func"])
        self.assertEqual(state["args"]["segment"], expected_args["segment"])

    def test_deepcopy(self) -> None:
        """Test deep copy functionality."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        area_copy = copy.deepcopy(area)
        
        self.assertIsNot(area_copy, area)
        self.assertEqual(area_copy.func, area.func)
        self.assertEqual(area_copy.segment, area.segment)
        self.assertEqual(area_copy.color, area.color)
        self.assertEqual(area_copy.opacity, area.opacity)

    def test_coordinate_conversion_accuracy(self) -> None:
        """Test that coordinate conversion is accurate between math and canvas coordinates."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
        # Test specific coordinate conversions
        test_coords = [(-150, 22500), (0, 0), (150, 22500)]  # (x, x^2) pairs
        
        for x_math, expected_y_math in test_coords:
            y_math = area._calculate_function_y_value(x_math)
            self.assertEqual(y_math, expected_y_math, f"Math calculation failed for x={x_math}")

    def test_function_with_domain_restrictions(self) -> None:
        """Test function with restricted domain bounds."""
        restricted_func = SimpleMock(
            name="restricted",
            function=lambda x: x**2,
            left_bound=-2,   # Restricted domain
            right_bound=2
        )
        
        area = FunctionSegmentBoundedColoredArea(restricted_func, self.segment)
        
        # Test intersection bounds calculation
        left_bound, right_bound = area._get_intersection_bounds(-150, 150)
        
        # Should intersect with function bounds: max(-150, -2) to min(150, 2)
        self.assertEqual(left_bound, -2)
        self.assertEqual(right_bound, 2)

    def test_function_evaluation_error_handling(self) -> None:
        """Test graceful handling of function evaluation errors."""
        # Create function that throws ZeroDivisionError 
        error_func = SimpleMock(
            name="error_func",
            function=lambda x: 1/0 if x == 0 else 1/x  # Division by zero exception
        )
        
        area = FunctionSegmentBoundedColoredArea(error_func, self.segment)
        
        # Test that ZeroDivisionError is caught and returns None
        result = area._calculate_function_y_value(0)
        self.assertIsNone(result, "ZeroDivisionError should be caught and return None")
        
        # Test that normal values work  
        result = area._calculate_function_y_value(2)
        self.assertAlmostEqual(result, 0.5, places=5)  # 1/2 = 0.5

    def test_segment_bounds_with_swapped_points(self) -> None:
        """Test segment bounds calculation when points are in reverse order."""
        # Create segment with points swapped (larger x first)
        swapped_segment = SimpleMock(
            name="BA",  # Reverse order
            point1=SimpleMock(
                x=150, y=-50
            ),
            point2=SimpleMock(
                x=-150, y=50
            )
        )
        
        area = FunctionSegmentBoundedColoredArea(self.func, swapped_segment)
        
        left_bound, right_bound = area._get_segment_bounds()
        
        # Should still return min, max correctly regardless of point order
        self.assertEqual(left_bound, -150)  # min(-150, 150)
        self.assertEqual(right_bound, 150)  # max(-150, 150)

    def test_generate_function_points_coordinate_conversion(self) -> None:
        """Test that _generate_function_points properly converts coordinates and generates points."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        
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

    def test_draw_method_integration(self) -> None:
        """Integration via renderable: build screen area paths."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        renderable = FunctionSegmentAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area(num_points=50)
        self.assertIsNotNone(closed_area, "ClosedArea should be produced")
        self.assertGreater(len(closed_area.forward_points), 0)
        self.assertEqual(len(closed_area.reverse_points), 2)
        self.assertFalse(closed_area.is_screen)
        expected_reverse = [
            (self.segment.point2.x, self.segment.point2.y),
            (self.segment.point1.x, self.segment.point1.y),
        ]
        self.assertEqual(closed_area.reverse_points, expected_reverse)

    def test_segment_points_follow_mapper_transformations(self) -> None:
        """Segment reverse path should honor mapper scale and offset."""
        area = FunctionSegmentBoundedColoredArea(self.func, self.segment)
        # Apply non-default mapper transforms
        self.coordinate_mapper.scale_factor = 1.5
        self.coordinate_mapper.offset.x = 25
        self.coordinate_mapper.offset.y = -40
        renderable = FunctionSegmentAreaRenderable(area, self.coordinate_mapper)
        closed_area = renderable.build_screen_area(num_points=25)
        self.assertIsNotNone(closed_area, "ClosedArea should be produced with mapper transforms applied")
        self.assertFalse(closed_area.is_screen)
        expected_reverse = [
            (self.segment.point2.x, self.segment.point2.y),
            (self.segment.point1.x, self.segment.point1.y),
        ]
        self.assertEqual(closed_area.reverse_points, expected_reverse)

    def test_edge_case_single_point_segment(self) -> None:
        """Test edge case where segment endpoints are the same."""
        # Create segment where both points are identical
        single_point_segment = SimpleMock(
            name="AA",
            point1=SimpleMock(
                x=0, y=0
            ),
            point2=SimpleMock(
                x=0, y=0
            )
        )
        
        area = FunctionSegmentBoundedColoredArea(self.func, single_point_segment)
        
        # Should still work without errors
        left_bound, right_bound = area._get_segment_bounds()
        self.assertEqual(left_bound, 0)
        self.assertEqual(right_bound, 0)
        
        # Should handle intersection bounds
        left_bound, right_bound = area._get_intersection_bounds(0, 0)
        self.assertEqual(left_bound, 0)  # max(0, -5)
        self.assertEqual(right_bound, 0)  # min(0, 5) 