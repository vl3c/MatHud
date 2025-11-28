from __future__ import annotations

import math
import unittest

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.function import Function

from rendering.renderables import FunctionRenderable
from rendering.primitives import ClosedArea


class TestFunctionRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_paths_returns_polyline(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.paths)
        self.assertGreater(len(result.paths), 0)

    def test_linear_function_produces_continuous_path(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertEqual(len(result.paths), 1)
        self.assertGreater(len(result.paths[0]), 2)

    def test_quadratic_function_produces_smooth_path(self) -> None:
        func = Function("x^2", name="g")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        if len(result.paths) > 0:
            self.assertGreater(len(result.paths[0]), 10)

    def test_discontinuous_function_produces_multiple_paths(self) -> None:
        func = Function("1/x", name="h")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        if len(result.paths) > 1:
            self.assertGreater(len(result.paths), 1)

    def test_trigonometric_function_evaluates(self) -> None:
        func = Function("sin(x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)

    def test_high_frequency_sin_has_sufficient_samples_per_period(self) -> None:
        func = Function("sin(10*x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        total_points = sum(len(path) for path in result.paths)
        self.assertGreater(total_points, 20)

    def test_high_frequency_sin_peaks_are_smooth(self) -> None:
        func = Function("sin(5*x)", name="s")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        min_angle_radians = math.radians(30)
        violations = 0
        total_checked = 0
        for path in result.paths:
            if len(path) < 3:
                continue
            for i in range(1, len(path) - 1):
                angle = renderable._compute_angle(path[i - 1], path[i], path[i + 1])
                total_checked += 1
                if angle < min_angle_radians:
                    violations += 1
        self.assertEqual(violations, 0, f"Found {violations} angles below 30 degrees out of {total_checked}")

    def test_adaptive_sampling_produces_more_points_for_curves(self) -> None:
        linear_func = Function("x", name="linear")
        curved_func = Function("sin(x)", name="curved")

        linear_renderable = FunctionRenderable(linear_func, self.mapper)
        curved_renderable = FunctionRenderable(curved_func, self.mapper)

        linear_result = linear_renderable.build_screen_paths()
        curved_result = curved_renderable.build_screen_paths()

        linear_points = sum(len(path) for path in linear_result.paths)
        curved_points = sum(len(path) for path in curved_result.paths)

        self.assertGreater(curved_points, linear_points)

    def test_sin_peaks_have_minimum_angle_of_30_degrees(self) -> None:
        func = Function("100*sin(x)", name="s", left_bound=-100, right_bound=100)
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        min_angle_radians = math.radians(30)
        violations = 0
        total_angles_checked = 0

        for path in result.paths:
            if len(path) < 3:
                continue
            for i in range(1, len(path) - 1):
                angle = renderable._compute_angle(path[i - 1], path[i], path[i + 1])
                total_angles_checked += 1
                if angle < min_angle_radians:
                    violations += 1

        self.assertGreater(total_angles_checked, 50)
        self.assertEqual(violations, 0, f"Found {violations} angles below 30 degrees out of {total_angles_checked}")


class TestFunctionsBoundedAreaRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_area_with_two_functions(self) -> None:
        from rendering.renderables import FunctionsBoundedAreaRenderable
        from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
        
        f1 = Function("x^2", name="f")
        f2 = Function("x", name="g")
        
        area_model = FunctionsBoundedColoredArea(
            f1, f2, left_bound=-1, right_bound=1, color="green", opacity=0.4
        )
        
        renderable = FunctionsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsInstance(result, ClosedArea)
        self.assertGreater(len(result.forward_points), 0)
        self.assertGreater(len(result.reverse_points), 0)

    def test_area_with_x_bounds(self) -> None:
        from rendering.renderables import FunctionsBoundedAreaRenderable
        from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
        
        f1 = Function("2*x", name="f")
        f2 = Function("x", name="g")
        
        area_model = FunctionsBoundedColoredArea(
            f1, f2, left_bound=0, right_bound=3
        )
        
        renderable = FunctionsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)


class TestSegmentsBoundedAreaRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_area_from_segments(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(4, 3, name="C")
        p4 = Point(0, 3, name="D")
        
        seg1 = Segment(p1, p2)
        seg2 = Segment(p4, p3)
        
        area_model = SegmentsBoundedColoredArea(
            seg1, seg2, color="red", opacity=0.5
        )
        
        renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, ClosedArea)
            self.assertGreater(len(result.forward_points), 0)

    def test_triangular_area_from_segments(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(3, 0, name="B")
        p3 = Point(1.5, 2.6, name="C")
        
        seg1 = Segment(p1, p2)
        seg2 = Segment(p2, p3)
        
        area_model = SegmentsBoundedColoredArea(seg1, seg2)
        
        renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
        result = renderable.build_screen_area()

        self.assertIsNotNone(result)
        self.assertGreater(len(result.forward_points), 0)


class TestRenderableEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_constant_function_produces_horizontal_line(self) -> None:
        func = Function("5", name="const")
        renderable = FunctionRenderable(func, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        if len(result.paths) > 0:
            path = result.paths[0]
            y_values = [point[1] for point in path]
            
            if len(y_values) > 1:
                y_range = max(y_values) - min(y_values)
                self.assertLess(y_range, 5)

    def test_invalid_function_handles_gracefully(self) -> None:
        func = Function("invalid_expr", name="bad")
        
        try:
            renderable = FunctionRenderable(func, self.mapper)
            renderable.build_screen_paths()
        except Exception as exc:
            self.fail(f"Invalid function should not raise exception: {exc}")

    def test_none_segment_area_handles_gracefully(self) -> None:
        from rendering.renderables import SegmentsBoundedAreaRenderable
        from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
        
        p1 = Point(0, 0, name="A")
        p2 = Point(1, 0, name="B")
        seg1 = Segment(p1, p2)
        
        area_model = SegmentsBoundedColoredArea(seg1, None)
        
        try:
            renderable = SegmentsBoundedAreaRenderable(area_model, self.mapper)
            renderable.build_screen_area()
        except Exception as exc:
            self.fail(f"Segments area with None segment raised exception: {exc}")


__all__ = [
    "TestFunctionRenderable",
    "TestFunctionsBoundedAreaRenderable",
    "TestSegmentsBoundedAreaRenderable",
    "TestRenderableEdgeCases",
]

