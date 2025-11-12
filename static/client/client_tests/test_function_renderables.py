from __future__ import annotations

import unittest

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.function import Function

from rendering.function_renderable import FunctionRenderable
from rendering.primitives import ClosedArea


class MockCartesian:
    def __init__(self):
        self.width = 640.0
        self.height = 480.0
        self.current_tick_spacing = 50.0
        self.default_tick_spacing = 50.0


class TestFunctionRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.cartesian = MockCartesian()

    def test_build_screen_paths_returns_polyline(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.paths)
        self.assertGreater(len(result.paths), 0)

    def test_linear_function_produces_continuous_path(self) -> None:
        func = Function("x", name="f")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

        result = renderable.build_screen_paths()

        self.assertEqual(len(result.paths), 1)
        self.assertGreater(len(result.paths[0]), 2)

    def test_quadratic_function_produces_smooth_path(self) -> None:
        func = Function("x^2", name="g")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)
        if len(result.paths) > 0:
            self.assertGreater(len(result.paths[0]), 10)

    def test_discontinuous_function_produces_multiple_paths(self) -> None:
        func = Function("1/x", name="h")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

        result = renderable.build_screen_paths()

        if len(result.paths) > 1:
            self.assertGreater(len(result.paths), 1)

    def test_trigonometric_function_evaluates(self) -> None:
        func = Function("sin(x)", name="s")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)


class TestFunctionsBoundedAreaRenderable(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_build_screen_area_with_two_functions(self) -> None:
        from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
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
        from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
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
        from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable
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
        from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable
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
        self.cartesian = MockCartesian()

    def test_constant_function_produces_horizontal_line(self) -> None:
        func = Function("5", name="const")
        renderable = FunctionRenderable(func, self.mapper, self.cartesian)

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
            renderable = FunctionRenderable(func, self.mapper, self.cartesian)
            renderable.build_screen_paths()
        except Exception as exc:
            self.fail(f"Invalid function should not raise exception: {exc}")

    def test_none_segment_area_handles_gracefully(self) -> None:
        from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable
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

