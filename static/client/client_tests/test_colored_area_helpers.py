from __future__ import annotations

import unittest
from types import SimpleNamespace
from client_tests.simple_mock import SimpleMock
from rendering.shared_drawable_renderers import (
    render_colored_area_helper,
    _points_close,
    _paths_form_single_loop,
    _filter_valid_points,
)


class TestPointsClose(unittest.TestCase):
    def test_identical_points(self) -> None:
        self.assertTrue(_points_close((0.0, 0.0), (0.0, 0.0)))

    def test_points_within_tolerance(self) -> None:
        self.assertTrue(_points_close((0.0, 0.0), (1e-10, 1e-10)))

    def test_points_outside_tolerance(self) -> None:
        self.assertFalse(_points_close((0.0, 0.0), (1e-8, 0.0)))

    def test_custom_tolerance(self) -> None:
        self.assertTrue(_points_close((0.0, 0.0), (0.5, 0.5), tol=1.0))
        self.assertFalse(_points_close((0.0, 0.0), (0.5, 0.5), tol=0.1))


class TestPathsFormSingleLoop(unittest.TestCase):
    def test_empty_forward(self) -> None:
        self.assertFalse(_paths_form_single_loop([], []))

    def test_too_few_points(self) -> None:
        self.assertFalse(_paths_form_single_loop([(0, 0), (1, 1)], [(1, 1), (0, 0)]))

    def test_mismatched_lengths(self) -> None:
        forward = [(0, 0), (1, 0), (1, 1)]
        reverse = [(1, 1), (1, 0)]
        self.assertFalse(_paths_form_single_loop(forward, reverse))

    def test_matching_loop(self) -> None:
        forward = [(0, 0), (1, 0), (1, 1)]
        reverse = [(1, 1), (1, 0), (0, 0)]
        self.assertTrue(_paths_form_single_loop(forward, reverse))

    def test_non_matching_loop(self) -> None:
        forward = [(0, 0), (1, 0), (1, 1)]
        reverse = [(2, 2), (1, 0), (0, 0)]
        self.assertFalse(_paths_form_single_loop(forward, reverse))


class TestFilterValidPoints(unittest.TestCase):
    def test_all_valid_points(self) -> None:
        points = [(0, 0), (1, 1), (2, 2)]
        result = _filter_valid_points(points)
        self.assertEqual(result, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])

    def test_filters_none_points(self) -> None:
        points = [(0, 0), None, (2, 2)]
        result = _filter_valid_points(points)
        self.assertEqual(result, [(0.0, 0.0), (2.0, 2.0)])

    def test_filters_points_with_none_coords(self) -> None:
        points = [(0, 0), (None, 1), (2, 2)]
        result = _filter_valid_points(points)
        self.assertEqual(result, [(0.0, 0.0), (2.0, 2.0)])

    def test_filters_empty_points(self) -> None:
        points = [(0, 0), (), (2, 2)]
        result = _filter_valid_points(points)
        self.assertEqual(result, [(0.0, 0.0), (2.0, 2.0)])

    def test_empty_input(self) -> None:
        self.assertEqual(_filter_valid_points([]), [])


class TestRenderColoredAreaHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.primitives = SimpleMock()
        self.primitives.fill_polygon = SimpleMock()
        self.primitives.fill_joined_area = SimpleMock()
        self.mapper = SimpleNamespace(
            math_to_screen=lambda x, y: (x * 10, y * 10),
            scale_factor=1.0,
        )
        self.style = {"area_fill_color": "blue", "area_opacity": 0.5}

    def test_none_area_returns_early(self) -> None:
        render_colored_area_helper(self.primitives, None, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 0)
        self.assertEqual(len(self.primitives.fill_joined_area.calls), 0)

    def test_empty_forward_returns_early(self) -> None:
        area = SimpleNamespace(forward_points=[], reverse_points=[(0, 0)])
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 0)

    def test_empty_reverse_returns_early(self) -> None:
        area = SimpleNamespace(forward_points=[(0, 0)], reverse_points=[])
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 0)

    def test_insufficient_points_returns_early(self) -> None:
        area = SimpleNamespace(
            forward_points=[(0, 0)],
            reverse_points=[(1, 1)],
            is_screen=False,
        )
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 0)

    def test_screen_space_points_not_transformed(self) -> None:
        area = SimpleNamespace(
            forward_points=[(0, 0), (10, 0), (10, 10)],
            reverse_points=[(10, 10), (10, 0), (0, 0)],
            is_screen=True,
            color="red",
            opacity=0.7,
        )
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 1)
        call_args, call_kwargs = self.primitives.fill_polygon.calls[0]
        points = call_args[0]
        self.assertIn((0.0, 0.0), points)
        self.assertIn((10.0, 0.0), points)

    def test_math_space_points_transformed(self) -> None:
        area = SimpleNamespace(
            forward_points=[(0, 0), (1, 0), (1, 1)],
            reverse_points=[(1, 1), (1, 0), (0, 0)],
            is_screen=False,
            color="green",
            opacity=0.5,
        )
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 1)
        call_args, call_kwargs = self.primitives.fill_polygon.calls[0]
        points = call_args[0]
        self.assertIn((0.0, 0.0), points)
        self.assertIn((10.0, 0.0), points)
        self.assertIn((10.0, 10.0), points)

    def test_uses_style_defaults(self) -> None:
        area = SimpleNamespace(
            forward_points=[(0, 0), (1, 0), (1, 1)],
            reverse_points=[(1, 1), (1, 0), (0, 0)],
            is_screen=False,
        )
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        self.assertEqual(len(self.primitives.fill_polygon.calls), 1)
        call_args, call_kwargs = self.primitives.fill_polygon.calls[0]
        fill = call_args[1]
        self.assertEqual(fill.color, "blue")
        self.assertEqual(fill.opacity, 0.5)

    def test_invalid_opacity_uses_default(self) -> None:
        area = SimpleNamespace(
            forward_points=[(0, 0), (1, 0), (1, 1)],
            reverse_points=[(1, 1), (1, 0), (0, 0)],
            is_screen=False,
            color="red",
            opacity="invalid",
        )
        render_colored_area_helper(self.primitives, area, self.mapper, self.style)
        call_args, call_kwargs = self.primitives.fill_polygon.calls[0]
        fill = call_args[1]
        self.assertEqual(fill.opacity, 0.3)


__all__ = [
    "TestPointsClose",
    "TestPathsFormSingleLoop",
    "TestFilterValidPoints",
    "TestRenderColoredAreaHelper",
]

