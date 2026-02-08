"""
Tests for PiecewiseFunction drawable and its integration with FunctionRenderable.
"""

from __future__ import annotations

import math
import unittest

from coordinate_mapper import CoordinateMapper
from drawables.piecewise_function import PiecewiseFunction
from drawables.piecewise_function_interval import PiecewiseFunctionInterval
from rendering.renderables import FunctionRenderable


class TestPiecewiseFunctionInterval(unittest.TestCase):
    """Tests for the PiecewiseFunctionInterval data class."""

    def test_contains_unbounded_left(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x^2",
            evaluator=lambda x: x**2,
            left=None,
            right=0,
            left_inclusive=True,
            right_inclusive=False,
        )
        self.assertTrue(interval.contains(-100))
        self.assertTrue(interval.contains(-1))
        self.assertFalse(interval.contains(0))
        self.assertFalse(interval.contains(1))

    def test_contains_unbounded_right(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x+1",
            evaluator=lambda x: x + 1,
            left=0,
            right=None,
            left_inclusive=True,
            right_inclusive=True,
        )
        self.assertTrue(interval.contains(0))
        self.assertTrue(interval.contains(1))
        self.assertTrue(interval.contains(100))
        self.assertFalse(interval.contains(-1))

    def test_contains_bounded_interval(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="2*x",
            evaluator=lambda x: 2 * x,
            left=-1,
            right=1,
            left_inclusive=True,
            right_inclusive=False,
        )
        self.assertTrue(interval.contains(-1))
        self.assertTrue(interval.contains(0))
        self.assertTrue(interval.contains(0.5))
        self.assertFalse(interval.contains(1))
        self.assertFalse(interval.contains(-2))

    def test_contains_right_inclusive(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x",
            evaluator=lambda x: x,
            left=0,
            right=1,
            left_inclusive=False,
            right_inclusive=True,
        )
        self.assertFalse(interval.contains(0))
        self.assertTrue(interval.contains(0.5))
        self.assertTrue(interval.contains(1))

    def test_evaluate(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x^2",
            evaluator=lambda x: x**2,
            left=None,
            right=None,
            left_inclusive=True,
            right_inclusive=True,
        )
        self.assertEqual(interval.evaluate(2), 4)
        self.assertEqual(interval.evaluate(-3), 9)

    def test_to_dict(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x+1",
            evaluator=lambda x: x + 1,
            left=-5,
            right=5,
            left_inclusive=True,
            right_inclusive=False,
        )
        result = interval.to_dict()
        self.assertEqual(result["expression"], "x+1")
        self.assertEqual(result["left"], -5)
        self.assertEqual(result["right"], 5)
        self.assertTrue(result["left_inclusive"])
        self.assertFalse(result["right_inclusive"])


class TestPiecewiseFunction(unittest.TestCase):
    """Tests for the PiecewiseFunction drawable class."""

    def test_create_two_piece_function(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertEqual(pf.name, "f")
        self.assertEqual(len(pf.intervals), 2)
        self.assertEqual(pf.get_class_name(), "PiecewiseFunction")

    def test_function_evaluation_left_piece(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertAlmostEqual(pf.function(-2), 4.0)
        self.assertAlmostEqual(pf.function(-1), 1.0)

    def test_function_evaluation_right_piece(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertAlmostEqual(pf.function(0), 1.0)
        self.assertAlmostEqual(pf.function(1), 2.0)
        self.assertAlmostEqual(pf.function(5), 6.0)

    def test_function_evaluation_outside_pieces_returns_nan(self) -> None:
        pieces = [
            {"expression": "x", "left": -1, "right": 1, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        result = pf.function(5)
        self.assertTrue(math.isnan(result))

    def test_discontinuity_at_piece_boundary(self) -> None:
        pieces = [
            {"expression": "0", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertIn(0, pf.point_discontinuities)

    def test_continuous_piecewise_no_discontinuity(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertNotIn(0, pf.point_discontinuities)

    def test_three_piece_function(self) -> None:
        pieces = [
            {"expression": "-1", "left": None, "right": -1, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x", "left": -1, "right": 1, "left_inclusive": True, "right_inclusive": False},
            {"expression": "1", "left": 1, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="clamp")

        self.assertEqual(len(pf.intervals), 3)
        self.assertAlmostEqual(pf.function(-5), -1.0)
        self.assertAlmostEqual(pf.function(0), 0.0)
        self.assertAlmostEqual(pf.function(5), 1.0)

    def test_get_state(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        state = pf.get_state()
        self.assertEqual(state["name"], "f")
        self.assertIn("pieces", state["args"])
        self.assertEqual(len(state["args"]["pieces"]), 2)

    def test_deepcopy(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        import copy
        pf_copy = copy.deepcopy(pf)

        self.assertEqual(pf_copy.name, "f")
        self.assertEqual(len(pf_copy.intervals), 2)
        self.assertIsNot(pf_copy, pf)

    def test_update_color(self) -> None:
        pieces = [{"expression": "x", "left": None, "right": None, "left_inclusive": True, "right_inclusive": True}]
        pf = PiecewiseFunction(pieces, name="f", color="blue")

        pf.update_color("red")
        self.assertEqual(pf.color, "red")

    def test_left_and_right_bounds_from_pieces(self) -> None:
        pieces = [
            {"expression": "x", "left": -5, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x^2", "left": 0, "right": 5, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertEqual(pf.left_bound, -5)
        self.assertEqual(pf.right_bound, 5)

    def test_unbounded_left_piece_gives_none_left_bound(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x^2", "left": 0, "right": 5, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertIsNone(pf.left_bound)
        self.assertEqual(pf.right_bound, 5)


class TestPiecewiseFunctionRendering(unittest.TestCase):
    """Tests for PiecewiseFunction rendering with FunctionRenderable."""

    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def test_constant_piecewise_function_renders(self) -> None:
        """Test that constant value piecewise functions render correctly."""
        pieces = [
            {"expression": "2", "left": -10, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "5", "left": 0, "right": 10, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="constant_piecewise")

        # Verify function evaluates correctly
        self.assertAlmostEqual(pf.function(-5), 2.0)
        self.assertAlmostEqual(pf.function(5), 5.0)

        # Verify bounds are computed
        self.assertEqual(pf.left_bound, -10)
        self.assertEqual(pf.right_bound, 10)

        # Verify discontinuity is detected at x=0 but NOT at the bounds
        self.assertIn(0, pf.point_discontinuities)
        self.assertNotIn(-10, pf.point_discontinuities, "Left bound should not be a discontinuity")
        self.assertNotIn(10, pf.point_discontinuities, "Right bound should not be a discontinuity")

        # Verify rendering works
        renderable = FunctionRenderable(pf, self.mapper)
        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.paths)
        self.assertGreater(len(result.paths), 0, "Constant piecewise function should produce paths")

        # Should have multiple paths due to discontinuity at x=0
        self.assertGreaterEqual(len(result.paths), 2, "Step function should produce at least 2 separate paths")

        # Should have points in each path
        for i, path in enumerate(result.paths):
            self.assertGreater(len(path), 0, f"Path {i} should have points")

    def test_piecewise_function_renders_with_function_renderable(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")
        renderable = FunctionRenderable(pf, self.mapper)

        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.paths)
        self.assertGreater(len(result.paths), 0)

    def test_piecewise_with_jump_discontinuity_produces_multiple_paths(self) -> None:
        pieces = [
            {"expression": "0", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="step")
        renderable = FunctionRenderable(pf, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 1)

    def test_continuous_piecewise_can_produce_single_path(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="linear")
        renderable = FunctionRenderable(pf, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)

    def test_absolute_value_like_piecewise(self) -> None:
        pieces = [
            {"expression": "-x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="abs_like")
        renderable = FunctionRenderable(pf, self.mapper)

        result = renderable.build_screen_paths()

        self.assertIsNotNone(result)
        total_points = sum(len(path) for path in result.paths)
        self.assertGreater(total_points, 2)

    def test_piecewise_with_asymptotes_in_one_piece(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "1/x", "left": 0, "right": None, "left_inclusive": False, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")
        renderable = FunctionRenderable(pf, self.mapper)

        result = renderable.build_screen_paths()

        self.assertGreater(len(result.paths), 0)


class TestPiecewiseFunctionUndefinedAt(unittest.TestCase):
    """Tests for undefined_at (hole) support in PiecewiseFunction."""

    def test_interval_with_single_undefined_point(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="2",
            evaluator=lambda x: 2,
            left=None,
            right=None,
            left_inclusive=True,
            right_inclusive=True,
            undefined_at=[0],
        )
        self.assertTrue(interval.contains(-1))
        self.assertTrue(interval.contains(1))
        self.assertFalse(interval.contains(0))
        self.assertTrue(math.isnan(interval.evaluate(0)))
        self.assertEqual(interval.evaluate(1), 2)

    def test_interval_with_multiple_undefined_points(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x",
            evaluator=lambda x: x,
            left=None,
            right=None,
            left_inclusive=True,
            right_inclusive=True,
            undefined_at=[-1, 0, 1],
        )
        self.assertFalse(interval.contains(0))
        self.assertFalse(interval.contains(1))
        self.assertFalse(interval.contains(-1))
        self.assertTrue(interval.contains(0.5))
        self.assertTrue(interval.contains(2))

    def test_piecewise_function_with_undefined_at(self) -> None:
        pieces = [
            {"expression": "2", "left": None, "right": None, "left_inclusive": True, "right_inclusive": True, "undefined_at": [0]},
        ]
        pf = PiecewiseFunction(pieces, name="f_with_hole")

        self.assertAlmostEqual(pf.function(-5), 2.0)
        self.assertAlmostEqual(pf.function(5), 2.0)
        self.assertTrue(math.isnan(pf.function(0)))
        self.assertIn(0, pf.point_discontinuities)

    def test_piecewise_with_multiple_holes_across_intervals(self) -> None:
        pieces = [
            {"expression": "x", "left": None, "right": 0, "right_inclusive": False, "undefined_at": [-2]},
            {"expression": "x^2", "left": 0, "right": None, "left_inclusive": True, "undefined_at": [1, 3]},
        ]
        pf = PiecewiseFunction(pieces, name="multi_hole")

        self.assertTrue(math.isnan(pf.function(-2)))
        self.assertTrue(math.isnan(pf.function(1)))
        self.assertTrue(math.isnan(pf.function(3)))
        self.assertAlmostEqual(pf.function(-1), -1.0)
        self.assertAlmostEqual(pf.function(2), 4.0)
        self.assertIn(-2, pf.point_discontinuities)
        self.assertIn(1, pf.point_discontinuities)
        self.assertIn(3, pf.point_discontinuities)

    def test_interval_to_dict_includes_undefined_at(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x",
            evaluator=lambda x: x,
            left=-5,
            right=5,
            left_inclusive=True,
            right_inclusive=False,
            undefined_at=[0, 2],
        )
        result = interval.to_dict()
        self.assertEqual(result["undefined_at"], [0, 2])

    def test_interval_to_dict_omits_undefined_at_when_empty(self) -> None:
        interval = PiecewiseFunctionInterval(
            expression="x",
            evaluator=lambda x: x,
            left=-5,
            right=5,
            left_inclusive=True,
            right_inclusive=False,
        )
        result = interval.to_dict()
        self.assertNotIn("undefined_at", result)


class TestPiecewiseFunctionEdgeCases(unittest.TestCase):
    """Edge case tests for PiecewiseFunction."""

    def test_single_piece_function(self) -> None:
        pieces = [
            {"expression": "x^2", "left": None, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertEqual(len(pf.intervals), 1)
        self.assertAlmostEqual(pf.function(3), 9.0)

    def test_empty_pieces_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            PiecewiseFunction([], name="f")

    def test_invalid_expression_raises_error(self) -> None:
        pieces = [
            {"expression": "invalid_function(x)", "left": None, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        with self.assertRaises(ValueError):
            PiecewiseFunction(pieces, name="f")

    def test_overlapping_intervals_uses_first_match(self) -> None:
        pieces = [
            {"expression": "1", "left": None, "right": 1, "left_inclusive": True, "right_inclusive": True},
            {"expression": "2", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        self.assertAlmostEqual(pf.function(0.5), 1.0)

    def test_gap_in_intervals_returns_nan(self) -> None:
        pieces = [
            {"expression": "1", "left": None, "right": -1, "left_inclusive": True, "right_inclusive": True},
            {"expression": "2", "left": 1, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        pf = PiecewiseFunction(pieces, name="f")

        result = pf.function(0)
        self.assertTrue(math.isnan(result))


__all__ = [
    "TestPiecewiseFunctionInterval",
    "TestPiecewiseFunction",
    "TestPiecewiseFunctionRendering",
    "TestPiecewiseFunctionUndefinedAt",
    "TestPiecewiseFunctionEdgeCases",
]

