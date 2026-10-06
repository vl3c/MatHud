"""Tests for numeric roots, extrema and intersections of plotted functions.

TestFunctionFeaturesAlgorithm exercises the pure module ``utils.function_features``;
TestFunctionFeaturesCanvas the ``find_function_features`` tool through the Canvas.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager
from utils.function_features import (
    KIND_INFLECTION,
    KIND_INTERSECTION,
    KIND_LOCAL_MAX,
    KIND_LOCAL_MIN,
    KIND_ROOT,
    find_function_features,
    find_intersections,
    sample_count,
)


def _of_kind(report: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [feature for feature in report["features"] if feature["kind"] == kind]


def _xs(report: Dict[str, Any], kind: str) -> List[float]:
    return [feature["x"] for feature in _of_kind(report, kind)]


class TestFunctionFeaturesAlgorithm(unittest.TestCase):
    """The pure root, extremum and intersection finder."""

    def assertValues(self, actual: List[float], expected: List[float], places: int = 7) -> None:
        self.assertEqual(len(actual), len(expected), f"{actual} != {expected}")
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=places)

    # ---- polynomials ----

    def test_quadratic_roots_and_minimum(self) -> None:
        report = find_function_features(lambda x: x * x - 4, -5, 5)
        self.assertEqual(_xs(report, KIND_ROOT), [-2.0, 2.0])
        minima = _of_kind(report, KIND_LOCAL_MIN)
        self.assertEqual([(m["x"], m["y"]) for m in minima], [(0.0, -4.0)])
        self.assertEqual(_of_kind(report, KIND_LOCAL_MAX), [])
        self.assertFalse(report["truncated"])

    def test_cubic_roots_and_both_extrema(self) -> None:
        report = find_function_features(lambda x: x**3 - 3 * x, -3, 3)
        self.assertValues(_xs(report, KIND_ROOT), [-math.sqrt(3), 0.0, math.sqrt(3)], places=9)
        self.assertEqual([(f["x"], f["y"]) for f in _of_kind(report, KIND_LOCAL_MAX)], [(-1.0, 2.0)])
        self.assertEqual([(f["x"], f["y"]) for f in _of_kind(report, KIND_LOCAL_MIN)], [(1.0, -2.0)])

    def test_features_are_sorted_by_x(self) -> None:
        report = find_function_features(lambda x: x**3 - 3 * x, -3, 3)
        xs = [feature["x"] for feature in report["features"]]
        self.assertEqual(xs, sorted(xs))

    def test_irrational_root_is_reported_with_ten_significant_digits(self) -> None:
        report = find_function_features(lambda x: x * x - 2, 0, 3, features=["roots"])
        self.assertEqual(_xs(report, KIND_ROOT), [1.414213562])

    # ---- double roots ----

    def test_double_root_is_a_touching_root_and_a_minimum(self) -> None:
        report = find_function_features(lambda x: x * x, -5, 4.3)
        roots = _of_kind(report, KIND_ROOT)
        self.assertEqual(len(roots), 1)
        self.assertEqual((roots[0]["x"], roots[0]["y"]), (0.0, 0.0))
        self.assertTrue(roots[0].get("touching"))
        self.assertEqual([(f["x"], f["y"]) for f in _of_kind(report, KIND_LOCAL_MIN)], [(0.0, 0.0)])

    def test_double_root_on_a_sample_point_is_not_duplicated(self) -> None:
        # 0 is a grid point on [-5, 5]; its exact zero sample and the minimum are one root
        report = find_function_features(lambda x: x * x, -5, 5)
        roots = _of_kind(report, KIND_ROOT)
        self.assertEqual(len(roots), 1)
        self.assertTrue(roots[0].get("touching"))

    def test_shifted_double_root(self) -> None:
        report = find_function_features(lambda x: (x - 1) ** 2 * (x + 2), -4, 4)
        roots = _of_kind(report, KIND_ROOT)
        self.assertEqual([root["x"] for root in roots], [-2.0, 1.0])
        self.assertFalse(roots[0].get("touching", False))
        self.assertTrue(roots[1].get("touching"))

    def test_kink_on_the_axis_is_a_touching_root(self) -> None:
        report = find_function_features(lambda x: abs(x - 0.3), -5, 10)
        self.assertEqual(_xs(report, KIND_ROOT), [0.3])
        self.assertEqual([(f["x"], f["y"]) for f in _of_kind(report, KIND_LOCAL_MIN)], [(0.3, 0.0)])

    def test_minimum_just_above_the_axis_is_not_a_root(self) -> None:
        report = find_function_features(lambda x: x * x + 0.001, -2, 2)
        self.assertEqual(_of_kind(report, KIND_ROOT), [])
        self.assertEqual(len(_of_kind(report, KIND_LOCAL_MIN)), 1)

    # ---- trigonometric ----

    def test_sine_roots_and_extrema(self) -> None:
        report = find_function_features(math.sin, -7, 7)
        self.assertValues(_xs(report, KIND_ROOT), [k * math.pi for k in (-2, -1, 0, 1, 2)], places=8)
        self.assertValues(_xs(report, KIND_LOCAL_MAX), [-1.5 * math.pi, 0.5 * math.pi], places=5)
        self.assertValues(_xs(report, KIND_LOCAL_MIN), [-0.5 * math.pi, 1.5 * math.pi], places=5)
        for feature in _of_kind(report, KIND_LOCAL_MAX):
            self.assertEqual(feature["y"], 1.0)

    def test_fast_oscillation_with_enough_samples(self) -> None:
        report = find_function_features(lambda x: math.sin(100 * x), 0.001, 0.1)
        self.assertEqual(len(_of_kind(report, KIND_ROOT)), 3)
        self.assertEqual(len(_of_kind(report, KIND_LOCAL_MAX)), 2)

    # ---- poles and discontinuities ----

    def test_reciprocal_has_no_false_root_at_its_pole(self) -> None:
        report = find_function_features(lambda x: 1 / x, -5, 5)
        self.assertEqual(report["features"], [])

    def test_reciprocal_with_breakpoint_has_no_features(self) -> None:
        report = find_function_features(lambda x: 1 / x, -5, 5, breakpoints=[0.0])
        self.assertEqual(report["features"], [])

    def test_tangent_roots_only_at_multiples_of_pi(self) -> None:
        report = find_function_features(math.tan, -5, 5)
        self.assertValues(_xs(report, KIND_ROOT), [-math.pi, 0.0, math.pi], places=8)
        self.assertEqual(_of_kind(report, KIND_LOCAL_MIN) + _of_kind(report, KIND_LOCAL_MAX), [])

    def test_even_pole_is_not_a_maximum(self) -> None:
        report = find_function_features(lambda x: 1 / (x * x), -5, 5)
        self.assertEqual(report["features"], [])

    def test_shifted_pole_keeps_the_real_root(self) -> None:
        report = find_function_features(lambda x: 1 / (x - 1) + 1, -5, 5)
        self.assertEqual(_xs(report, KIND_ROOT), [0.0])

    def test_jump_is_not_a_root(self) -> None:
        report = find_function_features(lambda x: -1.0 if x < 0.37 else 1.0, -2, 2)
        self.assertEqual(_of_kind(report, KIND_ROOT), [])

    def test_undefined_regions_are_skipped(self) -> None:
        report = find_function_features(lambda x: math.sqrt(x) - 2, -5, 10)
        self.assertEqual(_xs(report, KIND_ROOT), [4.0])
        report = find_function_features(math.log, -5, 10)
        self.assertEqual(_xs(report, KIND_ROOT), [1.0])

    # ---- empty results and intervals ----

    def test_no_real_roots(self) -> None:
        report = find_function_features(lambda x: x * x + 1, -5, 5, features=["roots"])
        self.assertEqual(report["features"], [])
        self.assertEqual(report["total_found"], 0)
        self.assertFalse(report["truncated"])

    def test_monotonic_and_constant_functions_have_no_extrema(self) -> None:
        self.assertEqual(find_function_features(math.exp, -5, 5)["features"], [])
        self.assertEqual(find_function_features(lambda x: 3.0, -5, 5)["features"], [])
        self.assertEqual(find_function_features(lambda x: x**3, -2, 2.1, features=["extrema"])["features"], [])

    def test_zero_function_reports_one_zero_interval(self) -> None:
        report = find_function_features(lambda x: 0.0, -5, 5)
        self.assertEqual(len(report["features"]), 1)
        self.assertEqual(report["features"][0]["zero_interval"], [-5.0, 5.0])

    def test_interval_bounds_limit_the_search(self) -> None:
        report = find_function_features(lambda x: x * x - 4, 0, 5)
        self.assertEqual(_xs(report, KIND_ROOT), [2.0])
        self.assertEqual(_of_kind(report, KIND_LOCAL_MIN), [])

    def test_feature_selection(self) -> None:
        roots_only = find_function_features(lambda x: x * x - 4, -5, 5, features=["roots"])
        self.assertEqual({f["kind"] for f in roots_only["features"]}, {KIND_ROOT})
        extrema_only = find_function_features(lambda x: x * x, -5, 4.3, features=["extrema"])
        self.assertEqual([f["kind"] for f in extrema_only["features"]], [KIND_LOCAL_MIN])

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 2, 1)
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 0, float("inf"))
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 0, 1, features=["inflection_points"])
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 0, 1, features=[])

    def test_results_are_capped_and_flagged(self) -> None:
        report = find_function_features(math.sin, 0.5, 100.5, features=["roots"], max_results=10)
        self.assertEqual(len(report["features"]), 10)
        self.assertEqual(report["total_found"], 31)
        self.assertTrue(report["truncated"])

    def test_sample_count_grows_with_interval_screen_and_period(self) -> None:
        self.assertEqual(sample_count(1.0), 500)
        self.assertEqual(sample_count(100.0), 2500)
        self.assertEqual(sample_count(10.0, pixel_span=2000.0), 4000)
        self.assertEqual(sample_count(10.0, period=0.05), 8000)
        self.assertEqual(sample_count(1e6), 10000)

    # ---- intersections ----

    def test_intersections_of_parabola_and_line(self) -> None:
        report = find_intersections(lambda x: x * x, lambda x: x + 2, -5, 5)
        self.assertEqual(
            [(f["x"], f["y"], f["kind"]) for f in report["features"]],
            [
                (-1.0, 1.0, KIND_INTERSECTION),
                (2.0, 4.0, KIND_INTERSECTION),
            ],
        )

    def test_intersections_of_sine_and_cosine(self) -> None:
        report = find_intersections(math.sin, math.cos, -4, 4)
        self.assertValues([f["x"] for f in report["features"]], [-3 * math.pi / 4, math.pi / 4, 5 * math.pi / 4], 8)
        self.assertAlmostEqual(report["features"][1]["y"], math.sqrt(0.5), places=9)

    def test_tangent_line_meets_in_one_touching_point(self) -> None:
        report = find_intersections(lambda x: x * x, lambda x: 2 * x - 1, -5, 5.1)
        self.assertEqual(len(report["features"]), 1)
        self.assertEqual((report["features"][0]["x"], report["features"][0]["y"]), (1.0, 1.0))
        self.assertTrue(report["features"][0].get("touching"))

    def test_disjoint_curves_do_not_intersect(self) -> None:
        report = find_intersections(lambda x: x * x + 1, lambda x: x - 5, -5, 5)
        self.assertEqual(report["features"], [])

    def test_intersections_skip_poles(self) -> None:
        # 1/x and -1/x never meet; their difference 2/x changes sign only at the pole
        report = find_intersections(lambda x: 1 / x, lambda x: -1 / x, -3, 3)
        self.assertEqual(report["features"], [])

    # ---- features closer than the sampling, and far from 0 ----

    def test_two_roots_inside_one_sample_step(self) -> None:
        report = find_function_features(lambda x: (x - 0.5) * (x - 0.5001), 0.0123, 1.1)
        self.assertEqual(_xs(report, KIND_ROOT), [0.5, 0.5001])
        self.assertEqual(_xs(report, KIND_LOCAL_MIN), [0.50005])

    def test_close_inflections_are_resolved(self) -> None:
        # (x^2 - 1e-6)^3 changes concavity at -0.001, -0.000447, 0.000447 and 0.001 (steps of 0.005)
        report = find_function_features(lambda x: (x * x - 1e-6) ** 3, -1.37, 1.3, features=["inflections"])
        s = 0.001 / math.sqrt(5)
        self.assertValues(_xs(report, KIND_INFLECTION), [-0.001, -s, s, 0.001], places=9)

    def test_steep_inflection_between_two_samples(self) -> None:
        report = find_function_features(lambda x: math.tanh(1e4 * (x - 0.123)), -1.37, 1.3, features=["inflections"])
        self.assertEqual(_xs(report, KIND_INFLECTION), [0.123])

    def test_inflections_far_from_zero(self) -> None:
        report = find_function_features(lambda x: (x - 1e7) ** 3, 1e7 - 5.3, 1e7 + 4.1, features=["inflections"])
        self.assertEqual(_xs(report, KIND_INFLECTION), [1e7])
        report = find_function_features(
            lambda x: math.exp(-((x - 1e6) ** 2)), 1e6 - 5.3, 1e6 + 4.1, features=["inflections"]
        )
        self.assertValues(_xs(report, KIND_INFLECTION), [1e6 - math.sqrt(0.5), 1e6 + math.sqrt(0.5)], places=2)

    def test_refined_samples_find_no_extrema_beside_poles(self) -> None:
        report = find_function_features(lambda x: 1 / (x - 0.37) ** 2, -2.5, 2.5, features=["extrema", "inflections"])
        self.assertEqual(report["features"], [])


class TestFunctionFeaturesCanvas(unittest.TestCase):
    """The find_function_features tool through Canvas -> DrawableManager -> FunctionManager."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _find(
        self,
        names: List[str],
        features: Optional[List[str]] = None,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        place_points: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self.canvas.find_function_features(
            names, features=features, left_bound=bounds[0], right_bound=bounds[1], place_points=place_points
        )

    def _points(self) -> List[Any]:
        return list(self.canvas.get_drawables_by_class_name("Point"))

    def test_roots_and_extremum_of_a_plotted_function(self) -> None:
        self.canvas.draw_function("x^2 - 4", name="f")
        result = self._find(["f"], bounds=(-5, 5))
        self.assertEqual(result["function_names"], ["f"])
        self.assertEqual(result["interval"], [-5.0, 5.0])
        self.assertEqual(
            [(f["x"], f["y"], f["kind"]) for f in result["features"]],
            [(-2.0, 0.0, "root"), (0.0, -4.0, "local_min"), (2.0, 0.0, "root")],
        )
        self.assertEqual(result["count"], 3)
        self.assertFalse(result["truncated"])
        self.assertNotIn("point_names", result)
        self.assertEqual(self._points(), [])

    def test_tangent_has_no_roots_at_its_asymptotes(self) -> None:
        self.canvas.draw_function("tan(x)", name="t")
        result = self._find(["t"], bounds=(-3, 3))
        self.assertEqual([(f["x"], f["kind"]) for f in result["features"]], [(0.0, "root")])

    def test_reciprocal_reports_nothing_with_a_note(self) -> None:
        self.canvas.draw_function("1/x", name="r")
        result = self._find(["r"], bounds=(-5, 5))
        self.assertEqual(result["features"], [])
        self.assertEqual(result["count"], 0)
        self.assertIn("No requested features", result["note"])

    def test_double_root_through_the_canvas(self) -> None:
        self.canvas.draw_function("x^2", name="f")
        result = self._find(["f"], bounds=(-3, 2.2))
        kinds = [(f["x"], f["kind"], f.get("touching", False)) for f in result["features"]]
        self.assertEqual(kinds, [(0.0, "root", True), (0.0, "local_min", False)])

    def test_feature_selection_through_the_canvas(self) -> None:
        self.canvas.draw_function("x^3 - 3*x", name="f")
        result = self._find(["f"], features=["extrema"], bounds=(-3, 3))
        self.assertEqual(
            [(f["x"], f["y"], f["kind"]) for f in result["features"]],
            [(-1.0, 2.0, "local_max"), (1.0, -2.0, "local_min")],
        )

    def test_inflection_points_through_the_canvas(self) -> None:
        self.canvas.draw_function("x^3 - 3*x", name="f")
        result = self._find(["f"], features=["inflections"], bounds=(-3, 3))
        self.assertEqual([(f["x"], f["y"], f["kind"]) for f in result["features"]], [(0.0, 0.0, "inflection")])
        self.canvas.draw_function("tan(x)", name="t")
        tangent = self._find(["t"], features=["inflections"], bounds=(-3, 3))
        self.assertEqual([(f["x"], f["kind"]) for f in tangent["features"]], [(0.0, "inflection")])

    def test_inflection_points_can_be_placed(self) -> None:
        self.canvas.draw_function("x^3", name="f")
        result = self._find(["f"], features=["inflections"], bounds=(-2, 2.1), place_points=True)
        self.assertEqual(len(result["created_point_names"]), 1)
        points = self._points()
        self.assertEqual([(p.x, p.y) for p in points], [(0.0, 0.0)])

    def test_function_bounds_are_the_default_interval(self) -> None:
        self.canvas.draw_function("x^2 - 1", name="f", left_bound=0, right_bound=5)
        result = self._find(["f"])
        self.assertEqual(result["interval"], [0.0, 5.0])
        self.assertEqual([(f["x"], f["kind"]) for f in result["features"]], [(1.0, "root")])

    def test_requested_interval_is_clipped_to_the_function_bounds(self) -> None:
        self.canvas.draw_function("x^2 - 1", name="f", left_bound=0, right_bound=5)
        result = self._find(["f"], bounds=(-10, 10))
        self.assertEqual(result["interval"], [0.0, 5.0])
        with self.assertRaises(ValueError):
            self._find(["f"], bounds=(-10, -5))

    def test_visible_range_is_the_default_without_bounds(self) -> None:
        self.canvas.draw_function("x - 3", name="f")
        mapper = self.canvas.coordinate_mapper
        result = self._find(["f"], features=["roots"])
        self.assertAlmostEqual(result["interval"][0], mapper.get_visible_left_bound(), places=6)
        self.assertAlmostEqual(result["interval"][1], mapper.get_visible_right_bound(), places=6)
        self.assertEqual([f["x"] for f in result["features"]], [3.0])

    def test_intersections_of_two_functions(self) -> None:
        self.canvas.draw_function("x^2", name="f")
        self.canvas.draw_function("x + 2", name="g")
        result = self._find(["f", "g"], bounds=(-5, 5))
        self.assertEqual(result["function_names"], ["f", "g"])
        self.assertEqual(
            [(f["x"], f["y"], f["kind"]) for f in result["features"]],
            [(-1.0, 1.0, "intersection"), (2.0, 4.0, "intersection")],
        )

    def test_intersections_use_the_shared_interval(self) -> None:
        self.canvas.draw_function("x^2", name="f", left_bound=0, right_bound=10)
        self.canvas.draw_function("x + 2", name="g", left_bound=-10, right_bound=3)
        result = self._find(["f", "g"])
        self.assertEqual(result["interval"], [0.0, 3.0])
        self.assertEqual([f["x"] for f in result["features"]], [2.0])

    def test_no_intersection_note(self) -> None:
        self.canvas.draw_function("x^2 + 1", name="f")
        self.canvas.draw_function("x - 5", name="g")
        result = self._find(["f", "g"], bounds=(-5, 5))
        self.assertEqual(result["features"], [])
        self.assertIn("No intersections", result["note"])

    def test_piecewise_functions_are_supported(self) -> None:
        pieces = [
            {"expression": "x^2 - 1", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x - 2", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ]
        self.canvas.draw_piecewise_function(pieces, name="p")
        result = self._find(["p"], features=["roots"], bounds=(-5, 5))
        self.assertEqual([f["x"] for f in result["features"]], [-1.0, 2.0])

    def test_bad_names_raise(self) -> None:
        self.canvas.draw_function("x", name="f")
        with self.assertRaises(ValueError):
            self._find(["missing"], bounds=(-1, 1))
        with self.assertRaises(ValueError):
            self._find([], bounds=(-1, 1))
        with self.assertRaises(ValueError):
            self._find(["f", "f"], bounds=(-1, 1))
        with self.assertRaises(ValueError):
            self._find(["f", "f", "f"], bounds=(-1, 1))

    def test_place_points_creates_named_points(self) -> None:
        self.canvas.draw_function("x^2 - 4", name="f")
        result = self._find(["f"], bounds=(-5, 5), place_points=True)
        self.assertEqual(len(result["point_names"]), 3)
        self.assertEqual(result["created_point_names"], result["point_names"])
        self.assertEqual(result["reused_point_names"], [])
        self.assertNotIn("note", result)
        points = {p.name: (p.x, p.y) for p in self._points()}
        for feature in result["features"]:
            self.assertEqual(points[feature["point_name"]], (feature["x"], feature["y"]))

    def test_touching_root_and_extremum_share_one_point(self) -> None:
        self.canvas.draw_function("x^2", name="f")
        result = self._find(["f"], bounds=(-3, 2.2), place_points=True)
        self.assertEqual(len(result["point_names"]), 1)
        self.assertEqual(len(self._points()), 1)
        self.assertEqual({f["point_name"] for f in result["features"]}, set(result["point_names"]))

    def test_placed_points_are_one_undo_step(self) -> None:
        self.canvas.draw_function("sin(x)", name="f")
        result = self._find(["f"], bounds=(-7, 7), place_points=True)
        self.assertEqual(len(self._points()), len(result["point_names"]))
        self.assertEqual(len(result["point_names"]), 9)

        self.assertTrue(self.canvas.undo())
        self.assertEqual(self._points(), [])
        self.assertIsNotNone(self.canvas.get_function("f"))

        self.assertTrue(self.canvas.redo())
        self.assertEqual(len(self._points()), 9)

    def test_existing_point_is_reused(self) -> None:
        self.canvas.draw_function("x^2 - 4", name="f")
        existing = self.canvas.create_point(2, 0, name="P", extra_graphics=False)
        result = self._find(["f"], features=["roots"], bounds=(-5, 5), place_points=True)
        self.assertIn(existing.name, result["point_names"])
        self.assertEqual(len(self._points()), 2)
        # The user's point is reported as reused, never as created (so the AI won't delete it)
        self.assertEqual(result["reused_point_names"], [existing.name])
        self.assertEqual(len(result["created_point_names"]), 1)
        self.assertNotIn(existing.name, result["created_point_names"])
        self.assertIn(existing.name, result["note"])

    def test_shared_point_of_touching_root_counts_as_created_once(self) -> None:
        self.canvas.draw_function("x^2", name="f")
        result = self._find(["f"], bounds=(-3, 2.2), place_points=True)
        self.assertEqual(result["created_point_names"], result["point_names"])
        self.assertEqual(len(result["created_point_names"]), 1)
        self.assertEqual(result["reused_point_names"], [])

    def test_without_place_points_nothing_is_archived(self) -> None:
        self.canvas.draw_function("x^2 - 4", name="f")
        undo_count = len(self.canvas.undo_redo_manager.undo_stack)
        self._find(["f"], bounds=(-5, 5))
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), undo_count)

    def _run_tool_calls(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run tool calls through the same registry and result processor the AI uses."""
        available = FunctionRegistry.get_available_functions(self.canvas, WorkspaceManager(self.canvas))
        undoable = FunctionRegistry.get_undoable_functions()
        return dict(ProcessFunctionCalls.get_results(calls, available, undoable, self.canvas))

    def test_tool_call_returns_the_values_to_the_ai(self) -> None:
        # Not in the undoable list, so the result is the payload, not "Call successful"
        self.assertNotIn("find_function_features", FunctionRegistry.get_undoable_functions())
        self.canvas.draw_function("x^2 - 4", name="f")
        arguments = {
            "function_names": ["f"],
            "features": None,
            "left_bound": -5,
            "right_bound": 5,
            "place_points": None,
        }
        results = self._run_tool_calls([{"function_name": "find_function_features", "arguments": arguments}])
        value = list(results.values())[0]
        self.assertIsInstance(value, dict)
        self.assertEqual([f["x"] for f in value["features"]], [-2.0, 0.0, 2.0])

    def test_tool_batch_with_plot_and_points_undoes_in_one_step(self) -> None:
        calls = [
            {
                "function_name": "draw_function",
                "arguments": {
                    "function_string": "x^2 - 1",
                    "name": "f",
                    "left_bound": None,
                    "right_bound": None,
                    "color": None,
                    "undefined_at": None,
                },
            },
            {
                "function_name": "find_function_features",
                "arguments": {
                    "function_names": ["f"],
                    "features": ["roots"],
                    "left_bound": -3,
                    "right_bound": 3,
                    "place_points": True,
                },
            },
        ]
        results = self._run_tool_calls(calls)
        feature_result = [v for k, v in results.items() if k.startswith("find_function_features")][0]
        self.assertEqual(len(feature_result["point_names"]), 2)
        self.assertEqual(len(self._points()), 2)

        self.assertTrue(self.canvas.undo())
        self.assertEqual(self._points(), [])
        self.assertIsNone(self.canvas.get_function("f"))
