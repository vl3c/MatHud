"""Pure Python tests for intersections of canvas objects (utils.object_intersections).

The Brython suite (static/client/client_tests/test_object_intersections.py) covers the
find_intersections tool on the canvas.
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Any, Dict, List, Tuple

from utils.function_features import scan_roots
from utils.object_intersections import (
    CircleShape,
    EllipseShape,
    FunctionShape,
    LineShape,
    ParametricShape,
    PointShape,
    circle_shape,
    ellipse_shape,
    find_object_intersections,
    line_shape,
)


def _xy(report: Dict[str, Any]) -> List[Tuple[float, float]]:
    return [(p["x"], p["y"]) for p in report["points"]]


class _Assertions(unittest.TestCase):
    def assertPoints(self, report: Dict[str, Any], expected: List[Tuple[float, float]], tol: float = 1e-9) -> None:
        actual = _xy(report)
        self.assertEqual(len(actual), len(expected), actual)
        for (x, y), (ex, ey) in zip(actual, expected):
            self.assertAlmostEqual(x, ex, delta=tol * max(1.0, abs(ex)))
            self.assertAlmostEqual(y, ey, delta=tol * max(1.0, abs(ey)))


class TestLines(_Assertions):
    def test_crossing_segments_with_parameters(self) -> None:
        report = find_object_intersections(line_shape("s1", (0, 0), (2, 2)), line_shape("s2", (0, 2), (2, 0)))
        self.assertEqual(report["points"], [{"x": 1.0, "y": 1.0, "params": {"s1": {"t": 0.5}, "s2": {"t": 0.5}}}])
        self.assertEqual(report["total_found"], 1)
        self.assertEqual(report["overlaps"], [])

    def test_segments_that_miss_meet_when_extended(self) -> None:
        a, b = ((0, 0), (1, 1)), ((3, 0), (4, -1))
        self.assertEqual(find_object_intersections(line_shape("s1", *a), line_shape("s2", *b))["points"], [])
        report = find_object_intersections(line_shape("s1", *a, bounded=False), line_shape("s2", *b, bounded=False))
        self.assertEqual(report["points"], [{"x": 1.5, "y": 1.5, "params": {"s1": {"t": 1.5}, "s2": {"t": -1.5}}}])

    def test_segment_end_on_the_other_segment(self) -> None:
        report = find_object_intersections(line_shape("s1", (0, 0), (2, 0)), line_shape("s2", (1, 0), (1, 5)))
        self.assertEqual(report["points"], [{"x": 1.0, "y": 0.0, "params": {"s1": {"t": 0.5}, "s2": {"t": 0.0}}}])

    def test_parallel_lines_have_a_note(self) -> None:
        report = find_object_intersections(line_shape("s1", (0, 0), (2, 0)), line_shape("s2", (0, 1), (3, 1)))
        self.assertEqual(report["points"], [])
        self.assertIn("parallel", report["notes"][0])

    def test_collinear_overlap_is_an_overlap_not_points(self) -> None:
        report = find_object_intersections(line_shape("s1", (0, 0), (2, 0)), line_shape("s2", (3, 0), (1, 0)))
        self.assertEqual(report["points"], [])
        self.assertEqual(report["overlaps"], [{"kind": "segment", "start": [1.0, 0.0], "end": [2.0, 0.0]}])

    def test_collinear_segments_touching_end_to_end_meet_in_one_point(self) -> None:
        report = find_object_intersections(line_shape("s1", (0, 0), (2, 0)), line_shape("s2", (2, 0), (3, 0)))
        self.assertEqual(report["points"], [{"x": 2.0, "y": 0.0, "params": {"s1": {"t": 1.0}, "s2": {"t": 0.0}}}])

    def test_collinear_apart_and_whole_lines(self) -> None:
        apart = find_object_intersections(line_shape("s1", (0, 0), (1, 1)), line_shape("s2", (2, 2), (3, 3)))
        self.assertEqual(apart["overlaps"], [])
        self.assertIn("one line", apart["notes"][0])
        lines = find_object_intersections(
            line_shape("s1", (0, 0), (1, 1), bounded=False), line_shape("s2", (2, 2), (3, 3), bounded=False)
        )
        self.assertEqual(lines["overlaps"], [{"kind": "line"}])

    def test_random_crossings_are_accurate(self) -> None:
        rng = random.Random(7)
        for _ in range(500):
            p = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(4)]
            a, b = line_shape("a", p[0], p[1], False), line_shape("b", p[2], p[3], False)
            assert isinstance(a, LineShape) and isinstance(b, LineShape)
            for point in find_object_intersections(a, b)["points"]:
                scale = max(1.0, abs(point["x"]), abs(point["y"]))
                self.assertLess(abs(a.implicit(point["x"], point["y"])), 1e-9 * scale)
                self.assertLess(abs(b.implicit(point["x"], point["y"])), 1e-9 * scale)


class TestCircles(_Assertions):
    def test_segment_through_a_circle(self) -> None:
        report = find_object_intersections(line_shape("s", (-5, 0), (5, 0)), circle_shape("c", (0, 0), 3))
        self.assertEqual(
            report["points"],
            [
                {"x": -3.0, "y": 0.0, "params": {"s": {"t": 0.2}, "c": {"angle": 3.141592654}}},
                {"x": 3.0, "y": 0.0, "params": {"s": {"t": 0.8}, "c": {"angle": 0.0}}},
            ],
        )

    def test_segment_inside_a_circle_meets_it_only_when_extended(self) -> None:
        inside = find_object_intersections(line_shape("s", (0, 0), (1, 0)), circle_shape("c", (0, 0), 3))
        self.assertEqual(inside["points"], [])
        extended = find_object_intersections(line_shape("s", (0, 0), (1, 0), False), circle_shape("c", (0, 0), 3))
        self.assertPoints(extended, [(-3.0, 0.0), (3.0, 0.0)])
        self.assertEqual([p["params"]["s"]["t"] for p in extended["points"]], [-3.0, 3.0])

    def test_tangent_line_touches_once(self) -> None:
        report = find_object_intersections(line_shape("s", (-5, 3), (5, 3)), circle_shape("c", (0, 0), 3))
        self.assertEqual(len(report["points"]), 1)
        self.assertEqual(report["points"][0]["x"], 0.0)
        self.assertEqual(report["points"][0]["y"], 3.0)
        self.assertTrue(report["points"][0]["tangent"])

    def test_line_just_outside_misses(self) -> None:
        report = find_object_intersections(line_shape("s", (-5, 3.000001), (5, 3.000001)), circle_shape("c", (0, 0), 3))
        self.assertEqual(report["points"], [])

    def test_two_circles(self) -> None:
        report = find_object_intersections(circle_shape("c1", (0, 0), 5), circle_shape("c2", (6, 0), 5))
        self.assertPoints(report, [(3.0, -4.0), (3.0, 4.0)])
        self.assertNotIn("tangent", report["points"][0])

    def test_external_and_internal_tangency(self) -> None:
        outside = find_object_intersections(circle_shape("c1", (0, 0), 2), circle_shape("c2", (5, 0), 3))
        self.assertEqual(outside["points"][0]["x"], 2.0)
        self.assertTrue(outside["points"][0]["tangent"])
        inside = find_object_intersections(circle_shape("c1", (0, 0), 5), circle_shape("c2", (2, 0), 3))
        self.assertEqual(_xy(inside), [(5.0, 0.0)])
        self.assertTrue(inside["points"][0]["tangent"])
        smaller_first = find_object_intersections(circle_shape("c1", (2, 0), 3), circle_shape("c2", (0, 0), 5))
        self.assertEqual(_xy(smaller_first), [(5.0, 0.0)])

    def test_apart_nested_and_concentric_circles(self) -> None:
        self.assertEqual(
            find_object_intersections(circle_shape("a", (0, 0), 1), circle_shape("b", (5, 0), 1))["points"], []
        )
        self.assertEqual(
            find_object_intersections(circle_shape("a", (0, 0), 5), circle_shape("b", (1, 0), 1))["points"], []
        )
        concentric = find_object_intersections(circle_shape("a", (0, 0), 1), circle_shape("b", (0, 0), 2))
        self.assertIn("concentric", concentric["notes"][0])

    def test_identical_circles_overlap(self) -> None:
        report = find_object_intersections(circle_shape("c1", (1, 2), 5), circle_shape("c2", (1, 2), 5))
        self.assertEqual(report["points"], [])
        self.assertEqual(report["overlaps"], [{"kind": "circle"}])

    def test_random_circle_pairs_are_accurate(self) -> None:
        rng = random.Random(3)
        for _ in range(500):
            a = circle_shape("a", (rng.uniform(-5, 5), rng.uniform(-5, 5)), rng.uniform(0.5, 5))
            b = circle_shape("b", (rng.uniform(-5, 5), rng.uniform(-5, 5)), rng.uniform(0.5, 5))
            line = line_shape(
                "s", (rng.uniform(-9, 9), rng.uniform(-9, 9)), (rng.uniform(-9, 9), rng.uniform(-9, 9)), False
            )
            assert isinstance(a, CircleShape) and isinstance(b, CircleShape)
            for point in find_object_intersections(a, b)["points"] + find_object_intersections(line, a)["points"]:
                # Rounded to 10 significant digits
                self.assertLess(abs(a.implicit(point["x"], point["y"])), 1e-9 * a.scale())


class TestArcs(_Assertions):
    def test_arc_keeps_only_points_on_its_sweep(self) -> None:
        upper = circle_shape("a", (0, 0), 2, arc_start=0.0, arc_sweep=math.pi)
        crossing = find_object_intersections(upper, line_shape("s", (-3, 1), (3, 1)))
        self.assertPoints(crossing, [(-math.sqrt(3), 1.0), (math.sqrt(3), 1.0)])
        self.assertAlmostEqual(crossing["points"][0]["params"]["a"]["angle"], 5 * math.pi / 6, places=9)
        self.assertEqual(find_object_intersections(upper, line_shape("s", (-3, -1), (3, -1)))["points"], [])

    def test_arc_against_a_circle(self) -> None:
        left_half = circle_shape("a", (0, 0), 2, arc_start=math.pi / 2, arc_sweep=math.pi)
        self.assertEqual(find_object_intersections(left_half, circle_shape("c", (2, 0), 2))["points"], [])
        right_half = circle_shape("a", (0, 0), 2, arc_start=-math.pi / 2, arc_sweep=math.pi)
        self.assertPoints(
            find_object_intersections(right_half, circle_shape("c", (2, 0), 2)),
            [(1.0, -math.sqrt(3)), (1.0, math.sqrt(3))],
        )

    def test_arc_end_counts(self) -> None:
        quarter = circle_shape("a", (0, 0), 1, arc_start=0.0, arc_sweep=math.pi / 2)
        report = find_object_intersections(quarter, line_shape("s", (-2, 1), (2, 1)))
        self.assertEqual(_xy(report), [(0.0, 1.0)])
        self.assertTrue(report["points"][0]["tangent"])

    def test_arcs_of_one_circle_overlap(self) -> None:
        report = find_object_intersections(
            circle_shape("a1", (0, 0), 1, 0.0, math.pi), circle_shape("a2", (0, 0), 1, math.pi / 2, math.pi)
        )
        self.assertEqual(report["overlaps"], [{"kind": "arc", "start": [0.0, 1.0], "end": [-1.0, 0.0]}])
        self.assertEqual(report["points"], [])

    def test_arcs_sharing_only_an_end_meet_there(self) -> None:
        report = find_object_intersections(
            circle_shape("a1", (0, 0), 1, 0.0, math.pi / 2), circle_shape("a2", (0, 0), 1, math.pi / 2, math.pi / 2)
        )
        self.assertEqual(_xy(report), [(0.0, 1.0)])
        self.assertEqual(report["overlaps"], [])

    def test_arcs_can_overlap_in_two_pieces(self) -> None:
        report = find_object_intersections(
            circle_shape("a1", (0, 0), 1, 0.0, 3.0), circle_shape("a2", (0, 0), 1, 2.0, 5.5)
        )
        self.assertEqual(len(report["overlaps"]), 2)

    def test_circle_and_its_arc_overlap_along_the_arc(self) -> None:
        report = find_object_intersections(circle_shape("c", (0, 0), 1), circle_shape("a", (0, 0), 1, 1.0, 1.0))
        self.assertEqual(len(report["overlaps"]), 1)
        self.assertEqual(report["overlaps"][0]["kind"], "arc")


class TestEllipses(_Assertions):
    def test_line_through_a_rotated_ellipse(self) -> None:
        ellipse = ellipse_shape("e", (0, 0), 4, 2, math.radians(30))
        report = find_object_intersections(line_shape("s", (-5, 0), (5, 0)), ellipse)
        # On y = 0: x^2 (cos^2/16 + sin^2/4) = 1
        x = 1.0 / math.sqrt(math.cos(math.radians(30)) ** 2 / 16 + math.sin(math.radians(30)) ** 2 / 4)
        self.assertPoints(report, [(-x, 0.0), (x, 0.0)])

    def test_line_tangent_to_a_rotated_ellipse(self) -> None:
        # Radii 2 (x) and 4 (y) turned a quarter: the top of the ellipse is (0, 2)
        report = find_object_intersections(
            line_shape("s", (-10, 2), (10, 2)), ellipse_shape("e", (0, 0), 2, 4, math.pi / 2)
        )
        self.assertEqual(_xy(report), [(0.0, 2.0)])
        self.assertTrue(report["points"][0]["tangent"])
        self.assertEqual(report["points"][0]["params"]["e"]["angle"], 0.0)

    def test_circle_and_ellipse_cross_four_times(self) -> None:
        report = find_object_intersections(circle_shape("c", (0, 0), 3), ellipse_shape("e", (0, 0), 4, 2, 0))
        # x^2 + y^2 = 9 and x^2/16 + y^2/4 = 1: x^2 = 20/3, y^2 = 7/3
        x, y = math.sqrt(20 / 3), math.sqrt(7 / 3)
        self.assertPoints(report, [(-x, -y), (-x, y), (x, -y), (x, y)])

    def test_circle_inscribed_in_an_ellipse_is_tangent_twice(self) -> None:
        report = find_object_intersections(circle_shape("c", (0, 0), 2), ellipse_shape("e", (0, 0), 4, 2, 0))
        self.assertPoints(report, [(0.0, -2.0), (0.0, 2.0)])
        self.assertTrue(all(p.get("tangent") for p in report["points"]))

    def test_two_ellipses_at_right_angles(self) -> None:
        report = find_object_intersections(
            ellipse_shape("e1", (0, 0), 4, 2, 0), ellipse_shape("e2", (0, 0), 4, 2, math.pi / 2)
        )
        v = 4 / math.sqrt(5)
        self.assertPoints(report, [(-v, -v), (-v, v), (v, -v), (v, v)])

    def test_ellipses_touching_externally(self) -> None:
        report = find_object_intersections(ellipse_shape("e1", (0, 0), 4, 2, 0), ellipse_shape("e2", (7, 0), 3, 1, 0))
        self.assertEqual(_xy(report), [(4.0, 0.0)])
        self.assertTrue(report["points"][0]["tangent"])

    def test_same_ellipse_described_two_ways_overlaps(self) -> None:
        report = find_object_intersections(
            ellipse_shape("e1", (0, 0), 4, 2, 0), ellipse_shape("e2", (0, 0), 2, 4, math.pi / 2)
        )
        self.assertEqual(report["overlaps"], [{"kind": "ellipse"}])
        turned = find_object_intersections(
            ellipse_shape("e1", (1, 1), 4, 2, 0.5), ellipse_shape("e2", (1, 1), 4, 2, 0.5 + math.pi)
        )
        self.assertEqual(turned["overlaps"], [{"kind": "ellipse"}])

    def test_random_rotated_ellipses_are_accurate(self) -> None:
        rng = random.Random(11)
        counts = set()
        for _ in range(150):
            e1 = ellipse_shape(
                "e1",
                (rng.uniform(-2, 2), rng.uniform(-2, 2)),
                rng.uniform(1, 5),
                rng.uniform(0.3, 3),
                rng.uniform(0, 3),
            )
            e2 = ellipse_shape(
                "e2",
                (rng.uniform(-2, 2), rng.uniform(-2, 2)),
                rng.uniform(1, 5),
                rng.uniform(0.3, 3),
                rng.uniform(0, 3),
            )
            assert isinstance(e1, EllipseShape) and isinstance(e2, EllipseShape)
            report = find_object_intersections(e1, e2)
            counts.add(len(report["points"]))
            self.assertIn(len(report["points"]), (0, 1, 2, 3, 4))
            for point in report["points"]:
                self.assertLess(abs(e1.implicit(point["x"], point["y"])), 1e-8)
                self.assertLess(abs(e2.implicit(point["x"], point["y"])), 1e-8)
        self.assertTrue({0, 2, 4} <= counts)


class TestCurves(_Assertions):
    def setUp(self) -> None:
        self.parabola = FunctionShape("f", lambda x: x * x, -10, 10)

    def test_function_and_segment(self) -> None:
        report = find_object_intersections(self.parabola, line_shape("s", (-3, 1), (3, 1)))
        self.assertPoints(report, [(-1.0, 1.0), (1.0, 1.0)])
        self.assertEqual(report["points"][0]["params"], {"s": {"t": 0.3333333333}})

    def test_function_tangent_to_a_segment(self) -> None:
        report = find_object_intersections(self.parabola, line_shape("s", (-3, 0), (3, 0)))
        self.assertEqual(_xy(report), [(0.0, 0.0)])
        self.assertTrue(report["points"][0]["tangent"])

    def test_tangency_off_the_sample_grid_is_located_precisely(self) -> None:
        shifted = FunctionShape("f", lambda x: (x - 0.123456789) ** 2 + 1, -10, 10)
        report = find_object_intersections(shifted, line_shape("s", (-5, 1), (5, 1)))
        self.assertPoints(report, [(0.123456789, 1.0)], tol=1e-9)
        self.assertTrue(report["points"][0]["tangent"])

    def test_crossing_at_an_inflection_point_is_a_tangency(self) -> None:
        cubic = FunctionShape("g", lambda x: x**3, -10, 10)
        report = find_object_intersections(cubic, line_shape("s", (-3, 0), (3, 0)))
        self.assertEqual(_xy(report), [(0.0, 0.0)])
        self.assertTrue(report["points"][0]["tangent"])

    def test_vertical_segment(self) -> None:
        report = find_object_intersections(self.parabola, line_shape("s", (2, -10), (2, 10)))
        self.assertEqual(_xy(report), [(2.0, 4.0)])

    def test_function_and_circle(self) -> None:
        report = find_object_intersections(self.parabola, circle_shape("c", (0, 0), 2))
        # x^2 + x^4 = 4
        y = (-1 + math.sqrt(17)) / 2
        self.assertPoints(report, [(-math.sqrt(y), y), (math.sqrt(y), y)])

    def test_sine_and_rotated_ellipse(self) -> None:
        ellipse = ellipse_shape("e", (0, 0), 3, 1, 0.2)
        assert isinstance(ellipse, EllipseShape)
        report = find_object_intersections(FunctionShape("f", math.sin, -10, 10), ellipse)
        self.assertEqual(len(report["points"]), 2)
        for point in report["points"]:
            self.assertLess(abs(ellipse.implicit(point["x"], point["y"])), 1e-9)
            self.assertAlmostEqual(point["y"], math.sin(point["x"]), places=9)

    def test_poles_are_not_intersections(self) -> None:
        tangent = FunctionShape(
            "t", math.tan, -5, 5, breakpoints=[-3 * math.pi / 2, -math.pi / 2, math.pi / 2, 3 * math.pi / 2]
        )
        report = find_object_intersections(tangent, line_shape("s", (-5, 1), (5, 1), False))
        self.assertPoints(report, [(-3 * math.pi / 4, 1.0), (math.pi / 4, 1.0), (5 * math.pi / 4, 1.0)])
        reciprocal = FunctionShape("r", lambda x: 1 / x, -5, 5, breakpoints=[0.0])
        self.assertEqual(_xy(find_object_intersections(reciprocal, line_shape("s", (-5, 0.5), (5, 0.5)))), [(2.0, 0.5)])

    def test_function_along_a_segment_is_an_overlap(self) -> None:
        report = find_object_intersections(
            FunctionShape("g", lambda x: 0.1 * x, -10, 10), line_shape("s", (0, 0), (5, 0.5))
        )
        self.assertEqual(report["points"], [])
        self.assertEqual(report["overlaps"], [{"kind": "curve", "start": [0.0, 0.0], "end": [5.0, 0.5]}])

    def test_partial_overlap_ends_where_the_function_bends_away(self) -> None:
        report = find_object_intersections(FunctionShape("f", abs, -5, 5), line_shape("s", (-2, -2), (3, 3)))
        self.assertEqual(report["overlaps"], [{"kind": "curve", "start": [0.0, 0.0], "end": [3.0, 3.0]}])
        self.assertEqual(report["points"], [])

    def test_upper_semicircle_function_overlaps_the_circle(self) -> None:
        report = find_object_intersections(
            FunctionShape("g", lambda x: math.sqrt(1 - x * x), -1, 1), circle_shape("c", (0, 0), 1)
        )
        self.assertEqual(report["overlaps"], [{"kind": "curve", "start": [-1.0, 0.0], "end": [1.0, 0.0]}])

    def test_function_outside_the_object_has_a_note(self) -> None:
        report = find_object_intersections(FunctionShape("f", lambda x: x, 10, 20), circle_shape("c", (0, 0), 2))
        self.assertEqual(report["points"], [])
        self.assertIn("not plotted", report["notes"][0])

    def test_two_functions(self) -> None:
        report = find_object_intersections(FunctionShape("g", lambda x: x + 2, -10, 10), self.parabola)
        self.assertEqual(report["points"], [{"x": -1.0, "y": 1.0}, {"x": 2.0, "y": 4.0}])

    def test_function_range_limits_the_search(self) -> None:
        report = find_object_intersections(FunctionShape("g", lambda x: x + 2, 0, 10), self.parabola)
        self.assertEqual(_xy(report), [(2.0, 4.0)])

    def test_parametric_curve_and_segment(self) -> None:
        unit = ParametricShape("p", math.cos, math.sin, 0, 2 * math.pi)
        report = find_object_intersections(unit, line_shape("s", (-2, 0.5), (2, 0.5)))
        self.assertPoints(report, [(-math.sqrt(3) / 2, 0.5), (math.sqrt(3) / 2, 0.5)])
        self.assertAlmostEqual(report["points"][0]["params"]["p"]["t"], 5 * math.pi / 6, places=9)

    def test_parametric_curve_tangent_to_a_circle(self) -> None:
        stretched = ParametricShape("p", lambda t: 2 * math.cos(t), math.sin, 0, 2 * math.pi)
        report = find_object_intersections(stretched, circle_shape("c", (0, 0), 1))
        self.assertPoints(report, [(0.0, -1.0), (0.0, 1.0)])
        self.assertTrue(all(p.get("tangent") for p in report["points"]))

    def test_parametric_curve_and_function(self) -> None:
        curve = ParametricShape("p", lambda t: t, lambda t: t * t, -3, 3)
        report = find_object_intersections(curve, FunctionShape("f", lambda x: 1.0, -10, 10))
        self.assertPoints(report, [(-1.0, 1.0), (1.0, 1.0)])

    def test_two_parametric_curves_are_not_supported(self) -> None:
        a = ParametricShape("p", math.cos, math.sin, 0, 1)
        b = ParametricShape("q", math.sin, math.cos, 0, 1)
        with self.assertRaises(ValueError):
            find_object_intersections(a, b)


class TestDegenerateAndReport(_Assertions):
    def test_zero_length_segment_is_a_point(self) -> None:
        point = line_shape("s", (1, 1), (1, 1))
        self.assertIsInstance(point, PointShape)
        report = find_object_intersections(point, circle_shape("c", (0, 0), math.sqrt(2)))
        self.assertEqual(_xy(report), [(1.0, 1.0)])
        self.assertIn("zero length", report["notes"][0])
        off = find_object_intersections(point, circle_shape("c", (0, 0), 1))
        self.assertEqual(off["points"], [])

    def test_zero_radius_circle_on_a_parametric_curve(self) -> None:
        dot = circle_shape("c0", (0.6, 0.8), 0)
        self.assertIsInstance(dot, PointShape)
        report = find_object_intersections(ParametricShape("p", math.cos, math.sin, 0, 2 * math.pi), dot)
        self.assertEqual(_xy(report), [(0.6, 0.8)])
        self.assertNotIn("tangent", report["points"][0])

    def test_ellipse_with_a_zero_radius_is_its_axis(self) -> None:
        axis = ellipse_shape("e", (0, 0), 3, 0, 0)
        self.assertIsInstance(axis, LineShape)
        report = find_object_intersections(axis, circle_shape("c", (0, 0), 2))
        self.assertEqual(_xy(report), [(-2.0, 0.0), (2.0, 0.0)])
        self.assertIn("zero radius", report["notes"][0])

    def test_same_name_twice_raises(self) -> None:
        with self.assertRaises(ValueError):
            find_object_intersections(circle_shape("c", (0, 0), 1), circle_shape("c", (1, 0), 1))

    def test_order_of_the_objects_does_not_change_the_points(self) -> None:
        circle, segment = circle_shape("c", (0, 0), 3), line_shape("s", (-5, 1), (5, 2))
        self.assertEqual(
            find_object_intersections(circle, segment)["points"], find_object_intersections(segment, circle)["points"]
        )

    def test_results_are_capped_and_flagged(self) -> None:
        wave = FunctionShape("f", lambda x: math.sin(10 * x), -20, 20)
        report = find_object_intersections(wave, line_shape("s", (-20, 0), (20, 0)), max_results=5)
        self.assertEqual(len(report["points"]), 5)
        self.assertTrue(report["truncated"])
        self.assertGreater(report["total_found"], 100)
        xs = [p["x"] for p in report["points"]]
        self.assertEqual(xs, sorted(xs))


class TestScanRoots(unittest.TestCase):
    def test_close_crossing_pair_between_samples_is_split(self) -> None:
        # Two roots 1e-4 apart, far closer than the 0.02 sample spacing
        roots = scan_roots(lambda x: (x - 0.3) * (x - 0.3001), -5, 5, samples=500)
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[0].x, 0.3, places=12)
        self.assertAlmostEqual(roots[1].x, 0.3001, places=12)
        self.assertFalse(roots[0].touching)

    def test_dip_below_zero_within_rounding_stays_one_touching_root(self) -> None:
        roots = scan_roots(lambda x: (x - 0.3) ** 2 - 1e-20, -5, 5, samples=500)
        self.assertEqual(len(roots), 1)
        self.assertTrue(roots[0].touching)

    def test_touching_and_zero_run(self) -> None:
        self.assertTrue(scan_roots(lambda x: (x - 1) ** 2, -3, 3)[0].touching)
        run = scan_roots(lambda x: 0.0 if 1 <= x <= 2 else x - 1 if x < 1 else x - 2, 0.5, 3, samples=250)
        self.assertEqual(len(run), 1)
        self.assertIsNotNone(run[0].end)


class TestToleranceAndOverlapEdgeCases(_Assertions):
    """Tolerances follow the objects' size, not their offset; overlaps of closed curves."""

    def test_small_circles_far_from_the_origin_are_not_one_circle(self) -> None:
        report = find_object_intersections(circle_shape("a", (1e6, 0), 1), circle_shape("b", (1e6 + 1e-5, 0), 1))
        self.assertEqual(report["overlaps"], [])
        self.assertEqual(len(report["points"]), 2)
        tiny = find_object_intersections(
            circle_shape("a", (1e6, 1e6), 1e-6), circle_shape("b", (1e6 + 1e-6, 1e6), 1e-6)
        )
        self.assertEqual(tiny["overlaps"], [])
        self.assertEqual(len(tiny["points"]), 2)

    def test_close_parallel_segments_far_from_the_origin_do_not_overlap(self) -> None:
        report = find_object_intersections(
            line_shape("s1", (1e6, 1e6), (1e6 + 1, 1e6)), line_shape("s2", (1e6, 1e6 + 1e-4), (1e6 + 1, 1e6 + 1e-4))
        )
        self.assertEqual(report["overlaps"], [])
        self.assertEqual(report["points"], [])
        self.assertIn("parallel", report["notes"][0])

    def test_identical_objects_far_from_the_origin_still_coincide(self) -> None:
        circles = find_object_intersections(circle_shape("a", (1e6, 1e6), 1e-3), circle_shape("b", (1e6, 1e6), 1e-3))
        self.assertEqual(circles["overlaps"], [{"kind": "circle"}])
        segments = find_object_intersections(
            line_shape("s1", (1e6, 1e6), (1e6 + 2, 1e6)), line_shape("s2", (1e6 + 1, 1e6), (1e6 + 3, 1e6))
        )
        self.assertEqual(len(segments["overlaps"]), 1)

    def test_nearly_tangent_line_on_a_large_circle_meets_it_twice(self) -> None:
        y = 1e6 - 1e-7
        report = find_object_intersections(line_shape("s", (-10, y), (10, y)), circle_shape("c", (0, 0), 1e6))
        # 1e6 - y is exact (Sterbenz), so this is the true half-chord of the line as stored
        half = math.sqrt((1e6 - y) * (1e6 + y))
        self.assertPoints(report, [(-half, y), (half, y)], tol=1e-6)
        self.assertFalse(any(p.get("tangent") for p in report["points"]))

    def test_exact_tangencies_far_from_the_origin_are_one_point(self) -> None:
        line = find_object_intersections(
            line_shape("s", (1e6 - 5, 1e6 + 3), (1e6 + 5, 1e6 + 3)), circle_shape("c", (1e6, 1e6), 3)
        )
        self.assertEqual(_xy(line), [(1e6, 1e6 + 3)])
        self.assertTrue(line["points"][0]["tangent"])
        circles = find_object_intersections(circle_shape("a", (1e6, 0), 2), circle_shape("b", (1e6 + 5, 0), 3))
        self.assertEqual(_xy(circles), [(1e6 + 2, 0.0)])
        self.assertTrue(circles["points"][0]["tangent"])

    def test_closed_parametric_curve_on_an_identical_circle_or_ellipse_overlaps(self) -> None:
        unit = ParametricShape("p", math.cos, math.sin, 0, 2 * math.pi)
        report = find_object_intersections(unit, circle_shape("c", (0, 0), 1))
        self.assertEqual(report["points"], [])
        self.assertEqual(report["overlaps"], [{"kind": "circle"}])
        stretched = ParametricShape("p", lambda t: 2 * math.cos(t), math.sin, 0, 2 * math.pi)
        ellipse = find_object_intersections(stretched, ellipse_shape("e", (0, 0), 2, 1, 0))
        self.assertEqual(ellipse["points"], [])
        self.assertEqual(ellipse["overlaps"], [{"kind": "ellipse"}])

    def test_closed_parametric_curve_on_an_arc_gives_one_overlap_and_no_stray_point(self) -> None:
        unit = ParametricShape("p", math.cos, math.sin, 0, 2 * math.pi)
        report = find_object_intersections(unit, circle_shape("a", (0, 0), 1, arc_start=0.0, arc_sweep=1.0))
        self.assertEqual(report["points"], [])
        self.assertEqual(
            report["overlaps"], [{"kind": "curve", "start": [1.0, 0.0], "end": [0.5403023059, 0.8414709848]}]
        )

    def test_arc_across_the_seam_of_a_closed_curve_is_one_overlap(self) -> None:
        unit = ParametricShape("p", math.cos, math.sin, 0, 2 * math.pi)
        report = find_object_intersections(unit, circle_shape("a", (0, 0), 1, arc_start=-0.5, arc_sweep=1.0))
        self.assertEqual(report["points"], [])
        self.assertEqual(len(report["overlaps"]), 1)
        overlap = report["overlaps"][0]
        self.assertAlmostEqual(overlap["start"][0], math.cos(0.5), places=9)
        self.assertAlmostEqual(overlap["start"][1], -math.sin(0.5), places=9)
        self.assertAlmostEqual(overlap["end"][1], math.sin(0.5), places=9)

    def test_flattened_ellipse_reports_its_angle(self) -> None:
        report = find_object_intersections(ellipse_shape("e", (0, 0), 3, 0, 0), circle_shape("c", (0, 0), 2))
        angles = [p["params"]["e"] for p in report["points"]]
        self.assertEqual([sorted(a) for a in angles], [["angle"], ["angle"]])
        # x = 3 cos(angle): -2 and 2
        self.assertAlmostEqual(angles[0]["angle"], math.acos(-2 / 3), places=9)
        self.assertAlmostEqual(angles[1]["angle"], math.acos(2 / 3), places=9)
        upright = find_object_intersections(ellipse_shape("e", (0, 0), 0, 3, 0), line_shape("s", (-1, 1.5), (1, 1.5)))
        self.assertAlmostEqual(upright["points"][0]["params"]["e"]["angle"], math.asin(0.5), places=9)

    def test_nearly_identical_ellipses_touch_instead_of_overlapping(self) -> None:
        report = find_object_intersections(
            ellipse_shape("e1", (0, 0), 4, 2, 0), ellipse_shape("e2", (0, 0), 4 + 1e-9, 2, 0)
        )
        self.assertEqual(report["overlaps"], [])
        self.assertPoints(report, [(0.0, -2.0), (0.0, 2.0)], tol=1e-6)
        self.assertTrue(all(p.get("tangent") for p in report["points"]))


class TestTangenciesFarFromTheOrigin(_Assertions):
    """Tangent objects built from rounded coordinates (as draw_tangent_line builds them) stay tangent."""

    def _assert_one_tangent_point(self, report: Dict[str, Any], label: str) -> None:
        self.assertEqual(len(report["points"]), 1, f"{label}: {report['points']}")
        self.assertTrue(report["points"][0].get("tangent"), label)

    def test_tangent_lines_to_circles(self) -> None:
        rng = random.Random(21)
        for offset in (500.0, 1000.0):
            for _ in range(150):
                cx, cy = offset * rng.choice((-1, 1)), offset * rng.uniform(-1, 1)
                r, theta = rng.uniform(0.3, 5.0), rng.uniform(0, 2 * math.pi)
                px, py = cx + r * math.cos(theta), cy + r * math.sin(theta)
                tx, ty = -math.sin(theta), math.cos(theta)
                line = line_shape("s", (px - 2 * tx, py - 2 * ty), (px + 2 * tx, py + 2 * ty))
                self._assert_one_tangent_point(find_object_intersections(line, circle_shape("c", (cx, cy), r)), "line")

    def test_tangent_lines_to_rotated_ellipses(self) -> None:
        rng = random.Random(22)
        for offset in (500.0, 1000.0):
            for _ in range(150):
                center = (offset * rng.choice((-1, 1)), offset * rng.uniform(-1, 1))
                ellipse = ellipse_shape("e", center, rng.uniform(1.0, 5.0), rng.uniform(0.3, 3.0), rng.uniform(0, 3))
                assert isinstance(ellipse, EllipseShape)
                theta = rng.uniform(0, 2 * math.pi)
                (px, py), (dx, dy) = ellipse.point_at_angle(theta), ellipse.derivative_at_angle(theta)
                norm = math.hypot(dx, dy)
                tx, ty = dx / norm, dy / norm
                line = line_shape("s", (px - 2 * tx, py - 2 * ty), (px + 2 * tx, py + 2 * ty))
                self._assert_one_tangent_point(find_object_intersections(line, ellipse), "ellipse")

    def test_touching_circles(self) -> None:
        rng = random.Random(23)
        for offset in (500.0, 1000.0):
            for _ in range(150):
                cx, cy = offset * rng.choice((-1, 1)), offset * rng.uniform(-1, 1)
                r1, r2, theta = rng.uniform(0.3, 5.0), rng.uniform(0.3, 5.0), rng.uniform(0, 2 * math.pi)
                outside = (cx + (r1 + r2) * math.cos(theta), cy + (r1 + r2) * math.sin(theta))
                big, small = max(r1, r2), min(r1, r2)
                inside = (cx + (big - small) * math.cos(theta), cy + (big - small) * math.sin(theta))
                first = circle_shape("a", (cx, cy), r1)
                self._assert_one_tangent_point(
                    find_object_intersections(first, circle_shape("b", outside, r2)), "outer"
                )
                self._assert_one_tangent_point(
                    find_object_intersections(circle_shape("a", (cx, cy), big), circle_shape("b", inside, small)),
                    "inner",
                )

    def test_circle_touching_an_ellipse(self) -> None:
        rng = random.Random(24)
        for offset in (500.0, 1000.0):
            for _ in range(60):
                center = (offset * rng.choice((-1, 1)), offset * rng.uniform(-1, 1))
                ellipse = ellipse_shape("e", center, rng.uniform(2.0, 5.0), rng.uniform(1.0, 2.0), rng.uniform(0, 3))
                assert isinstance(ellipse, EllipseShape)
                theta = rng.uniform(0, 2 * math.pi)
                (px, py), (dx, dy) = ellipse.point_at_angle(theta), ellipse.derivative_at_angle(theta)
                norm = math.hypot(dx, dy)
                r = rng.uniform(0.1, 0.5)
                # Outward normal of a counter-clockwise ellipse
                circle = circle_shape("c", (px + r * dy / norm, py - r * dx / norm), r)
                self._assert_one_tangent_point(find_object_intersections(circle, ellipse), "circle/ellipse")

    def test_far_points_print_only_the_digits_the_inputs_resolve(self) -> None:
        report = find_object_intersections(circle_shape("a", (1e6, 0), 1), circle_shape("b", (1e6 + 1e-5, 0), 1))
        self.assertEqual(_xy(report), [(1000000.000005, -1.0), (1000000.000005, 1.0)])
        tiny = find_object_intersections(
            circle_shape("a", (1e6, 1e6), 1e-6), circle_shape("b", (1e6 + 1e-6, 1e6), 1e-6)
        )
        self.assertEqual(_xy(tiny), [(1000000.0000005, 999999.999999134), (1000000.0000005, 1000000.000000866)])
        # Near the origin the 10 significant digits are unchanged
        near = find_object_intersections(line_shape("s", (-3, 1), (3, 1)), circle_shape("c", (0, 0), 2))
        self.assertEqual(_xy(near), [(-1.732050808, 1.0), (1.732050808, 1.0)])


if __name__ == "__main__":
    unittest.main()
