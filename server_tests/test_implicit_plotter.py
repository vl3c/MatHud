"""Pure Python tests for implicit curves and inequality regions.

Covers static/client/utils/implicit_relations.py (reading equations and inequalities,
every refusal and its hint), static/client/utils/implicit_plotter.py (marching squares on
the two-level grid: accuracy against exact curves, poles, undefined areas, saddles, small
loops, region rings and their orientation, the budget) and the polygon and dash helpers
of static/client/rendering/helpers/implicit_renderer.py. The Brython suite
(client_tests/test_implicit_curves.py) covers the managers, tools and rendering.
"""

from __future__ import annotations

import math
import unittest
from typing import Callable, Sequence, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
import compiled_expression
from rendering.helpers.implicit_renderer import bridged_polygon, dashed
from utils.implicit_plotter import PlotResult, plot, ring_area
from utils.implicit_relations import parse_equation, parse_inequality
from utils.implicit_summary import fractional_power_note, region_in_view

Point = Tuple[float, float]

RECT = (-12.0, -8.0, 12.0, 8.0)
CELL = 0.075  # 3 px at 40 px per unit


def _distance_to_zero(g: Callable[[float, float], float], x: float, y: float) -> float:
    """|g| over |grad g|: the distance to the zero set, to first order."""
    step = 1e-6
    gx = (g(x + step, y) - g(x - step, y)) / (2 * step)
    gy = (g(x, y + step) - g(x, y - step)) / (2 * step)
    return abs(g(x, y)) / max(1e-12, math.hypot(gx, gy))


def _winding(polygon: Sequence[Point], x: float, y: float) -> int:
    """The winding number of a closed polygon around (x, y)."""
    total = 0
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if y1 <= y < y2 and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
            total += 1
        elif y2 <= y < y1 and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
            total -= 1
    return total


def _curve(text: str, rect: Tuple[float, float, float, float] = RECT, cell: float = CELL, **options) -> PlotResult:
    equation = parse_equation(text)
    return plot(equation.g, rect, cell, factors=equation.factors, **options)


def _region(text: str, rect: Tuple[float, float, float, float] = RECT, cell: float = CELL) -> PlotResult:
    tree = parse_inequality(text)
    return plot(tree.value, rect, cell, fill=True, atom_of=tree.atom_of)


def _grid_rect(rect: Tuple[float, float, float, float], cell: float = CELL, block: int = 16) -> Tuple[float, ...]:
    """The block-aligned rectangle the plotter actually covers."""
    from utils.implicit_plotter import _layout

    i0, j0, phase, blocks_i, blocks_j = _layout(rect[0], rect[1], rect[2], rect[3], cell, block)
    x0, y0 = phase + i0 * cell, phase + j0 * cell
    return (x0, y0, x0 + blocks_i * block * cell, y0 + blocks_j * block * cell)


class TestEquationParsing(unittest.TestCase):
    def test_reads_standard_equations(self) -> None:
        cases = {
            "x^2 + y^2 = 25": "x^2 + y^2 = 25",
            "x^2/9 + y^2/4 = 1": "x^2/9 + y^2/4 = 1",
            "y^2 = x^3 - x": "y^2 = x^3 - x",
            "x^3 + y^3 = 3*x*y": "x^3 + y^3 = 3*x*y",
            "x = 3": "x = 3",
            "2x + 3y = 6": "2*x + 3*y = 6",
            "x² + y² = 1": "x^2 + y^2 = 1",
            "x == y^2": "x = y^2",
            "|x| + |y| = 1": "|x| + |y| = 1",
        }
        for text, stored in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_equation(text).text, stored)

    def test_g_is_left_minus_right(self) -> None:
        equation = parse_equation("x^2 + y^2 = 25")
        self.assertEqual(equation.g(3.0, 4.0), 0.0)
        self.assertEqual(equation.g(0.0, 0.0), -25.0)

    def test_refusals_say_what_to_write(self) -> None:
        cases = {
            "x^2 + y^2": "no '='",
            "y < x": "shade_inequality",
            "y = x = 2": "more than one",
            "xy = 1": "x\\*y",
            "sin x = y": "sin\\(x\\)",
            "sinx = y": "sin\\(x\\)",
            "xe^x = y": "x\\*e\\^x",
            "x = x": "holds for every x and y",
            "2 = 3": "no x or y",
            "a*x = y": "'a', which is not defined. Create the parameter first with set_parameter",
            "ax = y": "write a\\*x",
            "x2 + y2 = 25": "For a power write x\\^2 for x2, y\\^2 for y2",
            "y1 = x^2": "use draw_function",
            "y2 = 2*x + 1": "use draw_function",
            "x1 + x0 = 3": "write the relation in x and y",
            "x(x+1) = y": "function call",
            "(x+1)(y-1) = 1": "\\(x \\+ 1\\)\\*\\(y - 1\\)",
            "x^2 + y^2 = ((25)": "parentheses",
            "": "Give the equation",
            "y != 1": "inequality",
        }
        for text, pattern in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, pattern):
                    parse_equation(text)

    def test_factors_of_a_product_or_power_equal_to_zero(self) -> None:
        self.assertEqual(len(parse_equation("(x - y)*(x + y) = 0").factors), 2)
        self.assertEqual(len(parse_equation("0 = (x^2 + y^2 - 1)^2").factors), 1)
        self.assertEqual(len(parse_equation("x*y = 0").factors), 2)
        self.assertEqual(len(parse_equation("2*(x - y) = 0").factors), 1)  # the constant goes
        self.assertEqual(len(parse_equation("(x - y)*(x - y) = 0").factors), 1)  # repeats go
        squared = parse_equation("(x^2 + y^2 - 1)^2 = 0").factors[0]
        self.assertEqual(squared(2.0, 0.0), 3.0)  # the base, not the square
        self.assertEqual(len(parse_equation("(x - y)*(x + y) = 1").factors), 1)  # not = 0: g itself


class TestInequalityParsing(unittest.TestCase):
    def test_reads_standard_inequalities(self) -> None:
        cases = {
            "y < x^2": ("y < x^2", 1),
            "x^2 + y^2 <= 4": ("x^2 + y^2 <= 4", 1),
            "y ≥ |x|": ("y >= |x|", 1),
            "y ⩽ x^2": ("y <= x^2", 1),
            "((x > 0)) and y > 0": ("x > 0 and y > 0", 2),
            "x =< y": ("x <= y", 1),
            "y > x^2 - 2 and y < 1": ("y > x^2 - 2 and y < 1", 2),
            "x + y <= 10, x >= 0, y >= 0": ("x + y <= 10 and x >= 0 and y >= 0", 3),
            "y < -1 or y > 1": ("y < -1 or y > 1", 2),
            "-2 < x < 3": ("-2 < x < 3", 2),
            "x^2 < y <= 4": ("x^2 < y <= 4", 2),
            "(y > x and y < 2) or x < -1": ("y > x and y < 2 or x < -1", 3),
            "(y < x or y > 2) and x > 0": ("(y < x or y > 2) and x > 0", 3),
            "y < -1 or y > 1, x > 0": ("(y < -1 or y > 1) and x > 0", 3),
            "(x + 1)^2 < y": ("(x + 1)^2 < y", 1),
            "y < max(x, 1)": ("y < max(x, 1)", 1),
            "x > 2": ("x > 2", 1),
            "y AND x": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                if expected is None:
                    with self.assertRaises(ValueError):
                        parse_inequality(text)
                    continue
                tree = parse_inequality(text)
                self.assertEqual((tree.text, len(tree.atoms)), expected)

    def test_values_and_membership(self) -> None:
        tree = parse_inequality("y > x^2 - 2 and y < 1")
        self.assertTrue(tree.contains(0.0, 0.0))
        self.assertFalse(tree.contains(0.0, 2.0))
        self.assertFalse(tree.contains(3.0, 0.0))
        self.assertLess(tree.value(0.0, 0.0), 0.0)
        self.assertEqual(tree.atom_of(0.0, 0.9), 1)  # nearer y < 1 than y > x^2 - 2
        either = parse_inequality("y < -1 or y > 1")
        self.assertTrue(either.contains(0.0, 2.0))
        self.assertFalse(either.contains(0.0, 0.0))

    def test_boundary_points_belong_to_non_strict_inequalities(self) -> None:
        self.assertTrue(parse_inequality("x^2 + y^2 <= 4").contains(2.0, 0.0))
        self.assertFalse(parse_inequality("x^2 + y^2 < 4").contains(2.0, 0.0))

    def test_an_undefined_comparison_is_false(self) -> None:
        tree = parse_inequality("y < sqrt(x)")
        self.assertFalse(tree.contains(-1.0, -5.0))
        self.assertTrue(math.isnan(tree.value(-1.0, -5.0)))
        either = parse_inequality("y < sqrt(x) or y > 3")
        self.assertTrue(either.contains(-1.0, 4.0))
        self.assertFalse(either.contains(-1.0, 0.0))

    def test_refusals_say_what_to_write(self) -> None:
        cases = {
            "x = 1": "draw_implicit_curve",
            "y < x and x = 1": "draw_implicit_curve",
            "y != 1": "y < 1 or y > 1",
            "y ≠ 1": "y < 1 or y > 1",
            "not y < 1": "opposite comparison",
            "1 < 2": "no x or y",
            "y > x and": "empty part",
            "y": "no comparison",
            "sin x < y": "sin\\(x\\)",
            "xy < 1": "x\\*y",
            "y < a*x": "'a', which is not defined. Create the parameter first with set_parameter",
            "x1 + x2 <= 4, x1 >= 0, x2 >= 0": "write the relation in x and y",
            "(y < x": "parentheses",
            " ".join(["x > 0 and"] * 8) + " x > 0": "at most 8",
        }
        for text, pattern in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, pattern):
                    parse_inequality(text)


class TestParameters(unittest.TestCase):
    """Relations may read parameters (sliders); their values are read at call time."""

    def setUp(self) -> None:
        self._saved = compiled_expression.parameter_values()
        compiled_expression.set_parameter_values({"a": 2.0, "r": 3.0})

    def tearDown(self) -> None:
        compiled_expression.set_parameter_values(self._saved)

    def test_an_equation_follows_its_parameters(self) -> None:
        equation = parse_equation("x^2 + y^2 = r^2")
        self.assertEqual(equation.parameter_names(), ("r",))
        self.assertAlmostEqual(equation.g(3.0, 0.0), 0.0)
        compiled_expression.set_parameter_values({"a": 2.0, "r": 5.0})
        self.assertAlmostEqual(equation.g(5.0, 0.0), 0.0)
        result = plot(equation.g, (-8.0, -6.0, 8.0, 6.0), 0.05, factors=equation.factors)
        for x, y in result.curves[0]:
            self.assertAlmostEqual(math.hypot(x, y), 5.0, delta=1e-3)

    def test_factors_and_regions_read_parameters(self) -> None:
        equation = parse_equation("(x - a)*(y - a) = 0")
        self.assertEqual(len(equation.factors), 2)
        self.assertAlmostEqual(equation.factors[0](2.0, 7.0), 0.0)
        region = parse_inequality("y > a*x, x^2 + y^2 < r^2")
        self.assertEqual(region.parameter_names(), ("a", "r"))
        self.assertTrue(region.contains(0.0, 1.0))
        compiled_expression.set_parameter_values({"a": -2.0, "r": 0.5})
        self.assertFalse(region.contains(0.0, 1.0))

    def test_a_value_that_makes_it_hold_everywhere_is_accepted(self) -> None:
        compiled_expression.set_parameter_values({"a": 0.0})
        self.assertEqual(parse_equation("a*x*y = 0").parameter_names(), ("a",))

    def test_saved_relations_load_without_their_parameters(self) -> None:
        compiled_expression.set_parameter_values({})
        with self.assertRaisesRegex(ValueError, "set_parameter"):
            parse_equation("y = b*x^2")
        equation = parse_equation("y = b*x^2", check_names=False)
        self.assertEqual(equation.parameter_names(), ())
        self.assertTrue(math.isnan(_value_or_nan(equation.g, 1.0, 1.0)))
        parse_inequality("y < b", check_names=False)
        with self.assertRaisesRegex(ValueError, "set_parameter"):
            parse_inequality("y < b")


def _value_or_nan(g: Callable[[float, float], float], x: float, y: float) -> float:
    try:
        return float(g(x, y))
    except Exception:
        return float("nan")


class TestCurves(unittest.TestCase):
    def _assert_on_curve(self, text: str, result: PlotResult, tolerance: float = CELL / 4) -> None:
        g = parse_equation(text).g
        for path in result.curves:
            for x, y in path:
                self.assertLess(_distance_to_zero(g, x, y), tolerance, f"{text} at ({x}, {y})")

    def test_circle_is_one_closed_accurate_loop(self) -> None:
        result = _curve("x^2 + y^2 = 25")
        self.assertTrue(result.complete)
        self.assertEqual(len(result.curves), 1)
        loop = result.curves[0]
        self.assertEqual(loop[0], loop[-1])
        self.assertLess(max(abs(math.hypot(x, y) - 5.0) for x, y in loop), 1e-4)
        self.assertGreater(len(loop), 300)

    def test_common_curves_lie_on_their_equations(self) -> None:
        expected_pieces = {
            "x^2/9 + y^2/4 = 1": 1,
            "x^2 - y^2 = 1": 2,
            "y^2 = x^3 - x": 2,  # the oval and the open branch
            "x^3 + y^3 = 3*x*y": 1,  # the folium, through its node
            "x = 3": 1,
            "y = x": 1,
            "y = e^x": 1,
            "x^4 + y^4 = 1": 1,
            "x*y = 1": 2,
        }
        for text, pieces in expected_pieces.items():
            with self.subTest(text=text):
                result = _curve(text)
                self.assertTrue(result.complete)
                self.assertEqual(len(result.curves), pieces)
                self._assert_on_curve(text, result)

    def test_poles_are_not_drawn(self) -> None:
        result = _curve("y = 1/x")
        self.assertEqual(len(result.curves), 2)
        for path in result.curves:
            xs = [x for x, _ in path]
            self.assertTrue(min(xs) > 0 or max(xs) < 0, "a branch crossed the pole at x = 0")
        tangent = _curve("y = tan(x)")
        self._assert_on_curve("y = tan(x)", tangent, tolerance=CELL)
        for path in tangent.curves:
            ys = [y for _, y in path]
            self.assertTrue(all(abs(path[k][1] - path[k - 1][1]) < 1.0 for k in range(1, len(path))), ys[:3])

    def test_crossing_lines_at_a_saddle(self) -> None:
        result = _curve("y^2 = x^2")
        self.assertEqual(len(result.curves), 2)
        self._assert_on_curve("y^2 = x^2", result)

    def test_even_powers_are_drawn_through_their_factors(self) -> None:
        result = _curve("(x^2 + y^2 - 1)^2 = 0")
        self.assertEqual(len(result.curves), 1)
        self.assertLess(max(abs(math.hypot(x, y) - 1.0) for x, y in result.curves[0]), CELL / 100)
        both = _curve("(y - x)*(y + x - 1) = 0")
        self.assertEqual(len(both.curves), 2)

    def test_a_small_loop_inside_one_block_is_found(self) -> None:
        # Radius 0.2 = 8 px: no block corner is inside, the block centre is
        result = _curve("x^2 + y^2 = 0.04")
        self.assertEqual(len(result.curves), 1)

    def test_no_curve_without_real_points(self) -> None:
        self.assertEqual(_curve("x^2 + y^2 = -1").curves, [])
        self.assertEqual(_curve("x^2 + y^2 = 0").curves, [])  # one point: known gap

    def test_undefined_areas_are_skipped(self) -> None:
        result = _curve("y = sqrt(x)")
        self.assertEqual(len(result.curves), 1)
        self.assertTrue(all(x >= 0 for x, _ in result.curves[0]))
        self.assertLess(min(x for x, _ in result.curves[0]), 4 * CELL)

    def test_neighbouring_plots_agree_where_they_overlap(self) -> None:
        # Grids are anchored to the plane, so a pan gives the same points
        left = _curve("x^2 + y^2 = 25", rect=(-12.0, -8.0, 12.0, 8.0))
        right = _curve("x^2 + y^2 = 25", rect=(-6.0, -8.0, 18.0, 8.0))
        points = set(p for path in left.curves for p in path)
        self.assertTrue(set(p for path in right.curves for p in path) <= points)

    def test_budget_stops_the_search(self) -> None:
        result = _curve("sin(x*y) = 0.5", max_evals=3000)
        self.assertFalse(result.complete)
        self.assertLessEqual(result.evaluations, 3200)

    def test_dense_curves_are_plotted_coarser_rather_than_cut(self) -> None:
        result = _curve("sin(x*y) = 0.5", max_evals=30000)
        self.assertTrue(result.complete)
        self.assertGreater(result.cell, CELL)

    def test_deadline(self) -> None:
        ticks = iter(range(1000000))
        result = _curve("sin(x*y) = 0.5", clock=lambda: float(next(ticks)), deadline=50.0)
        self.assertFalse(result.complete)

    def test_bad_grid(self) -> None:
        with self.assertRaises(ValueError):
            plot(lambda x, y: x, (0.0, 0.0, 0.0, 1.0), 0.1)
        with self.assertRaises(ValueError):
            plot(lambda x, y: x, (0.0, 0.0, 1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            plot(lambda x, y: x, (0.0, 0.0, 1000.0, 1.0), 0.001)


class TestPolesAndSmallLoops(unittest.TestCase):
    """Cases the first PR review found: a pole within 1/64 of a cell from a node, and loops
    a few cells across between the coarse samples."""

    def _strokes_near_pole(self, curves: Sequence[Sequence[Tuple[float, float]]], pole: float, cell: float) -> int:
        return sum(1 for path in curves for x, y in path if abs(x - pole) < 0.5 * cell and abs(y) < 1 / (2 * cell))

    def test_asymptotes_near_a_node_are_not_joined(self) -> None:
        for scale in (20.0, 25.0, 40.0, 50.0, 80.0, 100.0):
            cell = 3.0 / scale
            rect = (-1200 / scale, -800 / scale, 1200 / scale, 800 / scale)
            for pole in range(-int(600 / scale) + 1, int(600 / scale)):
                with self.subTest(scale=scale, pole=pole):
                    equation = parse_equation(f"y = 1/(x - ({pole}))")
                    result = plot(equation.g, rect, cell, factors=equation.factors)
                    self.assertLessEqual(self._strokes_near_pole(result.curves, pole, result.cell), 2)

    def test_region_boundary_skips_the_asymptote(self) -> None:
        tree = parse_inequality("y < 1/(x - 1)")
        result = plot(tree.value, (-30.0, -20.0, 30.0, 20.0), 0.075, fill=True, atom_of=tree.atom_of)
        self.assertLessEqual(self._strokes_near_pole(result.curves, 1.0, result.cell), 2)

    def test_a_factor_is_drawn_only_where_the_product_is_defined(self) -> None:
        equation = parse_equation("x*sqrt(y) = 0")
        result = plot(equation.g, RECT, CELL, factors=equation.factors)
        points = [point for path in result.curves for point in path]
        self.assertTrue(points)
        self.assertGreater(min(y for _, y in points), -CELL)

    def test_small_loops_between_samples_are_found(self) -> None:
        rect = (-1200.0, -800.0, 1200.0, 800.0)  # the default view at 1 px per unit, with margins
        for cx, cy in ((10.0, 10.0), (-200.5, 31.0), (333.3, -250.0), (17.0, 230.0)):
            for radius in (3.0, 5.0, 8.0):
                with self.subTest(centre=(cx, cy), radius=radius):
                    result = plot(lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 - radius**2, rect, 3.0)
                    self.assertEqual(len(result.curves), 1)


class TestRegions(unittest.TestCase):
    def test_disc_area_and_orientation(self) -> None:
        result = _region("x^2 + y^2 < 4")
        self.assertEqual(len(result.rings), 1)
        self.assertAlmostEqual(ring_area(result.rings[0]), 4 * math.pi, delta=4 * math.pi * 1e-3)
        self.assertEqual(len(result.curves), 1)

    def test_annulus_has_a_clockwise_hole(self) -> None:
        result = _region("1 < x^2 + y^2 < 4")
        areas = sorted(ring_area(ring) for ring in result.rings)
        self.assertEqual(len(areas), 2)
        self.assertLess(areas[0], 0)
        self.assertAlmostEqual(sum(areas), 3 * math.pi, delta=0.02)
        self.assertEqual(sorted(result.curve_atoms), [0, 1])

    def test_regions_touching_the_border(self) -> None:
        x0, y0, x1, y1 = _grid_rect(RECT)
        cases = {
            # y < x^2: all of the grid but the part above the parabola
            "y < x^2": (x1 - x0) * (y1 - y0) - 4.0 / 3.0 * y1**1.5,
            "x > 2": (x1 - 2.0) * (y1 - y0),
            "y < -1 or y > 1": (x1 - x0) * (y1 - y0 - 2.0),
            "abs(x) + abs(y) <= 3": 18.0,
            "x + y <= 10, x >= 0, y >= 0": 50.0 - (10.0 - y1) ** 2 / 2.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = _region(text)
                self.assertTrue(result.complete)
                self.assertAlmostEqual(sum(ring_area(ring) for ring in result.rings), expected, delta=0.05)
                for ring in result.rings:
                    self.assertGreater(ring_area(ring), 0.0)

    def test_whole_view_and_empty_regions(self) -> None:
        x0, y0, x1, y1 = _grid_rect(RECT)
        whole = _region("y < 100")
        self.assertAlmostEqual(sum(ring_area(ring) for ring in whole.rings), (x1 - x0) * (y1 - y0), places=6)
        self.assertEqual(whole.curves, [])  # the border is not a boundary of the region
        self.assertEqual(_region("y > 100").rings, [])

    def test_undefined_area_is_outside_and_its_edge_is_not_stroked(self) -> None:
        x0, y0, x1, y1 = _grid_rect(RECT)
        result = _region("y < sqrt(x)")
        expected = 2.0 / 3.0 * x1**1.5 - y0 * x1
        self.assertAlmostEqual(sum(ring_area(ring) for ring in result.rings), expected, delta=0.1)
        for path in result.curves:
            for x, y in path:
                self.assertLess(_distance_to_zero(lambda u, v: v - math.sqrt(max(u, 0.0)), x, y), CELL)

    def test_pole_edges_bound_the_fill_but_are_not_stroked(self) -> None:
        result = _region("y < 1/x")
        self.assertEqual(len(result.curves), 2)
        for path in result.curves:
            xs = [x for x, _ in path]
            self.assertTrue(min(xs) > 0 or max(xs) < 0)

    def test_and_or_boundaries_belong_to_their_comparisons(self) -> None:
        tree = parse_inequality("x^2 + y^2 < 9 and y > x^2 - 4")
        result = plot(tree.value, RECT, CELL, fill=True, atom_of=tree.atom_of)
        self.assertEqual(sorted(set(result.curve_atoms)), [0, 1])
        for path, atom in zip(result.curves, result.curve_atoms):
            g = tree.atoms[atom].g
            for x, y in path[1:-1]:
                self.assertLess(_distance_to_zero(g, x, y), CELL / 2)


class TestRenderHelpers(unittest.TestCase):
    def test_bridged_polygon_keeps_holes_and_separate_parts(self) -> None:
        outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        hole = [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)]  # clockwise
        apart = [(20.0, 0.0), (25.0, 0.0), (25.0, 5.0), (20.0, 5.0)]
        polygon = bridged_polygon([outer, hole, apart])
        self.assertEqual(_winding(polygon, 1.0, 1.0), 1)
        self.assertEqual(_winding(polygon, 5.0, 5.0), 0)
        self.assertEqual(_winding(polygon, 22.0, 2.0), 1)
        self.assertEqual(_winding(polygon, 15.0, 2.0), 0)  # between the parts, on no bridge side
        self.assertEqual(_winding(polygon, 15.0, 0.0001), 0)

    def test_dashes(self) -> None:
        pieces = dashed([(0.0, 0.0), (20.0, 0.0)], on=6.0, off=4.0)
        self.assertEqual([[round(x, 6) for x, _ in piece] for piece in pieces], [[0.0, 6.0], [10.0, 16.0], [20.0]][:2])
        corner = dashed([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], on=6.0, off=4.0)
        self.assertEqual(corner[0], [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0)])


class TestFractionalPowerNote(unittest.TestCase):
    def test_fractions_and_decimals_in_exponents(self) -> None:
        for text in (
            "x^(2/3) + y^(2/3) = 1",
            "y = x^0.5",
            "(x - 1)^(1/3) = y",
            "y = x^(-1/2)",
            "y^(.5) <= x",
            "y = 2x^(1/3)",
            "x**(2/3) + y**(2/3) = 1",
            "(x^2 - y^2)^(1/2) = 1",
            "(x + y)^(1/3) = 1",
            "abs(x)^(2/3) + y^(1/3) = 1",
        ):
            self.assertIsNotNone(fractional_power_note(text), text)

    def test_a_power_over_a_number_is_not_a_fractional_power(self) -> None:
        for text in (
            "x^2/9 + y^2/4 = 1",
            "x^2/16 - y^2/9 = 1",
            "y = x^2/2",
            "4 + x^2/4 > y",
            "2^0.5*x = y",
            "x^2.0 = y",
            "abs(x)^(2/3) + abs(y)^(2/3) = 1",
            "(x^2 + y^2)^(1/2) = 3",
            "(x^2+y^2)^0.5 = 5",
            "(4*x^2 + 9*y^2 + 1)^(1/2) = 6",
        ):
            self.assertIsNone(fractional_power_note(text), text)


class TestRegionInView(unittest.TestCase):
    def test_a_region_smaller_than_the_samples_reports_its_drawn_area(self) -> None:
        text = "x + y <= 10, x >= 0, y >= 0"
        tree = parse_inequality(text)
        view = (-600.0, -480.0, 600.0, 480.0)
        drawn = plot(tree.value, view, 3.0, fill=True, atom_of=tree.atom_of)
        missed = region_in_view(text, tree.contains, view)
        self.assertEqual(missed["shaded_share"], 0.0)
        self.assertIn("No sample point", missed["note"])
        summary = region_in_view(text, tree.contains, view, drawn=drawn)
        self.assertGreater(summary["shaded_share"], 0.0)
        self.assertIn("small in this view", summary["note"])
        area = float(summary["note"].split("about ")[1].split(" ")[0])
        self.assertAlmostEqual(area, 50.0, delta=6.0)  # within a 3-unit cell along the edges

    def test_a_region_outside_the_view_still_says_none(self) -> None:
        text = "x > 1000"
        tree = parse_inequality(text)
        view = (-10.0, -10.0, 10.0, 10.0)
        drawn = plot(tree.value, (-20.0, -20.0, 20.0, 20.0), 0.1, fill=True, atom_of=tree.atom_of)
        summary = region_in_view(text, tree.contains, view, drawn=drawn)
        self.assertEqual(summary["shaded_share"], 0.0)
        self.assertIn("No sample point", summary["note"])


if __name__ == "__main__":
    unittest.main()
