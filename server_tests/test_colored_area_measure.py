"""Measuring coloured areas (``calculate_area`` on a coloured area's name), with stand-in drawables."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple

from utils.colored_area_measure import measure_colored_area

NORMAL_PDF_WITHIN_ONE_SIGMA = math.erf(1 / math.sqrt(2))  # 0.6826894921370859


def function(
    name: str,
    fn: Callable[[float], float],
    left: float = -10.0,
    right: float = 10.0,
    asymptotes: Optional[list[float]] = None,
) -> Any:
    return SimpleNamespace(name=name, function=fn, left_bound=left, right_bound=right, vertical_asymptotes=asymptotes)


def abs_sin_integral(k: float, b: float) -> float:
    """Integral of |sin(k x)| over [0, b]."""
    u = k * b
    n = math.floor(u / math.pi)
    return (2 * n + (1 - math.cos(u - n * math.pi))) / k


def point(x: float, y: float) -> Any:
    return SimpleNamespace(x=x, y=y)


def segment(name: str, a: Tuple[float, float], b: Tuple[float, float]) -> Any:
    return SimpleNamespace(name=name, point1=point(*a), point2=point(*b))


class FunctionsArea:
    """Stand-in for FunctionsBoundedColoredArea: two bounds (a function, a constant or None) and an interval."""

    def __init__(self, func1: Any, func2: Any, left: Optional[float], right: Optional[float]) -> None:
        self.func1, self.func2, self.left, self.right = func1, func2, left, right

    def get_class_name(self) -> str:
        return "FunctionsBoundedColoredArea"

    def _get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return self.left, self.right

    def _get_function_y_at_x(self, func: Any, x: float) -> Optional[float]:
        if func is None:
            return 0.0
        if isinstance(func, (int, float)):
            return float(func)
        try:
            y = func.function(x)
        except (ValueError, ZeroDivisionError):
            return None
        return None if not math.isfinite(y) else float(y)


class TestFunctionBoundedAreas(unittest.TestCase):
    def test_normal_pdf_within_one_sigma(self) -> None:
        pdf = function("normal", lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi), -4, 4)
        measure = measure_colored_area(FunctionsArea(pdf, None, -1, 1))
        self.assertAlmostEqual(measure["value"], NORMAL_PDF_WITHIN_ONE_SIGMA, places=10)
        self.assertAlmostEqual(round(measure["value"], 6), 0.682689)
        self.assertLess(measure["error_estimate"], 1e-9)
        self.assertEqual(measure["bounds"], [-1.0, 1.0])
        self.assertIn("|normal - the x-axis|", measure["method"])
        self.assertIn("Simpson", measure["method"])
        self.assertNotIn("crossings", measure)

    def test_crossing_bounds_are_split(self) -> None:
        sin, cos = function("f", math.sin), function("g", math.cos)
        measure = measure_colored_area(FunctionsArea(sin, cos, 0, math.pi))
        self.assertAlmostEqual(measure["value"], 2 * math.sqrt(2), places=8)
        # The reported error estimate is honest: the actual error is within twice of it.
        self.assertLessEqual(abs(measure["value"] - 2 * math.sqrt(2)), 2 * measure["error_estimate"])
        self.assertEqual(len(measure["crossings"]), 1)
        self.assertAlmostEqual(measure["crossings"][0], math.pi / 4, places=12)
        self.assertIn("split at 1 crossing(s)", measure["method"])

    def test_area_below_the_axis_counts_positive(self) -> None:
        cubic = function("c", lambda x: x**3)
        self.assertAlmostEqual(measure_colored_area(FunctionsArea(cubic, None, -1, 1))["value"], 0.5, places=10)

    def test_a_constant_bound(self) -> None:
        square = function("q", lambda x: x * x)
        measure = measure_colored_area(FunctionsArea(square, 4, -2, 2))
        self.assertAlmostEqual(measure["value"], 32 / 3, places=10)
        self.assertIn("y = 4", measure["method"])

    def test_an_asymptote_inside_the_interval_is_an_error(self) -> None:
        reciprocal = function("f", lambda x: 1 / (x - 1))
        with self.assertRaisesRegex(ValueError, "diverges .*near x ≈ 1.*f is undefined or not finite at x = 1"):
            measure_colored_area(FunctionsArea(reciprocal, None, 0, 2))
        # Away from it the area is the log: ln(3) on [2, 4] for 1/(x - 1).
        self.assertAlmostEqual(measure_colored_area(FunctionsArea(reciprocal, None, 2, 4))["value"], math.log(3), 8)

    def test_a_listed_asymptote_is_refused_before_sampling(self) -> None:
        tangent = function("t", math.tan, 0, 3, asymptotes=[math.pi / 2])
        with self.assertRaisesRegex(ValueError, "diverges near x ≈ 1.5708: t has a vertical asymptote"):
            measure_colored_area(FunctionsArea(tangent, None, 0, 3))
        # An asymptote outside the interval does not matter.
        self.assertAlmostEqual(
            measure_colored_area(FunctionsArea(tangent, None, 0, 1))["value"], -math.log(math.cos(1)), 8
        )

    def test_poles_off_the_grid_are_found_numerically(self) -> None:
        # No listed asymptotes: tan changes sign through its pole at pi/2, which no grid point hits.
        tangent = function("t", math.tan, 0, 3)
        with self.assertRaisesRegex(ValueError, "diverges near x ≈ 1.5708"):
            measure_colored_area(FunctionsArea(tangent, None, 0, 3))
        # A double pole keeps its sign: a sharp peak whose refined maximum grows without bound.
        double = function("d", lambda x: 1 / (x - 0.5001) ** 2, 0, 1)
        with self.assertRaisesRegex(ValueError, "diverges near x ≈ 0.5001"):
            measure_colored_area(FunctionsArea(double, None, 0, 1))
        midway = function("m", lambda x: 1 / (x - (0.5 + 1 / 1024)) ** 2, 0, 1)
        with self.assertRaisesRegex(ValueError, "diverges near x ≈ 0.500977"):
            measure_colored_area(FunctionsArea(midway, None, 0, 1))
        simple = function("s", lambda x: 1 / (x - 0.30007), 0, 1)
        with self.assertRaisesRegex(ValueError, "diverges near x ≈ 0.30007"):
            measure_colored_area(FunctionsArea(simple, None, 0, 1))

    def test_a_sharp_but_finite_peak_is_measured(self) -> None:
        # A narrow bump (Lorentzian of half-width 1e-3): finite, area pi * 1e-3 * 1e3 = pi.
        bump = function("b", lambda x: 1.0 / ((x - 0.4) ** 2 + 1e-6) * 1e-3, 0, 1)
        measure = measure_colored_area(FunctionsArea(bump, None, 0, 1))
        exact = math.atan(1000 * 0.6) + math.atan(1000 * 0.4)
        self.assertAlmostEqual(measure["value"], exact, places=4)

    def test_many_crossings_are_each_located(self) -> None:
        for k, b in ((30, 10), (50, 10), (1, 1000)):
            wave = function("w", lambda x, k=k: math.sin(k * x), -1e9, 1e9)
            measure = measure_colored_area(FunctionsArea(wave, None, 0, b))
            exact = abs_sin_integral(k, b)
            self.assertAlmostEqual(measure["value"], exact, delta=1e-6 * exact, msg=f"sin({k}x) on [0, {b}]")
            # The estimate is honest: the actual error is within it (with a margin for its own error).
            self.assertLessEqual(abs(measure["value"] - exact), 2 * measure["error_estimate"] + 1e-12)
            self.assertEqual(measure["crossing_count"], math.floor(k * b / math.pi))
        # sin(50x) on [0, 10]: about 6.3623 (the mean of |sin| times 10 gives 6.3662).
        self.assertAlmostEqual(abs_sin_integral(50, 10), 6.3623, places=4)

    def test_very_many_crossings_use_the_cell_rule_with_a_warning(self) -> None:
        wave = function("w", lambda x: math.sin(x), -1e9, 1e9)
        measure = measure_colored_area(FunctionsArea(wave, None, 0, 4000))
        exact = abs_sin_integral(1, 4000)
        self.assertGreater(measure["crossing_count"], 1024)
        self.assertIn("piecewise linear", measure["method"])
        self.assertIn("accuracy is limited", measure["warning"])
        self.assertLessEqual(abs(measure["value"] - exact), 2 * measure["error_estimate"])
        self.assertAlmostEqual(measure["value"], exact, delta=2e-3 * exact)

    def test_an_interval_that_is_missing_or_empty_is_an_error(self) -> None:
        f = function("f", math.sin)
        with self.assertRaisesRegex(ValueError, "no finite x-interval"):
            measure_colored_area(FunctionsArea(f, None, None, 1))
        with self.assertRaisesRegex(ValueError, "is empty"):
            measure_colored_area(FunctionsArea(f, None, 1, 1))


class TestSegmentBoundedAreas(unittest.TestCase):
    def test_function_and_segment(self) -> None:
        area = SimpleNamespace(
            get_class_name=lambda: "FunctionSegmentBoundedColoredArea",
            func=function("f", lambda x: x * x),
            segment=segment("AB", (0, 1), (2, 1)),
            _get_bounds=lambda: (0.0, 2.0),
        )
        area._get_function_y_at_x = lambda x: x * x
        # |x^2 - 1| on [0, 2]: 2/3 + 4/3.
        measure = measure_colored_area(area)
        self.assertAlmostEqual(measure["value"], 2.0, places=10)
        self.assertAlmostEqual(measure["crossings"][0], 1.0, places=12)

    def test_two_segments_over_their_overlap(self) -> None:
        area = SimpleNamespace(
            get_class_name=lambda: "SegmentsBoundedColoredArea",
            segment1=segment("AB", (0, 2), (4, 2)),
            segment2=segment("CD", (1, 0), (3, 0)),
        )
        measure = measure_colored_area(area)
        self.assertAlmostEqual(measure["value"], 4.0, places=12)
        self.assertEqual(measure["bounds"], [1.0, 3.0])

    def test_segment_and_the_x_axis_is_exact(self) -> None:
        area = SimpleNamespace(
            get_class_name=lambda: "SegmentsBoundedColoredArea",
            segment1=segment("AB", (-1, -1), (3, 3)),
            segment2=None,
        )
        # Two triangles: 1/2 below the axis on [-1, 0], 9/2 above it on [0, 3].
        self.assertAlmostEqual(measure_colored_area(area)["value"], 5.0, places=12)

    def test_segments_that_do_not_overlap(self) -> None:
        area = SimpleNamespace(
            get_class_name=lambda: "SegmentsBoundedColoredArea",
            segment1=segment("AB", (0, 0), (1, 0)),
            segment2=segment("CD", (2, 1), (3, 1)),
        )
        with self.assertRaisesRegex(ValueError, "is empty"):
            measure_colored_area(area)


class TestClosedShapes(unittest.TestCase):
    def shape(self, **attrs: Any) -> Any:
        return SimpleNamespace(get_class_name=lambda: "ClosedShapeColoredArea", **attrs)

    def test_exact_formulas(self) -> None:
        circle = self.shape(shape_type="circle", circle=SimpleNamespace(radius=2.0))
        self.assertAlmostEqual(measure_colored_area(circle)["value"], 4 * math.pi)
        ellipse = self.shape(shape_type="ellipse", ellipse=SimpleNamespace(radius_x=3.0, radius_y=2.0))
        measure = measure_colored_area(ellipse)
        self.assertAlmostEqual(measure["value"], 6 * math.pi)
        self.assertEqual(measure["error_estimate"], 0.0)
        self.assertTrue(measure["method"].startswith("exact"))

    def test_region_from_its_expression(self) -> None:
        square = [(math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)) for t in range(4)]
        region = self.shape(shape_type="region", expression="C(1)", points=square)
        measure = measure_colored_area(region, lambda expression: math.pi)
        self.assertEqual(measure["value"], math.pi)
        self.assertIn("region expression 'C(1)'", measure["method"])
        # The drawn outline (here a square inscribed in the circle) gives the estimate.
        self.assertAlmostEqual(measure["error_estimate"], math.pi - 2.0)

    def test_region_outline_alone_is_not_called_exact(self) -> None:
        region = self.shape(shape_type="region", points=[(0, 0), (4, 0), (4, 3), (0, 3)])
        measure = measure_colored_area(region)
        self.assertEqual(measure["value"], 12.0)
        self.assertFalse(measure["method"].startswith("exact"))
        self.assertIsNone(measure["error_estimate"])
        self.assertIn("approximation", measure["warning"])

    def test_polygon_from_segments(self) -> None:
        a, b, c = point(0, 0), point(4, 0), point(0, 3)
        sides = [SimpleNamespace(point1=a, point2=b), SimpleNamespace(point1=b, point2=c),
                 SimpleNamespace(point1=c, point2=a)]  # fmt: skip
        polygon = self.shape(shape_type="polygon", segments=sides)
        self.assertEqual(measure_colored_area(polygon)["value"], 6.0)

    def test_circle_segment_points_to_the_region_expression(self) -> None:
        chord = self.shape(shape_type="circle_segment", circle=SimpleNamespace(radius=1.0))
        with self.assertRaisesRegex(ValueError, "C\\(5\\) & AB"):
            measure_colored_area(chord)

    def test_unknown_kinds(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot measure"):
            measure_colored_area(SimpleNamespace(get_class_name=lambda: "Plot"))


if __name__ == "__main__":
    unittest.main()
