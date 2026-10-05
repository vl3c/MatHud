"""Pure Python tests for numeric roots, extrema and intersections (utils.function_features).

The Brython suite (static/client/client_tests/test_function_features.py) covers the same
module in the browser and the find_function_features tool on the canvas.
"""

from __future__ import annotations

import math
import unittest

from utils.function_features import find_function_features, find_intersections


def _summary(report: dict) -> list:
    return [(f["x"], f["y"], f["kind"]) for f in report["features"]]


class TestFunctionFeaturesPure(unittest.TestCase):
    def test_polynomial_roots_and_extrema(self) -> None:
        report = find_function_features(lambda x: x**3 - 3 * x, -3, 3)
        self.assertEqual(
            _summary(report),
            [
                (-1.732050808, 0.0, "root"),
                (-1.0, 2.0, "local_max"),
                (0.0, 0.0, "root"),
                (1.0, -2.0, "local_min"),
                (1.732050808, 0.0, "root"),
            ],
        )

    def test_double_root_is_touching_root_and_minimum(self) -> None:
        report = find_function_features(lambda x: (x - 1) ** 2, -5, 4.3)
        self.assertEqual(_summary(report), [(1.0, 0.0, "root"), (1.0, 0.0, "local_min")])
        self.assertTrue(report["features"][0]["touching"])

    def test_poles_give_no_false_roots(self) -> None:
        self.assertEqual(find_function_features(lambda x: 1 / x, -5, 5)["features"], [])
        self.assertEqual(find_function_features(lambda x: 1 / x**2, -5, 5)["features"], [])
        tangent = find_function_features(math.tan, -5, 5)
        self.assertEqual([f["x"] for f in tangent["features"]], [-3.141592654, 0.0, 3.141592654])

    def test_breakpoints_split_the_interval(self) -> None:
        report = find_function_features(lambda x: 1 / (x - 1) + 1, -5, 5, breakpoints=[1.0])
        self.assertEqual(_summary(report), [(0.0, 0.0, "root")])

    def test_empty_result(self) -> None:
        report = find_function_features(lambda x: x * x + 1, -5, 5, features=["roots"])
        self.assertEqual(report["features"], [])
        self.assertFalse(report["truncated"])

    def test_truncation(self) -> None:
        report = find_function_features(math.sin, 0.5, 100.5, features=["roots"], max_results=5)
        self.assertEqual(len(report["features"]), 5)
        self.assertEqual(report["total_found"], 31)
        self.assertTrue(report["truncated"])

    def test_intersections(self) -> None:
        report = find_intersections(lambda x: x * x, lambda x: x + 2, -5, 5)
        self.assertEqual(_summary(report), [(-1.0, 1.0, "intersection"), (2.0, 4.0, "intersection")])

    def test_coinciding_functions_report_a_zero_interval(self) -> None:
        report = find_intersections(lambda x: 2 * x, lambda x: x + x, -1, 1)
        self.assertEqual(len(report["features"]), 1)
        self.assertEqual(report["features"][0]["zero_interval"], [-1.0, 1.0])

    # ---- roots at the interval ends that rounding moved off zero ----

    def test_endpoint_roots_rounded_off_zero_are_found(self) -> None:
        # sin(2*pi) = -2.4e-16, cos(pi/2) = 6.1e-17, sqrt(2)^2 - 2 = 4.4e-16: no sign change brackets them
        sine = find_function_features(math.sin, 0, 2 * math.pi, features=["roots"])
        self.assertEqual([f["x"] for f in sine["features"]], [0.0, 3.141592654, 6.283185307])
        cosine = find_function_features(math.cos, 0, math.pi / 2, features=["roots"])
        self.assertEqual([f["x"] for f in cosine["features"]], [1.570796327])
        square = find_function_features(lambda x: x * x - 2, -5, math.sqrt(2), features=["roots"])
        self.assertEqual([f["x"] for f in square["features"]], [-1.414213562, 1.414213562])

    def test_endpoint_root_is_not_duplicated_by_a_nearby_sign_change(self) -> None:
        # cos just past pi/2 is -9e-16: the last bracket changes sign and the end is near zero too
        report = find_function_features(math.cos, 0, math.pi / 2 + 1e-15, features=["roots"])
        self.assertEqual(len(report["features"]), 1)
        self.assertAlmostEqual(report["features"][0]["x"], math.pi / 2, places=9)

    def test_tiny_endpoint_value_that_is_not_a_root_is_ignored(self) -> None:
        # exp(-40) is 4e-18, far below 1e-12 of exp(0), but exp does not vanish there
        self.assertEqual(find_function_features(math.exp, -40, 0, features=["roots"])["features"], [])
        self.assertEqual(find_function_features(lambda x: x * x + 1, -5, 5, features=["roots"])["features"], [])

    def test_endpoint_intersection_rounded_off_zero(self) -> None:
        report = find_intersections(math.sin, math.cos, 0, math.pi / 4)
        self.assertEqual([(f["x"], f["y"]) for f in report["features"]], [(0.7853981634, 0.7071067812)])

    # ---- steep roots versus poles (no sample lands on 0 in [-1.37, 1.3]) ----

    def test_roots_with_a_vertical_or_near_vertical_tangent_are_kept(self) -> None:
        def cbrt(x: float) -> float:
            return math.copysign(abs(x) ** (1.0 / 3.0), x)

        self.assertEqual(_summary(find_function_features(cbrt, -1.37, 1.3)), [(0.0, 0.0, "root")])
        steep = find_function_features(lambda x: math.tanh(1e5 * x), -1.37, 1.3)
        self.assertEqual(_summary(steep), [(0.0, 0.0, "root")])

    def test_poles_and_jumps_still_give_no_roots(self) -> None:
        cases = {
            "1/x": lambda x: 1 / x,
            "-1/x^3": lambda x: -1 / x**3,
            "1/(x-1)^2": lambda x: 1 / (x - 1) ** 2,
            "jump -1 to 1": lambda x: -1.0 if x < 0.37 else 1.0,
            "jump -1e-4 to 1": lambda x: -1e-4 if x < 0.37 else 1.0,
            "jump through 0 at the step": lambda x: (x > 0.37) - (x < 0.37),
        }
        for name, f in cases.items():
            with self.subTest(name):
                self.assertEqual(find_function_features(f, -1.37, 1.3, features=["roots"])["features"], [])
        tangent = find_function_features(math.tan, -1.37 - math.pi, 1.3 + math.pi, features=["roots"])
        self.assertEqual([f["x"] for f in tangent["features"]], [-3.141592654, 0.0, 3.141592654])

    def test_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 1, 1)


def _inflections(f, left: float, right: float) -> list:
    report = find_function_features(f, left, right, features=["inflections"])
    return [(feature["x"], feature["kind"]) for feature in report["features"]]


class TestInflectionPoints(unittest.TestCase):
    def test_default_features_do_not_include_inflections(self) -> None:
        kinds = {f["kind"] for f in find_function_features(lambda x: x**3 - 3 * x, -3, 3)["features"]}
        self.assertNotIn("inflection", kinds)

    def test_polynomial_inflections(self) -> None:
        self.assertEqual(_inflections(lambda x: x**3 - 3 * x, -3, 3), [(0.0, "inflection")])
        report = find_function_features(lambda x: (x - 1) ** 3 + 2, -2, 4, features=["inflections"])
        self.assertEqual(_summary(report), [(1.0, 2.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x**5, -1.3, 1.1), [(0.0, "inflection")])

    def test_smooth_functions_are_located_accurately(self) -> None:
        sine = [x for x, _ in _inflections(math.sin, -7, 7)]
        for found, expected in zip(sine, [-2 * math.pi, -math.pi, 0.0, math.pi, 2 * math.pi]):
            self.assertAlmostEqual(found, expected, places=4)
        self.assertEqual(len(sine), 5)
        gauss = [x for x, _ in _inflections(lambda x: math.exp(-x * x), -3, 3)]
        self.assertEqual(len(gauss), 2)
        self.assertAlmostEqual(gauss[1], 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(gauss[0], -1 / math.sqrt(2), places=5)
        self.assertEqual(_inflections(lambda x: 1 / (1 + math.exp(-x)), -8, 8), [(0.0, "inflection")])

    def test_no_inflection_without_a_concavity_change(self) -> None:
        for f in (
            lambda x: x**4,
            lambda x: x**2,
            lambda x: 2 * x + 1,
            lambda x: 1e6 * x + 3,
            lambda x: 7.0,
            math.exp,
            abs,
        ):
            self.assertEqual(_inflections(f, -2, 2.1), [])

    def test_poles_and_jumps_are_not_inflections(self) -> None:
        self.assertEqual(_inflections(lambda x: 1 / x, -1.37, 1.3), [])
        self.assertEqual(_inflections(lambda x: 1 / x**2, -1.37, 1.3), [])
        self.assertEqual(_inflections(lambda x: 1.0 if x >= 0.3 else -1.0, -2, 2), [])
        # tan changes concavity at its roots, not at its poles
        tangent = [x for x, _ in _inflections(math.tan, -5, 5)]
        self.assertEqual(len(tangent), 3)
        for found, expected in zip(tangent, [-math.pi, 0.0, math.pi]):
            self.assertAlmostEqual(found, expected, places=5)

    def test_steep_and_non_smooth_inflections(self) -> None:
        cbrt = lambda x: math.copysign(abs(x) ** (1 / 3), x)  # noqa: E731
        self.assertEqual(_inflections(cbrt, -1.37, 1.3), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x * abs(x), -2, 2.1), [(0.0, "inflection")])

    def test_scale_does_not_matter(self) -> None:
        self.assertEqual(_inflections(lambda x: 1e-9 * (x**3 - x), -2, 2), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: 1e9 * (x**3 - x), -2, 2), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: x**3 - 1e6 * x, -2, 2.3), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: (x - 1000) ** 3, 990, 1010), [(1000.0, "inflection")])

    def test_large_constant_offsets_are_found(self) -> None:
        # The curvature next to the inflection is below the rounding noise of a large offset
        # at a small step; the step grows until it is resolved
        self.assertEqual(_inflections(lambda x: 1e6 + x**3, -100, 100), [(0.0, "inflection")])
        self.assertEqual(_inflections(lambda x: 1e3 * x**3 + 1e8, -3, 3), [(0.0, "inflection")])
        offset_sine = [x for x, _ in _inflections(lambda x: 1e5 + math.sin(x), -10, 10)]
        self.assertEqual(len(offset_sine), 7)
        for found, k in zip(offset_sine, range(-3, 4)):
            self.assertAlmostEqual(found, k * math.pi, delta=0.01)

    def test_large_x_is_accurate_to_the_digits_reported(self) -> None:
        for centre in (1000.0, 1e4, 1e5):
            report = find_function_features(
                lambda x, c=centre: math.exp(-((x - c) ** 2)), centre - 5, centre + 5, features=["inflections"]
            )
            xs = [feature["x"] for feature in report["features"]]
            self.assertEqual(len(xs), 2)
            for found, expected in zip(xs, [centre - 1 / math.sqrt(2), centre + 1 / math.sqrt(2)]):
                # Rounded to the digits it is sure of: within one unit of the last digit
                digits = len(repr(found).split(".")[1]) if "." in repr(found) else 0
                self.assertLessEqual(abs(found - expected), 10.0**-digits)
            # y is evaluated at the reported x
            for feature in report["features"]:
                self.assertAlmostEqual(feature["y"], math.exp(-((feature["x"] - centre) ** 2)), places=9)

    def test_underflowing_values_do_not_crash(self) -> None:
        report = find_function_features(lambda x: 1e-200 * x**3, -3, 3, features=["roots", "inflections"])
        self.assertEqual(_summary(report), [(0.0, 0.0, "root"), (0.0, 0.0, "inflection")])

    def test_unknown_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 0, 1, features=["inflection_points"])


if __name__ == "__main__":
    unittest.main()
