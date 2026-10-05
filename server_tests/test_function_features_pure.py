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


if __name__ == "__main__":
    unittest.main()
