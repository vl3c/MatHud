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

    def test_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            find_function_features(math.sin, 1, 1)


if __name__ == "__main__":
    unittest.main()
