"""Pure Python tests for numeric integration helpers."""

from __future__ import annotations

import math
import unittest

from utils.numeric_integration import integrate


class TestNumericIntegrationPure(unittest.TestCase):
    def test_trapezoid_integrates_polynomial(self) -> None:
        result = integrate(
            eval_fn=lambda x: x * x,
            lower_bound=0.0,
            upper_bound=1.0,
            method="trapezoid",
            steps=400,
        )
        self.assertAlmostEqual(result["value"], 1.0 / 3.0, places=4)
        self.assertGreaterEqual(result["error_estimate"], 0.0)

    def test_midpoint_integrates_polynomial(self) -> None:
        result = integrate(
            eval_fn=lambda x: x * x + 2.0 * x + 1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            method="midpoint",
            steps=300,
        )
        self.assertAlmostEqual(result["value"], 26.0 / 3.0, places=4)

    def test_simpson_integrates_sine(self) -> None:
        result = integrate(
            eval_fn=math.sin,
            lower_bound=0.0,
            upper_bound=math.pi,
            method="simpson",
            steps=199,  # odd on purpose; implementation should normalize
        )
        self.assertAlmostEqual(result["value"], 2.0, places=8)
        self.assertEqual(result["method"], "simpson")
        self.assertEqual(result["steps"] % 2, 0)

    def test_rejects_invalid_method(self) -> None:
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 0.0, 1.0, method="romberg", steps=10)

    def test_rejects_non_finite_bounds(self) -> None:
        with self.assertRaises(ValueError):
            integrate(lambda x: x, float("nan"), 1.0, method="trapezoid", steps=10)
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 0.0, float("inf"), method="trapezoid", steps=10)

    def test_rejects_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 1.0, 1.0, method="trapezoid", steps=10)
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 2.0, 1.0, method="trapezoid", steps=10)

    def test_rejects_invalid_steps(self) -> None:
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 0.0, 1.0, method="trapezoid", steps=0)
        with self.assertRaises(TypeError):
            integrate(lambda x: x, 0.0, 1.0, method="trapezoid", steps=True)  # type: ignore[arg-type]

    def test_reports_steps_of_the_returned_value(self) -> None:
        # The value comes from the refined pass, so steps must be doubled too
        for method in ("trapezoid", "midpoint"):
            with self.subTest(method=method):
                result = integrate(lambda x: x * x, 0.0, 1.0, method=method, steps=100)
                self.assertEqual(result["steps"], 200)

    def test_smooth_integrand_has_no_warning(self) -> None:
        for method in ("trapezoid", "midpoint", "simpson"):
            with self.subTest(method=method):
                result = integrate(math.exp, 0.0, 50.0, method=method, steps=100)
                self.assertNotIn("warning", result)

    def test_flags_endpoint_singularity(self) -> None:
        result = integrate(lambda x: 1.0 / math.sqrt(x), 0.0, 1.0, method="midpoint", steps=200)
        self.assertIn("warning", result)
        self.assertIn("lower bound", result["warning"])

        result = integrate(lambda x: 1.0 / math.sqrt(1.0 - x), 0.0, 1.0, method="midpoint", steps=200)
        self.assertIn("upper bound", result["warning"])

    def test_flags_integrand_that_fails_near_endpoint(self) -> None:
        def eval_fn(x: float) -> float:
            if x < 1e-6:
                return float("inf")
            return 1.0 / math.sqrt(x)

        result = integrate(eval_fn, 0.0, 1.0, method="midpoint", steps=100)
        self.assertIn("warning", result)


if __name__ == "__main__":
    unittest.main()
