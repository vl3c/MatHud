"""Pure Python tests for numeric integration helpers."""

from __future__ import annotations

import math
import unittest

from utils.numeric_integration import find_interior_singularity, integrate


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

    def test_equal_bounds_give_zero(self) -> None:
        result = integrate(lambda x: x, 1.0, 1.0, method="trapezoid", steps=10)
        self.assertEqual(result["value"], 0.0)
        self.assertEqual(result["error_estimate"], 0.0)

    def test_reversed_bounds_negate_the_value(self) -> None:
        for method in ("trapezoid", "midpoint", "simpson"):
            with self.subTest(method=method):
                forward = integrate(lambda x: x * x, 0.0, 2.0, method=method, steps=50)
                backward = integrate(lambda x: x * x, 2.0, 0.0, method=method, steps=50)
                self.assertEqual(backward["value"], -forward["value"])
                self.assertEqual(backward["error_estimate"], forward["error_estimate"])

    def test_none_method_and_steps_use_defaults(self) -> None:
        result = integrate(math.sin, 0.0, math.pi, method=None, steps=None)
        self.assertEqual(result["method"], "simpson")
        self.assertEqual(result["steps"], 400)
        self.assertAlmostEqual(result["value"], 2.0, places=10)

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


class TestNumericIntegrationSingularities(unittest.TestCase):
    def test_interior_pole_is_an_error(self) -> None:
        cases = (
            ("1/x", lambda x: 1.0 / x, -1.0, 2.0, 0.0),
            ("1/x^2", lambda x: 1.0 / (x * x), -1.0, 1.0, 0.0),
            ("tan", math.tan, 0.0, 2.0, math.pi / 2),
        )
        for name, fn, a, b, pole in cases:
            for method in ("trapezoid", "midpoint", "simpson"):
                with self.subTest(integrand=name, method=method):
                    with self.assertRaises(ValueError) as ctx:
                        integrate(fn, a, b, method=method, steps=200)
                    message = str(ctx.exception)
                    self.assertIn("singular or undefined near", message)
                    self.assertIn(f"{pole:.6g}", message)

    def test_interior_pole_found_with_few_steps(self) -> None:
        with self.assertRaises(ValueError):
            integrate(math.tan, 0.0, 2.0, method="simpson", steps=1)

    def test_interior_pole_with_reversed_bounds(self) -> None:
        with self.assertRaises(ValueError):
            integrate(lambda x: 1.0 / x, 2.0, -1.0, method="simpson", steps=200)

    def test_removable_point_is_not_a_singularity(self) -> None:
        def sinc(x: float) -> float:
            return math.sin(x) / x

        expected = 2 * 0.9460830703671830  # Si(1) on each side
        for method in ("trapezoid", "midpoint", "simpson"):
            with self.subTest(method=method):
                result = integrate(sinc, -1.0, 1.0, method=method, steps=200)
                self.assertAlmostEqual(result["value"], expected, places=5)
                self.assertNotIn("warning", result)

    def test_removable_point_at_a_bound(self) -> None:
        result = integrate(lambda x: math.sin(x) / x, 0.0, 1.0, method="simpson", steps=100)
        self.assertEqual(result["method"], "simpson")
        self.assertAlmostEqual(result["value"], 0.9460830703671830, places=10)

    def test_smooth_sharp_peak_is_not_a_singularity(self) -> None:
        result = integrate(lambda x: math.exp(-(((x - 0.3) / 0.01) ** 2)), -1.0, 2.0, method="simpson", steps=1000)
        self.assertAlmostEqual(result["value"], 0.01 * math.sqrt(math.pi), places=4)

    def test_undefined_bound_falls_back_to_midpoint(self) -> None:
        for method in ("trapezoid", "simpson"):
            with self.subTest(method=method):
                result = integrate(math.log, 0.0, 1.0, method=method, steps=200)
                self.assertEqual(result["method"], "midpoint")
                self.assertAlmostEqual(result["value"], -1.0, places=2)
                self.assertIn("undefined at the lower bound", result["warning"])
                self.assertIn("approximate", result["warning"])

    def test_undefined_upper_bound_with_reversed_bounds(self) -> None:
        result = integrate(math.log, 1.0, 0.0, method="simpson", steps=200)
        self.assertAlmostEqual(result["value"], 1.0, places=2)
        self.assertIn("undefined at the upper bound", result["warning"])

    def test_integrand_failing_everywhere_raises_its_own_error(self) -> None:
        def eval_fn(x: float) -> float:
            raise NameError("unknown function ln")

        with self.assertRaises(NameError):
            integrate(eval_fn, 0.0, 1.0, method="simpson", steps=10)

    def test_find_interior_singularity(self) -> None:
        self.assertAlmostEqual(find_interior_singularity(math.tan, 0.0, 3.0), math.pi / 2, places=6)
        self.assertIsNone(find_interior_singularity(math.sin, 0.0, 10.0))
        self.assertIsNone(find_interior_singularity(lambda x: 1.0 / x, 0.0, 1.0))  # at a bound only
        self.assertIsNone(find_interior_singularity(math.sin, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
