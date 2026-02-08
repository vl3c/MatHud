"""
Pure Python tests for regression fitting algorithms.

These tests verify the core fitting algorithms in utils.statistics.regression
without requiring browser/Brython dependencies.
"""

from __future__ import annotations

import math
import unittest

from utils.statistics.regression import (
    SUPPORTED_MODEL_TYPES,
    fit_regression,
    fit_linear,
    fit_polynomial,
    fit_exponential,
    fit_logarithmic,
    fit_power,
    fit_logistic,
    fit_sinusoidal,
    calculate_r_squared,
    build_expression,
    _validate_data,
    _validate_positive,
    _matrix_multiply,
    _matrix_transpose,
    _matrix_inverse,
)


class TestValidation(unittest.TestCase):
    """Tests for input validation functions."""

    def test_validate_data_accepts_valid_input(self) -> None:
        # Should not raise for valid input
        _validate_data([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        _validate_data([1, 2, 3], [4, 5, 6])  # Integers also valid

    def test_validate_data_rejects_non_list(self) -> None:
        with self.assertRaises(TypeError):
            _validate_data((1, 2, 3), [4, 5, 6])  # type: ignore
        with self.assertRaises(TypeError):
            _validate_data([1, 2, 3], (4, 5, 6))  # type: ignore

    def test_validate_data_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            _validate_data([1.0, 2.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            _validate_data([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_validate_data_rejects_too_few_points(self) -> None:
        with self.assertRaises(ValueError):
            _validate_data([1.0], [2.0])  # Need at least 2
        with self.assertRaises(ValueError):
            _validate_data([], [])

    def test_validate_data_rejects_non_numeric(self) -> None:
        with self.assertRaises(TypeError):
            _validate_data([1.0, "a", 3.0], [1.0, 2.0, 3.0])  # type: ignore
        with self.assertRaises(TypeError):
            _validate_data([1.0, 2.0, 3.0], [1.0, None, 3.0])  # type: ignore

    def test_validate_data_rejects_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            _validate_data([1.0, float("inf"), 3.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            _validate_data([1.0, 2.0, 3.0], [1.0, float("nan"), 3.0])
        with self.assertRaises(ValueError):
            _validate_data([1.0, float("-inf"), 3.0], [1.0, 2.0, 3.0])

    def test_validate_positive_accepts_positive(self) -> None:
        _validate_positive([1.0, 2.0, 0.001], "test")

    def test_validate_positive_rejects_non_positive(self) -> None:
        with self.assertRaises(ValueError):
            _validate_positive([1.0, 0.0, 3.0], "test")
        with self.assertRaises(ValueError):
            _validate_positive([1.0, -1.0, 3.0], "test")


class TestMatrixOperations(unittest.TestCase):
    """Tests for pure Python matrix operations."""

    def test_matrix_transpose(self) -> None:
        A = [[1, 2, 3], [4, 5, 6]]
        At = _matrix_transpose(A)
        self.assertEqual(At, [[1, 4], [2, 5], [3, 6]])

    def test_matrix_transpose_empty(self) -> None:
        self.assertEqual(_matrix_transpose([]), [])

    def test_matrix_multiply(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = _matrix_multiply(A, B)
        self.assertEqual(C, [[19, 22], [43, 50]])

    def test_matrix_multiply_non_square(self) -> None:
        A = [[1, 2, 3], [4, 5, 6]]  # 2x3
        B = [[7], [8], [9]]  # 3x1
        C = _matrix_multiply(A, B)
        self.assertEqual(C, [[50], [122]])

    def test_matrix_inverse_2x2(self) -> None:
        A = [[4.0, 7.0], [2.0, 6.0]]
        A_inv = _matrix_inverse(A)
        # A * A_inv should be identity
        I = _matrix_multiply(A, A_inv)
        self.assertAlmostEqual(I[0][0], 1.0, places=10)
        self.assertAlmostEqual(I[0][1], 0.0, places=10)
        self.assertAlmostEqual(I[1][0], 0.0, places=10)
        self.assertAlmostEqual(I[1][1], 1.0, places=10)

    def test_matrix_inverse_3x3(self) -> None:
        A = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
        A_inv = _matrix_inverse(A)
        I = _matrix_multiply(A, A_inv)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(I[i][j], expected, places=9)

    def test_matrix_inverse_singular_raises(self) -> None:
        A = [[1.0, 2.0], [2.0, 4.0]]  # Singular (row 2 = 2 * row 1)
        with self.assertRaises(ValueError):
            _matrix_inverse(A)


class TestRSquared(unittest.TestCase):
    """Tests for R-squared calculation."""

    def test_perfect_fit(self) -> None:
        y_actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
        r2 = calculate_r_squared(y_actual, y_predicted)
        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_constant_prediction_at_mean(self) -> None:
        y_actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_predicted = [3.0, 3.0, 3.0, 3.0, 3.0]  # Mean of y_actual
        r2 = calculate_r_squared(y_actual, y_predicted)
        self.assertAlmostEqual(r2, 0.0, places=10)

    def test_partial_fit(self) -> None:
        y_actual = [1.0, 2.0, 3.0, 4.0]
        y_predicted = [1.1, 1.9, 3.1, 3.9]
        r2 = calculate_r_squared(y_actual, y_predicted)
        self.assertGreater(r2, 0.9)
        self.assertLess(r2, 1.0)

    def test_constant_y_actual(self) -> None:
        y_actual = [5.0, 5.0, 5.0, 5.0]
        y_predicted = [5.0, 5.0, 5.0, 5.0]
        r2 = calculate_r_squared(y_actual, y_predicted)
        self.assertEqual(r2, 1.0)  # Perfect match

    def test_mismatched_lengths_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_r_squared([1, 2, 3], [1, 2])

    def test_empty_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_r_squared([], [])


class TestBuildExpression(unittest.TestCase):
    """Tests for expression string generation."""

    def test_linear_expression(self) -> None:
        expr = build_expression("linear", {"m": 2.5, "b": -3.0})
        self.assertIn("2.5", expr)
        self.assertIn("x", expr)
        self.assertIn("3", expr)
        self.assertIn("-", expr)

    def test_polynomial_expression(self) -> None:
        expr = build_expression("polynomial", {"a0": 1.0, "a1": 2.0, "a2": 3.0})
        self.assertIn("x", expr)
        self.assertIn("x^2", expr)

    def test_exponential_expression(self) -> None:
        expr = build_expression("exponential", {"a": 2.0, "b": 0.5})
        self.assertIn("exp", expr)
        self.assertIn("x", expr)

    def test_logarithmic_expression(self) -> None:
        expr = build_expression("logarithmic", {"a": 1.0, "b": 2.0})
        self.assertIn("ln(x)", expr)

    def test_power_expression(self) -> None:
        expr = build_expression("power", {"a": 2.0, "b": 0.5})
        self.assertIn("x^", expr)

    def test_logistic_expression(self) -> None:
        expr = build_expression("logistic", {"L": 10.0, "k": 1.0, "x0": 5.0})
        self.assertIn("exp", expr)
        self.assertIn("1 +", expr)

    def test_sinusoidal_expression(self) -> None:
        expr = build_expression("sinusoidal", {"a": 2.0, "b": 1.0, "c": 0.0, "d": 0.0})
        self.assertIn("sin", expr)

    def test_unknown_model_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_expression("unknown", {})


class TestFitLinear(unittest.TestCase):
    """Tests for linear regression."""

    def test_perfect_linear_fit(self) -> None:
        # y = 2x + 1
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = fit_linear(x, y)

        self.assertEqual(result["model_type"], "linear")
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=10)
        self.assertAlmostEqual(result["coefficients"]["b"], 1.0, places=10)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_noisy_linear_fit(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.1, 3.9, 6.1, 7.9, 10.1]  # Approximately y = 2x
        result = fit_linear(x, y)

        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=1)
        self.assertGreater(result["r_squared"], 0.99)

    def test_zero_variance_x_raises(self) -> None:
        x = [5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_linear(x, y)


class TestFitPolynomial(unittest.TestCase):
    """Tests for polynomial regression."""

    def test_quadratic_fit(self) -> None:
        # y = x^2 + 2x + 1
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]
        result = fit_polynomial(x, y, degree=2)

        self.assertEqual(result["model_type"], "polynomial")
        self.assertAlmostEqual(result["coefficients"]["a0"], 1.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["a1"], 2.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["a2"], 1.0, places=8)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_cubic_fit(self) -> None:
        # y = x^3
        x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        y = [-8.0, -1.0, 0.0, 1.0, 8.0]
        result = fit_polynomial(x, y, degree=3)

        self.assertAlmostEqual(result["coefficients"]["a3"], 1.0, places=8)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_invalid_degree_raises(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [1.0, 4.0, 9.0]
        with self.assertRaises(ValueError):
            fit_polynomial(x, y, degree=0)
        with self.assertRaises(ValueError):
            fit_polynomial(x, y, degree=-1)

    def test_insufficient_points_raises(self) -> None:
        x = [1.0, 2.0]
        y = [1.0, 4.0]
        # Need at least 3 points for degree 2
        with self.assertRaises(ValueError):
            fit_polynomial(x, y, degree=2)


class TestFitExponential(unittest.TestCase):
    """Tests for exponential regression."""

    def test_exponential_fit(self) -> None:
        # y = 2 * e^(0.5x)
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [2.0 * math.exp(0.5 * xi) for xi in x]
        result = fit_exponential(x, y)

        self.assertEqual(result["model_type"], "exponential")
        self.assertAlmostEqual(result["coefficients"]["a"], 2.0, places=5)
        self.assertAlmostEqual(result["coefficients"]["b"], 0.5, places=5)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=8)

    def test_non_positive_y_raises(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, -2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_exponential(x, y)

    def test_zero_y_raises(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 0.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_exponential(x, y)


class TestFitLogarithmic(unittest.TestCase):
    """Tests for logarithmic regression."""

    def test_logarithmic_fit(self) -> None:
        # y = 1 + 2*ln(x)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0 + 2.0 * math.log(xi) for xi in x]
        result = fit_logarithmic(x, y)

        self.assertEqual(result["model_type"], "logarithmic")
        self.assertAlmostEqual(result["coefficients"]["a"], 1.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["b"], 2.0, places=8)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_non_positive_x_raises(self) -> None:
        x = [1.0, -2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_logarithmic(x, y)

    def test_zero_x_raises(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_logarithmic(x, y)


class TestFitPower(unittest.TestCase):
    """Tests for power regression."""

    def test_power_fit(self) -> None:
        # y = 2 * x^0.5 (square root)
        x = [1.0, 4.0, 9.0, 16.0, 25.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = fit_power(x, y)

        self.assertEqual(result["model_type"], "power")
        self.assertAlmostEqual(result["coefficients"]["a"], 2.0, places=5)
        self.assertAlmostEqual(result["coefficients"]["b"], 0.5, places=5)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=8)

    def test_non_positive_x_raises(self) -> None:
        x = [1.0, -2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_power(x, y)

    def test_non_positive_y_raises(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, -2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            fit_power(x, y)


class TestFitLogistic(unittest.TestCase):
    """Tests for logistic regression."""

    def test_logistic_fit_s_curve(self) -> None:
        # Generate S-curve data: y = 10 / (1 + e^(-1*(x-5)))
        L, k, x0 = 10.0, 1.0, 5.0
        x = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        y = [L / (1 + math.exp(-k * (xi - x0))) for xi in x]

        result = fit_logistic(x, y)

        self.assertEqual(result["model_type"], "logistic")
        # Check that fit is reasonably good
        self.assertGreater(result["r_squared"], 0.95)
        # Coefficients should be in the right ballpark
        self.assertAlmostEqual(result["coefficients"]["L"], L, delta=1.0)

    def test_logistic_returns_valid_result(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = fit_logistic(x, y)

        self.assertEqual(result["model_type"], "logistic")
        self.assertIn("L", result["coefficients"])
        self.assertIn("k", result["coefficients"])
        self.assertIn("x0", result["coefficients"])
        self.assertIn("expression", result)


class TestFitSinusoidal(unittest.TestCase):
    """Tests for sinusoidal regression."""

    def test_sinusoidal_fit(self) -> None:
        # y = 2*sin(x) + 1
        a, b, c, d = 2.0, 1.0, 0.0, 1.0
        x = [i * 0.5 for i in range(20)]
        y = [a * math.sin(b * xi + c) + d for xi in x]

        result = fit_sinusoidal(x, y)

        self.assertEqual(result["model_type"], "sinusoidal")
        # Should get a good fit
        self.assertGreater(result["r_squared"], 0.9)

    def test_sinusoidal_with_phase_shift(self) -> None:
        # y = sin(x + pi/4)
        x = [i * 0.5 for i in range(20)]
        y = [math.sin(xi + math.pi/4) for xi in x]

        result = fit_sinusoidal(x, y)
        self.assertGreater(result["r_squared"], 0.9)

    def test_sinusoidal_minimum_points(self) -> None:
        # Need at least 4 points
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 0.0]
        with self.assertRaises(ValueError):
            fit_sinusoidal(x, y)


class TestFitRegressionDispatcher(unittest.TestCase):
    """Tests for the main fit_regression dispatcher function."""

    def test_supported_model_types(self) -> None:
        expected = ("linear", "polynomial", "exponential", "logarithmic",
                    "power", "logistic", "sinusoidal")
        self.assertEqual(SUPPORTED_MODEL_TYPES, expected)

    def test_dispatch_linear(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_regression(x, y, "linear")
        self.assertEqual(result["model_type"], "linear")

    def test_dispatch_polynomial_requires_degree(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 9.0, 16.0]
        with self.assertRaises(ValueError):
            fit_regression(x, y, "polynomial")  # Missing degree

        result = fit_regression(x, y, "polynomial", degree=2)
        self.assertEqual(result["model_type"], "polynomial")

    def test_dispatch_exponential(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 8.0, 16.0]
        result = fit_regression(x, y, "exponential")
        self.assertEqual(result["model_type"], "exponential")

    def test_dispatch_logarithmic(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.69, 1.1, 1.39]
        result = fit_regression(x, y, "logarithmic")
        self.assertEqual(result["model_type"], "logarithmic")

    def test_dispatch_power(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 9.0, 16.0]
        result = fit_regression(x, y, "power")
        self.assertEqual(result["model_type"], "power")

    def test_dispatch_logistic(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = fit_regression(x, y, "logistic")
        self.assertEqual(result["model_type"], "logistic")

    def test_dispatch_sinusoidal(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        y = [0.0, 0.84, 0.91, 0.14, -0.76, -0.96]
        result = fit_regression(x, y, "sinusoidal")
        self.assertEqual(result["model_type"], "sinusoidal")

    def test_unsupported_model_raises(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            fit_regression(x, y, "unsupported")

    def test_case_insensitive_model_type(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_regression(x, y, "LINEAR")
        self.assertEqual(result["model_type"], "linear")

        result = fit_regression(x, y, "Linear")
        self.assertEqual(result["model_type"], "linear")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_empty_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_linear([], [])

    def test_single_point_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_linear([1.0], [2.0])

    def test_polynomial_degree_equals_points_minus_one(self) -> None:
        # 3 points, degree 2 - exactly solvable
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 4.0]
        result = fit_polynomial(x, y, degree=2)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=8)

    def test_polynomial_degree_too_high_raises(self) -> None:
        # 3 points, degree 3 requires 4 points
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 4.0]
        with self.assertRaises(ValueError):
            fit_polynomial(x, y, degree=3)

    def test_negative_slope_linear(self) -> None:
        # y = -2x + 10
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], -2.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["b"], 10.0, places=8)

    def test_horizontal_line(self) -> None:
        # y = 5 (constant)
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 5.0, 5.0, 5.0]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 0.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["b"], 5.0, places=8)

    def test_duplicate_x_values_linear(self) -> None:
        # Multiple y values at same x - should still fit
        x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        y = [1.9, 2.1, 3.9, 4.1, 5.9, 6.1]
        result = fit_linear(x, y)
        # Should approximate y = 2x
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=1)
        self.assertGreater(result["r_squared"], 0.99)

    def test_very_small_coefficients(self) -> None:
        # y = 0.0001x + 0.0002
        x = [0.0, 10000.0, 20000.0, 30000.0]
        y = [0.0002, 1.0002, 2.0002, 3.0002]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 0.0001, places=6)
        self.assertAlmostEqual(result["coefficients"]["b"], 0.0002, places=4)

    def test_very_large_coefficients(self) -> None:
        # y = 1000000x + 2000000
        x = [0.0, 1.0, 2.0, 3.0]
        y = [2000000.0, 3000000.0, 4000000.0, 5000000.0]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 1000000.0, places=2)
        self.assertAlmostEqual(result["coefficients"]["b"], 2000000.0, places=2)

    def test_negative_y_values_linear(self) -> None:
        # y = 2x - 10
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [-10.0, -8.0, -6.0, -4.0, -2.0]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=8)
        self.assertAlmostEqual(result["coefficients"]["b"], -10.0, places=8)

    def test_integer_input_converted(self) -> None:
        # Integers should work
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=8)

    def test_mixed_int_float_input(self) -> None:
        x = [1, 2.0, 3, 4.0]
        y = [2.0, 4, 6.0, 8]
        result = fit_linear(x, y)
        self.assertAlmostEqual(result["coefficients"]["m"], 2.0, places=8)

    def test_polynomial_zero_coefficients(self) -> None:
        # y = x^2 (no x^1 or x^0 terms)
        x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        y = [4.0, 1.0, 0.0, 1.0, 4.0]
        result = fit_polynomial(x, y, degree=2)
        self.assertAlmostEqual(result["coefficients"]["a0"], 0.0, places=6)
        self.assertAlmostEqual(result["coefficients"]["a1"], 0.0, places=6)
        self.assertAlmostEqual(result["coefficients"]["a2"], 1.0, places=6)

    def test_exponential_decay(self) -> None:
        # y = 10 * e^(-0.5x) - decaying exponential
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [10.0 * math.exp(-0.5 * xi) for xi in x]
        result = fit_exponential(x, y)
        self.assertAlmostEqual(result["coefficients"]["a"], 10.0, places=4)
        self.assertAlmostEqual(result["coefficients"]["b"], -0.5, places=4)

    def test_whitespace_in_model_type(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_regression(x, y, "  linear  ")
        self.assertEqual(result["model_type"], "linear")


class TestRegressionResultStructure(unittest.TestCase):
    """Tests that verify the structure of RegressionResult."""

    def test_result_has_required_keys(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_linear(x, y)

        self.assertIn("expression", result)
        self.assertIn("coefficients", result)
        self.assertIn("r_squared", result)
        self.assertIn("model_type", result)

    def test_expression_is_string(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_linear(x, y)
        self.assertIsInstance(result["expression"], str)

    def test_coefficients_is_dict(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_linear(x, y)
        self.assertIsInstance(result["coefficients"], dict)

    def test_r_squared_is_float_in_range(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = fit_linear(x, y)
        self.assertIsInstance(result["r_squared"], float)
        self.assertGreaterEqual(result["r_squared"], 0.0)
        self.assertLessEqual(result["r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()
