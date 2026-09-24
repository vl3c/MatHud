"""
Tests for the numeric solver package.

Tests cover individual helper functions, integration tests for the solver,
and fallback behavior in solve_system_of_equations.
"""

from __future__ import annotations

import json
import math
import unittest


class TestNumericSolverHelpers(unittest.TestCase):
    """Unit tests for individual helper functions."""

    def test_detect_variables(self) -> None:
        """Test variable detection extracts single letters and excludes functions."""
        from numeric_solver.expression_utils import detect_variables

        result = detect_variables(["sin(x) + y = 1"])
        self.assertEqual(result, ["x", "y"])

    def test_detect_variables_excludes_functions(self) -> None:
        """Test that log, exp, and other function names are excluded."""
        from numeric_solver.expression_utils import detect_variables

        result = detect_variables(["log(a) + exp(b) = 0"])
        self.assertEqual(result, ["a", "b"])

    def test_detect_variables_pi_excluded(self) -> None:
        """Test that 'pi' is excluded from variables."""
        from numeric_solver.expression_utils import detect_variables

        result = detect_variables(["x + pi = 0"])
        self.assertEqual(result, ["x"])

    def test_equation_to_residual_with_equals(self) -> None:
        """Test conversion of equation with equals sign."""
        from numeric_solver.expression_utils import equation_to_residual

        result = equation_to_residual("a + b = 1")
        self.assertEqual(result, "(a + b) - (1)")

    def test_equation_to_residual_without_equals(self) -> None:
        """Test that expressions without equals are returned as-is."""
        from numeric_solver.expression_utils import equation_to_residual

        result = equation_to_residual("x^2 - 1")
        self.assertEqual(result, "x^2 - 1")

    def test_gaussian_elimination(self) -> None:
        """Test Gaussian elimination with a known 3x3 system."""
        from numeric_solver.linear_algebra import solve_linear_system_gaussian

        # System: x + y + z = 6, 2y + 5z = -4, 2x + 5y - z = 27
        # Solution: x = 5, y = 3, z = -2
        A = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
        b = [6, -4, 27]

        result = solve_linear_system_gaussian(A, b)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 5.0, places=9)
        self.assertAlmostEqual(result[1], 3.0, places=9)
        self.assertAlmostEqual(result[2], -2.0, places=9)

    def test_gaussian_elimination_singular(self) -> None:
        """Test that singular matrices return None."""
        from numeric_solver.linear_algebra import solve_linear_system_gaussian

        # Singular matrix (row 2 = 2 * row 1)
        A = [[1, 2], [2, 4]]
        b = [3, 6]

        result = solve_linear_system_gaussian(A, b)
        self.assertIsNone(result)

    def test_gaussian_elimination_small_scale(self) -> None:
        """Pivots are judged relative to the matrix, so small-scale systems are not 'singular'."""
        from numeric_solver.linear_algebra import solve_linear_system_gaussian

        result = solve_linear_system_gaussian([[1e-13, 0.0], [0.0, 2e-13]], [1e-13, 6e-13])

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=9)
        self.assertAlmostEqual(result[1], 3.0, places=9)

    def test_deduplication(self) -> None:
        """Test that near-duplicate solutions are deduplicated."""
        from numeric_solver.utils import deduplicate_solutions

        solutions = [[1.0, 2.0], [1.0000001, 2.0000001], [3.0, 4.0]]
        variables = ["x", "y"]

        result = deduplicate_solutions(solutions, variables)

        self.assertEqual(len(result), 2)

    def test_generate_initial_guesses(self) -> None:
        """Test that initial guesses include structured guesses and random points."""
        from numeric_solver.utils import generate_initial_guesses

        guesses = generate_initial_guesses(2)

        # Should include origin
        self.assertIn([0.0, 0.0], guesses)

        # Should have reasonable count (structured + random)
        self.assertGreater(len(guesses), 20)


class TestNumericSolverIntegration(unittest.TestCase):
    """End-to-end integration tests for the solver."""

    def test_solve_numeric_linear_system(self) -> None:
        """Test solving a simple linear system."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["x + y = 4", "x - y = 2"])
        result = json.loads(result_json)

        self.assertEqual(result["method"], "newton_raphson")
        self.assertGreater(len(result["solutions"]), 0)

        sol = result["solutions"][0]
        self.assertAlmostEqual(sol["x"], 3.0, places=5)
        self.assertAlmostEqual(sol["y"], 1.0, places=5)

    def test_solve_numeric_transcendental(self) -> None:
        """Test solving sin(x) = 0.5."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["sin(x) = 0.5"])
        result = json.loads(result_json)

        self.assertGreater(len(result["solutions"]), 0)

        # Should find pi/6 ≈ 0.5236 or other solutions
        found_pi_6 = False
        for sol in result["solutions"]:
            x = sol["x"]
            # Check if sin(x) ≈ 0.5
            if abs(math.sin(x) - 0.5) < 0.001:
                found_pi_6 = True
                break
        self.assertTrue(found_pi_6, "Should find a solution where sin(x) = 0.5")

    def test_solve_numeric_nonlinear_system(self) -> None:
        """Test solving sin(x) + y = 1 and x^2 + y^2 = 4."""
        from numeric_solver import solve_numeric
        from numeric_solver.expression_utils import evaluate_residuals

        result_json = solve_numeric(["sin(x) + y = 1", "x^2 + y^2 = 4"])
        result = json.loads(result_json)

        # Verify solutions satisfy the equations
        for sol in result["solutions"]:
            x, y = sol["x"], sol["y"]
            # sin(x) + y should be close to 1
            r1 = abs(math.sin(x) + y - 1)
            # x^2 + y^2 should be close to 4
            r2 = abs(x * x + y * y - 4)
            self.assertLess(r1, 0.01, f"Residual 1 too large: {r1}")
            self.assertLess(r2, 0.01, f"Residual 2 too large: {r2}")

    def test_solve_numeric_circle_line(self) -> None:
        """Test intersection of circle and line."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["x^2 + y^2 = 25", "y = x + 1"])
        result = json.loads(result_json)

        # Should find two intersection points
        self.assertGreaterEqual(len(result["solutions"]), 1)

        for sol in result["solutions"]:
            x, y = sol["x"], sol["y"]
            # Verify on circle
            self.assertAlmostEqual(x * x + y * y, 25, places=3)
            # Verify on line
            self.assertAlmostEqual(y, x + 1, places=3)

    def test_solve_numeric_three_variables(self) -> None:
        """Test solving a 3-variable linear system."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["x + y + z = 6", "x - y + z = 2", "x + y - z = 0"])
        result = json.loads(result_json)

        self.assertGreater(len(result["solutions"]), 0)

        sol = result["solutions"][0]
        # Expected: x=1, y=2, z=3
        self.assertAlmostEqual(sol["x"], 1.0, places=3)
        self.assertAlmostEqual(sol["y"], 2.0, places=3)
        self.assertAlmostEqual(sol["z"], 3.0, places=3)

    def test_solve_numeric_no_real_solution(self) -> None:
        """Test that systems with no real solutions return empty."""
        from numeric_solver import solve_numeric

        # x^2 + 1 = 0 has no real solutions (single variable, square system)
        result_json = solve_numeric(["x^2 + 1 = 0"])
        result = json.loads(result_json)

        self.assertEqual(len(result["solutions"]), 0)
        self.assertIn("message", result)

    def test_solve_numeric_with_initial_guesses(self) -> None:
        """Test that user-provided initial guesses are used."""
        from numeric_solver import solve_numeric

        # Provide a guess close to pi/6 for sin(x) = 0.5
        result_json = solve_numeric(["sin(x) = 0.5"], initial_guesses=[[0.5]])
        result = json.loads(result_json)

        self.assertGreater(len(result["solutions"]), 0)

    def test_returns_json(self) -> None:
        """Test that result is valid JSON with expected structure."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["x = 1"])
        result = json.loads(result_json)

        self.assertIn("solutions", result)
        self.assertIn("variables", result)
        self.assertIn("method", result)
        self.assertEqual(result["method"], "newton_raphson")

    def test_solves_equations_with_large_magnitudes(self) -> None:
        """Residual noise near large roots exceeds any absolute tolerance."""
        from numeric_solver import solve_numeric

        cases = [
            ("x^3 = 2000000000", [2e9 ** (1.0 / 3.0)]),
            ("x^2 = 1000000000007", [-math.sqrt(1e12 + 7), math.sqrt(1e12 + 7)]),
            ("exp(x) = 10000000", [math.log(1e7)]),
        ]
        for equation, expected in cases:
            result = json.loads(solve_numeric([equation]))
            found = sorted(sol["x"] for sol in result["solutions"])
            self.assertEqual(len(found), len(expected), f"{equation}: {result}")
            for value, root in zip(found, sorted(expected)):
                self.assertTrue(math.isclose(value, root, rel_tol=1e-8), f"{equation}: {value} != {root}")

    def test_small_scale_equation_has_single_accurate_root(self) -> None:
        """An absolute residual tolerance accepts points far from the root of tiny-scale equations."""
        from numeric_solver import solve_numeric

        result = json.loads(solve_numeric(["0.000000000001*x = 0.001"]))

        self.assertEqual(len(result["solutions"]), 1, str(result))
        self.assertTrue(math.isclose(result["solutions"][0]["x"], 1e9, rel_tol=1e-8))

    def test_multiple_roots_reported_once(self) -> None:
        """Roots of multiplicity > 1 converge slowly but must not be reported many times."""
        from numeric_solver import solve_numeric

        for equation in ("(x-1)^2 = 0", "(x-1)^3 = 0", "x^2 = 0"):
            result = json.loads(solve_numeric([equation]))
            self.assertEqual(len(result["solutions"]), 1, f"{equation}: {result}")
            expected = 0.0 if equation == "x^2 = 0" else 1.0
            self.assertAlmostEqual(result["solutions"][0]["x"], expected, places=5)

    def test_deduplication_is_relative_and_keeps_best_residual(self) -> None:
        """Large-magnitude solutions are clustered relative to their size."""
        from numeric_solver.utils import deduplicate_solutions

        solutions = [[1e9 + 3.0], [1e9], [5.0]]
        result = deduplicate_solutions(solutions, ["x"], residuals=[1e-9, 1e-14, 1e-12])

        self.assertEqual(result, [{"x": 1e9}, {"x": 5.0}])

    def test_overdetermined_consistent_linear_system(self) -> None:
        """More equations than unknowns: Gauss-Newton finds the common solution."""
        from numeric_solver import solve_numeric

        result = json.loads(solve_numeric(["x + y = 5", "x - y = 1", "x + 2*y = 7"]))

        self.assertEqual(len(result["solutions"]), 1, str(result))
        self.assertAlmostEqual(result["solutions"][0]["x"], 3.0, places=8)
        self.assertAlmostEqual(result["solutions"][0]["y"], 2.0, places=8)
        self.assertIn("warning", result)

    def test_overdetermined_consistent_nonlinear_system(self) -> None:
        """Circle, line and hyperbola meeting at (3, 4) and (-4, -3)."""
        from numeric_solver import solve_numeric

        result = json.loads(solve_numeric(["x^2 + y^2 = 25", "y = x + 1", "x*y = 12"]))

        found = sorted((round(sol["x"], 6), round(sol["y"], 6)) for sol in result["solutions"])
        self.assertEqual(found, [(-4.0, -3.0), (3.0, 4.0)], str(result))

    def test_overdetermined_inconsistent_system_reports_message(self) -> None:
        """A least-squares minimum with non-zero residual is not a solution."""
        from numeric_solver import solve_numeric

        result = json.loads(solve_numeric(["x + y = 2", "x - y = 0", "x + 2*y = 5"]))

        self.assertEqual(result["solutions"], [])
        self.assertIn("inconsistent", result["message"])

    def test_underdetermined_system_returns_points_on_solution_set(self) -> None:
        """Fewer equations than unknowns: minimum-norm steps land on the solution set."""
        from numeric_solver import solve_numeric

        result = json.loads(solve_numeric(["x^2 + y^2 + z^2 = 1", "x + y + z = 0"]))

        self.assertGreater(len(result["solutions"]), 0, str(result))
        for sol in result["solutions"]:
            x, y, z = sol["x"], sol["y"], sol["z"]
            self.assertAlmostEqual(x * x + y * y + z * z, 1.0, places=6)
            self.assertAlmostEqual(x + y + z, 0.0, places=6)
        self.assertIn("not unique", result["warning"])

    def test_least_squares_helper_shapes(self) -> None:
        """Overdetermined systems use normal equations; underdetermined use minimum norm."""
        from numeric_solver.linear_algebra import solve_least_squares_gaussian

        over = solve_least_squares_gaussian([[1.0, 1.0], [1.0, -1.0], [1.0, 2.0]], [5.0, 1.0, 7.0])
        self.assertIsNotNone(over)
        self.assertAlmostEqual(over[0], 3.0, places=9)
        self.assertAlmostEqual(over[1], 2.0, places=9)

        under = solve_least_squares_gaussian([[1.0, 1.0]], [2.0])
        self.assertIsNotNone(under)
        self.assertAlmostEqual(under[0], 1.0, places=9)
        self.assertAlmostEqual(under[1], 1.0, places=9)

    def test_auto_detects_variables(self) -> None:
        """Test that variables are auto-detected when not provided."""
        from numeric_solver import solve_numeric

        result_json = solve_numeric(["a + b = 3", "a - b = 1"])
        result = json.loads(result_json)

        self.assertIn("a", result["variables"])
        self.assertIn("b", result["variables"])


class TestNumericSolverFallback(unittest.TestCase):
    """Tests for fallback integration with solve_system_of_equations."""

    def test_math_utils_solve_numeric_exists(self) -> None:
        """Test that MathUtils.solve_numeric method exists."""
        from utils.math_utils import MathUtils

        self.assertTrue(hasattr(MathUtils, "solve_numeric"))

    def test_math_utils_solve_numeric_works(self) -> None:
        """Test MathUtils.solve_numeric delegates to the package."""
        from utils.math_utils import MathUtils

        result_json = MathUtils.solve_numeric(["x + y = 4", "x - y = 2"])
        result = json.loads(result_json)

        self.assertEqual(result["method"], "newton_raphson")
        self.assertGreater(len(result["solutions"]), 0)

    def test_system_fallback_transcendental(self) -> None:
        """Test that solve_system_of_equations falls back for transcendental systems."""
        from utils.math_utils import MathUtils

        # This transcendental system should trigger fallback
        result = MathUtils.solve_system_of_equations(["sin(x) = 0.5"])

        # Should not be an error
        self.assertNotIn("Error", result)


class TestJacobianComputation(unittest.TestCase):
    """Tests for Jacobian computation."""

    def test_jacobian_linear_function(self) -> None:
        """Test Jacobian of linear functions."""
        from numeric_solver.jacobian import compute_jacobian

        # F(x, y) = [x + y, x - y]
        # Jacobian = [[1, 1], [1, -1]]
        residuals = ["x + y", "x - y"]
        variables = ["x", "y"]
        values = [1.0, 1.0]

        J = compute_jacobian(residuals, variables, values)

        self.assertIsNotNone(J)
        self.assertAlmostEqual(J[0][0], 1.0, places=5)
        self.assertAlmostEqual(J[0][1], 1.0, places=5)
        self.assertAlmostEqual(J[1][0], 1.0, places=5)
        self.assertAlmostEqual(J[1][1], -1.0, places=5)

    def test_jacobian_nonlinear_function(self) -> None:
        """Test Jacobian of x^2 + y^2 at point (1, 1)."""
        from numeric_solver.jacobian import compute_jacobian

        # F(x, y) = x^2 + y^2
        # dF/dx = 2x, dF/dy = 2y
        # At (1, 1): J = [2, 2]
        residuals = ["x^2 + y^2"]
        variables = ["x", "y"]
        values = [1.0, 1.0]

        J = compute_jacobian(residuals, variables, values)

        self.assertIsNotNone(J)
        self.assertAlmostEqual(J[0][0], 2.0, places=4)
        self.assertAlmostEqual(J[0][1], 2.0, places=4)

    def test_jacobian_step_scales_with_large_values(self) -> None:
        """A fixed step of 1e-7 is below the spacing of floats near 1e9."""
        from numeric_solver.jacobian import compute_jacobian

        J = compute_jacobian(["x^2", "x*y"], ["x", "y"], [1e9, 3.0])

        self.assertIsNotNone(J)
        self.assertTrue(math.isclose(J[0][0], 2e9, rel_tol=1e-6), f"d(x^2)/dx = {J[0][0]}")
        self.assertTrue(math.isclose(J[1][0], 3.0, rel_tol=1e-6), f"d(xy)/dx = {J[1][0]}")
        self.assertTrue(math.isclose(J[1][1], 1e9, rel_tol=1e-6), f"d(xy)/dy = {J[1][1]}")


class TestExpressionEvaluation(unittest.TestCase):
    """Tests for expression evaluation."""

    def test_evaluate_simple_expression(self) -> None:
        """Test evaluation of simple expressions."""
        from numeric_solver.expression_utils import evaluate_residuals

        result = evaluate_residuals(["x + y"], ["x", "y"], [1.0, 2.0])

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 3.0, places=10)

    def test_evaluate_trig_expression(self) -> None:
        """Test evaluation of trigonometric expressions."""
        from numeric_solver.expression_utils import evaluate_residuals

        result = evaluate_residuals(["sin(x)"], ["x"], [math.pi / 2])

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=10)

    def test_evaluate_invalid_returns_none(self) -> None:
        """Test that invalid expressions return None."""
        from numeric_solver.expression_utils import evaluate_residuals

        # Division by zero - math.js returns Infinity, which is non-finite
        result = evaluate_residuals(["1/0"], ["x"], [0.0])
        self.assertIsNone(result)

    def test_evaluate_undefined_variable(self) -> None:
        """Test that expressions with undefined variables return None."""
        from numeric_solver.expression_utils import evaluate_residuals

        result = evaluate_residuals(["x + z"], ["x"], [1.0])
        self.assertIsNone(result)
