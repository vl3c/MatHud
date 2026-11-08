"""The generic least-squares pieces of custom regression: QR solves, rank checks and Levenberg-Marquardt."""

from __future__ import annotations

import math
import unittest
from typing import List, Optional, Sequence

from utils.statistics.levenberg_marquardt import (
    EvaluationBudget,
    levenberg_marquardt,
    solve_least_squares,
    undetermined_columns,
)


class TestSolveLeastSquares(unittest.TestCase):
    def test_exact_solution(self) -> None:
        columns = [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]]
        solution, skipped = solve_least_squares(columns, [1.0, 3.0, 5.0, 7.0])
        self.assertEqual(skipped, [])
        self.assertAlmostEqual(solution[0], 1.0, places=12)
        self.assertAlmostEqual(solution[1], 2.0, places=12)

    def test_a_dependent_column_gets_zero(self) -> None:
        columns = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]
        solution, skipped = solve_least_squares(columns, [3.0, 5.0, 7.0])
        self.assertEqual(skipped, [1])
        self.assertEqual(solution[1], 0.0)
        self.assertAlmostEqual(solution[0], 2.0, places=10)
        self.assertAlmostEqual(solution[2], 1.0, places=10)

    def test_wildly_scaled_columns(self) -> None:
        xs = [2000.0 + i for i in range(10)]
        columns = [[1e-30 * x for x in xs], [1e20 for _ in xs]]
        rhs = [3e-30 * 1e30 * x + 5.0 for x in xs]
        solution, skipped = solve_least_squares(columns, rhs)
        self.assertEqual(skipped, [])
        self.assertAlmostEqual(solution[0] * 1e-30, 3.0, places=8)
        self.assertAlmostEqual(solution[1] * 1e20, 5.0, places=6)


class TestUndeterminedColumns(unittest.TestCase):
    def test_combinations_of_earlier_columns(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 0.0, 1.0, 0.0]
        self.assertEqual(undetermined_columns([a, b, [a[i] - 2 * b[i] for i in range(4)]]), [2])
        self.assertEqual(undetermined_columns([a, [0.0] * 4]), [1])
        self.assertEqual(undetermined_columns([a, b]), [])

    def test_scales_judge_tiny_columns(self) -> None:
        # A column 1e-9 of its own unprojected effect is undetermined at tolerance 1e-6
        self.assertEqual(undetermined_columns([[1e-9, 0.0, 0.0]], 1e-6, [1.0]), [0])
        self.assertEqual(undetermined_columns([[1e-9, 0.0, 0.0]], 1e-6, [1e-9]), [])


class TestLevenbergMarquardt(unittest.TestCase):
    def test_rosenbrock(self) -> None:
        budget = EvaluationBudget(100_000)

        def residuals(p: Sequence[float]) -> Optional[List[float]]:
            budget.spend(2)
            return [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]]

        run = levenberg_marquardt(residuals, [-1.2, 1.0], budget)
        assert run is not None
        self.assertTrue(run.converged)
        self.assertAlmostEqual(run.params[0], 1.0, places=6)
        self.assertAlmostEqual(run.params[1], 1.0, places=6)

    def test_exponential_decay(self) -> None:
        xs = [0.5 * i for i in range(20)]
        ys = [4.0 * math.exp(-0.7 * x) for x in xs]
        budget = EvaluationBudget(100_000)

        def residuals(p: Sequence[float]) -> Optional[List[float]]:
            budget.spend(len(xs))
            return [p[0] * math.exp(-p[1] * x) - y for x, y in zip(xs, ys)]

        run = levenberg_marquardt(residuals, [1.0, 0.1], budget)
        assert run is not None
        self.assertTrue(run.converged)
        self.assertAlmostEqual(run.params[0], 4.0, places=8)
        self.assertAlmostEqual(run.params[1], 0.7, places=8)

    def test_an_unevaluable_start_gives_none(self) -> None:
        self.assertIsNone(levenberg_marquardt(lambda p: None, [1.0], EvaluationBudget(100)))

    def test_the_budget_stops_the_iteration(self) -> None:
        budget = EvaluationBudget(30)

        def residuals(p: Sequence[float]) -> Optional[List[float]]:
            budget.spend(2)
            return [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]]

        run = levenberg_marquardt(residuals, [-1.2, 1.0], budget)
        assert run is not None
        self.assertFalse(run.converged)
        self.assertTrue(budget.exhausted)

    def test_steps_that_fail_to_evaluate_are_damped(self) -> None:
        budget = EvaluationBudget(100_000)

        def residuals(p: Sequence[float]) -> Optional[List[float]]:
            budget.spend(1)
            if p[0] <= 0.0:
                return None  # ln of a non-positive number
            return [math.log(p[0]) - 1.0]

        run = levenberg_marquardt(residuals, [0.05], budget)
        assert run is not None
        self.assertAlmostEqual(run.params[0], math.e, places=6)


if __name__ == "__main__":
    unittest.main()
