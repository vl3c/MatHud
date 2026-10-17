"""Tests for utils.linear_systems, the elimination behind solve_system_of_equations without SymPy."""

from __future__ import annotations

import unittest

from utils.linear_systems import solve_linear


class TestSolveLinear(unittest.TestCase):
    def test_one_solution(self) -> None:
        self.assertEqual(solve_linear([[1, 1], [1, -1]], [4, 2], ["x", "y"]), {"solutions": [{"x": "3", "y": "1"}]})
        # rounding in the elimination is dropped: 0.7, not 0.700000000000001
        self.assertEqual(
            solve_linear([[1 / 3, 1 / 7], [1, -1]], [1, 2], ["x", "y"]), {"solutions": [{"x": "2.7", "y": "0.7"}]}
        )
        self.assertEqual(
            solve_linear([[0, 1, 0], [1, 0, 0], [0, 0, 2]], [5, 6, 1], ["x", "y", "z"]),
            {"solutions": [{"x": "6", "y": "5", "z": "0.5"}]},
        )

    def test_inconsistent(self) -> None:
        self.assertEqual(solve_linear([[1, 1], [1, 1]], [1, 2], ["x", "y"]), {"none": True})
        self.assertEqual(solve_linear([[1, 1], [0, 0]], [1, 2], ["x", "y"]), {"none": True})

    def test_dependent(self) -> None:
        self.assertEqual(
            solve_linear([[1, 1], [2, 2]], [1, 2], ["x", "y"]),
            {"infinite": True, "general": {"x": "1-y"}, "free": ["y"], "rank": 1},
        )
        self.assertEqual(
            solve_linear([[1, -2, 0], [0, 0, 1]], [0, 3], ["x", "y", "z"]),
            {"infinite": True, "general": {"x": "2*y", "z": "3"}, "free": ["y"], "rank": 2},
        )
        self.assertEqual(
            solve_linear([[0, 0], [0, 0]], [0, 0], ["x", "y"]),
            {"infinite": True, "general": {}, "free": ["x", "y"], "rank": 0},
        )


if __name__ == "__main__":
    unittest.main()
