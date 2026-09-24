"""Regression tests for MathUtils equation solving, definite integrals, and asymptote detection."""

from __future__ import annotations

import json
import math
import re
import unittest
from typing import List, Tuple

from browser import window

from utils.math_utils import MathUtils


def _parse_point_pairs(result: str) -> List[Tuple[float, float]]:
    """Parse '(x = a, y = b), ...' strings into sorted (x, y) tuples."""
    pairs = re.findall(r"\(x = ([^,]+), y = ([^)]+)\)", result)
    return sorted((float(x), float(y)) for x, y in pairs)


def _parse_indexed_solutions(result: str) -> List[Tuple[float, float]]:
    """Parse 'x1 = a, y1 = b, x2 = c, y2 = d' (or 'x = a, y = b') into sorted (x, y) tuples."""
    values = dict(item.split(" = ") for item in result.split(", "))
    xs = sorted(key for key in values if key.startswith("x"))
    return sorted((float(values[key]), float(values["y" + key[1:]])) for key in xs)


def _parse_numeric_solutions(result: str) -> List[Tuple[float, float]]:
    """Parse the numeric solver JSON output into sorted (x, y) tuples."""
    data = json.loads(result)
    return sorted((float(s["x"]), float(s["y"])) for s in data["solutions"])


class TestMathUtilsSolving(unittest.TestCase):
    def assert_points_close(self, actual: List[Tuple[float, float]], expected: List[Tuple[float, float]]) -> None:
        self.assertEqual(len(actual), len(expected), f"Expected {expected}, got {actual}")
        for (ax, ay), (ex, ey) in zip(sorted(actual), sorted(expected)):
            self.assertAlmostEqual(ax, ex, places=6)
            self.assertAlmostEqual(ay, ey, places=6)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def _roots(self, result: str) -> List[object]:
        self.assertTrue(result.startswith("[") and result.endswith("]"), result)
        inner = result[1:-1]
        return [window.math.evaluate(part) for part in inner.split(",")] if inner else []

    def test_solve_drops_roots_that_do_not_satisfy_equation(self) -> None:
        roots = self._roots(MathUtils.solve("x^3-2=0", "x"))
        self.assertGreater(len(roots), 0)
        for root in roots:
            residual = window.math.abs(window.math.subtract(window.math.pow(root, 3), 2))
            self.assertLess(float(residual), 1e-9)
        self.assertTrue(any(abs(float(window.math.re(r)) - 2 ** (1 / 3)) < 1e-9 for r in roots))

    def test_solve_keeps_valid_complex_roots(self) -> None:
        self.assertEqual(MathUtils.solve("x^2+1=0", "x"), "[i,-i]")

    def test_solve_keeps_regular_output(self) -> None:
        self.assertEqual(MathUtils.solve("x^2 - 4", "x"), "[2,-2]")
        self.assertEqual(MathUtils.solve("x^2-2=0", "x"), "[sqrt(2),-sqrt(2)]")

    def test_solve_keeps_symbolic_roots_it_cannot_check(self) -> None:
        self.assertEqual(MathUtils.solve("x + y = 3", "x"), "[-(-3+y)]")
