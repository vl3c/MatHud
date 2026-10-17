"""Tests for MathUtils._scanned_zeros, the zero scan behind the asymptote analysis of drawn functions.

It finds the zeros of non-polynomial denominators, log arguments and abs arguments in the
drawing bounds, so it runs for every function drawn; the functions here are plain Python.
"""

from __future__ import annotations

import math
import unittest
from typing import Callable, List, Optional

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils imports)
from utils.math_utils import MathUtils


def _safe(function: Callable[[float], float]) -> Callable[[float], Optional[float]]:
    def evaluate(x: float) -> Optional[float]:
        try:
            value = function(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return float(value)

    return evaluate


def _zeros(function: Callable[[float], float], left: float, right: float) -> List[float]:
    return MathUtils._scanned_zeros(_safe(function), left, right)


class TestZeroScan(unittest.TestCase):
    def assert_zeros(self, found: List[float], expected: List[float], tolerance: float = 1e-9) -> None:
        self.assertEqual(len(found), len(expected), found)
        for value, target in zip(found, expected):
            self.assertAlmostEqual(value, target, delta=tolerance * max(1.0, abs(target)))

    def test_sign_changes(self) -> None:
        self.assert_zeros(_zeros(math.sin, -7, 7), [-2 * math.pi, -math.pi, 0.0, math.pi, 2 * math.pi])
        self.assert_zeros(_zeros(lambda x: math.exp(x) - 2, -5, 7), [math.log(2)])

    def test_poles_and_jumps_are_not_zeros(self) -> None:
        self.assertEqual(_zeros(lambda x: 1 / x, -5, 7), [])
        self.assert_zeros(_zeros(math.tan, -1, 4), [0.0, math.pi])
        self.assertEqual(_zeros(lambda x: 1.0 if x > 0.3 else -1.0, -5, 7), [])

    def test_touching_zeros(self) -> None:
        # |f| is below rounding within about 1e-8 of a double zero, so it is found that closely
        self.assert_zeros(_zeros(lambda x: 1 - math.cos(x), -5, 7), [0.0, 2 * math.pi], 1e-7)
        self.assert_zeros(_zeros(lambda x: math.sin(x) ** 2, -1, 4), [0.0, math.pi], 1e-7)

    def test_underflow_is_not_a_zero(self) -> None:
        self.assertEqual(_zeros(lambda x: math.exp(-x), -1000, 1000), [])
        self.assert_zeros(_zeros(lambda x: (1 - x) * math.exp(-x), -1000, 1000), [1.0])

    def test_the_number_of_zeros_is_capped(self) -> None:
        found = _zeros(lambda x: math.sin(50 * x), -1000, 1000)
        self.assertLessEqual(len(found), MathUtils._MAX_SCANNED_ZEROS)

    def test_empty_or_infinite_ranges(self) -> None:
        self.assertEqual(_zeros(math.sin, 1, 1), [])
        self.assertEqual(_zeros(math.sin, -math.inf, 1), [])


if __name__ == "__main__":
    unittest.main()
