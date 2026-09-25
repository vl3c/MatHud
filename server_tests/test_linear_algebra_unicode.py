"""Tests for the Unicode notation LinearAlgebraUtils rewrites before math.js sees it.

LinearAlgebraUtils is a Brython module; the browser stub makes it importable and a fake
math.js records the expression and scope each evaluation receives.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
import utils.linear_algebra_utils as linear_algebra_utils_module
from utils.linear_algebra_utils import LinearAlgebraUtils


class _RecordingMath:
    """The parts of math.js LinearAlgebraUtils touches; evaluate records its arguments."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.pi = 3.141592653589793
        self.e = 2.718281828459045

    def matrix(self, value: Any) -> Any:
        return value

    def cross(self, a: Any, b: Any) -> Any:
        return (a, b)

    def norm(self, value: Any) -> Any:
        return value

    inv = transpose = det = norm

    def evaluate(self, expression: str, scope: Dict[str, Any]) -> float:
        self.calls.append((expression, scope))
        return 1.0


class _Window:
    def __init__(self, math: _RecordingMath) -> None:
        self.math = math


VECTORS = [{"name": "u", "value": [1, 0, 0]}, {"name": "v", "value": [0, 1, 0]}]


class TestLinearAlgebraUnicode(unittest.TestCase):
    def setUp(self) -> None:
        self.original_window = linear_algebra_utils_module.window
        self.math = _RecordingMath()
        linear_algebra_utils_module.window = _Window(self.math)

    def tearDown(self) -> None:
        linear_algebra_utils_module.window = self.original_window

    def _evaluated(self, expression: str, objects: List[Dict[str, Any]] = VECTORS) -> str:
        LinearAlgebraUtils.evaluate_expression(objects, expression)
        return self.math.calls[-1][0]

    def test_unicode_notation_is_normalised(self) -> None:
        # Regression: A⁻¹ and u·v reached math.js as-is
        matrix = [{"name": "A", "value": [[1, 2], [3, 4]]}]
        self.assertEqual(self._evaluated("A⁻¹", matrix), "A^(-1)")
        self.assertEqual(self._evaluated("A²", matrix), "A^2")
        self.assertEqual(self._evaluated("u·v"), "u*v")
        self.assertEqual(self._evaluated("2π·u"), "2*pi*u")

    def test_times_between_names_is_the_cross_product(self) -> None:
        cases = {
            "u×v": "cross(u, v)",
            "u × v": "cross(u, v)",
            "u×v + v": "cross(u, v) + v",
            "-(u×v)": "-(cross(u, v))",
            "norm(u×v)": "norm(cross(u, v))",
            "2×3": "2*3",
            "2×u": "2*u",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self._evaluated(expression), expected)

    def test_ambiguous_times_is_an_error(self) -> None:
        # u×v×w, a×b^2 or 2*u×v would change meaning with precedence, so they are refused
        objects = VECTORS + [{"name": "w", "value": [0, 0, 1]}]
        for expression in ("u×v×w", "u×v^2", "2*u×v", "u/v×w", "u×(v+w)"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError) as context:
                    LinearAlgebraUtils.evaluate_expression(objects, expression)
                self.assertIn("cross(a, b)", str(context.exception))

    def test_greek_names_are_validated(self) -> None:
        objects: List[Dict[str, Any]] = [{"name": "α", "value": 2}, {"name": "ϕ", "value": [1, 2]}]
        self.assertEqual(self._evaluated("α*ϕ", objects), "α*φ")
        self.assertIn("φ", self.math.calls[-1][1])
        with self.assertRaises(ValueError) as context:
            LinearAlgebraUtils.evaluate_expression(objects, "α*β")
        self.assertIn("β", str(context.exception))

    def test_ascii_expressions_are_unchanged(self) -> None:
        matrix = [{"name": "A", "value": [[1, 2], [3, 4]]}, {"name": "B", "value": [[1, 0], [0, 1]]}]
        for expression in ("A + B", "inv(A) * B", "A^-1", "transpose(A)*2", "det(A) - 1e-3"):
            with self.subTest(expression=expression):
                self.assertEqual(self._evaluated(expression, matrix), expression)


if __name__ == "__main__":
    unittest.main()
