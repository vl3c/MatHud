"""Regression tests for numeric robustness of shape helpers in MathUtils."""

from __future__ import annotations

import math
import unittest

from utils.math_utils import MathUtils


class TestEllipseFormula(unittest.TestCase):
    @staticmethod
    def _evaluate_lhs(formula: str, x: float, y: float) -> float:
        lhs, rhs = formula.split("=")
        return float(eval(lhs, {"__builtins__": {}}, {"x": x, "y": y})) - float(rhs)

    def _assert_points_satisfy(self, cx: float, cy: float, rx: float, ry: float, rotation: float) -> None:
        formula = MathUtils.get_ellipse_formula(cx, cy, rx, ry, rotation)
        rot = math.radians(rotation)
        for k in range(12):
            t = k * math.pi / 6
            lx, ly = rx * math.cos(t), ry * math.sin(t)
            x = cx + lx * math.cos(rot) - ly * math.sin(rot)
            y = cy + lx * math.sin(rot) + ly * math.cos(rot)
            self.assertAlmostEqual(self._evaluate_lhs(formula, x, y), 0.0, places=6, msg=formula)

    def test_rotated_formula_uses_center_x_in_cross_term(self) -> None:
        self._assert_points_satisfy(5, -3, 3, 2, 30)

    def test_rotated_formula_keeps_precision_for_large_radii(self) -> None:
        formula = MathUtils.get_ellipse_formula(0, 0, 300, 200, 30)
        self.assertNotIn("0.0*", formula)
        self.assertNotIn("e-", formula)
        self._assert_points_satisfy(0, 0, 300, 200, 30)
        self._assert_points_satisfy(-120.5, 40.25, 300, 200, 75)

    def test_rotated_formula_negative_cross_term(self) -> None:
        formula = MathUtils.get_ellipse_formula(1, 2, 3, 2, 30)
        self.assertIn(" - ", formula)
        self._assert_points_satisfy(1, 2, 3, 2, 30)
        self._assert_points_satisfy(1, 2, 3, 2, 120)


if __name__ == "__main__":
    unittest.main()
