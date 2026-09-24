"""Regression tests for numeric robustness of shape helpers in MathUtils."""

from __future__ import annotations

import math
import unittest

from drawables.point import Position
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


class TestIsRectangleScale(unittest.TestCase):
    @staticmethod
    def _rotated_rectangle(cx: float, cy: float, w: float, h: float, angle: float) -> list[float]:
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -uy, ux
        coords: list[float] = []
        for i, j in ((0, 0), (1, 0), (1, 1), (0, 1)):
            coords.extend([cx + i * w * ux + j * h * vx, cy + i * w * uy + j * h * vy])
        return coords

    def test_rotated_rectangles_accepted_at_large_scale(self) -> None:
        for scale in (1.0, 200.0, 1000.0, 1e5):
            for k in range(12):
                angle = 0.1 + k * 0.5
                coords = self._rotated_rectangle(scale * 0.3, -scale * 0.7, scale, scale * 0.61, angle)
                self.assertTrue(MathUtils.is_rectangle(*coords), f"scale={scale} angle={angle}")

    def test_rotated_squares_accepted_at_large_scale(self) -> None:
        for scale in (200.0, 1000.0):
            coords = self._rotated_rectangle(17.0, 3.0, scale, scale, 0.7)
            self.assertTrue(MathUtils.is_rectangle(*coords))

    def test_non_rectangles_rejected_at_large_scale(self) -> None:
        # Parallelogram with a 1e-4 rad skew
        self.assertFalse(MathUtils.is_rectangle(0, 0, 1000, 0, 1000.1, 1000, 0.1, 1000))
        # Kite-like quadrilateral
        self.assertFalse(MathUtils.is_rectangle(0, 0, 1000, 0, 1000, 1000, 0, 1001))

    def test_is_right_angle_scale_invariant(self) -> None:
        self.assertTrue(MathUtils.is_right_angle(Position(0, 0), Position(3000.3, 0), Position(0, 4000.7)))
        self.assertFalse(MathUtils.is_right_angle(Position(0, 0), Position(1e-3, 0), Position(1e-6, 1e-3)))


class TestFindDiagonalPoints(unittest.TestCase):
    def test_rotated_square_returns_true_diagonal(self) -> None:
        points = [Position(0, 0), Position(1, 1), Position(0, 2), Position(-1, 1)]
        p1, p2 = MathUtils.find_diagonal_points(points, "R")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertAlmostEqual(math.hypot(p1.x - p2.x, p1.y - p2.y), 2.0)

    def test_rotated_rectangle_returns_true_diagonal(self) -> None:
        points = [Position(0, 0), Position(3, 3), Position(1, 5), Position(-2, 2)]
        p1, p2 = MathUtils.find_diagonal_points(points, "R")
        pair = {(p1.x, p1.y), (p2.x, p2.y)}
        self.assertIn(pair, ({(0, 0), (1, 5)}, {(3, 3), (-2, 2)}))


if __name__ == "__main__":
    unittest.main()
