from __future__ import annotations

import math
import unittest

from utils.polygon_canonicalizer import (
    PolygonCanonicalizationError,
    canonicalize_rectangle,
    canonicalize_triangle,
)


class TestPolygonCanonicalizer(unittest.TestCase):
    @staticmethod
    def _side_lengths(vertices: list[tuple[float, float]]) -> list[float]:
        lengths: list[float] = []
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            lengths.append(math.hypot(x2 - x1, y2 - y1))
        return lengths

    @staticmethod
    def _orientation(vertices: list[tuple[float, float]]) -> float:
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    def test_rectangle_vertices_axis_aligned(self) -> None:
        vertices = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (0.0, 2.0),
        ]
        result = canonicalize_rectangle(vertices)
        expected = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (0.0, 2.0),
        ]
        self.assertEqual(len(result), 4)
        for (rx, ry), (ex, ey) in zip(result, expected):
            self.assertAlmostEqual(rx, ex)
            self.assertAlmostEqual(ry, ey)

    def test_rectangle_vertices_rotated(self) -> None:
        # Rectangle centered at origin, rotated by 30 degrees
        angle = math.radians(30)
        width = 6.0
        height = 2.0
        ux = math.cos(angle)
        uy = math.sin(angle)
        vx = -math.sin(angle)
        vy = math.cos(angle)
        half_w = width / 2.0
        half_h = height / 2.0
        vertices = [
            (ux * -half_w + vx * -half_h, uy * -half_w + vy * -half_h),
            (ux * half_w + vx * -half_h, uy * half_w + vy * -half_h),
            (ux * half_w + vx * half_h, uy * half_w + vy * half_h),
            (ux * -half_w + vx * half_h, uy * -half_w + vy * half_h),
        ]
        # Introduce slight perturbation to simulate imperfect input
        noisy_vertices = [(x + 0.001, y - 0.001) for x, y in vertices]
        result = canonicalize_rectangle(noisy_vertices, tolerance=1e-3)
        self.assertEqual(len(result), 4)
        # Ensure the reconstructed rectangle has expected width/height
        lengths = []
        for i in range(4):
            x1, y1 = result[i]
            x2, y2 = result[(i + 1) % 4]
            lengths.append(math.hypot(x2 - x1, y2 - y1))
        lengths.sort()
        self.assertAlmostEqual(lengths[0], height, places=3)
        self.assertAlmostEqual(lengths[2], width, places=3)
        # Ensure the first corner is close to the first supplied vertex
        distance = math.hypot(
            result[0][0] - vertices[0][0],
            result[0][1] - vertices[0][1],
        )
        self.assertLess(distance, 0.01)

    def test_rectangle_vertices_requires_four_distinct_points(self) -> None:
        vertices = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 0.0),  # duplicate
            (0.0, 2.0),
        ]
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_rectangle(vertices)

    def test_rectangle_vertices_prioritize_first_diagonal(self) -> None:
        # Provide vertices in a noisy order; first and third define diagonal
        vertices = [
            (0.0, 0.0),
            (5.1, 0.0),
            (4.5, 3.9),
            (0.0, 4.0),
        ]
        result = canonicalize_rectangle(vertices, tolerance=0.2)
        for corner in result:
            if math.isclose(corner[0], vertices[0][0], abs_tol=0.1) and math.isclose(corner[1], vertices[0][1], abs_tol=0.1):
                break
        else:
            self.fail("Canonical rectangle did not preserve proximity to first vertex along the diagonal.")
        for corner in result:
            if math.isclose(corner[0], vertices[2][0], abs_tol=0.1) and math.isclose(corner[1], vertices[2][1], abs_tol=0.1):
                break
        else:
            self.fail("Canonical rectangle did not preserve proximity to third vertex along the diagonal.")

    def test_rectangle_diagonal_mode(self) -> None:
        diagonal = [(1.0, 1.0), (5.0, 4.0)]
        result = canonicalize_rectangle(diagonal, construction_mode="diagonal")
        expected = [
            (1.0, 1.0),
            (5.0, 1.0),
            (5.0, 4.0),
            (1.0, 4.0),
        ]
        self.assertEqual(result, expected)

    def test_rectangle_diagonal_requires_two_points(self) -> None:
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_rectangle([(0.0, 0.0)], construction_mode="diagonal")

    def test_rectangle_diagonal_rejects_shared_axis(self) -> None:
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_rectangle([(0.0, 0.0), (0.0, 4.0)], construction_mode="diagonal")

    def test_rectangle_diagonal_user_example(self) -> None:
        result = canonicalize_rectangle(
            [(97.0, 176.0), (144.0, 43.5)],
            construction_mode="diagonal",
        )
        expected = [
            (97.0, 176.0),
            (144.0, 176.0),
            (144.0, 43.5),
            (97.0, 43.5),
        ]
        self.assertEqual(expected, result)

    def test_rectangle_vertices_user_example(self) -> None:
        source_vertices = [
            (96.0, 176.0),
            (157.0, 164.0),
            (117.0, 31.5),
            (64.0, 51.5),
        ]
        result = canonicalize_rectangle(
            source_vertices,
            construction_mode="vertices",
            tolerance=1e-3,
        )
        expected = [
            (96.0, 176.0),
            (54.598627536401736, 52.40255569753407),
            (117.0, 31.5),
            (158.40137246359825, 155.09744430246596),
        ]
        self.assertEqual(len(expected), len(result))
        for actual, target in zip(result, expected):
            self.assertAlmostEqual(actual[0], target[0], places=6)
            self.assertAlmostEqual(actual[1], target[1], places=6)
        self.assertAlmostEqual(result[0][0], source_vertices[0][0])
        self.assertAlmostEqual(result[0][1], source_vertices[0][1])
        self.assertAlmostEqual(result[2][0], source_vertices[2][0])
        self.assertAlmostEqual(result[2][1], source_vertices[2][1])

    def test_rectangle_vertices_invalid(self) -> None:
        # Trapezoid should fall back to best-fit rectangle without raising
        vertices = [
            (0.0, 0.0),
            (3.0, 0.5),
            (4.0, 2.0),
            (0.5, 2.0),
        ]
        result = canonicalize_rectangle(vertices, tolerance=1e-3)
        self.assertEqual(len(result), 4)
        # Ensure diagonal anchors preserved
        anchor_pairs = []
        for anchor in (vertices[0], vertices[2]):
            anchor_match = min(result, key=lambda corner: math.hypot(corner[0] - anchor[0], corner[1] - anchor[1]))
            anchor_pairs.append(anchor_match)
        self.assertAlmostEqual(anchor_pairs[0][0], vertices[0][0], places=1)
        self.assertAlmostEqual(anchor_pairs[0][1], vertices[0][1], places=1)
        self.assertAlmostEqual(anchor_pairs[1][0], vertices[2][0], places=1)
        self.assertAlmostEqual(anchor_pairs[1][1], vertices[2][1], places=1)

    def test_rectangle_invalid_construction_mode(self) -> None:
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_rectangle([(0.0, 0.0), (1.0, 1.0)], construction_mode="unknown")

    def test_triangle_vertices_basic(self) -> None:
        vertices = [
            (0.0, 0.0),
            (4.0, 0.2),
            (1.5, 3.2),
        ]
        result = canonicalize_triangle(vertices)
        self.assertEqual(len(result), 3)
        self.assertGreater(abs(self._orientation(result)), 1e-6)
        distance = math.hypot(result[0][0] - vertices[0][0], result[0][1] - vertices[0][1])
        self.assertLess(distance, 0.5)

    def test_triangle_equilateral_subtype(self) -> None:
        vertices = [
            (0.0, 0.0),
            (2.0, 0.01),
            (1.02, 1.732),
        ]
        result = canonicalize_triangle(vertices, subtype="equilateral")
        lengths = sorted(self._side_lengths(result))
        self.assertAlmostEqual(lengths[0], lengths[1], places=6)
        self.assertAlmostEqual(lengths[1], lengths[2], places=6)

    def test_triangle_isosceles_subtype(self) -> None:
        vertices = [
            (0.0, 0.0),
            (3.0, 0.3),
            (1.5, 4.1),
        ]
        result = canonicalize_triangle(vertices, subtype="isosceles")
        lengths = self._side_lengths(result)
        equal_legs = sorted([lengths[0], lengths[2]])
        self.assertAlmostEqual(equal_legs[0], equal_legs[1], places=6)

        midpoint = (
            (result[1][0] + result[2][0]) / 2.0,
            (result[1][1] + result[2][1]) / 2.0,
        )
        apex_vector = (result[0][0] - midpoint[0], result[0][1] - midpoint[1])
        self.assertGreater(math.hypot(*apex_vector), 0.0)

    def test_triangle_right_subtype(self) -> None:
        vertices = [
            (0.0, 0.0),
            (4.1, -0.05),
            (0.05, 3.9),
        ]
        result = canonicalize_triangle(vertices, subtype="right")
        legs = [
            (result[1][0] - result[0][0], result[1][1] - result[0][1]),
            (result[2][0] - result[0][0], result[2][1] - result[0][1]),
        ]
        dot = legs[0][0] * legs[1][0] + legs[0][1] * legs[1][1]
        self.assertAlmostEqual(dot, 0.0, places=6)

    def test_triangle_right_isosceles_subtype(self) -> None:
        vertices = [
            (1.0, 1.0),
            (4.0, 1.2),
            (1.2, 4.0),
        ]
        result = canonicalize_triangle(vertices, subtype="right_isosceles")
        lengths = self._side_lengths(result)
        self.assertAlmostEqual(lengths[0], lengths[2], places=6)
        legs = [
            (result[1][0] - result[0][0], result[1][1] - result[0][1]),
            (result[2][0] - result[0][0], result[2][1] - result[0][1]),
        ]
        dot = legs[0][0] * legs[1][0] + legs[0][1] * legs[1][1]
        self.assertAlmostEqual(dot, 0.0, places=6)

    def test_triangle_invalid_vertices(self) -> None:
        vertices = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 0.0),
        ]
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_triangle(vertices)

    def test_triangle_invalid_subtype(self) -> None:
        vertices = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ]
        with self.assertRaises(PolygonCanonicalizationError):
            canonicalize_triangle(vertices, subtype="scalene")


if __name__ == "__main__":
    unittest.main()

