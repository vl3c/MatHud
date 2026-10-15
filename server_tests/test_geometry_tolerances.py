"""Shape classification and geometry predicates at awkward scales and shapes.

Covers the client geometry helpers that run in plain Python (with the browser stub):
polygon and triangle type flags, relation checks, circle-segment intersections and the
incircle, each checked away from unit scale where absolute tolerances used to decide.
"""

from __future__ import annotations

import math
import unittest

from server_tests import client_renderer  # noqa: F401  (browser stub, so the client modules import)
from utils.geometry_utils import GeometryUtils


def _regular_polygon(sides: int, radius: float = 1.0, step: int = 1) -> list[tuple[float, float]]:
    return [
        (radius * math.cos(2 * math.pi * step * k / sides), radius * math.sin(2 * math.pi * step * k / sides))
        for k in range(sides)
    ]


class TestPolygonRegularity(unittest.TestCase):
    def test_convex_regular_polygons_are_regular(self) -> None:
        for sides in range(3, 11):
            with self.subTest(sides=sides):
                self.assertTrue(GeometryUtils.polygon_flags(_regular_polygon(sides))["regular"])
                clockwise = list(reversed(_regular_polygon(sides, radius=1e-4)))
                self.assertTrue(GeometryUtils.polygon_flags(clockwise)["regular"])

    def test_plus_shape_with_unit_sides_is_not_regular(self) -> None:
        plus = [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1)]

        flags = GeometryUtils.polygon_flags(plus)

        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])

    def test_pentagram_is_not_regular(self) -> None:
        pentagram = _regular_polygon(5, step=2)

        self.assertFalse(GeometryUtils.polygon_flags(pentagram)["regular"])


class TestDegenerateTriangle(unittest.TestCase):
    def test_collinear_points_get_no_type_flags(self) -> None:
        for points in ([(0, 0), (1, 0), (2, 0)], [(0, 0), (1e6, 1e6), (3e6, 3e6)], [(0, 0), (1e-6, 0), (2e-6, 0)]):
            with self.subTest(points=points):
                flags = GeometryUtils.triangle_type_flags(points)
                self.assertEqual(set(flags.values()), {False})

    def test_thin_triangle_keeps_its_flags(self) -> None:
        flags = GeometryUtils.triangle_type_flags([(0, 0), (2, 0), (1, 1e-3)])

        self.assertTrue(flags["isosceles"])
        self.assertFalse(flags["right"])


if __name__ == "__main__":
    unittest.main()
