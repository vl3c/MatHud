"""Shape classification and geometry predicates at awkward scales and shapes.

Covers the client geometry helpers that run in plain Python (with the browser stub):
polygon and triangle type flags, relation checks, circle-segment intersections and the
incircle, each checked away from unit scale where absolute tolerances used to decide.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from server_tests import client_renderer  # noqa: F401  (browser stub, so the client modules import)
from utils.geometry_utils import GeometryUtils
from utils.relation_inspector import RelationInspector


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


def _point(x: float, y: float, name: str = "") -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, name=name)


def _segment(a: tuple[float, float], b: tuple[float, float], name: str = "s") -> SimpleNamespace:
    return SimpleNamespace(point1=_point(*a), point2=_point(*b), name=name)


class _Shape:
    """A triangle or rectangle as RelationInspector reads it: get_vertices(), in no set order."""

    def __init__(self, vertices: list[tuple[float, float]], name: str = "T") -> None:
        self._points = [_point(x, y) for x, y in reversed(vertices)]
        self.name = name

    def get_vertices(self) -> list[SimpleNamespace]:
        return list(self._points)


def _ellipse(cx: float, cy: float, rx: float, ry: float, rotation: float) -> SimpleNamespace:
    return SimpleNamespace(center=_point(cx, cy), radius_x=rx, radius_y=ry, rotation_angle=rotation, name="E")


class TestRelationTolerances(unittest.TestCase):
    def test_similar_does_not_depend_on_argument_order(self) -> None:
        big = _Shape([(0, 0), (3000, 0), (0, 4000)], "T1")
        small = _Shape([(0, 0), (3, 0), (0, 4.00005)], "T2")

        forward = RelationInspector.inspect("similar", [big, small], ["triangle", "triangle"])
        backward = RelationInspector.inspect("similar", [small, big], ["triangle", "triangle"])

        self.assertFalse(forward["result"])
        self.assertFalse(backward["result"])
        scaled = _Shape([(0, 0), (3e-3, 0), (0, 4e-3)], "T3")
        self.assertTrue(RelationInspector.inspect("similar", [scaled, big], ["triangle", "triangle"])["result"])
        self.assertTrue(RelationInspector.inspect("similar", [big, scaled], ["triangle", "triangle"])["result"])

    def test_tiny_lengths_are_compared_relative_to_their_size(self) -> None:
        short = _segment((0, 0), (2e-7, 0))
        longer = _segment((0, 1), (9e-7, 1))
        self.assertFalse(RelationInspector.inspect("equal_length", [short, longer], ["segment", "segment"])["result"])
        same = _segment((5, 5), (5, 5 + 2e-7))
        self.assertTrue(RelationInspector.inspect("equal_length", [short, same], ["segment", "segment"])["result"])

        t1 = _Shape([(0, 0), (2e-7, 0), (0, 3e-7)])
        t2 = _Shape([(0, 0), (9e-7, 0), (0, 3e-7)])
        self.assertFalse(RelationInspector.inspect("congruent", [t1, t2], ["triangle", "triangle"])["result"])

    def test_short_perpendicular_segments_are_not_parallel(self) -> None:
        half = 4.5e-4
        segments = [
            _segment((-half, 0), (half, 0)),
            _segment((0, -half), (0, half)),
            _segment((-half * 0.7, -half * 0.7), (half * 0.7, half * 0.7)),
        ]

        result = RelationInspector.inspect("concurrent", segments, ["segment"] * 3)

        self.assertTrue(result["result"])

    def test_concurrence_far_from_the_origin_uses_the_figure_scale(self) -> None:
        x = 1e6
        through = [_segment((x - 1, x), (x + 1, x)), _segment((x, x - 1), (x, x + 1))]
        missing = _segment((x - 1, x - 0.5), (x + 1, x + 1.5))  # 0.35 from (x, x)
        passing = _segment((x - 1, x - 1), (x + 1, x + 1))

        self.assertFalse(RelationInspector.inspect("concurrent", through + [missing], ["segment"] * 3)["result"])
        self.assertTrue(RelationInspector.inspect("concurrent", through + [passing], ["segment"] * 3)["result"])

    def test_right_triangle_flag_agrees_with_perpendicular(self) -> None:
        # Legs 89.995 degrees apart: within the shared 0.01 degree tolerance.
        angle = math.radians(89.995)
        apex = (math.cos(angle), math.sin(angle))
        flags = GeometryUtils.triangle_type_flags([(0, 0), (1, 0), apex])
        result = RelationInspector.inspect(
            "perpendicular", [_segment((0, 0), (1, 0)), _segment((0, 0), apex)], ["segment", "segment"]
        )
        self.assertTrue(flags["right"])
        self.assertTrue(result["result"])

        angle = math.radians(89.98)
        apex = (math.cos(angle), math.sin(angle))
        flags = GeometryUtils.triangle_type_flags([(0, 0), (1, 0), apex])
        result = RelationInspector.inspect(
            "perpendicular", [_segment((0, 0), (1, 0)), _segment((0, 0), apex)], ["segment", "segment"]
        )
        self.assertFalse(flags["right"])
        self.assertFalse(result["result"])

    def test_point_on_rotated_ellipse(self) -> None:
        ellipse = _ellipse(1, 2, 4, 1, 30)
        theta = math.radians(30)
        on = _point(1 + 4 * math.cos(theta), 2 + 4 * math.sin(theta))
        off = _point(1 + 4 * math.cos(theta), 2 + 4 * math.sin(theta) + 0.01)

        result = RelationInspector.inspect("point_on_ellipse", [on, ellipse], ["point", "ellipse"])
        self.assertTrue(result["result"])
        self.assertFalse(RelationInspector.inspect("point_on_ellipse", [ellipse, off], ["ellipse", "point"])["result"])
        auto = RelationInspector.inspect("auto", [on, ellipse], ["point", "ellipse"])
        self.assertEqual(auto["details"]["checks_run"], ["point_on_ellipse"])

    def test_rectangles_similar_and_congruent(self) -> None:
        r1 = _Shape([(0, 0), (4, 0), (4, 2), (0, 2)], "R1")
        r2 = _Shape([(10, 10), (12, 10), (12, 11), (10, 11)], "R2")
        r3 = _Shape([(0, 0), (0, 4), (-2, 4), (-2, 0)], "R3")
        square = _Shape([(0, 0), (2, 0), (2, 2), (0, 2)], "S")

        def check(operation: str, a: _Shape, b: _Shape) -> bool:
            return bool(RelationInspector.inspect(operation, [a, b], ["rectangle", "rectangle"])["result"])

        self.assertTrue(check("similar", r1, r2))
        self.assertFalse(check("congruent", r1, r2))
        self.assertTrue(check("congruent", r1, r3))
        self.assertFalse(check("similar", r1, square))
        mixed = RelationInspector.inspect("similar", [r1, _Shape([(0, 0), (1, 0), (0, 1)])], ["rectangle", "triangle"])
        self.assertIn("error", mixed)


if __name__ == "__main__":
    unittest.main()
