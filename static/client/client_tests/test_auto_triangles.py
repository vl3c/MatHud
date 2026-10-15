"""Triangles created automatically from segments that close them (PolygonManager.create_triangles_from_segments)."""

from __future__ import annotations

import unittest
from itertools import combinations
from typing import List, Tuple

from canvas import Canvas
from utils.geometry_utils import GeometryUtils
from utils.math_utils import MathUtils


def _brute_force_triangles(segments: List) -> List[List[str]]:
    """The point names of each triangle, found by checking every triple of segments in order."""
    found: List[List[str]] = []
    for s1, s2, s3 in combinations(segments, 3):
        names = GeometryUtils.get_unique_point_names_from_segments([s1, s2, s3])
        if len(names) == 3 and GeometryUtils.is_fully_connected_graph(names, segments) and names not in found:
            found.append(names)
    return found


class TestAutoTriangles(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _segment(self, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        self.canvas.create_segment(start[0], start[1], end[0], end[1], extra_graphics=False)

    def _connect(self) -> List[List[str]]:
        self.canvas.drawable_manager.create_drawables_from_new_connections()
        triangles = self.canvas.get_drawables_by_class_name("Triangle")
        return [sorted(vertex.name for vertex in triangle.get_vertices()) for triangle in triangles]

    def test_square_with_diagonals_gives_four_triangles(self) -> None:
        a, b, c, d = (0, 0), (4, 0), (4, 3), (0, 3)
        for start, end in ((a, b), (b, c), (c, d), (d, a), (a, c), (b, d)):
            self._segment(start, end)
        self.assertEqual(len(self._connect()), 4)

    def test_collinear_points_and_open_paths_give_none(self) -> None:
        self._segment((0, 0), (1, 0))
        self._segment((1, 0), (2, 0))
        self._segment((0, 0), (2, 0))
        self._segment((5, 5), (6, 7))
        self._segment((6, 7), (8, 5))
        self.assertEqual(self._connect(), [])

    def test_matches_a_scan_of_every_segment_triple_in_order(self) -> None:
        points = [(0, 0), (4, 0), (4, 3), (0, 3), (2, 6), (7, 2), (6, -3), (-3, 1)]
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (3, 4), (4, 2), (1, 5), (5, 2), (5, 6), (6, 1), (0, 7), (7, 3)]
        for i, j in edges:
            self._segment(points[i], points[j])
        segments = list(self.canvas.drawable_manager.drawables.Segments)
        expected = [
            names
            for names in _brute_force_triangles(segments)
            if MathUtils.points_orientation(*self._coords(names)) != 0
        ]
        self.assertGreater(len(expected), 3)
        self.assertEqual(self._connect(), expected)

    def _coords(self, names: List[str]) -> Tuple[float, ...]:
        coords: List[float] = []
        for name in names:
            point = self.canvas.get_point_by_name(name)
            coords.extend([point.x, point.y])
        return tuple(coords)


if __name__ == "__main__":
    unittest.main()
