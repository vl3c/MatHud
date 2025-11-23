from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.triangle import Triangle
from managers.drawables_container import DrawablesContainer
from managers.polygon_manager import PolygonManager
from managers.triangle_manager import TriangleManager
from .simple_mock import SimpleMock


class TestTriangleManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            draw_enabled=True,
            draw=SimpleMock(),
            undo_redo_manager=SimpleMock(
                name="UndoRedoMock",
                archive=SimpleMock(),
            ),
        )

        self.drawables = DrawablesContainer()
        self.name_generator = SimpleMock(name="NameGeneratorMock")
        self.name_generator.split_point_names = SimpleMock(side_effect=lambda _name, count: [""] * count)
        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
        )
        self.point_manager = SimpleMock(name="PointManagerMock")
        self.segment_manager = SimpleMock(name="SegmentManagerMock")
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
        )
        self.polygon_manager = PolygonManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            segment_manager=self.segment_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )
        self.drawable_manager_proxy.polygon_manager = self.polygon_manager

        self.triangle_manager = TriangleManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            segment_manager=self.segment_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_triangle(self, name: str = "ABC", color: str = "#111111") -> Triangle:
        a = Point(0.0, 0.0, name="A")
        b = Point(1.0, 0.0, name="B")
        c = Point(0.0, 1.0, name="C")
        s1 = Segment(a, b, color="#222222")
        s2 = Segment(b, c, color="#222222")
        s3 = Segment(c, a, color="#222222")
        triangle = Triangle(s1, s2, s3, color=color)
        triangle.name = name
        self.drawables.add(triangle)
        return triangle

    def test_update_triangle_changes_color_and_draws(self) -> None:
        triangle = self._add_triangle()
        self.assertFalse(triangle.is_renderable)

        result = self.triangle_manager.update_triangle("ABC", new_color="#ff00ff")

        self.assertTrue(result)
        self.assertEqual(triangle.color, "#ff00ff")
        self.assertEqual(triangle.segment1.color, "#ff00ff")
        self.assertEqual(triangle.segment2.color, "#ff00ff")
        self.assertEqual(triangle.segment3.color, "#ff00ff")
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_triangle_requires_existing_triangle(self) -> None:
        with self.assertRaises(ValueError):
            self.triangle_manager.update_triangle("missing", new_color="#00ff00")

    def test_update_triangle_requires_color_value(self) -> None:
        self._add_triangle()
        with self.assertRaises(ValueError):
            self.triangle_manager.update_triangle("ABC", new_color="  ")

    def test_update_triangle_requires_properties(self) -> None:
        self._add_triangle()
        with self.assertRaises(ValueError):
            self.triangle_manager.update_triangle("ABC")

if __name__ == "__main__":
    unittest.main()

