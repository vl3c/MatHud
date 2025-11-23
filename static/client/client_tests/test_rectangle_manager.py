from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.rectangle import Rectangle
from managers.drawables_container import DrawablesContainer
from managers.rectangle_manager import RectangleManager
from .simple_mock import SimpleMock


class TestRectangleManager(unittest.TestCase):
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

        self.rectangle_manager = RectangleManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            segment_manager=self.segment_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_rectangle(self, name: str = "ABCD", color: str = "#111111") -> Rectangle:
        a = Point(0.0, 0.0, name="A")
        b = Point(2.0, 0.0, name="B")
        c = Point(2.0, 1.0, name="C")
        d = Point(0.0, 1.0, name="D")
        s1 = Segment(a, b, color="#222222")
        s2 = Segment(b, c, color="#222222")
        s3 = Segment(c, d, color="#222222")
        s4 = Segment(d, a, color="#222222")
        rectangle = Rectangle(s1, s2, s3, s4, color=color)
        rectangle.name = name
        self.drawables.add(rectangle)
        return rectangle

    def test_update_rectangle_changes_color_and_draws(self) -> None:
        rectangle = self._add_rectangle()
        self.assertFalse(rectangle.is_renderable)

        result = self.rectangle_manager.update_rectangle("ABCD", new_color="#00aaff")

        self.assertTrue(result)
        self.assertEqual(rectangle.color, "#00aaff")
        self.assertEqual(rectangle.segment1.color, "#00aaff")
        self.assertEqual(rectangle.segment2.color, "#00aaff")
        self.assertEqual(rectangle.segment3.color, "#00aaff")
        self.assertEqual(rectangle.segment4.color, "#00aaff")
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_rectangle_requires_existing_rectangle(self) -> None:
        with self.assertRaises(ValueError):
            self.rectangle_manager.update_rectangle("missing", new_color="#00ff00")

    def test_update_rectangle_requires_color_value(self) -> None:
        self._add_rectangle()
        with self.assertRaises(ValueError):
            self.rectangle_manager.update_rectangle("ABCD", new_color="  ")

    def test_update_rectangle_requires_properties(self) -> None:
        self._add_rectangle()
        with self.assertRaises(ValueError):
            self.rectangle_manager.update_rectangle("ABCD")

if __name__ == "__main__":
    unittest.main()

