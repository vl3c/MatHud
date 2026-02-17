from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.circle import Circle
from managers.drawables_container import DrawablesContainer
from managers.circle_manager import CircleManager
from .simple_mock import SimpleMock


class TestCircleManager(unittest.TestCase):
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
        self.point_manager = SimpleMock(name="PointManagerMock")
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
        )

        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
            remove_drawable=SimpleMock(),
        )

        self.circle_manager = CircleManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_circle(self, name: str = "CircleA", color: str = "#111111") -> Circle:
        center = Point(0.0, 0.0, name="A")
        circle = Circle(center, radius=5.0, color=color)
        circle.name = name
        self.drawables.add(circle)
        return circle

    def test_update_circle_changes_color_and_center(self) -> None:
        circle = self._add_circle()

        self.dependency_manager.get_parents = lambda obj: {circle} if obj is circle.center else set()
        self.dependency_manager.get_children = lambda obj: set()

        result = self.circle_manager.update_circle(
            "CircleA",
            new_color="#ffaa00",
            new_center_x=3.0,
            new_center_y=4.0,
        )

        self.assertTrue(result)
        self.assertEqual(circle.color, "#ffaa00")
        self.assertEqual(circle.center.x, 3.0)
        self.assertEqual(circle.center.y, 4.0)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_circle_requires_existing_circle(self) -> None:
        with self.assertRaises(ValueError):
            self.circle_manager.update_circle("missing", new_color="#00ff00")

    def test_update_circle_requires_complete_center_pair(self) -> None:
        circle = self._add_circle()
        self.dependency_manager.get_parents = lambda obj: {circle} if obj is circle.center else set()
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError):
            self.circle_manager.update_circle("CircleA", new_center_x=3.0)

    def test_update_circle_rejects_non_solitary_center(self) -> None:
        other_parent = object()
        circle = self._add_circle()
        self.dependency_manager.get_parents = lambda obj: {circle, other_parent} if obj is circle.center else set()
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError):
            self.circle_manager.update_circle("CircleA", new_center_x=1.0, new_center_y=2.0)

    def test_update_circle_rejects_center_with_other_child(self) -> None:
        other_child = object()
        circle = self._add_circle()
        self.dependency_manager.get_parents = lambda obj: {circle} if obj is circle.center else set()
        self.dependency_manager.get_children = lambda obj: {circle, other_child} if obj is circle.center else set()

        with self.assertRaises(ValueError):
            self.circle_manager.update_circle("CircleA", new_center_x=5.0, new_center_y=6.0)

    def test_update_circle_requires_properties(self) -> None:
        self._add_circle()
        with self.assertRaises(ValueError):
            self.circle_manager.update_circle("CircleA")

    def test_delete_circle_removes_dependency_entry(self) -> None:
        circle = self._add_circle(name="CircleA")

        removed = self.circle_manager.delete_circle("CircleA")

        self.assertTrue(removed)
        self.dependency_manager.remove_drawable.assert_called_once_with(circle)


if __name__ == "__main__":
    unittest.main()
