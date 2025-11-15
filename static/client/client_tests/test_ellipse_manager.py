from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.ellipse import Ellipse
from managers.drawables_container import DrawablesContainer
from managers.ellipse_manager import EllipseManager
from .simple_mock import SimpleMock


class TestEllipseManager(unittest.TestCase):
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
        )

        self.ellipse_manager = EllipseManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_ellipse(self, name: str = "EllipseA", color: str = "#111111") -> Ellipse:
        center = Point(0.0, 0.0, name="A")
        ellipse = Ellipse(center, radius_x=4.0, radius_y=2.0, rotation_angle=10.0, color=color)
        ellipse.name = name
        self.drawables.add(ellipse)
        return ellipse

    def _allow_solitary(self, ellipse: Ellipse) -> None:
        def get_parents(obj):
            if obj is ellipse:
                return {ellipse.center}
            if obj is ellipse.center:
                return {ellipse}
            return set()

        self.dependency_manager.get_parents = get_parents
        self.dependency_manager.get_children = lambda obj: set()

    def test_update_ellipse_changes_all_fields(self) -> None:
        ellipse = self._add_ellipse()
        self._allow_solitary(ellipse)

        result = self.ellipse_manager.update_ellipse(
            "EllipseA",
            new_color="#ffaa00",
            new_radius_x=6.0,
            new_radius_y=3.0,
            new_rotation_angle=35.0,
            new_center_x=5.0,
            new_center_y=-2.0,
        )

        self.assertTrue(result)
        self.assertEqual(ellipse.color, "#ffaa00")
        self.assertEqual(ellipse.radius_x, 6.0)
        self.assertEqual(ellipse.radius_y, 3.0)
        self.assertEqual(ellipse.rotation_angle, 35.0 % 360)
        self.assertEqual(ellipse.center.x, 5.0)
        self.assertEqual(ellipse.center.y, -2.0)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_ellipse_requires_existing(self) -> None:
        with self.assertRaises(ValueError):
            self.ellipse_manager.update_ellipse("missing", new_color="#ff00ff")

    def test_update_ellipse_requires_complete_center_pair(self) -> None:
        ellipse = self._add_ellipse()
        self._allow_solitary(ellipse)

        with self.assertRaises(ValueError):
            self.ellipse_manager.update_ellipse("EllipseA", new_center_x=1.0)

    def test_update_ellipse_rejects_negative_radius(self) -> None:
        ellipse = self._add_ellipse()
        self._allow_solitary(ellipse)

        with self.assertRaises(ValueError):
            self.ellipse_manager.update_ellipse("EllipseA", new_radius_x=-1.0)

    def test_update_ellipse_rejects_center_with_other_parent(self) -> None:
        ellipse = self._add_ellipse()

        other_parent = object()
        self.dependency_manager.get_parents = (
            lambda obj: {ellipse, other_parent} if obj is ellipse.center else set()
        )
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError):
            self.ellipse_manager.update_ellipse("EllipseA", new_center_x=2.0, new_center_y=3.0)

    def test_update_ellipse_rejects_when_not_solitary(self) -> None:
        ellipse = self._add_ellipse()

        other_parent = object()
        self.dependency_manager.get_parents = (
            lambda obj: {other_parent} if obj is ellipse else set()
        )
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError):
            self.ellipse_manager.update_ellipse("EllipseA", new_radius_x=5.0)


if __name__ == "__main__":
    unittest.main()

