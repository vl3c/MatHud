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
            remove_drawable=SimpleMock(),
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
        self.dependency_manager.get_parents = lambda obj: {ellipse, other_parent} if obj is ellipse.center else set()
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError) as context:
            self.ellipse_manager.update_ellipse("EllipseA", new_center_x=2.0, new_center_y=3.0)
        message = str(context.exception)
        self.assertIn("cannot move its center", message)
        offset = (2.0 - ellipse.center.x, 3.0 - ellipse.center.y)
        self.assertIn(f"use translate_object with name '{ellipse.center.name}', x_offset {offset[0]:.12g}", message)

    def test_update_ellipse_rejects_when_not_solitary(self) -> None:
        ellipse = self._add_ellipse()

        other_parent = object()
        self.dependency_manager.get_parents = lambda obj: {other_parent} if obj is ellipse else set()
        self.dependency_manager.get_children = lambda obj: set()

        with self.assertRaises(ValueError) as context:
            self.ellipse_manager.update_ellipse("EllipseA", new_radius_x=5.0)
        # No working alternative for a radius: no move hint.
        self.assertNotIn("translate_object", str(context.exception))

    def test_delete_ellipse_removes_dependency_entry(self) -> None:
        ellipse = self._add_ellipse(name="EllipseA")

        removed = self.ellipse_manager.delete_ellipse("EllipseA")

        self.assertTrue(removed)
        self.dependency_manager.remove_drawable.assert_called_once_with(ellipse)


class TestEllipseReuseMatching(unittest.TestCase):
    """create_ellipse reuses an existing ellipse only when its orientation matches too."""

    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            draw_enabled=False,
            draw=SimpleMock(),
            undo_redo_manager=SimpleMock(name="UndoRedoMock", archive=SimpleMock()),
        )
        self.drawables = DrawablesContainer()
        self.center = Point(0.0, 0.0, name="A")
        self.ellipse_manager = EllipseManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=SimpleMock(name="NameGeneratorMock", split_point_names=lambda name, n: ["A"]),
            dependency_manager=SimpleMock(
                name="DependencyManagerMock",
                analyze_drawable_for_dependencies=SimpleMock(),
            ),
            point_manager=SimpleMock(name="PointManagerMock", create_point=lambda *args, **kwargs: self.center),
            drawable_manager_proxy=SimpleMock(name="DrawableManagerProxyMock"),
        )

    def _create(self, radius_x: float, radius_y: float, rotation_angle: float) -> Ellipse:
        return self.ellipse_manager.create_ellipse(0.0, 0.0, radius_x, radius_y, rotation_angle, extra_graphics=False)

    def test_same_centre_radii_and_rotation_reuses(self) -> None:
        first = self._create(4.0, 2.0, 30.0)
        second = self._create(4.0, 2.0, 30.0)

        self.assertIs(first, second)
        self.assertEqual(len(self.drawables.Ellipses), 1)

    def test_different_rotation_creates_new_ellipse(self) -> None:
        first = self._create(4.0, 2.0, 0.0)
        second = self._create(4.0, 2.0, 45.0)

        self.assertIsNot(first, second)
        self.assertEqual(second.rotation_angle, 45.0)
        self.assertEqual(len(self.drawables.Ellipses), 2)

    def test_rotation_differing_by_half_turn_reuses(self) -> None:
        first = self._create(4.0, 2.0, 20.0)

        self.assertIs(self._create(4.0, 2.0, 200.0), first)
        self.assertIs(self._create(4.0, 2.0, -160.0), first)
        self.assertEqual(len(self.drawables.Ellipses), 1)

    def test_rotation_near_half_turn_boundary_reuses(self) -> None:
        first = self._create(4.0, 2.0, 179.9999999)

        self.assertIs(self._create(4.0, 2.0, 0.0), first)

    def test_equal_radii_ignore_rotation(self) -> None:
        first = self._create(3.0, 3.0, 0.0)

        self.assertIs(self._create(3.0, 3.0, 73.0), first)
        self.assertEqual(len(self.drawables.Ellipses), 1)

    def test_get_ellipse_without_rotation_matches_any_orientation(self) -> None:
        first = self._create(4.0, 2.0, 30.0)

        self.assertIs(self.ellipse_manager.get_ellipse(0.0, 0.0, 4.0, 2.0), first)
        self.assertIsNone(self.ellipse_manager.get_ellipse(0.0, 0.0, 4.0, 2.0, 60.0))


if __name__ == "__main__":
    unittest.main()
