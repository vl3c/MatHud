import unittest
from unittest.mock import MagicMock

from drawables.circle import Circle
from drawables.ellipse import Ellipse
from managers.point_manager import PointManager
from managers.drawables_container import DrawablesContainer
from managers.drawable_dependency_manager import DrawableDependencyManager
from .simple_mock import SimpleMock


class TestPointManagerUpdates(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            undo_redo_manager=SimpleMock(name="UndoRedo", archive=MagicMock()),
            draw_enabled=True,
            draw=MagicMock(),
        )

        self.drawables = DrawablesContainer()
        self.dependency_manager = DrawableDependencyManager()
        self.dependency_manager.remove_drawable = MagicMock()
        self.generated_names: list[str] = []

        def generate_point_name(preferred: str | None) -> str:
            if preferred:
                return preferred
            name = f"P{len(self.generated_names)}"
            self.generated_names.append(name)
            return name

        self.name_generator = SimpleMock(
            name="NameGeneratorMock",
            generate_point_name=generate_point_name,
            filter_string=lambda value: value,
        )

        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            segment_manager=SimpleMock(
                name="SegmentManagerMock", _split_segments_with_point=MagicMock()
            ),
            create_drawables_from_new_connections=MagicMock(),
            delete_ellipse=MagicMock(),
            delete_circle=MagicMock(),
            delete_vector=MagicMock(),
            delete_segment=MagicMock(),
        )

        self.point_manager = PointManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def test_update_point_allows_solitary_edits(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)

        result = self.point_manager.update_point(
            "A", new_name="B", new_x=5.0, new_y=6.0, new_color="#123456"
        )

        self.assertTrue(result)
        self.assertEqual(point.name, "B")
        self.assertEqual(point.x, 5.0)
        self.assertEqual(point.y, 6.0)
        self.assertEqual(point.color, "#123456")
        self.canvas.undo_redo_manager.archive.assert_called()
        self.canvas.draw.assert_called()

    def test_update_point_allows_rename_only(self) -> None:
        point = self.point_manager.create_point(1, 2, "A", extra_graphics=False)

        result = self.point_manager.update_point("A", new_name="B")

        self.assertTrue(result)
        self.assertEqual(point.name, "B")
        self.assertEqual((point.x, point.y), (1.0, 2.0))

    def test_update_point_allows_move_only(self) -> None:
        point = self.point_manager.create_point(1, 2, "A", extra_graphics=False)

        result = self.point_manager.update_point("A", new_x=3.0, new_y=4.5)

        self.assertTrue(result)
        self.assertEqual(point.name, "A")
        self.assertEqual((point.x, point.y), (3.0, 4.5))

    def test_update_point_allows_color_only(self) -> None:
        point = self.point_manager.create_point(1, 2, "A", extra_graphics=False)

        result = self.point_manager.update_point("A", new_color="#ff0000")

        self.assertTrue(result)
        self.assertEqual(point.name, "A")
        self.assertEqual((point.x, point.y), (1.0, 2.0))
        self.assertEqual(point.color, "#ff0000")

    def test_update_point_rejects_dependencies(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        dependent_segment = SimpleMock(name="segment", get_class_name=lambda: "Segment")
        self.dependency_manager.register_dependency(dependent_segment, point)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_color="blue")

    def test_update_point_rejects_rename_when_dependent(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        dependent_segment = SimpleMock(name="segment", get_class_name=lambda: "Segment")
        self.dependency_manager.register_dependency(dependent_segment, point)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_name="B")

    def test_update_point_rejects_move_when_dependent(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        dependent_segment = SimpleMock(name="segment", get_class_name=lambda: "Segment")
        self.dependency_manager.register_dependency(dependent_segment, point)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_x=1.0, new_y=2.0)

    def test_update_point_rejects_when_point_is_circle_center(self) -> None:
        point = self.point_manager.create_point(0, 0, "C", extra_graphics=False)
        circle = Circle(point, radius=5.0)
        circle.name = "circle_C"
        self.drawables.add(circle)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("C", new_x=10.0, new_y=20.0)

    def test_update_point_rename_updates_circle_name(self) -> None:
        point = self.point_manager.create_point(0, 0, "C", extra_graphics=False)
        circle = Circle(point, radius=5.0)
        self.drawables.add(circle)

        result = self.point_manager.update_point("C", new_name="D")

        self.assertTrue(result)
        self.assertEqual(point.name, "D")
        self.assertEqual(circle.name, "D(5)")

    def test_update_point_rejects_when_point_is_ellipse_center(self) -> None:
        point = self.point_manager.create_point(0, 0, "E", extra_graphics=False)
        ellipse = Ellipse(point, radius_x=3.0, radius_y=2.0)
        ellipse.name = "ellipse_E"
        self.drawables.add(ellipse)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("E", new_x=4.0, new_y=5.0)

    def test_update_point_rename_updates_ellipse_name(self) -> None:
        point = self.point_manager.create_point(0, 0, "E", extra_graphics=False)
        ellipse = Ellipse(point, radius_x=3.0, radius_y=2.0)
        self.drawables.add(ellipse)

        result = self.point_manager.update_point("E", new_name="F")

        self.assertTrue(result)
        self.assertEqual(point.name, "F")
        self.assertEqual(ellipse.name, "F(3, 2)")

    def test_update_point_rejects_combined_edit_when_dependent(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        dependent_segment = SimpleMock(name="segment", get_class_name=lambda: "Segment")
        self.dependency_manager.register_dependency(dependent_segment, point)

        with self.assertRaises(ValueError):
            self.point_manager.update_point(
                "A", new_name="B", new_x=3.0, new_y=4.0, new_color="#00ff00"
            )

    def test_update_point_rename_conflict_preserves_original_name(self) -> None:
        self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        point_b = self.point_manager.create_point(2, 2, "B", extra_graphics=False)

        with self.assertRaises(ValueError) as context:
            self.point_manager.update_point("B", new_name="A")

        self.assertEqual("Another point named 'A' already exists.", str(context.exception))
        self.assertEqual(point_b.name, "B")

    def test_update_point_requires_full_coordinates(self) -> None:
        self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_x=1.0)

    def test_update_point_name_collision(self) -> None:
        self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        self.point_manager.create_point(1, 1, "B", extra_graphics=False)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_name="B")

    def test_update_point_coordinate_collision(self) -> None:
        point_a = self.point_manager.create_point(0, 0, "A", extra_graphics=False)
        point_b = self.point_manager.create_point(2, 2, "B", extra_graphics=False)

        with self.assertRaises(ValueError):
            self.point_manager.update_point("A", new_x=point_b.x, new_y=point_b.y)

    def test_delete_point_removes_dependency_entry(self) -> None:
        point = self.point_manager.create_point(0, 0, "A", extra_graphics=False)

        removed = self.point_manager.delete_point(point.x, point.y)

        self.assertTrue(removed)
        self.dependency_manager.remove_drawable.assert_called_with(point)


if __name__ == "__main__":
    unittest.main()
