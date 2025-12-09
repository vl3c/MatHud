from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from managers.drawables_container import DrawablesContainer
from managers.segment_manager import SegmentManager
from .simple_mock import SimpleMock


class TestSegmentManager(unittest.TestCase):
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
        self.name_generator = SimpleMock(
            name="NameGeneratorMock",
            split_point_names=lambda expr, count: ["", ""][:count],
        )
        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
            get_children=lambda segment: set(),
            get_all_children=lambda segment: set(),
            get_all_parents=lambda segment: set(),
            remove_drawable=SimpleMock(),
        )
        self.point_manager = SimpleMock(
            name="PointManagerMock",
            create_point=lambda x, y, name="", extra_graphics=True: Point(x, y, name=name),
        )
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
            delete_vector=SimpleMock(),
            angle_manager=None,
        )

        self.segment_manager = SegmentManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_segment(self, name: str = "AB", color: str = "#111111") -> Segment:
        p1 = Point(0.0, 0.0, name="A")
        p2 = Point(1.0, 0.0, name="B")
        segment = Segment(p1, p2, color=color)
        segment.name = name
        self.drawables.add(segment)
        return segment

    def test_update_segment_changes_color_and_draws(self) -> None:
        segment = self._add_segment()

        result = self.segment_manager.update_segment("AB", new_color="#ff00ff")

        self.assertTrue(result)
        self.assertEqual(segment.color, "#ff00ff")
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_segment_changes_label_text_and_visibility(self) -> None:
        segment = self._add_segment()

        result = self.segment_manager.update_segment("AB", new_label_text="mid", new_label_visible=True)

        self.assertTrue(result)
        self.assertEqual(segment.label.text, "mid")
        self.assertTrue(segment.label.visible)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_segment_requires_existing_segment(self) -> None:
        with self.assertRaises(ValueError):
            self.segment_manager.update_segment("missing", new_color="#00ff00")

    def test_update_segment_requires_color_value(self) -> None:
        self._add_segment()
        with self.assertRaises(ValueError):
            self.segment_manager.update_segment("AB", new_color="  ")

    def test_update_segment_requires_at_least_one_property(self) -> None:
        self._add_segment()
        with self.assertRaises(ValueError):
            self.segment_manager.update_segment("AB")


if __name__ == "__main__":
    unittest.main()

