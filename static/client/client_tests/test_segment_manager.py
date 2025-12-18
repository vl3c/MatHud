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

    def test_create_segment_from_points_creates_new_segment(self) -> None:
        """Test that create_segment_from_points creates a segment with the given Point objects."""
        p1 = Point(0.0, 0.0, name="X")
        p2 = Point(1.0, 0.0, name="Y")
        
        segment = self.segment_manager.create_segment_from_points(p1, p2)
        
        self.assertIs(segment.point1, p1)
        self.assertIs(segment.point2, p2)
        self.assertEqual(segment.name, "XY")

    def test_create_segment_from_points_returns_existing_if_same_points(self) -> None:
        """Test that it returns existing segment if it references the same Point objects."""
        p1 = Point(0.0, 0.0, name="A")
        p2 = Point(1.0, 0.0, name="B")
        
        segment1 = self.segment_manager.create_segment_from_points(p1, p2)
        segment2 = self.segment_manager.create_segment_from_points(p1, p2)
        
        self.assertIs(segment1, segment2)

    def test_create_segment_from_points_returns_existing_if_same_points_reversed(self) -> None:
        """Test that it returns existing segment if points are reversed but same objects."""
        p1 = Point(0.0, 0.0, name="A")
        p2 = Point(1.0, 0.0, name="B")
        
        segment1 = self.segment_manager.create_segment_from_points(p1, p2)
        segment2 = self.segment_manager.create_segment_from_points(p2, p1)
        
        self.assertIs(segment1, segment2)

    def test_create_segment_from_points_creates_new_if_different_points_same_coords(self) -> None:
        """Test that it creates a new segment if existing segment has different Point objects."""
        old_p1 = Point(0.0, 0.0, name="A")
        old_p2 = Point(1.0, 0.0, name="B")
        old_segment = Segment(old_p1, old_p2)
        self.drawables.add(old_segment)
        
        new_p1 = Point(0.0, 0.0, name="X")
        new_p2 = Point(1.0, 0.0, name="Y")
        
        new_segment = self.segment_manager.create_segment_from_points(new_p1, new_p2)
        
        self.assertIsNot(new_segment, old_segment)
        self.assertIs(new_segment.point1, new_p1)
        self.assertIs(new_segment.point2, new_p2)
        self.assertEqual(new_segment.name, "XY")

    def test_create_segment_from_points_removes_stale_segment(self) -> None:
        """Test that stale segment is removed when new one with same coords is created."""
        old_p1 = Point(0.0, 0.0, name="A")
        old_p2 = Point(1.0, 0.0, name="B")
        old_segment = Segment(old_p1, old_p2)
        self.drawables.add(old_segment)
        
        initial_count = len(self.drawables.Segments)
        
        new_p1 = Point(0.0, 0.0, name="X")
        new_p2 = Point(1.0, 0.0, name="Y")
        new_segment = self.segment_manager.create_segment_from_points(new_p1, new_p2)
        
        # Old segment should be removed, only new one should exist
        final_count = len(self.drawables.Segments)
        self.assertEqual(final_count, initial_count)
        self.assertIn(new_segment, self.drawables.Segments)
        self.assertNotIn(old_segment, self.drawables.Segments)


if __name__ == "__main__":
    unittest.main()

