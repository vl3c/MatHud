from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from managers.colored_area_manager import ColoredAreaManager
from managers.drawables_container import DrawablesContainer
from .simple_mock import SimpleMock


class TestColoredAreaManager(unittest.TestCase):
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
        self.dependency_manager = SimpleMock(name="DependencyManagerMock")
        self.drawable_manager_proxy = SimpleMock(name="DrawableManagerProxyMock")

        self.manager = ColoredAreaManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_functions_area(self) -> FunctionsBoundedColoredArea:
        func = SimpleMock(name="f1", function=lambda x: x)
        area = FunctionsBoundedColoredArea(func, None, left_bound=0.0, right_bound=5.0, color="#abcdef", opacity=0.3)
        self.drawables.add(area)
        return area

    def _add_segments_area(self) -> SegmentsBoundedColoredArea:
        s1 = Segment(Point(0.0, 0.0, "A"), Point(1.0, 1.0, "B"))
        area = SegmentsBoundedColoredArea(s1, None, color="#123456", opacity=0.4)
        self.drawables.add(area)
        return area

    def _add_function_segment_area(self) -> FunctionSegmentBoundedColoredArea:
        func = SimpleMock(name="f2", function=lambda x: 2 * x)
        seg = Segment(Point(0.0, 0.0, "S1"), Point(1.0, 0.0, "S2"))
        area = FunctionSegmentBoundedColoredArea(func, seg, color="#654321", opacity=0.5)
        self.drawables.add(area)
        return area

    def test_update_colored_area_updates_style_and_bounds(self) -> None:
        area = self._add_functions_area()

        result = self.manager.update_colored_area(
            area.name,
            new_color="#ff00ff",
            new_opacity=0.6,
            new_left_bound=1.0,
            new_right_bound=6.0,
        )

        self.assertTrue(result)
        self.assertEqual(area.color, "#ff00ff")
        self.assertEqual(area.opacity, 0.6)
        self.assertEqual(area.left_bound, 1.0)
        self.assertEqual(area.right_bound, 6.0)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_colored_area_blocks_bounds_for_segment_area(self) -> None:
        area = self._add_segments_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_left_bound=0.0)

    def test_update_colored_area_validates_color(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_color="not-a-color")

    def test_update_colored_area_validates_opacity(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_opacity=2.0)

    def test_update_colored_area_validates_bounds_relation(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_left_bound=10.0)

    def test_update_colored_area_style_only_segments_area(self) -> None:
        area = self._add_segments_area()

        result = self.manager.update_colored_area(area.name, new_color="#999999", new_opacity=0.2)

        self.assertTrue(result)
        self.assertEqual(area.color, "#999999")
        self.assertEqual(area.opacity, 0.2)

    def test_update_colored_area_style_only_function_segment_area(self) -> None:
        area = self._add_function_segment_area()

        result = self.manager.update_colored_area(area.name, new_color="#00ffaa", new_opacity=0.7)

        self.assertTrue(result)
        self.assertEqual(area.color, "#00ffaa")
        self.assertEqual(area.opacity, 0.7)

    def test_update_colored_area_missing_name(self) -> None:
        self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area("missing", new_color="#000000")


if __name__ == "__main__":
    unittest.main()

