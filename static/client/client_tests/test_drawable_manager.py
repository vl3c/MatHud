from __future__ import annotations

import unittest
from typing import Any

from .simple_mock import SimpleMock
from managers.drawable_manager import DrawableManager


class TestDrawableManagerRegionLookup(unittest.TestCase):
    def _build_manager(self) -> DrawableManager:
        manager = DrawableManager.__new__(DrawableManager)
        manager.polygon_manager = SimpleMock(get_polygon_by_name=SimpleMock(return_value=None))
        manager.circle_manager = SimpleMock(get_circle_by_name=SimpleMock(return_value=None))
        manager.ellipse_manager = SimpleMock(get_ellipse_by_name=SimpleMock(return_value=None))
        manager.arc_manager = SimpleMock(get_circle_arc_by_name=SimpleMock(return_value=None))
        manager.segment_manager = SimpleMock(get_segment_by_name=SimpleMock(return_value=None))
        return manager

    def test_get_region_capable_drawable_by_name_returns_none_for_empty_name(self) -> None:
        manager = self._build_manager()
        self.assertIsNone(manager.get_region_capable_drawable_by_name(""))

    def test_get_region_capable_drawable_by_name_uses_priority_order(self) -> None:
        manager = self._build_manager()
        polygon = SimpleMock(name="poly")
        circle = SimpleMock(name="circle")
        manager.polygon_manager.get_polygon_by_name = SimpleMock(return_value=polygon)
        manager.circle_manager.get_circle_by_name = SimpleMock(return_value=circle)

        drawable = manager.get_region_capable_drawable_by_name("X")

        self.assertIs(drawable, polygon)
        manager.polygon_manager.get_polygon_by_name.assert_called_once_with("X")
        manager.circle_manager.get_circle_by_name.assert_not_called()

    def test_get_region_capable_drawable_by_name_falls_back_to_segment(self) -> None:
        manager = self._build_manager()
        segment = SimpleMock(name="seg")
        manager.segment_manager.get_segment_by_name = SimpleMock(return_value=segment)

        drawable = manager.get_region_capable_drawable_by_name("S1")

        self.assertIs(drawable, segment)
        manager.polygon_manager.get_polygon_by_name.assert_called_once_with("S1")
        manager.circle_manager.get_circle_by_name.assert_called_once_with("S1")
        manager.ellipse_manager.get_ellipse_by_name.assert_called_once_with("S1")
        manager.arc_manager.get_circle_arc_by_name.assert_called_once_with("S1")
        manager.segment_manager.get_segment_by_name.assert_called_once_with("S1")

    def test_first_region_capable_match_returns_first_non_none(self) -> None:
        manager = self._build_manager()
        marker = SimpleMock(name="hit")
        order: list[str] = []

        def miss(tag: str) -> Any:
            order.append(tag)
            return None

        def hit(tag: str) -> Any:
            order.append(tag)
            return marker

        lookups = [
            lambda: miss("a"),
            lambda: miss("b"),
            lambda: hit("c"),
            lambda: hit("d"),
        ]

        result = manager._first_region_capable_match(lookups)

        self.assertIs(result, marker)
        self.assertEqual(order, ["a", "b", "c"])


class TestDrawableManagerColoredAreaDelegation(unittest.TestCase):
    def _build_manager(self) -> DrawableManager:
        manager = DrawableManager.__new__(DrawableManager)
        manager.colored_area_manager = SimpleMock(
            delete_colored_areas_for_function=SimpleMock(),
            delete_colored_areas_for_segment=SimpleMock(),
            delete_colored_areas_for_circle=SimpleMock(),
            delete_colored_areas_for_ellipse=SimpleMock(),
            delete_colored_areas_for_circle_arc=SimpleMock(),
            delete_region_expression_colored_areas_referencing_name=SimpleMock(),
        )
        return manager

    def test_delete_colored_areas_for_function_delegates(self) -> None:
        manager = self._build_manager()
        target = object()
        manager.delete_colored_areas_for_function(target, archive=False)
        manager.colored_area_manager.delete_colored_areas_for_function.assert_called_once_with(target, archive=False)

    def test_delete_colored_areas_for_segment_delegates(self) -> None:
        manager = self._build_manager()
        target = object()
        manager.delete_colored_areas_for_segment(target, archive=True)
        manager.colored_area_manager.delete_colored_areas_for_segment.assert_called_once_with(target, archive=True)

    def test_delete_colored_areas_for_circle_delegates(self) -> None:
        manager = self._build_manager()
        target = object()
        manager.delete_colored_areas_for_circle(target, archive=False)
        manager.colored_area_manager.delete_colored_areas_for_circle.assert_called_once_with(target, archive=False)

    def test_delete_colored_areas_for_ellipse_delegates(self) -> None:
        manager = self._build_manager()
        target = object()
        manager.delete_colored_areas_for_ellipse(target, archive=False)
        manager.colored_area_manager.delete_colored_areas_for_ellipse.assert_called_once_with(target, archive=False)

    def test_delete_colored_areas_for_circle_arc_delegates(self) -> None:
        manager = self._build_manager()
        target = object()
        manager.delete_colored_areas_for_circle_arc(target, archive=True)
        manager.colored_area_manager.delete_colored_areas_for_circle_arc.assert_called_once_with(target, archive=True)

    def test_delete_region_expression_colored_areas_referencing_name_delegates(self) -> None:
        manager = self._build_manager()
        manager.delete_region_expression_colored_areas_referencing_name("A", archive=False)
        manager.colored_area_manager.delete_region_expression_colored_areas_referencing_name.assert_called_once_with(
            "A",
            archive=False,
        )


if __name__ == "__main__":
    unittest.main()
