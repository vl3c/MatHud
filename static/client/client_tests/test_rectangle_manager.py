from __future__ import annotations

import math
import unittest
from typing import Any

from drawables.point import Point
from drawables.segment import Segment
from drawables.rectangle import Rectangle
from managers.drawables_container import DrawablesContainer
from managers.polygon_manager import PolygonManager
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
        self.name_generator.split_point_names = SimpleMock(side_effect=lambda _name, count: [""] * count)
        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
        )
        self.point_manager = SimpleMock(name="PointManagerMock")
        self.segment_manager = SimpleMock(
            name="SegmentManagerMock",
            delete_segment=SimpleMock(return_value=True),
        )
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
        )
        created_points: dict[tuple[float, float], Point] = {}
        point_counter = {"value": 0}

        def fake_create_point(x: float, y: float, *args: Any, **kwargs: Any) -> Point:
            key = (round(x, 6), round(y, 6))
            if key in created_points:
                return created_points[key]
            name = kwargs.get("name")
            if not name:
                name = f"P{point_counter['value']}"
                point_counter["value"] += 1
            color = kwargs.get("color")
            if isinstance(color, str):
                point = Point(x, y, name=name, color=color)
            else:
                point = Point(x, y, name=name)
            created_points[key] = point
            return point

        def fake_create_segment(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            *args: Any,
            **kwargs: Any,
        ) -> Segment:
            point1 = fake_create_point(x1, y1)
            point2 = fake_create_point(x2, y2)
            color = kwargs.get("color")
            return Segment(point1, point2, color=color) if isinstance(color, str) else Segment(point1, point2)

        self.point_manager.create_point = SimpleMock(side_effect=fake_create_point)
        self.segment_manager.create_segment = SimpleMock(side_effect=fake_create_segment)
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

    def test_create_rectangle_with_vertices_rotated(self) -> None:
        angle = math.radians(18)
        width = 5.0
        height = 2.5
        ux = math.cos(angle)
        uy = math.sin(angle)
        vx = -math.sin(angle)
        vy = math.cos(angle)
        half_w = width / 2.0
        half_h = height / 2.0
        vertices = [
            (ux * -half_w + vx * -half_h, uy * -half_w + vy * -half_h),
            (ux * half_w + vx * -half_h, uy * half_w + vy * -half_h),
            (ux * half_w + vx * half_h, uy * half_w + vy * half_h),
            (ux * -half_w + vx * half_h, uy * -half_w + vy * half_h),
        ]
        rectangle = self.rectangle_manager.create_rectangle(
            0.0,
            0.0,
            0.0,
            0.0,
            vertices=vertices,
        )
        self.assertIsInstance(rectangle, Rectangle)
        lengths = []
        for i in range(4):
            segment = getattr(rectangle, f"segment{i + 1}")
            lengths.append(math.hypot(
                segment.point2.x - segment.point1.x,
                segment.point2.y - segment.point1.y,
            ))
        lengths.sort()
        self.assertAlmostEqual(lengths[0], height, places=3)
        self.assertAlmostEqual(lengths[2], width, places=3)

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

