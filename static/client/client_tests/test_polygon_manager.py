from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.triangle import Triangle
from drawables.quadrilateral import Quadrilateral
from drawables.rectangle import Rectangle
from drawables.pentagon import Pentagon
from drawables.hexagon import Hexagon
from drawables.heptagon import Heptagon
from drawables.octagon import Octagon
from drawables.nonagon import Nonagon
from drawables.decagon import Decagon
from drawables.generic_polygon import GenericPolygon
from managers.drawables_container import DrawablesContainer
from managers.polygon_manager import PolygonManager
from .simple_mock import SimpleMock


class TestPolygonManager(unittest.TestCase):
    def setUp(self) -> None:
        self._point_cache: dict[tuple[float, float], Point] = {}

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
            split_point_names=lambda expr, count: list(expr[:count]) if expr else [""] * count,
        )
        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
        )
        self.point_manager = SimpleMock(
            name="PointManagerMock",
            create_point=self._create_point_mock,
            get_point_by_name=lambda name: None,
        )
        self.segment_manager = SimpleMock(
            name="SegmentManagerMock",
            create_segment=self._create_segment_mock,
            delete_segment=SimpleMock(),
        )
        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
        )

        self.polygon_manager = PolygonManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            point_manager=self.point_manager,
            segment_manager=self.segment_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _get_or_create_point(self, x: float, y: float, name: str = "") -> Point:
        key = (round(x, 9), round(y, 9))
        if key not in self._point_cache:
            point_name = name or f"P{x}_{y}"
            self._point_cache[key] = Point(x, y, name=point_name)
        return self._point_cache[key]

    def _create_point_mock(
        self,
        x: float,
        y: float,
        name: str = "",
        extra_graphics: bool = True,
    ) -> Point:
        return self._get_or_create_point(x, y, name)

    def _create_segment_mock(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        extra_graphics: bool = True,
        **kwargs: str,
    ) -> Segment:
        p1 = self._get_or_create_point(x1, y1)
        p2 = self._get_or_create_point(x2, y2)
        color = kwargs.get("color", "")
        return Segment(p1, p2, color=color)

    def test_create_triangle(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="triangle")
        self.assertIsInstance(polygon, Triangle)

    def test_create_quadrilateral(self) -> None:
        vertices = [(0, 0), (2, 0), (2.5, 1.5), (0.5, 1.5)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="quadrilateral")
        self.assertIsInstance(polygon, Quadrilateral)

    def test_create_rectangle(self) -> None:
        vertices = [(0, 0), (4, 0), (4, 2), (0, 2)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="rectangle")
        self.assertIsInstance(polygon, Rectangle)

    def test_create_square(self) -> None:
        vertices = [(0, 0), (2, 0), (2, 2), (0, 2)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="square")
        self.assertIsInstance(polygon, Rectangle)

    def test_create_pentagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)) for i in range(5)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="pentagon")
        self.assertIsInstance(polygon, Pentagon)

    def test_create_hexagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 6), math.sin(2 * math.pi * i / 6)) for i in range(6)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="hexagon")
        self.assertIsInstance(polygon, Hexagon)

    def test_create_heptagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 7), math.sin(2 * math.pi * i / 7)) for i in range(7)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="heptagon")
        self.assertIsInstance(polygon, Heptagon)

    def test_create_octagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)) for i in range(8)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="octagon")
        self.assertIsInstance(polygon, Octagon)

    def test_create_nonagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 9), math.sin(2 * math.pi * i / 9)) for i in range(9)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="nonagon")
        self.assertIsInstance(polygon, Nonagon)

    def test_create_decagon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 10), math.sin(2 * math.pi * i / 10)) for i in range(10)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="decagon")
        self.assertIsInstance(polygon, Decagon)

    def test_create_generic_polygon(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)) for i in range(12)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="generic")
        self.assertIsInstance(polygon, GenericPolygon)

    def test_infer_triangle_from_vertex_count(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        polygon = self.polygon_manager.create_polygon(vertices)
        self.assertIsInstance(polygon, Triangle)

    def test_infer_quadrilateral_from_vertex_count(self) -> None:
        vertices = [(0, 0), (2, 0), (2.5, 1.5), (0.5, 1.5)]
        polygon = self.polygon_manager.create_polygon(vertices)
        self.assertIsInstance(polygon, Quadrilateral)

    def test_infer_pentagon_from_vertex_count(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)) for i in range(5)]
        polygon = self.polygon_manager.create_polygon(vertices)
        self.assertIsInstance(polygon, Pentagon)

    def test_infer_generic_from_large_vertex_count(self) -> None:
        import math
        vertices = [(math.cos(2 * math.pi * i / 15), math.sin(2 * math.pi * i / 15)) for i in range(15)]
        polygon = self.polygon_manager.create_polygon(vertices)
        self.assertIsInstance(polygon, GenericPolygon)

    def test_wrong_vertex_count_raises(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        with self.assertRaises(ValueError):
            self.polygon_manager.create_polygon(vertices, polygon_type="quadrilateral")

    def test_invalid_polygon_type_raises(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        with self.assertRaises(ValueError):
            self.polygon_manager.create_polygon(vertices, polygon_type="invalid_type")

    def test_too_few_vertices_raises(self) -> None:
        vertices = [(0, 0), (3, 0)]
        with self.assertRaises(ValueError):
            self.polygon_manager.create_polygon(vertices)

    def test_create_with_dict_vertices(self) -> None:
        vertices = [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 1.5, "y": 2.6}]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="triangle")
        self.assertIsInstance(polygon, Triangle)

    def test_create_with_color(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", color="red")
        self.assertEqual(polygon.color, "red")

    def test_triangle_equilateral_subtype(self) -> None:
        vertices = [(0, 0), (2, 0), (1, 1.732)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", subtype="equilateral")
        self.assertIsInstance(polygon, Triangle)

    def test_quadrilateral_parallelogram_subtype(self) -> None:
        vertices = [(0, 0), (4, 0), (5, 3), (1, 3)]
        polygon = self.polygon_manager.create_polygon(vertices, polygon_type="quadrilateral", subtype="parallelogram")
        self.assertIsInstance(polygon, Quadrilateral)

    def test_get_polygon_by_name(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        created = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", name="ABC")
        found = self.polygon_manager.get_polygon_by_name(created.name)
        self.assertIs(found, created)

    def test_get_polygon_by_name_not_found(self) -> None:
        result = self.polygon_manager.get_polygon_by_name("nonexistent")
        self.assertIsNone(result)

    def test_get_polygon_by_vertices(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        created = self.polygon_manager.create_polygon(vertices, polygon_type="triangle")
        found = self.polygon_manager.get_polygon_by_vertices(vertices)
        self.assertIs(found, created)

    def test_delete_polygon_by_name(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        created = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", name="ABC")
        result = self.polygon_manager.delete_polygon(name=created.name)
        self.assertTrue(result)
        self.assertIsNone(self.polygon_manager.get_polygon_by_name(created.name))

    def test_delete_polygon_not_found(self) -> None:
        result = self.polygon_manager.delete_polygon(name="nonexistent")
        self.assertFalse(result)

    def test_update_polygon_color(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        created = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", name="ABC")
        result = self.polygon_manager.update_polygon(created.name, new_color="blue")
        self.assertTrue(result)
        self.assertEqual(created.color, "blue")

    def test_update_polygon_not_found_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.polygon_manager.update_polygon("nonexistent", new_color="blue")

    def test_update_polygon_requires_property(self) -> None:
        vertices = [(0, 0), (3, 0), (1.5, 2.6)]
        created = self.polygon_manager.create_polygon(vertices, polygon_type="triangle", name="ABC")
        with self.assertRaises(ValueError):
            self.polygon_manager.update_polygon(created.name)


if __name__ == "__main__":
    unittest.main()

