from __future__ import annotations

import unittest
from typing import Dict, Iterable, List, Tuple

from geometry import Point, Segment, Triangle, Rectangle, Circle, Ellipse
from drawables.closed_shape_colored_area import ClosedShapeColoredArea
from managers.transformations_manager import TransformationsManager
from client_tests.simple_mock import SimpleMock


class _IdentityDependencyManager:
    def __init__(self) -> None:
        self._edges: Dict[int, set[int]] = {}
        self._lookup: Dict[int, object] = {}

    def register(self, parent: object, child: object) -> None:
        parent_id = id(parent)
        child_id = id(child)
        self._edges.setdefault(parent_id, set()).add(child_id)
        self._lookup[parent_id] = parent
        self._lookup[child_id] = child

    def register_many(self, relationships: Iterable[Tuple[object, object]]) -> None:
        for parent, child in relationships:
            self.register(parent, child)

    def get_children(self, drawable: object) -> set:
        parent_id = id(drawable)
        child_ids = self._edges.get(parent_id, set())
        return {self._lookup[child_id] for child_id in child_ids if child_id in self._lookup}


def _build_canvas(primary_drawable: object, segments: List[Segment]) -> Tuple[SimpleMock, _IdentityDependencyManager, SimpleMock]:
    renderer = SimpleMock()
    renderer.invalidate_drawable_cache = SimpleMock()

    dependency_manager = _IdentityDependencyManager()
    for segment in segments:
        dependency_manager.register(segment, primary_drawable)

    drawables_container = SimpleMock(Segments=list(segments))
    drawable_manager = SimpleMock(
        get_drawables=SimpleMock(return_value=[primary_drawable]),
        drawables=drawables_container,
    )
    canvas = SimpleMock(
        renderer=renderer,
        dependency_manager=dependency_manager,
        drawable_manager=drawable_manager,
        draw_enabled=False,
        draw=SimpleMock(),
        undo_redo_manager=SimpleMock(archive=SimpleMock()),
    )
    return canvas, dependency_manager, renderer


class TestTransformationsManager(unittest.TestCase):
    def test_translate_triangle_refreshes_segments(self) -> None:
        p1 = Point(0, 0, name="P1")
        p2 = Point(4, 0, name="P2")
        p3 = Point(4, 3, name="P3")
        s1 = Segment(p1, p2, "red")
        s2 = Segment(p2, p3, "green")
        s3 = Segment(p3, p1, "blue")
        triangle = Triangle(s1, s2, s3, color="yellow")

        canvas, _, renderer = _build_canvas(triangle, [s1, s2, s3])
        triangle.canvas = canvas
        manager = TransformationsManager(canvas)

        manager.translate_object(triangle.name, 1.5, -2.0)

        self.assertEqual(s1.line_formula, "y = 0.0 * x - 2.0")
        self.assertTrue(renderer.invalidate_drawable_cache.calls)
        invalidated = {call_args[0][0] for call_args in renderer.invalidate_drawable_cache.calls}
        self.assertIn(s1, invalidated)
        self.assertIn(triangle, invalidated)

    def test_translate_rectangle_refreshes_segments(self) -> None:
        p1 = Point(0, 0, name="P1")
        p2 = Point(4, 0, name="P2")
        p3 = Point(4, 3, name="P3")
        p4 = Point(0, 3, name="P4")
        s1 = Segment(p1, p2, "red")
        s2 = Segment(p2, p3, "green")
        s3 = Segment(p3, p4, "blue")
        s4 = Segment(p4, p1, "orange")
        rectangle = Rectangle(s1, s2, s3, s4, color="purple")

        canvas, _, renderer = _build_canvas(rectangle, [s1, s2, s3, s4])
        rectangle.canvas = canvas
        manager = TransformationsManager(canvas)

        manager.translate_object(rectangle.name, -1.0, 2.5)

        self.assertEqual(s1.line_formula, "y = 0.0 * x + 2.5")
        self.assertTrue(renderer.invalidate_drawable_cache.calls)
        invalidated = {call_args[0][0] for call_args in renderer.invalidate_drawable_cache.calls}
        self.assertIn(s1, invalidated)
        self.assertIn(rectangle, invalidated)

    def test_translate_polygon_invalidates_closed_area_children(self) -> None:
        p1 = Point(0, 0, name="P1")
        p2 = Point(4, 0, name="P2")
        p3 = Point(4, 3, name="P3")
        p4 = Point(0, 3, name="P4")
        s1 = Segment(p1, p2, "red")
        s2 = Segment(p2, p3, "green")
        s3 = Segment(p3, p4, "blue")
        s4 = Segment(p4, p1, "orange")
        rectangle = Rectangle(s1, s2, s3, s4, color="purple")
        area = ClosedShapeColoredArea(
            shape_type="polygon",
            segments=[s1, s2, s3, s4],
            color="pink",
            opacity=0.4,
        )

        canvas, dependency_manager, renderer = _build_canvas(rectangle, [s1, s2, s3, s4])
        rectangle.canvas = canvas
        area.canvas = canvas
        dependency_manager.register_many(
            (segment, area) for segment in (s1, s2, s3, s4)
        )

        initial_snapshot = area.get_state()["args"]["geometry_snapshot"]

        manager = TransformationsManager(canvas)
        manager.translate_object(rectangle.name, 1.5, -2.0)

        updated_snapshot = area.get_state()["args"]["geometry_snapshot"]
        self.assertNotEqual(initial_snapshot["polygon_coords"], updated_snapshot["polygon_coords"])
        expected_coords = [[coord[0] + 1.5, coord[1] - 2.0] for coord in initial_snapshot["polygon_coords"]]
        self.assertEqual(updated_snapshot["polygon_coords"], expected_coords)

    def test_translate_circle_refreshes_dependents(self) -> None:
        center = Point(0, 0, name="C")
        circle = Circle(center, 5, color="cyan")
        area = ClosedShapeColoredArea(
            shape_type="circle",
            circle=circle,
            color="lightblue",
            opacity=0.5,
        )

        segments: List[Segment] = []
        canvas, dependency_manager, renderer = _build_canvas(circle, segments)
        circle.canvas = canvas
        area.canvas = canvas
        dependency_manager.register(circle, area)

        initial_snapshot = area.get_state()["args"]["geometry_snapshot"]

        manager = TransformationsManager(canvas)
        manager.translate_object(circle.name, 2.0, -1.0)

        self.assertEqual(circle.center.x, 2.0)
        self.assertEqual(circle.center.y, -1.0)
        updated_snapshot = area.get_state()["args"]["geometry_snapshot"]
        self.assertNotEqual(initial_snapshot["circle"]["center"], updated_snapshot["circle"]["center"])
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    def test_translate_ellipse_refreshes_dependents(self) -> None:
        center = Point(1, 1, name="E")
        ellipse = Ellipse(center, 6, 4, rotation_angle=30.0, color="magenta")
        area = ClosedShapeColoredArea(
            shape_type="ellipse",
            ellipse=ellipse,
            color="lavender",
            opacity=0.3,
        )

        segments: List[Segment] = []
        canvas, dependency_manager, renderer = _build_canvas(ellipse, segments)
        ellipse.canvas = canvas
        area.canvas = canvas
        dependency_manager.register(ellipse, area)

        initial_snapshot = area.get_state()["args"]["geometry_snapshot"]

        manager = TransformationsManager(canvas)
        manager.translate_object(ellipse.name, -3.0, 4.0)

        self.assertEqual(ellipse.center.x, -2.0)
        self.assertEqual(ellipse.center.y, 5.0)
        updated_snapshot = area.get_state()["args"]["geometry_snapshot"]
        self.assertNotEqual(initial_snapshot["ellipse"]["center"], updated_snapshot["ellipse"]["center"])
        self.assertTrue(renderer.invalidate_drawable_cache.calls)
