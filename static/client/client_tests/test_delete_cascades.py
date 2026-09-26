"""Regression tests for delete cascades run through the AI tool-call path.

Each test drives the canvas with the same function registry and result
processor the AI uses, so it covers the argument handling of the tools too.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager

Coord = Tuple[float, float]


class TestDeleteCascades(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.available_functions = FunctionRegistry.get_available_functions(self.canvas, self.workspace_manager)
        self.undoable_functions = FunctionRegistry.get_undoable_functions()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _call(self, function_name: str, **arguments: Any) -> Any:
        """Run one tool call through the AI path and return its result value."""
        calls = [{"function_name": function_name, "arguments": arguments}]
        results = ProcessFunctionCalls.get_results(
            calls, self.available_functions, self.undoable_functions, self.canvas
        )
        self.assertEqual(len(results), 1, f"Expected one result for {function_name}: {results}")
        value = list(results.values())[0]
        if isinstance(value, str):
            self.assertFalse(value.startswith("Error"), f"{function_name} failed: {value}")
        return value

    def _drawables(self, class_name: str) -> List[Any]:
        return list(self.canvas.get_drawables_by_class_name(class_name))

    def _coords(self, point: Any) -> Coord:
        return (round(point.x, 9), round(point.y, 9))

    def _segment_keys(self) -> List[Tuple[Coord, Coord]]:
        keys = []
        for segment in self._drawables("Segment"):
            ends = sorted([self._coords(segment.point1), self._coords(segment.point2)])
            keys.append((ends[0], ends[1]))
        return sorted(keys)

    def _point_coords(self) -> List[Coord]:
        return sorted(self._coords(p) for p in self._drawables("Point"))

    def _triangle(self, vertices: List[Coord], name: str) -> Any:
        self._call(
            "create_polygon",
            vertices=[{"x": x, "y": y} for x, y in vertices],
            polygon_type="triangle",
            name=name,
        )
        return self.canvas.get_polygon_by_vertices(vertices, polygon_type="triangle")

    def _assert_not_tracked(self, drawables: List[Any]) -> None:
        """Deleted drawables must leave no entries in the dependency manager."""
        dependency_manager = self.canvas.drawable_manager.dependency_manager
        for drawable in drawables:
            drawable_id = id(drawable)
            label = f"{drawable.get_class_name()} {getattr(drawable, 'name', '')}"
            self.assertNotIn(drawable_id, dependency_manager._object_lookup, f"{label} still tracked")
            self.assertNotIn(drawable_id, dependency_manager._parents, f"{label} still has parents")
            self.assertNotIn(drawable_id, dependency_manager._children, f"{label} still has children")
            for linked in list(dependency_manager._parents.values()) + list(dependency_manager._children.values()):
                self.assertNotIn(drawable_id, linked, f"{label} still linked from another drawable")

    def _assert_parents_present(self, drawable: Any) -> None:
        """Every dependency parent of a surviving drawable must still be on the canvas."""
        dependency_manager = self.canvas.drawable_manager.dependency_manager
        for parent in dependency_manager.get_parents(drawable):
            class_name = parent.get_class_name()
            self.assertIn(parent, self._drawables(class_name), f"Dangling parent {class_name} {parent.name}")

    # ------------------------------------------------------------------
    # K6: angle and polygon deletes must not take shared parts with them
    # ------------------------------------------------------------------
    def test_delete_angle_keeps_triangle_and_sides(self) -> None:
        triangle = self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3)
        angles = self._drawables("Angle")
        self.assertEqual(len(angles), 1)
        angle = angles[0]

        self._call("delete_angle", name=angle.name)

        self.assertEqual(self._drawables("Angle"), [])
        self.assertEqual(self._drawables("Triangle"), [triangle])
        self.assertEqual(len(self._drawables("Segment")), 3)
        self.assertEqual(self._point_coords(), [(0, 0), (0, 3), (4, 0)])
        self._assert_not_tracked([angle])
        self._assert_parents_present(triangle)

    def test_delete_angle_keeps_its_arm_segments(self) -> None:
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3)
        angle = self._drawables("Angle")[0]
        segments_before = self._segment_keys()
        self.assertEqual(len(segments_before), 2)

        self._call("delete_angle", name=angle.name)

        self.assertEqual(self._drawables("Angle"), [])
        self.assertEqual(self._segment_keys(), segments_before)
        self.assertEqual(len(self._drawables("Point")), 3)

    def test_delete_angle_then_undo_restores_angle(self) -> None:
        self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3)
        angle_name = self._drawables("Angle")[0].name

        self._call("delete_angle", name=angle_name)
        self._call("undo")

        self.assertEqual([a.name for a in self._drawables("Angle")], [angle_name])
        self.assertEqual(len(self._drawables("Triangle")), 1)
        self.assertEqual(len(self._drawables("Segment")), 3)

    def test_delete_polygon_keeps_edge_shared_with_other_triangle(self) -> None:
        self._triangle([(0, 0), (4, 0), (4, 4)], "ABC")
        other = self._triangle([(0, 0), (4, 4), (0, 4)], "ACD")
        self.assertEqual(len(self._drawables("Segment")), 5)
        deleted = self.canvas.get_polygon_by_vertices([(0, 0), (4, 0), (4, 4)], polygon_type="triangle")
        deleted_edges = [
            s
            for s in self._drawables("Segment")
            if self._coords(s.point1) == (4, 0) or self._coords(s.point2) == (4, 0)
        ]

        self._call(
            "delete_polygon",
            polygon_type="triangle",
            name="ABC",
            vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}],
        )

        self.assertEqual(self._drawables("Triangle"), [other])
        self.assertEqual(
            self._segment_keys(),
            [((0, 0), (0, 4)), ((0, 0), (4, 4)), ((0, 4), (4, 4))],
        )
        # Deleting a polygon keeps its vertex points.
        self.assertEqual(self._point_coords(), [(0, 0), (0, 4), (4, 0), (4, 4)])
        self._assert_not_tracked([deleted] + deleted_edges)
        self._assert_parents_present(other)

    def test_delete_polygon_keeps_edges_used_by_angle(self) -> None:
        self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        self._call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3)
        angle = self._drawables("Angle")[0]

        self._call("delete_polygon", polygon_type="triangle", name="ABC", vertices=None)

        self.assertEqual(self._drawables("Triangle"), [])
        self.assertEqual(self._drawables("Angle"), [angle])
        self.assertEqual(self._segment_keys(), [((0, 0), (0, 3)), ((0, 0), (4, 0))])
        self._assert_parents_present(angle)

    def test_delete_lone_polygon_still_removes_its_edges(self) -> None:
        triangle = self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        edges = self._drawables("Segment")

        self._call("delete_polygon", polygon_type="triangle", name="ABC", vertices=None)

        self.assertEqual(self._drawables("Triangle"), [])
        self.assertEqual(self._drawables("Segment"), [])
        self.assertEqual(len(self._drawables("Point")), 3)
        self._assert_not_tracked([triangle] + edges)

    def test_delete_polygon_then_undo_restores_everything(self) -> None:
        self._triangle([(0, 0), (4, 0), (4, 4)], "ABC")
        self._triangle([(0, 0), (4, 4), (0, 4)], "ACD")
        segments_before = self._segment_keys()

        self._call("delete_polygon", polygon_type="triangle", name="ABC", vertices=None)
        self._call("undo")

        self.assertEqual(len(self._drawables("Triangle")), 2)
        self.assertEqual(self._segment_keys(), segments_before)

    # ------------------------------------------------------------------
    # K7: a vector and a segment on the same endpoints are independent
    # ------------------------------------------------------------------
    def _segment_and_vector_on_same_endpoints(self) -> None:
        self._call("create_segment", x1=0, y1=0, x2=3, y2=0)
        self._call("create_vector", origin_x=0, origin_y=0, tip_x=3, tip_y=0)
        self.assertEqual(len(self._drawables("Segment")), 1)
        self.assertEqual(len(self._drawables("Vector")), 1)

    def test_delete_vector_keeps_segment_on_same_endpoints(self) -> None:
        self._segment_and_vector_on_same_endpoints()
        vector = self._drawables("Vector")[0]
        undo_depth = len(self.canvas.undo_redo_manager.undo_stack)

        self._call("delete_vector", origin_x=0, origin_y=0, tip_x=3, tip_y=0)

        self.assertEqual(self._drawables("Vector"), [])
        self.assertEqual(self._segment_keys(), [((0, 0), (3, 0))])
        self.assertEqual(self._point_coords(), [(0, 0), (3, 0)])
        self.assertLessEqual(len(self.canvas.undo_redo_manager.undo_stack), undo_depth + 2)
        self._assert_not_tracked([vector])

    def test_delete_segment_keeps_vector_on_same_endpoints(self) -> None:
        self._segment_and_vector_on_same_endpoints()

        self._call("delete_segment", x1=0, y1=0, x2=3, y2=0)

        self.assertEqual(self._drawables("Segment"), [])
        self.assertEqual(len(self._drawables("Vector")), 1)
        self.assertEqual(self._point_coords(), [(0, 0), (3, 0)])

    def test_delete_vector_then_undo_restores_vector(self) -> None:
        self._segment_and_vector_on_same_endpoints()

        self._call("delete_vector", origin_x=0, origin_y=0, tip_x=3, tip_y=0)
        self._call("undo")

        self.assertEqual(len(self._drawables("Vector")), 1)
        self.assertEqual(self._segment_keys(), [((0, 0), (3, 0))])


if __name__ == "__main__":
    unittest.main()
