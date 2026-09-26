"""Regression tests for delete cascades run through the AI tool-call path.

Each test drives the canvas with the same function registry and result
processor the AI uses, so it covers the argument handling of the tools too.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Tuple

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

    # ------------------------------------------------------------------
    # K18: adjacency matrix direction and graph ownership on delete
    # ------------------------------------------------------------------
    def _graph_args(self, **overrides: Any) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "name": "G",
            "graph_type": "graph",
            "directed": False,
            "root": None,
            "layout": None,
            "placement_box": None,
            "vertices": [],
            "edges": [],
            "adjacency_matrix": None,
        }
        args.update(overrides)
        return args

    def _matrix_graph_args(self, directed: bool) -> Dict[str, Any]:
        empty_vertex = {"name": None, "x": None, "y": None, "color": None, "label": None}
        return self._graph_args(
            name="M",
            directed=directed,
            layout="circular",
            placement_box={"x": 20, "y": 0, "width": 6, "height": 6},
            vertices=[dict(empty_vertex) for _ in range(3)],
            adjacency_matrix=[[0, 1, 0], [1, 0, 2], [0, 2, 0]],
        )

    def test_undirected_adjacency_matrix_creates_undirected_edges(self) -> None:
        self._call("generate_graph", **self._matrix_graph_args(directed=False))

        graph = self.canvas.get_graph("M")
        self.assertEqual(graph.get_class_name(), "UndirectedGraph")
        self.assertEqual(self._drawables("Vector"), [])
        self.assertEqual(len(self._drawables("Segment")), 2)
        self.assertEqual(sorted(e.get("weight") for e in graph.edges), [1.0, 2.0])
        for x, y in self._point_coords():
            self.assertTrue(20 <= x <= 26 and 0 <= y <= 6, f"Vertex ({x}, {y}) outside the placement box")

        names = [p.name for p in graph._isolated_points]
        result = self._call(
            "analyze_graph",
            graph_name="M",
            operation="shortest_path",
            params={"start": names[0], "goal": names[2]},
        )
        self.assertEqual(result.get("path"), names)
        self.assertEqual(result.get("cost"), 3.0)

    def test_directed_adjacency_matrix_keeps_one_vector_per_entry(self) -> None:
        self._call("generate_graph", **self._matrix_graph_args(directed=True))

        graph = self.canvas.get_graph("M")
        self.assertEqual(graph.get_class_name(), "DirectedGraph")
        self.assertEqual(len(self._drawables("Vector")), 4)
        self.assertEqual(self._drawables("Segment"), [])

    def _triangle_and_graph_on_its_vertex(self) -> Any:
        triangle = self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        self._call(
            "generate_graph",
            **self._graph_args(
                vertices=[
                    {"name": "X", "x": 0, "y": 0, "color": None, "label": None},
                    {"name": "Y", "x": 8, "y": 8, "color": None, "label": None},
                ],
                edges=[{"source": 0, "target": 1, "weight": 1, "name": None, "color": None, "directed": None}],
            ),
        )
        self.assertEqual(len(self._drawables("Segment")), 4)
        return triangle

    def _assert_graph_gone_triangle_kept(self, triangle: Any) -> None:
        self.assertIsNone(self.canvas.get_graph("G"))
        self.assertEqual(self._drawables("Triangle"), [triangle])
        self.assertEqual(self._segment_keys(), [((0, 0), (0, 3)), ((0, 0), (4, 0)), ((0, 3), (4, 0))])
        self.assertEqual(self._point_coords(), [(0, 0), (0, 3), (4, 0)])
        self._assert_parents_present(triangle)

    def test_delete_graph_keeps_preexisting_vertex_and_its_triangle(self) -> None:
        triangle = self._triangle_and_graph_on_its_vertex()
        graph = self.canvas.get_graph("G")
        graph_edges = list(graph.segments)
        created_point = self.canvas.get_point(8, 8)

        self._call("delete_graph", name="G")

        self._assert_graph_gone_triangle_kept(triangle)
        self._assert_not_tracked([graph, created_point] + graph_edges)

    def test_delete_graph_keeps_preexisting_edge(self) -> None:
        triangle = self._triangle([(0, 0), (4, 0), (0, 3)], "ABC")
        self._call(
            "generate_graph",
            **self._graph_args(
                vertices=[
                    {"name": None, "x": 0, "y": 0, "color": None, "label": None},
                    {"name": None, "x": 4, "y": 0, "color": None, "label": None},
                    {"name": None, "x": 9, "y": 9, "color": None, "label": None},
                ],
                edges=[
                    {"source": 0, "target": 1, "weight": None, "name": None, "color": None, "directed": None},
                    {"source": 1, "target": 2, "weight": None, "name": None, "color": None, "directed": None},
                ],
            ),
        )
        self.assertEqual(len(self._drawables("Segment")), 4)

        self._call("delete_graph", name="G")

        self._assert_graph_gone_triangle_kept(triangle)

    def test_delete_graph_still_removes_its_own_vertices(self) -> None:
        self._call(
            "generate_graph",
            **self._graph_args(
                vertices=[
                    {"name": "P", "x": 10, "y": 10, "color": None, "label": None},
                    {"name": "Q", "x": 12, "y": 10, "color": None, "label": None},
                    {"name": "R", "x": 11, "y": 12, "color": None, "label": None},
                ],
                edges=[{"source": 0, "target": 1, "weight": None, "name": None, "color": None, "directed": None}],
            ),
        )

        self._call("delete_graph", name="G")

        self.assertIsNone(self.canvas.get_graph("G"))
        self.assertEqual(self._drawables("Point"), [])
        self.assertEqual(self._drawables("Segment"), [])

    def test_delete_graph_ownership_survives_workspace_round_trip(self) -> None:
        self._triangle_and_graph_on_its_vertex()
        saved = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.workspace_manager._restore_workspace_state(json.loads(json.dumps(saved)))
        restored = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.assertEqual(restored.get("UndirectedGraphs"), saved.get("UndirectedGraphs"))
        triangle = self._drawables("Triangle")[0]

        self._call("delete_graph", name="G")

        self._assert_graph_gone_triangle_kept(triangle)

    def test_delete_graph_ownership_survives_undo(self) -> None:
        self._triangle_and_graph_on_its_vertex()
        self._call("create_point", x=-5, y=-5, name="Z")
        self._call("undo")
        triangle = self._drawables("Triangle")[0]

        self._call("delete_graph", name="G")

        self._assert_graph_gone_triangle_kept(triangle)

    def test_delete_graph_then_undo_restores_graph(self) -> None:
        self._triangle_and_graph_on_its_vertex()
        segments_before = self._segment_keys()
        points_before = self._point_coords()

        self._call("delete_graph", name="G")
        self._call("undo")

        self.assertIsNotNone(self.canvas.get_graph("G"))
        self.assertEqual(len(self._drawables("Triangle")), 1)
        self.assertEqual(self._segment_keys(), segments_before)
        self.assertEqual(self._point_coords(), points_before)


if __name__ == "__main__":
    unittest.main()
