"""Graph tools on a real canvas, driven through the AI tool-call path.

Covers generate_graph input validation, vertex placement (layout positions never merge
into existing points), parallel edge weights, edge direction, analyze_graph on the
created graphs, and translating a graph.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager


def _vertex(name: Optional[str], x: Optional[float] = None, y: Optional[float] = None) -> Dict[str, Any]:
    return {"name": name, "x": x, "y": y, "color": None, "label": None}


def _edge(source: Any, target: Any, weight: Optional[float] = None, directed: Optional[bool] = None) -> Dict[str, Any]:
    return {"source": source, "target": target, "weight": weight, "name": None, "color": None, "directed": directed}


class TestGraphTools(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.available_functions = FunctionRegistry.get_available_functions(self.canvas, self.workspace_manager)
        self.undoable_functions = FunctionRegistry.get_undoable_functions()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run(self, function_name: str, **arguments: Any) -> Any:
        calls = [{"function_name": function_name, "arguments": arguments}]
        results = ProcessFunctionCalls.get_results(
            calls, self.available_functions, self.undoable_functions, self.canvas
        )
        self.assertEqual(len(results), 1, f"Expected one result for {function_name}: {results}")
        return list(results.values())[0]

    def _call(self, function_name: str, **arguments: Any) -> Any:
        value = self._run(function_name, **arguments)
        if isinstance(value, str):
            self.assertFalse(value.startswith("Error"), f"{function_name} failed: {value}")
        return value

    def _call_error(self, function_name: str, expected: str, **arguments: Any) -> str:
        value = self._run(function_name, **arguments)
        text = str(value)
        self.assertIn("rror", text, f"{function_name} should have failed: {value}")
        self.assertIn(expected, text)
        return text

    def _graph(self, **overrides: Any) -> Any:
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
        return self._call("generate_graph", **args)

    def _graph_error(self, expected: str, **overrides: Any) -> str:
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
        return self._call_error("generate_graph", expected, **args)

    def _analyze(self, graph_name: str, operation: str, **params: Any) -> Dict[str, Any]:
        full = {key: None for key in ("start", "goal", "root", "a", "b", "new_root", "x", "y")}
        full.update(params)
        result = self._run("analyze_graph", graph_name=graph_name, operation=operation, params=full)
        self.assertIsInstance(result, dict)
        return result

    def _vertex_coords(self, graph_name: str) -> List[Tuple[float, float]]:
        state = self.canvas.capture_graph_state(graph_name)
        return [(round(v.x, 9), round(v.y, 9)) for v in state.vertices]

    def _edge_weights(self, graph_name: str) -> List[Any]:
        state = self.canvas.capture_graph_state(graph_name)
        return [(e.source, e.target, e.weight) for e in state.edges]

    def _weights(self, graph_name: str) -> List[float]:
        # Brython does not treat int 1 and float 1.0 as equal set members, so compare float lists.
        return sorted(float(w) for _, _, w in self._edge_weights(graph_name))

    # ------------------------------------------------------------------
    # Vertex placement
    # ------------------------------------------------------------------
    def test_second_graph_with_same_layout_gets_its_own_points(self) -> None:
        triangle = [_edge(0, 1, 1), _edge(1, 2, 1), _edge(2, 0, 1)]
        self._graph(name="G1", layout="circular", vertices=[_vertex("A"), _vertex("B"), _vertex("C")], edges=triangle)
        other = [_edge(0, 1, 7), _edge(1, 2, 7), _edge(2, 0, 7)]
        self._graph(name="G2", layout="circular", vertices=[_vertex("P"), _vertex("Q"), _vertex("R")], edges=other)

        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Point")), 6)
        self.assertTrue(set(self._vertex_coords("G1")).isdisjoint(self._vertex_coords("G2")))
        self.assertEqual(self._weights("G1"), [1.0, 1.0, 1.0])
        self.assertEqual(self._weights("G2"), [7.0, 7.0, 7.0])
        self.assertEqual(self._analyze("G2", "shortest_path", start="P", goal="Q").get("cost"), 7.0)

    def test_explicit_coordinates_reuse_an_existing_point(self) -> None:
        self._call("create_point", x=2, y=3, name="K", color=None)
        self._graph(vertices=[_vertex("X", 2, 3), _vertex("Y", 6, 3)], edges=[_edge(0, 1)])

        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Point")), 2)
        self.assertIn("K", [v.id for v in self.canvas.capture_graph_state("G").vertices])

    def test_explicit_coordinates_are_kept_when_other_vertices_are_laid_out(self) -> None:
        self._graph(vertices=[_vertex("A", 7, 9), _vertex("B"), _vertex("C")], edges=[_edge(0, 1), _edge(1, 2)])

        state = self.canvas.capture_graph_state("G")
        a = [v for v in state.vertices if v.id == "A"][0]
        self.assertEqual((a.x, a.y), (7.0, 9.0))

    def test_duplicate_explicit_coordinates_are_rejected(self) -> None:
        self._graph_error(
            "same coordinates",
            vertices=[_vertex("A", 0, 0), _vertex("B", 0, 0), _vertex("C", 5, 5)],
            edges=[_edge(0, 2), _edge(1, 2)],
        )
        self.assertIsNone(self.canvas.get_graph("G"))

    def test_grid_layout_keeps_components_apart(self) -> None:
        self._graph(
            layout="grid",
            vertices=[_vertex("A"), _vertex("B"), _vertex("C"), _vertex("D")],
            edges=[_edge(0, 1), _edge(2, 3)],
        )

        self.assertEqual(len(set(self._vertex_coords("G"))), 4)
        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Segment")), 2)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def test_out_of_range_edge_index_is_an_error(self) -> None:
        self._graph_error("out of range", vertices=[_vertex("A"), _vertex("B")], edges=[_edge(0, 5)])

    def test_non_integer_edge_index_is_an_error(self) -> None:
        self._graph_error("not a vertex index", vertices=[_vertex("A"), _vertex("B")], edges=[_edge(0, 1.5)])

    def test_unknown_root_is_an_error(self) -> None:
        self._graph_error(
            "Unknown root",
            graph_type="tree",
            root="Z",
            vertices=[_vertex("A"), _vertex("B")],
            edges=[_edge(0, 1)],
        )

    def test_tree_with_a_cycle_is_an_error(self) -> None:
        self._graph_error(
            "is not a tree",
            graph_type="tree",
            vertices=[_vertex("A"), _vertex("B"), _vertex("C")],
            edges=[_edge(0, 1), _edge(1, 2), _edge(2, 0)],
        )

    def test_dag_with_a_cycle_is_an_error(self) -> None:
        self._graph_error(
            "has a cycle",
            graph_type="dag",
            directed=None,
            vertices=[_vertex("A"), _vertex("B"), _vertex("C")],
            edges=[_edge(0, 1), _edge(1, 2), _edge(2, 0)],
        )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def test_undirected_parallel_edges_carry_the_cheapest_weight(self) -> None:
        self._graph(vertices=[_vertex("A"), _vertex("B")], edges=[_edge(0, 1, 5), _edge(0, 1, 1)])

        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Segment")), 1)
        self.assertEqual(self._weights("G"), [1.0, 1.0])
        self.assertEqual(self._analyze("G", "shortest_path", start="A", goal="B").get("cost"), 1.0)

    def test_directed_parallel_edges_carry_the_cheapest_weight(self) -> None:
        self._graph(directed=True, vertices=[_vertex("A"), _vertex("B")], edges=[_edge(0, 1, 1), _edge(0, 1, 5)])

        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Vector")), 1)
        self.assertEqual(self._analyze("G", "shortest_path", start="A", goal="B").get("cost"), 1.0)

    def test_edges_take_the_graph_direction(self) -> None:
        self._graph(
            vertices=[_vertex("A"), _vertex("B"), _vertex("C")],
            edges=[_edge(0, 1, 1), _edge(1, 2, 1, directed=True)],
        )
        self.assertEqual(len(self._edge_weights("G")), 2)
        self.assertEqual(self.canvas.get_drawables_by_class_name("Vector"), [])
        self.assertEqual(self._analyze("G", "shortest_path", start="A", goal="C").get("path"), ["A", "B", "C"])

        self._graph(
            name="D",
            directed=True,
            vertices=[_vertex("P"), _vertex("Q"), _vertex("R")],
            edges=[_edge(0, 1, 1), _edge(1, 2, 1, directed=False)],
        )
        self.assertEqual(len(self._edge_weights("D")), 2)
        self.assertEqual(self._analyze("D", "shortest_path", start="P", goal="R").get("path"), ["P", "Q", "R"])

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def test_unknown_vertex_lists_the_vertices(self) -> None:
        self._graph(vertices=[_vertex("A"), _vertex("B")], edges=[_edge(0, 1)])

        result = self._analyze("G", "shortest_path", start="A", goal="Z")
        self.assertIn("A, B", result.get("error", ""))

    def test_shortest_path_to_itself_on_isolated_vertex(self) -> None:
        self._graph(vertices=[_vertex("A"), _vertex("B"), _vertex("C")], edges=[_edge(0, 1)])

        result = self._analyze("G", "shortest_path", start="C", goal="C")
        self.assertEqual(result.get("path"), ["C"])
        self.assertEqual(result.get("cost"), 0.0)

    def test_tree_without_root_uses_the_vertex_no_edge_points_to(self) -> None:
        self._graph(
            graph_type="tree",
            vertices=[_vertex("A"), _vertex("B"), _vertex("C")],
            edges=[_edge(2, 0), _edge(2, 1)],
        )

        result = self._analyze("G", "levels")
        self.assertEqual(result.get("levels"), [["C"], ["A", "B"]])

    def test_lca_on_a_cycle_reports_it_is_not_a_tree(self) -> None:
        self._graph(
            vertices=[_vertex(n) for n in "ABCDE"],
            edges=[_edge(0, 1), _edge(1, 2), _edge(2, 3), _edge(3, 4), _edge(4, 0)],
        )

        result = self._analyze("G", "lca", root="A", a="C", b="D")
        self.assertIn("not one", result.get("error", ""))

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def test_translate_graph_moves_vertices_and_edges(self) -> None:
        self._graph(
            vertices=[_vertex("A", 0, 0), _vertex("B", 4, 0), _vertex("C", 0, 3)],
            edges=[_edge(0, 1, 2), _edge(1, 2, 5)],
        )

        self._call("translate_object", name="G", x_offset=1, y_offset=2)

        self.assertEqual(sorted(self._vertex_coords("G")), [(1.0, 2.0), (1.0, 5.0), (5.0, 2.0)])
        segment = self.canvas.get_graph("G").segments[0]
        self.assertEqual(
            sorted([(segment.point1.x, segment.point1.y), (segment.point2.x, segment.point2.y)]),
            [(1.0, 2.0), (5.0, 2.0)],
        )
        self.assertEqual(self._analyze("G", "shortest_path", start="A", goal="C").get("cost"), 7.0)

    def test_translate_directed_graph_and_undo(self) -> None:
        self._graph(directed=True, vertices=[_vertex("P", 10, 10), _vertex("Q", 12, 10)], edges=[_edge(0, 1, 3)])

        self._call("translate_object", name="G", x_offset=-1, y_offset=-1)
        self.assertEqual(sorted(self._vertex_coords("G")), [(9.0, 9.0), (11.0, 9.0)])

        self._call("undo")
        self.assertEqual(sorted(self._vertex_coords("G")), [(10.0, 10.0), (12.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
