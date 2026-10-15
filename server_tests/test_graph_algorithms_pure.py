"""Pure Python tests for the graph algorithms (utils.graph_utils, utils.graph_analyzer, utils.graph_layout).

The Brython suites (static/client/client_tests/test_graph_*.py) cover the same modules in the
browser and the graph tools on a canvas.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import unittest
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
from geometry.graph_state import GraphEdgeDescriptor, GraphState, GraphVertexDescriptor, TreeState
from utils.graph_analyzer import GraphAnalyzer
from utils.graph_layout import layout_vertices
from utils.graph_utils import Edge, GraphUtils


def _state(
    names: str,
    edges: Sequence[Tuple[Any, ...]],
    *,
    directed: bool = False,
    tree_root: Optional[str] = None,
    tree: bool = False,
) -> GraphState:
    vertices = [GraphVertexDescriptor(n, name=n, x=float(i), y=0.0) for i, n in enumerate(names)]
    descriptors = [
        GraphEdgeDescriptor(
            f"e{k}", spec[0], spec[1], weight=spec[2] if len(spec) > 2 else None, name=f"{spec[0]}{spec[1]}_{k}"
        )
        for k, spec in enumerate(edges)
    ]
    if tree:
        return TreeState("T", vertices, descriptors, root=tree_root)
    return GraphState("G", vertices, descriptors, directed=directed)


def _all_pairs_distances(names: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """Floyd-Warshall hop distances on the undirected graph, as ground truth."""
    inf = float("inf")
    dist = {(u, v): (0.0 if u == v else inf) for u in names for v in names}
    for u, v in edges:
        if u != v:
            dist[(u, v)] = dist[(v, u)] = 1.0
    for k, i, j in itertools.product(names, names, names):
        if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
            dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    return dist


class TestDiameter(unittest.TestCase):
    def test_matches_all_pairs_distances_on_random_connected_graphs(self) -> None:
        rng = random.Random(3)
        checked = 0
        while checked < 300:
            n = rng.randint(2, 8)
            names = [chr(65 + i) for i in range(n)]
            edges = {tuple(sorted(rng.sample(names, 2))) for _ in range(rng.randint(n - 1, 2 * n))}
            dist = _all_pairs_distances(names, list(edges))  # type: ignore[arg-type]
            if any(d == float("inf") for d in dist.values()):
                continue
            checked += 1
            result = GraphAnalyzer.analyze(_state("".join(names), sorted(edges)), "diameter")
            self.assertEqual(result["length"], max(dist.values()), sorted(edges))
            path = result["path"]
            self.assertEqual(len(path) - 1, result["length"])
            self.assertEqual(dist[(path[0], path[-1])], result["length"])
            self.assertTrue(result["connected"])

    def test_cycle_with_chord_where_a_double_sweep_is_wrong(self) -> None:
        # Hexagon A..F with chord A-D: B and E are 3 apart (B-A-D-E).
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A"), ("A", "D")]
        result = GraphAnalyzer.analyze(_state("ABCDEF", edges), "diameter")
        self.assertEqual(max(_all_pairs_distances("ABCDEF", edges).values()), 3)
        self.assertEqual(result["length"], 3)

    def test_directed_graph_uses_the_underlying_undirected_graph(self) -> None:
        result = GraphAnalyzer.analyze(_state("ABCD", [("A", "B"), ("B", "C"), ("C", "D")], directed=True), "diameter")
        self.assertEqual(result["length"], 3)
        self.assertEqual(sorted([result["path"][0], result["path"][-1]]), ["A", "D"])

    def test_forest_reports_disconnected_and_largest_component(self) -> None:
        edges = [("A", "B"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G")]
        result = GraphAnalyzer.analyze(_state("ABCDEFG", edges), "diameter")
        self.assertFalse(result["connected"])
        self.assertEqual(result["length"], 4)
        self.assertIn("disconnected", result["note"])


class TestVertexValidation(unittest.TestCase):
    def test_unknown_start_or_goal_lists_vertices(self) -> None:
        result = GraphAnalyzer.analyze(_state("ABC", [("A", "B")]), "shortest_path", {"start": "A", "goal": "Z"})
        self.assertIn("goal='Z'", result["error"])
        self.assertIn("A, B, C", result["error"])

    def test_missing_start_is_an_error(self) -> None:
        self.assertIn("start", GraphAnalyzer.analyze(_state("AB", [("A", "B")]), "bfs", {})["error"])

    def test_unknown_root_is_an_error(self) -> None:
        result = GraphAnalyzer.analyze(_state("ABC", [("A", "B"), ("A", "C")]), "levels", {"root": "Z"})
        self.assertIn("Unknown root", result["error"])

    def test_directed_default_root_is_the_unique_source(self) -> None:
        state = _state("ABC", [("B", "A"), ("B", "C")], directed=True)
        self.assertEqual(GraphAnalyzer.analyze(state, "levels")["levels"], [["B"], ["A", "C"]])

    def test_directed_without_unique_source_asks_for_a_root(self) -> None:
        state = _state("ABC", [("A", "B"), ("C", "B")], directed=True)
        self.assertIn("Pass a root", GraphAnalyzer.analyze(state, "levels")["error"])
        self.assertEqual(GraphAnalyzer.analyze(state, "levels", {"root": "B"})["levels"], [["B"], ["A", "C"]])

    def test_shortest_path_to_itself(self) -> None:
        state = _state("ABC", [("A", "B", 2.0)])
        isolated = GraphAnalyzer.analyze(state, "shortest_path", {"start": "C", "goal": "C"})
        self.assertEqual((isolated["path"], isolated["cost"]), (["C"], 0.0))
        connected = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "A"})
        self.assertEqual((connected["path"], connected["cost"]), (["A"], 0.0))


class TestTopologicalSort(unittest.TestCase):
    def test_undirected_graph_is_an_error(self) -> None:
        result = GraphAnalyzer.analyze(_state("AB", [("A", "B")]), "topological_sort")
        self.assertIn("directed graph", result["error"])

    def test_cycle_is_named(self) -> None:
        state = _state("ABCD", [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")], directed=True)
        self.assertIn("A -> B -> C -> A", GraphAnalyzer.analyze(state, "topological_sort")["error"])

    def test_order_is_lexicographically_smallest(self) -> None:
        state = _state("ABCDE", [("C", "A"), ("B", "A"), ("D", "E")], directed=True)
        self.assertEqual(GraphAnalyzer.analyze(state, "topological_sort")["order"], ["B", "C", "A", "D", "E"])

    def test_find_directed_cycle(self) -> None:
        self.assertIsNone(GraphUtils.find_directed_cycle({"A": {"B"}, "B": set()}))
        self.assertEqual(GraphUtils.find_directed_cycle({"A": {"A"}}), ["A", "A"])


class TestTreeOperations(unittest.TestCase):
    def test_tree_operations_refuse_graphs_with_cycles(self) -> None:
        cycle = _state("ABCDE", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")])
        for operation, params in [
            ("lca", {"root": "A", "a": "C", "b": "D"}),
            ("reroot", {"root": "A", "new_root": "C"}),
            ("balance_children", {}),
            ("invert_children", {}),
        ]:
            self.assertIn("has a cycle", GraphAnalyzer.analyze(cycle, operation, params)["error"], operation)

    def test_levels_of_a_graph_that_is_not_a_tree_are_bfs_distances(self) -> None:
        cycle = _state("ABCDE", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")])
        result = GraphAnalyzer.analyze(cycle, "levels", {"root": "A"})
        self.assertEqual(result["levels"], [["A"], ["B", "E"], ["C", "D"]])
        self.assertIn("not a tree", result["note"])
        dag = _state("ABCD", [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")], directed=True)
        self.assertEqual(GraphAnalyzer.analyze(dag, "levels")["levels"], [["A"], ["B", "C"], ["D"]])

    def test_tree_operations_refuse_forests(self) -> None:
        forest = _state("ABCD", [("A", "B"), ("C", "D")], tree=True, tree_root="A")
        self.assertIn("disconnected", GraphAnalyzer.analyze(forest, "lca", {"a": "A", "b": "B"})["error"])

    def test_lca_uses_depths_of_its_parent_map(self) -> None:
        tree = _state("ABCDEFG", [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("F", "G")])
        for a, b, expected in [("D", "E", "B"), ("E", "G", "A"), ("G", "C", "C"), ("A", "G", "A")]:
            result = GraphAnalyzer.analyze(tree, "lca", {"root": "A", "a": a, "b": b})
            self.assertEqual(result["lca"], expected, (a, b))

    def test_depths_from_parents(self) -> None:
        parent: Dict[str, Optional[str]] = {"A": None, "B": "A", "C": "B", "D": "A"}
        self.assertEqual(GraphUtils.depths_from_parents(parent), {"A": 0, "B": 1, "C": 2, "D": 1})


class TestParallelEdges(unittest.TestCase):
    def test_cheapest_parallel_edge_wins_everywhere(self) -> None:
        state = _state("AB", [("A", "B", 5.0), ("B", "A", 1.0)])
        self.assertEqual(GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "B"})["cost"], 1.0)
        self.assertEqual(GraphAnalyzer.analyze(state, "mst")["highlight_vectors"], ["BA_1"])
        matrix = GraphUtils.adjacency_matrix_from_descriptors(state.vertices, state.edges, directed=False)
        self.assertEqual(matrix, [[0.0, 1.0], [1.0, 0.0]])

    def test_unweighted_parallel_edge_counts_as_one(self) -> None:
        state = _state("AB", [("A", "B", 5.0), ("A", "B")], directed=True)
        self.assertEqual(GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "B"})["cost"], 1.0)

    def test_mst_on_directed_graph_is_the_undirected_mst(self) -> None:
        state = _state("ABC", [("A", "B", 1.0), ("C", "B", 2.0), ("A", "C", 5.0)], directed=True)
        result = GraphAnalyzer.analyze(state, "mst")
        self.assertEqual(sorted(sorted(e) for e in result["edges"]), [["A", "B"], ["B", "C"]])

    def test_cheapest_parallel_edges_keys(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "A")]
        weights = {("A", "B"): 3.0, ("B", "A"): 2.0}
        undirected = GraphUtils.cheapest_parallel_edges(edges, lambda e: weights[e.as_tuple()], directed=False)
        directed = GraphUtils.cheapest_parallel_edges(edges, lambda e: weights[e.as_tuple()], directed=True)
        self.assertEqual([w for _, w in undirected.values()], [2.0])
        self.assertEqual(sorted(w for _, w in directed.values()), [2.0, 3.0])


_DETERMINISM_SCRIPT = """
import json, sys
sys.path.insert(0, ".")
sys.path.insert(0, "static/client")
from server_tests import client_renderer
from geometry.graph_state import GraphEdgeDescriptor as E, GraphState, GraphVertexDescriptor as V
from utils.graph_analyzer import GraphAnalyzer as A
names = "ABCDEFGH"
star = [E("e%d" % k, "A", t) for k, t in enumerate(names[1:])]
diamond = [E("e0", "A", "B"), E("e1", "A", "C"), E("e2", "B", "D"), E("e3", "C", "D")]
directed = GraphState("D", [V(n) for n in names], star, directed=True)
undirected = GraphState("U", [V(n) for n in names], star)
square = GraphState("S", [V(n) for n in "ABCD"], diamond)
print(json.dumps([
    A.analyze(directed, "bfs", {"start": "A"}),
    A.analyze(directed, "topological_sort"),
    A.analyze(undirected, "levels", {"root": "A"}),
    A.analyze(square, "shortest_path", {"start": "A", "goal": "D"}),
    A.analyze(square, "bridges"),
    A.analyze(square, "articulation_points"),
]))
"""


class TestDeterminism(unittest.TestCase):
    def test_results_do_not_depend_on_hash_seed(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs: Set[str] = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            completed = subprocess.run(
                [sys.executable, "-c", _DETERMINISM_SCRIPT],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.add(completed.stdout)
        self.assertEqual(len(outputs), 1)
        bfs = json.loads(outputs.pop())[0]["order"]
        self.assertEqual(bfs, list("ABCDEFGH"))


class TestLayout(unittest.TestCase):
    box = {"x": -100.0, "y": -80.0, "width": 200.0, "height": 160.0}

    def _layout(self, ids: List[str], edges: List[Edge[str]], layout: Optional[str]) -> Dict[str, Tuple[float, float]]:
        positions: Dict[str, Tuple[float, float]] = layout_vertices(
            ids, edges, layout=layout, placement_box=self.box, canvas_width=1000.0, canvas_height=800.0
        )
        return positions

    def test_grid_layout_places_components_apart(self) -> None:
        positions = self._layout(["A", "B", "C", "D"], [Edge("A", "B"), Edge("C", "D")], "grid")
        self.assertEqual(len(set(positions.values())), 4)

    def test_explicit_layout_is_kept_for_a_tree(self) -> None:
        edges = [Edge("R", "A"), Edge("R", "B"), Edge("R", "C")]
        positions = self._layout(["R", "A", "B", "C"], edges, "circular")
        radii = {round((x * x + y * y) ** 0.5, 6) for x, y in positions.values()}
        self.assertEqual(len(radii), 1)

    def test_no_layout_stacks_vertices_or_leaves_the_box(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            n = rng.choice([1, 2, 3, 5, 8, 10])
            ids = [f"v{i}" for i in range(n)]
            edges = [Edge(rng.choice(ids), rng.choice(ids)) for _ in range(rng.randint(0, 2 * n))]
            layout = rng.choice([None, "circular", "grid", "radial", "tree", "hierarchical", "force"])
            positions = self._layout(ids, edges, layout)
            self.assertEqual(set(positions), set(ids))
            coords = list(positions.values())
            for (x1, y1), (x2, y2) in itertools.combinations(coords, 2):
                self.assertGreater(abs(x1 - x2) + abs(y1 - y2), 1e-6, (layout, edges))
            for x, y in coords:
                self.assertTrue(
                    -100.0 - 1e-6 <= x <= 100.0 + 1e-6 and -80.0 - 1e-6 <= y <= 80.0 + 1e-6, (layout, edges)
                )


if __name__ == "__main__":
    unittest.main()
