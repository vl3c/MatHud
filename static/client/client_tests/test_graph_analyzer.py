from __future__ import annotations

import unittest
from typing import List

from geometry.graph_state import (
    GraphEdgeDescriptor,
    GraphState,
    GraphVertexDescriptor,
    TreeState,
)
from utils.graph_analyzer import GraphAnalyzer


class TestAnalyzeGraphShortestPath(unittest.TestCase):
    """Tests for shortest_path operation."""

    def test_shortest_path_unweighted_simple(self) -> None:
        """Undirected graph G1: A-B, A-C, B-D, C-D, C-E, D-E, E-F. Path A to F."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E", "F"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "A", "C", name="AC"),
            GraphEdgeDescriptor("e3", "B", "D", name="BD"),
            GraphEdgeDescriptor("e4", "C", "D", name="CD"),
            GraphEdgeDescriptor("e5", "C", "E", name="CE"),
            GraphEdgeDescriptor("e6", "D", "E", name="DE"),
            GraphEdgeDescriptor("e7", "E", "F", name="EF"),
        ]
        state = GraphState("G1", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "F"})

        self.assertNotIn("error", result)
        path = result.get("path", [])
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "F")
        self.assertEqual(len(path), 4)  # A -> C -> E -> F

    def test_shortest_path_weighted_dijkstra(self) -> None:
        """Weighted graph W1: find min-cost path A to E."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=4.0, name="AB"),
            GraphEdgeDescriptor("e2", "A", "C", weight=2.0, name="AC"),
            GraphEdgeDescriptor("e3", "B", "C", weight=1.0, name="BC"),
            GraphEdgeDescriptor("e4", "B", "D", weight=5.0, name="BD"),
            GraphEdgeDescriptor("e5", "C", "D", weight=8.0, name="CD"),
            GraphEdgeDescriptor("e6", "C", "E", weight=10.0, name="CE"),
            GraphEdgeDescriptor("e7", "D", "E", weight=2.0, name="DE"),
        ]
        state = GraphState("W1", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "E"})

        self.assertNotIn("error", result)
        path = result.get("path", [])
        cost = result.get("cost")
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        self.assertIsNotNone(cost)
        # Best path: A->C=2, C->B=1, B->D=5, D->E=2 = 10
        self.assertEqual(cost, 10.0)

    def test_shortest_path_directed(self) -> None:
        """Directed graph: A->B, A->C, B->D, C->D, D->E."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "A", "C", name="AC"),
            GraphEdgeDescriptor("e3", "B", "D", name="BD"),
            GraphEdgeDescriptor("e4", "C", "D", name="CD"),
            GraphEdgeDescriptor("e5", "D", "E", name="DE"),
        ]
        state = GraphState("D1", vertices, edges, directed=True)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "E"})

        self.assertNotIn("error", result)
        path = result.get("path", [])
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        self.assertEqual(len(path), 4)  # A->B->D->E or A->C->D->E

    def test_shortest_path_no_path(self) -> None:
        """No path exists between disconnected nodes."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [GraphEdgeDescriptor("e1", "A", "B", name="AB")]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "C"})

        self.assertIsNone(result.get("path"))

    def test_shortest_path_missing_params(self) -> None:
        """Missing start/goal returns error."""
        vertices = [GraphVertexDescriptor("A"), GraphVertexDescriptor("B")]
        edges = [GraphEdgeDescriptor("e1", "A", "B")]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {})

        self.assertIn("error", result)

    def test_shortest_path_highlight_vectors(self) -> None:
        """Verify highlight_vectors contains edge names along path."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="seg_AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="seg_BC"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "C"})

        highlights = result.get("highlight_vectors", [])
        self.assertEqual(len(highlights), 2)
        self.assertIn("seg_AB", highlights)
        self.assertIn("seg_BC", highlights)


class TestAnalyzeGraphMST(unittest.TestCase):
    """Tests for minimum spanning tree (mst) operation."""

    def test_mst_weighted_graph(self) -> None:
        """MST on weighted graph W1."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=4.0, name="AB"),
            GraphEdgeDescriptor("e2", "A", "C", weight=2.0, name="AC"),
            GraphEdgeDescriptor("e3", "B", "C", weight=1.0, name="BC"),
            GraphEdgeDescriptor("e4", "B", "D", weight=5.0, name="BD"),
            GraphEdgeDescriptor("e5", "C", "D", weight=8.0, name="CD"),
            GraphEdgeDescriptor("e6", "C", "E", weight=10.0, name="CE"),
            GraphEdgeDescriptor("e7", "D", "E", weight=2.0, name="DE"),
        ]
        state = GraphState("W1", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "mst", {})

        self.assertNotIn("error", result)
        mst_edges = result.get("edges", [])
        self.assertEqual(len(mst_edges), 4)  # n-1 edges for 5 vertices
        highlights = result.get("highlight_vectors", [])
        self.assertEqual(len(highlights), 4)

    def test_mst_unweighted_graph(self) -> None:
        """MST on unweighted graph treats all edges as weight 1."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "D", name="CD"),
            GraphEdgeDescriptor("e4", "A", "D", name="AD"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "mst", {})

        mst_edges = result.get("edges", [])
        self.assertEqual(len(mst_edges), 3)


class TestAnalyzeGraphTopologicalSort(unittest.TestCase):
    """Tests for topological_sort operation."""

    def test_topological_sort_dag(self) -> None:
        """DAG: A->C, B->C, C->D, C->E, D->F, E->F."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E", "F"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "C"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "C", "E"),
            GraphEdgeDescriptor("e5", "D", "F"),
            GraphEdgeDescriptor("e6", "E", "F"),
        ]
        state = GraphState("DAG1", vertices, edges, directed=True, graph_type="dag")
        result = GraphAnalyzer.analyze(state, "topological_sort", {})

        self.assertNotIn("error", result)
        order = result.get("order", [])
        self.assertEqual(len(order), 6)
        # A and B must come before C; C before D,E; D,E before F
        self.assertLess(order.index("A"), order.index("C"))
        self.assertLess(order.index("B"), order.index("C"))
        self.assertLess(order.index("C"), order.index("D"))
        self.assertLess(order.index("C"), order.index("E"))
        self.assertLess(order.index("D"), order.index("F"))
        self.assertLess(order.index("E"), order.index("F"))

    def test_topological_sort_linear_chain(self) -> None:
        """Linear DAG: A->B->C->D."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
        ]
        state = GraphState("Chain", vertices, edges, directed=True, graph_type="dag")
        result = GraphAnalyzer.analyze(state, "topological_sort", {})

        order = result.get("order", [])
        self.assertEqual(order, ["A", "B", "C", "D"])


class TestAnalyzeGraphBridges(unittest.TestCase):
    """Tests for bridges operation."""

    def test_bridges_simple_chain(self) -> None:
        """Chain graph A-B-C-D has all edges as bridges."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "D", name="CD"),
        ]
        state = GraphState("Chain", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bridges", {})

        bridges = result.get("bridges", [])
        self.assertEqual(len(bridges), 3)
        highlights = result.get("highlight_vectors", [])
        self.assertEqual(len(highlights), 3)

    def test_bridges_cycle_no_bridges(self) -> None:
        """Cycle A-B-C-D-A has no bridges."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "D", name="CD"),
            GraphEdgeDescriptor("e4", "D", "A", name="DA"),
        ]
        state = GraphState("Cycle", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bridges", {})

        bridges = result.get("bridges", [])
        self.assertEqual(len(bridges), 0)

    def test_bridges_mixed(self) -> None:
        """Graph with one bridge: cycle A-B-C-A plus D connected only to C."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "A", name="CA"),
            GraphEdgeDescriptor("e4", "C", "D", name="CD"),
        ]
        state = GraphState("Mixed", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bridges", {})

        bridges = result.get("bridges", [])
        self.assertEqual(len(bridges), 1)
        bridge_vertices = bridges[0]
        self.assertIn("C", bridge_vertices)
        self.assertIn("D", bridge_vertices)


class TestAnalyzeGraphArticulationPoints(unittest.TestCase):
    """Tests for articulation_points operation."""

    def test_articulation_points_chain(self) -> None:
        """Chain A-B-C-D: B and C are articulation points."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
        ]
        state = GraphState("Chain", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "articulation_points", {})

        points = result.get("articulation_points", [])
        self.assertIn("B", points)
        self.assertIn("C", points)
        self.assertEqual(len(points), 2)

    def test_articulation_points_cycle(self) -> None:
        """Cycle has no articulation points."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "D", "A"),
        ]
        state = GraphState("Cycle", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "articulation_points", {})

        points = result.get("articulation_points", [])
        self.assertEqual(len(points), 0)


class TestAnalyzeGraphEulerStatus(unittest.TestCase):
    """Tests for euler_status operation."""

    def test_euler_status_eulerian_cycle(self) -> None:
        """Even-degree cycle has Euler cycle."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "D", "A"),
        ]
        state = GraphState("Cycle4", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "euler_status", {})

        status = result.get("status")
        self.assertEqual(status, "cycle")

    def test_euler_status_semi_eulerian(self) -> None:
        """Chain A-B-C has exactly two odd-degree vertices, so has Euler path."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
        ]
        state = GraphState("Chain", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "euler_status", {})

        status = result.get("status")
        self.assertEqual(status, "path")

    def test_euler_status_not_eulerian(self) -> None:
        """Star graph with 4 leaves: center has degree 4, leaves have degree 1 (4 odd vertices)."""
        vertices = [GraphVertexDescriptor(v) for v in ["C", "A", "B", "D", "E"]]
        edges = [
            GraphEdgeDescriptor("e1", "C", "A"),
            GraphEdgeDescriptor("e2", "C", "B"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "C", "E"),
        ]
        state = GraphState("Star", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "euler_status", {})

        status = result.get("status")
        self.assertIsNone(status)  # More than 2 odd-degree vertices


class TestAnalyzeGraphBipartite(unittest.TestCase):
    """Tests for bipartite operation."""

    def test_bipartite_true_even_cycle(self) -> None:
        """Even cycle is bipartite."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "D", "A"),
        ]
        state = GraphState("C4", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})

        self.assertTrue(result.get("is_bipartite"))
        coloring = result.get("coloring", {})
        self.assertEqual(len(coloring), 4)

    def test_bipartite_false_odd_cycle(self) -> None:
        """Odd cycle (triangle) is not bipartite."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "A"),
        ]
        state = GraphState("K3", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})

        self.assertFalse(result.get("is_bipartite"))

    def test_bipartite_complete_bipartite_k23(self) -> None:
        """K_{2,3} is bipartite: partition {A,B} and {C,D,E}."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D", "E"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "C"),
            GraphEdgeDescriptor("e2", "A", "D"),
            GraphEdgeDescriptor("e3", "A", "E"),
            GraphEdgeDescriptor("e4", "B", "C"),
            GraphEdgeDescriptor("e5", "B", "D"),
            GraphEdgeDescriptor("e6", "B", "E"),
        ]
        state = GraphState("K23", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})

        self.assertTrue(result.get("is_bipartite"))
        coloring = result.get("coloring", {})
        # A and B should have same color; C, D, E should have the other
        self.assertEqual(coloring["A"], coloring["B"])
        self.assertEqual(coloring["C"], coloring["D"])
        self.assertEqual(coloring["D"], coloring["E"])
        self.assertNotEqual(coloring["A"], coloring["C"])


class TestAnalyzeGraphBFSDFS(unittest.TestCase):
    """Tests for bfs and dfs operations."""

    def test_bfs_order(self) -> None:
        """BFS from A in graph A-B, A-C, B-D, C-D."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
            GraphEdgeDescriptor("e4", "C", "D"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bfs", {"start": "A"})

        order = result.get("order", [])
        self.assertEqual(order[0], "A")
        self.assertIn("B", order[1:3])
        self.assertIn("C", order[1:3])
        self.assertEqual(order[-1], "D")

    def test_dfs_order(self) -> None:
        """DFS from A in graph A-B, A-C, B-D."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "dfs", {"start": "A"})

        order = result.get("order", [])
        self.assertEqual(order[0], "A")
        self.assertEqual(len(order), 4)

    def test_bfs_missing_start(self) -> None:
        """BFS without start returns None order."""
        vertices = [GraphVertexDescriptor("A")]
        state = GraphState("G", vertices, [], directed=False)
        result = GraphAnalyzer.analyze(state, "bfs", {})

        self.assertIsNone(result.get("order"))


class TestAnalyzeGraphTreeOperations(unittest.TestCase):
    """Tests for tree-specific operations: levels, diameter, lca."""

    def _make_tree_state(self, name: str, vertex_names: List[str], edge_pairs: List[tuple], root: str) -> TreeState:
        vertices = [GraphVertexDescriptor(v) for v in vertex_names]
        edges = [
            GraphEdgeDescriptor(f"e{i}", src, tgt, name=f"{src}{tgt}")
            for i, (src, tgt) in enumerate(edge_pairs)
        ]
        return TreeState(name, vertices, edges, root=root)

    def test_levels_simple_tree(self) -> None:
        """Tree T1 rooted at A: A-B, A-C, B-D, B-E, C-F, C-G."""
        state = self._make_tree_state(
            "T1",
            ["A", "B", "C", "D", "E", "F", "G"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "levels", {"root": "A"})

        levels = result.get("levels", [])
        self.assertEqual(levels[0], ["A"])
        self.assertEqual(set(levels[1]), {"B", "C"})
        self.assertEqual(set(levels[2]), {"D", "E", "F", "G"})

    def test_diameter_linear_tree(self) -> None:
        """Linear tree A-B-C-D-E has diameter path of length 4 (5 nodes)."""
        state = self._make_tree_state(
            "Linear",
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
            root="C",
        )
        result = GraphAnalyzer.analyze(state, "diameter", {})

        path = result.get("path", [])
        self.assertEqual(len(path), 5)
        self.assertIn("A", path)
        self.assertIn("E", path)

    def test_diameter_star_tree(self) -> None:
        """Star tree with center C and leaves A,B,D,E has diameter 2 (3 nodes)."""
        state = self._make_tree_state(
            "Star",
            ["C", "A", "B", "D", "E"],
            [("C", "A"), ("C", "B"), ("C", "D"), ("C", "E")],
            root="C",
        )
        result = GraphAnalyzer.analyze(state, "diameter", {})

        path = result.get("path", [])
        self.assertEqual(len(path), 3)  # leaf - center - leaf

    def test_lca_sibling_nodes(self) -> None:
        """LCA of siblings D and E in tree rooted at A with A-B, B-D, B-E is B."""
        state = self._make_tree_state(
            "T",
            ["A", "B", "D", "E"],
            [("A", "B"), ("B", "D"), ("B", "E")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "lca", {"root": "A", "a": "D", "b": "E"})

        self.assertEqual(result.get("lca"), "B")

    def test_lca_ancestor_descendant(self) -> None:
        """LCA of A and D where D is descendant of A via B is A itself."""
        state = self._make_tree_state(
            "T",
            ["A", "B", "D"],
            [("A", "B"), ("B", "D")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "lca", {"root": "A", "a": "A", "b": "D"})

        self.assertEqual(result.get("lca"), "A")

    def test_lca_cousins(self) -> None:
        """LCA of cousins E and G in tree A-B, A-C, B-D, B-E, C-F, C-G is A."""
        state = self._make_tree_state(
            "T1",
            ["A", "B", "C", "D", "E", "F", "G"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "lca", {"root": "A", "a": "E", "b": "G"})

        self.assertEqual(result.get("lca"), "A")

    def test_lca_missing_params(self) -> None:
        """LCA without a and b returns error."""
        state = self._make_tree_state("T", ["A", "B"], [("A", "B")], root="A")
        result = GraphAnalyzer.analyze(state, "lca", {"root": "A"})

        self.assertIn("error", result)


class TestAnalyzeGraphTreeTransforms(unittest.TestCase):
    """Tests for tree transform operations: reroot, balance_children, invert_children."""

    def _make_tree_state(self, name: str, vertex_names: List[str], edge_pairs: List[tuple], root: str) -> TreeState:
        vertices = [GraphVertexDescriptor(v) for v in vertex_names]
        edges = [
            GraphEdgeDescriptor(f"e{i}", src, tgt)
            for i, (src, tgt) in enumerate(edge_pairs)
        ]
        return TreeState(name, vertices, edges, root=root)

    def test_reroot_simple(self) -> None:
        """Reroot tree A-B-C from A to B."""
        state = self._make_tree_state("T", ["A", "B", "C"], [("A", "B"), ("B", "C")], root="A")
        result = GraphAnalyzer.analyze(state, "reroot", {"root": "A", "new_root": "B"})

        self.assertNotIn("error", result)
        parent = result.get("parent", {})
        self.assertIsNone(parent.get("B"))  # B is new root
        self.assertEqual(parent.get("A"), "B")
        self.assertEqual(parent.get("C"), "B")

    def test_reroot_to_leaf(self) -> None:
        """Reroot tree A-B, A-C from A to C."""
        state = self._make_tree_state("T", ["A", "B", "C"], [("A", "B"), ("A", "C")], root="A")
        result = GraphAnalyzer.analyze(state, "reroot", {"root": "A", "new_root": "C"})

        parent = result.get("parent", {})
        self.assertIsNone(parent.get("C"))
        self.assertEqual(parent.get("A"), "C")
        self.assertEqual(parent.get("B"), "A")

    def test_reroot_missing_new_root(self) -> None:
        """Reroot without new_root returns error."""
        state = self._make_tree_state("T", ["A", "B"], [("A", "B")], root="A")
        result = GraphAnalyzer.analyze(state, "reroot", {"root": "A"})

        self.assertIn("error", result)

    def test_balance_children(self) -> None:
        """Balance children sorts by subtree size."""
        state = self._make_tree_state(
            "T",
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "balance_children", {"root": "A"})

        self.assertNotIn("error", result)
        children = result.get("children", {})
        self.assertIn("A", children)
        # B has 2 descendants, C has 0, so B should come after C in balanced order
        a_children = children.get("A", [])
        self.assertEqual(len(a_children), 2)

    def test_invert_children(self) -> None:
        """Invert children reverses child order."""
        state = self._make_tree_state(
            "T",
            ["A", "B", "C", "D"],
            [("A", "B"), ("A", "C"), ("A", "D")],
            root="A",
        )
        result = GraphAnalyzer.analyze(state, "invert_children", {"root": "A"})

        self.assertNotIn("error", result)
        children = result.get("children", {})
        a_children = children.get("A", [])
        self.assertEqual(len(a_children), 3)


class TestAnalyzeGraphEdgeCases(unittest.TestCase):
    """Edge case and error handling tests."""

    def test_unsupported_operation(self) -> None:
        """Unknown operation returns error."""
        vertices = [GraphVertexDescriptor("A")]
        state = GraphState("G", vertices, [], directed=False)
        result = GraphAnalyzer.analyze(state, "unknown_op", {})

        self.assertIn("error", result)

    def test_resolve_root_old_format(self) -> None:
        """Test that _resolve_root handles old internal ID format like 'v0'."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
        ]
        # Simulate old format: root is "v0" but adjacency uses point names
        state = TreeState("T", vertices, edges, root="v0")
        adjacency = {"A": {"B", "C"}, "B": {"A"}, "C": {"A"}}

        resolved = GraphAnalyzer._resolve_root(state, adjacency, {})
        self.assertEqual(resolved, "A")

    def test_resolve_root_already_valid(self) -> None:
        """Test that _resolve_root returns root as-is if already in adjacency."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
        ]
        state = TreeState("T", vertices, edges, root="A")
        adjacency = {"A": {"B", "C"}, "B": {"A"}, "C": {"A"}}

        resolved = GraphAnalyzer._resolve_root(state, adjacency, {})
        self.assertEqual(resolved, "A")

    def test_resolve_root_from_params(self) -> None:
        """Test that _resolve_root prefers params over state.root."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [GraphEdgeDescriptor("e1", "A", "B")]
        state = TreeState("T", vertices, edges, root="A")
        adjacency = {"A": {"B"}, "B": {"A"}, "C": set()}

        resolved = GraphAnalyzer._resolve_root(state, adjacency, {"root": "B"})
        self.assertEqual(resolved, "B")

    def test_empty_graph(self) -> None:
        """Operations on empty graph don't crash."""
        state = GraphState("Empty", [], [], directed=False)

        result = GraphAnalyzer.analyze(state, "bfs", {"start": "X"})
        self.assertIsNone(result.get("order"))

        result = GraphAnalyzer.analyze(state, "bridges", {})
        self.assertEqual(result.get("bridges", []), [])

    def test_single_vertex(self) -> None:
        """Single vertex graph operations."""
        vertices = [GraphVertexDescriptor("A")]
        state = GraphState("Single", vertices, [], directed=False)

        # BFS on single vertex with no edges - adjacency is empty, so returns None
        result = GraphAnalyzer.analyze(state, "bfs", {"start": "A"})
        self.assertIsNone(result.get("order"))

        # Euler status on empty graph: connected (trivially) with 0 odd-degree vertices
        result = GraphAnalyzer.analyze(state, "euler_status", {})
        self.assertEqual(result.get("status"), "cycle")


class TestAnalyzeGraphCompleteGraph(unittest.TestCase):
    """Tests using complete graphs (Kn)."""

    def test_k4_bridges(self) -> None:
        """K4 has no bridges."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor(f"e{i}", src, tgt)
            for i, (src, tgt) in enumerate([
                ("A", "B"), ("A", "C"), ("A", "D"),
                ("B", "C"), ("B", "D"), ("C", "D"),
            ])
        ]
        state = GraphState("K4", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bridges", {})

        self.assertEqual(len(result.get("bridges", [])), 0)

    def test_k4_not_bipartite(self) -> None:
        """K4 is not bipartite (contains K3)."""
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor(f"e{i}", src, tgt)
            for i, (src, tgt) in enumerate([
                ("A", "B"), ("A", "C"), ("A", "D"),
                ("B", "C"), ("B", "D"), ("C", "D"),
            ])
        ]
        state = GraphState("K4", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})

        self.assertFalse(result.get("is_bipartite"))


class TestAnalyzeGraphGridGraph(unittest.TestCase):
    """Tests on grid-structured graphs."""

    def test_grid_2x2_bipartite(self) -> None:
        """2x2 grid is bipartite."""
        vertices = [GraphVertexDescriptor(v) for v in ["r1c1", "r1c2", "r2c1", "r2c2"]]
        edges = [
            GraphEdgeDescriptor("e1", "r1c1", "r1c2"),
            GraphEdgeDescriptor("e2", "r1c1", "r2c1"),
            GraphEdgeDescriptor("e3", "r1c2", "r2c2"),
            GraphEdgeDescriptor("e4", "r2c1", "r2c2"),
        ]
        state = GraphState("Grid2x2", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})

        self.assertTrue(result.get("is_bipartite"))

    def test_grid_shortest_path(self) -> None:
        """Shortest path across 2x2 grid corners."""
        vertices = [GraphVertexDescriptor(v) for v in ["r1c1", "r1c2", "r2c1", "r2c2"]]
        edges = [
            GraphEdgeDescriptor("e1", "r1c1", "r1c2", name="e1"),
            GraphEdgeDescriptor("e2", "r1c1", "r2c1", name="e2"),
            GraphEdgeDescriptor("e3", "r1c2", "r2c2", name="e3"),
            GraphEdgeDescriptor("e4", "r2c1", "r2c2", name="e4"),
        ]
        state = GraphState("Grid2x2", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "r1c1", "goal": "r2c2"})

        path = result.get("path", [])
        self.assertEqual(path[0], "r1c1")
        self.assertEqual(path[-1], "r2c2")
        self.assertEqual(len(path), 3)  # Corner to corner in 2 steps


if __name__ == "__main__":
    unittest.main()

