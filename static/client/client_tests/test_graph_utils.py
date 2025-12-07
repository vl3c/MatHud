from __future__ import annotations

import unittest

from geometry.graph_state import GraphEdgeDescriptor, GraphState, GraphVertexDescriptor, TreeState
from utils.graph_analyzer import GraphAnalyzer
from utils.graph_utils import Edge, GraphUtils


class TestGraph(unittest.TestCase):

    # ------------------------------------------------------------------
    # Edge
    # ------------------------------------------------------------------

    def test_edge_creation(self) -> None:
        edge = Edge("A", "B")
        self.assertEqual(edge.source, "A")
        self.assertEqual(edge.target, "B")

    def test_edge_reversed(self) -> None:
        edge = Edge("A", "B")
        reversed_edge = edge.reversed()
        self.assertEqual(reversed_edge.source, "B")
        self.assertEqual(reversed_edge.target, "A")

    def test_edge_as_tuple(self) -> None:
        edge = Edge("X", "Y")
        self.assertEqual(edge.as_tuple(), ("X", "Y"))

    def test_edge_as_frozenset(self) -> None:
        edge1 = Edge("A", "B")
        edge2 = Edge("B", "A")
        self.assertEqual(edge1.as_frozenset(), edge2.as_frozenset())
        self.assertEqual(edge1.as_frozenset(), frozenset({"A", "B"}))

    def test_edge_equality_ordered(self) -> None:
        edge1 = Edge("A", "B")
        edge2 = Edge("A", "B")
        edge3 = Edge("B", "A")
        self.assertEqual(edge1, edge2)
        self.assertNotEqual(edge1, edge3)

    def test_edge_hash(self) -> None:
        edge1 = Edge("A", "B")
        edge2 = Edge("A", "B")
        edge3 = Edge("B", "A")
        self.assertEqual(hash(edge1), hash(edge2))
        self.assertNotEqual(hash(edge1), hash(edge3))

    def test_edge_repr(self) -> None:
        edge = Edge("A", "B")
        self.assertEqual(repr(edge), "Edge('A', 'B')")

    # ------------------------------------------------------------------
    # build_adjacency_map (undirected)
    # ------------------------------------------------------------------

    def test_build_adjacency_map_empty(self) -> None:
        adjacency = GraphUtils.build_adjacency_map([])
        self.assertEqual(adjacency, {})

    def test_build_adjacency_map_single_edge(self) -> None:
        edges = [Edge("A", "B")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B"})
        self.assertEqual(adjacency["B"], {"A"})

    def test_build_adjacency_map_triangle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B", "C"})
        self.assertEqual(adjacency["B"], {"A", "C"})
        self.assertEqual(adjacency["C"], {"A", "B"})

    def test_build_adjacency_map_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B"})
        self.assertEqual(adjacency["B"], {"A", "C"})
        self.assertEqual(adjacency["C"], {"B", "D"})
        self.assertEqual(adjacency["D"], {"C"})

    # ------------------------------------------------------------------
    # build_directed_adjacency_map
    # ------------------------------------------------------------------

    def test_build_directed_adjacency_map_single_edge(self) -> None:
        edges = [Edge("A", "B")]
        adjacency = GraphUtils.build_directed_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B"})
        self.assertEqual(adjacency["B"], set())

    def test_build_directed_adjacency_map_bidirectional(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "A")]
        adjacency = GraphUtils.build_directed_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B"})
        self.assertEqual(adjacency["B"], {"A"})

    def test_build_directed_adjacency_map_chain(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        adjacency = GraphUtils.build_directed_adjacency_map(edges)
        self.assertEqual(adjacency["A"], {"B"})
        self.assertEqual(adjacency["B"], {"C"})
        self.assertEqual(adjacency["C"], {"D"})
        self.assertEqual(adjacency["D"], set())

    # ------------------------------------------------------------------
    # get_all_vertices
    # ------------------------------------------------------------------

    def test_get_all_vertices_empty(self) -> None:
        vertices = GraphUtils.get_all_vertices([])
        self.assertEqual(vertices, set())

    def test_get_all_vertices_triangle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        vertices = GraphUtils.get_all_vertices(edges)
        self.assertEqual(vertices, {"A", "B", "C"})

    # ------------------------------------------------------------------
    # get_vertex_degrees (undirected)
    # ------------------------------------------------------------------

    def test_get_vertex_degrees_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        degrees = GraphUtils.get_vertex_degrees(adjacency)
        self.assertEqual(degrees["A"], 1)
        self.assertEqual(degrees["B"], 2)
        self.assertEqual(degrees["C"], 2)
        self.assertEqual(degrees["D"], 1)

    def test_get_vertex_degrees_cycle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        degrees = GraphUtils.get_vertex_degrees(adjacency)
        self.assertEqual(degrees["A"], 2)
        self.assertEqual(degrees["B"], 2)
        self.assertEqual(degrees["C"], 2)

    def test_get_vertex_degrees_t_structure(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("B", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        degrees = GraphUtils.get_vertex_degrees(adjacency)
        self.assertEqual(degrees["A"], 1)
        self.assertEqual(degrees["B"], 3)
        self.assertEqual(degrees["C"], 1)
        self.assertEqual(degrees["D"], 1)

    # ------------------------------------------------------------------
    # get_out_degrees / get_in_degrees (directed)
    # ------------------------------------------------------------------

    def test_get_out_degrees(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "C")]
        adjacency = GraphUtils.build_directed_adjacency_map(edges)
        out_degrees = GraphUtils.get_out_degrees(adjacency)
        self.assertEqual(out_degrees["A"], 2)
        self.assertEqual(out_degrees["B"], 1)
        self.assertEqual(out_degrees["C"], 0)

    def test_get_in_degrees(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "C")]
        adjacency = GraphUtils.build_directed_adjacency_map(edges)
        in_degrees = GraphUtils.get_in_degrees(adjacency)
        self.assertEqual(in_degrees["A"], 0)
        self.assertEqual(in_degrees["B"], 1)
        self.assertEqual(in_degrees["C"], 2)

    # ------------------------------------------------------------------
    # get_endpoints
    # ------------------------------------------------------------------

    def test_get_endpoints_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        endpoints = GraphUtils.get_endpoints(adjacency)
        self.assertEqual(endpoints, {"A", "D"})

    def test_get_endpoints_cycle_none(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        endpoints = GraphUtils.get_endpoints(adjacency)
        self.assertEqual(endpoints, set())

    def test_get_endpoints_single_edge(self) -> None:
        edges = [Edge("X", "Y")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        endpoints = GraphUtils.get_endpoints(adjacency)
        self.assertEqual(endpoints, {"X", "Y"})

    # ------------------------------------------------------------------
    # is_connected
    # ------------------------------------------------------------------

    def test_is_connected_empty(self) -> None:
        self.assertTrue(GraphUtils.is_connected({}))

    def test_is_connected_single_edge(self) -> None:
        edges = [Edge("A", "B")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertTrue(GraphUtils.is_connected(adjacency))

    def test_is_connected_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertTrue(GraphUtils.is_connected(adjacency))

    def test_is_connected_disconnected(self) -> None:
        edges = [Edge("A", "B"), Edge("C", "D")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertFalse(GraphUtils.is_connected(adjacency))

    def test_is_connected_triangle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        self.assertTrue(GraphUtils.is_connected(adjacency))

    # ------------------------------------------------------------------
    # get_connected_components
    # ------------------------------------------------------------------

    def test_get_connected_components_empty(self) -> None:
        components = GraphUtils.get_connected_components({})
        self.assertEqual(components, [])

    def test_get_connected_components_single(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        components = GraphUtils.get_connected_components(adjacency)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0], {"A", "B", "C"})

    def test_get_connected_components_two_components(self) -> None:
        edges = [Edge("A", "B"), Edge("C", "D"), Edge("D", "E")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        components = GraphUtils.get_connected_components(adjacency)
        self.assertEqual(len(components), 2)
        component_sets = [frozenset(c) for c in components]
        self.assertIn(frozenset({"A", "B"}), component_sets)
        self.assertIn(frozenset({"C", "D", "E"}), component_sets)

    def test_get_connected_components_three_isolated(self) -> None:
        edges = [Edge("A", "B"), Edge("C", "D"), Edge("E", "F")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        components = GraphUtils.get_connected_components(adjacency)
        self.assertEqual(len(components), 3)

    # ------------------------------------------------------------------
    # is_simple_path
    # ------------------------------------------------------------------

    def test_is_simple_path_empty(self) -> None:
        self.assertFalse(GraphUtils.is_simple_path([]))

    def test_is_simple_path_single_edge(self) -> None:
        edges = [Edge("A", "B")]
        self.assertTrue(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_chain_of_three(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        self.assertTrue(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_long_chain(self) -> None:
        edges = [
            Edge("A", "B"),
            Edge("B", "C"),
            Edge("C", "D"),
            Edge("D", "E"),
            Edge("E", "F"),
        ]
        self.assertTrue(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_rejects_cycle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        self.assertFalse(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_rejects_t_structure(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("B", "D")]
        self.assertFalse(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_rejects_star(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("A", "D")]
        self.assertFalse(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_rejects_disconnected(self) -> None:
        edges = [Edge("A", "B"), Edge("C", "D")]
        self.assertFalse(GraphUtils.is_simple_path(edges))

    def test_is_simple_path_rejects_cycle_with_tail(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A"), Edge("A", "D")]
        self.assertFalse(GraphUtils.is_simple_path(edges))

    # ------------------------------------------------------------------
    # is_simple_cycle
    # ------------------------------------------------------------------

    def test_is_simple_cycle_empty(self) -> None:
        self.assertFalse(GraphUtils.is_simple_cycle([]))

    def test_is_simple_cycle_single_edge(self) -> None:
        edges = [Edge("A", "B")]
        self.assertFalse(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_two_edges(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "A")]
        self.assertFalse(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_triangle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        self.assertTrue(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_square(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        self.assertTrue(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_pentagon(self) -> None:
        edges = [
            Edge("A", "B"),
            Edge("B", "C"),
            Edge("C", "D"),
            Edge("D", "E"),
            Edge("E", "A"),
        ]
        self.assertTrue(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_rejects_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        self.assertFalse(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_rejects_t_structure(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("B", "D")]
        self.assertFalse(GraphUtils.is_simple_cycle(edges))

    def test_is_simple_cycle_rejects_disconnected(self) -> None:
        edges = [
            Edge("A", "B"), Edge("B", "C"), Edge("C", "A"),
            Edge("D", "E"), Edge("E", "F"), Edge("F", "D"),
        ]
        self.assertFalse(GraphUtils.is_simple_cycle(edges))

    # ------------------------------------------------------------------
    # order_path_vertices
    # ------------------------------------------------------------------

    def test_order_path_vertices_single_edge(self) -> None:
        edges = [Edge("B", "A")]
        ordered = GraphUtils.order_path_vertices(edges)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(ordered, ["A", "B"])

    def test_order_path_vertices_chain(self) -> None:
        edges = [Edge("C", "D"), Edge("A", "B"), Edge("B", "C")]
        ordered = GraphUtils.order_path_vertices(edges)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(ordered, ["A", "B", "C", "D"])

    def test_order_path_vertices_reversed_edges(self) -> None:
        edges = [Edge("D", "C"), Edge("C", "B"), Edge("B", "A")]
        ordered = GraphUtils.order_path_vertices(edges)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(ordered, ["A", "B", "C", "D"])

    def test_order_path_vertices_invalid_cycle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        self.assertIsNone(GraphUtils.order_path_vertices(edges))

    def test_order_path_vertices_invalid_t_structure(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("B", "D")]
        self.assertIsNone(GraphUtils.order_path_vertices(edges))

    # ------------------------------------------------------------------
    # order_cycle_vertices
    # ------------------------------------------------------------------

    def test_order_cycle_vertices_triangle(self) -> None:
        edges = [Edge("C", "A"), Edge("A", "B"), Edge("B", "C")]
        ordered = GraphUtils.order_cycle_vertices(edges)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0], "A")
        self.assertEqual(set(ordered), {"A", "B", "C"})

    def test_order_cycle_vertices_square(self) -> None:
        edges = [Edge("D", "A"), Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        ordered = GraphUtils.order_cycle_vertices(edges)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(len(ordered), 4)
        self.assertEqual(ordered[0], "A")
        self.assertEqual(set(ordered), {"A", "B", "C", "D"})

    def test_order_cycle_vertices_invalid_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        self.assertIsNone(GraphUtils.order_cycle_vertices(edges))

    def test_order_cycle_vertices_invalid_t_structure(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("B", "D")]
        self.assertIsNone(GraphUtils.order_cycle_vertices(edges))

    # ------------------------------------------------------------------
    # shortest path / dijkstra
    # ------------------------------------------------------------------

    def test_shortest_path_unweighted(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("A", "C")]
        path = GraphUtils.shortest_path_unweighted(edges, "A", "C", directed=False)
        self.assertEqual(path, ["A", "C"])

    def test_shortest_path_directed_no_path(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C")]
        path = GraphUtils.shortest_path_unweighted(edges, "C", "A", directed=True)
        self.assertIsNone(path)

    def test_shortest_path_weighted_prefers_lower_cost(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("A", "C")]
        weights = {("A", "C"): 10.0, ("A", "B"): 2.0, ("B", "C"): 2.0}
        result = GraphUtils.shortest_path_dijkstra(edges, "A", "C", weight_lookup=weights, directed=False)
        assert result is not None
        path, cost = result
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(cost, 4.0)

    # ------------------------------------------------------------------
    # mst
    # ------------------------------------------------------------------

    def test_minimum_spanning_tree_basic(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("A", "C")]
        weights = {("A", "B"): 1.0, ("B", "C"): 5.0, ("A", "C"): 2.0}
        mst = GraphUtils.minimum_spanning_tree(edges, weight_lookup=weights)
        self.assertEqual(len(mst), 2)
        names = {e.as_frozenset() for e in mst}
        self.assertIn(frozenset({"A", "B"}), names)
        self.assertIn(frozenset({"A", "C"}), names)

    # ------------------------------------------------------------------
    # topo / bridges / articulation
    # ------------------------------------------------------------------

    def test_topological_sort_cycle_returns_none(self) -> None:
        adjacency = {"A": {"B"}, "B": {"A"}}
        order = GraphUtils.topological_sort(adjacency)
        self.assertIsNone(order)

    def test_topological_sort_simple_dag(self) -> None:
        adjacency = {"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": set()}
        order = GraphUtils.topological_sort(adjacency)
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")

    def test_bridges_and_articulation_points(self) -> None:
        adjacency = {
            "A": {"B"},
            "B": {"A", "C", "D"},
            "C": {"B"},
            "D": {"B", "E"},
            "E": {"D"},
        }
        bridges = GraphUtils.find_bridges(adjacency)
        self.assertIn(("D", "E"), bridges)
        aps = GraphUtils.find_articulation_points(adjacency)
        self.assertIn("B", aps)

    # ------------------------------------------------------------------
    # euler / bipartite
    # ------------------------------------------------------------------

    def test_euler_status_cycle_and_path(self) -> None:
        cycle_adj = GraphUtils.build_adjacency_map([Edge("A", "B"), Edge("B", "C"), Edge("C", "A")])
        path_adj = GraphUtils.build_adjacency_map([Edge("A", "B"), Edge("B", "C")])
        self.assertEqual(GraphUtils.euler_status(cycle_adj), "cycle")
        self.assertEqual(GraphUtils.euler_status(path_adj), "path")

    def test_is_bipartite(self) -> None:
        adjacency = GraphUtils.build_adjacency_map([Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")])
        is_bipartite, coloring = GraphUtils.is_bipartite(adjacency)
        self.assertTrue(is_bipartite)
        self.assertEqual(len(coloring), 4)

    def test_is_not_bipartite(self) -> None:
        adjacency = GraphUtils.build_adjacency_map([Edge("A", "B"), Edge("B", "C"), Edge("C", "A")])
        is_bipartite, _ = GraphUtils.is_bipartite(adjacency)
        self.assertFalse(is_bipartite)

    # ------------------------------------------------------------------
    # tree helpers
    # ------------------------------------------------------------------

    def test_tree_helpers_levels_diameter_lca(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("B", "E"), Edge("C", "F")]
        adjacency = GraphUtils.build_adjacency_map(edges)
        rooted = GraphUtils.root_tree(adjacency, "A")
        assert rooted is not None
        parent, children = rooted
        depths = GraphUtils.node_depths("A", adjacency) or {}
        levels = GraphUtils.tree_levels("A", adjacency)
        self.assertEqual(depths["D"], 2)
        self.assertEqual(levels[0], ["A"])
        diameter = GraphUtils.tree_diameter(adjacency)
        self.assertIsNotNone(diameter)
        lca = GraphUtils.lowest_common_ancestor(parent, depths, "D", "E")
        self.assertEqual(lca, "B")
        balanced = GraphUtils.balance_children("A", children)
        self.assertIn("B", balanced["A"])
        inverted = GraphUtils.invert_children(children)
        self.assertEqual(list(reversed(children["A"])), inverted["A"])
        rerooted = GraphUtils.reroot_tree(parent, children, "B")
        self.assertIsNotNone(rerooted)

    # ------------------------------------------------------------------
    # GraphAnalyzer integration
    # ------------------------------------------------------------------

    def test_graph_analyzer_shortest_path(self) -> None:
        vertices = [GraphVertexDescriptor("A"), GraphVertexDescriptor("B"), GraphVertexDescriptor("C")]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=1.0, name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", weight=1.0, name="BC"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {"start": "A", "goal": "C"})
        self.assertEqual(result.get("path"), ["A", "B", "C"])
        self.assertIn("AB", result.get("highlight_vectors", []))
        self.assertIn("BC", result.get("highlight_vectors", []))

    def test_graph_analyzer_lca(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
        ]
        state = TreeState("T", vertices, edges, root="A")
        result = GraphAnalyzer.analyze(state, "lca", {"root": "A", "a": "D", "b": "C"})
        self.assertEqual(result.get("lca"), "A")


if __name__ == "__main__":
    unittest.main()
