from __future__ import annotations

import unittest

from drawables.label import Label
from drawables.point import Point
from drawables.segment import Segment
from drawables.vector import Vector
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
            Edge("A", "B"),
            Edge("B", "C"),
            Edge("C", "A"),
            Edge("D", "E"),
            Edge("E", "F"),
            Edge("F", "D"),
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

    def test_drawables_to_descriptors(self) -> None:
        a = Point(0, 0, name="A")
        b = Point(1, 0, name="B")
        c = Point(1, 1, name="C")

        segment = Segment(a, b, label_text="5.5")
        vector = Vector(b, c)
        vector.segment.label.update_text("2")

        vertices, edges = GraphUtils.drawables_to_descriptors([segment], [vector])

        self.assertEqual({v.id for v in vertices}, {"A", "B", "C"})
        edge_lookup = {e.id: e for e in edges}

        self.assertEqual(len(edges), 2)

        seg_edge = edge_lookup.get(segment.name)
        self.assertIsNotNone(seg_edge)
        if seg_edge:
            self.assertFalse(seg_edge.directed)
            self.assertEqual(seg_edge.source, "A")
            self.assertEqual(seg_edge.target, "B")
            self.assertAlmostEqual(seg_edge.weight or 0.0, 5.5)

        vec_edge = edge_lookup.get(vector.name)
        self.assertIsNotNone(vec_edge)
        if vec_edge:
            self.assertTrue(vec_edge.directed)
            self.assertEqual(vec_edge.source, "B")
            self.assertEqual(vec_edge.target, "C")
            self.assertAlmostEqual(vec_edge.weight or 0.0, 2.0)

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

    # ------------------------------------------------------------------
    # adjacency_matrix_from_descriptors
    # ------------------------------------------------------------------

    def test_adjacency_matrix_from_descriptors_undirected(self) -> None:
        vertices = [
            GraphVertexDescriptor("A"),
            GraphVertexDescriptor("B"),
            GraphVertexDescriptor("C"),
        ]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=2.0),
            GraphEdgeDescriptor("e2", "B", "C", weight=3.0),
        ]
        matrix = GraphUtils.adjacency_matrix_from_descriptors(vertices, edges, directed=False)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(matrix[0][1], 2.0)  # A-B
        self.assertEqual(matrix[1][0], 2.0)  # B-A (symmetric)
        self.assertEqual(matrix[1][2], 3.0)  # B-C
        self.assertEqual(matrix[2][1], 3.0)  # C-B (symmetric)
        self.assertEqual(matrix[0][2], 0.0)  # A-C (no edge)

    def test_adjacency_matrix_from_descriptors_directed(self) -> None:
        vertices = [
            GraphVertexDescriptor("A"),
            GraphVertexDescriptor("B"),
            GraphVertexDescriptor("C"),
        ]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=2.0),
            GraphEdgeDescriptor("e2", "B", "C", weight=3.0),
        ]
        matrix = GraphUtils.adjacency_matrix_from_descriptors(vertices, edges, directed=True)
        self.assertEqual(matrix[0][1], 2.0)  # A->B
        self.assertEqual(matrix[1][0], 0.0)  # B->A (no reverse edge)
        self.assertEqual(matrix[1][2], 3.0)  # B->C
        self.assertEqual(matrix[2][1], 0.0)  # C->B (no reverse edge)

    def test_adjacency_matrix_default_weight(self) -> None:
        vertices = [GraphVertexDescriptor("A"), GraphVertexDescriptor("B")]
        edges = [GraphEdgeDescriptor("e1", "A", "B")]  # No weight specified
        matrix = GraphUtils.adjacency_matrix_from_descriptors(vertices, edges, directed=False)
        self.assertEqual(matrix[0][1], 1.0)  # Default weight
        self.assertEqual(matrix[1][0], 1.0)

    # ------------------------------------------------------------------
    # _extract_weight_from_label edge cases
    # ------------------------------------------------------------------

    def test_extract_weight_from_label_none(self) -> None:
        result = GraphUtils._extract_weight_from_label(None)
        self.assertIsNone(result)

    def test_extract_weight_from_label_empty_text(self) -> None:
        label = Label(0, 0, "")
        result = GraphUtils._extract_weight_from_label(label)
        self.assertIsNone(result)

    def test_extract_weight_from_label_non_numeric(self) -> None:
        label = Label(0, 0, "abc")
        result = GraphUtils._extract_weight_from_label(label)
        self.assertIsNone(result)

    def test_extract_weight_from_label_valid_float(self) -> None:
        label = Label(0, 0, "3.14")
        result = GraphUtils._extract_weight_from_label(label)
        self.assertAlmostEqual(result, 3.14)  # type: ignore[arg-type]

    def test_extract_weight_from_label_integer(self) -> None:
        label = Label(0, 0, "42")
        result = GraphUtils._extract_weight_from_label(label)
        self.assertEqual(result, 42.0)

    def test_extract_weight_from_label_whitespace(self) -> None:
        label = Label(0, 0, "  7.5  ")
        result = GraphUtils._extract_weight_from_label(label)
        self.assertAlmostEqual(result, 7.5)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # drawables_to_descriptors with isolated points
    # ------------------------------------------------------------------

    def test_drawables_to_descriptors_isolated_points(self) -> None:
        a = Point(0, 0, name="A")
        b = Point(1, 0, name="B")
        c = Point(2, 0, name="C")  # isolated
        d = Point(3, 0, name="D")  # isolated

        segment = Segment(a, b)
        vertices, edges = GraphUtils.drawables_to_descriptors([segment], [], isolated_points=[c, d])

        vertex_ids = {v.id for v in vertices}
        self.assertEqual(vertex_ids, {"A", "B", "C", "D"})
        self.assertEqual(len(edges), 1)

    def test_drawables_to_descriptors_empty(self) -> None:
        vertices, edges = GraphUtils.drawables_to_descriptors([], [])
        self.assertEqual(vertices, [])
        self.assertEqual(edges, [])

    # ------------------------------------------------------------------
    # GraphAnalyzer additional operations
    # ------------------------------------------------------------------

    def test_graph_analyzer_mst(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", weight=1.0, name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", weight=2.0, name="BC"),
            GraphEdgeDescriptor("e3", "A", "C", weight=5.0, name="AC"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "mst", {})
        self.assertEqual(len(result.get("edges", [])), 2)
        self.assertIn("AB", result.get("highlight_vectors", []))
        self.assertIn("BC", result.get("highlight_vectors", []))

    def test_graph_analyzer_bridges(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "D", name="CD"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bridges", {})
        bridges = result.get("bridges", [])
        self.assertEqual(len(bridges), 3)  # All edges are bridges in a path

    def test_graph_analyzer_articulation_points(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "articulation_points", {})
        aps = result.get("articulation_points", [])
        self.assertIn("B", aps)
        self.assertIn("C", aps)

    def test_graph_analyzer_euler_status(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "A"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "euler_status", {})
        self.assertEqual(result.get("status"), "cycle")

    def test_graph_analyzer_bipartite(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
            GraphEdgeDescriptor("e3", "C", "D"),
            GraphEdgeDescriptor("e4", "D", "A"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bipartite", {})
        self.assertTrue(result.get("is_bipartite"))

    def test_graph_analyzer_bfs(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "bfs", {"start": "A"})
        order = result.get("order", [])
        self.assertEqual(order[0], "A")
        self.assertEqual(len(order), 4)

    def test_graph_analyzer_dfs(self) -> None:
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

    def test_graph_analyzer_levels(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
        ]
        state = TreeState("T", vertices, edges, root="A")
        result = GraphAnalyzer.analyze(state, "levels", {"root": "A"})
        levels = result.get("levels", [])
        self.assertEqual(levels[0], ["A"])
        self.assertEqual(set(levels[1]), {"B", "C"})
        self.assertEqual(levels[2], ["D"])

    def test_graph_analyzer_diameter(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B", name="AB"),
            GraphEdgeDescriptor("e2", "B", "C", name="BC"),
            GraphEdgeDescriptor("e3", "C", "D", name="CD"),
        ]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "diameter", {})
        path = result.get("path", [])
        self.assertEqual(len(path), 4)
        self.assertIn("A", path)
        self.assertIn("D", path)

    def test_graph_analyzer_topological_sort(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C", "D"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "A", "C"),
            GraphEdgeDescriptor("e3", "B", "D"),
            GraphEdgeDescriptor("e4", "C", "D"),
        ]
        state = GraphState("G", vertices, edges, directed=True, graph_type="dag")
        result = GraphAnalyzer.analyze(state, "topological_sort", {})
        order = result.get("order", [])
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")

    def test_graph_analyzer_reroot(self) -> None:
        vertices = [GraphVertexDescriptor(v) for v in ["A", "B", "C"]]
        edges = [
            GraphEdgeDescriptor("e1", "A", "B"),
            GraphEdgeDescriptor("e2", "B", "C"),
        ]
        state = TreeState("T", vertices, edges, root="A")
        result = GraphAnalyzer.analyze(state, "reroot", {"root": "A", "new_root": "B"})
        new_parent = result.get("parent", {})
        self.assertIsNone(new_parent.get("B"))
        self.assertEqual(new_parent.get("A"), "B")
        self.assertEqual(new_parent.get("C"), "B")

    def test_graph_analyzer_error_missing_params(self) -> None:
        vertices = [GraphVertexDescriptor("A"), GraphVertexDescriptor("B")]
        edges = [GraphEdgeDescriptor("e1", "A", "B")]
        state = GraphState("G", vertices, edges, directed=False)
        result = GraphAnalyzer.analyze(state, "shortest_path", {})
        self.assertIn("error", result)

    def test_graph_analyzer_unsupported_operation(self) -> None:
        vertices = [GraphVertexDescriptor("A")]
        state = GraphState("G", vertices, [], directed=False)
        result = GraphAnalyzer.analyze(state, "unknown_op", {})
        self.assertIn("error", result)

    # ------------------------------------------------------------------
    # segments_cross
    # ------------------------------------------------------------------

    def test_segments_cross_basic_crossing(self) -> None:
        """Two segments that cross in the middle."""
        p1, p2 = (0.0, 0.0), (2.0, 2.0)
        p3, p4 = (0.0, 2.0), (2.0, 0.0)
        self.assertTrue(GraphUtils.segments_cross(p1, p2, p3, p4))

    def test_segments_cross_no_crossing_parallel(self) -> None:
        """Parallel segments don't cross."""
        p1, p2 = (0.0, 0.0), (2.0, 0.0)
        p3, p4 = (0.0, 1.0), (2.0, 1.0)
        self.assertFalse(GraphUtils.segments_cross(p1, p2, p3, p4))

    def test_segments_cross_no_crossing_separate(self) -> None:
        """Non-parallel segments that don't intersect."""
        p1, p2 = (0.0, 0.0), (1.0, 1.0)
        p3, p4 = (2.0, 0.0), (3.0, 1.0)
        self.assertFalse(GraphUtils.segments_cross(p1, p2, p3, p4))

    def test_segments_cross_touching_at_endpoint(self) -> None:
        """Segments touching at an endpoint don't count as crossing."""
        p1, p2 = (0.0, 0.0), (1.0, 1.0)
        p3, p4 = (1.0, 1.0), (2.0, 0.0)
        self.assertFalse(GraphUtils.segments_cross(p1, p2, p3, p4))

    def test_segments_cross_t_intersection(self) -> None:
        """T-intersection where one segment ends on another."""
        p1, p2 = (0.0, 0.0), (2.0, 0.0)
        p3, p4 = (1.0, -1.0), (1.0, 0.0)
        self.assertFalse(GraphUtils.segments_cross(p1, p2, p3, p4))

    def test_segments_cross_collinear_overlapping(self) -> None:
        """Collinear overlapping segments don't cross (parallel)."""
        p1, p2 = (0.0, 0.0), (2.0, 0.0)
        p3, p4 = (1.0, 0.0), (3.0, 0.0)
        self.assertFalse(GraphUtils.segments_cross(p1, p2, p3, p4))

    # ------------------------------------------------------------------
    # count_edge_crossings
    # ------------------------------------------------------------------

    def test_count_edge_crossings_square_no_crossings(self) -> None:
        """Square has no edge crossings."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        self.assertEqual(GraphUtils.count_edge_crossings(edges, positions), 0)

    def test_count_edge_crossings_crossing_diagonals(self) -> None:
        """Two crossing diagonals have 1 crossing."""
        edges = [Edge("A", "C"), Edge("B", "D")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        self.assertEqual(GraphUtils.count_edge_crossings(edges, positions), 1)

    def test_count_edge_crossings_k4_complete(self) -> None:
        """K4 with vertices at corners has 1 crossing (diagonals)."""
        edges = [
            Edge("A", "B"),
            Edge("A", "C"),
            Edge("A", "D"),
            Edge("B", "C"),
            Edge("B", "D"),
            Edge("C", "D"),
        ]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        # A-C and B-D cross
        self.assertEqual(GraphUtils.count_edge_crossings(edges, positions), 1)

    def test_count_edge_crossings_shared_vertex_no_crossing(self) -> None:
        """Edges sharing a vertex don't count as crossing."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 1.0), "C": (2.0, 0.0)}
        self.assertEqual(GraphUtils.count_edge_crossings(edges, positions), 0)

    # ------------------------------------------------------------------
    # is_edge_orthogonal
    # ------------------------------------------------------------------

    def test_is_edge_orthogonal_horizontal(self) -> None:
        """Horizontal edge is orthogonal."""
        self.assertTrue(GraphUtils.is_edge_orthogonal((0.0, 0.0), (5.0, 0.0)))

    def test_is_edge_orthogonal_vertical(self) -> None:
        """Vertical edge is orthogonal."""
        self.assertTrue(GraphUtils.is_edge_orthogonal((0.0, 0.0), (0.0, 5.0)))

    def test_is_edge_orthogonal_diagonal_45(self) -> None:
        """45-degree diagonal is not orthogonal."""
        self.assertFalse(GraphUtils.is_edge_orthogonal((0.0, 0.0), (5.0, 5.0)))

    def test_is_edge_orthogonal_slight_deviation(self) -> None:
        """Slight deviation from horizontal is still orthogonal within tolerance."""
        # 0.5% deviation on a length of 100 is 0.5, within 1% tolerance
        self.assertTrue(GraphUtils.is_edge_orthogonal((0.0, 0.0), (100.0, 0.5)))

    def test_is_edge_orthogonal_large_deviation(self) -> None:
        """Large deviation from horizontal is not orthogonal."""
        self.assertFalse(GraphUtils.is_edge_orthogonal((0.0, 0.0), (100.0, 10.0)))

    # ------------------------------------------------------------------
    # count_orthogonal_edges
    # ------------------------------------------------------------------

    def test_count_orthogonal_edges_all_orthogonal(self) -> None:
        """Square with all orthogonal edges."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        ortho, total = GraphUtils.count_orthogonal_edges(edges, positions)
        self.assertEqual(ortho, 4)
        self.assertEqual(total, 4)

    def test_count_orthogonal_edges_none_orthogonal(self) -> None:
        """Diamond shape has no orthogonal edges."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        # Rotated 45 degrees
        positions = {"A": (1.0, 0.0), "B": (2.0, 1.0), "C": (1.0, 2.0), "D": (0.0, 1.0)}
        ortho, total = GraphUtils.count_orthogonal_edges(edges, positions)
        self.assertEqual(ortho, 0)
        self.assertEqual(total, 4)

    def test_count_orthogonal_edges_mixed(self) -> None:
        """Triangle with one horizontal edge."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        positions = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.0)}
        ortho, total = GraphUtils.count_orthogonal_edges(edges, positions)
        self.assertEqual(ortho, 1)  # Only A-B is horizontal
        self.assertEqual(total, 3)

    # ------------------------------------------------------------------
    # edges_overlap (non-adjacent edges)
    # ------------------------------------------------------------------

    def test_edges_overlap_collinear_overlapping(self) -> None:
        """Collinear segments with overlapping spans."""
        p1, p2 = (0.0, 0.0), (3.0, 0.0)
        p3, p4 = (2.0, 0.0), (5.0, 0.0)
        self.assertTrue(GraphUtils.edges_overlap(p1, p2, p3, p4))

    def test_edges_overlap_collinear_no_overlap(self) -> None:
        """Collinear segments without overlapping spans."""
        p1, p2 = (0.0, 0.0), (1.0, 0.0)
        p3, p4 = (2.0, 0.0), (3.0, 0.0)
        self.assertFalse(GraphUtils.edges_overlap(p1, p2, p3, p4))

    def test_edges_overlap_collinear_touching(self) -> None:
        """Collinear segments touching at one point don't overlap."""
        p1, p2 = (0.0, 0.0), (1.0, 0.0)
        p3, p4 = (1.0, 0.0), (2.0, 0.0)
        self.assertFalse(GraphUtils.edges_overlap(p1, p2, p3, p4))

    def test_edges_overlap_not_collinear(self) -> None:
        """Non-collinear segments don't overlap."""
        p1, p2 = (0.0, 0.0), (1.0, 0.0)
        p3, p4 = (0.0, 1.0), (1.0, 1.0)
        self.assertFalse(GraphUtils.edges_overlap(p1, p2, p3, p4))

    def test_edges_overlap_vertical(self) -> None:
        """Vertical collinear overlapping segments."""
        p1, p2 = (0.0, 0.0), (0.0, 3.0)
        p3, p4 = (0.0, 2.0), (0.0, 5.0)
        self.assertTrue(GraphUtils.edges_overlap(p1, p2, p3, p4))

    # ------------------------------------------------------------------
    # point_on_segment
    # ------------------------------------------------------------------

    def test_point_on_segment_inside(self) -> None:
        """Point strictly inside segment."""
        self.assertTrue(GraphUtils.point_on_segment((1.0, 0.0), (0.0, 0.0), (2.0, 0.0)))

    def test_point_on_segment_at_start(self) -> None:
        """Point at segment start is not inside."""
        self.assertFalse(GraphUtils.point_on_segment((0.0, 0.0), (0.0, 0.0), (2.0, 0.0)))

    def test_point_on_segment_at_end(self) -> None:
        """Point at segment end is not inside."""
        self.assertFalse(GraphUtils.point_on_segment((2.0, 0.0), (0.0, 0.0), (2.0, 0.0)))

    def test_point_on_segment_outside_collinear(self) -> None:
        """Point collinear but outside segment."""
        self.assertFalse(GraphUtils.point_on_segment((3.0, 0.0), (0.0, 0.0), (2.0, 0.0)))

    def test_point_on_segment_not_collinear(self) -> None:
        """Point not on the line."""
        self.assertFalse(GraphUtils.point_on_segment((1.0, 1.0), (0.0, 0.0), (2.0, 0.0)))

    def test_point_on_segment_diagonal(self) -> None:
        """Point on diagonal segment."""
        self.assertTrue(GraphUtils.point_on_segment((1.0, 1.0), (0.0, 0.0), (2.0, 2.0)))

    def test_point_on_segment_vertical(self) -> None:
        """Point on vertical segment."""
        self.assertTrue(GraphUtils.point_on_segment((0.0, 1.0), (0.0, 0.0), (0.0, 2.0)))

    # ------------------------------------------------------------------
    # adjacent_edges_overlap
    # ------------------------------------------------------------------

    def test_adjacent_edges_overlap_point_inside(self) -> None:
        """Adjacent edges where one endpoint lies on the other edge."""
        # Edge 1: (0,0) to (2,0), Edge 2: (1,0) to (3,0)
        # They share the region from (1,0) to (2,0) conceptually,
        # but more importantly, (1,0) lies on edge 1
        p1, p2 = (0.0, 0.0), (2.0, 0.0)  # edge 1
        p3, p4 = (1.0, 0.0), (3.0, 0.0)  # edge 2
        self.assertTrue(GraphUtils.adjacent_edges_overlap(p1, p2, p3, p4, True))

    def test_adjacent_edges_overlap_no_overlap(self) -> None:
        """Adjacent edges that meet at a vertex without overlap."""
        # L-shape: (0,0)-(1,0) and (1,0)-(1,1)
        p1, p2 = (0.0, 0.0), (1.0, 0.0)
        p3, p4 = (1.0, 0.0), (1.0, 1.0)
        self.assertFalse(GraphUtils.adjacent_edges_overlap(p1, p2, p3, p4, True))

    def test_adjacent_edges_overlap_cap_on_edge(self) -> None:
        """Cap vertex lies on the main edge (I on B-C scenario)."""
        # B-C is (0,0) to (2,0), I is at (1,0)
        # Edge I-B: (1,0) to (0,0)
        # Edge B-C: (0,0) to (2,0)
        # I lies on B-C
        p1, p2 = (1.0, 0.0), (0.0, 0.0)  # I-B
        p3, p4 = (0.0, 0.0), (2.0, 0.0)  # B-C
        self.assertTrue(GraphUtils.adjacent_edges_overlap(p1, p2, p3, p4, True))

    def test_adjacent_edges_overlap_collinear_extending(self) -> None:
        """Adjacent collinear edges that extend in same direction."""
        # A-B: (0,0) to (1,0), B-C: (1,0) to (2,0)
        # These share only the endpoint, no overlap
        p1, p2 = (0.0, 0.0), (1.0, 0.0)
        p3, p4 = (1.0, 0.0), (2.0, 0.0)
        self.assertFalse(GraphUtils.adjacent_edges_overlap(p1, p2, p3, p4, True))

    # ------------------------------------------------------------------
    # count_edge_overlaps
    # ------------------------------------------------------------------

    def test_count_edge_overlaps_no_overlaps(self) -> None:
        """Square layout has no overlapping edges."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        self.assertEqual(GraphUtils.count_edge_overlaps(edges, positions), 0)

    def test_count_edge_overlaps_collinear_non_adjacent(self) -> None:
        """Non-adjacent collinear overlapping edges."""
        edges = [Edge("A", "B"), Edge("C", "D")]
        # A-B and C-D overlap on the x-axis
        positions = {"A": (0.0, 0.0), "B": (3.0, 0.0), "C": (2.0, 0.0), "D": (5.0, 0.0)}
        self.assertEqual(GraphUtils.count_edge_overlaps(edges, positions), 1)

    def test_count_edge_overlaps_adjacent_with_cap(self) -> None:
        """Adjacent edges with cap vertex on main edge."""
        # B at (0,0), C at (2,0), I at (1,0)
        # I connects to B and C
        edges = [Edge("B", "C"), Edge("I", "B"), Edge("I", "C")]
        positions = {"B": (0.0, 0.0), "C": (2.0, 0.0), "I": (1.0, 0.0)}
        # I-B overlaps with B-C, I-C overlaps with B-C
        overlaps = GraphUtils.count_edge_overlaps(edges, positions)
        self.assertEqual(overlaps, 2)

    def test_count_edge_overlaps_triangle_no_overlap(self) -> None:
        """Regular triangle has no overlapping edges."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        positions = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.732)}
        self.assertEqual(GraphUtils.count_edge_overlaps(edges, positions), 0)

    # ------------------------------------------------------------------
    # Edge length functions
    # ------------------------------------------------------------------

    def test_compute_edge_length_horizontal(self) -> None:
        """Horizontal edge length."""
        length = GraphUtils.compute_edge_length((0.0, 0.0), (3.0, 0.0))
        self.assertAlmostEqual(length, 3.0, places=5)

    def test_compute_edge_length_vertical(self) -> None:
        """Vertical edge length."""
        length = GraphUtils.compute_edge_length((0.0, 0.0), (0.0, 4.0))
        self.assertAlmostEqual(length, 4.0, places=5)

    def test_compute_edge_length_diagonal(self) -> None:
        """Diagonal edge length (3-4-5 triangle)."""
        length = GraphUtils.compute_edge_length((0.0, 0.0), (3.0, 4.0))
        self.assertAlmostEqual(length, 5.0, places=5)

    def test_compute_edge_length_zero(self) -> None:
        """Same point gives zero length."""
        length = GraphUtils.compute_edge_length((5.0, 5.0), (5.0, 5.0))
        self.assertAlmostEqual(length, 0.0, places=5)

    def test_get_edge_lengths_empty(self) -> None:
        """Empty edges list."""
        lengths = GraphUtils.get_edge_lengths([], {})
        self.assertEqual(lengths, [])

    def test_get_edge_lengths_single(self) -> None:
        """Single edge."""
        edges = [Edge("A", "B")]
        positions = {"A": (0.0, 0.0), "B": (5.0, 0.0)}
        lengths = GraphUtils.get_edge_lengths(edges, positions)
        self.assertEqual(len(lengths), 1)
        self.assertAlmostEqual(lengths[0], 5.0, places=5)

    def test_get_edge_lengths_multiple(self) -> None:
        """Multiple edges."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = {"A": (0.0, 0.0), "B": (3.0, 0.0), "C": (3.0, 4.0)}
        lengths = GraphUtils.get_edge_lengths(edges, positions)
        self.assertEqual(len(lengths), 2)
        self.assertAlmostEqual(lengths[0], 3.0, places=5)
        self.assertAlmostEqual(lengths[1], 4.0, places=5)

    def test_get_edge_lengths_missing_position(self) -> None:
        """Missing vertex position is skipped."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = {"A": (0.0, 0.0), "B": (3.0, 0.0)}  # C missing
        lengths = GraphUtils.get_edge_lengths(edges, positions)
        self.assertEqual(len(lengths), 1)

    def test_count_edges_with_same_length_all_same(self) -> None:
        """All edges have the same length."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        same_count, total, avg_len = GraphUtils.count_edges_with_same_length(edges, positions)
        self.assertEqual(same_count, 4)
        self.assertEqual(total, 4)
        self.assertAlmostEqual(avg_len, 1.0, places=5)

    def test_count_edges_with_same_length_mixed(self) -> None:
        """Mixed edge lengths."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        # A-B: 1.0, B-C: 1.0, C-D: 2.0
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0), "D": (4.0, 0.0)}
        same_count, total, _ = GraphUtils.count_edges_with_same_length(edges, positions)
        self.assertEqual(same_count, 2)  # Two edges of length 1.0
        self.assertEqual(total, 3)

    def test_count_edges_with_same_length_empty(self) -> None:
        """Empty edges."""
        same_count, total, avg_len = GraphUtils.count_edges_with_same_length([], {})
        self.assertEqual(same_count, 0)
        self.assertEqual(total, 0)
        self.assertAlmostEqual(avg_len, 0.0, places=5)

    def test_count_edges_with_same_length_tolerance(self) -> None:
        """Edges within tolerance are grouped together."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        # A-B: 1.0, B-C: 1.05, C-D: 1.1 (all within 10% tolerance)
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.05, 0.0), "D": (3.15, 0.0)}
        same_count, total, _ = GraphUtils.count_edges_with_same_length(edges, positions, tolerance=0.1)
        self.assertEqual(same_count, 3)  # All edges within tolerance

    def test_edge_length_variance_uniform(self) -> None:
        """Uniform edge lengths have zero variance."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (2.0, 2.0), "D": (0.0, 2.0)}
        variance = GraphUtils.edge_length_variance(edges, positions)
        self.assertAlmostEqual(variance, 0.0, places=5)

    def test_edge_length_variance_mixed(self) -> None:
        """Mixed edge lengths have positive variance."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        # A-B: 1.0, B-C: 3.0
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (4.0, 0.0)}
        variance = GraphUtils.edge_length_variance(edges, positions)
        # Mean = 2.0, variance = ((1-2)^2 + (3-2)^2) / 2 = 1.0
        self.assertAlmostEqual(variance, 1.0, places=5)

    def test_edge_length_variance_single_edge(self) -> None:
        """Single edge has zero variance."""
        edges = [Edge("A", "B")]
        positions = {"A": (0.0, 0.0), "B": (5.0, 0.0)}
        variance = GraphUtils.edge_length_variance(edges, positions)
        self.assertAlmostEqual(variance, 0.0, places=5)

    def test_edge_length_uniformity_ratio_all_same(self) -> None:
        """All same length gives ratio 1.0."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
        ratio = GraphUtils.edge_length_uniformity_ratio(edges, positions)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_edge_length_uniformity_ratio_half(self) -> None:
        """Half same length gives ratio 0.5."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "E")]
        # A-B: 1, B-C: 1, C-D: 2, D-E: 2
        positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0), "D": (4.0, 0.0), "E": (6.0, 0.0)}
        ratio = GraphUtils.edge_length_uniformity_ratio(edges, positions, tolerance=0.1)
        self.assertAlmostEqual(ratio, 0.5, places=5)

    def test_edge_length_uniformity_ratio_empty(self) -> None:
        """Empty edges gives ratio 1.0."""
        ratio = GraphUtils.edge_length_uniformity_ratio([], {})
        self.assertAlmostEqual(ratio, 1.0, places=5)


# Preserve legacy test case name expected by the runner
class TestGraphUtils(TestGraph):
    pass


if __name__ == "__main__":
    unittest.main()
