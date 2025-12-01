from __future__ import annotations

import unittest

from utils.graph_utils import Edge, GraphUtils


class TestGraphUtils(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
