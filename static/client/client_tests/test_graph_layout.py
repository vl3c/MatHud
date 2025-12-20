from __future__ import annotations

import math
import unittest

from utils.graph_layout import (
    layout_vertices,
    _circular_layout,
    _grid_layout,
    _simple_grid_placement,
    _orthogonal_tree_layout,
    _is_planar,
    _tree_layout,
    _force_directed_layout,
    _infer_root,
    _is_tree_structure,
)
from utils.graph_utils import Edge


class TestGraphLayout(unittest.TestCase):

    def setUp(self) -> None:
        self.box = {"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0}

    # ------------------------------------------------------------------
    # Circular layout
    # ------------------------------------------------------------------

    def test_circular_layout_single_node(self) -> None:
        positions = _circular_layout(["A"], self.box)
        self.assertEqual(len(positions), 1)
        x, y = positions["A"]
        self.assertAlmostEqual(x, 50.0 + 40.0, places=5)
        self.assertAlmostEqual(y, 50.0, places=5)

    def test_circular_layout_multiple_nodes(self) -> None:
        positions = _circular_layout(["A", "B", "C", "D"], self.box)
        self.assertEqual(len(positions), 4)
        for vid in ["A", "B", "C", "D"]:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 100.0)

    # ------------------------------------------------------------------
    # Grid layout (TSM orthogonal layout)
    # ------------------------------------------------------------------

    def test_grid_layout_single_node(self) -> None:
        """Single node should be centered in box."""
        positions = _grid_layout(["A"], [], self.box)
        self.assertEqual(len(positions), 1)
        x, y = positions["A"]
        # Should be at center
        self.assertAlmostEqual(x, 50.0, places=1)
        self.assertAlmostEqual(y, 50.0, places=1)

    def test_grid_layout_no_edges(self) -> None:
        """Multiple isolated nodes should be placed on grid."""
        positions = _grid_layout(["A", "B", "C", "D"], [], self.box)
        self.assertEqual(len(positions), 4)
        # All positions should be within box
        for vid in ["A", "B", "C", "D"]:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 100.0)

    def test_grid_layout_two_nodes(self) -> None:
        """Two connected nodes should be placed with separation."""
        edges = [Edge("A", "B")]
        positions = _grid_layout(["A", "B"], edges, self.box)
        self.assertEqual(len(positions), 2)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        
        # Nodes should be separated
        dist = math.sqrt((ax - bx)**2 + (ay - by)**2)
        self.assertGreater(dist, 5.0)

    def test_grid_layout_path(self) -> None:
        """A-B-C path should produce a valid layout."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = _grid_layout(["A", "B", "C"], edges, self.box)
        self.assertEqual(len(positions), 3)
        
        # All within box
        for vid in ["A", "B", "C"]:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 100.0)

    def test_grid_layout_triangle(self) -> None:
        """K3 (triangle) should produce valid planar layout."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        positions = _grid_layout(["A", "B", "C"], edges, self.box)
        self.assertEqual(len(positions), 3)
        
        # All positions distinct
        pos_set = set(positions.values())
        self.assertEqual(len(pos_set), 3)

    def test_grid_layout_square_cycle(self) -> None:
        """4-cycle should produce rectangular layout."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A")]
        positions = _grid_layout(["A", "B", "C", "D"], edges, self.box)
        self.assertEqual(len(positions), 4)
        
        # All within box
        for vid in ["A", "B", "C", "D"]:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)

    def test_grid_layout_k4(self) -> None:
        """K4 (complete graph on 4) is planar and should work."""
        edges = [
            Edge("A", "B"), Edge("A", "C"), Edge("A", "D"),
            Edge("B", "C"), Edge("B", "D"), Edge("C", "D"),
        ]
        positions = _grid_layout(["A", "B", "C", "D"], edges, self.box)
        self.assertEqual(len(positions), 4)

    def test_grid_layout_k5_non_planar(self) -> None:
        """K5 is non-planar but should still produce valid layout (via fallback)."""
        vertices = ["A", "B", "C", "D", "E"]
        edges = [
            Edge("A", "B"), Edge("A", "C"), Edge("A", "D"), Edge("A", "E"),
            Edge("B", "C"), Edge("B", "D"), Edge("B", "E"),
            Edge("C", "D"), Edge("C", "E"),
            Edge("D", "E"),
        ]
        positions = _grid_layout(vertices, edges, self.box)
        self.assertEqual(len(positions), 5)
        
        # All within box
        for vid in vertices:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)

    def test_grid_layout_binary_tree(self) -> None:
        """Binary tree should use orthogonal tree layout."""
        edges = [Edge("R", "A"), Edge("R", "B"), Edge("A", "C"), Edge("A", "D")]
        positions = _grid_layout(["R", "A", "B", "C", "D"], edges, self.box)
        self.assertEqual(len(positions), 5)
        
        # Root should be at top (higher y)
        rx, ry = positions["R"]
        ax, ay = positions["A"]
        self.assertGreater(ry, ay)

    def test_grid_layout_no_vertex_overlap(self) -> None:
        """No two vertices should share the exact same position."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "D")]
        positions = _grid_layout(["A", "B", "C", "D"], edges, self.box)
        
        pos_list = list(positions.values())
        for i, p1 in enumerate(pos_list):
            for p2 in pos_list[i + 1:]:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                self.assertGreater(dist, 0.01, "Vertices overlap")

    def test_grid_layout_fits_box(self) -> None:
        """All positions should be within bounding box."""
        box = {"x": 10.0, "y": 20.0, "width": 80.0, "height": 60.0}
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = _grid_layout(["A", "B", "C"], edges, box)
        
        for vid, (x, y) in positions.items():
            self.assertGreaterEqual(x, box["x"], f"{vid} x below box")
            self.assertLessEqual(x, box["x"] + box["width"], f"{vid} x above box")
            self.assertGreaterEqual(y, box["y"], f"{vid} y below box")
            self.assertLessEqual(y, box["y"] + box["height"], f"{vid} y above box")

    def test_grid_layout_via_layout_vertices(self) -> None:
        """Grid layout should work through main dispatcher."""
        edges = [Edge("A", "B"), Edge("B", "C")]
        positions = layout_vertices(
            ["A", "B", "C"],
            edges,
            layout="grid",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
        )
        self.assertEqual(len(positions), 3)

    def test_simple_grid_placement_single(self) -> None:
        """Simple grid placement with single node."""
        positions = _simple_grid_placement(["A"], self.box)
        self.assertEqual(len(positions), 1)
        x, y = positions["A"]
        self.assertAlmostEqual(x, 50.0, places=1)
        self.assertAlmostEqual(y, 50.0, places=1)

    def test_simple_grid_placement_multiple(self) -> None:
        """Simple grid placement arranges nodes in grid pattern."""
        positions = _simple_grid_placement(["A", "B", "C", "D", "E", "F"], self.box)
        self.assertEqual(len(positions), 6)
        
        # All unique positions
        pos_set = set()
        for x, y in positions.values():
            pos_set.add((round(x, 2), round(y, 2)))
        self.assertEqual(len(pos_set), 6)

    def test_orthogonal_tree_layout(self) -> None:
        """Orthogonal tree layout produces valid positions."""
        edges = [Edge("R", "A"), Edge("R", "B")]
        positions = _orthogonal_tree_layout(["R", "A", "B"], edges, self.box)
        self.assertEqual(len(positions), 3)
        
        # Root at top
        rx, ry = positions["R"]
        ax, ay = positions["A"]
        self.assertGreater(ry, ay)

    def test_is_planar_small_graphs(self) -> None:
        """Small graphs should be detected as planar."""
        # Triangle
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        is_planar, embedding = _is_planar(["A", "B", "C"], edges)
        self.assertTrue(is_planar)
        self.assertIsNotNone(embedding)

    def test_is_planar_k5_detection(self) -> None:
        """K5 should be detected as non-planar (too many edges)."""
        vertices = ["A", "B", "C", "D", "E"]
        edges = [
            Edge("A", "B"), Edge("A", "C"), Edge("A", "D"), Edge("A", "E"),
            Edge("B", "C"), Edge("B", "D"), Edge("B", "E"),
            Edge("C", "D"), Edge("C", "E"),
            Edge("D", "E"),
        ]
        is_planar, embedding = _is_planar(vertices, edges)
        self.assertFalse(is_planar)

    # ------------------------------------------------------------------
    # Tree layout
    # ------------------------------------------------------------------

    def test_tree_layout_single_node(self) -> None:
        positions = _tree_layout(["A"], [], self.box, "A")
        self.assertEqual(len(positions), 1)
        self.assertIn("A", positions)

    def test_tree_layout_binary_tree(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("B", "E")]
        positions = _tree_layout(["A", "B", "C", "D", "E"], edges, self.box, "A")
        self.assertEqual(len(positions), 5)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        cx, cy = positions["C"]
        dx, dy = positions["D"]
        ex, ey = positions["E"]
        
        self.assertGreater(ay, by)
        self.assertGreater(by, dy)
        self.assertAlmostEqual(by, cy, places=5)
        self.assertAlmostEqual(dy, ey, places=5)
        
        self.assertLess(bx, cx)
        self.assertLess(dx, ex)

    def test_tree_layout_children_centered_under_parent(self) -> None:
        edges = [Edge("R", "A"), Edge("R", "B")]
        positions = _tree_layout(["R", "A", "B"], edges, self.box, "R")
        
        rx, ry = positions["R"]
        ax, ay = positions["A"]
        bx, by = positions["B"]
        
        children_center = (ax + bx) / 2.0
        self.assertAlmostEqual(rx, children_center, places=5)

    def test_tree_layout_asymmetric_tree(self) -> None:
        """Tree: R -> A, B; A -> C, D, E (A has more children than B)."""
        edges = [
            Edge("R", "A"),
            Edge("R", "B"),
            Edge("A", "C"),
            Edge("A", "D"),
            Edge("A", "E"),
        ]
        positions = _tree_layout(["R", "A", "B", "C", "D", "E"], edges, self.box, "R")
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        self.assertLess(ax, bx)
        
        cx, _ = positions["C"]
        dx, _ = positions["D"]
        ex, _ = positions["E"]
        children_center = (cx + dx + ex) / 3.0
        self.assertAlmostEqual(ax, (cx + ex) / 2.0, places=3)

    def test_tree_layout_exact_coordinates_simple(self) -> None:
        """Test exact coordinate calculation for a simple tree: P -> L, Q."""
        box = {"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0}
        edges = [Edge("P", "L"), Edge("P", "Q")]
        positions = _tree_layout(["P", "L", "Q"], edges, box, "P")
        
        px, py = positions["P"]
        lx, ly = positions["L"]
        qx, qy = positions["Q"]
        
        self.assertAlmostEqual(py, 100.0 - 100.0 / 2 / 2, places=3)
        self.assertAlmostEqual(ly, 100.0 - 100.0 / 2 * 1.5, places=3)
        self.assertAlmostEqual(qy, ly, places=5)
        
        self.assertAlmostEqual(px, 50.0, places=3)
        self.assertAlmostEqual(lx, 25.0, places=3)
        self.assertAlmostEqual(qx, 75.0, places=3)

    def test_tree_layout_exact_coordinates_three_levels(self) -> None:
        """Test coordinates for a 3-level binary tree: root at top, leaves at bottom."""
        box = {"x": 0.0, "y": 0.0, "width": 120.0, "height": 90.0}
        edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("B", "E")]
        positions = _tree_layout(["A", "B", "C", "D", "E"], edges, box, "A")
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        cx, cy = positions["C"]
        dx, dy = positions["D"]
        ex, ey = positions["E"]
        
        layer_height = 90.0 / 3
        self.assertAlmostEqual(ay, 90.0 - layer_height * 0.5, places=3)
        self.assertAlmostEqual(by, 90.0 - layer_height * 1.5, places=3)
        self.assertAlmostEqual(dy, 90.0 - layer_height * 2.5, places=3)
        
        self.assertGreater(ay, by)
        self.assertGreater(by, dy)
        
        self.assertLess(bx, cx)
        self.assertLess(dx, ex)
        
        self.assertAlmostEqual(bx, (dx + ex) / 2, places=3)

    # ------------------------------------------------------------------
    # Force-directed layout
    # ------------------------------------------------------------------

    def test_force_layout_single_node(self) -> None:
        positions = _force_directed_layout(["A"], [], self.box)
        self.assertEqual(len(positions), 1)
        x, y = positions["A"]
        self.assertAlmostEqual(x, 50.0, places=5)
        self.assertAlmostEqual(y, 50.0, places=5)

    def test_force_layout_two_connected_nodes(self) -> None:
        edges = [Edge("A", "B")]
        positions = _force_directed_layout(["A", "B"], edges, self.box, iterations=50)
        self.assertEqual(len(positions), 2)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        # Nodes should be within box with margin
        self.assertGreater(ax, 5.0)
        self.assertLess(ax, 95.0)
        self.assertGreater(bx, 5.0)
        self.assertLess(bx, 95.0)
        self.assertGreaterEqual(bx, 0.0)
        self.assertLessEqual(bx, 100.0)

    def test_force_layout_triangle(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        positions = _force_directed_layout(["A", "B", "C"], edges, self.box, iterations=30)
        self.assertEqual(len(positions), 3)
        for vid in ["A", "B", "C"]:
            x, y = positions[vid]
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 100.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 100.0)

    def test_force_layout_no_edges(self) -> None:
        positions = _force_directed_layout(["A", "B", "C"], [], self.box, iterations=50)
        self.assertEqual(len(positions), 3)

    def test_force_layout_connected_closer_than_unconnected(self) -> None:
        """Connected nodes should be closer together than unconnected nodes."""
        # A-B connected, C unconnected
        edges = [Edge("A", "B")]
        box = {"x": 0.0, "y": 0.0, "width": 200.0, "height": 200.0}
        positions = _force_directed_layout(["A", "B", "C"], edges, box, iterations=100)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        cx, cy = positions["C"]
        
        # Distance A-B (connected)
        dist_ab = math.sqrt((ax - bx)**2 + (ay - by)**2)
        # Distance A-C (not connected)
        dist_ac = math.sqrt((ax - cx)**2 + (ay - cy)**2)
        # Distance B-C (not connected)
        dist_bc = math.sqrt((bx - cx)**2 + (by - cy)**2)
        
        # Connected pair should be closer than at least one unconnected pair
        self.assertLess(dist_ab, max(dist_ac, dist_bc))

    def test_force_layout_clusters_separate(self) -> None:
        """Two clusters connected by bridge should separate."""
        # Cluster 1: A-B, Cluster 2: C-D, Bridge: B-C
        edges = [Edge("A", "B"), Edge("C", "D"), Edge("B", "C")]
        box = {"x": 0.0, "y": 0.0, "width": 200.0, "height": 200.0}
        positions = _force_directed_layout(["A", "B", "C", "D"], edges, box, iterations=100)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        cx, cy = positions["C"]
        dx, dy = positions["D"]
        
        # Cluster 1 center
        c1x, c1y = (ax + bx) / 2, (ay + by) / 2
        # Cluster 2 center
        c2x, c2y = (cx + dx) / 2, (cy + dy) / 2
        
        # Cluster centers should be separated
        cluster_dist = math.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
        self.assertGreater(cluster_dist, 20.0)

    def test_force_layout_two_complete_squares_with_bridge(self) -> None:
        """
        Two K4 cliques (fully connected 4-node groups) connected by a bridge.
        Cluster 1: A,B,C,D (all edges)
        Cluster 2: E,F,G,H (all edges)
        Bridge: D-E
        Should produce two separated clusters.
        """
        vertices = ["A", "B", "C", "D", "E", "F", "G", "H"]
        edges = [
            # Cluster 1: complete square
            Edge("A", "B"), Edge("B", "C"), Edge("C", "D"), Edge("D", "A"),
            Edge("A", "C"), Edge("B", "D"),
            # Cluster 2: complete square
            Edge("E", "F"), Edge("F", "G"), Edge("G", "H"), Edge("H", "E"),
            Edge("E", "G"), Edge("F", "H"),
            # Bridge
            Edge("D", "E"),
        ]
        box = {"x": 0.0, "y": 0.0, "width": 1000.0, "height": 600.0}
        positions = _force_directed_layout(vertices, edges, box, iterations=100)
        
        self.assertEqual(len(positions), 8)
        
        # All nodes should be within box (with margin)
        for vid, (x, y) in positions.items():
            self.assertGreater(x, 50.0, f"{vid} x too small: {x}")
            self.assertLess(x, 950.0, f"{vid} x too large: {x}")
            self.assertGreater(y, 30.0, f"{vid} y too small: {y}")
            self.assertLess(y, 570.0, f"{vid} y too large: {y}")
        
        # Cluster 1 center (A,B,C,D)
        c1_x = sum(positions[v][0] for v in ["A","B","C","D"]) / 4
        c1_y = sum(positions[v][1] for v in ["A","B","C","D"]) / 4
        
        # Cluster 2 center (E,F,G,H)
        c2_x = sum(positions[v][0] for v in ["E","F","G","H"]) / 4
        c2_y = sum(positions[v][1] for v in ["E","F","G","H"]) / 4
        
        # Cluster centers should be well separated
        cluster_dist = math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        self.assertGreater(cluster_dist, 100.0, f"Cluster separation too small: {cluster_dist}")
        
        # Intra-cluster distances should be smaller than inter-cluster distances
        def avg_dist(v_list: list[str]) -> float:
            total = 0.0
            count = 0
            for i, v1 in enumerate(v_list):
                for v2 in v_list[i+1:]:
                    x1, y1 = positions[v1]
                    x2, y2 = positions[v2]
                    total += math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    count += 1
            return total / count if count > 0 else 0
        
        intra_c1 = avg_dist(["A", "B", "C", "D"])
        intra_c2 = avg_dist(["E", "F", "G", "H"])
        avg_intra = (intra_c1 + intra_c2) / 2
        
        # Intra-cluster average should be less than cluster center separation
        self.assertLess(avg_intra, cluster_dist)

    # ------------------------------------------------------------------
    # layout_vertices selector
    # ------------------------------------------------------------------

    def test_layout_vertices_default_with_root(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C")]
        positions = layout_vertices(
            ["A", "B", "C"],
            edges,
            layout=None,
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id="A",
        )
        self.assertEqual(len(positions), 3)
        ax, ay = positions["A"]
        bx, by = positions["B"]
        self.assertGreater(ay, by)

    def test_layout_vertices_default_without_root(self) -> None:
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        positions = layout_vertices(
            ["A", "B", "C"],
            edges,
            layout=None,
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id=None,
        )
        self.assertEqual(len(positions), 3)

    def test_layout_vertices_explicit_force(self) -> None:
        edges = [Edge("A", "B")]
        positions = layout_vertices(
            ["A", "B"],
            edges,
            layout="force",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
        )
        self.assertEqual(len(positions), 2)

    def test_layout_vertices_explicit_tree(self) -> None:
        edges = [Edge("A", "B"), Edge("A", "C")]
        positions = layout_vertices(
            ["A", "B", "C"],
            edges,
            layout="tree",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id="A",
        )
        self.assertEqual(len(positions), 3)

    # ------------------------------------------------------------------
    # Root inference
    # ------------------------------------------------------------------

    def test_infer_root_finds_vertex_with_no_incoming(self) -> None:
        """Infer root from directed edges: vertex with no incoming edges."""
        edges = [Edge("R", "A"), Edge("R", "B"), Edge("A", "C")]
        root = _infer_root(["R", "A", "B", "C"], edges)
        self.assertEqual(root, "R")

    def test_infer_root_first_vertex_when_no_edges(self) -> None:
        """When no edges, return first vertex."""
        root = _infer_root(["X", "Y", "Z"], [])
        self.assertEqual(root, "X")

    def test_infer_root_fallback_for_cycle(self) -> None:
        """When all vertices have incoming edges (cycle), return first vertex."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        root = _infer_root(["A", "B", "C"], edges)
        self.assertEqual(root, "A")

    def test_infer_root_binary_tree(self) -> None:
        """Infer root for a binary tree structure."""
        edges = [
            Edge("v0", "v1"), Edge("v0", "v2"),
            Edge("v1", "v3"), Edge("v1", "v4"),
            Edge("v2", "v5"), Edge("v2", "v6"),
        ]
        root = _infer_root(["v0", "v1", "v2", "v3", "v4", "v5", "v6"], edges)
        self.assertEqual(root, "v0")

    def test_layout_hierarchical_infers_root_when_none_provided(self) -> None:
        """Hierarchical layout should infer root when root_id=None."""
        edges = [Edge("R", "A"), Edge("R", "B")]
        positions = layout_vertices(
            ["R", "A", "B"],
            edges,
            layout="hierarchical",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id=None,
        )
        self.assertEqual(len(positions), 3)
        rx, ry = positions["R"]
        ax, ay = positions["A"]
        self.assertGreater(ry, ay)

    def test_layout_tree_infers_root_when_none_provided(self) -> None:
        """Tree layout should infer root when root_id=None."""
        edges = [Edge("P", "Q"), Edge("P", "S")]
        positions = layout_vertices(
            ["P", "Q", "S"],
            edges,
            layout="tree",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id=None,
        )
        self.assertEqual(len(positions), 3)
        px, py = positions["P"]
        qx, qy = positions["Q"]
        self.assertGreater(py, qy)

    # ------------------------------------------------------------------
    # Tree structure detection
    # ------------------------------------------------------------------

    def test_is_tree_structure_simple_tree(self) -> None:
        """Detect simple tree with root and children."""
        edges = [Edge("R", "A"), Edge("R", "B")]
        self.assertTrue(_is_tree_structure(["R", "A", "B"], edges))

    def test_is_tree_structure_binary_tree(self) -> None:
        """Detect binary tree structure."""
        edges = [
            Edge("v0", "v1"), Edge("v0", "v2"),
            Edge("v1", "v3"), Edge("v1", "v4"),
            Edge("v2", "v5"), Edge("v2", "v6"),
        ]
        self.assertTrue(_is_tree_structure(["v0", "v1", "v2", "v3", "v4", "v5", "v6"], edges))

    def test_is_tree_structure_not_tree_cycle(self) -> None:
        """Cycle is not a tree."""
        edges = [Edge("A", "B"), Edge("B", "C"), Edge("C", "A")]
        self.assertFalse(_is_tree_structure(["A", "B", "C"], edges))

    def test_is_tree_structure_not_tree_extra_edge(self) -> None:
        """Graph with extra edge is not a tree."""
        edges = [Edge("R", "A"), Edge("R", "B"), Edge("A", "B")]
        self.assertFalse(_is_tree_structure(["R", "A", "B"], edges))

    def test_is_tree_structure_not_tree_disconnected(self) -> None:
        """Disconnected graph is not a tree."""
        edges = [Edge("A", "B")]
        self.assertFalse(_is_tree_structure(["A", "B", "C"], edges))

    def test_is_tree_structure_single_node(self) -> None:
        """Single node with no edges is a tree."""
        self.assertTrue(_is_tree_structure(["A"], []))

    def test_circular_layout_overridden_for_tree(self) -> None:
        """When graph is a tree, circular layout is overridden to tree layout."""
        edges = [Edge("R", "A"), Edge("R", "B")]
        positions = layout_vertices(
            ["R", "A", "B"],
            edges,
            layout="circular",
            placement_box=self.box,
            canvas_width=100.0,
            canvas_height=100.0,
            root_id=None,
        )
        rx, ry = positions["R"]
        ax, ay = positions["A"]
        self.assertGreater(ry, ay)


class TestGraphLayoutLegacy(TestGraphLayout):
    pass


if __name__ == "__main__":
    unittest.main()

