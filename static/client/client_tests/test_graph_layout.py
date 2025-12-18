from __future__ import annotations

import unittest

from utils.graph_layout import (
    layout_vertices,
    _circular_layout,
    _grid_layout,
    _tree_layout,
    _force_directed_layout,
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
    # Grid layout
    # ------------------------------------------------------------------

    def test_grid_layout_single_node(self) -> None:
        positions = _grid_layout(["A"], self.box)
        x, y = positions["A"]
        self.assertAlmostEqual(x, 50.0, places=5)
        self.assertAlmostEqual(y, 50.0, places=5)

    def test_grid_layout_four_nodes(self) -> None:
        positions = _grid_layout(["A", "B", "C", "D"], self.box)
        self.assertEqual(len(positions), 4)

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
        positions = _force_directed_layout(["A", "B"], edges, self.box, iterations=20)
        self.assertEqual(len(positions), 2)
        
        ax, ay = positions["A"]
        bx, by = positions["B"]
        self.assertGreaterEqual(ax, 0.0)
        self.assertLessEqual(ax, 100.0)
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
        positions = _force_directed_layout(["A", "B", "C"], [], self.box, iterations=20)
        self.assertEqual(len(positions), 3)

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


class TestGraphLayoutLegacy(TestGraphLayout):
    pass


if __name__ == "__main__":
    unittest.main()

