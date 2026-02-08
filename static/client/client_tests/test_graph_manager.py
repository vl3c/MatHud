from __future__ import annotations

import unittest
from typing import Dict, List, Any

from drawables.point import Point
from drawables.segment import Segment
from drawables.tree import Tree
from managers.drawables_container import DrawablesContainer
from managers.graph_manager import GraphManager
from geometry.graph_state import TreeState, GraphVertexDescriptor, GraphEdgeDescriptor
from .simple_mock import SimpleMock


class TestGraphManager(unittest.TestCase):
    def setUp(self) -> None:
        self.drawables = DrawablesContainer()

        self.canvas = SimpleMock(
            name="CanvasMock",
            draw_enabled=True,
            draw=SimpleMock(),
            width=1000.0,
            height=800.0,
            undo_redo_manager=SimpleMock(
                name="UndoRedoMock",
                archive=SimpleMock(),
            ),
        )

        self.name_generator = SimpleMock(
            name="NameGeneratorMock",
            generate_point_name=lambda name: name if name else "P",
            split_point_names=lambda expr, count: ["", ""][:count],
        )

        self.dependency_manager = SimpleMock(
            name="DependencyManagerMock",
            analyze_drawable_for_dependencies=SimpleMock(),
            get_children=lambda d: set(),
            get_all_children=lambda d: set(),
            get_all_parents=lambda d: set(),
            remove_drawable=SimpleMock(),
        )

        self.points_created: List[Point] = []
        def create_point(x: float, y: float, name: str = "", color: str = None, extra_graphics: bool = True) -> Point:
            p = Point(x, y, name=name if name else f"P{len(self.points_created)}")
            self.points_created.append(p)
            self.drawables.add(p)
            return p

        self.point_manager = SimpleMock(
            name="PointManagerMock",
            create_point=create_point,
        )

        self.segments_created: List[Segment] = []
        def create_segment_from_points(p1: Point, p2: Point, name: str = "", color: str = None, label_text: str = "", label_visible: bool = False) -> Segment:
            seg = Segment(p1, p2, color=color or "#000000")
            self.segments_created.append(seg)
            self.drawables.add(seg)
            return seg

        self.segment_manager = SimpleMock(
            name="SegmentManagerMock",
            create_segment_from_points=create_segment_from_points,
            create_segment=lambda *args, **kwargs: create_segment_from_points(
                Point(args[0], args[1]), Point(args[2], args[3]), **{k: v for k, v in kwargs.items() if k != 'extra_graphics'}
            ),
        )

        self.vector_manager = SimpleMock(
            name="VectorManagerMock",
            create_vector_from_points=SimpleMock(return_value=None),
            create_vector=SimpleMock(return_value=None),
        )

        self.drawable_manager_proxy = SimpleMock(
            name="DrawableManagerProxyMock",
            create_drawables_from_new_connections=SimpleMock(),
        )

        self.graph_manager = GraphManager(
            self.canvas,
            self.drawables,
            self.name_generator,
            self.dependency_manager,
            self.point_manager,
            self.segment_manager,
            self.vector_manager,
            self.drawable_manager_proxy,
        )

    def test_tree_with_directed_edges_creates_segments_not_vectors(self) -> None:
        """Test that trees create segments even when edges have directed=True."""
        state = self.graph_manager.build_graph_state(
            name="tree_R",
            graph_type="tree",
            vertices=[
                {"name": "R"},
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D"},
                {"name": "E"},
                {"name": "F"},
                {"name": "G"},
                {"name": "H"},
            ],
            edges=[
                {"source": 0, "target": 1, "directed": True},
                {"source": 0, "target": 2, "directed": True},
                {"source": 0, "target": 3, "directed": True},
                {"source": 1, "target": 4, "directed": True},
                {"source": 1, "target": 5, "directed": True},
                {"source": 1, "target": 6, "directed": True},
                {"source": 1, "target": 7, "directed": True},
                {"source": 3, "target": 8, "directed": True},
            ],
            adjacency_matrix=None,
            directed=True,
            root="R",
            layout="hierarchical",
            placement_box={"x": -400, "y": -400, "width": 800, "height": 800},
            metadata=None,
        )

        self.assertIsInstance(state, TreeState)
        self.assertEqual(state.directed, False)

        graph = self.graph_manager.create_graph(state)

        self.assertIsInstance(graph, Tree)
        self.assertEqual(len(self.points_created), 9)
        self.assertEqual(len(self.segments_created), 8)
        self.vector_manager.create_vector_from_points.assert_not_called()

    def test_tree_root_resolution_by_name(self) -> None:
        """Test that root specified by vertex name is correctly resolved to vertex ID."""
        state = self.graph_manager.build_graph_state(
            name="simple_tree",
            graph_type="tree",
            vertices=[
                {"name": "Root"},
                {"name": "Child1"},
                {"name": "Child2"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="Root",
            layout=None,
            placement_box=None,
            metadata=None,
        )

        self.assertIsInstance(state, TreeState)
        self.assertEqual(state.root, "v0")

    def test_tree_root_resolution_by_index(self) -> None:
        """Test that root specified by index is correctly resolved to vertex ID."""
        state = self.graph_manager.build_graph_state(
            name="simple_tree",
            graph_type="tree",
            vertices=[
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="0",
            layout=None,
            placement_box=None,
            metadata=None,
        )

        self.assertIsInstance(state, TreeState)
        self.assertEqual(state.root, "v0")

    def test_tree_layout_positions_root_at_top(self) -> None:
        """Test that tree layout places root at highest y-coordinate."""
        state = self.graph_manager.build_graph_state(
            name="tree_layout_test",
            graph_type="tree",
            vertices=[
                {"name": "R"},
                {"name": "A"},
                {"name": "B"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="R",
            layout="tree",
            placement_box={"x": 0, "y": 0, "width": 100, "height": 100},
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)

        self.assertEqual(len(self.points_created), 3)

        root_point = None
        child_points = []
        for p in self.points_created:
            if p.name == "R":
                root_point = p
            else:
                child_points.append(p)

        self.assertIsNotNone(root_point)
        self.assertEqual(len(child_points), 2)

        for child in child_points:
            self.assertGreater(root_point.y, child.y)

    def test_tree_layout_children_centered_under_parent(self) -> None:
        """Test that children are horizontally centered under their parent."""
        state = self.graph_manager.build_graph_state(
            name="tree_centering_test",
            graph_type="tree",
            vertices=[
                {"name": "P"},
                {"name": "L"},
                {"name": "Q"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="P",
            layout="tree",
            placement_box={"x": 0, "y": 0, "width": 100, "height": 100},
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)

        root_point = None
        left_child = None
        right_child = None
        for p in self.points_created:
            if p.name == "P":
                root_point = p
            elif p.name == "L":
                left_child = p
            elif p.name == "Q":
                right_child = p

        self.assertIsNotNone(root_point)
        self.assertIsNotNone(left_child)
        self.assertIsNotNone(right_child)

        children_center_x = (left_child.x + right_child.x) / 2
        self.assertAlmostEqual(root_point.x, children_center_x, places=3)

    def test_tree_asymmetric_layout_proportional_spacing(self) -> None:
        """Test that asymmetric trees allocate horizontal space proportionally.

        Tree structure: R -> A, B, C; A -> D, E, F, G; C -> H
        """
        state = self.graph_manager.build_graph_state(
            name="asymmetric_tree",
            graph_type="tree",
            vertices=[
                {"name": "R"},
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D"},
                {"name": "E"},
                {"name": "F"},
                {"name": "G"},
                {"name": "H"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
                {"source": 0, "target": 3},
                {"source": 1, "target": 4},
                {"source": 1, "target": 5},
                {"source": 1, "target": 6},
                {"source": 1, "target": 7},
                {"source": 3, "target": 8},
            ],
            adjacency_matrix=None,
            directed=None,
            root="R",
            layout="tree",
            placement_box={"x": -400, "y": -400, "width": 800, "height": 800},
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)

        self.assertEqual(len(self.points_created), 9)
        self.assertEqual(len(self.segments_created), 8)

        points_by_name = {p.name: p for p in self.points_created}

        r_point = points_by_name["R"]
        a_point = points_by_name["A"]
        b_point = points_by_name["B"]
        c_point = points_by_name["C"]

        self.assertGreater(r_point.y, a_point.y)
        self.assertAlmostEqual(a_point.y, b_point.y, places=3)
        self.assertAlmostEqual(b_point.y, c_point.y, places=3)

        self.assertLess(a_point.x, b_point.x)
        self.assertLess(b_point.x, c_point.x)

        d_point = points_by_name["D"]
        g_point = points_by_name["G"]
        a_children_center = (d_point.x + g_point.x) / 2
        self.assertAlmostEqual(a_point.x, a_children_center, places=1)

    def test_segments_reference_correct_points(self) -> None:
        """Test that segments reference the exact Point objects from the graph."""
        state = self.graph_manager.build_graph_state(
            name="segment_ref_test",
            graph_type="tree",
            vertices=[
                {"name": "M"},
                {"name": "N"},
            ],
            edges=[
                {"source": 0, "target": 1},
            ],
            adjacency_matrix=None,
            directed=None,
            root="M",
            layout=None,
            placement_box=None,
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)

        self.assertEqual(len(self.segments_created), 1)
        segment = self.segments_created[0]

        self.assertIn(segment.point1, self.points_created)
        self.assertIn(segment.point2, self.points_created)

    def test_tree_root_stored_as_point_name(self) -> None:
        """Test that tree root is stored as actual point name, not internal vertex ID.

        This is critical for analyze_graph operations (levels, lca, etc.) which
        build adjacency maps keyed by point names.
        """
        state = self.graph_manager.build_graph_state(
            name="root_name_test",
            graph_type="tree",
            vertices=[
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="A",
            layout=None,
            placement_box=None,
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)

        # Root should be the point name "A", not internal ID "v0"
        self.assertIsInstance(graph, Tree)
        self.assertEqual(graph.root, "A")

    def test_capture_state_translates_old_root_format(self) -> None:
        """Test that capture_state translates old internal root IDs to point names.

        Trees created before the root translation fix have root stored as "v0", "v1", etc.
        The capture_state method should translate these to actual point names for
        analyze_graph compatibility.
        """
        # Create tree with the fix (root stored as point name)
        state = self.graph_manager.build_graph_state(
            name="backward_compat_test",
            graph_type="tree",
            vertices=[
                {"name": "X"},
                {"name": "Y"},
                {"name": "Z"},
            ],
            edges=[
                {"source": 0, "target": 1},
                {"source": 0, "target": 2},
            ],
            adjacency_matrix=None,
            directed=None,
            root="X",
            layout=None,
            placement_box=None,
            metadata=None,
        )

        graph = self.graph_manager.create_graph(state)
        self.assertIsInstance(graph, Tree)

        # Simulate old format by setting root to internal ID
        graph.root = "v0"

        # Capture state should translate "v0" back to the point name "X"
        captured = self.graph_manager.capture_state("backward_compat_test")
        self.assertIsNotNone(captured)
        self.assertEqual(captured.root, "X")


if __name__ == "__main__":
    unittest.main()

