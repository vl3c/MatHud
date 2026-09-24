"""Round-trip tests: serialize every drawable class, restore it, and compare."""

from __future__ import annotations

import json
import math
import unittest
from typing import Any, Callable, Dict, List, Tuple

from canvas import Canvas
from utils.graph_analyzer import GraphAnalyzer
from workspace_manager import WorkspaceManager


# Every drawable bucket that a saved workspace can contain. Bars are derived from
# plot composites and are intentionally not persisted.
EXPECTED_DRAWABLE_KEYS: Tuple[str, ...] = (
    "Points",
    "Segments",
    "Vectors",
    "Labels",
    "Triangles",
    "Rectangles",
    "Quadrilaterals",
    "Pentagons",
    "Hexagons",
    "Heptagons",
    "Octagons",
    "Nonagons",
    "Decagons",
    "GenericPolygons",
    "Circles",
    "Ellipses",
    "CircleArcs",
    "Angles",
    "Functions",
    "PiecewiseFunctions",
    "ParametricFunctions",
    "FunctionsBoundedColoredAreas",
    "SegmentsBoundedColoredAreas",
    "FunctionSegmentBoundedColoredAreas",
    "ClosedShapeColoredAreas",
    "ContinuousPlots",
    "DiscretePlots",
    "BarsPlots",
    "UndirectedGraphs",
    "DirectedGraphs",
    "Trees",
)


def _regular_polygon(sides: int, radius: float = 3.0) -> List[Tuple[float, float]]:
    return [
        (
            round(radius * math.cos(2 * math.pi * i / sides), 3),
            round(radius * math.sin(2 * math.pi * i / sides), 3),
        )
        for i in range(sides)
    ]


def _graph_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "adjacency_matrix": None,
        "directed": None,
        "root": None,
        "layout": None,
        "placement_box": None,
        "metadata": None,
    }
    kwargs.update(overrides)
    return kwargs


# ----------------------------------------------------------------------
# Scene builders. Scenes are kept small (restore cost grows superlinearly
# with the number of segments), and together they cover every class above.
# ----------------------------------------------------------------------
def _build_primitives(c: Canvas) -> None:
    c.create_point(-90, 90, name="P")
    c.create_segment(-90, 80, -80, 80, label_text="seg label", label_visible=True, extra_graphics=False)
    c.create_segment(-90, 70, -80, 72, extra_graphics=False)
    c.create_vector(-70, 80, -65, 85)
    c.create_label(-60, 80, "hello", name="L1", font_size=18)


def _build_small_polygons(c: Canvas) -> None:
    c.create_polygon([(0, 0), (4, 0), (2, 3)], polygon_type="triangle", extra_graphics=False)
    c.create_polygon([(10, 0), (14, 0), (14, 2), (10, 2)], polygon_type="rectangle", extra_graphics=False)
    c.create_polygon([(20, 0), (25, 0), (24, 3), (21, 2)], polygon_type="quadrilateral", extra_graphics=False)


def _polygon_builder(sides: int, polygon_type: str) -> Callable[[Canvas], None]:
    def build(c: Canvas) -> None:
        c.create_polygon(_regular_polygon(sides), polygon_type=polygon_type, extra_graphics=False)

    return build


def _build_curves(c: Canvas) -> None:
    circle = c.create_circle(0, -30, 5)
    c.create_ellipse(20, -30, 6, 3, rotation_angle=30)
    c.create_circle_arc(
        point1_x=5,
        point1_y=-30,
        point2_x=0,
        point2_y=-25,
        circle_name=circle.name,
        arc_name="ArcMin_Q",
        use_major_arc=False,
        extra_graphics=False,
    )
    c.create_angle(40, -30, 45, -30, 40, -25)
    c.create_region_colored_area(circle_name=circle.name)


def _build_functions_and_areas(c: Canvas) -> None:
    c.draw_function("x^2", name="f", left_bound=-5, right_bound=5)
    c.draw_function("x", name="g", left_bound=-5, right_bound=5)
    c.draw_piecewise_function(
        [
            {"expression": "x^2", "left": None, "right": 0, "left_inclusive": True, "right_inclusive": False},
            {"expression": "x+1", "left": 0, "right": None, "left_inclusive": True, "right_inclusive": True},
        ],
        name="pw",
    )
    c.draw_parametric_function("cos(t)", "sin(t)", name="pc", t_min=0.0, t_max=6.0)
    s1 = c.create_segment(-4, 1, 4, 2, extra_graphics=False)
    s2 = c.create_segment(-4, -1, 4, -2, extra_graphics=False)
    c.create_colored_area("f", "g", left_bound=-1, right_bound=1)
    c.create_colored_area(s1.name, s2.name)
    c.create_colored_area("f", s1.name)


def _build_plots(c: Canvas) -> None:
    c.plot_distribution(
        name="Cont",
        representation="continuous",
        distribution_type="normal",
        distribution_params={"mean": 0.0, "sigma": 1.0},
        plot_bounds={"left_bound": -2.0, "right_bound": 2.0},
    )
    c.plot_distribution(
        name="Disc",
        representation="discrete",
        distribution_type="normal",
        distribution_params={"mean": 0.0, "sigma": 1.0},
        plot_bounds={"left_bound": -2.0, "right_bound": 2.0},
        bar_count=4,
    )
    c.plot_bars(name="Poll", values=[1.0, 2.0], labels_below=["a", "b"], x_start=60.0, y_base=60.0)


def _build_graphs(c: Canvas) -> None:
    c.generate_graph(
        name="G1",
        graph_type="graph",
        vertices=[
            {"name": "A", "x": -50, "y": -60},
            {"name": "B", "x": -40, "y": -60},
            {"name": "C", "x": -45, "y": -52},
            {"name": "I", "x": -35, "y": -52},
        ],
        edges=[
            {"source": 0, "target": 1, "weight": 7},
            {"source": 1, "target": 2, "weight": 1},
            {"source": 0, "target": 2, "weight": 2},
        ],
        **_graph_kwargs(directed=False),
    )
    c.generate_graph(
        name="D1",
        graph_type="dag",
        vertices=[
            {"name": "D", "x": -20, "y": -60},
            {"name": "E", "x": -10, "y": -60},
            {"name": "F", "x": -15, "y": -52},
        ],
        edges=[
            {"source": 0, "target": 1, "weight": 5},
            {"source": 1, "target": 2, "weight": 3},
        ],
        **_graph_kwargs(directed=True),
    )
    c.generate_graph(
        name="T1",
        graph_type="tree",
        vertices=[
            {"name": "R", "x": 20, "y": -50},
            {"name": "S", "x": 15, "y": -60},
            {"name": "U", "x": 25, "y": -60},
        ],
        edges=[{"source": 0, "target": 1}, {"source": 0, "target": 2}],
        **_graph_kwargs(root="R"),
    )


SCENES: Tuple[Tuple[str, Callable[[Canvas], None]], ...] = (
    ("primitives", _build_primitives),
    ("small_polygons", _build_small_polygons),
    ("pentagon", _polygon_builder(5, "pentagon")),
    ("hexagon", _polygon_builder(6, "hexagon")),
    ("heptagon", _polygon_builder(7, "heptagon")),
    ("octagon", _polygon_builder(8, "octagon")),
    ("nonagon", _polygon_builder(9, "nonagon")),
    ("decagon", _polygon_builder(10, "decagon")),
    ("generic_polygon", _polygon_builder(11, "generic")),
    ("curves", _build_curves),
    ("functions_and_areas", _build_functions_and_areas),
    ("plots", _build_plots),
    ("graphs", _build_graphs),
)


class TestWorkspaceRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.manager = WorkspaceManager(self.canvas)

    def _saved_state(self, manager: WorkspaceManager) -> Dict[str, Any]:
        # Mirror a real save/load: persistable snapshot, JSON encoded by the server.
        return json.loads(json.dumps(manager._snapshot_persistable_canvas_state()))

    def _drawable_buckets(self, state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        return {
            key: sorted(
                [self._strip_volatile(item) for item in value],
                key=lambda item: str(item.get("name", "")),
            )
            for key, value in state.items()
            if isinstance(value, list) and value and key != "computations"
        }

    def _strip_volatile(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Keys prefixed with "_" are render-cache hints, not persisted semantics.
        return {key: value for key, value in item.items() if not key.startswith("_")}

    def _round_trip(self, manager: WorkspaceManager) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        before = self._saved_state(manager)
        manager._restore_workspace_state(json.loads(json.dumps(before)))
        after = self._saved_state(manager)
        return before, after

    def test_round_trip_preserves_every_drawable_class(self) -> None:
        covered: set[str] = set()
        for scene_name, build in SCENES:
            with self.subTest(scene=scene_name):
                canvas = Canvas(500, 500, draw_enabled=False)
                manager = WorkspaceManager(canvas)
                build(canvas)
                before, after = self._round_trip(manager)
                before_buckets = self._drawable_buckets(before)
                after_buckets = self._drawable_buckets(after)
                covered.update(before_buckets.keys())
                self.assertEqual(
                    {key: len(items) for key, items in after_buckets.items()},
                    {key: len(items) for key, items in before_buckets.items()},
                    f"Drawable counts changed after restoring scene '{scene_name}'",
                )
                for key, expected in before_buckets.items():
                    self.assertEqual(after_buckets.get(key), expected, f"{key} state changed in '{scene_name}'")

        missing = [key for key in EXPECTED_DRAWABLE_KEYS if key not in covered]
        self.assertEqual(missing, [], f"Round-trip scenes are missing drawable classes: {missing}")
        unexpected = sorted(key for key in covered if key not in EXPECTED_DRAWABLE_KEYS)
        self.assertEqual(
            unexpected,
            [],
            f"New serializable drawable classes must be added to the round-trip test: {unexpected}",
        )

    def _graph_vertex_names(self, graph_name: str) -> List[str]:
        # Vertex points in creation order (the name generator may rename requested names).
        return [point.name for point in self.canvas.get_graph(graph_name)._isolated_points]

    def _analyze_graph(self, graph_name: str, operation: str, params: Any = None) -> Dict[str, Any]:
        # Analyze the manager-captured graph state (what the analyze_graph tool consumes).
        state = self.canvas.drawable_manager.graph_manager.capture_state(graph_name)
        self.assertIsNotNone(state, f"Graph {graph_name} should be analyzable")
        return GraphAnalyzer.analyze(state, operation, params)

    def test_restored_graphs_support_analysis(self) -> None:
        _build_graphs(self.canvas)
        start, goal, middle, isolated = self._graph_vertex_names("G1")
        root_before = self.canvas.get_graph("T1").root
        shortest_params = {"start": start, "goal": goal}
        expected_path = self._analyze_graph("G1", "shortest_path", shortest_params)
        expected_topo = self._analyze_graph("D1", "topological_sort")
        self._round_trip(self.manager)

        path = self._analyze_graph("G1", "shortest_path", shortest_params)
        self.assertEqual(path.get("path"), [start, middle, goal])
        self.assertEqual(path.get("cost"), 3.0)
        self.assertEqual(path, expected_path)
        self.assertEqual(self._analyze_graph("D1", "topological_sort"), expected_topo)

        g1 = self.canvas.get_graph("G1")
        d1 = self.canvas.get_graph("D1")
        t1 = self.canvas.get_graph("T1")
        self.assertEqual(g1.get_class_name(), "UndirectedGraph")
        self.assertFalse(g1.directed)
        self.assertIn(isolated, g1.vertices.values(), "Isolated vertex should stay part of the graph")
        self.assertEqual(d1.get_class_name(), "DirectedGraph")
        self.assertTrue(d1.directed)
        self.assertEqual(sorted(e.get("weight") for e in d1.edges), [3.0, 5.0])
        self.assertEqual(t1.get_class_name(), "Tree")
        self.assertIsNotNone(root_before)
        self.assertEqual(t1.root, root_before)

    def test_legacy_graph_state_without_new_fields_restores_edges(self) -> None:
        _build_graphs(self.canvas)
        root_before = self.canvas.get_graph("T1").root
        state = self._saved_state(self.manager)
        for key in ("UndirectedGraphs", "DirectedGraphs", "Trees"):
            for item in state.get(key, []):
                item["args"].pop("isolated_points", None)
        for item in state.get("Vectors", []):
            item["args"].pop("label", None)

        self.manager._restore_workspace_state(state)

        g1 = self.canvas.get_graph("G1")
        d1 = self.canvas.get_graph("D1")
        self.assertIsNotNone(g1)
        self.assertIsNotNone(d1)
        self.assertEqual(len(g1.edges), 3)
        self.assertEqual(len(d1.edges), 2)
        self.assertEqual(self.canvas.get_graph("T1").root, root_before)

    def test_restore_does_not_synthesize_triangles_from_segments(self) -> None:
        # Three connected segments without a saved Triangle must not gain one on reload.
        self.canvas.create_segment(0, 0, 4, 0, extra_graphics=False)
        self.canvas.create_segment(4, 0, 2, 3, extra_graphics=False)
        self.canvas.create_segment(2, 3, 0, 0, extra_graphics=False)
        self.assertEqual(self.canvas.get_drawables_by_class_name("Triangle"), [])

        self._round_trip(self.manager)

        self.assertEqual(len(self.canvas.get_drawables_by_class_name("Segment")), 3)
        self.assertEqual(self.canvas.get_drawables_by_class_name("Triangle"), [])


if __name__ == "__main__":
    unittest.main()
