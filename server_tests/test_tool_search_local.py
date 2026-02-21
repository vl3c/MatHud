"""
Offline benchmark and latency tests for the local tool search engine.

Runs the 190-case benchmark dataset against ``search_tools_local()`` without
any API key.  Executes with standard ``pytest`` in under a second.

Usage::

    python -m pytest server_tests/test_tool_search_local.py -v
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from static.tool_search_service import ToolSearchService, clear_search_cache

DATASET_PATH = Path("server_tests/data/tool_discovery_cases.yaml")


def _load_dataset() -> Dict[str, Any]:
    raw = DATASET_PATH.read_text(encoding="utf-8")
    parsed: Dict[str, Any] = json.loads(raw)
    return parsed


def _get_tool_name(tool: Dict[str, Any]) -> str:
    name: str = tool.get("function", {}).get("name", "")
    return name


class TestLocalSearchBenchmark:
    """Run benchmark dataset against local search (no API key needed)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        clear_search_cache()
        self.service = ToolSearchService.__new__(ToolSearchService)
        self.service._client = None
        self.service._client_initialized = False
        self.service.default_model = None
        self.service.last_error = None

    def _run_benchmark(self, max_results: int = 10) -> Dict[str, Any]:
        """Run the full benchmark and return metrics."""
        dataset = _load_dataset()
        cases = dataset.get("cases", [])

        positive_total = 0
        positive_evaluated = 0
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0

        negative_total = 0
        negative_pass = 0

        failed_cases: List[Dict[str, Any]] = []

        for case in cases:
            query = str(case.get("query", "")).strip()
            expected_any = [str(x) for x in case.get("expected_any", []) if isinstance(x, str)]

            results = self.service.search_tools_local(query, max_results)
            ranked = [_get_tool_name(t) for t in results]

            if not expected_any:
                negative_total += 1
                if not ranked:
                    negative_pass += 1
                continue

            positive_total += 1
            positive_evaluated += 1

            top1 = ranked[0] if ranked else ""
            top3_set = set(ranked[:3])
            top5_set = set(ranked[:5])
            expected_set = set(expected_any)

            is_top1 = top1 in expected_set
            is_top3 = bool(expected_set & top3_set)
            is_top5 = bool(expected_set & top5_set)

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if is_top5:
                top5_hits += 1

            if not is_top5:
                failed_cases.append({
                    "id": case.get("id", "?"),
                    "query": query,
                    "expected": expected_any,
                    "got_top5": ranked[:5],
                })

        top1_rate = top1_hits / positive_evaluated if positive_evaluated else 0.0
        top3_rate = top3_hits / positive_evaluated if positive_evaluated else 0.0
        top5_rate = top5_hits / positive_evaluated if positive_evaluated else 0.0

        return {
            "positive_total": positive_total,
            "positive_evaluated": positive_evaluated,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "top5_hits": top5_hits,
            "top1_rate": top1_rate,
            "top3_rate": top3_rate,
            "top5_rate": top5_rate,
            "negative_total": negative_total,
            "negative_pass": negative_pass,
            "failed_cases": failed_cases,
        }

    def test_local_search_accuracy(self) -> None:
        """Local search should meet accuracy thresholds on benchmark dataset."""
        metrics = self._run_benchmark()

        print(
            f"\nLocal search benchmark: "
            f"top1={metrics['top1_rate']:.3f} ({metrics['top1_hits']}/{metrics['positive_evaluated']}), "
            f"top3={metrics['top3_rate']:.3f} ({metrics['top3_hits']}/{metrics['positive_evaluated']}), "
            f"top5={metrics['top5_rate']:.3f} ({metrics['top5_hits']}/{metrics['positive_evaluated']})"
        )

        if metrics["failed_cases"]:
            print(f"Failed cases ({len(metrics['failed_cases'])}):")
            for fc in metrics["failed_cases"][:15]:
                print(f"  - {fc['id']}: query={fc['query']!r}")
                print(f"    expected={fc['expected']}, got_top5={fc['got_top5']}")

        assert metrics["positive_evaluated"] > 0, "No positive cases evaluated"
        assert metrics["top1_rate"] >= 0.80, (
            f"Top-1 accuracy {metrics['top1_rate']:.3f} below threshold 0.80"
        )
        assert metrics["top3_rate"] >= 0.88, (
            f"Top-3 accuracy {metrics['top3_rate']:.3f} below threshold 0.88"
        )

    def test_local_search_top5_rate(self) -> None:
        """Local search top-5 accuracy should be reasonable."""
        metrics = self._run_benchmark()
        assert metrics["top5_rate"] >= 0.90, (
            f"Top-5 accuracy {metrics['top5_rate']:.3f} below threshold 0.90"
        )


class TestLocalSearchLatency:
    """Ensure local search completes fast enough."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        clear_search_cache()
        self.service = ToolSearchService.__new__(ToolSearchService)
        self.service._client = None
        self.service._client_initialized = False
        self.service.default_model = None
        self.service.last_error = None

    def test_local_search_latency_p99(self) -> None:
        """Local search should complete in under 5ms per query (p99)."""
        queries = [
            "draw a circle",
            "solve x^2 - 4 = 0",
            "create a point at 3, 4",
            "calculate derivative of sin(x)",
            "plot normal distribution mean 0 sigma 1",
            "generate a weighted graph with vertices A B C",
            "undo the last action",
            "save workspace as MyProject",
            "shade area under curve",
            "translate point A by dx=2 dy=3",
            "fit regression to data",
            "compute mean and median of [1,2,3,4,5]",
            "draw a parametric curve",
            "construct perpendicular bisector of segment AB",
            "find shortest path from A to D",
            "zoom in on the canvas",
            "clear everything",
            "create segment from origin to (5,5)",
            "reflect triangle across y-axis",
            "integrate x^2 from 0 to 1",
        ]

        # Warm up
        for q in queries[:3]:
            clear_search_cache()
            self.service.search_tools_local(q)

        # Measure
        latencies: List[float] = []
        for q in queries:
            clear_search_cache()
            start = time.perf_counter()
            self.service.search_tools_local(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
        p99 = latencies[p99_idx]
        avg = sum(latencies) / len(latencies)

        print(f"\nLocal search latency: avg={avg:.2f}ms, p50={p50:.2f}ms, p99={p99:.2f}ms")

        assert p99 < 5.0, f"p99 latency {p99:.2f}ms exceeds 5ms threshold"

    def test_local_search_consistent_results(self) -> None:
        """Same query should return same results (deterministic)."""
        query = "draw a circle at center 0,0 with radius 5"

        clear_search_cache()
        result1 = self.service.search_tools_local(query)
        clear_search_cache()
        result2 = self.service.search_tools_local(query)

        names1 = [_get_tool_name(t) for t in result1]
        names2 = [_get_tool_name(t) for t in result2]
        assert names1 == names2


# ---------------------------------------------------------------------------
# Creative real-world prompt tests
# ---------------------------------------------------------------------------

def _top_n(service: ToolSearchService, query: str, n: int = 5) -> List[str]:
    """Return top-n tool names for a query."""
    clear_search_cache()
    results = service.search_tools_local(query, max_results=n)
    return [_get_tool_name(t) for t in results]


class TestCreativePrompts:
    """Test the classifier against creative, realistic, and tricky user prompts.

    These go beyond the benchmark dataset to probe edge cases, slang,
    multi-step requests, and domain-specific phrasing.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        clear_search_cache()
        self.service = ToolSearchService.__new__(ToolSearchService)
        self.service._client = None
        self.service._client_initialized = False
        self.service.default_model = None
        self.service.last_error = None

    # -- Casual / conversational phrasing --

    def test_casual_circle(self) -> None:
        """Casual 'gimme a circle' should find create_circle."""
        names = _top_n(self.service, "gimme a circle centered at the origin")
        assert "create_circle" in names

    def test_casual_undo(self) -> None:
        """'oops go back' should find undo."""
        names = _top_n(self.service, "oops go back")
        assert "undo" in names

    def test_casual_delete(self) -> None:
        """'get rid of that triangle' should find delete_polygon."""
        names = _top_n(self.service, "get rid of that triangle")
        assert "delete_polygon" in names

    def test_casual_zoom(self) -> None:
        """'I can't see anything, zoom out' should find zoom."""
        names = _top_n(self.service, "I can't see anything, zoom out")
        assert "zoom" in names

    # -- Math class homework scenarios --

    def test_homework_quadratic(self) -> None:
        """Student solving a quadratic."""
        names = _top_n(self.service, "What are the roots of x^2 - 5x + 6?")
        assert "solve" in names

    def test_homework_derivative_chain_rule(self) -> None:
        """Chain rule derivative."""
        names = _top_n(self.service, "take the derivative of sin(3x^2 + 1)")
        assert "derive" in names

    def test_homework_integral_area(self) -> None:
        """Find area under a curve via integration."""
        names = _top_n(self.service,
                       "what's the area under the curve y=x^3 between 0 and 2?")
        assert "integrate" in names or "create_colored_area" in names

    def test_homework_system_word_problem(self) -> None:
        """Word-problem style system of equations."""
        names = _top_n(self.service,
                       "If apples cost $2 and bananas cost $3, and I spent $13 on "
                       "5 fruits, how many of each did I buy?")
        assert "solve_system_of_equations" in names or "solve" in names

    def test_homework_factor_polynomial(self) -> None:
        """Factor a cubic polynomial."""
        names = _top_n(self.service, "factor x^3 - 27 completely")
        assert "factor" in names

    def test_homework_limit(self) -> None:
        """L'Hopital's rule limit."""
        names = _top_n(self.service,
                       "What is the limit of sin(x)/x as x approaches 0?")
        assert "limit" in names

    # -- Engineering / physics style --

    def test_physics_projectile(self) -> None:
        """Projectile motion parametric curve."""
        names = _top_n(self.service,
                       "plot the trajectory: x(t) = 10t, y(t) = 10t - 4.9t^2")
        assert "draw_parametric_function" in names

    def test_physics_unit_conversion(self) -> None:
        """Physics unit conversion."""
        names = _top_n(self.service, "convert 9.8 meters per second squared to feet")
        assert "convert" in names

    def test_engineering_matrix(self) -> None:
        """Stiffness matrix computation."""
        names = _top_n(self.service,
                       "find the inverse of the 3x3 matrix [[1,2,3],[0,1,4],[5,6,0]]")
        assert "evaluate_linear_algebra_expression" in names

    # -- Geometry constructions --

    def test_geometry_circumscribed_circle(self) -> None:
        """Circumscribed circle of a triangle."""
        names = _top_n(self.service,
                       "construct the circumscribed circle of triangle ABC")
        assert "construct_circumcircle" in names

    def test_geometry_incircle(self) -> None:
        """Inscribed circle."""
        names = _top_n(self.service,
                       "draw the inscribed circle inside the triangle")
        assert "construct_incircle" in names

    def test_geometry_midpoint(self) -> None:
        """Find midpoint of a segment."""
        names = _top_n(self.service, "mark the midpoint of segment PQ")
        assert "construct_midpoint" in names

    def test_geometry_parallel(self) -> None:
        """Construct parallel line."""
        names = _top_n(self.service,
                       "draw a line through point C parallel to segment AB")
        assert "construct_parallel_line" in names

    # -- Statistics / data science --

    def test_stats_bell_curve(self) -> None:
        """Bell curve is a normal distribution."""
        names = _top_n(self.service,
                       "show me a bell curve with mean 100 and std dev 15")
        assert "plot_distribution" in names

    def test_stats_bar_chart(self) -> None:
        """Simple bar chart."""
        names = _top_n(self.service,
                       "make a bar chart comparing sales: Q1=100, Q2=150, Q3=80, Q4=200")
        assert "plot_bars" in names

    def test_stats_regression_fit(self) -> None:
        """Fit a trend line to data."""
        names = _top_n(self.service,
                       "fit a line of best fit through these data points")
        assert "fit_regression" in names

    def test_stats_descriptive(self) -> None:
        """Basic descriptive statistics."""
        names = _top_n(self.service,
                       "give me the mean, median, and standard deviation of "
                       "[88, 92, 76, 95, 83, 91, 78]")
        assert "compute_descriptive_statistics" in names

    # -- Graph theory --

    def test_graph_shortest_path(self) -> None:
        """Dijkstra / shortest path."""
        names = _top_n(self.service,
                       "what's the shortest path from node S to node T in the graph?")
        assert "analyze_graph" in names

    def test_graph_minimum_spanning_tree(self) -> None:
        """MST."""
        names = _top_n(self.service,
                       "compute the minimum spanning tree of graph G1")
        assert "analyze_graph" in names

    def test_graph_topological_sort(self) -> None:
        """Topological sort."""
        names = _top_n(self.service,
                       "topologically sort the DAG")
        assert "analyze_graph" in names

    def test_graph_create_network(self) -> None:
        """Create a network graph."""
        names = _top_n(self.service,
                       "build a weighted undirected network with 6 nodes and 8 edges")
        assert "generate_graph" in names

    # -- Workspace management --

    def test_workspace_checkpoint(self) -> None:
        """Save a checkpoint."""
        names = _top_n(self.service, "save my progress as 'homework_ch7'")
        assert "save_workspace" in names

    def test_workspace_resume(self) -> None:
        """Resume previous work."""
        names = _top_n(self.service,
                       "pick up where I left off on the calculus_project workspace")
        assert "load_workspace" in names

    def test_workspace_browse(self) -> None:
        """Browse workspaces."""
        names = _top_n(self.service, "what workspaces do I have saved?")
        assert "list_workspaces" in names

    # -- Canvas operations --

    def test_canvas_wipe(self) -> None:
        """Wipe the canvas."""
        names = _top_n(self.service, "wipe everything clean and start fresh")
        assert "clear_canvas" in names

    def test_canvas_grid_toggle(self) -> None:
        """Toggle grid."""
        names = _top_n(self.service, "hide the grid lines")
        assert "set_grid_visible" in names

    def test_canvas_polar_mode(self) -> None:
        """Switch to polar coordinates."""
        names = _top_n(self.service, "switch to polar coordinate mode")
        assert "set_coordinate_system" in names

    # -- Transforms --

    def test_transform_slide(self) -> None:
        """Translate using casual language."""
        names = _top_n(self.service,
                       "slide the rectangle 3 units to the right and 2 up")
        assert "translate_object" in names

    def test_transform_flip(self) -> None:
        """Reflect using 'flip'."""
        names = _top_n(self.service, "flip the triangle over the x-axis")
        assert "reflect_object" in names

    def test_transform_double_size(self) -> None:
        """Scale up."""
        names = _top_n(self.service,
                       "make the circle twice as big")
        assert "scale_object" in names

    def test_transform_rotate_45(self) -> None:
        """Rotate by 45 degrees."""
        names = _top_n(self.service, "rotate the square 45 degrees clockwise")
        assert "rotate_object" in names

    # -- Colored areas --

    def test_area_shade_between_curves(self) -> None:
        """Shade between two curves."""
        names = _top_n(self.service,
                       "shade the region between y=x^2 and y=x")
        assert "create_region_colored_area" in names or "create_colored_area" in names

    def test_area_highlight_integral(self) -> None:
        """Highlight the area for a definite integral."""
        names = _top_n(self.service,
                       "highlight the area under e^(-x) from 0 to infinity")
        assert "create_colored_area" in names

    # -- Edge cases and tricky phrasing --

    def test_ambiguous_graph_word(self) -> None:
        """'graph' meaning plot, not graph theory."""
        names = _top_n(self.service,
                       "graph the absolute value function |x|")
        assert "draw_function" in names or "draw_piecewise_function" in names

    def test_ambiguous_normal(self) -> None:
        """'normal' meaning distribution, not normal line."""
        names = _top_n(self.service, "plot a normal distribution")
        assert "plot_distribution" in names

    def test_ambiguous_normal_line(self) -> None:
        """'normal line' meaning perpendicular to tangent."""
        names = _top_n(self.service,
                       "draw the normal line to the curve at x=2")
        assert "draw_normal_line" in names

    def test_no_verb_query(self) -> None:
        """Query with no clear verb."""
        names = _top_n(self.service, "circle radius 5 center (0,0)")
        assert "create_circle" in names

    def test_emoji_and_noise(self) -> None:
        """Query with noise characters."""
        names = _top_n(self.service, "!!! draw a big triangle please!!!")
        assert "create_polygon" in names

    def test_very_long_query(self) -> None:
        """Verbose multi-sentence request."""
        names = _top_n(self.service,
                       "I'm working on my geometry homework and I need to create "
                       "a point at coordinates (3, 7). This point represents the "
                       "location of a lighthouse on my map. Could you help me "
                       "place it on the canvas?")
        assert "create_point" in names

    def test_mixed_math_notation(self) -> None:
        """Math notation mixed with words."""
        names = _top_n(self.service,
                       "compute integral from 0 to pi of sin(x) dx")
        assert "integrate" in names

    def test_creative_lissajous(self) -> None:
        """Lissajous curve (parametric)."""
        names = _top_n(self.service,
                       "draw a Lissajous figure with x=sin(3t) y=sin(2t)")
        assert "draw_parametric_function" in names

    def test_creative_rose_curve(self) -> None:
        """Rose curve expressed parametrically."""
        names = _top_n(self.service,
                       "plot the rose curve r=cos(4*theta) in polar")
        assert "draw_parametric_function" in names or "draw_function" in names

    def test_inspect_perpendicularity(self) -> None:
        """Check if two segments are perpendicular."""
        names = _top_n(self.service,
                       "are segments AB and CD perpendicular to each other?")
        assert "inspect_relation" in names

    def test_numeric_approximation(self) -> None:
        """Numerical root finding."""
        names = _top_n(self.service,
                       "find an approximate solution to x = cos(x)")
        assert "solve_numeric" in names

    def test_expand_binomial(self) -> None:
        """Binomial expansion."""
        names = _top_n(self.service, "expand (2x - 3)^4")
        assert "expand" in names

    def test_simplify_trig_identity(self) -> None:
        """Simplify a trig expression."""
        names = _top_n(self.service,
                       "simplify sin^2(x) + cos^2(x) - 1")
        assert "simplify" in names

    def test_delete_specific_plot(self) -> None:
        """Delete a named plot."""
        names = _top_n(self.service,
                       "remove the distribution plot called 'bell1'")
        assert "delete_plot" in names

    def test_update_circle_color(self) -> None:
        """Change a circle's color."""
        names = _top_n(self.service, "change circle C1 to red")
        assert "update_circle" in names

    def test_ellipse_creation(self) -> None:
        """Create an ellipse."""
        names = _top_n(self.service,
                       "draw an ellipse with semi-major axis 5 and semi-minor axis 3")
        assert "create_ellipse" in names

    def test_canvas_state_inspection(self) -> None:
        """Inspect canvas state."""
        names = _top_n(self.service, "what objects are currently on the canvas?")
        assert "get_current_canvas_state" in names

    def test_angle_creation(self) -> None:
        """Create an angle."""
        names = _top_n(self.service,
                       "show the angle between rays BA and BC")
        assert "create_angle" in names

    def test_vector_creation(self) -> None:
        """Create a vector."""
        names = _top_n(self.service,
                       "draw a vector from (1,1) pointing to (4,5)")
        assert "create_vector" in names

    def test_piecewise_absolute_value(self) -> None:
        """Absolute value as piecewise."""
        names = _top_n(self.service,
                       "draw f(x) = x when x >= 0 and -x when x < 0")
        assert "draw_piecewise_function" in names

    def test_tangent_line_at_point(self) -> None:
        """Tangent line at a specific point."""
        names = _top_n(self.service,
                       "draw the tangent to y=x^3 at the point where x=1")
        assert "draw_tangent_line" in names

    def test_convert_degrees_radians(self) -> None:
        """Convert between angle units."""
        names = _top_n(self.service, "how many radians is 270 degrees?")
        assert "convert" in names

    def test_bisect_angle(self) -> None:
        """Bisect an angle."""
        names = _top_n(self.service,
                       "bisect the angle at vertex B in triangle ABC")
        assert "construct_angle_bisector" in names

    def test_colored_area_function(self) -> None:
        """Shade under a specific function."""
        names = _top_n(self.service,
                       "fill the area under y=1/x from x=1 to x=e with blue")
        assert "create_colored_area" in names

    def test_delete_workspace(self) -> None:
        """Delete a workspace."""
        names = _top_n(self.service,
                       "remove the workspace named 'old_draft'")
        assert "delete_workspace" in names
