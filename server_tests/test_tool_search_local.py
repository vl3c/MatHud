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

    # -- "Exact" answers --

    def test_exact_value_goes_to_simplify(self) -> None:
        assert _top_n(self.service, "Give 1/3 + 1/4 as an exact fraction.")[0] == "simplify"
        assert _top_n(self.service, "What is the exact value of sqrt(12)?")[0] == "simplify"

    def test_exact_function_value_or_area_keeps_its_tools_first(self) -> None:
        """'Exact' adds simplify behind the tools a function value or an area needs."""
        assert _top_n(self.service, "What is f(3) as a fraction?")[0] == "evaluate_expression"
        assert _top_n(self.service, "the exact area of a circle of radius 3")[0] == "calculate_area"
        assert "simplify" in _top_n(self.service, "the exact area of a circle of radius 3")

    def test_exact_function_value_at_a_constant_keeps_evaluate_expression(self) -> None:
        """A one-letter function at a fraction or at pi is a function value too, f(x) is not."""
        for query in ("exact value of f(1/3)", "exact value of h(1/2)", "exact value of f(pi/4)"):
            assert _top_n(self.service, query)[0] == "evaluate_expression", query
        assert "simplify" in _top_n(self.service, "exact value of f(1/3)")
        assert "evaluate_expression" in _top_n(self.service, "exact value of f(sqrt(2))")
        assert _top_n(self.service, "exact value of f(x)")[0] == "simplify"

    def test_letter_before_a_fraction_is_no_function_value_without_exact(self) -> None:
        """Outside exact queries only f(3)-style calls are function values, not a(1/2)."""
        assert _top_n(self.service, "plot y = a(1/2)x")[0] == "draw_function"

    def test_exact_distance_or_length_goes_to_simplify(self) -> None:
        """No tool measures a distance or a length, so the exact radical comes from simplify."""
        assert _top_n(self.service, "Find the exact distance between A and B")[0] == "simplify"
        assert "simplify" in _top_n(self.service, "exact length of segment AB")[:3]
        assert "simplify" in _top_n(self.service, "distance between (1,2) and (4,6) as a radical")
        assert "simplify" in _top_n(self.service, "give the angle in radians as an exact multiple of pi")

    def test_curve_length_goes_to_arc_length(self) -> None:
        """A curve's length is arc_length's; a segment's or a side's is not."""
        for query in (
            "arc length of y = x^2 from 0 to 1",
            "how long is the curve y = ln(x) from 1 to e",
            "length of the parametric curve x = cos(t)^3, y = sin(t)^3",
            "perimeter of the cardioid r = 1 + cos(theta)",
            "length of the spiral r = theta from 0 to 2pi",
            "distance traveled by a particle at (t^2, t^3) from t = 0 to 2",
            "length of one arch of the cycloid x = t - sin(t), y = 1 - cos(t)",
            "plot the cardioid r = 1 + cos(θ) and compute its perimeter",
            "perimeter of an ellipse with semi-axes 3 and 1",
            "length of sin(x) from 0 to pi",
        ):
            assert _top_n(self.service, query)[0] == "arc_length", query
        for query in ("what is the length of segment AB", "length of vector v", "perimeter of polygon ABCD"):
            assert "arc_length" not in _top_n(self.service, query)[:3], query

    def test_loops_and_arcs_of_a_graph_stay_with_graph_tools(self) -> None:
        """A loop, arc or perimeter of a graph of vertices and edges is not a curve's length."""
        for query in (
            "length of the loop in graph G",
            "what is the length of the loop A-B-C-A in G1",
            "the perimeter of the graph G",
        ):
            top = _top_n(self.service, query)
            assert top[0] == "analyze_graph", (query, top)
            assert "arc_length" not in top, (query, top)

    def test_exact_arc_length_keeps_arc_length_first(self) -> None:
        """The exact form is the integral of the speed: integrate and simplify follow arc_length."""
        top = _top_n(self.service, "exact arc length of y = x^(3/2) from 0 to 4")
        assert top[0] == "arc_length"
        assert {"integrate", "simplify"} <= set(top)

    def test_distance_travelled_from_a_velocity_is_an_integral(self) -> None:
        """Along a line, the distance travelled is the integral of |v(t)|."""
        top = _top_n(self.service, "total distance traveled with velocity v(t) = t^2 - 4 on [0, 3]")
        assert top[0] == "integrate"
        for query in (
            "velocity v(t) = 3t^2, find the distance travelled from t=0 to t=2",
            "given v(t)=t^2-1, what is the total distance traveled from t=0 to t=3",
            "find the displacement and total distance for v(t) = cos(t), 0 <= t <= pi",
            "the particle's position is x(t) = t^3 - 6t^2, find the total distance traveled from t=0 to t=5",
            "v(t) = 10 - 2t, total distance from t = 0 to t = 8",
        ):
            top = _top_n(self.service, query)
            assert "integrate" in top and "numeric_integrate" in top, (query, top)
            assert "arc_length" not in top, (query, top)

    def test_distance_travelled_in_the_plane_is_an_arc_length(self) -> None:
        for query in (
            "x = 3t, y = t^2, find the total distance travelled from t=0 to t=2",
            "a particle moves with x(t) = cos(t), y(t) = sin(t); find the distance travelled from t=0 to 2pi",
            "distance traveled by the particle with position (cos(t), sin(2t)) for t in [0, pi]",
            "a particle moves along y = x^2; find the distance travelled from x=0 to x=1",
        ):
            assert _top_n(self.service, query)[0] == "arc_length", query

    def test_distances_that_are_not_a_path_length(self) -> None:
        """Velocity components give the integral of the speed; a projectile's horizontal distance is
        a coordinate; a path between named points is a polyline or a graph path."""
        for query in (
            "a particle moves in the xy-plane so that dx/dt = 3t^2 and dy/dt = 2t; find the total distance traveled from t = 0 to 2",
            "a particle moving in the plane has velocity vector (2t, 3t^2); find the distance traveled from t = 0 to 1",
            "the velocity of a particle moving in the plane is v(t) = (3t, 4t); how far does it travel from t = 0 to 2",
        ):
            top = _top_n(self.service, query)
            assert "integrate" in top and "numeric_integrate" in top, (query, top)
            assert "arc_length" not in top, (query, top)
        for query in (
            "a ball is thrown with x(t) = 15t and y(t) = 2 + 10t - 4.9t^2; how far does it travel horizontally before hitting the ground",
            "find the total distance along the shortest path from A to D",
            "total distance walked along the path from A to B to C",
        ):
            assert _top_n(self.service, query)[0] != "arc_length", query

    def test_its_length_of_a_polygon_near_a_curve_is_not_an_arc_length(self) -> None:
        """"its perimeter" is a rectangle's or a segment's when the query has one, or optimises it."""
        for query in (
            "a rectangle is inscribed under the curve y = 4 - x^2; maximize its perimeter",
            "a segment joins the origin to a point on the curve y = 1/x; minimize its length",
            "a triangle has vertices on the curve y = x^2; compute its perimeter",
        ):
            assert _top_n(self.service, query)[0] != "arc_length", query
        assert _top_n(self.service, "plot the cardioid r = 1 + cos(θ) and compute its perimeter")[0] == "arc_length"

    def test_draw_and_measure_keeps_the_draw_tool(self) -> None:
        top = _top_n(self.service, "draw y = x^2 and find its arc length from 0 to 1")
        assert top[:2] == ["arc_length", "draw_function"]

    # -- Implicit curves and inequality regions --

    def test_equations_in_x_and_y_draw_an_implicit_curve(self) -> None:
        for query in (
            "draw x^2 + y^2 = 25",
            "graph the hyperbola x^2 - y^2 = 1",
            "plot x = y^2 - 4",
            "sketch the folium x^3 + y^3 = 6xy",
            "draw the vertical line x = -2",
            "plot the line 2x + 3y = 6",
            "draw the curve where x^2 + y^2 equals 9",
        ):
            assert _top_n(self.service, query)[0] == "draw_implicit_curve", query

    def test_inequalities_shade_a_region(self) -> None:
        for query in (
            "graph the inequality y > 2x + 1",
            "shade x^2 + y^2 <= 9",
            "shade the feasible region x + y <= 10, x >= 0, y >= 0",
            "graph the system y < x^2",
            "graph -1 < x <= 2",
            "shade where y < -1 or y > 1",
        ):
            assert _top_n(self.service, query)[0] == "shade_inequality", query

    def test_relations_for_other_tools_keep_their_tools(self) -> None:
        for query, tool in (
            ("plot y = x^2 from x = -5 to 5", "draw_function"),
            ("graph the parabola y = x^2 - 4", "draw_function"),
            ("draw a point B at x=50, y=-30", "create_point"),
            ("find (x,y) for 2x+3y=7 and x-y=4", "solve_system_of_equations"),
            ("solve x^2 - 4 >= 0", "solve"),
            ("solve for y: x^2 + y^2 = 25", "solve"),
            ("plot x(t) = cos(t), y(t) = sin(t) for 0 <= t <= 2*pi", "draw_parametric_function"),
            ("Graph f(x) with two rules: -x when x<2, and x-4 when x>=2.", "draw_piecewise_function"),
            ("implicit differentiation of x^2 + y^2 = 25", "derive"),
            # Bare relations are equations to solve; measuring and other objects keep their tools
            ("2x + 3y = 7 and x - y = 4", "solve_system_of_equations"),
            ("x^2 + y^2 = 25 and x + y = 7", "solve_system_of_equations"),
            ("calculate the area of the region x^2 + y^2 <= 4", "calculate_area"),
            ("what is the area of the circle x^2 + y^2 = 25", "calculate_area"),
            ("arc length of x^2 + y^2 = 1", "arc_length"),
            ("integrate x*y over the region x^2 + y^2 < 1", "integrate"),
            ("zoom so that -10 < x < 10 and -5 < y < 5", "zoom"),
            ("draw a segment from x = 0 to x = 5 at y = 2", "create_segment"),
            ("plot the histogram where x > 5", "plot_histogram"),
            ("plot y = x^2 - 1 and shade between it and y = 0 for -1 < x < 1", "create_colored_area"),
            ("plot the function y = x^2 - 4 and shade where y < 0", "draw_function"),
            ("graph the circle x^2 + y^2 = 4", "create_circle"),
        ):
            assert _top_n(self.service, query)[0] == tool, query
        for query, tool in (
            ("2x + y = 5, x - 3y = -1", "solve"),
            ("what are the intercepts of 2x + 3y = 6", "solve"),
        ):
            assert tool in _top_n(self.service, query), query

    def test_drawing_and_measuring_keeps_both_tools(self) -> None:
        for query, first in (
            ("draw x^2/9 + y^2/4 = 1 and calculate the arc length", "arc_length"),
            ("draw x*y = 4 and find the intercepts", "solve"),
        ):
            assert _top_n(self.service, query)[:2] == [first, "draw_implicit_curve"], query
        for query in (
            "plot x^2 + y^2 = 4 and find its perimeter",
            "graph x^2 + y^2 = 25 and compute the circumference",
        ):
            assert "draw_implicit_curve" in _top_n(self.service, query)[:2], query
        for query in ("how does x^2 + y^2 = 25 look", "what does x^3 + y^3 = 6xy look like"):
            assert _top_n(self.service, query)[0] == "draw_implicit_curve", query
        # draw_function cannot draw these relations, so it goes below the implicit tool
        for query in ("plot x^2 + y^2 = 4 and find its perimeter", "graph 2x + 3y = 6 and find the intercepts"):
            top = _top_n(self.service, query)
            assert top.index("draw_implicit_curve") < top.index("draw_function"), query
        top = _top_n(self.service, "shade x + y <= 4, x >= 0, y >= 0 and find the corner points")
        assert "shade_inequality" in top[:2]

    def test_explicit_functions_after_short_words_stay_functions(self) -> None:
        for query, tool in (
            ("plot the curve given by y = x^2", "draw_function"),
            ("graph the line given by y = 2x + 1", "draw_function"),
            ("plot for me y = sin(x)", "draw_function"),
            ("shade the region bounded by y = x^2 and y = 4", "create_region_colored_area"),
            ("fill region R1 inside circle c1 with red", "create_region_colored_area"),
        ):
            assert _top_n(self.service, query)[0] == tool, query

    def test_editing_implicit_curves_and_shaded_regions(self) -> None:
        for query, tool in (
            ("delete the implicit curve c1", "delete_implicit_curve"),
            ("make implicit curve c1 blue", "update_implicit_curve"),
            ("erase the curve x^2 + y^2 = 4", "delete_implicit_curve"),
            ("delete the shaded region R1", "delete_colored_area"),
            ("remove the feasible region", "delete_colored_area"),
            ("make the region R1 red", "update_colored_area"),
            ("change region R2 opacity to 0.1", "update_colored_area"),
            ("shade region R2 blue", "update_colored_area"),
            ("fill region R1 with yellow", "update_colored_area"),
            ("Color the polygonal region enclosed by segments AB, BC, CD, DA.", "create_region_colored_area"),
            ("fill the region inside circle c1 with red", "create_region_colored_area"),
            ("delete point P in the shaded region", "delete_point"),
            ("change the triangle so its area is 10", "calculate_area"),
        ):
            assert _top_n(self.service, query)[0] == tool, query

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

    def test_stats_correlation(self) -> None:
        """Correlation and covariance of paired data."""
        names = _top_n(self.service,
                       "what is the correlation coefficient and covariance of "
                       "x = 1, 2, 3, 4, 5 and y = 2, 4, 5, 4, 5?")
        assert "compute_descriptive_statistics" in names

    def test_stats_correlation_of_a_fit(self) -> None:
        """A fit asked for its correlation: the linear fit reports r itself."""
        names = _top_n(self.service, "fit a line and give the correlation")
        assert names[:2] == ["fit_regression", "compute_descriptive_statistics"]

    def test_stats_correlate_two_lists(self) -> None:
        """The verb alone still finds the statistics tool."""
        names = _top_n(self.service, "correlate x = 1, 2, 3 with y = 2, 4, 7")
        assert names[0] == "compute_descriptive_statistics"

    def test_covariance_matrix_is_linear_algebra(self) -> None:
        """A covariance matrix is computed by the linear algebra tool, not the statistics one."""
        for query in ("covariance matrix", "correlation matrix", "compute the covariance matrix of A"):
            names = _top_n(self.service, query)
            assert names[0] == "evaluate_linear_algebra_expression", (query, names)

    def test_correlation_on_a_plotted_graph_is_statistics(self) -> None:
        """A plain "graph" is usually a plot, so its correlation is still the statistics tool."""
        for query in (
            "graph the correlation of x and y",
            "correlation graph",
            "correlation of points on the graph",
            "do these graphs correlate",
        ):
            names = _top_n(self.service, query)
            assert names[0] == "compute_descriptive_statistics", (query, names)

    def test_correlation_of_everyday_tree_or_path_data_is_statistics(self) -> None:
        """One everyday word such as "tree", "path" or "vertex" names data, as do number lists."""
        for query in (
            "find the correlation between tree age and height: ages [1,2,3], heights [2,4,5]",
            "correlation between tree height and trunk diameter",
            "correlation between edge length and weight [1,2,3] [4,5,6]",
            "covariance of path length and travel time",
            "correlate the number of nodes and runtime",
            "correlation between the x and y coordinates of the triangle vertices",
            "correlation between parabola vertex height and coefficient a",
            "correlation between network size and path length [1,2,3] [2,3,4]",
            "find the covariance of path lengths [1,2,3] and node counts [3,4,5]",
        ):
            names = _top_n(self.service, query)
            assert names[0] == "compute_descriptive_statistics", (query, names)

    def test_correlating_graph_structure_keeps_the_graph_tool(self) -> None:
        """Edges and trees to correlate are a graph analysis, not paired lists."""
        names = _top_n(self.service, "correlate the edge weights of the tree")
        assert "analyze_graph" in names[:3]
        assert "compute_descriptive_statistics" not in names[:3]

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
