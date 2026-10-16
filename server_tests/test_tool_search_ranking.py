"""
Ranking guards for the local tool search.

Destructive tools must not outrank everything else for harmless queries,
function-value questions must find evaluate_expression, and run_tests is only
offered when the query is about tests.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from static.tool_search_service import CONFIDENCE_THRESHOLD, ToolSearchService, clear_search_cache

DESTRUCTIVE_PREFIXES = ("delete_", "clear_", "reset_")


@pytest.fixture()
def service() -> ToolSearchService:
    clear_search_cache()
    return ToolSearchService()


def _names(service: ToolSearchService, query: str, max_results: int = 10) -> List[str]:
    return [tool["function"]["name"] for tool in service.search_tools_local(query, max_results)]


def test_function_value_query_ranks_evaluate_first(service: ToolSearchService) -> None:
    names = _names(service, "what is f(3) for the function I drew")
    assert names[0] == "evaluate_expression"
    assert not any(name.startswith(DESTRUCTIVE_PREFIXES) for name in names[:5])


def test_table_of_values_ranks_evaluate_first(service: ToolSearchService) -> None:
    names = _names(service, "make a table of values of f")
    assert names[0] == "evaluate_expression"


def test_export_image_query_has_no_destructive_tools_or_tests(service: ToolSearchService) -> None:
    names = _names(service, "export my canvas as an image")
    assert not any(name.startswith(DESTRUCTIVE_PREFIXES) for name in names[:5])
    assert "run_tests" not in names


def test_destructive_tools_rank_after_other_matches_without_destructive_verb(service: ToolSearchService) -> None:
    names = _names(service, "show me the function on the canvas", max_results=20)
    destructive = [i for i, name in enumerate(names) if name.startswith(DESTRUCTIVE_PREFIXES)]
    others = [i for i, name in enumerate(names) if not name.startswith(DESTRUCTIVE_PREFIXES)]
    assert others, names
    assert all(i > max(others) for i in destructive), names


@pytest.mark.parametrize(
    "query, expected",
    [
        ("delete the function f", "delete_function"),
        ("remove point A", "delete_point"),
        ("clear the canvas", "clear_canvas"),
        ("wipe everything clean and start fresh", "clear_canvas"),
        ("get rid of the circle", "delete_circle"),
    ],
)
def test_destructive_verbs_still_find_destructive_tools(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


def test_reset_verb_finds_reset_canvas(service: ToolSearchService) -> None:
    assert "reset_canvas" in _names(service, "reset the canvas view to default")[:3]


def test_run_tests_only_offered_for_test_queries(service: ToolSearchService) -> None:
    assert _names(service, "run the tests")[0] == "run_tests"
    assert "run_tests" not in _names(service, "clear the canvas", max_results=20)
    assert "run_tests" not in _names(service, "zoom the canvas view", max_results=20)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("deletes the function f", "delete_function"),
        ("deleting the circle", "delete_circle"),
        ("erasing segment AB", "delete_segment"),
        ("drop the triangle", "delete_polygon"),
        ("removing point A", "delete_point"),
        ("clearing the canvas", "clear_canvas"),
    ],
)
def test_inflected_destructive_verbs_find_destructive_tools(
    service: ToolSearchService, query: str, expected: str
) -> None:
    assert _names(service, query)[0] == expected


def test_drop_a_perpendicular_is_not_a_delete(service: ToolSearchService) -> None:
    names = _names(service, "drop a perpendicular from A to line BC")
    assert "construct_perpendicular_from_point" in names[:3]
    assert not any(name.startswith(DESTRUCTIVE_PREFIXES) for name in names[:5])


def test_confidence_is_best_raw_score_even_when_demoted(
    service: ToolSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boost_delete_circle(query_tokens: List[str], scores: Dict[str, float], raw_query: str = "") -> None:
        scores["delete_circle"] += 50.0

    monkeypatch.setattr(ToolSearchService, "_apply_intent_boosts", staticmethod(boost_delete_circle))
    names = _names(service, "show the circle")
    assert names[0] != "delete_circle"
    assert service._last_local_top_score >= 50.0


# Queries a local model sent while building the weighted graph of scenario GR-01, and
# other ways of asking for a graph of vertices and edges.
GRAPH_CREATION_QUERIES = [
    "draw points, segments, and labels on canvas to create a graph",
    "draw points and labeled segments on canvas to create a graph",
    "create points A B C D and connect them with segments as a graph",
    "create a weighted graph from points and segments",
    "create a graph",
    "build graph G1",
    "draw a graph with vertices A, B, C and edges A-B, B-C",
    "create nodes A, B, C and edges A-B, B-C",
    "build a network with nodes and edges",
]


@pytest.mark.parametrize("query", GRAPH_CREATION_QUERIES)
def test_graph_creation_queries_rank_generate_graph_first(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert names[0] == "generate_graph", names


@pytest.mark.parametrize("query", GRAPH_CREATION_QUERIES[:4])
def test_graph_creation_queries_also_offer_analyze_graph(service: ToolSearchService, query: str) -> None:
    assert "analyze_graph" in _names(service, query)[:3]


def test_graph_creation_query_skips_api_search_in_hybrid_mode(
    service: ToolSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_api_search(*args: object, **kwargs: object) -> List[object]:
        raise AssertionError("hybrid search fell back to the API")

    monkeypatch.setenv("TOOL_SEARCH_MODE", "hybrid")
    monkeypatch.setattr(service, "_search_tools_api", no_api_search)
    results = service.search_tools("draw points, segments, and labels on canvas to create a graph", max_results=10)
    names = [tool["function"]["name"] for tool in results]
    assert names[:2] == ["generate_graph", "analyze_graph"]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("draw a segment from A to B", "create_segment"),
        ("create a triangle", "create_polygon"),
        ("create a polygon with vertices A B C", "create_polygon"),
        ("create a point at (2, 3)", "create_point"),
        ("connect points A and B with a segment", "create_segment"),
        ("connect the points with segments", "create_segment"),
        ("draw edges of a triangle", "create_polygon"),
        ("draw the edges of the triangle", "create_polygon"),
        ("color the edges of the square", "create_colored_area"),
    ],
)
def test_geometry_queries_keep_geometry_tools_first(service: ToolSearchService, query: str, expected: str) -> None:
    names = _names(service, query)
    assert names[0] == expected, names
    assert "generate_graph" not in names[:3], names


@pytest.mark.parametrize(
    "query",
    [
        "draw a graph of sin(x)",
        "plot a graph of the function f(x) = x^3",
        "graph the absolute value function |x|",
        "make a graph of the function x^2 - 4",
        "graph y = x^2",
        "graph f(x) = x^2",
        "graph x squared",
        "make a graph of x squared",
    ],
)
def test_function_graph_queries_keep_draw_function_first(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert names[0] == "draw_function", names


# "graph" as the object of another tool, or a function plot that the local search does not
# recognise: none of these is a graph of vertices and edges.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("create a bar graph", "plot_bars"),
        ("create a parametric graph", "draw_parametric_function"),
        ("make a polar graph", "set_coordinate_system"),
        ("make a graph of the piecewise rule x if x > 0 else -x", "draw_piecewise_function"),
        ("generate a graph of the normal distribution", "plot_distribution"),
        ("create a graph of the circle", "create_circle"),
        ("create a graph of the ellipse", "create_ellipse"),
        ("make a graph of the regression line", "fit_regression"),
        ("create a graph of the derivative", "derive"),
        ("make a graph of the integral", "integrate"),
        ("make a coordinate graph", "set_coordinate_system"),
        ("make a graph of f", None),
        ("create a graph of 1/x", None),
        ("create a graph of 2x + 1", None),
        ("create a graph of abs(x)", None),
        ("create a graph of y equals x squared", None),
        ("create a graph of the inequality y > 2x", None),
        ("make a graph of x > 3", None),
        ("create a graph of the absolute value of x", None),
        ("create a line graph of the data", None),
        ("make a graph of the data", None),
    ],
)
def test_graph_of_something_else_does_not_rank_generate_graph_first(
    service: ToolSearchService, query: str, expected: str | None
) -> None:
    names = _names(service, query)
    assert names[0] != "generate_graph", names
    if expected is not None:
        assert names[0] == expected, names


# A shape, point or chart given by an equation or coordinates keeps its own tool; the
# function-notation boost for draw_function is only for functions.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("graph the circle x^2 + y^2 = 4", "create_circle"),
        ("graph the ellipse x^2/4 + y^2/9 = 1", "create_ellipse"),
        ("graph the point A = (2, 3)", "create_point"),
        ("graph the vector v = (3, 4)", "create_vector"),
        ("graph the triangle with A=(0,0), B=(4,0), C=(0,3)", "create_polygon"),
        ("graph the bars with values = [1, 2, 3]", "plot_bars"),
    ],
)
def test_graphing_a_shape_with_an_equation_keeps_its_tool(
    service: ToolSearchService, query: str, expected: str
) -> None:
    assert _names(service, query)[0] == expected


@pytest.mark.parametrize(
    "query",
    ["graph y = 2x + 1", "graph f(x) = x/2", "graph the line y = 2x + 1"],
)
def test_graphing_a_function_with_notation_ranks_draw_function_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "draw_function"


@pytest.mark.parametrize(
    "query",
    [
        "connect A and B",
        "connect points A and B",
        "connect the points with segments",
        "connect the midpoints of AB and CD",
        "draw a line connecting A and B",
    ],
)
def test_connecting_points_ranks_create_segment_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "create_segment"


# Without points to join, "connect"/"join" boosts nothing: as on main, the first four stay
# below the hybrid confidence threshold, so hybrid mode still asks the model.
@pytest.mark.parametrize(
    "query, expected, low_confidence",
    [
        ("join the two graphs", None, True),
        ("join the graphs G1 and G2", None, True),
        ("connect the two functions", None, True),
        ("connect the dots", None, True),
        ("join the endpoints to form a triangle", "create_polygon", False),
        ("draw a graph where A connects to B, C and D", None, False),
    ],
)
def test_connect_without_points_to_join_does_not_boost_create_segment(
    service: ToolSearchService, query: str, expected: str | None, low_confidence: bool
) -> None:
    names = _names(service, query)
    assert not names or names[0] != "create_segment", names
    if expected is not None:
        assert names[0] == expected, names
    if low_confidence:
        assert service._last_local_top_score < CONFIDENCE_THRESHOLD


@pytest.mark.parametrize(
    "query",
    [
        "create a graph with edges A->B, B->C",
        "create a graph with A-B, B-C, C-A",
        "create a bipartite graph",
        "create a complete graph on 5 nodes",
        "draw a star graph",
        "make a cycle graph with 6 vertices",
    ],
)
def test_written_edges_and_graph_families_rank_generate_graph_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "generate_graph"


def test_triangle_from_vertex_coordinates_keeps_generate_graph_low(service: ToolSearchService) -> None:
    names = _names(service, "Draw a triangle with vertices at (0,0), (4,0), (2,3).")
    assert names[0] == "create_polygon"
    assert "generate_graph" not in names[:3], names


@pytest.mark.parametrize(
    "query, expected",
    [
        ("connect P and Q with a vector", "create_vector"),
        ("join A and B with an arc", "create_circle_arc"),
        ("connect the centre to point A with a circle", "create_circle"),
    ],
)
def test_connect_with_a_named_drawable_keeps_that_tool(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


@pytest.mark.parametrize("query", ["I want to connect to the server", "connect A and B with a ray"])
def test_connect_without_two_points_to_join_falls_back_like_main(service: ToolSearchService, query: str) -> None:
    assert "create_segment" not in _names(service, query)
    assert service._last_local_top_score < CONFIDENCE_THRESHOLD


@pytest.mark.parametrize(
    "query, expected",
    [
        ("graph the function x -> 2x", "draw_function"),
        # x^2 is function notation, so draw_function leads (main had analyze_graph first)
        ("graph the map x -> x^2", "draw_function"),
        ("graph f: R -> R", "analyze_graph"),
    ],
)
def test_arrow_in_function_notation_is_not_a_graph_edge(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


@pytest.mark.parametrize("query", ["build a graph: A->B, B->C", "create graph with A->B"])
def test_arrow_between_vertex_names_is_a_graph_edge(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "generate_graph"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("graph y = sin(x) and mark the points where it crosses zero", "draw_function"),
        ("graph f(x) = x^2 and plot the point where x = 1", "draw_function"),
        ("graph the point A = (2, 3)", "create_point"),
    ],
)
def test_function_definition_with_points_mentioned_still_plots_the_function(
    service: ToolSearchService, query: str, expected: str
) -> None:
    assert _names(service, query)[0] == expected


# The solution set of an inequality comes from solve; piecewise rules and shading keep their tools.
@pytest.mark.parametrize(
    "query",
    [
        "Solve x^2 - 5x + 6 < 0.",
        "For which x is |x - 1| >= 2?",
        "Solve sin(x) > 1/2 for x in [0, 2pi].",
        "for which x is x >= 2 and x^2 < 9",
        "when is x > 3 - x",
        "solve the inequality 2x + 3 ≤ 7",
    ],
)
def test_inequalities_to_solve_rank_solve_first(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert names[0] == "solve", names
    assert "draw_piecewise_function" not in names, names


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Graph f(x)=1 for x<0 and -1 for x>=0.", "draw_piecewise_function"),
        ("Plot a piecewise function: x^2 for x<0 and 2x+1 for x>=0.", "draw_piecewise_function"),
        ("Shade the region where y > x^2", "create_region_colored_area"),
        ("Compute lim x->infinity (1+1/x)^x.", "limit"),
    ],
)
def test_drawing_an_inequality_or_an_arrow_is_not_solving_one(
    service: ToolSearchService, query: str, expected: str
) -> None:
    assert _names(service, query)[0] == expected


# A condition on x that only qualifies another operation, or a function defined by cases,
# is not an inequality to solve.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("define f(x) = x for x<0 and x^2 when x>=0", "draw_piecewise_function"),
        ("f(x) = x if x < 0, x^2 when x >= 0", "draw_piecewise_function"),
        ("when x < 0 the function is -x, when x >= 0 it is x^2", "draw_piecewise_function"),
        ("find the area under x^2 for 0 < x < 2", "create_colored_area"),
        ("find the integral of x^2 from 0 to 2 where x > 0", "integrate"),
        ("find the derivative of f(x) = x^2 when x > 0", "derive"),
        ("find the maximum of f(x)=x^2 when 0<x<3", "find_function_features"),
    ],
)
def test_conditions_that_qualify_another_operation_do_not_rank_solve_first(
    service: ToolSearchService, query: str, expected: str
) -> None:
    assert _names(service, query)[0] == expected
