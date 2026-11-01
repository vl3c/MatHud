"""
Ranking guards for the local tool search.

Destructive tools must not outrank everything else for harmless queries,
function-value questions must find evaluate_expression, and run_tests is only
offered when the query is about tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
        "when is 2x+1 > 7",
        "find the zeros of sin(x) for 0 < x < 2pi",
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


# A probability's condition and a condition on points are no inequalities to solve either
@pytest.mark.parametrize(
    "query, expected",
    [
        ("find P(X > 2) for a normal with mean 0 and sigma 1", "plot_distribution"),
        ("find p(x < 3) for the normal distribution with mean 2", "plot_distribution"),
        ("find the distance between points A and B where A.x < B.x", None),
    ],
)
def test_probabilities_and_distances_do_not_rank_solve_first(
    service: ToolSearchService, query: str, expected: Optional[str]
) -> None:
    names = _names(service, query)
    assert names[0] != "solve", names
    if expected is not None:
        assert expected in names[:3], names


# Statistics, geometric measurements and objects, a relation between two named objects (AB >
# BC) and a value asked under a condition keep their own tools
@pytest.mark.parametrize(
    "query, expected",
    [
        ("find the standard deviation for values > 5", "compute_descriptive_statistics"),
        ("find the median of data where x > 3", "compute_descriptive_statistics"),
        ("find the angle ABC where AB > BC", "create_angle"),
        ("find the length of segment AB when AB > 3", "create_segment"),
        ("find the perimeter of triangle ABC if AB < 5", "create_polygon"),
        ("which points have x > 3", None),
        ("when x < 0 what is |x|", None),
    ],
)
def test_measurements_and_named_relations_do_not_rank_solve_first(
    service: ToolSearchService, query: str, expected: Optional[str]
) -> None:
    names = _names(service, query)
    assert names[0] != "solve", names
    if expected is not None:
        assert expected in names[:3], names


# A derivation to check ranks check_steps first; a step-by-step request for another operation,
# a simplification and a construction keep their own tools
@pytest.mark.parametrize(
    "query",
    [
        "Check my steps: (x+1)^2 - 1, x^2 + 2x, x(x+2).",
        "Which step of this derivation is wrong?",
        "Is x^2 - 1 equivalent to (x - 1)(x + 1)?",
        "find the mistake in my algebra: 2(x+3) = 2x+3",
        "is my simplification of (x^2-4)/(x-2) to x+2 correct?",
    ],
)
def test_derivations_to_check_rank_check_steps_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "check_steps"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Check whether AB and CD are perpendicular", "inspect_relation"),
        ("simplify (x^2-4)/(x-2)", "simplify"),
        ("Solve x^2 - 5x + 6 = 0 step by step.", "solve"),
        ("Follow these steps to construct the perpendicular bisector of AB.", "construct_perpendicular_bisector"),
    ],
)
def test_other_operations_do_not_rank_check_steps_first(service: ToolSearchService, query: str, expected: str) -> None:
    names = _names(service, query)
    assert names[0] == expected, names
    assert "check_steps" not in names[:3], names


# A request that names another operation keeps its tool, though it says step, check or working
@pytest.mark.parametrize(
    "query, expected",
    [
        ("follow these steps: plot sin(x), then plot cos(x)", "draw_function"),
        ("follow the steps below and plot the function x^3", "draw_function"),
        ("the plot is not working correctly, please check", "draw_function"),
        ("plot the function step by step and check", "draw_function"),
        ("evaluate sin(pi/4) step by step and check", "evaluate_expression"),
        ("compute the mean step by step and verify", "compute_descriptive_statistics"),
        ("evaluate the integral of x^2 from 0 to 1, check each step", "evaluate_expression"),
    ],
)
def test_named_operations_keep_their_tools(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


def test_a_derivative_is_not_a_derivation(service: ToolSearchService) -> None:
    assert _names(service, "What is the derivative of x^3 + 2x - 1?")[0] == "derive"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("What is the equivalent of 5 miles in kilometers?", "convert"),
        ("equivalent fractions of 1/2", "simplify"),
    ],
)
def test_equivalent_units_and_fractions_are_not_derivations(
    service: ToolSearchService, query: str, expected: str
) -> None:
    names = _names(service, query)
    assert names[0] == expected, names
    assert "check_steps" not in names[:3], names


# Taylor and Maclaurin polynomials rank taylor_polynomial first, also when "expand" or a
# plot is asked alongside; a series to test for convergence does not
@pytest.mark.parametrize(
    "query",
    [
        "Taylor polynomial of e^x at 0 to order 4.",
        "Maclaurin series of sin(x).",
        "What is the power series of 1/(1-x) up to x^5?",
        "Taylor expand ln(1+x) at 0",
        "expand e^x in a power series",
        "series expansion of tan(x) around 0",
        "Plot e^x and its Taylor polynomial of order 3",
        "write cos(x) as a series around 0",
        "quadratic approximation of sqrt(x) near 4",
        "approximate sin(x) with a polynomial of degree 5",
        "linear approximation of sqrt(x) near 4",
        "find the linearization of f(x)=sqrt(x) at a=4",
    ],
)
def test_taylor_requests_rank_taylor_polynomial_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "taylor_polynomial"


def test_a_convergence_question_is_not_a_taylor_polynomial(service: ToolSearchService) -> None:
    assert "taylor_polynomial" not in _names(service, "Does the series of 1/n^2 converge?")[:3]


# Closed forms of sums come from simplify(summation(...)); a sum of numbers and a normal
# distribution's sigma do not
@pytest.mark.parametrize(
    "query",
    [
        "Find a closed form for the sum of k^2 from 1 to n.",
        "What is the sum of 1/k^2 from 1 to infinity?",
        "Sum of the geometric series 3*(1/2)^k for k from 0 to infinity.",
        "What is the formula for the sum of the first n odd numbers?",
        "Does the series of 1/n^2 converge?",
        "Is the infinite series of (-1)^k/k convergent?",
        "sum of 1/2^k for k from 0 to ∞",
    ],
)
def test_sums_rank_simplify_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "simplify"


@pytest.mark.parametrize(
    "query",
    [
        "Plot a normal distribution with mean 0 and sigma 1, shade from 0 to infinity.",
        "Sum of the values 3, 5, 9.",
        "Maclaurin series of sin(x).",
        "Compute the sum of k from 1 to 10",
        "Draw a series of circles with radius from 1 to n",
    ],
)
def test_other_sums_do_not_rank_simplify_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] != "simplify"


# Partial fractions rank expand first; an exact fraction stays with simplify
@pytest.mark.parametrize(
    "query",
    [
        "Decompose 2/(x^2-1) into partial fractions.",
        "Partial fraction decomposition of (3x+5)/((x+1)(x+2))",
        "decompose 1/(x(x+1)) into simpler fractions",
        "write 2/(x^2-1) as a sum of simpler fractions",
    ],
)
def test_partial_fractions_rank_expand_first(service: ToolSearchService, query: str) -> None:
    assert _names(service, query)[0] == "expand"


def test_splitting_a_number_fraction_is_not_partial_fractions(service: ToolSearchService) -> None:
    assert _names(service, "split the fraction 5/6 into a sum of unit fractions")[0] != "expand"


@pytest.mark.parametrize("query", ["Give 1/3 + 1/4 as an exact fraction.", "write 0.75 as a fraction"])
def test_exact_fractions_keep_simplify_first(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert names[0] == "simplify", names
    assert "expand" not in names[:3], names


def test_d_by_dx_is_a_derivative(service: ToolSearchService) -> None:
    assert _names(service, "Compute d/dx of ln(x^2+1).")[0] == "derive"


# "the normal to f at x = 2" is a normal line; a normal with distribution words is a distribution
@pytest.mark.parametrize(
    "query, expected",
    [
        ("draw the normal to f at x = 2", "draw_normal_line"),
        ("draw the normal to the parabola at x = 2", "draw_normal_line"),
        ("draw the normal to the circle at point A", "draw_normal_line"),
        ("draw the normal of f at x=3", "draw_normal_line"),
        ("show the normal to g at x = 0", "draw_normal_line"),
        ("draw the tangent and normal to f at x=2", "draw_normal_line"),
        ("plot a normal distribution with mean 0 and sigma 1", "plot_distribution"),
        ("draw a normal curve", "plot_distribution"),
        ("shade the normal from -1 to 1", "plot_distribution"),
        ("what is P(X<3) for a normal with mean 5", "compute_distribution"),
        ("make a binomial distribution with n=10 and p=0.5", "plot_distribution"),
        ("put a chi-square distribution with 3 df on the canvas", "plot_distribution"),
    ],
)
def test_normal_line_and_normal_distribution(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


# Hypothesis tests, confidence intervals and chi-squared tests: matched on the raw text, because the
# tokenizer drops "t" and "p" ("t-test", "p-value") and χ².
@pytest.mark.parametrize(
    "query, expected",
    [
        ("run a t-test on the data 5 6 7 8 against mean 5", "hypothesis_test"),
        ("two sample t test", "hypothesis_test"),
        ("paired t test before after", "hypothesis_test"),
        ("test the hypothesis that the mean is 50", "hypothesis_test"),
        ("1-PropZTest with x=45 n=100 p0=0.5", "hypothesis_test"),
        ("2-SampTTest on lists 1 2 3 and 4 5 6", "hypothesis_test"),
        ("is the coin fair? 60 heads in 100 flips, test at alpha 0.05", "hypothesis_test"),
        ("is the difference between the two group means significant at the 5% level?", "hypothesis_test"),
        ("compare two proportions with a test: 45 of 100 vs 30 of 100", "hypothesis_test"),
        ("95% confidence interval for the mean", "confidence_interval"),
        ("margin of error for a proportion", "confidence_interval"),
        ("ZInterval with sigma 15 mean 103 n 36", "confidence_interval"),
        ("with 95% confidence, estimate the mean weight from 12 13 14", "confidence_interval"),
        ("χ² test for the table [[10,20],[30,40]]", "chi_squared_test"),
        ("chi-square test of independence for this table", "chi_squared_test"),
        ("chi-square goodness of fit test with observed 20 30 50", "chi_squared_test"),
        ("find the p-value for t = 2.1 with 15 df, two-tailed", "compute_distribution"),
        ("p-value for z = -1.8, left-tailed", "compute_distribution"),
        ("critical value of t with 5 df", "compute_distribution"),
        ("what is P(X<3) for a normal with mean 5", "compute_distribution"),
        ("plot a chi-square distribution with 3 degrees of freedom", "plot_distribution"),
        ("run tests", "run_tests"),
        ("run the client tests", "run_tests"),
        ("draw the normal to f at x = 2", "draw_normal_line"),
        ("solve x^2 < 4 on the interval [0, 3]", "solve"),
        ("integrate x^2 over the interval [0, 2]", "integrate"),
    ],
)
def test_inference_requests(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


@pytest.mark.parametrize(
    "query",
    ["run a t-test on the data 5 6 7 8", "paired t test before after", "two sample t test"],
)
def test_a_t_test_is_not_the_test_suite(service: ToolSearchService, query: str) -> None:
    assert "run_tests" not in _names(service, query)


@pytest.mark.parametrize(
    "query",
    ["interval where f is increasing", "On which interval is f increasing?", "solve x^2 < 4 on the interval [0, 3]"],
)
def test_an_interval_of_x_is_not_a_confidence_interval(service: ToolSearchService, query: str) -> None:
    assert "confidence_interval" not in _names(service, query)[:3]


@pytest.mark.parametrize(
    "query",
    ["test if A, B, C are collinear", "round to 3 significant figures", "are these vectors linearly independent"],
)
def test_other_test_words_stay_away_from_inference(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert not names or names[0] not in ("hypothesis_test", "chi_squared_test", "confidence_interval"), names


# Review round 1: "test", "interval" and "significance" outside statistics, pictures and critical values.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("is the sequence convergent? test with the ratio test", "evaluate_expression"),
        ("test for symmetry of y = x^3", "evaluate_expression"),
        ("what is the significance of the second derivative", "derive"),
        ("plot a normal distribution and shade the p-value for z = 2.1", "plot_distribution"),
        ("plot a t distribution with 10 df and shade the p-value for t = 2.3", "plot_distribution"),
        ("draw the 95% confidence interval on a normal curve", "plot_distribution"),
        ("critical z value for 95% confidence", "compute_distribution"),
        ("z* for 90% confidence", "compute_distribution"),
        ("t* for 95% confidence with 14 df", "compute_distribution"),
        (
            "fit a linear regression to x=[1,2,3] y=[2,4,7] and give the confidence interval of the slope",
            "fit_regression",
        ),
        ("run a t-test on data 5 6 7 8 against 5 and give the critical value", "hypothesis_test"),
        ("test H0: mu = 50 against mu > 50 with data 51 52 53", "hypothesis_test"),
        ("test whether the mean is 50 using 48 52 49 51 47", "hypothesis_test"),
    ],
)
def test_inference_words_in_other_requests(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


@pytest.mark.parametrize(
    "query",
    [
        "plot points h0 and h1",
        "create point H0 at (1, 2)",
        "p values of these points",
        "use the ratio test on sum 1/n!",
    ],
)
def test_names_and_series_tests_are_not_hypothesis_tests(service: ToolSearchService, query: str) -> None:
    assert not {"hypothesis_test", "chi_squared_test"} & set(_names(service, query)[:3])


def test_outlier_test_keeps_descriptive_statistics(service: ToolSearchService) -> None:
    names = _names(service, "find the standard deviation and test for outliers in 2, 4, 6, 100")
    assert "compute_descriptive_statistics" in names[:2], names


# Review round 2: products of t or z, calculus critical values, "make"/"show" without a picture, textbook wording.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("compute z * conj(z) for z = 3 + 4i", "evaluate_expression"),
        ("evaluate 2*z*3 where z = 4", "evaluate_expression"),
        ("find t such that t*2 = 10", "solve"),
        ("find the critical values of f(x) = x^3 - 3x", "evaluate_expression"),
        ("make a 95% confidence interval for the mean of 4, 5, 6, 7, 8", "confidence_interval"),
        ("display the 95% confidence interval for mean 12.3, sd 2.1, n 50", "confidence_interval"),
        ("show the p-value for a two sample t test of [1,2,3] and [4,5,6]", "hypothesis_test"),
        ("test the claim that the mean weight is 500 g; sample mean 495, s = 10, n = 25", "hypothesis_test"),
        ("is there evidence that the mean exceeds 50? sample mean 52, sd 5, n 40, test at 5%", "hypothesis_test"),
        ("carry out a test at the 5% level that the proportion is 0.3; 45 successes in 120", "hypothesis_test"),
        ("test whether angle alpha is a right angle", "create_angle"),
        ("what z gives 95% confidence", "compute_distribution"),
    ],
)
def test_inference_review_round_two(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


def test_theorem_hypotheses_are_not_a_hypothesis_test(service: ToolSearchService) -> None:
    names = _names(service, "verify the hypotheses of the mean value theorem for f(x) = x^2 on [0, 2]")
    assert names[0] != "hypothesis_test", names


# Review round 3: p-values of a known statistic or a named distribution, regression significance, sample size.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("p-value for a chi-squared statistic of 7.8 with 3 degrees of freedom", "compute_distribution"),
        ("p-value when chi-square is 7.8 with 3 df", "compute_distribution"),
        ("the chi-square statistic is 11.2 with 4 df; find the p-value", "compute_distribution"),
        ("the test statistic is t = 2.3 with 12 df, find the p-value", "compute_distribution"),
        (
            "using the binomial distribution, find the p-value of getting 15 or more heads in 20 flips",
            "compute_distribution",
        ),
        ("find the p-value P(X >= 15) for a binomial with n=20 p=0.5", "compute_distribution"),
        ("is the slope of the regression line significantly different from 0", "fit_regression"),
        ("test whether the regression slope is zero at alpha 0.05", "fit_regression"),
        ("what is the p-value of the regression", "fit_regression"),
        ("find the critical t value for 95% confidence and 10 df", "compute_distribution"),
    ],
)
def test_inference_review_round_three(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


def test_sample_size_planning_is_not_an_interval(service: ToolSearchService) -> None:
    names = _names(service, "what sample size is needed for a margin of error of 3 with sigma 15 at 95% confidence")
    assert not {"confidence_interval", "hypothesis_test"} & set(names), names


# Review round 4: a sample size given as data, exact discrete tails asked as "significant", calculus in t.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("for a sample size of 30 with mean 98.2 and s 0.6, test H0: mu = 98.6 at alpha 0.05", "hypothesis_test"),
        ("a sample size of 25 has mean 52 and sd 6; construct a 95% confidence interval", "confidence_interval"),
        (
            "the probability of 9 or more successes out of 10 with p = 0.5; is it significant at 0.05?",
            "compute_distribution",
        ),
        ("binomial: is 9 out of 10 successes significant at the 5% level when p = 0.5", "compute_distribution"),
        (
            "for a binomial with n = 20 and p = 0.5, find P(X >= 15) and decide whether it is significant at the 5% level",
            "compute_distribution",
        ),
        (
            "assume a normal distribution; find the p-value for H0: mu = 50 with sample mean 52, sd 5, n 30",
            "hypothesis_test",
        ),
        ("find the critical values of s(t) = t^2 - 4t where t is 0 to 5", "evaluate_expression"),
        ("p-value for chi2(3) = 7.8", "compute_distribution"),
        ("make a binomial distribution with n=10 and p=0.5", "plot_distribution"),
    ],
)
def test_inference_review_round_four(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


def test_survey_size_planning_is_not_an_interval(service: ToolSearchService) -> None:
    names = _names(service, "how many people must be surveyed for a margin of error of 3%?")
    assert "confidence_interval" not in names, names


# Review round 5: "geometric", p(x) and "test" in algebra stay with their tools; a binomial p-value from a sample
# is an exact tail; "test whether p > 0.5 at alpha 0.05" crosses the decimal point.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("use the ratio test on the geometric series sum (1/2)^n", "simplify"),
        ("test whether the geometric series 1 + 2 + 4 + ... converges", "simplify"),
        ("use the rational root test on p(x) = x^3 - 6x^2 + 11x - 6", "evaluate_expression"),
        ("if alpha and beta are the roots of p(x) = x^2 + 3x + 2, find alpha^2 + beta^2", "solve"),
        ("geometric construction: bisect angle alpha", "construct_angle_bisector"),
        ("goodness of fit test: do the observed counts 30, 50, 20 follow a binomial distribution", "chi_squared_test"),
        ("in a sample of 30 with p = 0.5, find the binomial p-value of 21 or more successes", "compute_distribution"),
        (
            "a random sample of 15 is binomial with p = 0.4; find the p-value of 10 or more successes",
            "compute_distribution",
        ),
        (
            "the number of defects follows a poisson distribution; 15 defects observed; test whether the mean is 10",
            "compute_distribution",
        ),
        ("a survey of 500 voters finds 270 support the measure; test whether p > 0.5 at alpha 0.05", "hypothesis_test"),
        ("test whether mu > 3.5 at alpha 0.05 given x̄ = 3.8, s = 0.9, n = 40", "hypothesis_test"),
        ("test whether p differs from 0.4 at alpha = 0.01; 52 of 150 said yes", "hypothesis_test"),
        ("test mu = 2.5 at alpha 0.05 with mean 2.7, sd 0.4, n 20", "hypothesis_test"),
        (
            "using the normal approximation to the binomial, test whether p = 0.5 when 60 of 100 are heads",
            "hypothesis_test",
        ),
    ],
)
def test_inference_review_round_five(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected


# Review round 6: a named binomial test that mentions a proportion is an exact tail; p(x) and logic p, q are not p.
@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "a manufacturer claims the proportion of faulty bulbs is 0.05; 4 of 25 are faulty; "
            "use a binomial test at the 5% level",
            "compute_distribution",
        ),
        (
            "in a binomial experiment the success proportion is 0.4; find the p-value of 12 or more successes in 20",
            "compute_distribution",
        ),
        ("test whether p(x) = x^2 + 1 has real roots", "solve"),
        ("test if p(x) = x^2 - 4 has a root at x = 2", "evaluate_expression"),
        ("test whether p(t) = 3t^2 has a maximum", "find_function_features"),
        ("test whether p and q are logically equivalent", "check_steps"),
        ("a survey of 500 voters finds 270 support the measure; test whether p > 0.5 at alpha 0.05", "hypothesis_test"),
        ("test whether mu exceeds 3.5 at alpha 0.05 given mean 3.8, sd 0.9, n = 40", "hypothesis_test"),
    ],
)
def test_inference_review_round_six(service: ToolSearchService, query: str, expected: str) -> None:
    assert _names(service, query)[0] == expected
