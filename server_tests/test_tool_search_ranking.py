"""
Ranking guards for the local tool search.

Destructive tools must not outrank everything else for harmless queries,
function-value questions must find evaluate_expression, and run_tests is only
offered when the query is about tests.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from static.tool_search_service import ToolSearchService, clear_search_cache

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
    ],
)
def test_function_graph_queries_keep_draw_function_first(service: ToolSearchService, query: str) -> None:
    names = _names(service, query)
    assert names[0] == "draw_function", names
    assert "generate_graph" not in names[:3], names
