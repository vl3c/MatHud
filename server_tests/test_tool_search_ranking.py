"""
Ranking guards for the local tool search.

Destructive tools must not outrank everything else for harmless queries,
function-value questions must find evaluate_expression, and run_tests is only
offered when the query is about tests.
"""

from __future__ import annotations

from typing import List

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
