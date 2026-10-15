"""Every AI tool is defined, registered in the browser, searchable and of exactly one kind.

``static/client/function_registry.py`` imports browser modules, so its tool names are read
with ``ast``. The kinds live in the pure module ``static/client/tool_kinds.py``.
"""

from __future__ import annotations

import ast
import os
import unittest
from collections import Counter
from typing import Set

from static.functions_definitions import FUNCTIONS
from static.tool_search_service import TOOL_CATEGORIES
from tool_kinds import META_TOOLS, UNDOABLE_TOOLS, VALUE_TOOLS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "static", "client", "function_registry.py")

# The search tool is how the model finds the others, so it belongs to no search category.
UNCATEGORIZED_TOOLS = {"search_tools"}


def _defined_tools() -> Set[str]:
    return {entry["function"]["name"] for entry in FUNCTIONS}


def _registered_tools() -> Set[str]:
    """Keys of the ``functions`` dict in ``get_available_functions``, plus ``functions["x"] = ...`` additions."""
    tree = ast.parse(open(REGISTRY_PATH, encoding="utf-8").read())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "get_available_functions"
    )
    names: Set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "functions":
            assert isinstance(node.value, ast.Dict)
            names.update(key.value for key in node.value.keys if isinstance(key, ast.Constant))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "functions"
                    and isinstance(target.slice, ast.Constant)
                ):
                    names.add(target.slice.value)
    return names


class TestToolWiring(unittest.TestCase):
    def test_tool_names_are_unique(self) -> None:
        counts = Counter(entry["function"]["name"] for entry in FUNCTIONS)
        self.assertEqual([name for name, count in counts.items() if count > 1], [])

    def test_every_defined_tool_is_registered_in_the_browser(self) -> None:
        registered = _registered_tools()
        self.assertGreater(len(registered), 50)
        self.assertEqual(sorted(_defined_tools() - registered), [])
        self.assertEqual(sorted(registered - _defined_tools()), [])

    def test_every_tool_has_a_search_category(self) -> None:
        categorized = {tool for category in TOOL_CATEGORIES.values() for tool in category["tools"]}
        self.assertEqual(sorted(_defined_tools() - categorized - UNCATEGORIZED_TOOLS), [])
        self.assertEqual(sorted(categorized - _defined_tools()), [])

    def test_every_tool_is_of_exactly_one_kind(self) -> None:
        kinds = list(UNDOABLE_TOOLS) + list(VALUE_TOOLS) + list(META_TOOLS)
        self.assertEqual([name for name, count in Counter(kinds).items() if count > 1], [])
        self.assertEqual(sorted(_defined_tools() - set(kinds)), [])
        self.assertEqual(sorted(set(kinds) - _defined_tools()), [])


if __name__ == "__main__":
    unittest.main()
