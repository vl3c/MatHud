from __future__ import annotations

import unittest
from typing import Any, Dict, List

from static.functions_definitions import FUNCTIONS


def _find_tool(name: str) -> Dict[str, Any]:
    for entry in FUNCTIONS:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "function":
            continue
        fn = entry.get("function")
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == name:
            return entry
    raise AssertionError(f"Tool not found: {name}")


def _require_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be dict")
    return value


def _require_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise AssertionError(f"{label} must be list")
    return value


class TestCanvasStateToolSchema(unittest.TestCase):
    def test_get_current_canvas_state_schema(self) -> None:
        tool = _find_tool("get_current_canvas_state")
        fn = _require_dict(tool.get("function"), "function")
        self.assertTrue(fn.get("strict"))

        params = _require_dict(fn.get("parameters"), "get_current_canvas_state.parameters")
        self.assertEqual(params.get("type"), "object")
        self.assertEqual(params.get("additionalProperties"), False)

        props = _require_dict(params.get("properties"), "get_current_canvas_state.properties")
        required = _require_list(params.get("required"), "get_current_canvas_state.required")

        self.assertEqual(props, {})
        self.assertEqual(required, [])


