"""Every strict tool schema must satisfy OpenAI strict mode.

Strict mode requires each object schema to list all of its properties in
``required`` (optional parameters are nullable instead) and to set
``additionalProperties`` to false, at every nesting level.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Iterator, List, Tuple

from static.functions_definitions import FUNCTIONS


def _strict_tools() -> List[Dict[str, Any]]:
    tools = []
    for entry in FUNCTIONS:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and fn.get("strict"):
            tools.append(fn)
    return tools


def _object_schemas(schema: Any, path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (path, schema) for every object schema with properties, nested ones included."""
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("properties"), dict):
        yield path, schema
        for name, child in schema["properties"].items():
            yield from _object_schemas(child, f"{path}.{name}")
    if isinstance(schema.get("items"), dict):
        yield from _object_schemas(schema["items"], f"{path}[]")
    for key in ("anyOf", "oneOf", "allOf"):
        for index, option in enumerate(schema.get(key) or []):
            yield from _object_schemas(option, f"{path}.{key}[{index}]")


class TestStrictToolSchemas(unittest.TestCase):
    def test_there_are_strict_tools(self) -> None:
        self.assertGreater(len(_strict_tools()), 10)

    def test_every_property_is_required_and_extras_are_forbidden(self) -> None:
        for fn in _strict_tools():
            for path, schema in _object_schemas(fn.get("parameters"), fn["name"]):
                with self.subTest(schema=path):
                    properties = set(schema["properties"])
                    self.assertEqual(set(schema.get("required") or []), properties)
                    self.assertIs(schema.get("additionalProperties"), False)

    def test_numeric_integrate_optional_parameters_are_nullable(self) -> None:
        fn = next(tool for tool in _strict_tools() if tool["name"] == "numeric_integrate")
        properties = fn["parameters"]["properties"]
        self.assertEqual(properties["method"]["type"], ["string", "null"])
        self.assertIn(None, properties["method"]["enum"])
        self.assertEqual(properties["steps"]["type"], ["integer", "null"])


if __name__ == "__main__":
    unittest.main()
