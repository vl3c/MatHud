"""Null spelled as a string ("null", "None", "undefined") in tool arguments means null.

Local models fill unused optional arguments with the string "null" (the first live
scenario run on Qwen: ``"color": "null"``, ``"subtype": "null"``, ``"new_name": "null"``).
The routes turn those strings into None wherever the tool's schema allows null,
before the client validates and runs the calls; required and free-text arguments
keep the string.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

from static.tool_argument_validator import NULL_STRINGS, ToolArgumentValidator
from static.tool_call_processor import ToolCallProcessor


def normalize(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return ToolArgumentValidator.normalize_null_strings(tool, arguments)


class TestNormalizeNullStrings(unittest.TestCase):
    def test_every_spelling_becomes_none_for_nullable_arguments(self) -> None:
        for spelling in sorted(NULL_STRINGS):
            args = normalize("draw_function", {"function_string": "x^2", "color": spelling, "name": spelling})
            self.assertEqual(args, {"function_string": "x^2", "color": None, "name": None}, spelling)

    def test_the_live_run_arguments(self) -> None:
        polygon = normalize(
            "create_polygon",
            {"vertices": [{"x": 0, "y": 0}], "polygon_type": "triangle", "color": "blue", "name": "ABC",
             "subtype": "null"},
        )  # fmt: skip
        self.assertIsNone(polygon["subtype"])
        self.assertEqual(polygon["color"], "blue")
        point = normalize("update_point", {"point_name": "A", "new_name": "null", "new_x": 1, "new_y": 1,
                                           "new_color": "null"})  # fmt: skip
        self.assertEqual(point, {"point_name": "A", "new_name": None, "new_x": 1, "new_y": 1, "new_color": None})
        circumcircle = normalize("construct_circumcircle", {"triangle_name": "ABC", "p1_name": "null"})
        self.assertIsNone(circumcircle["p1_name"])
        self.assertEqual(circumcircle["triangle_name"], "ABC")

    def test_nullable_numbers_and_arrays_too(self) -> None:
        args = normalize("draw_function", {"function_string": "x", "left_bound": "null", "undefined_at": "None"})
        self.assertIsNone(args["left_bound"])
        self.assertIsNone(args["undefined_at"])

    def test_required_arguments_keep_the_string(self) -> None:
        # update_point's point_name is a plain string: a point named "null" is the model's problem.
        self.assertEqual(normalize("update_point", {"point_name": "null"})["point_name"], "null")
        self.assertEqual(normalize("draw_function", {"function_string": "null"})["function_string"], "null")

    def test_free_text_arguments_keep_the_string(self) -> None:
        segment = normalize("create_segment", {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "label_text": "null",
                                               "color": "null"})  # fmt: skip
        self.assertEqual(segment["label_text"], "null")
        self.assertIsNone(segment["color"])
        self.assertEqual(normalize("update_label", {"label_name": "L", "new_text": "None"})["new_text"], "None")
        area = normalize("create_region_colored_area", {"expression": "null", "circle_name": "null"})
        self.assertEqual(area["expression"], "null")
        self.assertIsNone(area["circle_name"])

    def test_only_the_exact_strings(self) -> None:
        for value in ("NULL", "nil", " null", "none", "nullable", ""):
            self.assertEqual(normalize("draw_function", {"function_string": "x", "color": value})["color"], value)

    def test_nested_optional_fields(self) -> None:
        args = normalize(
            "plot_distribution",
            {"name": "null", "representation": "continuous", "distribution_type": "normal",
             "distribution_params": {"mean": "null", "sigma": 1}, "shade_bounds": "null"},
        )  # fmt: skip
        self.assertIsNone(args["name"])
        self.assertEqual(args["distribution_params"], {"mean": None, "sigma": 1})
        self.assertIsNone(args["shade_bounds"])
        graph = normalize(
            "generate_graph",
            {"name": "G", "vertices": [{"name": "A", "color": "null", "label": "null"}], "edges": []},
        )
        self.assertEqual(graph["vertices"][0], {"name": "A", "color": None, "label": "null"})

    def test_input_is_not_modified_and_unknown_tools_pass_through(self) -> None:
        original = {"function_string": "x", "color": "null"}
        normalize("draw_function", original)
        self.assertEqual(original["color"], "null")
        self.assertEqual(normalize("no_such_tool", {"color": "null"}), {"color": "null"})

    def test_validation_accepts_the_normalized_call(self) -> None:
        args = {"vertices": [{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], "polygon_type": "triangle",
                "color": "null", "name": "ABC", "subtype": "null"}  # fmt: skip
        result = ToolArgumentValidator.validate("create_polygon", args)
        self.assertTrue(result["valid"], result["errors"])
        self.assertIsNone(result["arguments"]["subtype"])
        self.assertIsNone(result["arguments"]["color"])

    def test_a_failed_validation_still_returns_null_for_null_strings(self) -> None:
        result = ToolArgumentValidator.validate("create_point", {"x": "far", "y": 1, "color": "null", "name": None})
        self.assertFalse(result["valid"])
        self.assertIsNone(result["arguments"]["color"])
        self.assertEqual(result["arguments"]["x"], "far")

    def test_normalize_tool_calls_copies_each_call(self) -> None:
        calls: List[Dict[str, Any]] = [
            {"id": "c1", "function_name": "draw_function", "arguments": {"function_string": "x", "color": "null"}},
            {"id": "c2", "function_name": "search_tools", "arguments": {"query": "null"}},
            {"function_name": "broken", "arguments": "not a dict"},
        ]
        normalized = ToolArgumentValidator.normalize_tool_calls(calls)
        self.assertEqual(normalized[0], {"id": "c1", "function_name": "draw_function",
                                         "arguments": {"function_string": "x", "color": None}})  # fmt: skip
        self.assertEqual(normalized[1]["arguments"], {"query": "null"})
        self.assertEqual(normalized[2], calls[2])
        self.assertEqual(calls[0]["arguments"]["color"], "null")

    def test_non_streaming_completion_path(self) -> None:
        tool_call = MagicMock()
        tool_call.function.name = "update_point"
        tool_call.function.arguments = json.dumps(
            {"point_name": "A", "new_name": "null", "new_x": 1, "new_y": 1, "new_color": "undefined"}
        )
        processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        self.assertIsNone(processed["arguments"]["new_name"])
        self.assertIsNone(processed["arguments"]["new_color"])


class TestStreamingRouteNormalizesNullStrings(unittest.TestCase):
    """The streaming route hands the client the calls with null strings as null."""

    def setUp(self) -> None:
        from static.app_manager import AppManager

        self.original = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"
        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original is None:
            os.environ.pop("REQUIRE_AUTH", None)
        else:
            os.environ["REQUIRE_AUTH"] = self.original

    def test_stream_final_event(self) -> None:
        from static.openai_completions_api import OpenAIChatCompletionsAPI

        call = {"id": "c1", "function_name": "create_polygon",
                "arguments": {"vertices": [{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}],
                              "polygon_type": "triangle", "color": "null", "name": "ABC", "subtype": "null"}}  # fmt: skip

        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [call], "finish_reason": "tool_calls"}

        body = {"user_message": "triangle", "use_vision": False, "ai_model": "gpt-4.1"}
        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream", return_value=events()):
            response = self.client.post("/send_message_stream", json={"message": json.dumps(body)})
            lines = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
        final = next(line for line in lines if line.get("type") == "final")
        arguments = final["ai_tool_calls"][0]["arguments"]
        self.assertIsNone(arguments["subtype"])
        self.assertIsNone(arguments["color"])
        self.assertEqual(arguments["name"], "ABC")


if __name__ == "__main__":
    unittest.main()
