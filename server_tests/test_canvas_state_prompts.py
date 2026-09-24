"""
How canvas state reaches the model through each provider.

Covers the <canvas> block in user messages (MATHUD_CANVAS_FORMAT=text/min_json),
the legacy prompt-JSON path (json), history cleanup, the system prompt, the
[canvas changes] note appended to the last tool result of a batch, and how
get_current_canvas_state results are rendered.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest.mock import patch

from server_tests.test_canvas_state_formatter import load_scene
from static.ai_model import AIModel
from static.canvas_state_formatter import render_text
from static.token_estimation import estimate_tokens_from_text
from static.openai_api_base import OpenAIAPIBase, build_developer_message
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.providers.anthropic_api import AnthropicAPI
from static.providers.local.local_agent_api import LocalAgentAPI

STATE: Dict[str, Any] = {
    "Points": [
        {"name": "A", "args": {"position": {"x": 0, "y": 0}}},
        {"name": "B", "args": {"position": {"x": 3, "y": 4.000000000000001}}},
    ],
    "Segments": [{"name": "AB", "args": {"p1": "A", "p2": "B"}, "_p1_coords": [0, 0], "_p2_coords": [3, 4]}],
    "Cartesian_System_Visibility": {"left_bound": -10, "right_bound": 10, "top_bound": 5, "bottom_bound": -5},
    "current_tick_spacing": 1,
    "coordinate_system": {"mode": "cartesian"},
}

CANVAS_BLOCK = "\n".join(
    [
        "<canvas>",
        "view x [-10, 10] y [-5, 5]; grid 1",
        "A = (0, 0)",
        "B = (3, 4)",
        "AB = Segment(A, B)  len 5",
        "</canvas>",
    ]
)

# The system prompt before canvas formats existed; json format keeps it (full tool mode).
LEGACY_DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. Use the provided tools for calculations rather than computing results yourself, so every result shown comes from the math engine. Canvas state is included with user messages; base your actions on it. For large scenes it may be summarized to reduce noise; when you need complete details, call get_current_canvas_state. Canvas state may be stale after tool calls, so re-check live state between actions when needed. Never use emoticons or emoji in your responses. When performing multiple steps, include a succinct summary of all actions taken in your final response. INFO: Point labels and coordinates are hardcoded to be shown next to all points on the canvas."""


def user_prompt(text: str = "How long is AB?", state: Dict[str, Any] = STATE, **extra: Any) -> str:
    """The prompt JSON the client sends with a user message (static/client/ai_interface.py)."""
    prompt: Dict[str, Any] = {
        "canvas_state": state,
        "user_message": text,
        "tool_call_results": None,
        "use_vision": False,
        "ai_model": "gpt-4.1-mini",
    }
    prompt.update(extra)
    return json.dumps(prompt)


class CanvasFormatEnv(unittest.TestCase):
    """Runs each test with a fixed canvas format and no budget override."""

    canvas_format = "text"

    def setUp(self) -> None:
        self._env = patch.dict(
            os.environ,
            {"MATHUD_CANVAS_FORMAT": self.canvas_format, "OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"},
        )
        self._env.start()
        os.environ.pop("MATHUD_CANVAS_BUDGET_TOKENS", None)
        os.environ.pop("AI_CANVAS_SUMMARY_MODE", None)
        self.addCleanup(self._env.stop)

    def chat_api(self) -> OpenAIChatCompletionsAPI:
        with patch("static.openai_api_base.OpenAI"):
            return OpenAIChatCompletionsAPI()

    def responses_api(self) -> OpenAIResponsesAPI:
        with patch("static.openai_api_base.OpenAI"):
            return OpenAIResponsesAPI()

    def anthropic_api(self) -> AnthropicAPI:
        return AnthropicAPI(model=AIModel.from_identifier("claude-haiku-4-5"))

    def local_api(self) -> LocalAgentAPI:
        return LocalAgentAPI(model=AIModel.from_identifier("local-model"))


class TestTextCanvasBlock(CanvasFormatEnv):
    def test_chat_completions_user_message_is_canvas_block_plus_text(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        self.assertEqual(api.messages[-1], {"role": "user", "content": f"{CANVAS_BLOCK}\n\nHow long is AB?"})

    def test_prompt_json_noise_is_not_sent(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        content = api.messages[-1]["content"]
        for noise in ("tool_call_results", "use_vision", "ai_model", "_p1_coords", "4.000000000000001"):
            self.assertNotIn(noise, content)

    def test_local_agent_now_sees_the_canvas(self) -> None:
        api = self.local_api()
        message = api._parse_and_prepare_message(user_prompt())
        self.assertEqual(message, {"role": "user", "content": f"{CANVAS_BLOCK}\n\nHow long is AB?"})

    def test_anthropic_and_responses_use_the_same_content(self) -> None:
        anthropic_message = self.anthropic_api()._parse_and_prepare_message(user_prompt())
        self.assertEqual(anthropic_message, {"role": "user", "content": f"{CANVAS_BLOCK}\n\nHow long is AB?"})
        api = self.responses_api()
        api._prepare_messages_for_stream(user_prompt())
        self.assertEqual(
            api._get_latest_user_message_for_input(),
            [{"role": "user", "content": f"{CANVAS_BLOCK}\n\nHow long is AB?"}],
        )

    def test_vision_request_keeps_the_canvas_block(self) -> None:
        api = self.chat_api()
        with patch.object(
            api, "_create_enhanced_prompt_with_image", wraps=api._create_enhanced_prompt_with_image
        ) as spy:
            content = api._prepare_message_content(
                user_prompt(attached_images=["data:image/png;base64,AAAA"], use_vision=False)
            )
        self.assertEqual(spy.call_args.kwargs["user_message"], f"{CANVAS_BLOCK}\n\nHow long is AB?")
        self.assertIsInstance(content, list)
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": f"{CANVAS_BLOCK}\n\nHow long is AB?"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_vision_snapshot_request_keeps_the_canvas_block(self) -> None:
        api = self.chat_api()
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch("static.openai_api_base._logger"),
            patch("builtins.print"),
        ):
            content = api._prepare_message_content(user_prompt(use_vision=True))
        # No snapshot on disk: falls back to the text content, still with the canvas.
        self.assertEqual(content, f"{CANVAS_BLOCK}\n\nHow long is AB?")

    def test_image_only_message_sends_just_the_canvas(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(user_prompt(text="", attached_images=["data:image/png;base64,AAAA"]))
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": CANVAS_BLOCK})

    def test_min_json_format(self) -> None:
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": "min_json"}):
            api = self.chat_api()
            content = api._prepare_message_content(user_prompt())
        assert isinstance(content, str)
        block, text = content.split("\n\n", 1)
        self.assertEqual(text, "How long is AB?")
        payload = json.loads(block.removeprefix("<canvas>\n").removesuffix("\n</canvas>"))
        self.assertEqual(payload["Points"][1], {"name": "B", "position": [3, 4]})

    def test_budget_env_trims_large_scenes(self) -> None:
        state = json.loads(json.dumps(STATE))
        state["Points"] += [{"name": f"P{i}", "args": {"position": {"x": i, "y": -i}}} for i in range(200)]
        with patch.dict(os.environ, {"MATHUD_CANVAS_BUDGET_TOKENS": "300"}):
            content = self.chat_api()._prepare_message_content(user_prompt(state=state))
        assert isinstance(content, str)
        self.assertIn("more points omitted; call get_current_canvas_state", content)
        with patch.dict(os.environ, {"MATHUD_CANVAS_BUDGET_TOKENS": "0"}):
            content = self.chat_api()._prepare_message_content(user_prompt(state=state))
        self.assertIn("P199 = (199, -199)", content)

    def test_local_default_budget_is_tighter_than_cloud(self) -> None:
        self.assertLess(
            self.local_api()._get_canvas_budget_tokens() or 0, self.chat_api()._get_canvas_budget_tokens() or 0
        )

    def test_unknown_format_falls_back_to_provider_default(self) -> None:
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": "yaml"}), patch("static.openai_api_base._logger"):
            self.assertEqual(self.chat_api()._get_canvas_format(), "text")
            self.assertEqual(self.local_api()._get_canvas_format(), "text")


class TestCanvasBlockHistory(CanvasFormatEnv):
    def _finish_turn(self, api: OpenAIChatCompletionsAPI, text: str = "done") -> None:
        api._finalize_stream(text, [])

    def test_latest_canvas_block_survives_cleanup(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        self._finish_turn(api)
        self.assertEqual(api.messages[1]["content"], f"{CANVAS_BLOCK}\n\nHow long is AB?")

    def test_older_canvas_blocks_are_stripped_by_marker(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt("first"))
        self._finish_turn(api)
        api._prepare_messages_for_request(user_prompt("second"))
        user_contents = [m["content"] for m in api.messages if m["role"] == "user"]
        self.assertEqual(user_contents, ["first", f"{CANVAS_BLOCK}\n\nsecond"])

    def test_stripping_handles_multimodal_text_parts(self) -> None:
        api = self.chat_api()
        api.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{CANVAS_BLOCK}\n\nold"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        )
        api._prepare_messages_for_request(user_prompt("new"))
        self.assertEqual(api.messages[1]["content"][0]["text"], "old")

    def test_user_text_that_mentions_canvas_is_kept(self) -> None:
        api = self.chat_api()
        api.messages.append({"role": "user", "content": "what is <canvas>?"})
        api._prepare_messages_for_request(user_prompt("new"))
        self.assertEqual(api.messages[1]["content"], "what is <canvas>?")


class TestSystemPrompt(CanvasFormatEnv):
    def test_text_format_describes_the_canvas_block(self) -> None:
        api = self.chat_api()
        prompt = api.messages[0]["content"]
        self.assertEqual(prompt, OpenAIAPIBase.DEV_MSG)
        self.assertIn("<canvas> block", prompt)
        self.assertIn("can be quoted directly", prompt)
        self.assertNotIn("summarized", prompt)

    def test_json_format_keeps_the_legacy_prompt(self) -> None:
        self.assertEqual(build_developer_message("json"), LEGACY_DEV_MSG)
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": "json"}):
            self.assertEqual(self.chat_api().messages[0]["content"], LEGACY_DEV_MSG)
            self.assertTrue(self.local_api().messages[0]["content"].startswith(LEGACY_DEV_MSG))
            self.assertTrue(self.anthropic_api()._build_system_prompt().startswith(LEGACY_DEV_MSG))


class TestJsonFormatKeepsTheCanvasPayload(CanvasFormatEnv):
    """MATHUD_CANVAS_FORMAT=json sends the same canvas payload as before.

    Not a byte-for-byte replay of old requests: the hybrid ``metrics`` block is no
    longer in the prompt, search tool mode adds SEARCH_MODE_MSG to the system prompt,
    and each tool call's result goes into its own tool message.
    """

    canvas_format = "json"

    def test_chat_completions_sends_the_prompt_json_then_strips_the_state(self) -> None:
        api = self.chat_api()
        prompt = user_prompt()
        api._prepare_messages_for_request(prompt)
        self.assertEqual(api.messages[-1], {"role": "user", "content": prompt})
        api._finalize_stream("done", [])
        expected = json.loads(prompt)
        del expected["canvas_state"]
        self.assertEqual(api.messages[1]["content"], json.dumps(expected))

    def test_local_agent_sends_object_counts(self) -> None:
        message = self.local_api()._parse_and_prepare_message(user_prompt())
        self.assertEqual(message, {"role": "user", "content": "How long is AB?\n[Canvas: 2 Points, 1 Segments]"})

    def test_vision_request_drops_the_state(self) -> None:
        api = self.chat_api()
        content = api._prepare_message_content(user_prompt(attached_images=["data:image/png;base64,AAAA"]))
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "How long is AB?"})

    def test_tool_results_get_no_canvas_changes(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api._prepare_messages_for_request(results_prompt(STATE_AFTER))
        self.assertEqual(tool_contents(api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})


TWO_CALLS: List[Dict[str, Any]] = [
    {"id": "call_a", "function": {"name": "create_point", "arguments": '{"x": 1, "y": 1, "name": "C"}'}},
    {"id": "call_b", "function": {"name": "update_point", "arguments": '{"point_name": "B", "new_y": 0}'}},
]
RESULT_A = {"create_point(x:1, y:1, name:C)": "Call successful!"}
RESULT_B = {"update_point(point_name:B, new_y:0)": "Call successful!"}

STATE_AFTER: Dict[str, Any] = json.loads(json.dumps(STATE))
STATE_AFTER["Points"][1]["args"]["position"] = {"x": 3, "y": 0}
STATE_AFTER["Points"].append({"name": "C", "args": {"position": {"x": 1, "y": 1}}})

CHANGES = "\n".join(
    [
        "[canvas changes]",
        "+ C = (1, 1)",
        "~ B = (3, 4)  ->  (3, 0)",
        "~ AB = Segment(A, B)  len 5  ->  Segment(A, B)  len 3",
    ]
)


def results_prompt(state: Dict[str, Any], order: Sequence[str] = ("call_b", "call_a")) -> str:
    """The prompt the client sends after running a tool batch (per-call results plus state_after)."""
    entries = {
        "call_a": {"tool_call_id": "call_a", "result": RESULT_A},
        "call_b": {"tool_call_id": "call_b", "result": RESULT_B},
    }
    return json.dumps(
        {
            "canvas_state": state,
            "user_message": None,
            "tool_call_results": json.dumps([entries[key] for key in order]),
            "use_vision": False,
            "ai_model": "gpt-4.1-mini",
        }
    )


def tool_contents(api: OpenAIAPIBase) -> Dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in api.messages if m.get("role") == "tool"}


class TestCanvasChangesAfterToolCalls(CanvasFormatEnv):
    """After a tool batch the last tool message says what changed on the canvas."""

    EXPECTED = {"call_a": json.dumps(RESULT_A), "call_b": f"{json.dumps(RESULT_B)}\n{CHANGES}"}

    def test_chat_completions(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api._prepare_messages_for_request(results_prompt(STATE_AFTER))
        # Results are matched by id even though they arrive in reverse order.
        self.assertEqual(tool_contents(api), self.EXPECTED)

    def test_responses_api(self) -> None:
        api = self.responses_api()
        api._prepare_messages_for_stream(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api._prepare_messages_for_stream(results_prompt(STATE_AFTER))
        self.assertEqual(tool_contents(api), self.EXPECTED)
        converted = api._convert_messages_to_input()
        self.assertTrue(converted[-1]["content"].endswith(CHANGES))
        # The user message keeps its canvas block for the continuation request.
        self.assertEqual(converted[1]["content"], f"{CANVAS_BLOCK}\n\nHow long is AB?")

    def test_anthropic_keeps_changes_inside_the_tool_result_block(self) -> None:
        api = self.anthropic_api()
        api.messages.append(api._parse_and_prepare_message(user_prompt()) or {})
        api._finalize_anthropic_stream("", TWO_CALLS)
        self.assertIsNone(api._parse_and_prepare_message(results_prompt(STATE_AFTER)))
        self.assertEqual(tool_contents(api), self.EXPECTED)
        blocks = api._convert_messages_to_anthropic()[-1]["content"]
        self.assertEqual([b["type"] for b in blocks], ["tool_result", "tool_result"])
        self.assertEqual({b["tool_use_id"]: b["content"] for b in blocks}, self.EXPECTED)

    def test_local_agent(self) -> None:
        api = self.local_api()
        api.messages.append(api._parse_and_prepare_message(user_prompt()) or {})
        api._finalize_stream("", TWO_CALLS)
        self.assertIsNone(api._parse_and_prepare_message(results_prompt(STATE_AFTER)))
        self.assertEqual(tool_contents(api), self.EXPECTED)

    def test_unchanged_canvas_adds_nothing(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api._prepare_messages_for_request(results_prompt(STATE))
        self.assertEqual(tool_contents(api), {"call_a": json.dumps(RESULT_A), "call_b": json.dumps(RESULT_B)})

    def test_second_batch_reports_changes_since_the_first(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api._prepare_messages_for_request(results_prompt(STATE_AFTER))
        api._finalize_stream("", [{"id": "call_c", "function": {"name": "delete_point", "arguments": "{}"}}])
        final_state = json.loads(json.dumps(STATE_AFTER))
        final_state["Points"].pop()
        entries = [{"tool_call_id": "call_c", "result": {"delete_point(point_name:C)": "Call successful!"}}]
        prompt = json.loads(results_prompt(final_state))
        prompt["tool_call_results"] = json.dumps(entries)
        api._prepare_messages_for_request(json.dumps(prompt))
        self.assertTrue(tool_contents(api)["call_c"].endswith("\n[canvas changes]\n- C (point) removed"))

    def test_changes_follow_a_server_side_answer(self) -> None:
        """A call answered by the server (e.g. a dropped tool) still gets the changes appended last."""
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", TWO_CALLS)
        api.record_tool_call_result("call_b", "Error: tool 'update_point' is not loaded")
        api._prepare_messages_for_request(results_prompt(STATE_AFTER, order=("call_a",)))
        contents = tool_contents(api)
        self.assertEqual(contents["call_a"], json.dumps(RESULT_A))
        self.assertTrue(contents["call_b"].startswith("Error: tool 'update_point' is not loaded\n[canvas changes]"))

    def test_reset_conversation_forgets_the_shown_state(self) -> None:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api.reset_conversation()
        api.messages.append({"role": "assistant", "content": "", "tool_calls": TWO_CALLS})
        api._append_tool_messages([SimpleNamespace(id="call_a"), SimpleNamespace(id="call_b")])
        api._prepare_messages_for_request(results_prompt(STATE_AFTER))
        self.assertTrue(tool_contents(api)["call_b"].endswith("[canvas now]\n" + CANVAS_AFTER))


CANVAS_AFTER = "\n".join(
    [
        "view x [-10, 10] y [-5, 5]; grid 1",
        "A = (0, 0)",
        "B = (3, 0)",
        "C = (1, 1)",
        "AB = Segment(A, B)  len 3",
    ]
)

STATE_KEY = "get_current_canvas_state(drawable_types:None, object_names:None, include_computations:None)"
GET_STATE_CALLS: List[Dict[str, Any]] = [
    {"id": "call_s", "function": {"name": "get_current_canvas_state", "arguments": "{}"}},
]


def get_state_results_prompt(result: Any, state: Dict[str, Any] = STATE) -> str:
    entries = [{"tool_call_id": "call_s", "result": result}]
    return json.dumps({"canvas_state": state, "user_message": None, "tool_call_results": json.dumps(entries)})


class TestCanvasStateToolResult(CanvasFormatEnv):
    """get_current_canvas_state results are rendered like the user-message canvas."""

    def _api_waiting_for_state(self) -> OpenAIChatCompletionsAPI:
        api = self.chat_api()
        api._prepare_messages_for_request(user_prompt())
        api._finalize_stream("", GET_STATE_CALLS)
        return api

    def test_full_state_is_rendered_as_text(self) -> None:
        api = self._api_waiting_for_state()
        scene = load_scene("mixed_medium")
        result = {STATE_KEY: {"type": "canvas_state", "value": scene}}
        api._prepare_messages_for_request(get_state_results_prompt(result))
        self.assertEqual(tool_contents(api)["call_s"], render_text(scene))

    def _state_result_content(self, points: int, budget: str, fmt: str = "text") -> str:
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": fmt}):
            api = self._api_waiting_for_state()
            state = json.loads(json.dumps(STATE))
            state["Points"] += [{"name": f"P{i}", "args": {"position": {"x": i, "y": i}}} for i in range(points)]
            result = {STATE_KEY: {"type": "canvas_state", "value": state}}
            with patch.dict(os.environ, {"MATHUD_CANVAS_BUDGET_TOKENS": budget}):
                api._prepare_messages_for_request(get_state_results_prompt(result))
        return tool_contents(api)["call_s"]

    def test_tool_result_gets_twice_the_canvas_budget(self) -> None:
        # Over the canvas budget but within twice it: nothing is dropped.
        content = self._state_result_content(points=30, budget="200")
        self.assertGreater(estimate_tokens_from_text(content), 200)
        self.assertIn("P29 = (29, 29)", content)
        self.assertNotIn("omitted", content)

    def test_huge_tool_result_is_truncated_with_a_note(self) -> None:
        content = self._state_result_content(points=300, budget="200")
        self.assertLessEqual(estimate_tokens_from_text(content), 400)
        self.assertIn("more points omitted; call get_current_canvas_state with object_names", content)
        self.assertIn("P299", self._state_result_content(points=300, budget="0"))

    def test_huge_min_json_tool_result_is_truncated_with_a_note(self) -> None:
        content = self._state_result_content(points=300, budget="200", fmt="min_json")
        self.assertLessEqual(estimate_tokens_from_text(content), 400)
        payload = json.loads(content)
        self.assertGreater(payload["omitted"]["Points"], 0)
        self.assertIn("object_names", payload["note"])

    def test_filtered_state_renders_only_the_requested_objects(self) -> None:
        api = self._api_waiting_for_state()
        filtered = {
            "Points": [STATE["Points"][0]],
            **{k: v for k, v in STATE.items() if k not in ("Points", "Segments")},
        }
        key = "get_current_canvas_state(drawable_types:None, object_names:['A'], include_computations:False)"
        api._prepare_messages_for_request(get_state_results_prompt({key: {"type": "canvas_state", "value": filtered}}))
        self.assertEqual(tool_contents(api)["call_s"], "view x [-10, 10] y [-5, 5]; grid 1\nA = (0, 0)")

    def test_other_results_stay_json(self) -> None:
        api = self._api_waiting_for_state()
        api._prepare_messages_for_request(get_state_results_prompt({"2+3": 5}))
        self.assertEqual(tool_contents(api)["call_s"], json.dumps({"2+3": 5}))

    def test_legacy_combined_results_render_the_state_inside_json(self) -> None:
        api = self._api_waiting_for_state()
        legacy = {"2+3": 5, STATE_KEY: {"type": "canvas_state", "value": STATE}}
        prompt = json.dumps({"canvas_state": STATE, "user_message": None, "tool_call_results": json.dumps(legacy)})
        api._prepare_messages_for_request(prompt)
        content = json.loads(tool_contents(api)["call_s"])
        self.assertEqual(content, {"2+3": 5, STATE_KEY: CANVAS_BLOCK.split("\n", 1)[1].rsplit("\n", 1)[0]})

    def test_json_format_sends_the_raw_state(self) -> None:
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": "json"}):
            api = self._api_waiting_for_state()
            result = {STATE_KEY: {"type": "canvas_state", "value": STATE}}
            api._prepare_messages_for_request(get_state_results_prompt(result))
        self.assertEqual(tool_contents(api)["call_s"], json.dumps(result))

    def test_local_agent_renders_the_state(self) -> None:
        api = self.local_api()
        api.messages.append(api._parse_and_prepare_message(user_prompt()) or {})
        api._finalize_stream("", GET_STATE_CALLS)
        result = {STATE_KEY: {"type": "canvas_state", "value": STATE}}
        self.assertIsNone(api._parse_and_prepare_message(get_state_results_prompt(result)))
        self.assertEqual(tool_contents(api)["call_s"], render_text(STATE))


if __name__ == "__main__":
    unittest.main()
