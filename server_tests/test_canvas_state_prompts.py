"""
How canvas state reaches the model through each provider.

Covers the <canvas> block in user messages (MATHUD_CANVAS_FORMAT=text/min_json),
the legacy prompt-JSON path (json), history cleanup and the system prompt.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict
from unittest.mock import patch

from static.ai_model import AIModel
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

# The system prompt before canvas formats existed; json format must keep it byte for byte.
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


class TestJsonFormatIsUnchanged(CanvasFormatEnv):
    """MATHUD_CANVAS_FORMAT=json reproduces the original behaviour byte for byte."""

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


if __name__ == "__main__":
    unittest.main()
