"""
Tests for the request bodies the Anthropic provider sends through the real SDK.

The anthropic client is kept real and only its HTTP transport is mocked, so a
keyword argument the installed SDK no longer accepts (such as ``temperature``,
removed from ``messages.create``/``messages.stream`` in anthropic 1.0) fails
here instead of being swallowed by the provider's error handling at runtime.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import anthropic
import httpx2

from static.ai_model import PROVIDER_ANTHROPIC, AIModel
from static.providers.anthropic_api import AnthropicAPI

# Model ID -> configured effort for the adaptive-thinking models; Haiku 4.5 has no effort support.
_REASONING_MODELS: Dict[str, str] = {
    "claude-fable-5-1": "low",
    "claude-opus-5-5": "medium",
    "claude-sonnet-5": "medium",
}


def _claude_model(model_id: str, reasoning_effort: Optional[str]) -> AIModel:
    """Build the model directly so the tests do not depend on the model registry."""
    return AIModel(
        model_id,
        has_vision=True,
        is_reasoning_model=reasoning_effort is not None,
        reasoning_effort=reasoning_effort,
        provider=PROVIDER_ANTHROPIC,
    )


_MESSAGE_RESPONSE: Dict[str, Any] = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _sse_body() -> str:
    message_start = dict(_MESSAGE_RESPONSE, content=[], stop_reason=None)
    events = [
        ("message_start", {"type": "message_start", "message": message_start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class TestAnthropicRequestBodies(unittest.TestCase):
    """AnthropicAPI must build requests the installed anthropic SDK accepts."""

    def setUp(self) -> None:
        self.request_bodies: List[Dict[str, Any]] = []

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.request_bodies.append(body)
        if body.get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=_sse_body())
        return httpx2.Response(200, json=_MESSAGE_RESPONSE)

    def _make_api(self, model_id: str) -> AnthropicAPI:
        return self._make_api_for(AIModel.from_identifier(model_id))

    def _make_api_for(self, model: AIModel) -> AnthropicAPI:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            api = AnthropicAPI(model=model, temperature=0.3, tools=[])
        http_client = httpx2.Client(transport=httpx2.MockTransport(self._handler))
        self.addCleanup(http_client.close)
        api._anthropic_client = anthropic.Anthropic(api_key="test-key", http_client=http_client, max_retries=0)
        return api

    def test_non_reasoning_model_sends_temperature(self) -> None:
        api = self._make_api("claude-haiku-4-5")
        api.create_chat_completion("Hi")
        self.assertEqual(len(self.request_bodies), 1)
        self.assertEqual(self.request_bodies[0]["temperature"], 0.3)

    def test_non_reasoning_model_stream_sends_temperature(self) -> None:
        api = self._make_api("claude-haiku-4-5")
        events = list(api.create_chat_completion_stream("Hi"))
        self.assertEqual(len(self.request_bodies), 1)
        self.assertEqual(self.request_bodies[0]["temperature"], 0.3)
        tokens = [event.get("text") for event in events if event.get("type") == "token"]
        self.assertIn("Hello", tokens)

    def test_reasoning_model_omits_temperature(self) -> None:
        api = self._make_api("claude-sonnet-5")
        api.create_chat_completion("Hi")
        self.assertEqual(len(self.request_bodies), 1)
        self.assertNotIn("temperature", self.request_bodies[0])

    def test_reasoning_model_stream_omits_temperature(self) -> None:
        api = self._make_api("claude-sonnet-5")
        list(api.create_chat_completion_stream("Hi"))
        self.assertEqual(len(self.request_bodies), 1)
        self.assertNotIn("temperature", self.request_bodies[0])

    def test_reasoning_models_send_configured_effort(self) -> None:
        for model_id, effort in _REASONING_MODELS.items():
            with self.subTest(model=model_id):
                self.request_bodies.clear()
                api = self._make_api_for(_claude_model(model_id, effort))
                api.create_chat_completion("Hi")
                list(api.create_chat_completion_stream("Hi"))
                self.assertEqual(len(self.request_bodies), 2)
                for body in self.request_bodies:
                    self.assertEqual(body["output_config"], {"effort": effort})
                    self.assertNotIn("temperature", body)

    def test_haiku_sends_temperature_without_effort(self) -> None:
        api = self._make_api_for(_claude_model("claude-haiku-4-5", None))
        api.create_chat_completion("Hi")
        list(api.create_chat_completion_stream("Hi"))
        self.assertEqual(len(self.request_bodies), 2)
        for body in self.request_bodies:
            self.assertNotIn("output_config", body)
            self.assertEqual(body["temperature"], 0.3)

    def test_unknown_effort_is_not_sent(self) -> None:
        api = self._make_api_for(_claude_model("claude-opus-5-5", "extreme"))
        with self.assertLogs("mathud", level="WARNING"):
            api.create_chat_completion("Hi")
        self.assertNotIn("output_config", self.request_bodies[0])

    def test_streaming_max_tokens_leaves_room_for_thinking(self) -> None:
        api = self._make_api_for(_claude_model("claude-opus-5-5", "medium"))
        list(api.create_chat_completion_stream("Hi"))
        api.create_chat_completion("Hi")
        self.assertEqual(self.request_bodies[0]["max_tokens"], 32000)
        self.assertEqual(self.request_bodies[1]["max_tokens"], 16000)

    def test_requests_send_no_thinking_config_or_tool_choice(self) -> None:
        # Thinking runs adaptive by default; Fable 5.1 and Opus 5.5 reject forced tool_choice.
        api = self._make_api_for(_claude_model("claude-fable-5-1", "low"))
        api.create_chat_completion("Hi")
        list(api.create_chat_completion_stream("Hi"))
        for body in self.request_bodies:
            self.assertNotIn("thinking", body)
            self.assertNotIn("tool_choice", body)


if __name__ == "__main__":
    unittest.main()
