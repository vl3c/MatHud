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
from typing import Any, Dict, List
from unittest.mock import patch

import anthropic
import httpx2

from static.ai_model import AIModel
from static.providers.anthropic_api import AnthropicAPI

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
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            api = AnthropicAPI(model=AIModel.from_identifier(model_id), temperature=0.3, tools=[])
        api._anthropic_client = anthropic.Anthropic(
            api_key="test-key",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self._handler)),
            max_retries=0,
        )
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


if __name__ == "__main__":
    unittest.main()
