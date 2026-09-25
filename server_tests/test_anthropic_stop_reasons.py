"""
Tests for how the Anthropic provider handles the reason a Claude reply stopped.

Like test_anthropic_api_requests.py, these keep the anthropic client real and
mock only its HTTP transport, so streamed events and responses are parsed by
the installed SDK exactly as in production. No request leaves the process.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import anthropic
import httpx2

from static.ai_model import PROVIDER_ANTHROPIC, AIModel
from static.providers.anthropic_api import AnthropicAPI

SseEvent = Tuple[str, Dict[str, Any]]

_REFUSAL_DETAILS: Dict[str, Any] = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "This request touches on a restricted topic.",
}


def _opus_model() -> AIModel:
    """Claude Opus 5.5, built directly so the test does not depend on the model registry."""
    return AIModel(
        "claude-opus-5-5",
        has_vision=True,
        is_reasoning_model=True,
        reasoning_effort="medium",
        provider=PROVIDER_ANTHROPIC,
    )


def _message(content: List[Dict[str, Any]], stop_reason: Optional[str], **extra: Any) -> Dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        **extra,
    }


def _text_block(index: int, text: str) -> List[SseEvent]:
    return [
        (
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _tool_block(index: int, tool_id: str, partial_json: str, closed: bool = True) -> List[SseEvent]:
    events: List[SseEvent] = [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": "create_point", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            },
        ),
    ]
    if closed:
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    return events


def _thinking_block(index: int) -> List[SseEvent]:
    return [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": "sig"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _sse(blocks: List[SseEvent], stop_reason: str, stop_details: Optional[Dict[str, Any]] = None) -> str:
    events: List[SseEvent] = [("message_start", {"type": "message_start", "message": _message([], None)})]
    events.extend(blocks)
    delta: Dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events.append(("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 1}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


_COMPLETE_ARGS = '{"x": 1, "y": 2}'
_TRUNCATED_ARGS = '{"x": 1, "y'


class _AnthropicTransportTest(unittest.TestCase):
    """Runs AnthropicAPI against queued fake HTTP responses."""

    def setUp(self) -> None:
        self.request_bodies: List[Dict[str, Any]] = []
        self.responses: List[httpx2.Response] = []

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.request_bodies.append(json.loads(request.content))
        return self.responses.pop(0)

    def _queue_stream(self, body: str) -> None:
        self.responses.append(httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body))

    def _queue_message(self, message: Dict[str, Any]) -> None:
        self.responses.append(httpx2.Response(200, json=message))

    def _make_api(self) -> AnthropicAPI:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            api = AnthropicAPI(model=_opus_model(), tools=[])
        http_client = httpx2.Client(transport=httpx2.MockTransport(self._handler))
        self.addCleanup(http_client.close)
        api._anthropic_client = anthropic.Anthropic(api_key="test-key", http_client=http_client, max_retries=0)
        return api

    @staticmethod
    def _final(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        finals = [event for event in events if event.get("type") == "final"]
        assert len(finals) == 1
        return finals[0]

    @staticmethod
    def _streamed_text(events: List[Dict[str, Any]]) -> str:
        return "".join(str(event.get("text", "")) for event in events if event.get("type") == "token")

    def _assistant_turns(self, api: AnthropicAPI) -> List[Dict[str, Any]]:
        return [dict(message) for message in api.messages if message.get("role") == "assistant"]


class TestStreamingMaxTokens(_AnthropicTransportTest):
    """A reply cut off at max_tokens must not run a tool call with incomplete arguments."""

    def test_cut_off_tool_call_is_dropped_and_reported(self) -> None:
        api = self._make_api()
        blocks = _text_block(0, "Placing the point.") + _tool_block(1, "toolu_cut", _TRUNCATED_ARGS)
        self._queue_stream(_sse(blocks, "max_tokens"))

        events = list(api.create_chat_completion_stream("Hi"))

        final = self._final(events)
        self.assertEqual(final["ai_tool_calls"], [])
        self.assertEqual(final["finish_reason"], "length")
        self.assertIn("cut off", self._streamed_text(events))
        self.assertIn("cut off", final["ai_message"])
        self.assertEqual(final["metrics"]["finish_reason"], "length")

    def test_unclosed_tool_block_is_dropped(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_tool_block(0, "toolu_cut", _TRUNCATED_ARGS, closed=False), "max_tokens"))

        final = self._final(list(api.create_chat_completion_stream("Hi")))

        self.assertEqual(final["ai_tool_calls"], [])
        self.assertEqual(final["finish_reason"], "length")

    def test_dropped_tool_call_is_left_out_of_history(self) -> None:
        api = self._make_api()
        blocks = _text_block(0, "Placing the point.") + _tool_block(1, "toolu_cut", _TRUNCATED_ARGS)
        self._queue_stream(_sse(blocks, "max_tokens"))

        list(api.create_chat_completion_stream("Hi"))

        self.assertEqual(self._assistant_turns(api), [{"role": "assistant", "content": "Placing the point."}])
        self.assertFalse(any(message.get("role") == "tool" for message in api.messages))

    def test_finished_tool_call_before_the_cut_off_still_runs(self) -> None:
        api = self._make_api()
        blocks = _tool_block(0, "toolu_done", _COMPLETE_ARGS) + _tool_block(1, "toolu_cut", _TRUNCATED_ARGS)
        self._queue_stream(_sse(blocks, "max_tokens"))

        events = list(api.create_chat_completion_stream("Hi"))

        final = self._final(events)
        self.assertEqual([call["id"] for call in final["ai_tool_calls"]], ["toolu_done"])
        self.assertEqual(final["ai_tool_calls"][0]["arguments"], {"x": 1, "y": 2})
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertIn("not run", self._streamed_text(events))
        tool_messages = [message for message in api.messages if message.get("role") == "tool"]
        self.assertEqual([message.get("tool_call_id") for message in tool_messages], ["toolu_done"])

    def test_normal_tool_use_is_unchanged(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_tool_block(0, "toolu_ok", _COMPLETE_ARGS), "tool_use"))

        events = list(api.create_chat_completion_stream("Hi"))

        final = self._final(events)
        self.assertEqual([call["id"] for call in final["ai_tool_calls"]], ["toolu_ok"])
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual(self._streamed_text(events), "")


class TestStreamingRefusal(_AnthropicTransportTest):
    """A refusal is surfaced to the user and neither runs tools nor enters history."""

    def test_refusal_surfaces_category_and_explanation(self) -> None:
        api = self._make_api()
        blocks = _text_block(0, "Sure, here is") + _tool_block(1, "toolu_x", _COMPLETE_ARGS)
        self._queue_stream(_sse(blocks, "refusal", _REFUSAL_DETAILS))

        events = list(api.create_chat_completion_stream("Hi"))

        final = self._final(events)
        self.assertEqual(final["finish_reason"], "refusal")
        self.assertEqual(final["ai_tool_calls"], [])
        streamed = self._streamed_text(events)
        self.assertIn("declined", streamed)
        self.assertIn("cyber", streamed)
        self.assertIn(_REFUSAL_DETAILS["explanation"], streamed)
        self.assertIn("declined", final["ai_message"])

    def test_refusal_without_details_still_explains(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse([], "refusal"))

        events = list(api.create_chat_completion_stream("Hi"))

        self.assertEqual(self._final(events)["finish_reason"], "refusal")
        self.assertIn("declined", self._streamed_text(events))

    def test_refused_reply_is_not_stored(self) -> None:
        api = self._make_api()
        blocks = _text_block(0, "Sure, here is") + _tool_block(1, "toolu_x", _COMPLETE_ARGS)
        self._queue_stream(_sse(blocks, "refusal", _REFUSAL_DETAILS))

        list(api.create_chat_completion_stream("Hi"))

        self.assertEqual(self._assistant_turns(api), [])
        self.assertFalse(any(message.get("role") == "tool" for message in api.messages))

    def test_refused_prompt_is_removed_from_history(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_text_block(0, "Sure"), "refusal", _REFUSAL_DETAILS))

        list(api.create_chat_completion_stream("Hi"))

        self.assertFalse(any(message.get("role") == "user" for message in api.messages))


class TestStreamingPauseTurn(_AnthropicTransportTest):
    def test_pause_turn_stops_with_a_note(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_text_block(0, "Working on it."), "pause_turn"))

        events = list(api.create_chat_completion_stream("Hi"))

        final = self._final(events)
        self.assertEqual(final["finish_reason"], "stop")
        self.assertIn("paused", self._streamed_text(events))


class TestEmptyAssistantTurns(_AnthropicTransportTest):
    """A reply with no text and no tool calls must never be stored or sent back."""

    def test_thinking_only_reply_is_not_stored(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_thinking_block(0), "end_turn"))

        list(api.create_chat_completion_stream("Hi"))

        self.assertEqual(self._assistant_turns(api), [])

    def test_next_request_has_no_empty_assistant_message(self) -> None:
        api = self._make_api()
        self._queue_stream(_sse(_thinking_block(0), "end_turn"))
        self._queue_stream(_sse(_text_block(0, "Done."), "end_turn"))

        list(api.create_chat_completion_stream("First"))
        list(api.create_chat_completion_stream("Second"))

        sent = self.request_bodies[1]["messages"]
        self.assertEqual([message["role"] for message in sent], ["user", "user"])

    def test_empty_assistant_message_in_history_is_skipped(self) -> None:
        api = self._make_api()
        api.messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Second"},
        ]

        converted = api._convert_messages_to_anthropic()

        self.assertEqual([message["role"] for message in converted], ["user", "user"])

    def test_non_streaming_empty_reply_is_not_stored(self) -> None:
        api = self._make_api()
        self._queue_message(_message([{"type": "thinking", "thinking": "", "signature": "sig"}], "end_turn"))

        api.create_chat_completion("Hi")

        self.assertEqual(self._assistant_turns(api), [])


class TestNonStreamingStopReasons(_AnthropicTransportTest):
    def test_cut_off_tool_call_is_dropped(self) -> None:
        api = self._make_api()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Placing the point."},
            {"type": "tool_use", "id": "toolu_cut", "name": "create_point", "input": {}},
        ]
        self._queue_message(_message(content, "max_tokens"))

        result = api.create_chat_completion("Hi")

        self.assertEqual(result.finish_reason, "length")
        self.assertFalse(result.message.tool_calls)
        self.assertIn("cut off", result.message.content)
        self.assertEqual(self._assistant_turns(api), [{"role": "assistant", "content": "Placing the point."}])

    def test_finished_tool_call_before_the_cut_off_still_runs(self) -> None:
        api = self._make_api()
        content: List[Dict[str, Any]] = [
            {"type": "tool_use", "id": "toolu_done", "name": "create_point", "input": {"x": 1}},
            {"type": "tool_use", "id": "toolu_cut", "name": "create_point", "input": {}},
        ]
        self._queue_message(_message(content, "max_tokens"))

        result = api.create_chat_completion("Hi")

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual([call.id for call in result.message.tool_calls], ["toolu_done"])

    def test_refusal_is_surfaced_and_not_stored(self) -> None:
        api = self._make_api()
        content: List[Dict[str, Any]] = [{"type": "text", "text": "Sure, here is"}]
        self._queue_message(_message(content, "refusal", stop_details=_REFUSAL_DETAILS))

        result = api.create_chat_completion("Hi")

        self.assertEqual(result.finish_reason, "refusal")
        self.assertFalse(result.message.tool_calls)
        self.assertIn("declined", result.message.content)
        self.assertIn("cyber", result.message.content)
        self.assertEqual(self._assistant_turns(api), [])
        self.assertFalse(any(message.get("role") == "user" for message in api.messages))


if __name__ == "__main__":
    unittest.main()
