"""
Tests for per-request response metrics (static/response_metrics.py).

Unit tests drive ``ResponseMetricsTracker`` with a fake clock. Provider tests
stream mocked responses through each provider path and check the ``metrics``
record on the final event: Chat Completions and LocalAgent go through the real
openai SDK with a mocked HTTP transport (so llama-server's extra ``timings``
field is parsed exactly as in production), Anthropic through the real anthropic
SDK, and the Responses API through mocked stream events.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import anthropic
import httpx2
from openai import OpenAI

from static.ai_model import AIModel
from static.log_manager import LogManager
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.providers.anthropic_api import AnthropicAPI
from static.providers.local.local_agent_api import LocalAgentAPI
from static.response_metrics import (
    ResponseMetricsTracker,
    llama_timings_from,
    usage_from_anthropic,
    usage_from_chat_completions,
    usage_from_responses,
)


class FakeClock:
    """Monotonic clock advanced by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tracker(clock: FakeClock, streamed: bool = True) -> ResponseMetricsTracker:
    return ResponseMetricsTracker("local_agent", "qwen", "chat_completions", streamed, clock, lambda: 1700000000.0)


class TestResponseMetricsTracker(unittest.TestCase):
    def test_time_to_first_token_and_usage_rate(self) -> None:
        clock = FakeClock()
        tracker = _tracker(clock)
        clock.advance(0.5)
        tracker.mark_output("content")
        clock.advance(2.0)
        tracker.record_usage({"prompt_tokens": 120, "completion_tokens": 50, "cached_tokens": 100})
        metrics = tracker.finish("stop", 0)

        self.assertEqual(metrics["time_to_first_token_s"], 0.5)
        self.assertEqual(metrics["total_latency_s"], 2.5)
        self.assertEqual(metrics["output_tokens"], 50)
        self.assertFalse(metrics["output_tokens_estimated"])
        self.assertEqual(metrics["output_tokens_per_s"], 25.0)
        self.assertEqual(metrics["tokens_per_s_source"], "usage")
        self.assertEqual(metrics["total_tokens"], 170)
        self.assertEqual(metrics["cached_tokens"], 100)
        self.assertEqual(metrics["request_started_at"], 1700000000.0)
        self.assertEqual(metrics["schema"], 1)

    def test_reasoning_starts_generation_window_but_not_first_token(self) -> None:
        clock = FakeClock()
        tracker = _tracker(clock)
        clock.advance(1.0)
        tracker.mark_output("reasoning")
        clock.advance(3.0)
        tracker.mark_output("tool_call")
        clock.advance(1.0)
        tracker.record_usage({"completion_tokens": 200})
        metrics = tracker.finish("tool_calls", 2)

        self.assertEqual(metrics["time_to_first_token_s"], 4.0)
        self.assertEqual(metrics["output_tokens_per_s"], 50.0)  # 200 tokens over 4 s since first output
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["finish_reason"], "tool_calls")

    def test_estimates_tokens_when_provider_reports_no_usage(self) -> None:
        clock = FakeClock()
        tracker = _tracker(clock)
        clock.advance(0.2)
        tracker.mark_output("content")
        tracker.add_output_text("The answer is 42.")
        clock.advance(1.0)
        metrics = tracker.finish("stop", 0)

        self.assertTrue(metrics["output_tokens_estimated"])
        self.assertGreater(metrics["output_tokens"] or 0, 0)
        self.assertEqual(metrics["tokens_per_s_source"], "estimated")
        self.assertIsNone(metrics["prompt_tokens"])
        self.assertIsNone(metrics["total_tokens"])

    def test_server_timings_take_precedence(self) -> None:
        clock = FakeClock()
        tracker = _tracker(clock)
        tracker.mark_output("content")
        clock.advance(1.0)
        tracker.record_server_timings(
            {
                "cache_n": 30,
                "prompt_n": 90,
                "prompt_per_second": 812.345,
                "predicted_n": 64,
                "predicted_per_second": 38.456,
            }
        )
        metrics = tracker.finish("stop", 0)

        self.assertEqual(metrics["output_tokens_per_s"], 38.46)
        self.assertEqual(metrics["tokens_per_s_source"], "server_timings")
        self.assertEqual(metrics["output_tokens"], 64)
        self.assertFalse(metrics["output_tokens_estimated"])
        self.assertEqual(metrics["prompt_tokens_per_s"], 812.35)
        self.assertEqual(metrics["cached_tokens"], 30)
        self.assertEqual(metrics["server_timings"]["prompt_n"], 90)

    def test_non_streamed_request_has_no_first_token(self) -> None:
        clock = FakeClock()
        tracker = _tracker(clock, streamed=False)
        clock.advance(2.0)
        tracker.record_usage({"completion_tokens": 10})
        metrics = tracker.finish(None, 0, error=None)

        self.assertIsNone(metrics["time_to_first_token_s"])
        self.assertFalse(metrics["streamed"])
        self.assertEqual(metrics["output_tokens_per_s"], 5.0)
        self.assertEqual(metrics["finish_reason"], "stop")
        self.assertNotIn("error", metrics)

    def test_metrics_are_json_serializable(self) -> None:
        tracker = _tracker(FakeClock())
        tracker.record_server_timings({"predicted_per_second": 10.0})
        json.dumps(tracker.finish("error", 0, error="boom"))


class TestUsageExtraction(unittest.TestCase):
    def test_chat_completions_usage_with_details(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
            completion_tokens_details={"reasoning_tokens": 5},
        )
        self.assertEqual(
            usage_from_chat_completions(usage),
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_tokens": 64,
                "reasoning_tokens": 5,
            },
        )

    def test_responses_usage(self) -> None:
        usage = {
            "input_tokens": 300,
            "output_tokens": 40,
            "total_tokens": 340,
            "input_tokens_details": {"cached_tokens": 128},
            "output_tokens_details": {"reasoning_tokens": 30},
        }
        result = usage_from_responses(usage)
        self.assertEqual(result["prompt_tokens"], 300)
        self.assertEqual(result["cached_tokens"], 128)
        self.assertEqual(result["reasoning_tokens"], 30)

    def test_anthropic_usage_sums_cache_reads_and_writes(self) -> None:
        usage = {
            "input_tokens": 10,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 20,
            "output_tokens": 7,
        }
        self.assertEqual(
            usage_from_anthropic(usage), {"prompt_tokens": 530, "completion_tokens": 7, "cached_tokens": 500}
        )

    def test_llama_timings_from_dict_and_missing(self) -> None:
        self.assertEqual(
            llama_timings_from({"timings": {"predicted_per_second": 12, "junk": "x"}}), {"predicted_per_second": 12.0}
        )
        self.assertIsNone(llama_timings_from({"choices": []}))
        self.assertIsNone(llama_timings_from(MagicMock()))


def _chat_chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    chunk: Dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    chunk.update(extra)
    return chunk


def _usage_chunk(usage: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    chunk: Dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": usage,
    }
    chunk.update(extra)
    return chunk


def _sse(chunks: List[Dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


LLAMA_TIMINGS = {
    "cache_n": 40,
    "prompt_n": 812,
    "prompt_ms": 950.2,
    "prompt_per_token_ms": 1.17,
    "prompt_per_second": 854.5,
    "predicted_n": 31,
    "predicted_ms": 815.8,
    "predicted_per_token_ms": 26.3,
    "predicted_per_second": 38.0,
}


class _OpenAIStreamCase(unittest.TestCase):
    """Serves one SSE body through a real openai client and records request bodies."""

    sse_body = ""

    def setUp(self) -> None:
        self.request_bodies: List[Dict[str, Any]] = []

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.request_bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=self.sse_body)

    def _client(self) -> OpenAI:
        http_client = httpx2.Client(transport=httpx2.MockTransport(self._handler))
        self.addCleanup(http_client.close)
        return OpenAI(api_key="k", base_url="http://mock/v1", http_client=http_client, max_retries=0)

    @staticmethod
    def _final(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        finals = [event for event in events if event.get("type") == "final"]
        assert len(finals) == 1
        return finals[0]


class TestChatCompletionsStreamMetrics(_OpenAIStreamCase):
    def _make_api(self) -> OpenAIChatCompletionsAPI:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            api = OpenAIChatCompletionsAPI(model=AIModel.from_identifier("gpt-4.1"), tools=[])
        api.client = self._client()
        return api

    def test_usage_from_trailing_chunk(self) -> None:
        self.sse_body = _sse(
            [
                _chat_chunk({"role": "assistant", "content": "Hi"}),
                _chat_chunk({"content": " there"}),
                _chat_chunk({}, finish_reason="stop"),
                _usage_chunk(
                    {
                        "prompt_tokens": 1500,
                        "completion_tokens": 3,
                        "total_tokens": 1503,
                        "prompt_tokens_details": {"cached_tokens": 1024},
                    }
                ),
            ]
        )
        api = self._make_api()
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"}))))

        self.assertEqual(self.request_bodies[0]["stream_options"], {"include_usage": True})
        self.assertEqual(final["ai_message"], "Hi there")
        metrics = final["metrics"]
        self.assertEqual(metrics["provider"], "openai")
        self.assertEqual(metrics["model"], "gpt-4.1")
        self.assertEqual(metrics["api"], "chat_completions")
        self.assertEqual(metrics["prompt_tokens"], 1500)
        self.assertEqual(metrics["completion_tokens"], 3)
        self.assertEqual(metrics["cached_tokens"], 1024)
        self.assertFalse(metrics["output_tokens_estimated"])
        self.assertIsNotNone(metrics["time_to_first_token_s"])
        self.assertLessEqual(metrics["time_to_first_token_s"], metrics["total_latency_s"])
        self.assertEqual(metrics["tool_calls"], 0)
        self.assertEqual(metrics["finish_reason"], "stop")
        self.assertEqual(api.last_response_metrics, metrics)

    def test_tool_call_counts_as_first_token(self) -> None:
        tool_delta = {
            "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "undo", "arguments": "{}"}}
            ]
        }
        self.sse_body = _sse([_chat_chunk(tool_delta), _chat_chunk({}, finish_reason="tool_calls")])
        api = self._make_api()
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "undo"}))))

        metrics = final["metrics"]
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertIsNotNone(metrics["time_to_first_token_s"])
        self.assertTrue(metrics["output_tokens_estimated"])  # no usage chunk was sent
        self.assertEqual(metrics["finish_reason"], "tool_calls")

    def test_error_final_event_carries_metrics(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            api = OpenAIChatCompletionsAPI(tools=[])
        api.client = MagicMock()
        api.client.chat.completions.create.side_effect = RuntimeError("boom")
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"}))))

        self.assertEqual(final["metrics"]["finish_reason"], "error")
        self.assertEqual(final["metrics"]["error"], "boom")
        self.assertIsNone(final["metrics"]["time_to_first_token_s"])

    @patch("static.openai_api_base.OpenAI")
    def test_non_streaming_completion_records_last_metrics(self, mock_openai: Mock) -> None:
        api = OpenAIChatCompletionsAPI(tools=[])
        message = SimpleNamespace(content="ok", tool_calls=None)
        api.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2, total_tokens=11),
        )
        api.create_chat_completion(json.dumps({"user_message": "hi"}))

        metrics = api.last_response_metrics
        assert metrics is not None
        self.assertFalse(metrics["streamed"])
        self.assertIsNone(metrics["time_to_first_token_s"])
        self.assertEqual(metrics["total_tokens"], 11)


class TestLocalAgentStreamMetrics(_OpenAIStreamCase):
    def _make_api(self) -> LocalAgentAPI:
        model = AIModel("qwen3.8-27b", has_vision=False, provider="local_agent")
        api = LocalAgentAPI(model=model, tools=[])
        api.client = self._client()
        return api

    def test_llama_server_usage_and_timings(self) -> None:
        # llama-server with stream_options.include_usage: usage and timings on a trailing empty-choices chunk.
        self.sse_body = _sse(
            [
                _chat_chunk({"role": "assistant", "content": "x = 2"}),
                _chat_chunk({}, finish_reason="stop"),
                _usage_chunk(
                    {"prompt_tokens": 812, "completion_tokens": 31, "total_tokens": 843},
                    timings=LLAMA_TIMINGS,
                ),
            ]
        )
        api = self._make_api()
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "solve"}))))

        self.assertEqual(self.request_bodies[0]["stream_options"], {"include_usage": True})
        metrics = final["metrics"]
        self.assertEqual(metrics["provider"], "local_agent")
        self.assertEqual(metrics["model"], "qwen3.8-27b")
        self.assertEqual(metrics["prompt_tokens"], 812)
        self.assertEqual(metrics["completion_tokens"], 31)
        self.assertEqual(metrics["output_tokens_per_s"], 38.0)
        self.assertEqual(metrics["tokens_per_s_source"], "server_timings")
        self.assertEqual(metrics["prompt_tokens_per_s"], 854.5)
        self.assertEqual(metrics["cached_tokens"], 40)
        self.assertEqual(metrics["server_timings"]["predicted_ms"], 815.8)
        self.assertLessEqual(metrics["time_to_first_token_s"], metrics["total_latency_s"])

    def test_timings_on_finish_chunk_without_usage(self) -> None:
        # Older llama-server builds attach timings to the finish chunk and send no usage chunk.
        self.sse_body = _sse(
            [
                _chat_chunk({"content": "ok"}),
                _chat_chunk({}, finish_reason="stop", timings=LLAMA_TIMINGS),
            ]
        )
        api = self._make_api()
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "hi"}))))

        metrics = final["metrics"]
        self.assertEqual(metrics["output_tokens"], 31)
        self.assertFalse(metrics["output_tokens_estimated"])
        self.assertEqual(metrics["tokens_per_s_source"], "server_timings")

    def test_estimated_when_server_reports_nothing(self) -> None:
        self.sse_body = _sse([_chat_chunk({"content": "Plotted y = x^2."}), _chat_chunk({}, finish_reason="stop")])
        api = self._make_api()
        final = self._final(list(api.create_chat_completion_stream(json.dumps({"user_message": "plot"}))))

        metrics = final["metrics"]
        self.assertTrue(metrics["output_tokens_estimated"])
        self.assertGreater(metrics["output_tokens"], 0)
        self.assertIsNone(metrics["prompt_tokens"])
        self.assertNotIn("server_timings", metrics)


class TestResponsesStreamMetrics(unittest.TestCase):
    @patch("static.openai_api_base.OpenAI")
    def test_usage_and_first_token_from_response_events(self, mock_openai: Mock) -> None:
        api = OpenAIResponsesAPI()
        api.model = AIModel.from_identifier("gpt-5.5")
        usage = SimpleNamespace(
            input_tokens=900,
            output_tokens=120,
            total_tokens=1020,
            input_tokens_details=SimpleNamespace(cached_tokens=512),
            output_tokens_details=SimpleNamespace(reasoning_tokens=100),
        )
        events = [
            SimpleNamespace(
                type="response.output_item.added", output_index=0, item=SimpleNamespace(type="reasoning", summary=[])
            ),
            SimpleNamespace(type="response.output_text.delta", delta="Done."),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1", status="completed", output=[], usage=usage),
            ),
        ]
        api.client.responses.create.return_value = iter(events)
        output = list(api.create_response_stream(json.dumps({"user_message": "hi"})))
        final = [event for event in output if event.get("type") == "final"][0]

        metrics = final["metrics"]
        self.assertEqual(metrics["api"], "responses")
        self.assertEqual(metrics["model"], "gpt-5.5")
        self.assertEqual(metrics["prompt_tokens"], 900)
        self.assertEqual(metrics["completion_tokens"], 120)
        self.assertEqual(metrics["cached_tokens"], 512)
        self.assertEqual(metrics["reasoning_tokens"], 100)
        self.assertIsNotNone(metrics["time_to_first_token_s"])
        self.assertEqual(metrics["tokens_per_s_source"], "usage")

    @patch("static.openai_api_base.OpenAI")
    def test_stream_error_carries_metrics(self, mock_openai: Mock) -> None:
        api = OpenAIResponsesAPI()
        api.client.responses.create.side_effect = RuntimeError("down")
        output = list(api.create_response_stream(json.dumps({"user_message": "hi"})))
        final = [event for event in output if event.get("type") == "final"][0]
        self.assertEqual(final["metrics"]["finish_reason"], "error")
        self.assertEqual(final["metrics"]["error"], "down")


def _anthropic_sse() -> str:
    message_start: Dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "cache_read_input_tokens": 2000, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message_start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sure."}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "undo", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class TestAnthropicStreamMetrics(unittest.TestCase):
    def test_usage_events_and_tool_calls(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=_anthropic_sse())

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            api = AnthropicAPI(model=AIModel.from_identifier("claude-haiku-4-5"), tools=[])
        http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
        self.addCleanup(http_client.close)
        api._anthropic_client = anthropic.Anthropic(api_key="k", http_client=http_client, max_retries=0)

        output = list(api.create_chat_completion_stream(json.dumps({"user_message": "undo"})))
        final = [event for event in output if event.get("type") == "final"][0]

        metrics = final["metrics"]
        self.assertEqual(metrics["provider"], "anthropic")
        self.assertEqual(metrics["api"], "anthropic_messages")
        self.assertEqual(metrics["prompt_tokens"], 2012)
        self.assertEqual(metrics["cached_tokens"], 2000)
        self.assertEqual(metrics["completion_tokens"], 42)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["finish_reason"], "tool_calls")
        self.assertLessEqual(metrics["time_to_first_token_s"], metrics["total_latency_s"])


class TestMetricsLogging(unittest.TestCase):
    def test_log_response_metrics_writes_json_line(self) -> None:
        manager = LogManager.__new__(LogManager)
        manager._logger = MagicMock()
        manager.log_response_metrics({"model": "m", "total_latency_s": 1.5})

        fmt, payload = manager._logger.info.call_args[0]
        self.assertEqual(fmt, "response_metrics %s")
        self.assertEqual(json.loads(payload), {"model": "m", "total_latency_s": 1.5})


class TestRoutesForwardMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.original_require_auth = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"
        from static.app_manager import AppManager

        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def _post_stream(self, prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = {"message": json.dumps({"use_vision": False, "ai_model": "gpt-4.1", **prompt}), "svg_state": None}
        response = self.client.post("/send_message_stream", json=payload)
        return [json.loads(line) for line in response.data.decode("utf-8").splitlines() if line.strip()]

    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream")
    def test_stream_final_event_metrics_are_tagged_and_logged(self, mock_stream: Mock) -> None:
        def stream() -> Iterator[Dict[str, Any]]:
            yield {"type": "token", "text": "ok"}
            yield {
                "type": "final",
                "ai_message": "ok",
                "ai_tool_calls": [],
                "finish_reason": "stop",
                "metrics": {"model": "gpt-4.1", "total_latency_s": 0.4},
            }

        mock_stream.side_effect = lambda message: stream()
        with patch.object(self.app.log_manager, "log_response_metrics") as log_metrics:
            events = self._post_stream({"user_message": None, "tool_call_results": "[]x"})

        self.assertEqual(events[-1]["metrics"]["request_kind"], "tool_results")
        log_metrics.assert_called_once()
        self.assertEqual(log_metrics.call_args[0][0]["request_kind"], "tool_results")

    @patch.object(OpenAIChatCompletionsAPI, "create_chat_completion")
    def test_send_message_returns_provider_metrics(self, mock_chat: Mock) -> None:
        def complete(message: str) -> Any:
            self.app.ai_api.last_response_metrics = {"model": "gpt-4.1", "total_latency_s": 1.0}
            return SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")

        mock_chat.side_effect = complete
        payload = {"message": json.dumps({"user_message": "hi", "use_vision": False, "ai_model": "gpt-4.1"})}
        data = json.loads(self.client.post("/send_message", json=payload).data)

        self.assertEqual(data["data"]["metrics"]["request_kind"], "user_message")
        self.assertEqual(data["data"]["metrics"]["total_latency_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
