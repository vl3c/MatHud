"""Tests for per-turn response metrics: aggregation, footer text and the chat footer element."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from browser import html

from chat_ui_manager import ChatUIManager
from message_menu_manager import MessageMenuManager
from tool_call_log_manager import ToolCallLogManager
from turn_metrics import (
    MAX_TURN_HISTORY,
    TurnMetricsCollector,
    aggregate_turn,
    format_metrics_details,
    format_metrics_footer,
    format_seconds,
    short_model_name,
    turn_outcome,
)


def _request(**overrides: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "schema": 1,
        "provider": "local_agent",
        "model": "qwen3.8-27b",
        "api": "chat_completions",
        "streamed": True,
        "time_to_first_token_s": 0.8,
        "total_latency_s": 2.0,
        "output_tokens": 38,
        "output_tokens_estimated": False,
        "output_tokens_per_s": 38.0,
        "tokens_per_s_source": "server_timings",
        "prompt_tokens": 800,
        "completion_tokens": 38,
        "cached_tokens": 700,
        "reasoning_tokens": None,
        "total_tokens": 838,
        "tool_calls": 0,
        "finish_reason": "stop",
    }
    metrics.update(overrides)
    return metrics


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestTurnAggregation(unittest.TestCase):
    def test_sums_requests_and_counts_tool_errors(self) -> None:
        requests = [
            _request(tool_calls=3, finish_reason="tool_calls", output_tokens=76, output_tokens_per_s=38.0),
            _request(time_to_first_token_s=0.5, total_latency_s=1.0, output_tokens=38, output_tokens_per_s=19.0),
        ]
        tool_results = [
            {"function_name": "create_point", "result": "Point A created", "is_error": False},
            {"function_name": "create_segment", "result": "Error: point B not found", "is_error": True},
            {"function_name": "evaluate", "result": "Error: bad expression", "is_error": False},
        ]
        turn = aggregate_turn(requests, tool_results, 4.2, "stop")

        self.assertEqual(turn["requests"], 2)
        self.assertEqual(turn["tool_calls"], 3)
        self.assertEqual(turn["tool_executions"], 3)
        self.assertEqual(turn["tool_errors"], 2)
        self.assertEqual(turn["tool_error_messages"][0], "create_segment: Error: point B not found")
        self.assertEqual(turn["time_to_first_token_s"], 0.8)
        self.assertEqual(turn["model_time_s"], 3.0)
        self.assertEqual(turn["wall_time_s"], 4.2)
        self.assertEqual(turn["prompt_tokens"], 1600)
        self.assertEqual(turn["total_tokens"], 1676)
        self.assertIsNone(turn["reasoning_tokens"])
        self.assertEqual(turn["output_tokens"], 114)
        # 114 tokens over 2 s + 2 s of generation
        self.assertEqual(turn["output_tokens_per_s"], 28.5)
        self.assertEqual(turn["model"], "qwen3.8-27b")
        self.assertEqual(len(turn["per_request"]), 2)

    def test_estimated_flag_propagates(self) -> None:
        turn = aggregate_turn([_request(), _request(output_tokens_estimated=True)], [], 1.0, "stop")
        self.assertTrue(turn["output_tokens_estimated"])

    def test_missing_rates_and_tokens_stay_none(self) -> None:
        request = _request(output_tokens_per_s=None, prompt_tokens=None, completion_tokens=None, total_tokens=None)
        turn = aggregate_turn([request], [], None, "error")
        self.assertIsNone(turn["output_tokens_per_s"])
        self.assertIsNone(turn["prompt_tokens"])
        self.assertIsNone(turn["wall_time_s"])
        self.assertEqual(turn["outcome"], "error")


class TestFooterFormatting(unittest.TestCase):
    def test_footer_for_multi_request_turn(self) -> None:
        requests = [_request(tool_calls=2), _request(tool_calls=1)]
        turn = aggregate_turn(requests, [], 4.2, "stop")
        self.assertEqual(
            format_metrics_footer(turn),
            "qwen3.8-27b · 4.2 s · first token 0.8 s · 38 tok/s · 2 requests · 3 tool calls",
        )

    def test_footer_for_single_request_omits_counts(self) -> None:
        turn = aggregate_turn([_request()], [], 1.34, "stop")
        self.assertEqual(format_metrics_footer(turn), "qwen3.8-27b · 1.3 s · first token 0.8 s · 38 tok/s")

    def test_footer_marks_estimates_and_errors(self) -> None:
        request = _request(output_tokens_estimated=True, tool_calls=1)
        tool_results = [{"function_name": "f", "result": "Error: x", "is_error": True}]
        footer = format_metrics_footer(aggregate_turn([request], tool_results, 12.4, "stop"))
        self.assertIn("~38 tok/s", footer)
        self.assertIn("12 s", footer)
        self.assertIn("1 tool call", footer)
        self.assertIn("1 tool error", footer)

    def test_footer_empty_without_requests(self) -> None:
        self.assertEqual(format_metrics_footer(None), "")
        self.assertEqual(format_metrics_footer(aggregate_turn([], [], 1.0, "stop")), "")

    def test_details_include_tokens_breakdown_and_errors(self) -> None:
        tool_results = [{"function_name": "solve", "result": "Error: no solution", "is_error": True}]
        details = format_metrics_details(
            aggregate_turn([_request(tool_calls=1), _request()], tool_results, 3.0, "stop")
        )
        self.assertIn("prompt 1600 (cached 1400)", details)
        self.assertIn("completion 76", details)
        self.assertIn("llama-server timings", details)
        self.assertIn("Request 1:", details)
        self.assertIn("Request 2:", details)
        self.assertIn("- solve: Error: no solution", details)

    def test_seconds_and_model_names(self) -> None:
        self.assertEqual(format_seconds(0.84), "0.8 s")
        self.assertEqual(format_seconds(42.4), "42 s")
        self.assertEqual(format_seconds(75), "1 m 15 s")
        self.assertEqual(format_seconds(None), "?")
        self.assertEqual(short_model_name("C:\\models\\Qwen3.8-27B-Q4_K_M.gguf"), "Qwen3.8-27B-Q4_K_M")
        self.assertEqual(short_model_name("google/gemini-3.1-pro-preview"), "gemini-3.1-pro-preview")


class TestTurnMetricsCollector(unittest.TestCase):
    def test_turn_lifecycle_and_history(self) -> None:
        clock = _Clock()
        collector = TurnMetricsCollector(clock)
        collector.start_turn("draw a circle")
        collector.record_request(_request(tool_calls=1, finish_reason="tool_calls"))
        collector.record_tool_results([{"function_name": "create_circle", "result": "ok", "is_error": False}])
        collector.record_request(_request())
        clock.now += 2500.0
        turn = collector.finish_turn("stop")

        assert turn is not None
        self.assertEqual(turn["turn_id"], 1)
        self.assertEqual(turn["user_message"], "draw a circle")
        self.assertEqual(turn["wall_time_s"], 2.5)
        self.assertEqual(turn["requests"], 2)
        self.assertEqual(turn["tool_executions"], 1)
        self.assertFalse(collector.is_active)
        self.assertIs(collector.last_turn(), turn)
        json.dumps(collector.history())

    def test_ignores_events_outside_a_turn(self) -> None:
        collector = TurnMetricsCollector(_Clock())
        collector.record_request(_request())
        collector.record_tool_results([{"function_name": "f", "result": "ok"}])
        self.assertIsNone(collector.finish_turn())
        self.assertIsNone(collector.last_turn())

    def test_history_is_bounded_and_clearable(self) -> None:
        collector = TurnMetricsCollector(_Clock(), max_history=3)
        for index in range(5):
            collector.start_turn(f"turn {index}")
            collector.finish_turn()
        history = collector.history()
        self.assertEqual([turn["turn_id"] for turn in history], [3, 4, 5])
        collector.clear()
        self.assertEqual(collector.history(), [])
        self.assertGreater(MAX_TURN_HISTORY, 3)

    def test_missing_metrics_still_counts_turn(self) -> None:
        collector = TurnMetricsCollector(_Clock())
        collector.start_turn("hi")
        collector.record_request(None)
        turn = collector.finish_turn("error")
        assert turn is not None
        self.assertEqual(turn["requests"], 0)
        self.assertEqual(format_metrics_footer(turn), "")


class TestMetricsFooterElement(unittest.TestCase):
    def _chat_ui(self) -> ChatUIManager:
        return ChatUIManager(message_menu=MessageMenuManager(), tool_call_log=ToolCallLogManager())

    @staticmethod
    def _footers(container: Any) -> List[Any]:
        return [child for child in container.children if "chat-metrics-footer" in str(child.attrs.get("class", ""))]

    def test_footer_appended_with_tooltip(self) -> None:
        container = html.DIV(Class="chat-message normal")
        turn = aggregate_turn([_request()], [], 1.0, "stop")
        self._chat_ui().append_metrics_footer(container, turn)

        footers = self._footers(container)
        self.assertEqual(len(footers), 1)
        self.assertEqual(footers[0].text, format_metrics_footer(turn))
        self.assertIn("Request 1:", footers[0].attrs["title"])

    def test_no_footer_without_metrics(self) -> None:
        container = html.DIV(Class="chat-message normal")
        chat_ui = self._chat_ui()
        empty_turn: Optional[Dict[str, Any]] = None
        chat_ui.append_metrics_footer(container, empty_turn)
        chat_ui.append_metrics_footer(container, aggregate_turn([], [], 1.0, "stop"))
        chat_ui.append_metrics_footer(None, aggregate_turn([_request()], [], 1.0, "stop"))
        self.assertEqual(self._footers(container), [])

    def test_pending_metrics_cleared_by_reset(self) -> None:
        chat_ui = self._chat_ui()
        chat_ui.pending_turn_metrics = aggregate_turn([_request()], [], 1.0, "stop")
        chat_ui.reset_streaming_state()
        self.assertIsNone(chat_ui.pending_turn_metrics)


class TestTurnOutcomes(unittest.TestCase):
    def test_finish_reasons_map_to_outcomes(self) -> None:
        self.assertEqual(turn_outcome("stop"), "stop")
        self.assertEqual(turn_outcome("completed"), "stop")
        self.assertEqual(turn_outcome(None), "stop")
        self.assertEqual(turn_outcome("error"), "error")
        for reason in ("length", "max_tokens", "max_output_tokens", "incomplete"):
            self.assertEqual(turn_outcome(reason), "truncated")
        for reason in ("content_filter", "refusal"):
            self.assertEqual(turn_outcome(reason), "filtered")

    def test_stale_turn_token_is_ignored(self) -> None:
        collector = TurnMetricsCollector(_Clock())
        collector.start_turn("first")
        old_token = collector.turn_token
        collector.start_turn("second")

        collector.record_request(_request(), turn_token=old_token)
        self.assertIsNone(collector.finish_turn("stop", turn_token=old_token))
        self.assertTrue(collector.is_active)

        collector.record_request(_request(), turn_token=collector.turn_token)
        turn = collector.finish_turn("stop", turn_token=collector.turn_token)
        assert turn is not None
        self.assertEqual(turn["requests"], 1)
        self.assertEqual(turn["user_message"], "second")


class _FakeRequest:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self.text = "response text"
        self.json = payload


class TestTurnBookkeeping(unittest.TestCase):
    """AIInterface closes the turn on every way a request can end."""

    def _ai(self) -> Any:
        from ai_interface import AIInterface

        ai = AIInterface.__new__(AIInterface)
        tool_call_log = ToolCallLogManager()
        ai._tool_call_log = tool_call_log
        ai._chat_ui = ChatUIManager(message_menu=MessageMenuManager(), tool_call_log=tool_call_log)
        ai._turn_metrics_collector = TurnMetricsCollector(_Clock())
        ai._last_user_message = ""
        ai.is_processing = True
        ai._stop_requested = False
        ai._response_timeout_id = None
        ai._finalize_stream_message = lambda msg=None: None
        ai._enable_send_controls = lambda: None
        ai._restore_user_message_on_error = lambda: None
        ai._turn_metrics.start_turn("hi")
        return ai

    def _last_outcome(self, ai: Any) -> Optional[str]:
        turn = ai._turn_metrics.last_turn()
        return None if turn is None else turn["outcome"]

    def test_request_error_finishes_turn(self) -> None:
        ai = self._ai()
        ai._on_error(_FakeRequest(500))
        self.assertFalse(ai._turn_metrics.is_active)
        self.assertEqual(self._last_outcome(ai), "error")

    def test_non_200_response_finishes_turn(self) -> None:
        ai = self._ai()
        ai._on_complete(_FakeRequest(502, {"message": "bad gateway"}))
        self.assertEqual(self._last_outcome(ai), "error")

    def test_response_without_data_finishes_turn(self) -> None:
        ai = self._ai()
        ai._on_complete(_FakeRequest(200, {"message": "Invalid response format"}))
        self.assertEqual(self._last_outcome(ai), "error")

    def test_failing_final_handler_finishes_turn(self) -> None:
        ai = self._ai()

        def broken(event: Any) -> Dict[str, Any]:
            raise ValueError("unreadable event")

        ai._normalize_stream_event = broken
        ai._on_stream_final({"finish_reason": "stop"})
        self.assertEqual(self._last_outcome(ai), "error")

    def test_truncated_answer_has_its_own_outcome(self) -> None:
        ai = self._ai()
        event = {"finish_reason": "length", "ai_tool_calls": [], "ai_message": "partial", "metrics": _request()}
        ai._on_stream_final(event, ai._turn_metrics.turn_token)
        self.assertEqual(self._last_outcome(ai), "truncated")

    def test_late_final_event_of_an_old_turn_is_ignored(self) -> None:
        ai = self._ai()
        old_token = ai._turn_metrics.turn_token
        ai._turn_metrics.start_turn("next question")

        ai._on_stream_final({"finish_reason": "stop", "ai_tool_calls": [], "ai_message": "old"}, old_token)

        self.assertTrue(ai._turn_metrics.is_active)
        self.assertIsNone(ai._turn_metrics.last_turn())
