"""Tool results must reach the model as JSON.

Brython's ``json.dumps`` rejects tuples, and after such a failure it reports
"Circular reference detected" for the same objects on every later dump. The
``analyze_graph`` MST result held tuples, so a real turn could not send it back.
These tests run results through the actual send path and round-trip the JSON.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

import result_processor as result_processor_module
from canvas import Canvas
from function_registry import FunctionRegistry
from json_safe import CIRCULAR_REFERENCE_TEXT, to_json_safe
from managers.action_trace_collector import ActionTraceCollector
from process_function_calls import ProcessFunctionCalls
from result_processor import ResultProcessor
from turn_metrics import TurnMetricsCollector
from workspace_manager import WorkspaceManager

# Scenario GR-06: square A-B-C-D with weighted edges; the MST is AB, BC, CD (weight 4)
GR06_VERTICES: List[Dict[str, Any]] = [
    {"name": "A", "x": 0, "y": 0},
    {"name": "B", "x": 4, "y": 0},
    {"name": "C", "x": 4, "y": 4},
    {"name": "D", "x": 0, "y": 4},
]
GR06_EDGES: List[Dict[str, Any]] = [
    {"source": 0, "target": 1, "weight": 1},
    {"source": 1, "target": 2, "weight": 2},
    {"source": 0, "target": 2, "weight": 4},
    {"source": 2, "target": 3, "weight": 1},
]


def _make_ai(canvas: Canvas) -> Any:
    """An AIInterface without __init__ (no DOM), wired to a real canvas and registry."""
    from ai_interface import AIInterface

    ai = AIInterface.__new__(AIInterface)
    ai.canvas = canvas
    ai.workspace_manager = WorkspaceManager(canvas)
    ai.available_functions = FunctionRegistry.get_available_functions(canvas, ai.workspace_manager)
    ai.undoable_functions = FunctionRegistry.get_undoable_functions()
    ai.is_processing = False
    ai._trace_collector = ActionTraceCollector()
    ai._turn_metrics_collector = TurnMetricsCollector(clock_ms=lambda: 0.0)
    return ai


class _Recorder:
    """Accepts any method call."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _Drawable:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_class_name(self) -> str:
        return "Point"


class TestJsonSafe(unittest.TestCase):
    """to_json_safe turns any result into data json.dumps accepts."""

    def test_plain_data_is_unchanged(self) -> None:
        value = {"a": [1, 2.5, "x", None, True], "b": {"c": False}}
        self.assertEqual(to_json_safe(value), value)

    def test_tuples_and_sets_become_lists(self) -> None:
        value = {"edges": [("A", "B"), ("B", "C")], "points": {"C", "A", "B"}, "pair": frozenset({2, 1})}
        self.assertEqual(
            to_json_safe(value),
            {"edges": [["A", "B"], ["B", "C"]], "points": ["A", "B", "C"], "pair": [1, 2]},
        )

    def test_keys_are_written_like_json_dumps(self) -> None:
        value = {2: "two", True: "yes", None: "nothing", ("A", "B"): "edge"}
        converted = to_json_safe(value)
        self.assertEqual(converted["2"], "two")
        self.assertEqual(converted["true"], "yes")
        self.assertEqual(converted["null"], "nothing")
        self.assertEqual(len(converted), 4)
        self.assertTrue(all(isinstance(key, str) for key in converted))

    def test_non_finite_floats_become_strings(self) -> None:
        value = [float("nan"), float("inf"), float("-inf"), 1.5]
        self.assertEqual(to_json_safe(value), ["NaN", "Infinity", "-Infinity", 1.5])

    def test_self_reference_becomes_text(self) -> None:
        value: Dict[str, Any] = {"name": "loop"}
        value["self"] = value
        self.assertEqual(to_json_safe(value), {"name": "loop", "self": CIRCULAR_REFERENCE_TEXT})

    def test_shared_values_are_not_circular(self) -> None:
        shared = ["A", "B"]
        self.assertEqual(to_json_safe({"x": shared, "y": [shared]}), {"x": ["A", "B"], "y": [["A", "B"]]})

    def test_objects_become_descriptions(self) -> None:
        self.assertEqual(to_json_safe({"point": _Drawable("A")}), {"point": "Point 'A'"})

        class Thing:
            def __str__(self) -> str:
                return "a thing"

        self.assertEqual(to_json_safe([Thing()]), ["a thing"])

    def test_result_round_trips_through_json(self) -> None:
        value = {"edges": [("A", "B")], "seen": {"A"}, "ratio": float("inf"), "point": _Drawable("P")}
        safe = to_json_safe(value)
        self.assertEqual(json.loads(json.dumps(safe)), safe)


class TestGraphAnalysisResultsAreJson(unittest.TestCase):
    """analyze_graph results are plain JSON data, sent intact through the real send path."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.ai = _make_ai(self.canvas)
        self.ai.execute_tool_batch(
            [
                {
                    "function_name": "generate_graph",
                    "arguments": {
                        "name": "G1",
                        "graph_type": "graph",
                        "directed": False,
                        "vertices": GR06_VERTICES,
                        "edges": GR06_EDGES,
                    },
                }
            ]
        )

    def _analyze_call(self, operation: str, call_id: str) -> Dict[str, Any]:
        return {
            "id": call_id,
            "function_name": "analyze_graph",
            "arguments": {"graph_name": "G1", "operation": operation},
        }

    def test_mst_result_round_trips_through_send_serialization(self) -> None:
        calls = [self._analyze_call("mst", "call_mst")]
        batch = self.ai.execute_tool_batch(calls)

        text = ProcessFunctionCalls.serialize_tool_call_results(calls, batch["traced_calls"])
        sent = json.loads(text)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["tool_call_id"], "call_mst")
        result = list(sent[0]["result"].values())[0]
        self.assertEqual(result["operation"], "mst")
        self.assertEqual(result["edges"], [["A", "B"], ["B", "C"], ["C", "D"]])
        self.assertEqual(sorted(result["highlight_vectors"]), ["AB", "BC", "CD"])
        self.assertTrue(result["connected"])
        self.assertEqual(json.loads(json.dumps(sent)), sent)

    def test_mst_result_still_serializes_for_traces_after_sending(self) -> None:
        calls = [self._analyze_call("mst", "call_mst")]
        batch = self.ai.execute_tool_batch(calls)
        ProcessFunctionCalls.serialize_tool_call_results(calls, batch["traced_calls"])

        # The same result objects are dumped again by the trace and metrics exports
        traced = json.loads(json.dumps(batch["traced_calls"]))
        self.assertEqual(traced[0]["result"]["edges"], [["A", "B"], ["B", "C"], ["C", "D"]])
        exported = json.loads(json.dumps(self.ai._trace_collector.export_traces_json()))
        self.assertEqual(len(exported), 2)

    def test_bridges_result_is_lists(self) -> None:
        calls = [self._analyze_call("bridges", "call_bridges")]
        batch = self.ai.execute_tool_batch(calls)

        sent = json.loads(ProcessFunctionCalls.serialize_tool_call_results(calls, batch["traced_calls"]))
        result = list(sent[0]["result"].values())[0]
        self.assertEqual(result["bridges"], [["C", "D"]])
        self.assertEqual(result["highlight_vectors"], ["CD"])

    def test_streamed_turn_sends_the_mst_result(self) -> None:
        sent: List[Any] = []
        self.ai._stop_requested = False
        self.ai._tool_call_log = _Recorder()
        self.ai._chat_ui = _Recorder(stream_container=object(), stream_content=object(), stream_buffer="")
        self.ai._start_response_timeout = lambda use_reasoning_timeout=False: None
        self.ai._normalize_stream_event = lambda event: event
        self.ai._enable_send_controls = lambda: None
        self.ai._send_prompt_to_ai = lambda user_message, tool_call_results=None, **kwargs: sent.append(
            tool_call_results
        )
        self.ai._turn_metrics.start_turn("Find the minimum spanning tree of G1")

        calls = [self._analyze_call("mst", "call_mst")]
        self.ai._on_stream_final({"finish_reason": "tool_calls", "ai_tool_calls": calls, "ai_message": ""})

        self.assertTrue(self.ai._turn_metrics.is_active)  # the turn goes on; it did not end in error
        self.assertEqual(len(sent), 1)
        result = list(json.loads(sent[0])[0]["result"].values())[0]
        self.assertEqual(result["edges"], [["A", "B"], ["B", "C"], ["C", "D"]])


class TestSendPathFallback(unittest.TestCase):
    """A result that cannot be serialized becomes an error result for its call only."""

    def test_unserializable_result_becomes_an_error_entry(self) -> None:
        calls = [
            {"id": "c1", "function_name": "evaluate_expression", "arguments": {"expression": "1+1"}},
            {"id": "c2", "function_name": "broken_tool", "arguments": {}},
        ]
        traced = [
            {"function_name": "evaluate_expression", "result_key": "1+1", "result": 2},
            {"function_name": "broken_tool", "result_key": "broken_tool()", "result": {"value": object()}},
        ]
        original = result_processor_module.to_json_safe
        result_processor_module.to_json_safe = lambda value: value  # disable the sanitiser
        try:
            text = ResultProcessor.serialize_tool_call_results(calls, traced)
        finally:
            result_processor_module.to_json_safe = original

        sent = json.loads(text)
        self.assertEqual(sent[0], {"tool_call_id": "c1", "result": {"1+1": 2}})
        self.assertEqual(sent[1]["tool_call_id"], "c2")
        message = sent[1]["result"]["broken_tool()"]
        self.assertTrue(message.startswith("Error: the result of broken_tool could not be serialized"))
        self.assertTrue(ResultProcessor.is_error_result(message))

    def test_passthrough_check_keeps_plain_dicts_only(self) -> None:
        self.assertTrue(ResultProcessor._is_small_passthrough_result({"edges": [["A", "B"]]}))
        self.assertFalse(ResultProcessor._is_small_passthrough_result({"edges": [("A", "B")]}))
        self.assertFalse(ResultProcessor._is_small_passthrough_result({"point": _Drawable("A")}))
