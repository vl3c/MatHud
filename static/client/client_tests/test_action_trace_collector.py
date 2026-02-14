"""Tests for ActionTraceCollector — trace building, storage, delta computation, and export."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from managers.action_trace_collector import ActionTraceCollector, _MAX_TRACES


class TestComputeStateDelta(unittest.TestCase):
    """Verify state delta computation between canvas snapshots."""

    def test_added(self) -> None:
        before: Dict[str, Any] = {"Point": {}}
        after: Dict[str, Any] = {"Point": {"A": {"x": 1, "y": 2}}}
        delta = ActionTraceCollector.compute_state_delta(before, after)
        self.assertEqual(delta["added"], ["A"])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["modified"], [])

    def test_removed(self) -> None:
        before: Dict[str, Any] = {"Point": {"A": {"x": 1, "y": 2}}}
        after: Dict[str, Any] = {"Point": {}}
        delta = ActionTraceCollector.compute_state_delta(before, after)
        self.assertEqual(delta["added"], [])
        self.assertEqual(delta["removed"], ["A"])
        self.assertEqual(delta["modified"], [])

    def test_modified(self) -> None:
        before: Dict[str, Any] = {"Point": {"A": {"x": 1, "y": 2}}}
        after: Dict[str, Any] = {"Point": {"A": {"x": 3, "y": 4}}}
        delta = ActionTraceCollector.compute_state_delta(before, after)
        self.assertEqual(delta["added"], [])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["modified"], ["A"])

    def test_no_change(self) -> None:
        state: Dict[str, Any] = {"Point": {"A": {"x": 1, "y": 2}}}
        delta = ActionTraceCollector.compute_state_delta(state, state)
        self.assertEqual(delta["added"], [])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["modified"], [])

    def test_multiple_categories(self) -> None:
        before: Dict[str, Any] = {
            "Point": {"A": {"x": 0, "y": 0}},
            "Segment": {"s1": {"start": "A", "end": "B"}},
        }
        after: Dict[str, Any] = {
            "Point": {"A": {"x": 0, "y": 0}, "B": {"x": 1, "y": 1}},
            "Segment": {},
        }
        delta = ActionTraceCollector.compute_state_delta(before, after)
        self.assertIn("B", delta["added"])
        self.assertIn("s1", delta["removed"])
        self.assertEqual(delta["modified"], [])

    def test_empty_states(self) -> None:
        delta = ActionTraceCollector.compute_state_delta({}, {})
        self.assertEqual(delta, {"added": [], "removed": [], "modified": []})


class TestBuildTrace(unittest.TestCase):
    """Verify build_trace produces correct structure."""

    def setUp(self) -> None:
        self.collector = ActionTraceCollector()

    def test_structure(self) -> None:
        before: Dict[str, Any] = {"Point": {}}
        after: Dict[str, Any] = {"Point": {"A": {"x": 1, "y": 2}}}
        calls: List[Dict[str, Any]] = [{
            "seq": 0,
            "function_name": "create_point",
            "arguments": {"x": 1, "y": 2},
            "result": "Success",
            "is_error": False,
            "duration_ms": 1.5,
        }]
        trace = self.collector.build_trace(before, after, calls, 2.0)

        self.assertIn("trace_id", trace)
        self.assertIn("timestamp", trace)
        self.assertEqual(trace["tool_calls"], calls)
        self.assertEqual(trace["total_duration_ms"], 2.0)
        self.assertIn("state_delta", trace)
        self.assertEqual(trace["state_delta"]["added"], ["A"])
        self.assertIn("canvas_state_before", trace)
        self.assertIn("canvas_state_after", trace)

    def test_monotonic_ids(self) -> None:
        """Trace IDs should increment even within the same millisecond."""
        t1 = self.collector.build_trace({}, {}, [], 0)
        t2 = self.collector.build_trace({}, {}, [], 0)
        # Counter portion should differ
        self.assertNotEqual(t1["trace_id"], t2["trace_id"])


class TestStoreAndRetrieve(unittest.TestCase):
    """Verify store, get_traces, and get_last_trace."""

    def setUp(self) -> None:
        self.collector = ActionTraceCollector()

    def _make_trace(self, **overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "trace_id": "test-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "tool_calls": [],
            "state_delta": {"added": [], "removed": [], "modified": []},
            "total_duration_ms": 0,
            "canvas_state_before": {"Point": {}},
            "canvas_state_after": {"Point": {}},
        }
        base.update(overrides)
        return base

    def test_store_and_retrieve(self) -> None:
        trace = self._make_trace()
        self.collector.store(trace)
        self.assertEqual(len(self.collector.get_traces()), 1)
        self.assertEqual(self.collector.get_last_trace(), trace)

    def test_store_strips_snapshots_from_older(self) -> None:
        t1 = self._make_trace(trace_id="t1")
        t2 = self._make_trace(trace_id="t2")
        self.collector.store(t1)
        self.collector.store(t2)

        traces = self.collector.get_traces()
        # First trace should have snapshots stripped
        self.assertNotIn("canvas_state_before", traces[0])
        self.assertNotIn("canvas_state_after", traces[0])
        # Latest trace keeps them
        self.assertIn("canvas_state_before", traces[1])
        self.assertIn("canvas_state_after", traces[1])

    def test_store_cap(self) -> None:
        for i in range(_MAX_TRACES + 20):
            self.collector.store(self._make_trace(trace_id=f"t{i}"))
        self.assertEqual(len(self.collector.get_traces()), _MAX_TRACES)

    def test_clear(self) -> None:
        self.collector.store(self._make_trace())
        self.collector.clear()
        self.assertEqual(len(self.collector.get_traces()), 0)
        self.assertIsNone(self.collector.get_last_trace())

    def test_get_last_trace_empty(self) -> None:
        self.assertIsNone(self.collector.get_last_trace())


class TestExportTracesJson(unittest.TestCase):
    """Verify export produces serializable output with truncated results."""

    def test_truncated_results(self) -> None:
        collector = ActionTraceCollector()
        long_result = "x" * 1000
        trace: Dict[str, Any] = {
            "trace_id": "t1",
            "timestamp": "2024-01-01T00:00:00Z",
            "tool_calls": [{
                "seq": 0,
                "function_name": "eval",
                "arguments": {},
                "result": long_result,
                "is_error": False,
                "duration_ms": 1.0,
            }],
            "state_delta": {"added": [], "removed": [], "modified": []},
            "total_duration_ms": 1.0,
            "canvas_state_before": {},
            "canvas_state_after": {},
        }
        collector.store(trace)
        exported = collector.export_traces_json()
        self.assertEqual(len(exported), 1)
        result_val = exported[0]["tool_calls"][0]["result"]
        self.assertLessEqual(len(result_val), 503)  # 500 + "..."

    def test_serializable(self) -> None:
        collector = ActionTraceCollector()
        trace: Dict[str, Any] = {
            "trace_id": "t1",
            "timestamp": "2024-01-01T00:00:00Z",
            "tool_calls": [{
                "seq": 0,
                "function_name": "create_point",
                "arguments": {"x": 1},
                "result": "OK",
                "is_error": False,
                "duration_ms": 0.5,
            }],
            "state_delta": {"added": ["A"], "removed": [], "modified": []},
            "total_duration_ms": 0.5,
            "canvas_state_before": {},
            "canvas_state_after": {"Point": {"A": {}}},
        }
        collector.store(trace)
        exported = collector.export_traces_json()
        # Must be JSON-serializable
        serialized = json.dumps(exported)
        self.assertIsInstance(serialized, str)


class TestCompactSummary(unittest.TestCase):
    """Verify build_compact_summary for server logging."""

    def test_summary_structure(self) -> None:
        collector = ActionTraceCollector()
        trace: Dict[str, Any] = {
            "trace_id": "t1",
            "timestamp": "2024-01-01T00:00:00Z",
            "tool_calls": [
                {"seq": 0, "function_name": "f1", "arguments": {}, "result": "ok",
                 "is_error": False, "duration_ms": 1.0},
                {"seq": 1, "function_name": "f2", "arguments": {}, "result": "Error: bad",
                 "is_error": True, "duration_ms": 0.5},
            ],
            "state_delta": {"added": ["A"], "removed": [], "modified": []},
            "total_duration_ms": 1.5,
        }
        summary = collector.build_compact_summary(trace)
        self.assertEqual(summary["trace_id"], "t1")
        self.assertEqual(summary["tool_count"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["total_duration_ms"], 1.5)
        self.assertEqual(len(summary["calls"]), 2)
        # Per-call summary should not include full result values
        self.assertNotIn("result", summary["calls"][0])
        self.assertIn("function_name", summary["calls"][0])
