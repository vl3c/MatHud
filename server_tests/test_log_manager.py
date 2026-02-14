"""Tests for LogManager — structured action trace logging."""

from __future__ import annotations

import json
import logging
import unittest

from static.log_manager import LogManager


class TestLogActionTrace(unittest.TestCase):
    """Verify log_action_trace writes structured JSON."""

    def setUp(self) -> None:
        self.log_manager = LogManager.__new__(LogManager)
        self.log_manager._logger = logging.getLogger("test_action_trace")
        self.log_manager._logger.handlers = []
        self.log_manager._logger.setLevel(logging.DEBUG)
        self._captured: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: self._captured.append(record.getMessage())  # type: ignore[assignment]
        self.log_manager._logger.addHandler(handler)

    def test_log_action_trace(self) -> None:
        trace_summary = {
            "trace_id": "123-1",
            "tool_count": 2,
            "error_count": 0,
            "total_duration_ms": 5.0,
            "state_delta": {"added": ["A"], "removed": [], "modified": []},
            "calls": [
                {"function_name": "create_point", "duration_ms": 3.0, "is_error": False},
                {"function_name": "create_segment", "duration_ms": 2.0, "is_error": False},
            ],
        }
        self.log_manager.log_action_trace(trace_summary)

        self.assertEqual(len(self._captured), 1)
        msg = self._captured[0]
        self.assertTrue(msg.startswith("action_trace "))
        payload = json.loads(msg.split("action_trace ", 1)[1])
        self.assertEqual(payload["trace_id"], "123-1")
        self.assertEqual(payload["tool_count"], 2)

    def test_log_action_trace_sorted_keys(self) -> None:
        trace_summary = {"z_field": 1, "a_field": 2}
        self.log_manager.log_action_trace(trace_summary)

        msg = self._captured[0]
        json_str = msg.split("action_trace ", 1)[1]
        # Keys should be sorted
        self.assertLess(json_str.index("a_field"), json_str.index("z_field"))
