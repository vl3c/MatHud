"""Tests for LogManager — structured action trace logging."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from static.log_manager import MAX_LOG_FILES, LogManager, prune_old_log_files


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


class TestLogRetention(unittest.TestCase):
    """Verify startup pruning keeps only the newest session logs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.logs_dir = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_file(self, name: str, age_rank: int) -> None:
        """Create a file whose mtime is older the higher age_rank is."""
        path = os.path.join(self.logs_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
        mtime = time.time() - 1000 - age_rank * 60
        os.utime(path, (mtime, mtime))

    def _session_name(self, index: int) -> str:
        # Alternate the current daily form and the older timestamped form.
        if index % 2:
            return f"mathud_session_24_{index // 28 + 1:02d}_{index % 28 + 1:02d}.log"
        return f"mathud_session_24_03_12_{index // 60:02d}_{index % 60:02d}_00.log"

    def test_prune_keeps_newest_session_logs_only(self) -> None:
        names = [self._session_name(i) for i in range(MAX_LOG_FILES + 7)]
        for rank, name in enumerate(names):
            self._make_file(name, age_rank=rank)

        deleted = prune_old_log_files(self.logs_dir)

        self.assertEqual(sorted(deleted), sorted(names[MAX_LOG_FILES:]))
        self.assertEqual(sorted(os.listdir(self.logs_dir)), sorted(names[:MAX_LOG_FILES]))

    def test_prune_ignores_files_not_written_by_log_manager(self) -> None:
        unrelated = ["notes.txt", "mathud_log_24_03_12_21_49_12.log", "other_session_24_01_01.log", "keep.log"]
        for rank, name in enumerate(unrelated):
            self._make_file(name, age_rank=1000 + rank)
        os.mkdir(os.path.join(self.logs_dir, "mathud_session_24_01_01.log"))
        for rank in range(3):
            self._make_file(self._session_name(rank), age_rank=rank)

        deleted = prune_old_log_files(self.logs_dir, keep=1)

        self.assertEqual(len(deleted), 2)
        for name in unrelated + ["mathud_session_24_01_01.log", self._session_name(0)]:
            self.assertTrue(os.path.exists(os.path.join(self.logs_dir, name)), name)

    def test_prune_never_deletes_current_log_and_counts_it(self) -> None:
        current = "mathud_session_24_09_24.log"
        self._make_file(current, age_rank=500)  # oldest, but still in use
        for rank in range(5):
            self._make_file(self._session_name(rank), age_rank=rank)

        prune_old_log_files(self.logs_dir, keep=3, current_file_name=current)

        remaining = sorted(os.listdir(self.logs_dir))
        self.assertEqual(remaining, sorted([current, self._session_name(0), self._session_name(1)]))

    def test_setup_logging_prunes_on_startup(self) -> None:
        manager = LogManager.__new__(LogManager)
        manager.logs_dir = self.logs_dir
        manager._logger = logging.getLogger("test_log_retention")
        manager._logger.handlers = [logging.NullHandler()]
        root_handlers = logging.getLogger().handlers
        with (
            patch("static.log_manager.prune_old_log_files") as mock_prune,
            patch.object(logging.getLogger(), "handlers", root_handlers or [logging.NullHandler()]),
        ):
            manager._setup_logging()

        mock_prune.assert_called_once_with(self.logs_dir, current_file_name=manager._get_log_file_name())
