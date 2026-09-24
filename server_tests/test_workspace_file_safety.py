"""Tests for crash-safe workspace saves, backups, and recoverable deletes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

from static.workspace_manager import WorkspaceManager


def _state(label: str) -> Dict[str, Any]:
    return {"Points": [{"name": label, "args": {"position": {"x": 1, "y": 2}}}]}


class TestWorkspaceFileSafety(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspaces_dir = self._tmp.name
        self.manager = WorkspaceManager(self.workspaces_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _path(self, *parts: str) -> str:
        return os.path.join(self.workspaces_dir, *parts)

    def _read_state(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["state"]

    def _entries(self) -> List[str]:
        return sorted(os.listdir(self.workspaces_dir))

    def test_save_leaves_no_temp_files(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))
        self.assertEqual(self._entries(), ["ws.json"])
        self.assertEqual(self._read_state(self._path("ws.json")), _state("A"))

    def test_failed_save_keeps_previous_workspace_intact(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))

        with mock.patch("static.workspace_manager.json.dump", side_effect=OSError("disk full")):
            self.assertFalse(self.manager.save_workspace(_state("B"), "ws"))

        self.assertEqual(self._read_state(self._path("ws.json")), _state("A"))
        self.assertEqual(self._entries(), ["ws.json"], "Temp files must be cleaned up after a failed save")

    def test_failed_serialization_keeps_previous_workspace_intact(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))

        unserializable: Any = {"Points": [object()]}
        with self.assertRaises(TypeError):
            self.manager.save_workspace(unserializable, "ws")

        self.assertEqual(self._read_state(self._path("ws.json")), _state("A"))
        self.assertEqual(self._entries(), ["ws.json"])

    def test_overwrite_keeps_single_backup_of_previous_version(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))
        self.assertFalse(os.path.exists(self._path("ws.json.bak")))

        self.assertTrue(self.manager.save_workspace(_state("B"), "ws"))
        self.assertEqual(self._read_state(self._path("ws.json")), _state("B"))
        self.assertEqual(self._read_state(self._path("ws.json.bak")), _state("A"))

        self.assertTrue(self.manager.save_workspace(_state("C"), "ws"))
        self.assertEqual(self._read_state(self._path("ws.json")), _state("C"))
        self.assertEqual(self._read_state(self._path("ws.json.bak")), _state("B"))
        self.assertEqual(self._entries(), ["ws.json", "ws.json.bak"])

    def test_delete_moves_workspace_to_trash(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))

        self.assertTrue(self.manager.delete_workspace("ws"))

        self.assertFalse(os.path.exists(self._path("ws.json")))
        trash_files = os.listdir(self._path(".trash"))
        self.assertEqual(len(trash_files), 1)
        self.assertTrue(trash_files[0].startswith("ws_"))
        self.assertTrue(trash_files[0].endswith(".json"))
        self.assertEqual(self._read_state(self._path(".trash", trash_files[0])), _state("A"))

    def test_repeated_deletes_keep_every_trashed_version(self) -> None:
        for label in ("A", "B"):
            self.assertTrue(self.manager.save_workspace(_state(label), "ws"))
            self.assertTrue(self.manager.delete_workspace("ws"))

        trashed = sorted(os.listdir(self._path(".trash")))
        self.assertEqual(len(trashed), 2)
        states = sorted(self._read_state(self._path(".trash", name))["Points"][0]["name"] for name in trashed)
        self.assertEqual(states, ["A", "B"])

    def test_trash_names_never_overwrite_within_one_clock_tick(self) -> None:
        fixed = mock.Mock(wraps=__import__("datetime").datetime)
        fixed.now.return_value = __import__("datetime").datetime(2026, 1, 2, 3, 4, 5, 6)
        with mock.patch("static.workspace_manager.datetime", fixed):
            for label in ("A", "B", "C"):
                self.assertTrue(self.manager.save_workspace(_state(label), "ws"))
                self.assertTrue(self.manager.delete_workspace("ws"))

        trashed = [n for n in os.listdir(self._path(".trash")) if n.endswith(".json")]
        states = sorted(self._read_state(self._path(".trash", n))["Points"][0]["name"] for n in trashed)
        self.assertEqual(states, ["A", "B", "C"])

    def test_delete_moves_backup_to_trash(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))
        self.assertTrue(self.manager.save_workspace(_state("B"), "ws"))
        self.assertTrue(os.path.exists(self._path("ws.json.bak")))

        self.assertTrue(self.manager.delete_workspace("ws"))

        self.assertFalse(os.path.exists(self._path("ws.json.bak")))
        backups = [n for n in os.listdir(self._path(".trash")) if n.endswith(".json.bak")]
        self.assertEqual(len(backups), 1)
        self.assertEqual(self._read_state(self._path(".trash", backups[0])), _state("A"))

    def test_save_retries_when_target_is_briefly_locked(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))
        real_replace = os.replace
        calls: List[int] = []

        def flaky_replace(src: str, dst: str) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise PermissionError(5, "Access is denied")
            real_replace(src, dst)

        with (
            mock.patch("static.workspace_manager.os.replace", side_effect=flaky_replace),
            mock.patch("static.workspace_manager.time.sleep"),
        ):
            self.assertTrue(self.manager.save_workspace(_state("B"), "ws"))

        self.assertEqual(len(calls), 2)
        self.assertEqual(self._read_state(self._path("ws.json")), _state("B"))

    def test_delete_in_test_dir_uses_local_trash(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws", "Sub"))

        self.assertTrue(self.manager.delete_workspace("ws", "Sub"))

        self.assertEqual(len(os.listdir(self._path("Sub", ".trash"))), 1)
        self.assertFalse(os.path.exists(self._path(".trash")))

    def test_list_workspaces_ignores_backups_and_trash(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "keep"))
        self.assertTrue(self.manager.save_workspace(_state("B"), "keep"))
        self.assertTrue(self.manager.save_workspace(_state("C"), "gone"))
        self.assertTrue(self.manager.delete_workspace("gone"))

        self.assertTrue(os.path.exists(self._path("keep.json.bak")))
        self.assertEqual(self.manager.list_workspaces(), ["keep"])

    def test_deleted_workspace_cannot_be_loaded(self) -> None:
        self.assertTrue(self.manager.save_workspace(_state("A"), "ws"))
        self.assertTrue(self.manager.delete_workspace("ws"))

        with self.assertRaises(FileNotFoundError):
            self.manager.load_workspace("ws")


if __name__ == "__main__":
    unittest.main()
