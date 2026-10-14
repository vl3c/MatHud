"""Tests for the browser-storage canvas backup (``CanvasBackupManager``).

Storage and timers are test doubles, so the tests never touch the page's own
``localStorage`` backup.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

from canvas import Canvas
from canvas_backup_manager import (
    BACKUP_DELAY_MS,
    BACKUP_STORAGE_KEY,
    BACKUP_VERSION,
    UNRESTORED_STORAGE_KEY,
    CanvasBackupManager,
)
from workspace_manager import WorkspaceManager


class FakeStorage:
    """A ``localStorage`` double; ``refuse_writes`` makes setItem raise like a full storage."""

    def __init__(self, items: Optional[Dict[str, str]] = None) -> None:
        self.items: Dict[str, str] = dict(items or {})
        self.writes: List[str] = []
        self.refuse_writes = False

    def getItem(self, key: str) -> Optional[str]:
        return self.items.get(key)

    def setItem(self, key: str, value: str) -> None:
        if self.refuse_writes:
            raise RuntimeError("QuotaExceededError")
        self.writes.append(key)
        self.items[key] = value

    def removeItem(self, key: str) -> None:
        self.items.pop(key, None)


class FakeTimers:
    """Records scheduled callbacks; ``run_all`` fires them."""

    def __init__(self) -> None:
        self.scheduled: List[Tuple[Callable[[], None], int]] = []

    def set_timeout(self, callback: Callable[[], None], delay_ms: int) -> int:
        self.scheduled.append((callback, delay_ms))
        return len(self.scheduled)

    def run_all(self) -> None:
        pending, self.scheduled = self.scheduled, []
        for callback, _delay in pending:
            callback()


def _new_canvas() -> Canvas:
    return Canvas(500, 500, draw_enabled=False)


def _backup_for(canvas: Canvas, storage: FakeStorage, timers: Optional[FakeTimers] = None) -> CanvasBackupManager:
    timers = timers or FakeTimers()
    return CanvasBackupManager(canvas, WorkspaceManager(canvas), storage=storage, set_timeout=timers.set_timeout)


def _point_names(canvas: Canvas) -> List[str]:
    return sorted(point.name for point in canvas.drawable_manager.drawables.Points)


class TestCanvasBackup(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = FakeStorage()
        self.timers = FakeTimers()
        self.canvas = _new_canvas()
        self.backup = _backup_for(self.canvas, self.storage, self.timers)

    def _stored_backup(self) -> Dict[str, Any]:
        return dict(json.loads(self.storage.items[BACKUP_STORAGE_KEY]))

    def test_backup_stores_the_canvas_state_and_view(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.assertTrue(self.backup.backup_now())
        stored = self._stored_backup()
        self.assertEqual(stored["version"], BACKUP_VERSION)
        self.assertEqual([p["name"] for p in stored["state"]["Points"]], ["A"])
        self.assertEqual(stored["view"]["scale_factor"], self.canvas.get_view_state()["scale_factor"])

    def test_restore_brings_back_drawables_view_and_computations_without_undo_history(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.canvas.create_point(3, 4, name="B")
        self.canvas.create_segment(1, 2, 3, 4)
        self.canvas.add_computation("1+1", 2)
        self.canvas.restore_view_state({"scale_factor": 2.5, "offset": [30.0, -40.0], "cartesian_grid_visible": False})
        self.backup.backup_now()

        reloaded = _new_canvas()
        restored = _backup_for(reloaded, self.storage).restore()

        self.assertTrue(restored)
        self.assertEqual(_point_names(reloaded), ["A", "B"])
        self.assertEqual(len(reloaded.drawable_manager.drawables.Segments), 1)
        self.assertEqual([c["expression"] for c in reloaded.computations], ["1+1"])
        view = reloaded.get_view_state()
        self.assertEqual(view["scale_factor"], 2.5)
        self.assertEqual(view["offset"], [30.0, -40.0])
        self.assertFalse(view["cartesian_grid_visible"])
        self.assertEqual(reloaded.undo_redo_manager.undo_stack, [])
        self.assertEqual(reloaded.undo_redo_manager.redo_stack, [])

    def test_restore_without_a_backup_leaves_the_canvas_alone(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        undo_depth = len(self.canvas.undo_redo_manager.undo_stack)
        self.assertFalse(self.backup.restore())
        self.assertEqual(_point_names(self.canvas), ["A"])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), undo_depth)

    def test_unreadable_backup_is_kept_aside_and_not_restored(self) -> None:
        for raw in ("{not json", json.dumps({"version": BACKUP_VERSION + 1, "state": {}}), json.dumps({"version": 1})):
            storage = FakeStorage({BACKUP_STORAGE_KEY: raw})
            canvas = _new_canvas()
            self.assertFalse(_backup_for(canvas, storage).restore())
            self.assertEqual(storage.items[UNRESTORED_STORAGE_KEY], raw)
            self.assertEqual(_point_names(canvas), [])

    def test_failed_restore_backs_up_again_after_keeping_the_old_copy(self) -> None:
        storage = FakeStorage({BACKUP_STORAGE_KEY: "{not json"})
        canvas = _new_canvas()
        backup = _backup_for(canvas, storage)
        backup.restore()
        canvas.create_point(5, 5, name="C")
        self.assertTrue(backup.backup_now())
        self.assertEqual(storage.items[UNRESTORED_STORAGE_KEY], "{not json")
        self.assertIn('"C"', storage.items[BACKUP_STORAGE_KEY])

    def test_items_skipped_by_the_restore_are_kept_aside(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.backup_now()
        stored = self._stored_backup()
        # A point the restore cannot rebuild (no position): it is skipped with a warning.
        stored["state"]["Points"].append({"name": "Z", "args": {}})
        raw = json.dumps(stored)
        storage = FakeStorage({BACKUP_STORAGE_KEY: raw})

        reloaded = _new_canvas()
        backup = _backup_for(reloaded, storage)
        self.assertTrue(backup.restore())
        self.assertEqual(_point_names(reloaded), ["A"])
        self.assertEqual(storage.items[UNRESTORED_STORAGE_KEY], raw)
        self.assertTrue(backup.backup_now())
        self.assertEqual(storage.items[UNRESTORED_STORAGE_KEY], raw)
        self.assertNotEqual(storage.items[BACKUP_STORAGE_KEY], raw)

    def test_restore_brings_back_polar_mode_on_a_drawing_canvas(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.canvas.coordinate_system_manager.set_mode("polar", redraw=False)
        self.canvas.restore_view_state({"scale_factor": 0.5, "offset": [10.0, 0.0]})
        self.backup.backup_now()

        reloaded = _new_canvas()
        reloaded.renderer = None
        reloaded.draw_enabled = True
        self.assertTrue(_backup_for(reloaded, FakeStorage(dict(self.storage.items))).restore())
        view = reloaded.get_view_state()
        self.assertEqual(view["coordinate_mode"], "polar")
        self.assertEqual(view["scale_factor"], 0.5)
        self.assertEqual(_point_names(reloaded), ["A"])

    def test_tab_whose_backup_another_tab_replaced_writes_only_after_its_canvas_changes(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.backup_now()
        newer = '{"version": 1, "state": {}, "view": {}}'
        self.storage.items[BACKUP_STORAGE_KEY] = newer
        self.backup.mark_superseded()

        self.canvas.restore_view_state({"scale_factor": 3.0, "offset": [0.0, 0.0]})  # a pan or zoom
        self.assertFalse(self.backup.backup_now())
        self.assertEqual(self.storage.items[BACKUP_STORAGE_KEY], newer)

        self.canvas.create_point(3, 4, name="B")
        self.assertTrue(self.backup.backup_now())
        self.assertEqual(sorted(p["name"] for p in self._stored_backup()["state"]["Points"]), ["A", "B"])

    def test_unchanged_canvas_is_not_written_again(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.backup_now()
        self.backup.backup_now()
        self.assertEqual(self.storage.writes, [BACKUP_STORAGE_KEY])
        self.canvas.create_point(3, 4, name="B")
        self.backup.backup_now()
        self.assertEqual(self.storage.writes, [BACKUP_STORAGE_KEY, BACKUP_STORAGE_KEY])

    def test_restored_backup_is_not_rewritten_until_the_canvas_changes(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.backup_now()
        reloaded = _new_canvas()
        storage = FakeStorage(dict(self.storage.items))
        backup = _backup_for(reloaded, storage)
        backup.restore()
        backup.backup_now()
        self.assertEqual(storage.writes, [])

    def test_schedule_backup_writes_once_after_the_delay(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.schedule_backup()
        self.backup.schedule_backup()
        self.assertEqual([delay for _callback, delay in self.timers.scheduled], [BACKUP_DELAY_MS])
        self.assertNotIn(BACKUP_STORAGE_KEY, self.storage.items)
        self.timers.run_all()
        self.assertEqual([p["name"] for p in self._stored_backup()["state"]["Points"]], ["A"])
        self.backup.schedule_backup()
        self.assertEqual(len(self.timers.scheduled), 1)

    def test_refused_write_removes_the_stale_backup(self) -> None:
        self.canvas.create_point(1, 2, name="A")
        self.backup.backup_now()
        self.storage.refuse_writes = True
        self.canvas.create_point(3, 4, name="B")
        self.assertFalse(self.backup.backup_now())
        self.assertNotIn(BACKUP_STORAGE_KEY, self.storage.items)
        self.storage.refuse_writes = False
        self.assertTrue(self.backup.backup_now())
        self.assertEqual(
            sorted(p["name"] for p in self._stored_backup()["state"]["Points"]),
            ["A", "B"],
        )

    def test_canvas_draw_schedules_a_backup(self) -> None:
        self.canvas.renderer = None
        self.canvas.draw_enabled = True
        self.canvas.add_draw_listener(self.backup.schedule_backup)
        self.canvas.draw()
        self.assertEqual(len(self.timers.scheduled), 1)

    def test_failing_draw_listener_does_not_break_drawing(self) -> None:
        calls: List[str] = []

        def failing() -> None:
            raise RuntimeError("boom")

        self.canvas.renderer = None
        self.canvas.draw_enabled = True
        self.canvas.add_draw_listener(failing)
        self.canvas.add_draw_listener(lambda: calls.append("after"))
        self.canvas.draw()
        self.assertEqual(calls, ["after"])


if __name__ == "__main__":
    unittest.main()
