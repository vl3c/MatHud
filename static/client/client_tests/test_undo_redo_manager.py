from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from managers.undo_redo_manager import UndoRedoManager
from .simple_mock import SimpleMock


class TestUndoRedoManager(unittest.TestCase):
    def setUp(self) -> None:
        self.drawables = SimpleMock(
            _drawables={"Points": [], "Segments": []},
            rebuild_renderables=MagicMock(),
        )
        self.dependency_manager = SimpleMock(
            _parents={},
            _children={},
            analyze_drawable_for_dependencies=MagicMock(),
        )
        self.canvas = SimpleMock(
            drawable_manager=SimpleMock(drawables=self.drawables),
            computations=[],
            dependency_manager=self.dependency_manager,
            draw=MagicMock(),
        )
        self.manager = UndoRedoManager(self.canvas)

    def test_archive_is_noop_while_suspended(self) -> None:
        self.manager.suspend_archiving()
        self.manager.archive()
        self.assertEqual(len(self.manager.undo_stack), 0)

        self.manager.resume_archiving()
        self.manager.archive()
        self.assertEqual(len(self.manager.undo_stack), 1)

    def test_push_undo_state_deep_copies_and_clears_redo(self) -> None:
        self.manager.redo_stack = [{"drawables": {}, "computations": []}]
        state = {"drawables": {"Points": [SimpleMock(name="A")]}, "computations": [1]}

        self.manager.push_undo_state(state)
        state["computations"].append(2)

        self.assertEqual(len(self.manager.undo_stack), 1)
        stored = self.manager.undo_stack[0]
        self.assertEqual(stored["computations"], [1])
        self.assertEqual(self.manager.redo_stack, [])

    def test_restore_state_rebuilds_dependencies_and_redraws(self) -> None:
        p1 = SimpleMock(name="P1")
        s1 = SimpleMock(name="S1")
        state = {
            "drawables": {"Points": [p1], "Segments": [s1]},
            "computations": [{"expression": "2+2", "result": 4}],
        }

        self.manager.restore_state(state)

        self.drawables.rebuild_renderables.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.assertEqual(self.canvas.computations, [{"expression": "2+2", "result": 4}])
        self.assertEqual(self.dependency_manager.analyze_drawable_for_dependencies.call_count, 2)

    def test_capture_state_returns_deep_copy(self) -> None:
        self.canvas.computations = [{"expression": "x", "result": 1}]
        point = SimpleMock(name="P")
        self.drawables._drawables = {"Points": [point]}

        snapshot = self.manager.capture_state()
        self.canvas.computations[0]["result"] = 99

        self.assertEqual(snapshot["computations"][0]["result"], 1)

    def test_batch_collapses_archives_into_one_entry_with_the_baseline(self) -> None:
        self.canvas.computations = ["before"]
        self.manager.redo_stack = [{"drawables": {}, "computations": []}]

        self.manager.begin_batch()
        self.canvas.computations = ["during"]
        self.manager.archive()
        self.manager.archive()
        self.manager.push_undo_state({"drawables": {}, "computations": ["inner"]})
        self.assertEqual(self.manager.undo_stack, [])
        self.manager.end_batch()

        self.assertEqual(len(self.manager.undo_stack), 1)
        self.assertEqual(self.manager.undo_stack[0]["computations"], ["before"])
        self.assertEqual(self.manager.redo_stack, [])

    def test_batch_without_changes_adds_no_entry(self) -> None:
        self.manager.redo_stack = [{"drawables": {}, "computations": []}]

        self.manager.begin_batch()
        self.manager.end_batch()

        self.assertEqual(self.manager.undo_stack, [])
        self.assertEqual(len(self.manager.redo_stack), 1)

    def test_nested_batches_commit_once_at_the_outermost_end(self) -> None:
        self.manager.begin_batch()
        self.manager.begin_batch()
        self.manager.archive()
        self.manager.end_batch()
        self.assertEqual(self.manager.undo_stack, [])

        self.manager.end_batch()
        self.assertEqual(len(self.manager.undo_stack), 1)

    def test_undo_inside_a_batch_reverts_the_batch_changes_so_far(self) -> None:
        self.canvas.computations = ["before"]
        self.manager.begin_batch()
        self.canvas.computations = ["during"]
        self.manager.archive()

        self.assertTrue(self.manager.undo())
        self.manager.end_batch()

        self.assertEqual(self.manager.undo_stack, [])
        self.assertEqual(len(self.manager.redo_stack), 1)

    def test_batch_change_mark_can_be_reset(self) -> None:
        self.assertFalse(self.manager.is_batch_changed())
        self.manager.begin_batch()
        self.manager.archive()
        self.assertTrue(self.manager.is_batch_changed())

        self.manager.set_batch_changed(False)
        self.manager.end_batch()

        self.assertEqual(self.manager.undo_stack, [])

    def test_state_comparison_with_the_batch_baseline(self) -> None:
        self.assertTrue(self.manager.state_differs_from_batch_baseline())
        self.canvas.computations = [{"expression": "x", "result": 1}]
        self.manager.begin_batch()
        self.assertFalse(self.manager.state_differs_from_batch_baseline())

        self.canvas.computations = [{"expression": "x", "result": 2}]

        self.assertTrue(self.manager.state_differs_from_batch_baseline())
        self.manager.end_batch()

    def test_archive_outside_a_batch_is_unchanged(self) -> None:
        self.manager.begin_batch()
        self.manager.end_batch()

        self.manager.archive()
        self.manager.archive()

        self.assertEqual(len(self.manager.undo_stack), 2)
