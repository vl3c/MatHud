"""
MatHud Undo/Redo State Management System

Provides comprehensive undo and redo functionality for canvas operations through state archiving
and restoration. Maintains operation history and handles complex object relationships during
state transitions.

State Management Architecture:
    - Snapshot System: Deep copying of entire canvas state for each operation
    - Dual Stack System: Separate undo and redo stacks for bidirectional navigation
    - Automatic Archiving: State capture before any destructive operation
    - State Restoration: Complete canvas reconstruction from archived states

Archived State Components:
    - Drawable Objects: Complete deep copy of all geometric objects and their properties
    - Computation History: Mathematical operation results and expressions
    - View: Zoom and pan, coordinate-system mode, grid visibility and grid spacing
      (``Canvas.get_view_state``). Mouse panning and zooming do not archive, and undoing
      a step that did not change the view keeps the current view.
    - Object Relationships: Preservation of parent-child dependencies
    - Canvas References: Proper object-to-canvas relationship maintenance

Operation Flow:
    - archive(): Captures current state before modifications
    - undo(): Restores previous state and moves current to redo stack
    - redo(): Restores next state and moves current to undo stack
    - State clearing: Automatic redo stack clearing on new operations

Complex State Handling:
    - Dependency Rebuilding: _rebuild_dependency_graph() recreates object relationships
    - Deep Copy Management: Handles nested object structures and circular references
    - Memory Efficiency: Strategic state limitation to prevent memory bloat

Integration Points:
    - DrawableManager: State capture of all drawable objects
    - DrawableDependencyManager: Dependency graph reconstruction
    - Canvas: Automatic redrawing after state changes
    - Mathematical Operations: Computation history preservation

Error Recovery:
    - Graceful Degradation: Continues operation even if some references can't be restored
    - Validation: Checks for required manager instances before operations
    - Logging: Comprehensive warning system for debugging state issues
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from canvas import Canvas

# Entry key: False when the step the entry undoes did not change the view, so undoing it
# (and redoing it) keeps the current view, e.g. after the user panned with the mouse.
# A missing key means unknown, and the entry's view is restored.
RESTORES_VIEW_KEY = "restores_view"
# View keys that follow from the zoom level: restored, but left out of comparisons.
_DERIVED_VIEW_KEYS = ("grid_spacing",)


class UndoRedoManager:
    """
    Manages undo and redo operations for a Canvas object.

    This class is responsible for:
    - Archiving canvas states (for undo operations)
    - Handling undo operations (restore previous state)
    - Handling redo operations (restore undone state)
    """

    def __init__(self, canvas: "Canvas") -> None:
        """
        Initialize the UndoRedoManager.

        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas: "Canvas" = canvas
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        self._archive_suspension_depth: int = 0
        self._batch_depth: int = 0
        self._batch_baseline: Optional[Dict[str, Any]] = None
        self._batch_signature: Optional[str] = None
        self._batch_changed: bool = False

    def archive(self) -> None:
        """
        Archives the current state of the canvas for undo operations.

        This method should be called whenever a change is made to the canvas
        that should be undoable. Inside an undo batch it only records that the
        batch changed something; the batch pushes one entry when it ends.
        """
        if self._archive_suspension_depth > 0:
            return
        if self._batch_depth > 0:
            self._batch_changed = True
            return
        self.push_undo_state(self.capture_state())

    def capture_state(self) -> Dict[str, Any]:
        """Capture the current canvas state snapshot: drawables, computations and the view."""
        state: Dict[str, Any] = {
            "drawables": copy.deepcopy(self.canvas.drawable_manager.drawables._drawables),
            "computations": copy.deepcopy(self.canvas.computations),
        }
        view = self._capture_view()
        if view is not None:
            state["view"] = view
        return state

    def _capture_view(self) -> Optional[Dict[str, Any]]:
        """The canvas view, or None for a canvas without one (test doubles)."""
        get_view_state = getattr(self.canvas, "get_view_state", None)
        if not callable(get_view_state):
            return None
        return cast(Dict[str, Any], get_view_state())

    def push_undo_state(self, state: Dict[str, Any]) -> None:
        """Push a prior state onto the undo stack and clear redo history.

        Inside an undo batch the batch's own baseline is kept instead, so a
        composite operation within the batch does not add a separate entry.
        """
        if self._batch_depth > 0:
            self._batch_changed = True
            return
        self.undo_stack.append(copy.deepcopy(state))
        self.redo_stack = []

    def restore_state(self, state: Dict[str, Any], redraw: bool = True) -> None:
        """Restore a captured state snapshot, including its view."""
        self._apply_state(state, restore_view=True, redraw=redraw)

    def _apply_state(self, state: Dict[str, Any], restore_view: bool, redraw: bool) -> None:
        """Replace the drawables and computations with the state's, and optionally the view."""
        self.canvas.drawable_manager.drawables._drawables = copy.deepcopy(state["drawables"])
        self.canvas.drawable_manager.drawables.rebuild_renderables()
        self.canvas.computations = copy.deepcopy(state.get("computations", []))
        zoom_changed = restore_view and self._restore_view(state)
        self._rebuild_dependency_graph()
        if redraw:
            self.canvas.draw(apply_zoom=zoom_changed)

    def _restore_view(self, state: Dict[str, Any]) -> bool:
        """Apply the state's view, if it has one; return True when the zoom level changed."""
        view = state.get("view")
        restore_view_state = getattr(self.canvas, "restore_view_state", None)
        if view is None or not callable(restore_view_state):
            return False
        return bool(restore_view_state(view))

    def suspend_archiving(self) -> None:
        """Suspend archive() calls for composite operations."""
        self._archive_suspension_depth += 1

    def resume_archiving(self) -> None:
        """Resume archive() calls after a composite operation."""
        if self._archive_suspension_depth > 0:
            self._archive_suspension_depth -= 1

    def begin_batch(self) -> None:
        """Start grouping every change until the matching end_batch() into one undo step.

        Batches nest; only the outermost one captures the baseline and pushes the entry.
        """
        if self._batch_depth == 0:
            self._start_batch_from_current_state()
        self._batch_depth += 1

    def end_batch(self) -> None:
        """Close a batch; the outermost close pushes one undo entry if anything changed."""
        if self._batch_depth == 0:
            return
        self._batch_depth -= 1
        if self._batch_depth == 0:
            self._commit_batch()
            self._batch_baseline = None
            self._batch_signature = None

    def is_batch_changed(self) -> bool:
        """Return True when the open batch has recorded a change."""
        return self._batch_depth > 0 and self._batch_changed

    def set_batch_changed(self, changed: bool) -> None:
        """Overwrite the open batch's change mark, e.g. to drop the mark of a call that failed."""
        if self._batch_depth > 0:
            self._batch_changed = changed

    def state_differs_from_batch_baseline(self) -> bool:
        """Return True when the canvas no longer matches the open batch's baseline.

        Compares each live drawable's ``get_state()``, the computations and the view (zoom,
        pan, coordinate mode and grid visibility), serialized as sorted JSON, with the same
        serialization of the live objects taken when the batch started. The deep-copied
        baseline is not used: copying rebuilds some drawables (a polygon recomputes its
        types), so it can serialize differently from the unchanged live objects. Outside a batch, or when a state cannot be serialized, the canvas is
        assumed to differ so that a change is never dropped from the undo history.
        """
        if self._batch_depth == 0 or self._batch_signature is None:
            return True
        current = self._live_signature()
        return current is None or current != self._batch_signature

    def _start_batch_from_current_state(self) -> None:
        """Take the batch baseline (what undo restores) and the live signature (what comparisons use)."""
        self._batch_baseline = self.capture_state()
        self._batch_signature = self._live_signature()
        self._batch_changed = False

    def _live_signature(self) -> Optional[str]:
        """Serialized live state, or None when it cannot be serialized."""
        try:
            return self._serialize_state(self._live_state())
        except Exception:
            return None

    def _live_state(self) -> Dict[str, Any]:
        """The live drawables, computations and view, without copying (for comparison only)."""
        return {
            "drawables": self.canvas.drawable_manager.drawables._drawables,
            "computations": self.canvas.computations,
            "view": self._capture_view(),
        }

    @staticmethod
    def _serialize_state(state: Dict[str, Any]) -> str:
        drawables = {
            bucket: [drawable.get_state() for drawable in items]
            for bucket, items in state["drawables"].items()
            if items
        }
        payload = {
            "drawables": drawables,
            "computations": state.get("computations", []),
            "view": UndoRedoManager._comparable_view(state.get("view")),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _comparable_view(view: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The view without the keys that follow from the zoom level."""
        if view is None:
            return None
        return {key: value for key, value in view.items() if key not in _DERIVED_VIEW_KEYS}

    def _view_changed_since(self, state: Dict[str, Any]) -> bool:
        """True when the live view differs from the state's view, or when either is unknown."""
        before = self._comparable_view(state.get("view"))
        after = self._comparable_view(self._capture_view())
        if before is None or after is None:
            return True
        return json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)

    def _commit_batch(self) -> None:
        """Push the batch baseline as one undo entry when the batch changed something.

        The entry records whether the batch changed the view, so undoing a batch that only
        changed objects keeps the view the user has now.
        """
        if not self._batch_changed or self._batch_baseline is None:
            return
        self._batch_baseline[RESTORES_VIEW_KEY] = self._view_changed_since(self._batch_baseline)
        self.undo_stack.append(self._batch_baseline)
        self.redo_stack = []
        self._batch_changed = False

    def _rebase_batch(self) -> None:
        """After an undo or redo inside a batch, later changes start from the restored state."""
        if self._batch_depth > 0:
            self._start_batch_from_current_state()

    def undo(self) -> bool:
        """
        Restores the last archived state from the undo stack.

        Inside a batch, changes made earlier in the batch are committed first, so
        undo reverts them rather than the step before the batch.

        Returns:
            bool: True if an undo was performed, False otherwise
        """
        self._commit_batch()
        if not self.undo_stack:
            return False

        last_state = self.undo_stack.pop()
        self.redo_stack.append(self._capture_counterpart(last_state))
        self._restore_history_entry(last_state)

        self._rebase_batch()
        return True

    def redo(self) -> bool:
        """
        Restores the last undone state from the redo stack.

        Inside a batch, changes made earlier in the batch are committed first,
        which clears the redo history as any new change does.

        Returns:
            bool: True if a redo was performed, False otherwise
        """
        self._commit_batch()
        if not self.redo_stack:
            return False

        next_state = self.redo_stack.pop()
        self.undo_stack.append(self._capture_counterpart(next_state))
        self._restore_history_entry(next_state)

        self._rebase_batch()
        return True

    def _capture_counterpart(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """The current state, for the opposite stack; it restores the view only if ``entry`` does."""
        current = self.capture_state()
        current[RESTORES_VIEW_KEY] = self._entry_restores_view(entry)
        return current

    def _restore_history_entry(self, entry: Dict[str, Any]) -> None:
        """Restore an undo or redo entry: its objects always, its view when its step changed the view."""
        self._apply_state(entry, restore_view=self._entry_restores_view(entry), redraw=True)

    @staticmethod
    def _entry_restores_view(entry: Dict[str, Any]) -> bool:
        return bool(entry.get(RESTORES_VIEW_KEY, True))

    def can_undo(self) -> bool:
        """
        Checks if there are any states that can be undone.

        Returns:
            bool: True if undo is possible, False otherwise
        """
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """
        Checks if there are any states that can be redone.

        Returns:
            bool: True if redo is possible, False otherwise
        """
        return len(self.redo_stack) > 0

    def _rebuild_dependency_graph(self) -> None:
        """
        Rebuilds the dependency relationships between drawables.

        This is necessary after loading a saved state, as the serialization
        process may lose dependency links which need to be re-established
        with the new object instances.
        """
        all_drawables = []
        # Collect all drawable objects from the restored state
        for drawable_type in self.canvas.drawable_manager.drawables._drawables:
            for drawable in self.canvas.drawable_manager.drawables._drawables[drawable_type]:
                all_drawables.append(drawable)

        # This assumes self.canvas has a 'dependency_manager' attribute
        # which is an instance of DrawableDependencyManager.
        if hasattr(self.canvas, "dependency_manager") and self.canvas.dependency_manager is not None:
            dependency_manager = self.canvas.dependency_manager

            # Clear existing dependency relationships from the manager
            dependency_manager._parents.clear()
            dependency_manager._children.clear()

            # Re-analyze each drawable to rebuild the dependency graph
            # The analyze_drawable_for_dependencies method in DrawableDependencyManager
            # is responsible for handling various drawable types and their specific dependencies.
            for drawable in all_drawables:
                dependency_manager.analyze_drawable_for_dependencies(drawable)
        else:
            # Log a warning if the dependency manager isn't found on the canvas object.
            # This helps in debugging if the expected structure isn't met.
            print(
                "UndoRedoManager: Warning - Canvas instance does not have a 'dependency_manager' "
                + "attribute or it is None. Skipping dependency graph rebuild for undo/redo operation."
            )

    def clear(self) -> None:
        """
        Clears all undo and redo history.
        """
        self.undo_stack = []
        self.redo_stack = []
