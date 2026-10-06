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
      a step restores only the parts of the view it changed (zoom and pan as one part,
      the coordinate mode, each grid's visibility).
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, cast

if TYPE_CHECKING:
    from canvas import Canvas

# The view (Canvas.get_view_state) in parts that undo restores independently. Zoom and pan
# are one part; the grid spacing follows the zoom level, so it is restored with it but
# never compared.
VIEW_PARTS: Dict[str, Tuple[str, ...]] = {
    "zoom": ("scale_factor", "offset", "grid_spacing"),
    "coordinate_mode": ("coordinate_mode",),
    "cartesian_grid": ("cartesian_grid_visible",),
    "polar_grid": ("polar_grid_visible",),
}
_DERIVED_VIEW_KEYS = ("grid_spacing",)
# Entry key: the view parts the entry's step changed. Undoing (and redoing) the step
# restores only those parts and keeps the rest as the user has them now, e.g. a grid
# toggle made afterwards. A missing key means unknown: every part is restored.
VIEW_CHANGES_KEY = "view_changes"
# Drawable attributes outside get_state() that update tools change (colour, attached labels).
_APPEARANCE_ATTRIBUTES = ("color", "opacity", "font_size", "rotation_degrees", "text", "visible")


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
        # View tracking for a batch that spans user activity (a chat turn's undo group):
        # only the view parts its own tool batches changed belong to its step, so a pan or
        # zoom the user makes between the batches is neither undone with it nor counted as
        # its change. Off (False) for ordinary batches, which compare the whole view.
        self._view_tracking: bool = False
        self._tracked_view_parts: Set[str] = set()
        self._view_mark: Optional[Dict[str, Any]] = None
        # The open batch's serialization without the view, and its comparable view.
        self._batch_core_signature: Optional[str] = None
        self._batch_view: Optional[Dict[str, Any]] = None

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
        """Restore a captured state snapshot, including its whole view."""
        self._apply_state(state, tuple(VIEW_PARTS), redraw=redraw)

    def _apply_state(self, state: Dict[str, Any], view_parts: Sequence[str], redraw: bool) -> None:
        """Replace the drawables and computations with the state's, and the given parts of its view."""
        self.canvas.drawable_manager.drawables._drawables = copy.deepcopy(state["drawables"])
        self.canvas.drawable_manager.drawables.rebuild_renderables()
        self.canvas.computations = copy.deepcopy(state.get("computations", []))
        zoom_changed = self._restore_view(state, view_parts)
        self._rebuild_dependency_graph()
        if redraw:
            self.canvas.draw(apply_zoom=zoom_changed)

    def _restore_view(self, state: Dict[str, Any], view_parts: Sequence[str]) -> bool:
        """Apply the given parts of the state's view, if it has one; True when the zoom level changed."""
        view = state.get("view")
        restore_view_state = getattr(self.canvas, "restore_view_state", None)
        if view is None or not view_parts or not callable(restore_view_state):
            return False
        keys = {key for part in view_parts for key in VIEW_PARTS.get(part, ())}
        return bool(restore_view_state({key: value for key, value in view.items() if key in keys}))

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
            self._view_tracking = False
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
            self._batch_core_signature = None
            self._batch_view = None
            self._view_tracking = False

    def track_batch_view(self) -> None:
        """Count only the view changes made between ``mark_batch_view`` and ``note_batch_view``.

        For a batch that stays open while the user works (a chat turn's undo group):
        a pan, zoom or grid toggle the user makes between the marked stretches is not
        the batch's change, so it neither makes the batch count as changed nor is
        undone with it. Drawables are compared as usual. Ignored outside a batch.
        """
        if self._batch_depth == 0:
            return
        self._view_tracking = True
        self._tracked_view_parts = set()
        self._view_mark = self._comparable_view(self._capture_view())

    def mark_batch_view(self) -> None:
        """Start a stretch whose view changes belong to the tracked batch (before a tool batch)."""
        if self._view_tracking:
            self._view_mark = self._comparable_view(self._capture_view())

    def note_batch_view(self) -> None:
        """End the stretch: the view parts it changed become the tracked batch's (after a tool batch)."""
        if self._view_tracking:
            self._tracked_view_parts.update(self._parts_changed_since_mark())
            self._view_mark = None

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
        baseline is not used: copying rebuilds each drawable, so it need not serialize
        exactly like the unchanged live objects. Each drawable's colour and attached label
        count too, since ``get_state()`` leaves them out. Outside a batch, or when a state
        cannot be serialized, the canvas is assumed to differ so that a change is never
        dropped from the undo history.
        """
        if self._batch_depth == 0 or self._batch_signature is None:
            return True
        if self._view_tracking:
            return not self._matches_batch_start()
        current = self._live_signature()
        return current is None or current != self._batch_signature

    def _start_batch_from_current_state(self) -> None:
        """Take the batch baseline (what undo restores) and the live signature (what comparisons use)."""
        self._batch_baseline = self.capture_state()
        self._batch_signature = self._live_signature()
        self._batch_core_signature = self._live_signature(with_view=False)
        self._batch_view = self._comparable_view(self._capture_view())
        self._batch_changed = False
        if self._view_tracking:
            # After an undo or redo inside a tracked batch the new group starts here.
            self._tracked_view_parts = set()
            self._view_mark = self._batch_view

    def _live_signature(self, with_view: bool = True) -> Optional[str]:
        """Serialized live state (optionally without the view), or None when it cannot be serialized."""
        try:
            state = self._live_state()
            if not with_view:
                state["view"] = None
            return self._serialize_state(state)
        except Exception:
            return None

    def _parts_changed_since_mark(self) -> Set[str]:
        """View parts changed since ``mark_batch_view`` (none when no stretch is open)."""
        if self._view_mark is None:
            return set()
        return set(self._view_parts_between(self._view_mark, self._comparable_view(self._capture_view())))

    def _owned_view_parts(self) -> List[str]:
        """The view parts the tracked batch changed, in ``VIEW_PARTS`` order."""
        owned = self._tracked_view_parts | self._parts_changed_since_mark()
        return [part for part in VIEW_PARTS if part in owned]

    def _matches_batch_start(self) -> bool:
        """Tracked batch: drawables and computations as at the start, and so are the view parts it owns."""
        if self._batch_core_signature is None:
            return False
        current = self._live_signature(with_view=False)
        if current is None or current != self._batch_core_signature:
            return False
        before, after = self._batch_view, self._comparable_view(self._capture_view())
        if before is None or after is None:
            return before is None and after is None
        keys = [key for part in self._owned_view_parts() for key in VIEW_PARTS[part] if key not in _DERIVED_VIEW_KEYS]
        return all(before.get(key) == after.get(key) for key in keys)

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
            bucket: [UndoRedoManager._comparable_drawable(drawable) for drawable in items]
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
    def _comparable_drawable(drawable: Any) -> Dict[str, Any]:
        """``get_state()`` plus the appearance it leaves out: colour and the attached label."""
        comparable: Dict[str, Any] = {
            "state": drawable.get_state(),
            "appearance": UndoRedoManager._appearance(drawable),
        }
        label = getattr(drawable, "label", None)
        if label is not None:
            comparable["label"] = UndoRedoManager._appearance(label)
        return comparable

    @staticmethod
    def _appearance(item: Any) -> Dict[str, Any]:
        appearance: Dict[str, Any] = {}
        for attribute in _APPEARANCE_ATTRIBUTES:
            value = getattr(item, attribute, None)
            if value is None or isinstance(value, (str, bool, int, float)):
                appearance[attribute] = value
        return appearance

    @staticmethod
    def _comparable_view(view: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The view without the keys that follow from the zoom level, reading -0.0 as 0.0."""
        if view is None:
            return None
        return {
            key: UndoRedoManager._without_negative_zero(value)
            for key, value in view.items()
            if key not in _DERIVED_VIEW_KEYS
        }

    @staticmethod
    def _without_negative_zero(value: Any) -> Any:
        if isinstance(value, float):
            return value + 0.0
        if isinstance(value, (list, tuple)):
            return [UndoRedoManager._without_negative_zero(item) for item in value]
        return value

    def _changed_view_parts(self, state: Dict[str, Any]) -> List[str]:
        """The view parts in which the live view differs from the state's; all of them when either is unknown."""
        return self._view_parts_between(
            self._comparable_view(state.get("view")), self._comparable_view(self._capture_view())
        )

    @staticmethod
    def _view_parts_between(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> List[str]:
        """The view parts in which two comparable views differ; all of them when either is unknown."""
        if before is None or after is None:
            return list(VIEW_PARTS)
        return [part for part, keys in VIEW_PARTS.items() if any(before.get(key) != after.get(key) for key in keys)]

    def _commit_batch(self) -> None:
        """Push the batch baseline as one undo entry when the batch changed something.

        A batch marked changed that left the canvas as it found it (a clear of an empty
        canvas, a zoom and back) pushes nothing and keeps the redo history. The entry
        records which view parts the batch changed, so undoing it keeps the other parts
        as the user has them then. A tracked batch (``track_batch_view``) records only the
        parts its own stretches changed, never a pan or zoom the user made in between.
        """
        if not self._batch_changed or self._batch_baseline is None:
            return
        self._batch_changed = False
        if self._batch_left_canvas_unchanged():
            return
        if self._view_tracking:
            self._batch_baseline[VIEW_CHANGES_KEY] = self._owned_view_parts()
        else:
            self._batch_baseline[VIEW_CHANGES_KEY] = self._changed_view_parts(self._batch_baseline)
        self.undo_stack.append(self._batch_baseline)
        self.redo_stack = []

    def _batch_left_canvas_unchanged(self) -> bool:
        """True only when both signatures exist and match; a signature that failed counts as a change."""
        if self._view_tracking:
            return self._matches_batch_start()
        if self._batch_signature is None:
            return False
        current = self._live_signature()
        return current is not None and current == self._batch_signature

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
        """The current state, for the opposite stack; it restores the same view parts as ``entry``."""
        current = self.capture_state()
        current[VIEW_CHANGES_KEY] = list(self._entry_view_parts(entry))
        return current

    def _restore_history_entry(self, entry: Dict[str, Any]) -> None:
        """Restore an undo or redo entry: its objects, and the view parts its step changed."""
        self._apply_state(entry, self._entry_view_parts(entry), redraw=True)

    @staticmethod
    def _entry_view_parts(entry: Dict[str, Any]) -> Tuple[str, ...]:
        parts = entry.get(VIEW_CHANGES_KEY)
        return tuple(VIEW_PARTS) if parts is None else tuple(parts)

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
