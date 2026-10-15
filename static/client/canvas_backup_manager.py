"""Browser-storage backup of the canvas, so a page reload does not lose work.

The canvas is mirrored to ``localStorage`` in the same form a saved workspace
stores it (``WorkspaceManager.snapshot_canvas_state``), plus the view (zoom, pan,
coordinate mode and grid visibility). When the page loads, ``restore`` brings
that canvas back before anything else touches it.

When the backup is written:

- after the canvas draws (every visible change redraws it), at most once per
  ``BACKUP_DELAY_MS``, so dragging or panning does not serialize every frame;
- at once when the page is hidden or unloaded (reload, tab switch, window close).

Unchanged canvases are not written again. The backup is per origin, so every tab
of the app on the same host and port shares it. When another tab writes it, this
tab stops writing until its own canvas changes (an undo step), so panning an old
tab does not replace the newer work of another.

A backup that cannot be fully restored (unreadable, a restore that raised, or a
bucket that came back with fewer items) is copied to ``UNRESTORED_STORAGE_KEY``
before the canvas is backed up again, so a failed restore never overwrites the
only copy. A write the browser refuses (storage full or disabled) removes the
backup instead of leaving an older canvas to come back.

Chat history is not part of the backup; saved workspaces keep the chat.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from browser import document, window
from json_safe import ensure_json_safe

if TYPE_CHECKING:
    from canvas import Canvas
    from workspace_manager import WorkspaceManager

BACKUP_STORAGE_KEY = "mathud.canvas.backup"
UNRESTORED_STORAGE_KEY = "mathud.canvas.backup.unrestored"
BACKUP_VERSION = 1
# Shortest time between two backups while the canvas keeps changing.
BACKUP_DELAY_MS = 1000

TimeoutScheduler = Callable[[Callable[[], None], int], Any]


class CanvasBackupManager:
    """Mirrors the canvas to browser storage and restores it when the page loads.

    Attributes:
        canvas: The canvas to back up and restore.
        workspace_manager: Serializes and restores the canvas state.
        _storage: ``localStorage`` (or a test double); None when the browser refuses it.
        _set_timeout: Schedules the delayed backup (``window.setTimeout`` by default).
        _pending: True while a delayed backup is scheduled.
        _last_written: The text last written to storage, to skip unchanged canvases.
        _write_failed: True after a refused write, so the warning is printed once.
        _superseded_at: The undo depths when another tab last wrote the backup; this tab
            skips its backups until its own canvas changes. None when this tab wrote last.
    """

    def __init__(
        self,
        canvas: "Canvas",
        workspace_manager: "WorkspaceManager",
        storage: Optional[Any] = None,
        set_timeout: Optional[TimeoutScheduler] = None,
    ) -> None:
        self.canvas = canvas
        self.workspace_manager = workspace_manager
        self._storage: Optional[Any] = storage if storage is not None else _browser_local_storage()
        self._set_timeout: TimeoutScheduler = set_timeout or _window_set_timeout
        self._pending = False
        self._last_written: Optional[str] = None
        self._write_failed = False
        self._superseded_at: Optional[Tuple[int, int]] = None

    @property
    def enabled(self) -> bool:
        """True when browser storage is available."""
        return self._storage is not None

    def restore(self) -> bool:
        """Replace the canvas with the backup, if there is one; True when it was restored.

        The restore leaves no undo history, as a freshly loaded page has none.
        """
        raw = self._read(BACKUP_STORAGE_KEY)
        if not raw:
            return False
        try:
            backup = self._parse_backup(raw)
            self.workspace_manager.restore_canvas_state(backup["state"])
            self._restore_view(backup.get("view"))
            self.canvas.undo_redo_manager.clear()
        except Exception as exc:
            print(f"Warning: could not restore the canvas backup: {exc}")
            self._keep_unrestored(raw)
            return False
        missing = self._buckets_missing_items(backup["state"])
        if missing:
            # Items that failed one by one were skipped with a warning; keep the full copy.
            print(f"Warning: the canvas backup was restored without some {', '.join(missing)}")
            self._keep_unrestored(raw)
            return True
        self._last_written = raw
        return True

    def start(self) -> None:
        """Back the canvas up after it draws and whenever the page is hidden or unloaded."""
        if not self.enabled:
            return
        self.canvas.add_draw_listener(self.schedule_backup)
        window.bind("storage", self._on_storage_event)
        window.bind("pagehide", lambda _event: self.backup_now())
        document.bind("visibilitychange", self._on_visibility_change)

    def schedule_backup(self) -> None:
        """Back up after ``BACKUP_DELAY_MS`` unless a backup is already scheduled."""
        if not self.enabled or self._pending:
            return
        self._pending = True
        self._set_timeout(self._run_scheduled_backup, BACKUP_DELAY_MS)

    def backup_now(self) -> bool:
        """Write the canvas to storage now; True when the backup is current."""
        if not self.enabled:
            return False
        if self._superseded_at is not None:
            if self._superseded_at == self._undo_depths():
                return False
            self._superseded_at = None
        try:
            text = self._serialize_backup()
            if text != self._last_written:
                self._storage.setItem(BACKUP_STORAGE_KEY, text)
                self._last_written = text
            self._write_failed = False
            return True
        except Exception as exc:
            self._on_write_failure(exc)
            return False

    def _run_scheduled_backup(self) -> None:
        self._pending = False
        self.backup_now()

    def mark_superseded(self) -> None:
        """Another tab wrote the backup: skip backups until this tab's canvas changes."""
        self._superseded_at = self._undo_depths()
        self._last_written = None

    def _on_storage_event(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if key is None or str(key) == BACKUP_STORAGE_KEY:  # None: the storage was cleared
            self.mark_superseded()

    def _undo_depths(self) -> Tuple[int, int]:
        """Every change to the canvas made in this tab moves one of the undo or redo depths."""
        manager = self.canvas.undo_redo_manager
        return len(manager.undo_stack), len(manager.redo_stack)

    def _buckets_missing_items(self, saved_state: Dict[str, Any]) -> List[str]:
        """The drawable buckets that have fewer items on the canvas than in the backup."""
        restored = self.workspace_manager.snapshot_canvas_state()
        restored = restored if isinstance(restored, dict) else {}
        missing: List[str] = []
        for bucket, items in saved_state.items():
            # Drawable buckets are capitalized ("Points"); computations skip workspace calls on purpose.
            if not isinstance(items, list) or not bucket[:1].isupper():
                continue
            restored_items = restored.get(bucket)
            restored_count = len(restored_items) if isinstance(restored_items, list) else 0
            if restored_count < len(items):
                missing.append(bucket)
        return missing

    def _on_visibility_change(self, _event: Any) -> None:
        if str(getattr(document, "visibilityState", "")) == "hidden":
            self.backup_now()

    def _serialize_backup(self) -> str:
        backup: Dict[str, Any] = {
            "version": BACKUP_VERSION,
            "state": self.workspace_manager.snapshot_canvas_state(),
            "view": self.canvas.get_view_state(),
        }
        return json.dumps(ensure_json_safe(backup))

    @staticmethod
    def _parse_backup(raw: str) -> Dict[str, Any]:
        backup = json.loads(raw)
        if not isinstance(backup, dict) or backup.get("version") != BACKUP_VERSION:
            raise ValueError("unknown backup format")
        if not isinstance(backup.get("state"), dict):
            raise ValueError("the backup has no canvas state")
        return backup

    def _restore_view(self, view: Any) -> None:
        """Bring back the zoom, pan, coordinate mode and grids the canvas had (a workspace restore resets them)."""
        if not isinstance(view, dict):
            return
        zoom_changed = self.canvas.restore_view_state(view)
        self.canvas.draw(apply_zoom=bool(zoom_changed))

    def _keep_unrestored(self, raw: str) -> None:
        try:
            if self._storage is not None:
                self._storage.setItem(UNRESTORED_STORAGE_KEY, raw)
        except Exception as exc:
            print(f"Warning: could not keep the unrestored canvas backup: {exc}")

    def _on_write_failure(self, exc: Exception) -> None:
        if not self._write_failed:
            print(f"Warning: could not back up the canvas to browser storage: {exc}")
        self._write_failed = True
        self._last_written = None
        try:
            if self._storage is not None:
                self._storage.removeItem(BACKUP_STORAGE_KEY)
        except Exception:
            pass

    def _read(self, key: str) -> Optional[str]:
        if self._storage is None:
            return None
        try:
            value = self._storage.getItem(key)
        except Exception:
            return None
        return str(value) if value else None


def _browser_local_storage() -> Optional[Any]:
    """``window.localStorage``, or None when the browser refuses access (privacy settings, sandboxed frames)."""
    try:
        storage = window.localStorage
        storage.getItem(BACKUP_STORAGE_KEY)
        return storage
    except Exception:
        return None


def _window_set_timeout(callback: Callable[[], None], delay_ms: int) -> Any:
    return window.setTimeout(callback, delay_ms)
