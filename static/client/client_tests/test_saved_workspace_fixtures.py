"""Workspaces saved before the 2026-10 refactors must still load unchanged.

The files in ``data/saved_workspaces_2026-10-15/`` were written by the server's
WorkspaceManager from the round-trip scenes (``test_workspace_round_trip.py``) before the
expression, renderer and plot refactors. Each one is restored and saved again, and the
result must match the file. The files are never regenerated: when the saved format
changes on purpose, loading these files must still give the same drawables.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from browser import ajax

from canvas import Canvas
from workspace_manager import WorkspaceManager

FIXTURE_DIR = "/static/client/client_tests/data/saved_workspaces_2026-10-15"

# Every file in FIXTURE_DIR (server_tests/test_drawable_wiring.py checks the list).
FIXTURE_NAMES = (
    "primitives",
    "small_polygons",
    "pentagon",
    "hexagon",
    "heptagon",
    "octagon",
    "nonagon",
    "decagon",
    "generic_polygon",
    "curves",
    "functions_and_areas",
    "plots",
    "graphs",
)


def _load_fixture(name: str) -> Dict[str, Any]:
    request = ajax.Ajax()
    request.open("GET", f"{FIXTURE_DIR}/{name}.json", False)
    request.send()
    if request.status != 200:
        raise AssertionError(f"Could not fetch the saved workspace '{name}' (HTTP {request.status})")
    return json.loads(request.text)


def _drawable_buckets(state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Each non-empty drawable bucket, its items sorted by name and without render-cache keys."""
    return {
        key: sorted(
            [{field: value for field, value in item.items() if not field.startswith("_")} for item in items],
            key=lambda item: str(item.get("name", "")),
        )
        for key, items in state.items()
        if isinstance(items, list) and items and key != "computations"
    }


class TestSavedWorkspaceFixtures(unittest.TestCase):
    def test_saved_workspaces_restore_unchanged(self) -> None:
        for name in FIXTURE_NAMES:
            with self.subTest(workspace=name):
                saved = _load_fixture(name)["state"]
                canvas = Canvas(500, 500, draw_enabled=False)
                manager = WorkspaceManager(canvas)
                manager._restore_workspace_state(json.loads(json.dumps(saved)))
                restored = json.loads(json.dumps(manager.snapshot_canvas_state()))
                self.assertEqual(_drawable_buckets(restored), _drawable_buckets(saved))

    def test_unknown_saved_lists_are_reported_and_the_rest_restores(self) -> None:
        saved = _load_fixture("primitives")["state"]
        saved["Widgets"] = [{"name": "w1", "args": {}}]
        saved["Gadgets"] = []
        canvas = Canvas(500, 500, draw_enabled=False)
        manager = WorkspaceManager(canvas)
        self.assertEqual(manager._unrestored_buckets(saved), ["Widgets"])
        manager._restore_workspace_state(saved)
        self.assertEqual(len(canvas.get_drawables_by_class_name("Point")), len(saved["Points"]))


if __name__ == "__main__":
    unittest.main()
