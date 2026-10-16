"""Workspaces saved before the 2026-10 refactors must still load unchanged.

The files in ``data/saved_workspaces_2026-10-15/`` were written by the server's
WorkspaceManager from the round-trip scenes (``test_workspace_round_trip.py``) before the
expression, renderer and plot refactors. Each one is restored and saved again, and the
result must match the file. The files are never regenerated: when the saved format
changes on purpose, loading these files must still give the same drawables.

``legacy_plots.json`` is the exception: no released build wrote a ``Plots`` list that names
its parts, so it was assembled by hand from the shapes the restore has accepted since the
plot classes were split, plus a discrete plot saved without a ``distribution_type`` and a
stale saved bar. Its ``expected_state`` records what the restore made of it before the plot
refactor: legacy plots move to ``ContinuousPlots`` and ``DiscretePlots`` or stay a base
``Plot``, and a missing ``distribution_type`` stays null. It uses schema version 1 (a file
saved before workspaces kept the chat), which the server loads unchanged.

The bars that discrete and bar-chart plots rebuild on restore are not saved, so
``FIXTURE_BARS`` records each one's state as the restore built it before the plot refactor.
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
    "legacy_plots",
)

# The derived bars each fixture's plots rebuild (``Bar.get_state()["args"]``); a fixture not
# listed rebuilds none. Recorded before the plot refactor.
FIXTURE_BARS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "plots": {
        "Disc_bar_0": {
            "x_left": -2,
            "x_right": -1,
            "y_bottom": 0,
            "y_top": 0.12951759566589174,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": None,
            "label_text": None,
        },
        "Disc_bar_1": {
            "x_left": -1,
            "x_right": 0,
            "y_bottom": 0,
            "y_top": 0.3520653267642995,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": None,
            "label_text": None,
        },
        "Disc_bar_2": {
            "x_left": 0,
            "x_right": 1,
            "y_bottom": 0,
            "y_top": 0.3520653267642995,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": None,
            "label_text": None,
        },
        "Disc_bar_3": {
            "x_left": 1,
            "x_right": 2,
            "y_bottom": 0,
            "y_top": 0.12951759566589174,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": None,
            "label_text": None,
        },
        "Poll_bar_0": {
            "x_left": 60,
            "x_right": 61,
            "y_bottom": 60,
            "y_top": 61,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": "a",
            "label_text": None,
        },
        "Poll_bar_1": {
            "x_left": 61.2,
            "x_right": 62.2,
            "y_bottom": 60,
            "y_top": 62,
            "stroke_color": "",
            "fill_color": "lightblue",
            "fill_opacity": 0.3,
            "label_above_text": None,
            "label_below_text": "b",
            "label_text": None,
        },
    },
    "legacy_plots": {
        "Untyped_bar_0": {
            "x_left": -1.5,
            "x_right": -0.5,
            "y_bottom": 0,
            "y_top": 0.24197072451914337,
            "stroke_color": "red",
            "fill_color": "orange",
            "fill_opacity": 0.5,
            "label_above_text": "a",
            "label_below_text": None,
            "label_text": "a",
        },
        "Untyped_bar_1": {
            "x_left": -0.5,
            "x_right": 0.5,
            "y_bottom": 0,
            "y_top": 0.3989422804014327,
            "stroke_color": "red",
            "fill_color": "orange",
            "fill_opacity": 0.5,
            "label_above_text": "b",
            "label_below_text": None,
            "label_text": "b",
        },
        "Untyped_bar_2": {
            "x_left": 0.5,
            "x_right": 1.5,
            "y_bottom": 0,
            "y_top": 0.24197072451914337,
            "stroke_color": "red",
            "fill_color": "orange",
            "fill_opacity": 0.5,
            "label_above_text": "c",
            "label_below_text": None,
            "label_text": "c",
        },
    },
}


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class TestSavedWorkspaceFixtures(unittest.TestCase):
    def test_saved_workspaces_restore_unchanged(self) -> None:
        for name in FIXTURE_NAMES:
            with self.subTest(workspace=name):
                fixture = _load_fixture(name)
                canvas = Canvas(500, 500, draw_enabled=False)
                manager = WorkspaceManager(canvas)
                manager._restore_workspace_state(json.loads(json.dumps(fixture["state"])))
                restored = json.loads(json.dumps(manager.snapshot_canvas_state()))
                expected = fixture.get("expected_state", fixture["state"])
                self.assertEqual(_drawable_buckets(restored), _drawable_buckets(expected))
                self._assert_bars(canvas, FIXTURE_BARS.get(name, {}))

    def _assert_bars(self, canvas: Canvas, expected: Dict[str, Dict[str, Any]]) -> None:
        bars = {bar.name: bar.get_state()["args"] for bar in canvas.get_drawables_by_class_name("Bar")}
        self.assertEqual(sorted(bars), sorted(expected))
        for bar_name, expected_args in expected.items():
            args = bars[bar_name]
            self.assertEqual(sorted(args), sorted(expected_args), bar_name)
            for field, value in expected_args.items():
                if _is_number(value):
                    self.assertTrue(_is_number(args[field]), f"{bar_name}.{field}: {args[field]!r}")
                    self.assertAlmostEqual(args[field], value, delta=1e-12, msg=f"{bar_name}.{field}")
                else:
                    self.assertEqual(args[field], value, f"{bar_name}.{field}")

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
