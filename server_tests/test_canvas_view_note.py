"""Tests for the view note in static/canvas_state_formatter.py.

The app never moves the view on its own. When the shapes are too small on screen or
outside the view, the canvas the model sees carries one "View note:" line, so the
model can offer the user a zoom.
"""

from __future__ import annotations

import copy
import json
import math
import unittest
from typing import Any, Dict, List, Optional

from server_tests.test_canvas_state_formatter import load_scene, point
from static.canvas_state_formatter import (
    CANVAS_SIZE_KEY,
    CHANGES_HEADER,
    CURRENT_HEADER,
    OMITTED_NOTE,
    VIEW_NOTE_PREFIX,
    Box,
    render_min_json,
    render_state,
    render_text,
    render_update,
    summarize_view,
    view_note,
)
from static.token_estimation import estimate_tokens_from_text

# The app's default view on an 800 x 600 px canvas: one math unit per pixel.
DEFAULT_VIEW: Dict[str, Any] = {
    "Cartesian_System_Visibility": {"left_bound": -400, "right_bound": 400, "top_bound": 300, "bottom_bound": -300},
    "current_tick_spacing": 100,
    "coordinate_system": {"mode": "cartesian"},
    CANVAS_SIZE_KEY: {"width": 800, "height": 600},
}


def scene(points: List[Dict[str, Any]], view: Optional[Dict[str, Any]] = None, **buckets: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {"Points": points}
    state.update(buckets)
    state.update(copy.deepcopy(view if view is not None else DEFAULT_VIEW))
    return state


def with_bounds(state: Dict[str, Any], left: float, right: float, bottom: float, top: float) -> Dict[str, Any]:
    moved = copy.deepcopy(state)
    moved["Cartesian_System_Visibility"] = {
        "left_bound": left,
        "right_bound": right,
        "bottom_bound": bottom,
        "top_bound": top,
    }
    return moved


def tiny_triangle_with_circumcircle() -> Dict[str, Any]:
    """The reported case: triangle (0,0), (6,0), (2,4) and its circumcircle at the default view."""
    return scene(
        [point("A", 0, 0), point("B", 6, 0), point("C", 2, 4), point("O", 3, 1)],
        Triangles=[{"name": "ABC", "args": {"p1": "A", "p2": "B", "p3": "C"}}],
        Circles=[{"name": "c", "args": {"center": "O", "radius": math.sqrt(10)}}],
    )


def big_triangle() -> Dict[str, Any]:
    return scene(
        [point("A", -300, -200), point("B", 300, -200), point("C", 0, 250)],
        Triangles=[{"name": "ABC", "args": {"p1": "A", "p2": "B", "p3": "C"}}],
    )


class TestTooSmall(unittest.TestCase):
    maxDiff = None

    def test_tiny_triangle_at_the_default_view_gets_the_hint(self) -> None:
        note = view_note(None, tiny_triangle_with_circumcircle())
        self.assertEqual(
            note,
            "View note: the shapes span only ~6x6 px on screen (shapes x -0.2..6.2, y -2.2..4.2; "
            "view x -400..400, y -300..300). Offer to zoom to about x -2.3..8.3, y -3..5 "
            "(zoom center_x=3, center_y=1, range_val=5.3, range_axis=x); "
            "don't change the view unless the user agrees.",
        )

    def test_suggested_view_contains_the_shapes(self) -> None:
        summary = summarize_view(tiny_triangle_with_circumcircle())
        assert summary is not None and summary.content is not None
        suggested = Box.around(3, 1, 5.3, 5.3 * 600 / 800)
        self.assertTrue(suggested.left < summary.content.left and summary.content.right < suggested.right)
        self.assertTrue(suggested.bottom < summary.content.bottom and summary.content.top < suggested.top)

    def test_normal_size_drawing_gets_no_hint(self) -> None:
        self.assertIsNone(view_note(None, big_triangle()))
        # 60 px across at one unit per pixel: above the 40 px threshold.
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 60, 0)])))

    def test_threshold_is_forty_pixels_on_a_normal_canvas(self) -> None:
        self.assertIsNotNone(view_note(None, scene([point("A", 0, 0), point("B", 39, 0)])))
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 41, 0)])))

    def test_threshold_grows_with_very_large_canvases(self) -> None:
        # 3% of a 2000 px smaller side is 60 px; the view still maps one unit to one pixel.
        big_canvas = dict(DEFAULT_VIEW)
        big_canvas["Cartesian_System_Visibility"] = {
            "left_bound": -1500,
            "right_bound": 1500,
            "top_bound": 1000,
            "bottom_bound": -1000,
        }
        big_canvas[CANVAS_SIZE_KEY] = {"width": 3000, "height": 2000}
        self.assertIsNotNone(view_note(None, scene([point("A", 0, 0), point("B", 50, 0)], big_canvas)))

    def test_pixels_follow_the_zoom(self) -> None:
        # The same triangle, zoomed so that one unit is 20 px: 126 px across, readable.
        zoomed = with_bounds(tiny_triangle_with_circumcircle(), -20, 20, -15, 15)
        self.assertIsNone(view_note(None, zoomed))

    def test_a_lone_point_is_never_too_small(self) -> None:
        self.assertIsNone(view_note(None, scene([point("A", 3, 4)])))
        self.assertIsNone(view_note(None, scene([point("A", 3, 4), point("B", 3, 4)])))

    def test_without_a_canvas_size_the_view_fraction_is_used(self) -> None:
        state = tiny_triangle_with_circumcircle()
        del state[CANVAS_SIZE_KEY]
        note = view_note(None, state)
        assert note is not None
        self.assertIn("the shapes span only ~1.1% of the view", note)
        # A captured scene from an older client (view +-491 x +-249, a 7-unit drawing).
        self.assertIn("% of the view", view_note(None, load_scene("triangle_circle")) or "")


class TestOutsideTheView(unittest.TestCase):
    def test_content_entirely_off_screen(self) -> None:
        state = scene([point("A", 1000, 0), point("B", 1100, 0), point("C", 1050, 80)])
        note = view_note(None, state)
        assert note is not None
        self.assertIn("the shapes are entirely outside the view (shapes x 1000..1100, y 0..80;", note)
        # Readable at the current zoom, so the suggestion only moves the view.
        self.assertIn("Offer to move the view to about x 650..1450", note)
        self.assertIn("range_val=400", note)

    def test_content_mostly_off_screen(self) -> None:
        state = scene([point("A", 0, 0), point("B", 900, 0), point("C", 450, 200)])
        note = view_note(None, state)
        assert note is not None
        self.assertIn("only ~44% of the shapes' extent is inside the view", note)
        self.assertIn("Offer to zoom to about", note)

    def test_content_half_visible_is_fine(self) -> None:
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 700, 0), point("C", 300, 200)])))

    def test_changed_object_off_screen_in_a_visible_scene(self) -> None:
        before = scene([point("A", -390, -200), point("B", 390, -200), point("C", 0, 250)])
        after = copy.deepcopy(before)
        after["Points"].append(point("P", 450, 0))
        note = view_note(before, after)
        assert note is not None
        self.assertIn("new or changed P is outside the view (x -400..400, y -300..300)", note)
        self.assertIsNone(view_note(after, after))

    def test_many_changed_objects_are_counted(self) -> None:
        before = scene([point("A", -390, -200), point("B", 390, -200), point("C", 0, 250)])
        after = copy.deepcopy(before)
        after["Points"].extend(point(name, 420 + i, 0) for i, name in enumerate("PQRST"))
        self.assertIn("new or changed P, Q, R (+2 more) are outside the view", view_note(before, after) or "")


class TestNoHint(unittest.TestCase):
    def test_empty_canvas(self) -> None:
        self.assertIsNone(view_note(None, scene([])))
        self.assertIsNone(view_note(tiny_triangle_with_circumcircle(), scene([])))

    def test_only_unbounded_objects(self) -> None:
        state = scene([], Functions=[{"name": "f", "args": {"function_string": "x^2"}}])
        self.assertIsNone(view_note(None, state))

    def test_no_view(self) -> None:
        self.assertIsNone(view_note(None, {"Points": [point("A", 0, 0), point("B", 1, 0)]}))
        broken = with_bounds(tiny_triangle_with_circumcircle(), 10, -10, -5, 5)
        self.assertIsNone(view_note(None, broken))

    def test_malformed_state_never_raises(self) -> None:
        state = scene(
            [point("A", 0, 0), {"name": "B", "args": {"position": "nowhere"}}, "junk"],  # type: ignore[list-item]
            Circles=[{"name": "c", "args": {"center": "missing", "radius": 1}}, {"args": None}],
            Ellipses="not a list",
            Bars=[{"name": "b", "args": {"x_left": "a"}}],
        )
        state[CANVAS_SIZE_KEY] = {"width": "wide"}
        self.assertIsNone(view_note(None, state))
        self.assertIsNone(view_note("not a state", state))  # type: ignore[arg-type]


class TestRepeatSuppression(unittest.TestCase):
    """A note comes back only when the drawing changes, never for the user's own pan or zoom."""

    def test_same_drawing_is_not_noted_again(self) -> None:
        state = tiny_triangle_with_circumcircle()
        self.assertIsNotNone(view_note(None, state))
        self.assertIsNone(view_note(state, copy.deepcopy(state)))

    def test_user_pan_or_zoom_does_not_bring_it_back(self) -> None:
        state = tiny_triangle_with_circumcircle()
        self.assertIsNone(view_note(state, with_bounds(state, -800, 800, -600, 600)))
        self.assertIsNone(view_note(state, with_bounds(state, 1000, 1800, -300, 300)))

    def test_user_zooming_into_a_big_scene_is_not_noted(self) -> None:
        state = big_triangle()
        self.assertIsNone(view_note(state, with_bounds(state, -10, 10, -7.5, 7.5)))

    def test_object_inside_the_drawing_does_not_bring_it_back(self) -> None:
        state = tiny_triangle_with_circumcircle()
        after = copy.deepcopy(state)
        after["Points"].append(point("M", 3, 0))
        self.assertIsNone(view_note(state, after))

    def test_drawing_that_grows_is_noted_again(self) -> None:
        state = tiny_triangle_with_circumcircle()
        after = copy.deepcopy(state)
        after["Points"].append(point("D", 9, 9))
        self.assertIsNotNone(view_note(state, after))

    def test_float_noise_does_not_count_as_a_change(self) -> None:
        state = tiny_triangle_with_circumcircle()
        after = copy.deepcopy(state)
        after["Points"][1]["args"]["position"]["x"] = 6.000000000000001
        self.assertIsNone(view_note(state, after))


class TestExtents(unittest.TestCase):
    def _content(self, **buckets: Any) -> Box:
        points = buckets.pop("Points", [])
        summary = summarize_view(scene(points, **buckets))
        assert summary is not None and summary.content is not None
        return summary.content

    def assertBox(self, box: Box, expected: tuple) -> None:  # noqa: N802
        for actual, wanted in zip((box.left, box.right, box.bottom, box.top), expected):
            self.assertAlmostEqual(actual, wanted)

    def test_rotated_ellipse(self) -> None:
        box = self._content(
            Points=[point("E", 1, 1)],
            Ellipses=[{"name": "e", "args": {"center": "E", "radius_x": 4, "radius_y": 2, "rotation_angle": 90}}],
        )
        self.assertBox(box, (-1, 3, -3, 5))

    def test_arc_bar_and_label(self) -> None:
        box = self._content(
            CircleArcs=[{"name": "a", "args": {"center_x": 0, "center_y": 0, "radius": 2}}],
            Bars=[{"name": "b", "args": {"x_left": 5, "x_right": 6, "y_bottom": 0, "y_top": -3}}],
            Labels=[{"name": "l", "args": {"position": {"x": -4, "y": 1}, "text": "hi"}}],
        )
        self.assertBox(box, (-4, 6, -3, 2))

    def test_bar_chart(self) -> None:
        box = self._content(
            BarsPlots=[
                {
                    "name": "sales",
                    "args": {"values": [12, -3, 7], "bar_width": 1, "bar_spacing": 0.2, "x_start": -2, "y_base": 1},
                }
            ]
        )
        self.assertBox(box, (-2, 1.4, -2, 13))

    def test_canvas_size_is_not_rendered_as_an_object(self) -> None:
        state = tiny_triangle_with_circumcircle()
        self.assertNotIn("canvas_size_px", render_text(state))
        self.assertNotIn("canvas_size_px", render_min_json(state))


class TestNoteInEveryFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.state = tiny_triangle_with_circumcircle()
        note = view_note(None, self.state)
        assert note is not None
        self.note = note

    def test_text_puts_the_note_under_the_view_line(self) -> None:
        lines = render_state(self.state, "text", view_note=self.note).split("\n")
        self.assertEqual(lines[0], "view x [-400, 400] y [-300, 300]; grid 100")
        self.assertEqual(lines[1], self.note)

    def test_min_json_and_json_carry_a_view_note_key(self) -> None:
        self.assertEqual(json.loads(render_state(self.state, "min_json", view_note=self.note))["view_note"], self.note)
        self.assertEqual(json.loads(render_state(self.state, "json", view_note=self.note))["view_note"], self.note)
        self.assertNotIn("view_note", render_state(self.state, "json"))

    def test_budget_trimming_keeps_the_note(self) -> None:
        crowded = scene([point(f"P{i}", (i % 20) * 0.3, (i // 20) * 0.3) for i in range(200)])
        note = view_note(None, crowded)
        assert note is not None and note.startswith(VIEW_NOTE_PREFIX)
        text = render_text(crowded, budget_tokens=400, view_note=note)
        self.assertIn(OMITTED_NOTE, text)
        self.assertEqual(text.split("\n")[1], note)
        self.assertLessEqual(estimate_tokens_from_text(text), 400)
        trimmed = json.loads(render_min_json(crowded, budget_tokens=400, view_note=note))
        self.assertIn("omitted", trimmed)
        self.assertEqual(trimmed["view_note"], note)

    def test_canvas_changes_end_with_the_note(self) -> None:
        before = scene([])
        update = render_update(before, self.state, "text", view_note=self.note)
        self.assertTrue(update.startswith(CHANGES_HEADER + "\n+ A = (0, 0)"))
        self.assertTrue(update.endswith("\n" + self.note))
        self.assertTrue(render_update(before, self.state, "min_json", view_note=self.note).endswith(self.note))

    def test_full_canvas_update_carries_the_note(self) -> None:
        update = render_update(None, self.state, "text", view_note=self.note)
        self.assertTrue(
            update.startswith(CURRENT_HEADER + "\nview x [-400, 400] y [-300, 300]; grid 100\n" + self.note)
        )

    def test_unchanged_canvas_still_reports_a_note(self) -> None:
        self.assertEqual(
            render_update(self.state, self.state, "text", view_note=self.note), f"{CHANGES_HEADER}\n{self.note}"
        )
        self.assertEqual(render_update(self.state, self.state, "text"), "")


if __name__ == "__main__":
    unittest.main()
