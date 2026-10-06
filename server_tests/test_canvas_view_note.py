"""Tests for static/canvas_view_note.py and how the render functions carry its note.

The app never moves the view on its own. When the shapes are too small on screen,
too flat or outside the view, the canvas the model sees carries one "View note:"
line, so the model can offer the user a zoom.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unittest
from typing import Any, Dict, List, Optional, Tuple

from server_tests.test_canvas_state_formatter import load_scene, point
from static.canvas_state_formatter import (
    CANVAS_SIZE_KEY,
    CHANGES_HEADER,
    CURRENT_HEADER,
    CURVE_EXTENTS_KEY,
    OMITTED_NOTE,
    render_min_json,
    render_state,
    render_text,
    render_update,
)
from static.canvas_view_note import VIEW_NOTE_PREFIX, Box, summarize_view, view_note
from static.token_estimation import estimate_tokens_from_text

# The app's default view on an 800 x 600 px canvas: one math unit per pixel.
DEFAULT_VIEW: Dict[str, Any] = {
    "Cartesian_System_Visibility": {"left_bound": -400, "right_bound": 400, "top_bound": 300, "bottom_bound": -300},
    "current_tick_spacing": 100,
    "min_tick_spacing": 1e-06,
    "coordinate_system": {"mode": "cartesian"},
    CANVAS_SIZE_KEY: {"width": 800, "height": 600},
}
# The default view of a 1256 x 963 px canvas (headless Chrome at the CLI's window size).
WIDE_VIEW: Dict[str, Any] = dict(
    DEFAULT_VIEW,
    Cartesian_System_Visibility={"left_bound": -628, "right_bound": 628, "top_bound": 481.5, "bottom_bound": -481.5},
    **{CANVAS_SIZE_KEY: {"width": 1256, "height": 963}},
)


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


def triangle(name: str = "ABC") -> Dict[str, Any]:
    return {"name": name, "args": {"p1": name[0], "p2": name[1], "p3": name[2]}}


def circle(name: str, center: str, radius: float) -> Dict[str, Any]:
    return {"name": name, "args": {"center": center, "radius": radius}}


def segment(p1: str, p2: str) -> Dict[str, Any]:
    return {"name": p1 + p2, "args": {"p1": p1, "p2": p2}}


TINY_POINTS = [point("A", 0, 0), point("B", 6, 0), point("C", 2, 4)]


def tiny_triangle_with_circumcircle() -> Dict[str, Any]:
    """The reported case: triangle (0,0), (6,0), (2,4) and its circumcircle at the default view."""
    return scene(TINY_POINTS + [point("O", 3, 1)], Triangles=[triangle()], Circles=[circle("c", "O", math.sqrt(10))])


def big_triangle() -> Dict[str, Any]:
    return scene(
        [point("A", -300, -200), point("B", 300, -200), point("C", 0, 250)],
        Segments=[segment("A", "B"), segment("B", "C"), segment("C", "A")],
        Triangles=[triangle()],
    )


def graph(
    name: str, box: Tuple[float, float, float, float], clipped: bool, view: Dict[str, Any], **args: Any
) -> Dict[str, Any]:
    """A scene with one function graph whose client-sampled box is ``box``."""
    state = scene([], view, Functions=[{"name": name, "args": dict({"function_string": "f(x)"}, **args)}])
    state[CURVE_EXTENTS_KEY] = {"Functions": {name: {"box": list(box), "clipped": clipped}}}
    return state


def add(state: Dict[str, Any], *points: Dict[str, Any], **buckets: Any) -> Dict[str, Any]:
    grown = copy.deepcopy(state)
    grown["Points"].extend(points)
    for bucket, items in buckets.items():
        grown.setdefault(bucket, []).extend(items)
    return grown


class TestTooSmall(unittest.TestCase):
    maxDiff = None

    def test_tiny_triangle_at_the_default_view_gets_the_hint(self) -> None:
        self.assertEqual(
            view_note(None, tiny_triangle_with_circumcircle()),
            "View note: the shapes span only ~6x6 px on screen (shapes x -0.2..6.2, y -2.2..4.2; "
            "view x -400..400, y -300..300). Offer to zoom to about x -2.3..8.3, y -3..5 "
            "(zoom center_x=3, center_y=1, range_val=5.3, range_axis=x); "
            "don't change the view unless the user agrees.",
        )

    def test_view_bounds_are_rounded_like_the_view_line(self) -> None:
        note = view_note(None, scene(TINY_POINTS, WIDE_VIEW)) or ""
        self.assertIn("view x -628..628, y -481.5..481.5", note)

    def test_suggested_view_contains_the_shapes(self) -> None:
        summary = summarize_view(tiny_triangle_with_circumcircle())
        assert summary is not None and summary.content is not None
        suggested = Box.around(3, 1, 5.3, 5.3 * 600 / 800)
        self.assertTrue(suggested.left < summary.content.left and summary.content.right < suggested.right)
        self.assertTrue(suggested.bottom < summary.content.bottom and summary.content.top < suggested.top)

    def test_normal_size_drawing_gets_no_hint(self) -> None:
        self.assertIsNone(view_note(None, big_triangle()))
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 60, 0)])))

    def test_threshold_is_forty_pixels_on_a_normal_canvas(self) -> None:
        self.assertIsNotNone(view_note(None, scene([point("A", 0, 0), point("B", 39, 0)])))
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 41, 0)])))

    def test_threshold_grows_with_very_large_canvases(self) -> None:
        big_canvas = with_bounds(scene([]), -1500, 1500, -1000, 1000)
        big_canvas[CANVAS_SIZE_KEY] = {"width": 3000, "height": 2000}
        # 3% of a 2000 px smaller side is 60 px; the view still maps one unit to one pixel.
        self.assertIsNotNone(view_note(None, add(big_canvas, point("A", 0, 0), point("B", 50, 0))))

    def test_pixels_follow_the_zoom(self) -> None:
        self.assertIsNone(view_note(None, with_bounds(tiny_triangle_with_circumcircle(), -20, 20, -15, 15)))

    def test_a_lone_point_is_never_too_small(self) -> None:
        self.assertIsNone(view_note(None, scene([point("A", 3, 4)])))
        self.assertIsNone(view_note(None, scene([point("A", 3, 4), point("B", 3, 4)])))

    def test_without_a_canvas_size_the_view_fraction_is_used(self) -> None:
        state = tiny_triangle_with_circumcircle()
        del state[CANVAS_SIZE_KEY]
        self.assertIn("the shapes span only ~1.1% of the view", view_note(None, state) or "")
        # A captured scene from an older client (view +-491 x +-249, a 7-unit drawing).
        self.assertIn("% of the view", view_note(None, load_scene("triangle_circle")) or "")


class TestDegenerateSizes(unittest.TestCase):
    """L3: extents too small to zoom into count as a point; huge numbers stay short."""

    def test_points_closer_than_the_finest_grid_count_as_one(self) -> None:
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 1e-9, 0)])))
        far = with_bounds(scene([point("A", 1e12, 0), point("B", 1e12 + 1e-4, 0)]), 1e12 - 400, 1e12 + 400, -300, 300)
        self.assertIsNone(view_note(None, far))

    def test_a_small_but_zoomable_drawing_gets_a_nonzero_range(self) -> None:
        note = view_note(None, scene([point("A", 0, 0), point("B", 1e-4, 0)])) or ""
        self.assertIn("range_val=6.3e-5", note)
        self.assertNotIn("range_val=0,", note)

    def test_huge_coordinates_are_written_in_scientific_notation(self) -> None:
        note = view_note(None, scene([point("A", 1e300, 0), point("B", -1e300, 0)])) or ""
        self.assertIn("shapes x -1e300..1e300", note)
        self.assertIsNone(re.search(r"\d{16}", note), note)


class TestTinyNewShapesInALargeDrawing(unittest.TestCase):
    """M1: the new shapes are measured on their own, with the small shapes right next to them."""

    def setUp(self) -> None:
        self.circle_only = scene([point("O", 0, 0)], Circles=[circle("c", "O", 250)])
        self.with_triangle = add(self.circle_only, *TINY_POINTS, Triangles=[triangle()])

    def test_tiny_triangle_next_to_a_big_circle(self) -> None:
        note = view_note(self.circle_only, self.with_triangle) or ""
        self.assertIn("the new or changed shapes (ABC) span only ~6x4 px on screen (shapes x 0..6, y 0..4;", note)
        self.assertIn("zoom center_x=3, center_y=2", note)

    def test_more_shapes_in_the_same_speck_are_not_noted_again(self) -> None:
        circumcircle = add(self.with_triangle, point("D", 3, 1), Circles=[circle("d", "D", math.sqrt(10))])
        self.assertIsNone(view_note(self.with_triangle, circumcircle))

    def test_a_new_speck_elsewhere_is_noted(self) -> None:
        second = add(self.with_triangle, point("E", 200, 100), point("F", 203, 100))
        self.assertIn("(E, F) span only ~3x<1 px", view_note(self.with_triangle, second) or "")

    def test_readable_new_shapes_are_fine(self) -> None:
        bigger = add(self.circle_only, point("A", 0, 0), point("B", 80, 0), point("C", 30, 50), Triangles=[triangle()])
        self.assertIsNone(view_note(self.circle_only, bigger))

    def test_first_message_judges_the_whole_drawing(self) -> None:
        self.assertIsNone(view_note(None, self.with_triangle))


class TestFunctionsAndCurves(unittest.TestCase):
    """M2: graphs and curves are measured from the boxes the client samples."""

    maxDiff = None

    def test_flat_sine_at_the_default_view(self) -> None:
        state = graph("f", (-628, 628, -1, 1), True, WIDE_VIEW)
        self.assertEqual(
            view_note(None, state),
            "View note: f varies only ~2 px vertically on screen (y -1..1 over x -628..628; "
            "view x -628..628, y -481.5..481.5). Offer to zoom to about x -5.3..5.3, y -4.1..4.1 "
            "(zoom center_x=0, center_y=0, range_val=5.3, range_axis=x); "
            "don't change the view unless the user agrees.",
        )

    def test_bounded_parabola_is_tiny(self) -> None:
        state = graph("f", (-2, 2, -2, 2), False, WIDE_VIEW, left_bound=-2, right_bound=2)
        self.assertIn("the shapes span only ~4x4 px on screen", view_note(None, state) or "")

    def test_readable_graph_gets_no_hint(self) -> None:
        self.assertIsNone(view_note(None, graph("f", (-400, 400, 0, 160000), True, DEFAULT_VIEW)))
        self.assertIsNone(view_note(None, graph("f", (-400, 400, -100, 100), True, DEFAULT_VIEW)))

    def test_a_constant_is_a_readable_line(self) -> None:
        self.assertIsNone(view_note(None, graph("f", (-400, 400, 3, 3), True, DEFAULT_VIEW)))

    def test_new_graph_above_the_view(self) -> None:
        before = scene([])
        after = graph("f", (-400, 400, 1000, 1160), True, DEFAULT_VIEW)
        self.assertIn("new or changed f is outside the view", view_note(before, after) or "")

    def test_flat_graph_unchanged_is_not_noted_again(self) -> None:
        state = graph("f", (-628, 628, -1, 1), True, WIDE_VIEW)
        self.assertIsNone(view_note(state, copy.deepcopy(state)))
        zoomed_out = with_bounds(state, -6280, 6280, -4815, 4815)
        self.assertIsNone(view_note(state, zoomed_out))

    def test_redefined_graph_is_measured_again(self) -> None:
        before = graph("f", (-400, 400, -100, 100), True, DEFAULT_VIEW)
        after = graph("f", (-400, 400, -1, 1), True, DEFAULT_VIEW)
        after["Functions"][0]["args"]["function_string"] = "sin(x)"
        self.assertIn("f varies only ~2 px", view_note(before, after) or "")

    def test_parametric_curve(self) -> None:
        state = scene([], ParametricFunctions=[{"name": "p", "args": {"x_expression": "cos(t)"}}])
        state[CURVE_EXTENTS_KEY] = {"ParametricFunctions": {"p": {"box": [-1, 1, -1, 1], "clipped": False}}}
        self.assertIn("the shapes span only ~2x2 px", view_note(None, state) or "")

    def test_area_under_a_graph_between_bounds(self) -> None:
        state = graph("f", (-400, 400, -1, 1), True, DEFAULT_VIEW)
        state["FunctionsBoundedColoredAreas"] = [
            {"name": "a", "args": {"func1": "f", "func2": "x_axis", "left_bound": 0, "right_bound": 3}}
        ]
        summary = summarize_view(state)
        assert summary is not None and summary.content is not None
        self.assertEqual(summary.content, Box(0, 3, -1, 1))
        self.assertIn("the shapes span only ~3x2 px", view_note(None, state) or "")

    def test_without_curve_extents_graphs_are_not_measured(self) -> None:
        state = graph("f", (-628, 628, -1, 1), True, WIDE_VIEW)
        del state[CURVE_EXTENTS_KEY]
        self.assertIsNone(view_note(None, state))


class TestOutsideTheView(unittest.TestCase):
    def test_content_entirely_off_screen(self) -> None:
        state = scene([point("A", 1000, 0), point("B", 1100, 0), point("C", 1050, 80)])
        note = view_note(None, state) or ""
        self.assertIn("the shapes are entirely outside the view (shapes x 1000..1100, y 0..80;", note)
        # Readable at the current zoom, so the suggestion only moves the view.
        self.assertIn("Offer to move the view to about x 650..1450", note)
        self.assertIn("range_val=400", note)

    def test_one_point_off_screen(self) -> None:
        self.assertIn("(shapes at (1000, 0);", view_note(None, scene([point("A", 1000, 0)])) or "")

    def test_content_mostly_off_screen(self) -> None:
        state = scene([point("A", 0, 0), point("B", 900, 0), point("C", 450, 200)])
        note = view_note(scene([]), state) or ""
        self.assertIn("only ~33% of the drawing is inside the view", note)
        self.assertIn("Offer to zoom to about", note)

    def test_visible_share_follows_the_outline(self) -> None:
        """L4: a diagonal segment through a +-400 view shows 30% of its length, not 0.4 x 0.3."""
        state = scene([point("A", -1000, -1000), point("B", 1000, 1000)], Segments=[segment("A", "B")])
        summary = summarize_view(state)
        assert summary is not None
        self.assertAlmostEqual(summary.visible_fraction(), 0.3, delta=0.05)
        self.assertIn("% of the drawing is inside the view", view_note(scene([]), state) or "")

    def test_half_visible_is_fine(self) -> None:
        self.assertIsNone(view_note(None, scene([point("A", 0, 0), point("B", 700, 0), point("C", 300, 200)])))

    def test_start_of_a_conversation_zoomed_into_a_big_drawing(self) -> None:
        """Part of the drawing on screen at the first message: the user's chosen view, no note."""
        zoomed = with_bounds(big_triangle(), -310, -190, -245, -155)
        summary = summarize_view(zoomed)
        assert summary is not None
        self.assertTrue(0 < summary.visible_fraction() < 0.5)
        self.assertIsNone(view_note(None, zoomed))

    def test_changed_object_off_screen_in_a_visible_scene(self) -> None:
        before = scene([point("A", -390, -200), point("B", 390, -200), point("C", 0, 250)])
        after = add(before, point("P", 450, 0))
        self.assertIn(
            "new or changed P is outside the view (view x -400..400, y -300..300)", view_note(before, after) or ""
        )
        self.assertIsNone(view_note(after, after))

    def test_ends_of_a_new_segment_across_the_view_are_not_outside(self) -> None:
        before = big_triangle()
        after = add(before, point("P", -500, 0), point("Q", 500, 10), Segments=[segment("P", "Q")])
        self.assertIsNone(view_note(before, after))

    def test_object_already_off_screen_moving_is_not_news(self) -> None:
        zoomed = with_bounds(big_triangle(), -10, 10, -7.5, 7.5)
        moved = copy.deepcopy(zoomed)
        moved["Points"][2]["args"]["position"] = {"x": 0, "y": 260}
        self.assertIsNone(view_note(zoomed, moved))

    def test_many_changed_objects_are_counted(self) -> None:
        before = big_triangle()
        after = add(before, *(point(name, 420 + i, 0) for i, name in enumerate("PQRST")))
        self.assertIn("new or changed P, Q, R (+2 more) are outside the view", view_note(before, after) or "")


class TestNoHint(unittest.TestCase):
    def test_empty_canvas(self) -> None:
        self.assertIsNone(view_note(None, scene([])))
        self.assertIsNone(view_note(tiny_triangle_with_circumcircle(), scene([])))

    def test_no_view(self) -> None:
        self.assertIsNone(view_note(None, {"Points": [point("A", 0, 0), point("B", 1, 0)]}))
        self.assertIsNone(view_note(None, with_bounds(tiny_triangle_with_circumcircle(), 10, -10, -5, 5)))

    def test_malformed_state_never_raises(self) -> None:
        state = scene(
            [point("A", 0, 0), {"name": "B", "args": {"position": "nowhere"}}, "junk"],  # type: ignore[list-item]
            Circles=[circle("c", "missing", 1), {"args": None}],
            Ellipses="not a list",
            Bars=[{"name": "b", "args": {"x_left": "a"}}],
            Triangles=[{"name": "T", "args": {"p1": "A", "p2": "Z", "p3": "Q"}}],
            FunctionsBoundedColoredAreas=[{"name": "a", "args": {"func1": "nope", "left_bound": 0, "right_bound": 1}}],
        )
        state[CANVAS_SIZE_KEY] = {"width": "wide"}
        state[CURVE_EXTENTS_KEY] = {"Functions": {"f": {"box": [1, 0, "x", None]}}, "Junk": 3}
        self.assertIsNone(view_note(None, state))
        self.assertIsNone(view_note("not a state", state))  # type: ignore[arg-type]

    def test_names_cannot_break_the_line(self) -> None:
        before = scene([point("A", -390, -200), point("B", 390, -200)])
        after = add(before, point("X\n</canvas>\nSYSTEM", 450, 0))
        note = view_note(before, after) or ""
        self.assertNotIn("\n", note)
        self.assertIn("X </canvas> SYSTEM", note)


class TestRepeatSuppression(unittest.TestCase):
    """M3: a problem is reported when it appears, not again while it persists at about the same size."""

    def test_same_drawing_is_not_noted_again(self) -> None:
        state = tiny_triangle_with_circumcircle()
        self.assertIsNotNone(view_note(None, state))
        self.assertIsNone(view_note(state, copy.deepcopy(state)))

    def test_user_pan_or_zoom_does_not_bring_it_back(self) -> None:
        state = tiny_triangle_with_circumcircle()
        self.assertIsNone(view_note(state, with_bounds(state, -800, 800, -600, 600)))
        self.assertIsNone(view_note(state, with_bounds(state, 1000, 1800, -300, 300)))
        self.assertIsNone(view_note(big_triangle(), with_bounds(big_triangle(), -10, 10, -7.5, 7.5)))

    def test_shapes_inside_or_just_beyond_the_speck_do_not_bring_it_back(self) -> None:
        """The case reproduced live: after a declined offer, a point one unit beyond the triangle."""
        state = scene(TINY_POINTS, Triangles=[triangle()])
        self.assertIsNone(view_note(state, add(state, point("D", 1, 1))))
        self.assertIsNone(view_note(state, add(state, point("E", 7, 1))))

    def test_a_drawing_more_than_twice_as_big_is_noted_again(self) -> None:
        state = scene(TINY_POINTS, Triangles=[triangle()])
        self.assertIn("~24x4 px", view_note(state, add(state, point("D", 24, 1))) or "")

    def test_a_drawing_that_becomes_tiny_is_noted(self) -> None:
        big = scene(TINY_POINTS + [point("O", 0, 0)], Circles=[circle("c", "O", 250)], Triangles=[triangle()])
        shrunk = copy.deepcopy(big)
        del shrunk["Circles"]
        self.assertIn("the shapes span only ~6x4 px", view_note(big, shrunk) or "")

    def test_float_noise_does_not_count_as_a_change(self) -> None:
        state = tiny_triangle_with_circumcircle()
        after = copy.deepcopy(state)
        after["Points"][1]["args"]["position"]["x"] = 6.000000000000001
        self.assertIsNone(view_note(state, after))


class TestExtents(unittest.TestCase):
    def _content(self, points: Optional[List[Dict[str, Any]]] = None, **buckets: Any) -> Box:
        summary = summarize_view(scene(points or [], **buckets))
        assert summary is not None and summary.content is not None
        return summary.content

    def assertBox(self, box: Box, expected: tuple) -> None:  # noqa: N802
        for actual, wanted in zip((box.left, box.right, box.bottom, box.top), expected):
            self.assertAlmostEqual(actual, wanted)

    def test_rotated_ellipse(self) -> None:
        ellipse = {"name": "e", "args": {"center": "E", "radius_x": 4, "radius_y": 2, "rotation_angle": 90}}
        self.assertBox(self._content([point("E", 1, 1)], Ellipses=[ellipse]), (-1, 3, -3, 5))

    def test_arc_bar_and_label(self) -> None:
        box = self._content(
            CircleArcs=[{"name": "a", "args": {"center_x": 0, "center_y": 0, "radius": 2}}],
            Bars=[{"name": "b", "args": {"x_left": 5, "x_right": 6, "y_bottom": 0, "y_top": -3}}],
            Labels=[{"name": "l", "args": {"position": {"x": -4, "y": 1}, "text": "hi"}}],
        )
        self.assertBox(box, (-4, 6, -3, 2))

    def test_bar_chart(self) -> None:
        plot = {"values": [12, -3, 7], "bar_width": 1, "bar_spacing": 0.2, "x_start": -2, "y_base": 1}
        self.assertBox(self._content(BarsPlots=[{"name": "sales", "args": plot}]), (-2, 1.4, -2, 13))

    def test_polygon_from_vertex_list(self) -> None:
        polygon = {"name": "P", "args": {"points": ["A", "B", "C"]}}
        box = self._content([point("A", 0, 0), point("B", 5, 0), point("C", 5, 7)], GenericPolygons=[polygon])
        self.assertBox(box, (0, 5, 0, 7))

    def test_view_keys_are_not_rendered_as_objects(self) -> None:
        state = graph("f", (-628, 628, -1, 1), True, WIDE_VIEW)
        for rendered in (render_text(state), render_min_json(state)):
            self.assertNotIn(CANVAS_SIZE_KEY, rendered)
            self.assertNotIn(CURVE_EXTENTS_KEY, rendered)


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
