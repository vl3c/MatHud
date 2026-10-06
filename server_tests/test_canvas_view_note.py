"""Tests for static/canvas_view_note.py: telling the model when something it just drew can't be seen.

A note is only about the objects a tool batch created or moved, and a missed note is
far better than a false one, so most tests check that ordinary drawing stays silent.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unittest
from typing import Any, Dict, List, Optional, Tuple

from server_tests.test_canvas_state_formatter import point
from static.canvas_state_formatter import (
    CANVAS_SIZE_KEY,
    CHANGES_HEADER,
    CURRENT_HEADER,
    CURVE_EXTENTS_KEY,
    OMITTED_NOTE,
    render_min_json,
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
# A view of +-10 on an 800 x 600 canvas: 40 px per unit.
VIEW_10: Dict[str, Any] = dict(
    DEFAULT_VIEW,
    Cartesian_System_Visibility={"left_bound": -10, "right_bound": 10, "top_bound": 7.5, "bottom_bound": -7.5},
)


def scene(points: List[Dict[str, Any]], view: Optional[Dict[str, Any]] = None, **buckets: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {"Points": list(points)}
    state.update(buckets)
    state.update(copy.deepcopy(view if view is not None else DEFAULT_VIEW))
    return state


def empty(view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return scene([], view)


def add(state: Dict[str, Any], *points: Dict[str, Any], **buckets: Any) -> Dict[str, Any]:
    grown = copy.deepcopy(state)
    grown["Points"].extend(points)
    for bucket, items in buckets.items():
        grown.setdefault(bucket, []).extend(items)
    return grown


def with_bounds(state: Dict[str, Any], left: float, right: float, bottom: float, top: float) -> Dict[str, Any]:
    moved = copy.deepcopy(state)
    moved["Cartesian_System_Visibility"] = {
        "left_bound": left,
        "right_bound": right,
        "bottom_bound": bottom,
        "top_bound": top,
    }
    return moved


def polygon(name: str, bucket_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"name": name, "args": bucket_args or {f"p{i + 1}": vertex for i, vertex in enumerate(name)}}


def circle(name: str, center: str, radius: float, **args: Any) -> Dict[str, Any]:
    return {"name": name, "args": dict({"center": center, "radius": radius}, **args)}


def segment(p1: str, p2: str) -> Dict[str, Any]:
    return {"name": p1 + p2, "args": {"p1": p1, "p2": p2}}


def edges(names: str) -> List[Dict[str, Any]]:
    return [segment(a, b) for a, b in zip(names, names[1:] + names[0])]


def triangle_scene(
    name: str, corners: List[Tuple[float, float]], view: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """A triangle as create_polygon leaves it: three points, three edges and the triangle."""
    points = [point(n, x, y) for n, (x, y) in zip(name, corners)]
    return scene(points, view, Segments=edges(name), Triangles=[polygon(name)])


def with_graph(
    state: Dict[str, Any],
    name: str,
    box: Tuple[float, float, float, float],
    clipped: bool,
    waves: bool = True,
    spiky: bool = False,
    resolved: bool = False,
    period: Optional[float] = 6.3,
    **args: Any,
) -> Dict[str, Any]:
    """``state`` plus a function graph whose client-sampled box is ``box``.

    By default an aliased wave with sin's period, as the client reports sin at the default view.
    """
    grown = add(state, Functions=[{"name": name, "args": dict({"function_string": f"{name}(x)"}, **args)}])
    grown.setdefault(CURVE_EXTENTS_KEY, {}).setdefault("Functions", {})[name] = {
        "box": list(box),
        "clipped": clipped,
        "waves": waves,
        "spiky": spiky,
        "resolved": resolved,
        "period": period,
    }
    return grown


TINY = [(0, 0), (6, 0), (2, 4)]


class TestTooSmall(unittest.TestCase):
    maxDiff = None

    def test_tiny_triangle_drawn_at_the_default_view(self) -> None:
        self.assertEqual(
            view_note(empty(), triangle_scene("ABC", TINY)),
            "View note: the new ABC spans only ~6x4 px on screen (x 0..6, y 0..4; view x -400..400, "
            "y -300..300). Offer to zoom to about x -0.8..6.8, y -0.8..4.8 (zoom center_x=3, center_y=2, "
            "range_val=3.8, range_axis=x); don't change the view unless the user agrees.",
        )

    def test_triangle_and_circumcircle_are_named_together(self) -> None:
        after = add(triangle_scene("ABC", TINY), point("O", 3, 1), Circles=[circle("O(3.16)", "O", math.sqrt(10))])
        note = view_note(empty(), after) or ""
        self.assertIn("the new O(3.16), ABC span only ~6x6 px on screen", note)

    def test_view_bounds_are_rounded_like_the_view_line(self) -> None:
        note = view_note(empty(WIDE_VIEW), triangle_scene("ABC", TINY, WIDE_VIEW)) or ""
        self.assertIn("view x -628..628, y -481.5..481.5", note)

    def test_a_readable_drawing_is_fine(self) -> None:
        self.assertIsNone(view_note(empty(), triangle_scene("ABC", [(0, 0), (300, 0), (100, 200)])))
        self.assertIsNone(view_note(empty(), triangle_scene("ABC", [(0, 0), (17, 0), (5, 9)])))

    def test_the_threshold_is_sixteen_pixels(self) -> None:
        self.assertIsNotNone(view_note(empty(), triangle_scene("ABC", [(0, 0), (15, 0), (5, 9)])))

    def test_pixels_follow_the_zoom(self) -> None:
        self.assertIsNone(view_note(empty(VIEW_10), triangle_scene("ABC", TINY, VIEW_10)))

    def test_points_and_labels_are_never_too_small(self) -> None:
        self.assertIsNone(view_note(empty(), scene([point("A", 0, 0), point("B", 1, 0)])))
        label = {"name": "L", "args": {"position": {"x": 0, "y": 0}, "text": "hi"}}
        self.assertIsNone(view_note(empty(), scene([], Labels=[label])))

    def test_without_a_canvas_size_the_view_fraction_is_used(self) -> None:
        after = triangle_scene("ABC", TINY)
        del after[CANVAS_SIZE_KEY]
        self.assertIn("the new ABC spans only a sliver of the view", view_note(empty(), after) or "")

    def test_extents_below_the_finest_grid_count_as_a_point(self) -> None:
        self.assertIsNone(view_note(empty(), triangle_scene("ABC", [(0, 0), (1e-9, 0), (0, 1e-9)])))
        note = view_note(empty(), triangle_scene("ABC", [(0, 0), (1e-4, 0), (0, 1e-4)])) or ""
        self.assertIn("range_val=8.4e-5", note)

    def test_huge_coordinates_are_written_in_scientific_notation(self) -> None:
        after = triangle_scene("ABC", [(1e300, 0), (1e300, 1e290), (1.0000001e300, 0)])
        note = view_note(empty(), after) or ""
        self.assertIsNone(re.search(r"\d{16}", note), note)


class TestMarkersOnReadableShapes(unittest.TestCase):
    """Small shapes drawn on a readable shape are part of it: never a note.

    Each marker is checked drawn after the figure (it builds on an earlier shape) and drawn in
    the same batch as the figure (it is attached to a readable shape).
    """

    def setUp(self) -> None:
        self.figure = triangle_scene("ABC", [(0, 0), (240, 0), (0, 180)])

    def assertSilent(self, after: Dict[str, Any], figure: Optional[Dict[str, Any]] = None) -> None:  # noqa: N802
        self.assertIsNone(view_note(figure or self.figure, after))
        self.assertIsNone(view_note(empty(), after))

    def test_right_angle_square_at_a_vertex(self) -> None:
        marker = add(
            self.figure,
            point("D", 14, 0),
            point("E", 14, 14),
            point("F", 0, 14),
            Quadrilaterals=[{"name": "ADEF", "args": {"p1": "A", "p2": "D", "p3": "E", "p4": "F"}}],
            Segments=edges("ADEF"),
        )
        self.assertSilent(marker)

    def test_highlight_circle_on_a_vertex(self) -> None:
        self.assertSilent(add(self.figure, Circles=[circle("B(8)", "B", 8)]))
        self.assertSilent(add(self.figure, point("G", 240, 0), Circles=[circle("G(8)", "G", 8)]))

    def test_angle_arc_at_a_vertex(self) -> None:
        arc = {"name": "arc", "args": {"center_x": 240, "center_y": 0, "radius": 12, "point1_name": "B"}}
        self.assertSilent(add(self.figure, CircleArcs=[arc]))

    def test_a_mark_beside_a_corner_touching_nothing(self) -> None:
        """The corner check alone: a small square just off vertex B, clear of both edges."""
        mark = add(
            self.figure,
            point("P", 242, -14),
            point("Q", 254, -14),
            point("R", 254, -2),
            point("S", 242, -2),
            Quadrilaterals=[{"name": "PQRS", "args": {"p1": "P", "p2": "Q", "p3": "R", "p4": "S"}}],
        )
        # Drawn after the figure, without touching it: only the corner check keeps it silent.
        self.assertIsNone(view_note(self.figure, mark))
        apart = copy.deepcopy(mark)
        for item in apart["Points"][3:]:
            item["args"]["position"]["x"] += 100
        self.assertIn("the new PQRS spans only ~12x12 px", view_note(self.figure, apart) or "")

    def test_tick_marks_across_an_edge(self) -> None:
        """The outline check alone: ticks across the middle of an edge, far from its corners."""
        self.assertSilent(add(self.figure, point("T1", 60, -6), point("T2", 60, 6), Segments=[segment("T1", "T2")]))

    def test_small_circle_on_a_readable_circle(self) -> None:
        figure = scene([point("O", 0, 0)], Circles=[circle("O(200)", "O", 200)])
        self.assertSilent(add(figure, point("P", 200, 0), Circles=[circle("P(5)", "P", 5)]), figure)

    def test_label_or_point_next_to_a_vertex(self) -> None:
        label = {"name": "alpha", "args": {"position": {"x": 0.35, "y": 0.15}, "text": "alpha"}}
        small = triangle_scene("ABC", [(0, 0), (4, 0), (1, 3)], VIEW_10)
        self.assertIsNone(view_note(small, add(small, Labels=[label])))
        self.assertIsNone(view_note(small, add(small, point("D", 0.5, 0.2))))

    def test_a_speck_away_from_everything_is_noted(self) -> None:
        speck = add(
            self.figure,
            point("P", 300, 200),
            point("Q", 304, 200),
            point("R", 302, 203),
            Segments=edges("PQR"),
            Triangles=[polygon("PQR")],
        )
        self.assertIn("the new PQR spans only ~4x3 px", view_note(self.figure, speck) or "")


class TestBuildingOnEarlierShapes(unittest.TestCase):
    """A small new shape that touches what was drawn before builds on a scene the user already sees."""

    def test_circumcircle_after_a_declined_tiny_triangle(self) -> None:
        speck = triangle_scene("ABC", TINY)
        self.assertIsNotNone(view_note(empty(), speck))
        circumcircle = add(speck, point("D", 3, 1), Circles=[circle("D(3.16)", "D", math.sqrt(10))])
        self.assertIsNone(view_note(speck, circumcircle))

    def test_a_diagonal_after_the_user_zoomed_out(self) -> None:
        square = scene(
            [point("A", 0, 0), point("B", 4, 0), point("C", 4, 4), point("D", 0, 4)],
            VIEW_10,
            Segments=edges("ABCD"),
            Quadrilaterals=[polygon("ABCD")],
        )
        zoomed_out = with_bounds(square, -400, 400, -300, 300)
        self.assertIsNone(view_note(square, zoomed_out))
        self.assertIsNone(view_note(zoomed_out, add(zoomed_out, Segments=[segment("A", "C")])))

    def test_a_ring_around_a_point_that_was_there(self) -> None:
        lone = scene([point("P", 100, 50)])
        self.assertIsNone(view_note(lone, add(lone, Circles=[circle("P(5)", "P", 5)])))

    def test_a_separate_speck_is_still_noted(self) -> None:
        speck = triangle_scene("ABC", TINY)
        far = add(speck, point("P", 300, 300), point("Q", 304, 300), point("R", 302, 303), Triangles=[polygon("PQR")])
        self.assertIn("the new PQR spans only", view_note(speck, far) or "")

    def test_a_clipped_graph_has_no_size_of_its_own(self) -> None:
        after = with_graph(empty(), "h", (0, 5, 0, 5), True, waves=False)
        self.assertIsNone(view_note(empty(), after))
        bounded = with_graph(empty(), "h", (0, 5, 0, 5), False, waves=False, left_bound=0, right_bound=5)
        self.assertIsNotNone(view_note(empty(), bounded))


class TestWhatCountsAsNew(unittest.TestCase):
    def test_recolouring_is_no_change(self) -> None:
        before = triangle_scene("ABC", TINY)
        after = copy.deepcopy(before)
        after["Triangles"][0]["args"]["color"] = "red"
        after["Points"][0]["args"]["color"] = "blue"
        self.assertIsNone(view_note(before, after))

    def test_an_unchanged_speck_is_not_noted_again(self) -> None:
        before = triangle_scene("ABC", TINY)
        self.assertIsNone(view_note(before, copy.deepcopy(before)))

    def test_moving_a_point_changes_the_shapes_through_it(self) -> None:
        before = scene([point("O", 0, 0)], Circles=[circle("O(5)", "O", 5)])
        after = scene([point("O", 5000, 0)], Circles=[circle("O(5)", "O", 5)])
        self.assertIn("the new or changed O(5) is outside the view", view_note(before, after) or "")

    def test_new_or_changed_follows_the_named_objects(self) -> None:
        before = scene([point("O", 0, 0)], Circles=[circle("O(5)", "O", 5)])
        after = add(before, point("P", 5000, 0), Circles=[circle("P(5)", "P", 5)])
        self.assertIn("the new P(5) is outside", view_note(before, after) or "")

    def test_a_view_change_alone_is_never_noted(self) -> None:
        before = triangle_scene("ABC", [(0, 0), (300, 0), (100, 200)])
        for view in ((-40000, 40000, -30000, 30000), (5000, 5800, -300, 300)):
            self.assertIsNone(view_note(before, with_bounds(before, *view)))
        graph = with_graph(empty(), "f", (-400, 400, -100, 100), True)
        panned = with_graph(empty(), "f", (1000, 1800, -100, 100), True)
        self.assertIsNone(view_note(graph, with_bounds(panned, 1000, 1800, 1000, 1600)))

    def test_without_a_previous_canvas_nothing_is_said(self) -> None:
        self.assertIsNone(view_note(None, triangle_scene("ABC", TINY)))


class TestFunctionsAndCurves(unittest.TestCase):
    maxDiff = None

    def test_flat_sine_at_the_default_view(self) -> None:
        self.assertEqual(
            view_note(empty(WIDE_VIEW), with_graph(empty(WIDE_VIEW), "f", (-628, 628, -1, 1), True)),
            "View note: the new f varies only ~2 px vertically on screen (y -1..1 over x -628..628; "
            "view x -628..628, y -481.5..481.5). Offer to zoom to about x -5.3..5.3, y -4.1..4.1 "
            "(zoom center_x=0, center_y=0, range_val=5.3, range_axis=x); "
            "don't change the view unless the user agrees.",
        )

    def test_bounded_parabola_is_tiny(self) -> None:
        after = with_graph(empty(WIDE_VIEW), "f", (-2, 2, -2, 2), False, left_bound=-2, right_bound=2)
        self.assertIn("the new f spans only ~4x4 px on screen", view_note(empty(WIDE_VIEW), after) or "")

    def test_readable_graphs_are_fine(self) -> None:
        for box in ((-400, 400, 0, 160000), (-400, 400, -100, 100)):
            self.assertIsNone(view_note(empty(), with_graph(empty(), "f", box, True)))

    def test_lines_and_constants_are_never_flat(self) -> None:
        """Zooming cannot make y = 0.01x + 5 or a constant more readable."""
        self.assertIsNone(view_note(empty(), with_graph(empty(), "f", (-400, 400, 3, 3), True, waves=False)))
        self.assertIsNone(
            view_note(empty(WIDE_VIEW), with_graph(empty(WIDE_VIEW), "h", (-628, 628, -1.3, 11.3), True, waves=False))
        )

    def test_graphs_with_a_spike_or_asymptote_are_never_flat(self) -> None:
        """1/x or tan sampled across the view look flat once the pole's samples are trimmed."""
        base = empty(WIDE_VIEW)
        self.assertIsNone(view_note(base, with_graph(base, "r", (-628, 628, -0.03, 0.03), True, spiky=True)))
        listed = with_graph(base, "r", (-628, 628, -0.03, 0.03), True, vertical_asymptotes=[0])
        self.assertIsNone(view_note(base, listed))
        elsewhere = with_graph(base, "r", (-628, 628, -0.03, 0.03), True, period=0.2, vertical_asymptotes=[5000])
        self.assertIsNotNone(view_note(base, elsewhere))

    def test_a_wave_the_samples_resolve_is_drawn_as_it_is(self) -> None:
        """0.1*sin(x) at +-10: a low wave, and zooming keeps the aspect ratio, so no note."""
        resolved = with_graph(empty(VIEW_10), "a", (-10, 10, -0.1, 0.1), True, resolved=True, period=None)
        self.assertIsNone(view_note(empty(VIEW_10), resolved))

    def test_a_wave_already_wide_on_screen_is_not_flat(self) -> None:
        """sin(x) at +-100: a 40 px period, a low wave the screen draws as one."""
        view = with_bounds(empty(), -100, 100, -75, 75)
        self.assertIsNone(view_note(view, with_graph(view, "s", (-100, 100, -1, 1), True, period=6.3)))

    def test_no_note_when_the_zoom_would_show_less_than_a_period(self) -> None:
        """0.001*sin(x) at the default view: a zoom high enough to show it shows a straight piece."""
        tiny_wave = with_graph(empty(WIDE_VIEW), "b", (-628, 628, -0.001, 0.001), True, period=6.3)
        self.assertIsNone(view_note(empty(WIDE_VIEW), tiny_wave))
        unknown = with_graph(empty(WIDE_VIEW), "f", (-628, 628, -1, 1), True, period=None)
        self.assertIsNone(view_note(empty(WIDE_VIEW), unknown))

    def test_a_flat_graph_mostly_beside_the_view_is_not_flat(self) -> None:
        after = with_graph(empty(), "f", (395, 1000, -1, 1), False, left_bound=395, right_bound=1000)
        self.assertIsNone(view_note(empty(), after))

    def test_a_long_bounded_wave_at_a_close_view_is_fine(self) -> None:
        after = with_graph(empty(VIEW_10), "s", (-1000, 1000, -1, 1), False, left_bound=-1000, right_bound=1000)
        self.assertIsNone(view_note(empty(VIEW_10), after))

    def test_a_graph_drawn_above_the_view(self) -> None:
        after = with_graph(empty(), "f", (-400, 400, 1000, 1160), True)
        self.assertIn("the new f is outside the view", view_note(empty(), after) or "")

    def test_a_graph_without_measurement_is_never_noted(self) -> None:
        after = with_graph(empty(WIDE_VIEW), "f", (-628, 628, -1, 1), True)
        del after[CURVE_EXTENTS_KEY]
        self.assertIsNone(view_note(empty(WIDE_VIEW), after))

    def test_a_tangent_on_a_graph_is_part_of_it(self) -> None:
        before = with_graph(empty(VIEW_10), "f", (-10, 10, -18, 18), True)
        after = add(before, point("T1", 0.9, -2.1), point("T2", 1.1, -1.9), Segments=[segment("T1", "T2")])
        self.assertIsNone(view_note(before, after))

    def test_a_small_shape_next_to_an_unmeasured_graph_is_left_alone(self) -> None:
        """Only a batch's new curves are measured; a tangent on an older graph could lie on it."""
        before = scene([], Functions=[{"name": "f", "args": {"function_string": "x^2"}}])
        after = add(before, point("T1", 0.1, -0.8), point("T2", 1.9, 2.8), Segments=[segment("T1", "T2")])
        self.assertIsNone(view_note(before, after))

    def test_parametric_curve(self) -> None:
        after = scene([], ParametricFunctions=[{"name": "p", "args": {"x_expression": "cos(t)"}}])
        after[CURVE_EXTENTS_KEY] = {"ParametricFunctions": {"p": {"box": [-1, 1, -1, 1], "clipped": False}}}
        self.assertIn("the new p spans only ~2x2 px", view_note(empty(), after) or "")


class TestOutsideTheView(unittest.TestCase):
    def test_a_circle_drawn_far_away(self) -> None:
        before = scene([point("A", 0, 0)], Circles=[circle("A(100)", "A", 100)])
        after = add(before, point("B", 5000, 5000), Circles=[circle("B(100)", "B", 100)])
        note = view_note(before, after) or ""
        self.assertIn("the new B(100) is outside the view (view x -400..400, y -300..300)", note)
        self.assertIn("Offer to move the view to about x 4600..5400, y 4700..5300", note)

    def test_the_suggestion_keeps_the_batch_on_screen_objects_in_view(self) -> None:
        """Data points fitted partly above the view: zoom out to all of them, not to the hidden ones."""
        after = scene([point("D1", 0, 0), point("D2", 100, 150), point("D3", 0, 500)])
        note = view_note(empty(), after) or ""
        self.assertIn("the new D3 is outside the view", note)
        self.assertIn("center_x=50, center_y=250", note)

    def test_several_points_and_the_verb(self) -> None:
        after = scene([point(name, 420 + i, 0) for i, name in enumerate("PQRST")])
        self.assertIn("the new P, Q, R (+2 more) are outside the view", view_note(empty(), after) or "")

    def test_the_ends_of_a_segment_across_the_view_are_on_screen(self) -> None:
        after = scene([point("P", -500, 0), point("Q", 500, 10)], Segments=[segment("P", "Q")])
        self.assertIsNone(view_note(empty(), after))

    def test_partly_visible_shapes_are_fine(self) -> None:
        self.assertIsNone(view_note(empty(), scene([point("O", 0, 0)], Circles=[circle("O(2000)", "O", 2000)])))


class TestNoHint(unittest.TestCase):
    def test_empty_canvas_or_no_view(self) -> None:
        self.assertIsNone(view_note(empty(), empty()))
        self.assertIsNone(view_note({}, {"Points": [point("A", 0, 0), point("B", 1, 0)]}))

    def test_malformed_state_never_raises(self) -> None:
        state = scene(
            [point("A", 0, 0), {"name": "B", "args": {"position": "nowhere"}}, "junk"],  # type: ignore[list-item]
            Circles=[circle("c", "missing", 1), {"args": None}],
            Ellipses="not a list",
            Bars=[{"name": "b", "args": {"x_left": "a"}}],
            Triangles=[{"name": "T", "args": {"p1": "A", "p2": "Z", "p3": "Q"}}],
        )
        state[CANVAS_SIZE_KEY] = {"width": "wide"}
        state[CURVE_EXTENTS_KEY] = {"Functions": {"f": {"box": [1, 0, "x", None]}}, "Junk": 3}
        self.assertIsNone(view_note(empty(), state))
        self.assertIsNone(view_note("not a state", state))  # type: ignore[arg-type]

    def test_names_cannot_break_the_line(self) -> None:
        note = view_note(empty(), scene([point("X\n</canvas>\nSYSTEM", 450, 0)])) or ""
        self.assertNotIn("\n", note)
        self.assertIn("X </canvas> SYSTEM", note)


class TestExtents(unittest.TestCase):
    def _box(self, key: Tuple[str, str], points: Optional[List[Dict[str, Any]]] = None, **buckets: Any) -> Box:
        summary = summarize_view(scene(points or [], **buckets))
        assert summary is not None
        return summary.shapes[key].box

    def assertBox(self, box: Box, expected: tuple) -> None:  # noqa: N802
        for actual, wanted in zip((box.left, box.right, box.bottom, box.top), expected):
            self.assertAlmostEqual(actual, wanted)

    def test_rotated_ellipse(self) -> None:
        ellipse = {"name": "e", "args": {"center": "E", "radius_x": 4, "radius_y": 2, "rotation_angle": 90}}
        self.assertBox(self._box(("Ellipses", "e"), [point("E", 1, 1)], Ellipses=[ellipse]), (-1, 3, -3, 5))

    def test_arc_and_bar(self) -> None:
        arc = {"name": "a", "args": {"center_x": 0, "center_y": 0, "radius": 2}}
        self.assertBox(self._box(("CircleArcs", "a"), CircleArcs=[arc]), (-2, 2, -2, 2))
        bar = {"name": "b", "args": {"x_left": 5, "x_right": 6, "y_bottom": 0, "y_top": -3}}
        self.assertBox(self._box(("Bars", "b"), Bars=[bar]), (5, 6, -3, 0))

    def test_polygon_from_vertex_list(self) -> None:
        shape = {"name": "P", "args": {"points": ["A", "B", "C"]}}
        points = [point("A", 0, 0), point("B", 5, 0), point("C", 5, 7)]
        self.assertBox(self._box(("GenericPolygons", "P"), points, GenericPolygons=[shape]), (0, 5, 0, 7))

    def test_measuring_keys_are_not_rendered_as_objects(self) -> None:
        state = with_graph(empty(WIDE_VIEW), "f", (-628, 628, -1, 1), True)
        for rendered in (render_text(state), render_min_json(state)):
            self.assertNotIn(CANVAS_SIZE_KEY, rendered)
            self.assertNotIn(CURVE_EXTENTS_KEY, rendered)


class TestPlacement(unittest.TestCase):
    """The note ends [canvas changes]; when the full canvas is sent instead it heads it."""

    def setUp(self) -> None:
        self.state = triangle_scene("ABC", TINY)
        note = view_note(empty(), self.state)
        assert note is not None
        self.note = note

    def test_canvas_changes_end_with_the_note(self) -> None:
        update = render_update(empty(), self.state, "text", view_note=self.note)
        self.assertTrue(update.startswith(CHANGES_HEADER + "\n+ A = (0, 0)"))
        self.assertTrue(update.endswith("\n" + self.note))
        self.assertTrue(render_update(empty(), self.state, "min_json", view_note=self.note).endswith(self.note))

    def test_full_canvas_update_carries_the_note(self) -> None:
        update = render_update(None, self.state, "text", view_note=self.note)
        self.assertTrue(
            update.startswith(CURRENT_HEADER + "\nview x [-400, 400] y [-300, 300]; grid 100\n" + self.note)
        )
        self.assertEqual(json.loads(render_min_json(self.state, view_note=self.note))["view_note"], self.note)

    def test_budget_trimming_keeps_the_note(self) -> None:
        crowded = scene([point(f"P{i}", (i % 20) * 0.3, (i // 20) * 0.3) for i in range(200)])
        note = f"{VIEW_NOTE_PREFIX} test"
        text = render_text(crowded, budget_tokens=400, view_note=note)
        self.assertIn(OMITTED_NOTE, text)
        self.assertEqual(text.split("\n")[1], note)
        self.assertLessEqual(estimate_tokens_from_text(text), 400)
        self.assertEqual(json.loads(render_min_json(crowded, budget_tokens=400, view_note=note))["view_note"], note)

    def test_unchanged_canvas_still_reports_a_note(self) -> None:
        self.assertEqual(
            render_update(self.state, self.state, "text", view_note=self.note), f"{CHANGES_HEADER}\n{self.note}"
        )
        self.assertEqual(render_update(self.state, self.state, "text"), "")


if __name__ == "__main__":
    unittest.main()
