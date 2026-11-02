"""Tests for parameters and sliders on a real canvas: the tools, objects following them, undo,
refusals, animation with a fake clock, render caches, copies and the slider panel.

The pure rules (names, ranges, live namespace, hints) are covered by pytest in
server_tests/test_parameters_pure.py.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional

import compiled_expression
from browser import html
from managers.parameter_manager import ParameterManager
from parameter_animation import ParameterAnimator, _Run
from parameter_panel import ParameterPanel
from rendering.signature_state import is_undefined, signature_state

from .test_tool_batch_results import _ToolBatchTestCase

_SET = {"value": None, "min": None, "max": None, "step": None}
_DRAW = {"name": None, "left_bound": None, "right_bound": None, "color": None, "undefined_at": None}


class FakeFrames:
    """A requestAnimationFrame double: frames run only when the test calls ``run``."""

    def __init__(self) -> None:
        self.callbacks: List[Callable[[float], None]] = []

    def request(self, callback: Callable[[float], None]) -> int:
        self.callbacks.append(callback)
        return len(self.callbacks)

    def run(self, timestamp_ms: float) -> None:
        pending, self.callbacks = self.callbacks, []
        for callback in pending:
            callback(timestamp_ms)


class _ParameterTestCase(_ToolBatchTestCase):
    """A canvas that owns the parameter namespace for the test (the page's is given back after)."""

    def setUp(self) -> None:
        super().setUp()
        self.manager: ParameterManager = self.canvas.drawable_manager.parameter_manager
        self._previous_owner = self.manager.claim_namespace()
        self.frames = FakeFrames()
        self.draws = 0
        self.manager.animator = ParameterAnimator(self.manager, self._count_draw, request_frame=self.frames.request)

    def tearDown(self) -> None:
        self.manager.stop_all_animations()
        self.manager.release_namespace(self._previous_owner)
        super().tearDown()

    def _count_draw(self) -> None:
        self.draws += 1

    def set(self, name: str, **args: Any) -> Dict[str, Any]:
        call = dict(_SET)
        call.update(args)
        call["name"] = name
        _, traced = self.run_batch(("set_parameter", call))
        return traced[0]

    def draw_function(self, expression: str, **args: Any) -> Dict[str, Any]:
        call = dict(_DRAW)
        call.update(args)
        call["function_string"] = expression
        _, traced = self.run_batch(("draw_function", call))
        return traced[0]

    def point(self, x: Any, y: Any, name: Optional[str] = None) -> Dict[str, Any]:
        _, traced = self.run_batch(("create_point", {"x": x, "y": y, "color": None, "name": name}))
        return traced[0]

    def assert_ok(self, traced: Dict[str, Any]) -> None:
        self.assertFalse(traced.get("is_error"), traced)

    def assert_error(self, traced: Dict[str, Any], *fragments: str) -> None:
        self.assertTrue(traced.get("is_error"), traced)
        for fragment in fragments:
            self.assertIn(fragment, str(traced["result"]))


class TestSetParameter(_ParameterTestCase):
    def test_create_with_defaults(self) -> None:
        traced = self.set("a")
        self.assert_ok(traced)
        parameter = self.manager.get_parameter("a")
        self.assertEqual((parameter.value, parameter.min, parameter.max, parameter.step), (1.0, -10.0, 10.0, 0.1))
        self.assertEqual(compiled_expression.parameter_value("a"), 1.0)
        self.assertFalse(parameter.is_renderable)

    def test_value_outside_given_range_is_refused_and_changes_nothing(self) -> None:
        depth = self.undo_depth()
        self.assert_error(self.set("a", value=12, min=-5, max=5), "above the max 5")
        self.assertIsNone(self.manager.get_parameter("a"))
        self.assertEqual(self.undo_depth(), depth)

    def test_value_outside_stored_range_widens_it(self) -> None:
        self.set("a", value=1, min=0, max=5)
        traced = self.set("a", value=8)
        self.assert_ok(traced)
        self.assertEqual(self.manager.get_parameter("a").max, 8.0)
        self.assertIn("range widened", str(traced["result"]))

    def test_each_set_is_one_undo_step(self) -> None:
        self.set("a", value=1)
        self.set("a", value=3)
        self.assertEqual(compiled_expression.parameter_value("a"), 3.0)
        self.canvas.undo_redo_manager.undo()
        self.assertEqual(self.manager.get_parameter("a").value, 1.0)
        self.assertEqual(compiled_expression.parameter_value("a"), 1.0)
        self.canvas.undo_redo_manager.undo()
        self.assertIsNone(self.manager.get_parameter("a"))
        self.assertIsNone(compiled_expression.parameter_value("a"))

    def test_refused_names(self) -> None:
        for name in ("x", "θ", "pi", "sin", "2a"):
            with self.subTest(name=name):
                self.assert_error(self.set(name, value=1), "cannot name a parameter")
        self.point(1, 2, "A")
        self.assert_error(self.set("A", value=1), "another object")

    def test_delete_is_refused_while_used(self) -> None:
        self.set("a", value=2)
        self.assert_ok(self.draw_function("a*x", name="f"))
        _, traced = self.run_batch(("delete_parameter", {"name": "a"}))
        self.assert_error(traced[0], "used by f")
        self.run_batch(("delete_function", {"name": "f"}))
        _, traced = self.run_batch(("delete_parameter", {"name": "a"}))
        self.assert_ok(traced[0])
        self.assertIsNone(compiled_expression.parameter_value("a"))


class TestObjectsFollowParameters(_ParameterTestCase):
    def test_function_follows_its_parameter(self) -> None:
        self.set("a", value=2)
        self.assert_ok(self.draw_function("a*x^2", name="f"))
        function = self.canvas.drawable_manager.drawables.Functions[0]
        self.assertEqual(function.function(3.0), 18.0)
        self.set("a", value=-1)
        self.assertEqual(function.function(3.0), -9.0)
        self.assertEqual(function.parameter_names(), ("a",))

    def test_vertical_asymptote_follows_the_parameter(self) -> None:
        self.set("a", value=1)
        self.draw_function("1/(x - a)", name="f")
        function = self.canvas.drawable_manager.drawables.Functions[0]
        self.assertTrue(any(abs(x - 1.0) < 1e-6 for x in function.vertical_asymptotes))
        self.set("a", value=3)
        self.assertTrue(any(abs(x - 3.0) < 1e-6 for x in function.vertical_asymptotes))

    def test_undefined_name_is_refused_with_hints(self) -> None:
        self.assert_error(self.draw_function("k*x"), "set_parameter", "search_tools")
        self.assert_error(self.draw_function("ax^2 + bx"), "a*x")
        self.set("a", value=1)
        self.assert_error(self.draw_function("a(x+1)"), "write a*(x+1)")
        self.assertEqual(len(self.canvas.drawable_manager.drawables.Functions), 0)

    def test_defined_point_follows_and_segments_follow_it(self) -> None:
        self.set("a", value=2)
        self.assert_ok(self.point("a", "a^2", "P"))
        point = self.canvas.get_point_by_name("P")
        self.assertEqual((point.x, point.y), (2.0, 4.0))
        self.run_batch(
            (
                "create_segment",
                {
                    "x1": 0,
                    "y1": 0,
                    "x2": 2,
                    "y2": 4,
                    "name": None,
                    "color": None,
                    "label_text": None,
                    "label_visible": None,
                },
            )
        )
        segment = self.canvas.drawable_manager.drawables.Segments[0]
        self.assertIs(segment.point2, point)
        self.set("a", value=1)
        self.assertEqual((point.x, point.y), (1.0, 1.0))
        expected = segment._calculate_line_algebraic_formula()
        self.assertEqual(segment.line_formula, expected)

    def test_defined_point_never_reuses_a_point_at_its_spot(self) -> None:
        self.point(2, 4, "B")
        self.set("a", value=2)
        self.point("a", "a^2", "P")
        names = sorted(p.name for p in self.canvas.drawable_manager.drawables.Points)
        self.assertEqual(names, ["B", "P"])

    def test_undefined_point_is_marked_and_skipped_by_renderers(self) -> None:
        self.set("a", value=4)
        self.point("sqrt(a)", "0", "P")
        point = self.canvas.get_point_by_name("P")
        self.set("a", value=-1)
        self.assertTrue(point.undefined)
        self.assertTrue(is_undefined(point))
        self.assertEqual(point.x, 2.0)  # stays at its last position
        self.set("a", value=9)
        self.assertFalse(point.undefined)
        self.assertEqual(point.x, 3.0)

    def test_transforms_refuse_to_move_a_defined_point(self) -> None:
        self.set("a", value=1)
        self.point("a", "0", "P")
        depth = self.undo_depth()
        _, traced = self.run_batch(("translate_object", {"name": "P", "x_offset": 1, "y_offset": 0}))
        self.assert_error(traced[0], "defined by parameters", "update_point")
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.canvas.get_point_by_name("P").x, 1.0)

    def test_update_point_with_numbers_frees_and_with_expressions_defines(self) -> None:
        self.set("a", value=1)
        self.point(5, 5, "P")
        args = {"point_name": "P", "new_name": None, "new_color": None}
        _, traced = self.run_batch(("update_point", dict(args, new_x="a", new_y="2*a")))
        self.assert_ok(traced[0])
        point = self.canvas.get_point_by_name("P")
        self.assertEqual((point.x, point.y, point.x_expression), (1.0, 2.0, "a"))
        _, traced = self.run_batch(("update_point", dict(args, new_x=7, new_y=8)))
        self.assert_ok(traced[0])
        self.assertFalse(point.has_definition)
        self.set("a", value=3)
        self.assertEqual((point.x, point.y), (7.0, 8.0))

    def test_update_point_frees_a_used_point_and_one_where_it_stands(self) -> None:
        self.set("a", value=1)
        self.point("a", "0", "P")
        segment_args = {"name": None, "color": None, "label_text": None, "label_visible": None}
        self.run_batch(("create_segment", dict(segment_args, x1=1, y1=0, x2=4, y2=4)))
        point = self.canvas.get_point_by_name("P")
        segment = self.canvas.drawable_manager.drawables.Segments[0]
        self.assertIs(segment.point1, point)
        args = {"point_name": "P", "new_name": None, "new_color": None}
        depth = self.undo_depth()
        _, traced = self.run_batch(("update_point", dict(args, new_x=1, new_y=0)))  # where it stands
        self.assert_ok(traced[0])
        self.assertFalse(point.has_definition)
        self.assertEqual(self.undo_depth(), depth + 1)
        self.set("a", value=4)
        self.assertEqual((point.x, point.y), (1.0, 0.0))
        self.run_batch(("update_point", dict(args, new_x="a", new_y="0")))
        _, traced = self.run_batch(("update_point", dict(args, new_x=2, new_y=3)))  # used by the segment
        self.assert_ok(traced[0])
        self.assertEqual((point.x, point.y, point.has_definition), (2.0, 3.0, False))
        self.assertIs(segment.point1, point)
        self.assertEqual(segment.line_formula, segment._calculate_line_algebraic_formula())
        _, traced = self.run_batch(("translate_object", {"name": "P", "x_offset": 1, "y_offset": 0}))
        self.assert_ok(traced[0])

    def test_circle_radius_follows_and_is_named_after_it(self) -> None:
        self.set("b", value=2)
        _, traced = self.run_batch(
            ("create_circle", {"center_x": 0, "center_y": 0, "radius": "b", "color": None, "name": None})
        )
        self.assert_ok(traced[0])
        circle = self.canvas.drawable_manager.drawables.Circles[0]
        self.assertTrue(circle.name.endswith("(b)"), circle.name)
        self.set("b", value=-3)
        self.assertEqual(circle.radius, 3.0)
        self.set("b", value=0)
        self.assertTrue(is_undefined(circle))
        _, traced = self.run_batch(("scale_object", {"name": circle.name, "sx": 2, "sy": 2, "cx": 0, "cy": 0}))
        self.assert_error(traced[0], "radius defined by parameters")

    def test_evaluate_expression_reads_parameters(self) -> None:
        self.set("a", value=2)
        _, traced = self.run_batch(("evaluate_expression", {"expression": "a*3 + 1", "variables": None}))
        self.assert_ok(traced[0])
        self.assertEqual(float(traced[0]["result"]), 7.0)

    def test_render_signature_changes_with_the_value(self) -> None:
        self.set("a", value=1)
        self.draw_function("a*x", name="f")
        self.run_batch(
            (
                "draw_parametric_function",
                {
                    "x_expression": "a*cos(t)",
                    "y_expression": "sin(t)",
                    "name": None,
                    "t_min": None,
                    "t_max": None,
                    "color": None,
                },
            )
        )
        self.run_batch(
            (
                "draw_polar_function",
                {"r_expression": "a*(1 + cos(θ))", "name": None, "theta_min": None, "theta_max": None, "color": None},
            )
        )
        drawables = self.canvas.drawable_manager.drawables
        curves = [drawables.Functions[0], drawables.ParametricFunctions[0], drawables.PolarFunctions[0]]
        before = [repr(signature_state(curve)) for curve in curves]
        self.set("a", value=2)
        after = [repr(signature_state(curve)) for curve in curves]
        for curve, old, new in zip(curves, before, after):
            with self.subTest(curve=curve.get_class_name()):
                self.assertNotEqual(old, new)


class TestCopiesAndRestore(_ParameterTestCase):
    def test_copies_never_evaluate_with_the_namespace_emptied(self) -> None:
        self.set("a", value=2)
        self.draw_function("1/(x - a)", name="f")
        self.run_batch(
            (
                "draw_piecewise_function",
                {
                    "pieces": [
                        {
                            "expression": "a*x",
                            "left": None,
                            "right": 0,
                            "left_inclusive": None,
                            "right_inclusive": None,
                            "undefined_at": None,
                        },
                        {
                            "expression": "1/(x-a)",
                            "left": 0,
                            "right": None,
                            "left_inclusive": None,
                            "right_inclusive": None,
                            "undefined_at": None,
                        },
                    ],
                    "name": None,
                    "color": None,
                },
            )
        )
        self.run_batch(
            (
                "draw_polar_function",
                {"r_expression": "a + cos(θ)", "name": None, "theta_min": None, "theta_max": None, "color": None},
            )
        )
        drawables = list(self.canvas.drawable_manager.drawables.get_all())
        compiled_expression.set_parameter_values({})
        try:
            copies = [copy.deepcopy(drawable) for drawable in drawables]
        finally:
            self.manager.sync_namespace()
        piecewise = [c for c in copies if c.get_class_name() == "PiecewiseFunction"][0]
        original = [d for d in drawables if d.get_class_name() == "PiecewiseFunction"][0]
        self.assertEqual(piecewise.vertical_asymptotes, original.vertical_asymptotes)

    def test_undo_resyncs_the_namespace_and_stops_animation(self) -> None:
        self.set("a", value=0, min=0, max=1)
        _, traced = self.run_batch(
            ("animate_parameter", {"name": "a", "start": None, "end": None, "duration_seconds": 1, "repeat": "loop"})
        )
        self.assert_ok(traced[0])
        self.assertTrue(self.manager.animating)
        self.canvas.undo_redo_manager.undo()
        self.assertFalse(self.manager.animating)
        self.assertEqual(self.manager.animator.running, [])

    def test_workspace_round_trip_restores_parameters_before_their_users(self) -> None:
        self.set("a", value=2, min=-5, max=5, step=0.5)
        self.point(2, 4, "B")
        self.point("a", "a^2", "P")
        self.draw_function("a*x", name="f")
        state = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.canvas.clear()
        self.assertIsNone(compiled_expression.parameter_value("a"))
        self.workspace_manager._restore_workspace_state(state)
        parameter = self.manager.get_parameter("a")
        self.assertEqual((parameter.value, parameter.min, parameter.max, parameter.step), (2.0, -5.0, 5.0, 0.5))
        names = sorted(p.name for p in self.canvas.drawable_manager.drawables.Points)
        self.assertEqual(names, ["B", "P"])
        self.assertEqual(self.canvas.get_point_by_name("P").y_expression, "a^2")
        self.assertEqual(self.canvas.drawable_manager.drawables.Functions[0].function(1.0), 2.0)


class TestNamespaceOwner(_ParameterTestCase):
    def test_other_canvases_never_write_the_namespace(self) -> None:
        self.set("a", value=2)
        from canvas import Canvas

        other = Canvas(300, 300, draw_enabled=False)
        other.drawable_manager.parameter_manager.set_parameter("q", value=5)
        self.assertIsNone(compiled_expression.parameter_value("q"))
        self.assertEqual(compiled_expression.parameter_value("a"), 2.0)

    def test_live_globals_see_adds_and_deletes(self) -> None:
        compiled = compiled_expression.compile_expression("c*x + d", ("x",))
        f = compiled.bind(parameters=True)
        compiled_expression.set_parameter_values({"c": 2.0, "d": 1.0})
        try:
            self.assertEqual(f(3.0), 7.0)
            compiled_expression.set_parameter_values({"c": 2.0})
            with self.assertRaises(NameError):
                f(3.0)
        finally:
            self.manager.sync_namespace()


class TestAnimation(_ParameterTestCase):
    def animate(self, **args: Any) -> Dict[str, Any]:
        call = {"name": "a", "start": None, "end": None, "duration_seconds": None, "repeat": None}
        call.update(args)
        _, traced = self.run_batch(("animate_parameter", call))
        return traced[0]

    def test_once_runs_from_start_to_end_and_is_one_undo_step(self) -> None:
        self.set("a", value=5, min=0, max=10)
        self.draw_function("a*x", name="f")
        depth = self.undo_depth()
        self.assert_ok(self.animate(start=0, end=1, duration_seconds=1))
        self.assertEqual(self.undo_depth(), depth + 1)
        self.frames.run(1000.0)
        self.assertEqual(self.manager.get_parameter("a").value, 0.0)
        self.frames.run(1500.0)
        self.assertAlmostEqual(self.manager.get_parameter("a").value, 0.5)
        self.assertAlmostEqual(self.canvas.drawable_manager.drawables.Functions[0].function(2.0), 1.0)
        self.frames.run(2000.0)
        self.assertEqual(self.manager.get_parameter("a").value, 1.0)
        self.assertFalse(self.manager.animating)
        self.assertEqual(self.frames.callbacks, [])
        self.assertEqual(self.undo_depth(), depth + 1)
        self.canvas.undo_redo_manager.undo()
        self.assertEqual(self.manager.get_parameter("a").value, 5.0)

    def test_stop_keeps_the_current_value(self) -> None:
        self.set("a", value=0, min=0, max=10)
        self.animate(start=0, end=10, duration_seconds=10, repeat="loop")
        self.frames.run(0.0)
        self.frames.run(3000.0)
        traced = self.animate(repeat="stop")
        self.assert_ok(traced)
        self.assertFalse(self.manager.animating)
        self.assertAlmostEqual(self.manager.get_parameter("a").value, 3.0)

    def test_set_parameter_stops_its_animation(self) -> None:
        self.set("a", value=0, min=0, max=10)
        self.animate(repeat="bounce")
        traced = self.set("a", value=4)
        self.assert_ok(traced)
        self.assertFalse(self.manager.animating)
        self.assertIn("animation was stopped", str(traced["result"]))

    def test_a_failing_frame_ends_the_animation_cleanly(self) -> None:
        self.set("a", value=0, min=0, max=1)
        self.animate(repeat="loop")

        def broken_draw() -> None:
            raise RuntimeError("boom")

        self.manager.animator._draw = broken_draw
        self.frames.run(0.0)
        self.assertFalse(self.manager.animating)
        self.assertEqual(self.frames.callbacks, [])

    def test_repeat_modes(self) -> None:
        loop = _Run(0.0, 10.0, 1.0, "loop")
        bounce = _Run(0.0, 10.0, 1.0, "bounce")
        loop.value_at(0.0)
        bounce.value_at(0.0)
        self.assertAlmostEqual(loop.value_at(1250.0)[0], 2.5)
        self.assertAlmostEqual(bounce.value_at(1250.0)[0], 7.5)
        self.assertFalse(bounce.value_at(5000.0)[1])

    def test_duration_is_checked(self) -> None:
        self.set("a", value=0)
        self.assert_error(self.animate(duration_seconds=0), "duration_seconds")
        self.assert_error(self.animate(duration_seconds=601), "duration_seconds")


class TestSliderPanel(_ParameterTestCase):
    def make_panel(self) -> ParameterPanel:
        self.element = html.DIV()
        return ParameterPanel(
            self.manager, self.canvas, element=self.element, request_frame=self.frames.request
        ).attach()

    def test_rows_follow_the_parameters(self) -> None:
        panel = self.make_panel()
        self.assertEqual(self.element.style.display, "none")
        self.set("a", value=1.5, min=-5, max=5, step=0.1)
        self.set("b", value=2)
        self.assertEqual(sorted(panel._rows), ["a", "b"])
        self.assertEqual(self.element.style.display, "block")
        self.assertEqual(panel._rows["a"].value_text.text, "= 1.5")
        self.assertEqual(panel._rows["a"].slider.attrs["step"], "any")
        self.set("a", value=-2.4)
        panel.update_values()
        self.assertEqual(panel._rows["a"].value_text.text, "= -2.4")

    def test_dragging_sets_live_values_with_one_undo_entry(self) -> None:
        panel = self.make_panel()
        self.set("a", value=0, min=0, max=10, step=0.5)
        depth = self.undo_depth()
        for position in ("1.2", "3.9", "7.74"):
            panel._on_input("a", _FakeEvent(position))
        self.assertEqual(self.manager.get_parameter("a").value, 7.5)
        self.assertEqual(self.undo_depth(), depth + 1)
        self.assertEqual(len(self.frames.callbacks), 1)  # one coalesced draw
        panel._on_change("a")
        self.canvas.undo_redo_manager.undo()
        self.assertEqual(self.manager.get_parameter("a").value, 0.0)

    def test_play_toggles_a_bounce(self) -> None:
        panel = self.make_panel()
        self.set("a", value=0, min=0, max=2)
        panel._on_play("a")
        self.assertTrue(self.manager.animating)
        self.assertEqual(self.manager.get_parameter("a").animation["repeat"], "bounce")
        panel._on_play("a")
        self.assertFalse(self.manager.animating)


class _FakeEvent:
    def __init__(self, value: str) -> None:
        self.target = _FakeTarget(value)


class _FakeTarget:
    def __init__(self, value: str) -> None:
        self.value = value
