"""Tests for scenario_hooks: the window hooks used by the scenario-testing harness."""

from __future__ import annotations

import json
import types
import unittest
from typing import Any, Dict, List

from browser import document, html
from canvas import Canvas
from function_registry import FunctionRegistry
from managers.action_trace_collector import ActionTraceCollector
from scenario_hooks import (
    WORKSPACE_TOOLS,
    ScenarioHooks,
    build_inspection,
    content_extent,
    fit_window,
    normalize_tool_calls,
    parse_options,
    reset_canvas_session,
    turn_status,
)
from turn_metrics import TurnMetricsCollector
from workspace_manager import WorkspaceManager


def _make_ai(canvas: Canvas) -> Any:
    """An AIInterface without __init__ (no DOM), wired to a real canvas and registry."""
    from ai_interface import AIInterface

    ai = AIInterface.__new__(AIInterface)
    ai.canvas = canvas
    ai.workspace_manager = WorkspaceManager(canvas)
    ai.available_functions = FunctionRegistry.get_available_functions(canvas, ai.workspace_manager)
    ai.undoable_functions = FunctionRegistry.get_undoable_functions()
    ai.is_processing = False
    ai._trace_collector = ActionTraceCollector()
    ai._turn_metrics_collector = TurnMetricsCollector(clock_ms=lambda: 0.0)
    return ai


def _call(tool: str, **arguments: Any) -> Dict[str, Any]:
    return {"function_name": tool, "arguments": arguments}


def _point_names(state: Dict[str, Any]) -> List[str]:
    return sorted(point["name"] for point in state.get("Points", []))


class TestScenarioHookHelpers(unittest.TestCase):
    """Pure helpers: option parsing and call normalization."""

    def test_parse_options_empty_values(self) -> None:
        for raw in (None, "", "  ", "undefined", "null"):
            self.assertEqual(parse_options(raw), {})

    def test_parse_options_object(self) -> None:
        self.assertEqual(parse_options('{"inspect": true}'), {"inspect": True})

    def test_parse_options_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_options("[1, 2]")

    def test_normalize_accepts_both_call_shapes(self) -> None:
        calls = normalize_tool_calls(
            [
                {"function_name": "create_point", "arguments": {"x": 1, "y": 2}, "id": "c1"},
                {"tool": "undo", "args": {}},
                {"tool": "redo"},
            ]
        )
        self.assertEqual(calls[0], {"function_name": "create_point", "arguments": {"x": 1, "y": 2}, "id": "c1"})
        self.assertEqual(calls[1], {"function_name": "undo", "arguments": {}})
        self.assertEqual(calls[2], {"function_name": "redo", "arguments": {}})

    def test_normalize_rejects_bad_calls(self) -> None:
        for raw in ({"tool": "undo"}, [1], [{"arguments": {}}], [{"tool": "undo", "args": [1]}]):
            with self.assertRaises(ValueError):
                normalize_tool_calls(raw)

    def test_turn_status_summarizes_collector(self) -> None:
        collector = TurnMetricsCollector(clock_ms=lambda: 0.0)
        idle = turn_status(False, collector)
        self.assertEqual(idle["completed_turns"], 0)
        self.assertIsNone(idle["last_outcome"])

        collector.start_turn("hi")
        collector.record_request({"model": "m"})
        collector.record_tool_results([{"function_name": "create_point", "result": "ok"}])
        running = turn_status(True, collector)
        self.assertTrue(running["processing"])
        self.assertEqual((running["requests"], running["tool_batches"], running["tool_calls"]), (1, 1, 1))

        collector.finish_turn("stop")
        done = turn_status(False, collector)
        self.assertEqual(done["completed_turns"], 1)
        self.assertEqual(done["last_outcome"], "stop")
        self.assertEqual(done["requests"], 0)


class TestScenarioHookCanvas(unittest.TestCase):
    """Reset and inspection against a real canvas."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.ai = _make_ai(self.canvas)

    def _run(self, *calls: Dict[str, Any]) -> None:
        self.ai.execute_tool_batch(list(calls))

    def test_reset_clears_drawables_history_view_and_modes(self) -> None:
        self._run(_call("create_point", x=1, y=2, name="A"))
        self._run(_call("zoom", center_x=0, center_y=0, range_val=2, range_axis="x"))
        self.canvas.coordinate_system_manager.set_mode("polar")
        self.canvas.set_grid_visible(False)
        self.assertGreater(len(self.canvas.undo_redo_manager.undo_stack), 0)

        reset_canvas_session(self.canvas)

        state = self.canvas.get_canvas_state()
        self.assertEqual(state.get("Points", []), [])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), 0)
        self.assertEqual(len(self.canvas.undo_redo_manager.redo_stack), 0)
        self.assertEqual(self.canvas.coordinate_system_manager.mode, "cartesian")
        self.assertTrue(self.canvas.is_grid_visible())
        self.assertTrue(self.canvas.coordinate_system_manager.polar_grid.visible)
        self.assertNotAlmostEqual(state["Cartesian_System_Visibility"]["right_bound"], 2.0)

    def test_reset_restores_the_polar_grid_spacing(self) -> None:
        polar_grid = self.canvas.coordinate_system_manager.polar_grid
        default_spacing = polar_grid._default_radial_spacing
        polar_grid._current_radial_spacing = 0.2  # as after zooming in (Canvas.reset leaves it, K25)

        self.assertEqual(build_inspection(self.canvas)["polar_radial_spacing"], 0.2)
        reset_canvas_session(self.canvas)

        self.assertEqual(polar_grid._current_radial_spacing, default_spacing)
        self.assertEqual(build_inspection(self.canvas)["polar_radial_spacing"], default_spacing)

    def test_reset_rewinds_point_name_hints(self) -> None:
        self._run(_call("create_point", x=1, y=1, name="K"))
        reset_canvas_session(self.canvas)
        self._run(_call("create_point", x=3, y=3, name="K"))
        self.assertEqual(_point_names(self.canvas.get_canvas_state()), ["K"])

    def test_inspection_reports_hidden_attributes(self) -> None:
        self._run(
            _call(
                "create_polygon",
                vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}],
                polygon_type="triangle",
                name="ABC",
                color="green",
            ),
            _call("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3),
            _call("draw_function", function_string="1/(x-1)", name="f", left_bound=-5, right_bound=5),
        )

        inspection = build_inspection(self.canvas, samples={"f": [3]})

        by_name = {(item["class"], item["name"]): item for item in inspection["drawables"]}
        self.assertEqual(by_name[("Triangle", "ABC")]["color"], "green")
        self.assertEqual(len(by_name[("Triangle", "ABC")]["vertices"]), 3)
        angle = next(item for item in inspection["drawables"] if item["class"] == "Angle")
        self.assertAlmostEqual(angle["angle_degrees"], 90.0)
        self.assertEqual(angle["vertex"], [0, 0])
        function = by_name[("Function", "f")]
        self.assertAlmostEqual(function["samples"][0][1], 0.5)
        probe = function["asymptote_probes"][0]
        self.assertAlmostEqual(probe[0], 1.0)
        self.assertGreater(abs(probe[2]), 1e4)
        self.assertEqual(inspection["undo_depth"], len(self.canvas.undo_redo_manager.undo_stack))
        self.assertEqual(inspection["coordinate_mode"], "cartesian")
        self.assertTrue(inspection["grid_visible"]["active"])

    def test_inspection_reports_segment_label_and_parametric_samples(self) -> None:
        self._run(
            _call("create_segment", x1=0, y1=0, x2=3, y2=0, label_text="s", label_visible=True, color="red"),
            _call("draw_parametric_function", x_expression="cos(t)", y_expression="sin(t)", name="p"),
        )

        inspection = build_inspection(self.canvas, t_samples={"*": [0]})

        segment = next(item for item in inspection["drawables"] if item["class"] == "Segment")
        self.assertEqual(segment["label"], {"text": "s", "visible": True})
        self.assertEqual(segment["color"], "red")
        curve = next(item for item in inspection["drawables"] if item["class"] == "ParametricFunction")
        t, x, y = curve["t_samples"][0]
        self.assertEqual(t, 0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)


class TestScenarioHookEndpoints(unittest.TestCase):
    """The hook methods themselves, as the harness calls them (JSON in, JSON out)."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.ai = _make_ai(self.canvas)
        self.hooks = ScenarioHooks(self.ai)

    def test_run_tool_calls_uses_the_model_batch_path(self) -> None:
        reply = json.loads(
            self.hooks.run_tool_calls(json.dumps([{"tool": "create_point", "args": {"x": 1, "y": 2, "name": "P"}}]))
        )

        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["traced"][0]["function_name"], "create_point")
        self.assertFalse(reply["traced"][0]["is_error"])
        self.assertEqual(reply["undoable"], [True])
        self.assertEqual(_point_names(reply["state"]), ["P"])
        self.assertEqual(reply["undo_depth_before"], 0)
        self.assertGreater(reply["undo_depth_after"], 0)
        # The batch left an action trace, like a model batch does, with a real delta (K20).
        trace = self.ai._trace_collector.get_last_trace()
        self.assertEqual(trace["trace_id"], reply["trace_id"])
        self.assertIn("P", trace["state_delta"]["added"])

    def test_run_tool_calls_reports_tool_errors(self) -> None:
        reply = json.loads(self.hooks.run_tool_calls(json.dumps([_call("no_such_tool")])))
        self.assertEqual(reply["status"], "ok")
        self.assertTrue(reply["traced"][0]["is_error"])
        self.assertEqual(reply["undoable"], [False])

    def test_run_tool_calls_rejects_invalid_json(self) -> None:
        reply = json.loads(self.hooks.run_tool_calls("not json"))
        self.assertEqual(reply["status"], "error")

    def test_run_tool_calls_refuses_during_a_chat_turn(self) -> None:
        self.ai._turn_metrics.start_turn("draw")
        reply = json.loads(self.hooks.run_tool_calls(json.dumps([_call("create_point", x=0, y=0)])))
        self.assertEqual(reply["status"], "busy")
        self.assertEqual(self.ai._turn_metrics.progress()["tool_calls"], 0)
        self.assertEqual(self.canvas.get_canvas_state().get("Points", []), [])

    def test_get_canvas_state_with_and_without_inspection(self) -> None:
        self.hooks.run_tool_calls(json.dumps([_call("create_point", x=1, y=1, name="Q", color="red")]))

        plain = json.loads(self.hooks.get_canvas_state())
        self.assertNotIn("inspection", plain)
        self.assertEqual(_point_names(plain["state"]), ["Q"])

        inspected = json.loads(self.hooks.get_canvas_state('{"inspect": true}'))
        point = next(item for item in inspected["inspection"]["drawables"] if item["class"] == "Point")
        self.assertEqual(point["color"], "red")

    def test_reset_session_restores_a_fixture_without_history(self) -> None:
        self.hooks.run_tool_calls(json.dumps([_call("create_point", x=5, y=5, name="Z")]))
        fixture = {
            "state": {
                "Points": [
                    {"name": "A", "args": {"position": {"x": 0, "y": 0}}},
                    {"name": "B", "args": {"position": {"x": 4, "y": 0}}},
                ],
                "Segments": [{"name": "AB", "args": {"p1": "A", "p2": "B"}}],
            }
        }

        reply = json.loads(self.hooks.reset_session(json.dumps({"fixture": fixture, "chat": False})))

        self.assertEqual(reply, {"status": "ok", "fixture": True})
        state = self.canvas.get_canvas_state()
        self.assertEqual(_point_names(state), ["A", "B"])
        self.assertEqual([segment["name"] for segment in state["Segments"]], ["AB"])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), 0)
        self.assertEqual(self.ai._trace_collector.get_traces(), [])

    def test_turn_status_and_stop_when_idle(self) -> None:
        status = json.loads(self.hooks.get_turn_status())
        self.assertFalse(status["processing"])
        self.assertEqual(json.loads(self.hooks.stop_turn()), {"status": "idle"})

    def test_send_message_refuses_while_busy(self) -> None:
        self.ai.is_processing = True
        self.assertEqual(json.loads(self.hooks.send_message("hello")), {"status": "busy"})

    def test_send_message_passes_the_turn_limits(self) -> None:
        sent: List[Any] = []
        ai_class = type(self.ai)
        original = ai_class.send_user_message
        ai_class.send_user_message = lambda _self, *args: sent.append(args)
        try:
            options = json.dumps({"max_requests": 3, "response_timeout_ms": 330000})
            self.assertEqual(json.loads(self.hooks.send_message("hello", None, options)), {"status": "started"})
            self.assertEqual(json.loads(self.hooks.send_message("again")), {"status": "started"})
        finally:
            ai_class.send_user_message = original
        self.assertEqual(sent, [("hello", 3, 330000), ("again", None, None)])

    def test_send_message_refuses_while_images_are_attached(self) -> None:
        sent: List[Any] = []
        self.ai._image_attachment = types.SimpleNamespace(images=["data:image/png;base64,AAAA"])
        ai_class = type(self.ai)
        original = ai_class.send_user_message
        ai_class.send_user_message = lambda _self, *args: sent.append(args)
        try:
            reply = json.loads(self.hooks.send_message("hello"))
        finally:
            ai_class.send_user_message = original
        self.assertEqual(reply["status"], "error")
        self.assertIn("images are attached", reply["error"])
        self.assertEqual(sent, [])

    def test_send_message_restores_the_model_selection(self) -> None:
        selector = document["ai-model-selector"]
        before = str(selector.value)
        added = [html.OPTION("zz-a", value="zz-test-a"), html.OPTION("zz-b", value="zz-test-b")]
        for option in added:
            selector <= option
        selector.value = "zz-test-a"
        seen: List[str] = []
        ai_class = type(self.ai)
        original = ai_class.send_user_message
        ai_class.send_user_message = lambda _self, *args: seen.append(str(document["ai-model-selector"].value))
        try:
            reply = json.loads(self.hooks.send_message("hello", "zz-test-b"))
            self.assertEqual(reply, {"status": "started"})
            self.assertEqual(seen, ["zz-test-b"])  # the request was built with the requested model
            self.assertEqual(str(selector.value), "zz-test-a")  # and the user's choice is back
        finally:
            ai_class.send_user_message = original
            for option in added:
                option.remove()
            selector.value = before

    def test_automation_guards_block_and_restore_workspace_tools(self) -> None:
        originals = {name: self.ai.available_functions[name] for name in WORKSPACE_TOOLS}

        reply = json.loads(self.hooks.set_automation_guards('{"block_workspace_tools": true}'))
        self.assertEqual(reply, {"status": "ok", "workspace_tools_blocked": True})
        self.hooks.set_automation_guards('{"block_workspace_tools": true}')  # twice keeps the originals
        batch = json.loads(self.hooks.run_tool_calls(json.dumps([_call("save_workspace", name="mine")])))
        call = batch["traced"][0]
        self.assertTrue(call["is_error"])
        self.assertIn("workspace tools are disabled", str(call["result"]))
        self.assertEqual(batch["undo_depth_after"], batch["undo_depth_before"])

        reply = json.loads(self.hooks.set_automation_guards('{"block_workspace_tools": false}'))
        self.assertEqual(reply["workspace_tools_blocked"], False)
        for name, function in originals.items():
            self.assertIs(self.ai.available_functions[name], function)


class _Recorder:
    """Accepts any method call, recording (name, args)."""

    def __init__(self, **attrs: Any) -> None:
        self.calls: List[Any] = []
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args))

        return record


class _FailingTraceCollector(ActionTraceCollector):
    def store(self, trace: Dict[str, Any]) -> None:
        raise TypeError("'<' not supported between instances of 'int' and 'str'")


class TestToolBatchTraceFailure(unittest.TestCase):
    """A failure while storing the action trace must not end the user's turn."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.ai = _make_ai(self.canvas)
        self.ai._trace_collector = _FailingTraceCollector()
        self.ai._stop_requested = False
        self.sent: List[Any] = []
        self.ai._tool_call_log = _Recorder()
        self.ai._chat_ui = _Recorder(stream_container=object(), stream_content=object(), stream_buffer="")
        self.ai._start_response_timeout = lambda use_reasoning_timeout=False: None
        self.ai._normalize_stream_event = lambda event: event
        self.ai._enable_send_controls = lambda: None

        def send(user_message: Any, tool_call_results: Any = None, **kwargs: Any) -> None:
            self.sent.append((tool_call_results, kwargs.get("action_trace")))

        self.ai._send_prompt_to_ai = send

    def test_execute_tool_batch_returns_without_a_trace(self) -> None:
        batch = self.ai.execute_tool_batch([_call("create_point", x=1, y=1, name="P")])
        self.assertIsNone(batch["trace"])
        self.assertEqual(_point_names(batch["state_after"]), ["P"])

    def test_streamed_turn_continues_with_its_tool_log(self) -> None:
        calls = [_call("create_point", x=1, y=1, name="P")]
        self.ai._turn_metrics.start_turn("draw")
        self.ai._on_stream_final({"finish_reason": "tool_calls", "ai_tool_calls": calls, "ai_message": ""})

        self.assertTrue(self.ai._turn_metrics.is_active)  # still running, not ended as an error
        self.assertIsNone(self.ai._turn_metrics.last_turn())
        self.assertEqual([name for name, _ in self.ai._tool_call_log.calls], ["ensure_element", "add_entries"])
        self.assertEqual(len(self.sent), 1)
        self.assertIsNone(self.sent[0][1])  # no trace summary to send
        self.assertEqual(_point_names(self.canvas.get_canvas_state()), ["P"])

    def test_stop_during_tools_still_reports_stopped(self) -> None:
        self.ai._stop_requested = True
        self.ai._finalize_stream_message = lambda final_message=None: None
        messages: List[str] = []
        self.ai._print_system_message_in_chat = messages.append
        self.ai._turn_metrics.start_turn("draw")
        self.ai._on_stream_final(
            {"finish_reason": "tool_calls", "ai_tool_calls": [_call("create_point", x=1, y=1)], "ai_message": ""}
        )
        self.assertFalse(self.ai._turn_metrics.is_active)
        self.assertEqual(self.ai._turn_metrics.last_turn()["outcome"], "stopped")
        self.assertEqual(messages, ["Generation stopped."])
        self.assertEqual(self.sent, [])


class TestScenarioHookFitView(unittest.TestCase):
    """fitMatHudView: a presentation-only zoom to the content, for automated runs."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.ai = _make_ai(self.canvas)
        self.hooks = ScenarioHooks(self.ai)

    def _run(self, *calls: Dict[str, Any]) -> None:
        self.ai.execute_tool_batch(list(calls))

    def _bounds(self) -> Dict[str, float]:
        return dict(self.canvas.get_canvas_state()["Cartesian_System_Visibility"])

    def test_fit_window_picks_the_binding_axis_and_a_minimum_span(self) -> None:
        center_x, center_y, half, axis = fit_window([0, 0, 10, 2], 500, 500, padding=0.1)
        self.assertEqual((center_x, center_y, axis), (5.0, 1.0, "x"))
        self.assertAlmostEqual(half, 6.0)
        _, _, half, axis = fit_window([0, 0, 2, 10], 1000, 500, padding=0.1)
        self.assertEqual(axis, "y")
        self.assertAlmostEqual(half, 6.0)
        center_x, center_y, half, _ = fit_window([3, 4, 3, 4], 500, 500, min_half_span=1.0)
        self.assertEqual((center_x, center_y, half), (3.0, 4.0, 1.0))

    def test_extent_covers_points_circles_rotated_ellipses_and_labels(self) -> None:
        self.assertIsNone(content_extent(self.canvas))
        self._run(
            _call("create_point", x=-1, y=1, name="A"),
            _call("create_circle", center_x=5, center_y=0, radius=2),
            _call("create_ellipse", center_x=0, center_y=-5, radius_x=3, radius_y=1, rotation_angle=90),
            _call("create_label", x=0, y=6, text="top"),
        )
        x_min, y_min, x_max, y_max = content_extent(self.canvas)
        self.assertAlmostEqual(x_min, -1.0)
        self.assertAlmostEqual(x_max, 7.0)
        self.assertAlmostEqual(y_min, -8.0)  # the ellipse turned upright reaches y = -5 - 3
        self.assertAlmostEqual(y_max, 6.0)

    def test_function_samples_ignore_asymptote_spikes(self) -> None:
        self._run(_call("draw_function", function_string="1/x", name="f", left_bound=-4, right_bound=4))
        x_min, y_min, x_max, y_max = content_extent(self.canvas)
        self.assertEqual((x_min, x_max), (-4.0, 4.0))
        self.assertLess(y_max, 20)
        self.assertGreater(y_min, -20)

    def test_fit_zooms_without_touching_undo_redo_or_drawables(self) -> None:
        self._run(
            _call(
                "create_polygon",
                vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}],
                polygon_type="triangle",
                name="ABC",
            )
        )
        self._run(_call("create_point", x=1, y=1, name="P"))
        self.canvas.undo()
        manager = self.canvas.undo_redo_manager
        depths = (len(manager.undo_stack), len(manager.redo_stack))
        state = self.canvas.get_canvas_state()
        drawables = {key: value for key, value in state.items() if key in ("Points", "Segments", "Triangles")}

        reply = json.loads(self.hooks.fit_view())

        self.assertEqual(reply["status"], "ok")
        self.assertTrue(reply["fitted"])
        self.assertEqual((len(manager.undo_stack), len(manager.redo_stack)), depths)
        after = self.canvas.get_canvas_state()
        for key, value in drawables.items():
            self.assertEqual(after.get(key), value)
        bounds = self._bounds()
        self.assertLess(bounds["left_bound"], 0)
        self.assertGreater(bounds["right_bound"], 6)
        self.assertLess(bounds["bottom_bound"], 0)
        self.assertGreater(bounds["top_bound"], 4)
        self.assertLess(bounds["right_bound"] - bounds["left_bound"], 10)
        # The redo entry still brings P back.
        self.canvas.redo()
        self.assertIn("P", _point_names(self.canvas.get_canvas_state()))

    def test_fit_keeps_polar_mode_and_adapts_its_grid(self) -> None:
        self._run(_call("set_coordinate_system", mode="polar"))
        self._run(_call("create_circle", center_x=0, center_y=0, radius=3))
        polar_grid = self.canvas.coordinate_system_manager.polar_grid
        default_spacing = polar_grid._default_radial_spacing

        reply = json.loads(self.hooks.fit_view())

        self.assertTrue(reply["fitted"])
        self.assertEqual(self.canvas.coordinate_system_manager.mode, "polar")
        bounds = self._bounds()
        width = bounds["right_bound"] - bounds["left_bound"]
        self.assertAlmostEqual(width, bounds["top_bound"] - bounds["bottom_bound"])
        self.assertGreater(bounds["right_bound"], 3)
        self.assertLess(bounds["right_bound"], 5)
        self.assertLessEqual(polar_grid._current_radial_spacing, default_spacing)

    def test_empty_canvas_leaves_the_view_alone(self) -> None:
        before = self._bounds()
        reply = json.loads(self.hooks.fit_view())
        self.assertEqual(reply["status"], "ok")
        self.assertFalse(reply["fitted"])
        self.assertEqual(self._bounds(), before)

    def test_single_point_gets_a_small_window_around_it(self) -> None:
        self._run(_call("create_point", x=100, y=-50, name="A"))
        json.loads(self.hooks.fit_view())
        bounds = self._bounds()
        self.assertAlmostEqual((bounds["left_bound"] + bounds["right_bound"]) / 2, 100)
        self.assertAlmostEqual((bounds["top_bound"] + bounds["bottom_bound"]) / 2, -50)
        self.assertAlmostEqual(bounds["right_bound"] - bounds["left_bound"], 2.0)

    def test_fit_refuses_during_a_chat_turn(self) -> None:
        self._run(_call("create_point", x=1, y=1, name="A"))
        before = self._bounds()
        self.ai.is_processing = True
        self.assertEqual(json.loads(self.hooks.fit_view())["status"], "busy")
        self.assertEqual(self._bounds(), before)
