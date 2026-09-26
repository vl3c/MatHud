"""Tests for scenario_hooks: the window hooks used by the scenario-testing harness."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from canvas import Canvas
from function_registry import FunctionRegistry
from managers.action_trace_collector import ActionTraceCollector
from scenario_hooks import (
    ScenarioHooks,
    build_inspection,
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

    def test_run_tool_calls_records_into_an_active_turn(self) -> None:
        self.ai._turn_metrics.start_turn("draw")
        self.hooks.run_tool_calls(json.dumps([_call("create_point", x=0, y=0)]))
        self.assertEqual(self.ai._turn_metrics.progress()["tool_calls"], 1)

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
