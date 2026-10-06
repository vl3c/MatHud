"""Tests for prompt_canvas_state: the canvas size and the new curves' extents sent after a tool batch."""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from prompt_canvas_state import (
    CANVAS_SIZE_KEY,
    CURVE_EXTENTS_KEY,
    CURVE_SAMPLES,
    MAX_MEASURED_CURVES,
    _PEAKS_CHECKED,
    canvas_size_px,
    curve_extents,
    new_curve_names,
    with_view_info,
)

from .test_turn_metrics import TestTurnBookkeeping


def _frozen() -> float:
    """A clock that never moves: the time limit never cuts a test short."""
    return 0.0


class _Sized:
    def __init__(self, width: Any, height: Any) -> None:
        self.width = width
        self.height = height


class _Mapper:
    def __init__(self, left: float, right: float) -> None:
        self.left, self.right = left, right

    def get_visible_left_bound(self) -> float:
        return self.left

    def get_visible_right_bound(self) -> float:
        return self.right


class _Graph:
    def __init__(self, name: str, f: Any, left: Optional[float] = None, right: Optional[float] = None) -> None:
        self.name = name
        self.function = f
        self.left_bound = left
        self.right_bound = right


class _Curve:
    def __init__(self, name: str, x: Any, y: Any, t_min: float = 0.0, t_max: float = 2 * math.pi) -> None:
        self.name = name
        self.t_min, self.t_max = t_min, t_max
        self.evaluate = lambda t: (x(t), y(t))


class _Canvas(_Sized):
    """width/height, a coordinate mapper and drawables by class name, like Canvas."""

    def __init__(self, drawables: Optional[Dict[str, List[Any]]] = None, view: tuple = (-400.0, 400.0)) -> None:
        super().__init__(800, 600)
        self.coordinate_mapper = _Mapper(*view)
        self._drawables = drawables or {}

    def get_drawables_by_class_name(self, class_name: str) -> List[Any]:
        return self._drawables.get(class_name, [])


def _state(**functions: str) -> Dict[str, Any]:
    return {"Functions": [{"name": name, "args": {"function_string": expr}} for name, expr in functions.items()]}


def _all(canvas: _Canvas) -> Dict[str, List[str]]:
    """Every curve on the canvas, as if all were new."""
    buckets = {
        "Function": "Functions",
        "PiecewiseFunction": "PiecewiseFunctions",
        "ParametricFunction": "ParametricFunctions",
    }
    return {buckets[c]: [d.name for d in drawables] for c, drawables in canvas._drawables.items()}


class TestPromptCanvasSize(unittest.TestCase):
    def test_size_of_a_canvas(self) -> None:
        self.assertEqual(canvas_size_px(_Sized(800, 600)), {"width": 800.0, "height": 600.0})
        self.assertEqual(canvas_size_px(_Sized(812.3456, 600.004)), {"width": 812.35, "height": 600.0})

    def test_unknown_or_invalid_sizes_give_none(self) -> None:
        for sized in (None, _Sized(0, 600), _Sized(800, -1), _Sized("wide", 600), _Sized(float("nan"), 600)):
            self.assertIsNone(canvas_size_px(sized))

    def test_info_is_added_to_a_copy(self) -> None:
        state = _state(f="sin(x)")
        result = with_view_info(state, _Canvas({"Function": [_Graph("f", math.sin)]}), {}, _frozen)
        self.assertEqual(result[CANVAS_SIZE_KEY], {"width": 800.0, "height": 600.0})
        self.assertIn("f", result[CURVE_EXTENTS_KEY]["Functions"])
        self.assertNotIn(CANVAS_SIZE_KEY, state)

    def test_state_is_returned_unchanged_without_a_size(self) -> None:
        state: Dict[str, Any] = {"Points": []}
        self.assertIs(with_view_info(state, None), state)
        self.assertIs(with_view_info(None, _Sized(800, 600)), None)

    def test_live_canvas_exposes_its_size(self) -> None:
        from canvas import Canvas

        self.assertEqual(canvas_size_px(Canvas(640, 480, draw_enabled=False)), {"width": 640.0, "height": 480.0})


class TestNewCurves(unittest.TestCase):
    """Only the curves a batch created or redefined are measured, decided from the states."""

    def test_new_and_redefined_curves(self) -> None:
        before = _state(f="sin(x)", g="x^2")
        after = _state(f="sin(x)", g="x^3", h="cos(x)")
        self.assertEqual(new_curve_names(after, before), {"Functions": ["g", "h"]})

    def test_a_recolour_is_no_change(self) -> None:
        before = _state(f="sin(x)")
        after = _state(f="sin(x)")
        after["Functions"][0]["args"]["color"] = "red"
        self.assertEqual(new_curve_names(after, before), {})

    def test_unchanged_curves_are_not_measured(self) -> None:
        canvas = _Canvas({"Function": [_Graph("f", math.sin), _Graph("g", math.cos)]})
        result = with_view_info(_state(f="sin(x)", g="cos(x)"), canvas, _state(f="sin(x)"), _frozen)
        self.assertEqual(list(result[CURVE_EXTENTS_KEY]["Functions"]), ["g"])


class TestCurveExtents(unittest.TestCase):
    def assertBox(self, entry: Dict[str, Any], expected: List[float], places: int = 2) -> None:  # noqa: N802
        for actual, wanted in zip(entry["box"], expected):
            self.assertAlmostEqual(actual, wanted, places=places)

    def _measure(self, canvas: _Canvas) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return curve_extents(canvas, _all(canvas), _frozen)

    def test_unbounded_graph_is_sampled_over_the_view(self) -> None:
        entry = self._measure(_Canvas({"Function": [_Graph("f", math.sin)]}, view=(-628.0, 628.0)))["Functions"]["f"]
        self.assertTrue(entry["clipped"])
        self.assertBox(entry, [-628, 628, -1, 1], places=1)

    def test_bounded_graph_is_sampled_over_its_bounds(self) -> None:
        entry = self._measure(_Canvas({"Function": [_Graph("g", lambda x: x * x - 2, -2.0, 2.0)]}))["Functions"]["g"]
        self.assertFalse(entry["clipped"])
        self.assertBox(entry, [-2, 2, -2, 2], places=1)

    def test_undefined_values_are_skipped(self) -> None:
        def sqrt_or_fail(x: float) -> float:
            if x < 0:
                raise ValueError("math domain error")
            return math.sqrt(x)

        entry = self._measure(_Canvas({"Function": [_Graph("s", sqrt_or_fail, -4.0, 4.0)]}))["Functions"]["s"]
        self.assertBox(entry, [0, 4, 0, 2], places=0)

    def test_parametric_curve(self) -> None:
        extents = self._measure(_Canvas({"ParametricFunction": [_Curve("c", math.cos, math.sin)]}))
        self.assertBox(extents["ParametricFunctions"]["c"], [-1, 1, -1, 1])

    def test_waves_tells_a_wave_from_a_line_or_a_bump(self) -> None:
        graphs = [
            _Graph("s", math.sin, -10.0, 10.0),
            _Graph("l", lambda x: 0.01 * x + 5, 0.0, 500.0),
            _Graph("b", lambda x: 0.01 * math.exp(-x * x), -5.0, 5.0),
        ]
        extents = self._measure(_Canvas({"Function": graphs}))
        self.assertTrue(extents["Functions"]["s"]["waves"])
        self.assertFalse(extents["Functions"]["l"]["waves"])
        self.assertFalse(extents["Functions"]["b"]["waves"])

    def test_spiky_marks_a_pole_between_samples(self) -> None:
        graphs = [
            _Graph("r", lambda x: 1.0 / x),
            _Graph("q", lambda x: 1.0 / (x * x)),
            _Graph("p", lambda x: 1.0 / (x - 0.3)),
            _Graph("s", math.sin),
        ]
        extents = self._measure(_Canvas({"Function": graphs}, view=(-628.0, 628.0)))
        for name in ("r", "q", "p"):
            self.assertTrue(extents["Functions"][name]["spiky"], name)
        self.assertFalse(extents["Functions"]["s"]["spiky"])

    def test_the_count_cap_is_deterministic(self) -> None:
        calls: List[float] = []

        def counted(x: float) -> float:
            calls.append(x)
            return x

        graphs = [_Graph(f"f{i}", counted, -1.0, 1.0) for i in range(MAX_MEASURED_CURVES + 5)]
        extents = self._measure(_Canvas({"Function": graphs}))
        self.assertEqual(list(extents["Functions"]), [f"f{i}" for i in range(MAX_MEASURED_CURVES)])
        self.assertLessEqual(len(calls), MAX_MEASURED_CURVES * (CURVE_SAMPLES + 2 * _PEAKS_CHECKED))

    def test_running_past_the_time_limit_sends_nothing(self) -> None:
        ticks = iter(i * 0.06 for i in range(1000))
        graphs = [_Graph(f"f{i}", math.sin, -1.0, 1.0) for i in range(5)]
        canvas = _Canvas({"Function": graphs})
        self.assertEqual(curve_extents(canvas, _all(canvas), clock=lambda: next(ticks)), {})

    def test_real_function_drawable(self) -> None:
        from drawables.function import Function

        canvas = _Canvas({"Function": [Function("x^2 - 2", name="f", left_bound=-2, right_bound=2)]})
        self.assertBox(self._measure(canvas)["Functions"]["f"], [-2, 2, -2, 2], places=1)


class TestPromptCarriesViewInfo(unittest.TestCase):
    """AIInterface sends the view info after tool batches only, without touching the state it was given."""

    def _ai(self) -> Any:
        ai = TestTurnBookkeeping()._ai()
        ai._turn_request_limit = None
        ai._turn_requests_sent = 0
        self.sent: List[Any] = []
        ai._send_prompt_json = lambda *args: self.sent.append(args)
        return ai

    def test_tool_batch_prompt_carries_the_info(self) -> None:
        ai = self._ai()
        ai.canvas = _Canvas({"Function": [_Graph("f", math.sin)]})
        state_after = _state(f="sin(x)")
        ai._send_prompt_to_ai(None, "[]", canvas_state=state_after, previous_state={})
        sent_state = self.sent[-1][0]["canvas_state"]
        self.assertEqual(sent_state[CANVAS_SIZE_KEY], {"width": 800.0, "height": 600.0})
        self.assertIn("f", sent_state[CURVE_EXTENTS_KEY]["Functions"])
        self.assertNotIn(CANVAS_SIZE_KEY, state_after)

    def test_user_messages_carry_none(self) -> None:
        ai = self._ai()
        ai.canvas = _Canvas({"Function": [_Graph("f", math.sin)]})
        ai._send_prompt_to_ai("hi", None, canvas_state=_state(f="sin(x)"))
        self.assertNotIn(CANVAS_SIZE_KEY, self.sent[-1][0]["canvas_state"])

    def test_prompt_is_sent_when_measuring_fails(self) -> None:
        import ai_interface

        def broken(*args: Any) -> Any:
            raise RuntimeError("measuring failed")

        original = ai_interface.with_view_info
        ai_interface.with_view_info = broken
        try:
            ai = self._ai()
            ai.canvas = _Canvas()
            ai._send_prompt_to_ai(None, "[]", canvas_state={"Points": []})
        finally:
            ai_interface.with_view_info = original
        self.assertEqual(self.sent[-1][0]["canvas_state"], {"Points": []})


__all__ = ["TestPromptCanvasSize", "TestNewCurves", "TestCurveExtents", "TestPromptCarriesViewInfo"]
