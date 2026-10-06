"""Tests for prompt_canvas_state: the canvas size and curve extents added to a prompt's canvas state."""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from prompt_canvas_state import (
    CANVAS_SIZE_KEY,
    CURVE_EXTENTS_KEY,
    CURVE_SAMPLES,
    MAX_MEASURED_CURVES,
    canvas_size_px,
    curve_extents,
    with_view_info,
)

from .test_turn_metrics import TestTurnBookkeeping


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
        self.calls = 0

    def get_drawables_by_class_name(self, class_name: str) -> List[Any]:
        return self._drawables.get(class_name, [])


class TestPromptCanvasSize(unittest.TestCase):
    def test_size_of_a_canvas(self) -> None:
        self.assertEqual(canvas_size_px(_Sized(800, 600)), {"width": 800.0, "height": 600.0})

    def test_fractional_css_pixels_are_rounded(self) -> None:
        self.assertEqual(canvas_size_px(_Sized(812.3456, 600.004)), {"width": 812.35, "height": 600.0})

    def test_unknown_or_invalid_sizes_give_none(self) -> None:
        self.assertIsNone(canvas_size_px(None))
        self.assertIsNone(canvas_size_px(_Sized(0, 600)))
        self.assertIsNone(canvas_size_px(_Sized(800, -1)))
        self.assertIsNone(canvas_size_px(_Sized("wide", 600)))
        self.assertIsNone(canvas_size_px(_Sized(float("nan"), 600)))

    def test_info_is_added_to_a_copy(self) -> None:
        state: Dict[str, Any] = {"Points": [], "Cartesian_System_Visibility": {"left_bound": -400}}
        result = with_view_info(state, _Canvas({"Function": [_Graph("f", math.sin)]}))
        self.assertEqual(result[CANVAS_SIZE_KEY], {"width": 800.0, "height": 600.0})
        self.assertIn("f", result[CURVE_EXTENTS_KEY]["Functions"])
        self.assertEqual(result["Points"], [])
        self.assertNotIn(CANVAS_SIZE_KEY, state)
        self.assertNotIn(CURVE_EXTENTS_KEY, state)

    def test_no_curves_adds_no_extents(self) -> None:
        self.assertNotIn(CURVE_EXTENTS_KEY, with_view_info({}, _Canvas()))

    def test_state_is_returned_unchanged_without_a_size(self) -> None:
        state: Dict[str, Any] = {"Points": []}
        self.assertIs(with_view_info(state, None), state)
        self.assertIs(with_view_info(None, _Sized(800, 600)), None)

    def test_live_canvas_exposes_its_size(self) -> None:
        """The real Canvas exposes its CSS pixel size as width and height."""
        from canvas import Canvas

        canvas = Canvas(640, 480, draw_enabled=False)
        self.assertEqual(canvas_size_px(canvas), {"width": 640.0, "height": 480.0})


class TestCurveExtents(unittest.TestCase):
    def assertBox(self, entry: Dict[str, Any], expected: List[float], places: int = 2) -> None:  # noqa: N802
        for actual, wanted in zip(entry["box"], expected):
            self.assertAlmostEqual(actual, wanted, places=places)

    def test_unbounded_graph_is_sampled_over_the_view(self) -> None:
        extents = curve_extents(_Canvas({"Function": [_Graph("f", math.sin)]}, view=(-628.0, 628.0)))
        entry = extents["Functions"]["f"]
        self.assertTrue(entry["clipped"])
        self.assertBox(entry, [-628, 628, -1, 1], places=1)

    def test_bounded_graph_is_sampled_over_its_bounds(self) -> None:
        graph = _Graph("g", lambda x: x * x - 2, left=-2.0, right=2.0)
        entry = curve_extents(_Canvas({"Function": [graph]}))["Functions"]["g"]
        self.assertFalse(entry["clipped"])
        self.assertBox(entry, [-2, 2, -2, 2], places=1)

    def test_half_bounded_graph_is_clipped_to_the_view(self) -> None:
        graph = _Graph("h", lambda x: x, left=0.0)
        entry = curve_extents(_Canvas({"Function": [graph]}, view=(-10.0, 10.0)))["Functions"]["h"]
        self.assertTrue(entry["clipped"])
        self.assertBox(entry, [0, 10, 0, 10], places=0)

    def test_steep_ends_near_an_asymptote_are_trimmed(self) -> None:
        graph = _Graph("t", lambda x: 1.0 / x if x else float("inf"), left=-1.0, right=1.0)
        entry = curve_extents(_Canvas({"Function": [graph]}))["Functions"]["t"]
        nearest = min(abs(-1.0 + 2.0 * i / (CURVE_SAMPLES - 1)) for i in range(CURVE_SAMPLES))
        self.assertLess(entry["box"][3], 1.0 / nearest)

    def test_undefined_values_are_skipped(self) -> None:
        def sqrt_or_fail(x: float) -> float:
            if x < 0:
                raise ValueError("math domain error")
            return math.sqrt(x)

        entry = curve_extents(_Canvas({"Function": [_Graph("s", sqrt_or_fail, -4.0, 4.0)]}))["Functions"]["s"]
        self.assertBox(entry, [0, 4, 0, 2], places=0)

    def test_parametric_curve(self) -> None:
        extents = curve_extents(_Canvas({"ParametricFunction": [_Curve("c", math.cos, math.sin)]}))
        self.assertBox(extents["ParametricFunctions"]["c"], [-1, 1, -1, 1], places=2)

    def test_piecewise_bucket(self) -> None:
        extents = curve_extents(_Canvas({"PiecewiseFunction": [_Graph("p", abs, -1.0, 1.0)]}))
        self.assertIn("p", extents["PiecewiseFunctions"])

    def test_cost_is_bounded(self) -> None:
        calls: List[float] = []

        def counted(x: float) -> float:
            calls.append(x)
            return x

        graphs = [_Graph(f"f{i}", counted, -1.0, 1.0) for i in range(MAX_MEASURED_CURVES + 5)]
        extents = curve_extents(_Canvas({"Function": graphs}))
        self.assertEqual(len(extents["Functions"]), MAX_MEASURED_CURVES)
        self.assertEqual(len(calls), MAX_MEASURED_CURVES * CURVE_SAMPLES)

    def test_real_function_drawable(self) -> None:
        from drawables.function import Function

        entry = curve_extents(_Canvas({"Function": [Function("x^2 - 2", name="f", left_bound=-2, right_bound=2)]}))
        self.assertBox(entry["Functions"]["f"], [-2, 2, -2, 2], places=1)


class TestPromptCarriesViewInfo(unittest.TestCase):
    """AIInterface sends the view info with every prompt, without touching the state it was given."""

    def _ai(self) -> Any:
        ai = TestTurnBookkeeping()._ai()
        ai._turn_request_limit = None
        ai._turn_requests_sent = 0
        self.sent: List[Any] = []
        ai._send_prompt_json = lambda *args: self.sent.append(args)
        return ai

    def test_tool_batch_prompt_carries_the_size(self) -> None:
        ai = self._ai()
        ai.canvas = _Canvas({"Function": [_Graph("f", math.sin)]})
        state_after: Dict[str, Any] = {"Points": []}
        ai._send_prompt_to_ai(None, "[]", canvas_state=state_after)
        prompt_json = self.sent[-1][0]
        self.assertEqual(prompt_json["canvas_state"][CANVAS_SIZE_KEY], {"width": 800.0, "height": 600.0})
        self.assertIn("f", prompt_json["canvas_state"][CURVE_EXTENTS_KEY]["Functions"])
        self.assertNotIn(CANVAS_SIZE_KEY, state_after)

    def test_prompt_without_a_canvas_is_sent_as_is(self) -> None:
        ai = self._ai()
        ai._send_prompt_to_ai(None, "[]", canvas_state={})
        self.assertEqual(self.sent[-1][0]["canvas_state"], {})


__all__ = ["TestPromptCanvasSize", "TestCurveExtents", "TestPromptCarriesViewInfo"]
