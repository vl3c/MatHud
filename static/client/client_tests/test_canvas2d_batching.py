"""Tests for Canvas2D batching, draw order and render-plan reprojection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple

from browser import document

from cartesian_system_2axis import Cartesian2Axis
from coordinate_mapper import CoordinateMapper

from rendering.base_telemetry import BaseRendererTelemetry
from rendering.cached_render_plan import (
    OptimizedPrimitivePlan,
    PrimitiveCommand,
    _affine_params,
    _math_to_screen_point,
    build_plan_for_cartesian,
)
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.primitives import FillStyle, FontStyle, StrokeStyle, TextAlignment
from rendering.style_manager import get_renderer_style


class _RecordingContext:
    """Minimal Python stand-in for a CanvasRenderingContext2D that logs calls."""

    def __init__(self) -> None:
        self.log: List[Tuple[Any, ...]] = []
        self.strokeStyle = "#000"
        self.fillStyle = "#000"
        self.lineWidth = 1
        self.globalAlpha = 1.0

    def beginPath(self) -> None:
        self.log.append(("beginPath",))

    def moveTo(self, x: float, y: float) -> None:
        self.log.append(("moveTo", x, y))

    def lineTo(self, x: float, y: float) -> None:
        self.log.append(("lineTo", x, y))

    def closePath(self) -> None:
        self.log.append(("closePath",))

    def stroke(self) -> None:
        self.log.append(("stroke", self.strokeStyle))

    def fill(self) -> None:
        self.log.append(("fill", self.fillStyle))

    def fillText(self, text: str, x: float, y: float) -> None:
        self.log.append(("fillText", text))

    def save(self) -> None:
        pass

    def restore(self) -> None:
        pass


def _make_adapter(telemetry: Any = None) -> Tuple[Canvas2DPrimitiveAdapter, _RecordingContext]:
    ctx = _RecordingContext()
    canvas_el = SimpleNamespace(getContext=lambda _kind: ctx, width=200, height=100)
    return Canvas2DPrimitiveAdapter(canvas_el, telemetry=telemetry), ctx


def _ops(ctx: _RecordingContext, *names: str) -> List[Tuple[Any, ...]]:
    return [entry for entry in ctx.log if entry[0] in names]


class TestCanvas2DBatching(unittest.TestCase):
    def setUp(self) -> None:
        self.stroke = StrokeStyle(color="#111", width=2, line_join="round")
        self.fill = FillStyle(color="#abc", opacity=None)

    def test_python_context_uses_python_fallback(self) -> None:
        adapter, _ = _make_adapter()
        self.assertIsNone(adapter._js_stroke_paths)
        self.assertIsNone(adapter._js_trace_polygons)

    def test_polyline_is_one_subpath(self) -> None:
        adapter, ctx = _make_adapter()
        points = ((0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, 5.0))
        adapter.execute_optimized(PrimitiveCommand("stroke_polyline", (points, self.stroke), {}))
        adapter.end_frame()
        self.assertEqual(len(_ops(ctx, "moveTo")), 1)
        self.assertEqual(len(_ops(ctx, "lineTo")), 3)
        self.assertEqual(len(_ops(ctx, "stroke")), 1)

    def test_lines_and_polylines_with_same_style_share_one_stroke(self) -> None:
        adapter, ctx = _make_adapter()
        adapter.execute_optimized(
            PrimitiveCommand("stroke_line", ((0.0, 0.0), (5.0, 5.0), self.stroke), {"include_width": True})
        )
        adapter.execute_optimized(
            PrimitiveCommand("stroke_polyline", (((1.0, 1.0), (2.0, 2.0), (3.0, 1.0)), self.stroke), {})
        )
        adapter.end_frame()
        self.assertEqual(len(_ops(ctx, "beginPath")), 1)
        self.assertEqual(len(_ops(ctx, "moveTo")), 2)
        self.assertEqual(len(_ops(ctx, "stroke")), 1)

    def test_line_between_polygons_keeps_draw_order(self) -> None:
        adapter, ctx = _make_adapter()
        square_a = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        square_b = ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0))
        adapter.execute_optimized(PrimitiveCommand("fill_polygon", (square_a, self.fill, None), {}))
        adapter.execute_optimized(
            PrimitiveCommand("stroke_line", ((0.0, 5.0), (30.0, 5.0), self.stroke), {"include_width": True})
        )
        adapter.execute_optimized(PrimitiveCommand("fill_polygon", (square_b, self.fill, None), {}))
        adapter.end_frame()
        paint_ops = [entry[0] for entry in _ops(ctx, "fill", "stroke")]
        self.assertEqual(paint_ops, ["fill", "stroke", "fill"])

    def test_direct_text_is_drawn_after_pending_lines(self) -> None:
        adapter, ctx = _make_adapter()
        adapter.execute_optimized(
            PrimitiveCommand("stroke_line", ((0.0, 0.0), (5.0, 5.0), self.stroke), {"include_width": True})
        )
        adapter.draw_text("label", (1.0, 1.0), FontStyle("Arial", 12, None), "#000", TextAlignment("left", "top"))
        paint_ops = [entry[0] for entry in _ops(ctx, "stroke", "fillText")]
        self.assertEqual(paint_ops, ["stroke", "fillText"])

    def test_line_telemetry_is_aggregated_per_flush(self) -> None:
        telemetry = BaseRendererTelemetry()
        adapter, _ = _make_adapter(telemetry)
        points = tuple((float(i), float(i % 3)) for i in range(11))
        adapter.execute_optimized(PrimitiveCommand("stroke_polyline", (points, self.stroke), {}))
        adapter.execute_optimized(
            PrimitiveCommand("stroke_line", ((0.0, 0.0), (5.0, 5.0), self.stroke), {"include_width": True})
        )
        adapter.end_frame()
        events = telemetry.snapshot()["adapter_events"]
        self.assertEqual(events["line_batch_segments"], 11)
        self.assertEqual(events["stroke_calls"], 1)

    def test_js_helpers_draw_on_real_canvas(self) -> None:
        canvas_el = document.createElement("canvas")
        canvas_el.width = 40
        canvas_el.height = 40
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        self.assertIsNotNone(adapter._js_stroke_paths, "canvas2d_paths.js helpers were not loaded")
        stroke = StrokeStyle(color="#ff0000", width=4)
        adapter.execute_optimized(PrimitiveCommand("stroke_polyline", (((0.0, 10.0), (40.0, 10.0)), stroke), {}))
        square = ((5.0, 25.0), (35.0, 25.0), (35.0, 35.0), (5.0, 35.0))
        adapter.execute_optimized(PrimitiveCommand("fill_polygon", (square, FillStyle(color="#0000ff"), None), {}))
        adapter.end_frame()
        ctx = canvas_el.getContext("2d")
        line_pixel = ctx.getImageData(20, 10, 1, 1).data
        fill_pixel = ctx.getImageData(20, 30, 1, 1).data
        self.assertGreater(line_pixel[0], 200)
        self.assertGreater(line_pixel[3], 200)
        self.assertGreater(fill_pixel[2], 200)
        self.assertGreater(fill_pixel[3], 200)


class TestRenderPlanReprojection(unittest.TestCase):
    OLD = {"scale": 40.0, "origin_x": 300.0, "origin_y": 200.0, "offset_x": 12.0, "offset_y": -7.0}
    NEW = {"scale": 55.0, "origin_x": 300.0, "origin_y": 200.0, "offset_x": -31.5, "offset_y": 18.25}

    def _screen_to_math(self, point: Tuple[float, float], state: dict) -> Tuple[float, float]:
        scale = state["scale"]
        return (
            (point[0] - state["offset_x"] - state["origin_x"]) / scale,
            (state["origin_y"] + state["offset_y"] - point[1]) / scale,
        )

    def test_affine_map_matches_math_round_trip(self) -> None:
        k, tx, ty = _affine_params(self.OLD, self.NEW)
        for point in ((0.0, 0.0), (123.5, -40.25), (-900.0, 1200.0)):
            expected = _math_to_screen_point(self._screen_to_math(point, self.OLD), self.NEW)
            self.assertAlmostEqual(k * point[0] + tx, expected[0], places=9)
            self.assertAlmostEqual(k * point[1] + ty, expected[1], places=9)

    def test_plan_reprojects_points_and_radii(self) -> None:
        stroke = StrokeStyle(color="#000", width=1)
        commands = [
            PrimitiveCommand("stroke_polyline", (((10.0, 20.0), (30.0, 40.0)), stroke), {}),
            PrimitiveCommand("stroke_circle", ((50.0, 60.0), 8.0, stroke), {}),
        ]
        plan = OptimizedPrimitivePlan(
            drawable=None, commands=commands, plan_key="p", metadata={"map_state": dict(self.OLD)}
        )
        plan.update_map_state(dict(self.NEW))
        k = self.NEW["scale"] / self.OLD["scale"]
        expected_start = _math_to_screen_point(self._screen_to_math((10.0, 20.0), self.OLD), self.NEW)
        new_start = plan.commands[0].args[0][0]
        self.assertAlmostEqual(new_start[0], expected_start[0], places=9)
        self.assertAlmostEqual(new_start[1], expected_start[1], places=9)
        self.assertAlmostEqual(plan.commands[1].args[1], 8.0 * k, places=9)

    def test_pan_shifts_bounds_like_a_full_rescan(self) -> None:
        stroke = StrokeStyle(color="#000", width=1)
        points = tuple((float(i), float((i * 7) % 13)) for i in range(20))
        commands = [PrimitiveCommand("stroke_polyline", (points, stroke), {})]
        plan = OptimizedPrimitivePlan(
            drawable=None, commands=commands, plan_key="p", metadata={"map_state": dict(self.OLD)}
        )
        panned = dict(self.OLD)
        panned["offset_x"] += 37.5
        panned["offset_y"] -= 12.0
        plan.update_map_state(panned)
        shifted = plan._screen_bounds
        plan._recompute_bounds_from_commands()
        for got, expected in zip(shifted, plan._screen_bounds):
            self.assertAlmostEqual(got, expected, places=9)

    def test_plan_bounds_include_circle_radius(self) -> None:
        fill = FillStyle(color="#000")
        commands = [PrimitiveCommand("fill_circle", ((-5.0, 50.0), 20.0, fill, None), {"screen_space": False})]
        plan = OptimizedPrimitivePlan(
            drawable=None, commands=commands, plan_key="p", metadata={"map_state": dict(self.OLD)}
        )
        # Centre is off-screen but the circle reaches into the viewport.
        self.assertTrue(plan.is_visible(100, 100))

    def test_supports_transform_request_is_ignored(self) -> None:
        mapper = CoordinateMapper(640, 480)
        plan = build_plan_for_cartesian(Cartesian2Axis(mapper), mapper, get_renderer_style(), supports_transform=True)
        self.assertFalse(plan.supports_transform())
        self.assertIsNone(plan.get_transform())


__all__ = ["TestCanvas2DBatching", "TestRenderPlanReprojection"]
