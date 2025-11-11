from __future__ import annotations

import sys
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "static" / "client"
CLIENT_ROOT_STR = str(CLIENT_ROOT)
if CLIENT_ROOT.exists() and CLIENT_ROOT_STR not in sys.path:
    sys.path.insert(0, CLIENT_ROOT_STR)

import unittest
from collections import deque
from types import SimpleNamespace

from rendering import canvas2d_renderer

from .renderer_fixtures import CanvasContextRecorder, PlanStub, TelemetryRecorder


class TestCanvas2DRendererPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.original_build_plan = canvas2d_renderer.build_plan_for_cartesian

    def tearDown(self) -> None:
        canvas2d_renderer.build_plan_for_cartesian = self.original_build_plan

    def _make_renderer(self) -> canvas2d_renderer.Canvas2DRenderer:
        renderer = canvas2d_renderer.Canvas2DRenderer.__new__(canvas2d_renderer.Canvas2DRenderer)
        renderer.style = {}
        renderer._telemetry = TelemetryRecorder()
        renderer._cartesian_cache = None
        renderer._plan_cache = {}
        renderer.canvas_el = SimpleNamespace(width=640, height=480)
        renderer.ctx = CanvasContextRecorder()
        renderer._use_layer_compositing = False
        renderer._offscreen_canvas = None
        renderer._is_cached_plan_valid = lambda entry, signature: bool(entry and entry.get("signature") == signature)
        return renderer

    def test_cartesian_plan_cache_rebuilds_on_signature_change(self) -> None:
        renderer = self._make_renderer()

        plans = deque([PlanStub(plan_key="plan-1"), PlanStub(plan_key="plan-2")])

        def fake_builder(cartesian, mapper, style, supports_transform=False):
            return plans.popleft()

        canvas2d_renderer.build_plan_for_cartesian = fake_builder

        cartesian = SimpleNamespace()
        mapper = object()
        map_state = {"scale": 1.0}

        signature_a = ("sig",)
        plan_a_first = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, signature_a, "Cartesian2Axis")
        plan_a_second = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, signature_a, "Cartesian2Axis")

        self.assertIs(plan_a_first, plan_a_second)
        self.assertEqual(plan_a_first.update_calls, 2)

        signature_b = ("sig", 2)
        plan_b = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, signature_b, "Cartesian2Axis")

        self.assertIsNot(plan_b, plan_a_first)
        self.assertEqual(plan_b.update_calls, 1)

    def test_flush_offscreen_draws_back_to_main_canvas(self) -> None:
        renderer = self._make_renderer()
        renderer._use_layer_compositing = True
        renderer._offscreen_canvas = object()

        renderer._flush_offscreen_to_main()

        self.assertEqual(len(renderer.ctx.draw_image_calls), 1)
        self.assertIs(renderer.ctx.draw_image_calls[0][0], renderer._offscreen_canvas)


__all__ = ["TestCanvas2DRendererPlan"]
