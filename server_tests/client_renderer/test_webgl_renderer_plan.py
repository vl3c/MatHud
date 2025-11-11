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

from rendering import webgl_renderer

from .renderer_fixtures import PlanStub


class TestWebGLRendererPlan(unittest.TestCase):
    def _make_renderer(self) -> webgl_renderer.WebGLRenderer:
        renderer = webgl_renderer.WebGLRenderer.__new__(webgl_renderer.WebGLRenderer)
        renderer.style = {}
        renderer.canvas_el = SimpleNamespace(width=640, height=480)
        renderer._shared_primitives = SimpleNamespace()
        renderer._prepare_cartesian_dimensions = lambda _cartesian: (640, 480)
        renderer._ensure_scratch_context = lambda _color: SimpleNamespace(fillStyle="")
        return renderer

    def test_render_cartesian_skips_invisible_plan(self) -> None:
        renderer = self._make_renderer()
        invisible_plan = PlanStub(visible=False, plan_key="cart-plan-1")
        visible_plan = PlanStub(visible=True, plan_key="cart-plan-2")
        plans = deque([invisible_plan, visible_plan])

        renderer._build_cartesian_plan = lambda _c, _m: plans.popleft()
        renderer._should_apply_plan = webgl_renderer.WebGLRenderer._should_apply_plan.__get__(renderer)

        cartesian = SimpleNamespace()
        mapper = object()

        renderer.render_cartesian(cartesian, mapper)
        renderer.render_cartesian(cartesian, mapper)

        self.assertEqual(invisible_plan.apply_calls, 0)
        self.assertEqual(visible_plan.apply_calls, 1)

    def test_parse_color_normalizes_rgba_alpha(self) -> None:
        renderer = self._make_renderer()
        renderer._ensure_scratch_context = lambda _color: SimpleNamespace(fillStyle="")

        result = webgl_renderer.WebGLRenderer._parse_color(renderer, "rgba(255, 128, 64, 200)")

        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 128 / 255, places=6)
        self.assertAlmostEqual(result[2], 64 / 255, places=6)
        self.assertAlmostEqual(result[3], 200 / 255, places=6)


__all__ = ["TestWebGLRendererPlan"]
