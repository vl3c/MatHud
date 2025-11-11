from __future__ import annotations

import sys
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "static" / "client"
CLIENT_ROOT_STR = str(CLIENT_ROOT)
if CLIENT_ROOT.exists() and CLIENT_ROOT_STR not in sys.path:
    sys.path.insert(0, CLIENT_ROOT_STR)

import unittest
from types import SimpleNamespace

from rendering import svg_renderer

from .renderer_fixtures import PlanStub, TelemetryRecorder


class TestSvgRendererPlan(unittest.TestCase):
    def _make_renderer(self) -> svg_renderer.SvgRenderer:
        renderer = svg_renderer.SvgRenderer.__new__(svg_renderer.SvgRenderer)
        renderer.style = {}
        renderer._telemetry = TelemetryRecorder()
        renderer._shared_primitives = SimpleNamespace(
            reserve_usage_counts=lambda *_args, **_kwargs: None,
            drop_group=lambda *_args, **_kwargs: None,
            push_group_to_back=lambda *_args, **_kwargs: None,
        )
        renderer._plan_cache = {}
        renderer._cartesian_cache = None
        renderer._frame_seen_plan_keys = set()
        renderer._cartesian_rendered_this_frame = False
        renderer._get_surface_dimensions = lambda: (640, 480)
        renderer._mark_screen_space_plan_dirty = lambda plan: plan.mark_dirty()
        renderer._drop_plan_group = lambda plan: None
        renderer._create_cartesian_plan_context = lambda plan: {
            "plan": plan,
            "plan_key": plan.plan_key,
            "usage_counts": plan.get_usage_counts(),
            "reserve": None,
            "supports_transform": plan.supports_transform(),
            "needs_apply": plan.needs_apply(),
        }
        return renderer

    def test_cartesian_plan_cache_reuse(self) -> None:
        renderer = self._make_renderer()
        renderer._is_cached_plan_valid = lambda entry, signature: bool(entry and entry.get("signature") == signature)

        first_plan = PlanStub(visible=True, plan_key="cart-plan")
        second_plan = PlanStub(visible=True, plan_key="cart-plan-2")
        plans = [first_plan, second_plan]

        def build_plan(cartesian, mapper, map_state, drawable_name):
            plan = plans.pop(0)
            plan.update_map_state(map_state)
            return plan

        renderer._build_cartesian_plan_with_metrics = build_plan

        cartesian = SimpleNamespace()
        mapper = object()
        map_state = {"scale": 1.0}
        signature = ("sig",)

        ctx1 = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, signature, "Cartesian2Axis")
        ctx2 = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, signature, "Cartesian2Axis")

        self.assertIs(ctx1["plan"], first_plan)
        self.assertIs(ctx2["plan"], first_plan)
        self.assertEqual(first_plan.update_calls, 2)

        new_signature = ("sig", 2)
        ctx3 = renderer._resolve_cartesian_plan(cartesian, mapper, map_state, new_signature, "Cartesian2Axis")
        self.assertIs(ctx3["plan"], second_plan)

    def test_prune_unused_plan_entries_drops_orphans(self) -> None:
        renderer = self._make_renderer()
        dropped: list[str] = []

        def drop_group(plan_key: str) -> None:
            dropped.append(plan_key)

        renderer._shared_primitives = SimpleNamespace(
            reserve_usage_counts=lambda *_args, **_kwargs: None,
            drop_group=drop_group,
            push_group_to_back=lambda *_args, **_kwargs: None,
        )

        stale_plan = PlanStub(plan_key="group-1")
        renderer._plan_cache = {"cache-key": {"plan": stale_plan, "signature": ("sig",)}}
        renderer._frame_seen_plan_keys = set()

        renderer._prune_unused_plan_entries()

        self.assertEqual(renderer._plan_cache, {})
        self.assertEqual(dropped, ["group-1"])
        self.assertEqual(renderer._frame_seen_plan_keys, set())


__all__ = ["TestSvgRendererPlan"]
