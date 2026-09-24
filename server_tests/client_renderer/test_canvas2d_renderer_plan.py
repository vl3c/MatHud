from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

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

    def _install_drawable_plan_builder(self) -> list:
        built: list = []
        original = canvas2d_renderer.build_plan_for_drawable

        class _OffscreenPlan(canvas2d_renderer.OptimizedPrimitivePlan):
            def __init__(self) -> None:
                pass

            def update_map_state(self, _map_state) -> None:
                pass

            def is_visible(self, _width, _height, margin=0.0) -> bool:
                return False

        def fake_builder(drawable, mapper, style, supports_transform=False):
            built.append(drawable)
            return _OffscreenPlan()

        canvas2d_renderer.build_plan_for_drawable = fake_builder
        self.addCleanup(setattr, canvas2d_renderer, "build_plan_for_drawable", original)
        return built

    def _make_frame_renderer(self) -> canvas2d_renderer.Canvas2DRenderer:
        renderer = self._make_renderer()
        renderer._frame_seen_plan_keys = set()
        renderer._shared_primitives = SimpleNamespace(begin_frame=lambda: None, end_frame=lambda: None)
        renderer._telemetry.begin_frame = lambda: None
        renderer._telemetry.end_frame = lambda: None
        del renderer._is_cached_plan_valid
        return renderer

    def _resolve(self, renderer, drawable) -> None:
        renderer._render_drawable(drawable, None)

    def test_invalidate_drawable_cache_forces_rebuild(self) -> None:
        renderer = self._make_frame_renderer()
        built = self._install_drawable_plan_builder()
        drawable = SimpleNamespace(name="A", get_state=lambda: {"x": 1})

        self._resolve(renderer, drawable)
        self._resolve(renderer, drawable)
        self.assertEqual(len(built), 1)

        renderer.invalidate_drawable_cache(drawable)
        self._resolve(renderer, drawable)
        self.assertEqual(len(built), 2)

        renderer.invalidate_all_drawable_caches()
        self.assertEqual(renderer._plan_cache, {})

    def test_end_frame_prunes_plans_for_drawables_not_rendered(self) -> None:
        renderer = self._make_frame_renderer()
        self._install_drawable_plan_builder()
        kept = SimpleNamespace(name="kept", get_state=lambda: {})
        deleted = SimpleNamespace(name="deleted", get_state=lambda: {})

        renderer.begin_frame()
        self._resolve(renderer, kept)
        self._resolve(renderer, deleted)
        renderer.end_frame()
        self.assertEqual(len(renderer._plan_cache), 2)

        renderer.begin_frame()
        self._resolve(renderer, kept)
        renderer.end_frame()
        self.assertEqual(list(renderer._plan_cache.keys()), ["SimpleNamespace:kept"])

    def test_signature_tracks_dependent_point_coordinates(self) -> None:
        renderer = self._make_renderer()
        center = SimpleNamespace(x=1.0, y=2.0, name="A")
        circle = SimpleNamespace(name="c", center=center, get_state=lambda: {"args": {"center": "A"}})
        vertex = SimpleNamespace(x=0.0, y=0.0)
        angle = SimpleNamespace(
            name="ang",
            vertex_point=vertex,
            arm1_point=SimpleNamespace(x=1.0, y=0.0),
            arm2_point=SimpleNamespace(x=0.0, y=1.0),
            get_state=lambda: {"args": {"segment1_name": "s1"}},
        )

        circle_before = renderer._compute_drawable_signature(circle)
        angle_before = renderer._compute_drawable_signature(angle)
        center.x = 5.0
        vertex.y = -3.0

        self.assertNotEqual(renderer._compute_drawable_signature(circle), circle_before)
        self.assertNotEqual(renderer._compute_drawable_signature(angle), angle_before)

    def test_function_signature_changes_when_canvas_resizes(self) -> None:
        renderer = self._make_renderer()
        function = SimpleNamespace(name="f", get_class_name=lambda: "Function", get_state=lambda: {})
        mapper = SimpleNamespace(
            scale_factor=1.0, offset=SimpleNamespace(x=0.0, y=0.0), canvas_width=800, canvas_height=600
        )

        before = renderer._compute_drawable_signature(function, mapper)
        mapper.canvas_width = 1200

        self.assertNotEqual(renderer._compute_drawable_signature(function, mapper), before)

    def test_flush_offscreen_draws_back_to_main_canvas(self) -> None:
        renderer = self._make_renderer()
        renderer._use_layer_compositing = True
        renderer._offscreen_canvas = object()

        renderer._flush_offscreen_to_main()

        self.assertEqual(len(renderer.ctx.draw_image_calls), 1)
        self.assertIs(renderer.ctx.draw_image_calls[0][0], renderer._offscreen_canvas)


__all__ = ["TestCanvas2DRendererPlan"]
