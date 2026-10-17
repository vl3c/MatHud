from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple

from rendering import cached_render_plan as optimized
from rendering import shared_drawable_renderers as shared
from rendering import style_manager
from rendering.canvas2d_renderer import Canvas2DRenderer
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.primitives import FontStyle
from rendering.svg_renderer import SvgRenderer


class CoordinateMapperStub:
    def __init__(self) -> None:
        self.scale_factor = 1.0

    def math_to_screen(self, x: float, y: float):
        return (x + 10.0, y + 5.0)

    def scale_value(self, value: float) -> float:
        return value


class PrimitiveRecorder(shared.RendererPrimitives):
    def __init__(self) -> None:
        self.calls = []

    def fill_circle(self, center, radius, fill, stroke=None, *, screen_space=False):
        self.calls.append(("fill_circle", center, radius, fill, screen_space))

    def draw_text(
        self,
        text,
        position,
        font,
        color,
        alignment,
        style_overrides=None,
        *,
        screen_space=False,
        metadata=None,
    ):
        self.calls.append(("draw_text", text, position, metadata))


class TestRendererLogic(unittest.TestCase):
    def _view_changes(self, renderer: Any, class_name: str) -> Tuple[bool, bool, bool]:
        """Whether a small pan, a pan of a large part of the screen and a zoom change the plan signature."""
        drawable = SimpleNamespace(name="d", color="red", get_class_name=lambda: class_name, get_state=lambda: {})

        def mapper(offset_x: float, scale: float = 1.0) -> Any:
            return SimpleNamespace(
                scale_factor=scale, offset=SimpleNamespace(x=offset_x, y=0.0), canvas_width=800, canvas_height=600
            )

        def changes(before: Any, after: Any) -> bool:
            return renderer._compute_drawable_signature(drawable, before) != renderer._compute_drawable_signature(
                drawable, after
            )

        return (
            changes(mapper(0.0), mapper(10.0)),
            changes(mapper(0.0), mapper(500.0)),
            changes(mapper(0.0), mapper(0.0, 2.0)),
        )

    def test_canvas2d_plan_signatures_follow_the_view_mode(self) -> None:
        renderer = Canvas2DRenderer.__new__(Canvas2DRenderer)
        renderer.style = {"function_view_margin": 0.5}  # half-screen buckets: 400 px wide here
        expected = {
            "Point": (False, False, False),
            "ParametricFunction": (False, False, True),
            "Function": (False, True, True),
            "PiecewiseFunction": (False, True, True),
            "FunctionsBoundedColoredArea": (True, True, True),
            "PolarGrid": (True, True, True),
        }
        for class_name, changes in expected.items():
            self.assertEqual(self._view_changes(renderer, class_name), changes, class_name)

    def test_svg_plan_signatures_change_with_any_view_change_when_view_dependent(self) -> None:
        renderer = SvgRenderer.__new__(SvgRenderer)
        self.assertEqual(self._view_changes(renderer, "Point"), (False, False, False))
        for class_name in ("ParametricFunction", "Function", "FunctionSegmentBoundedColoredArea", "PolarGrid"):
            self.assertEqual(self._view_changes(renderer, class_name), (True, True, True), class_name)

    def test_both_renderers_draw_every_class_with_a_plan_helper_by_class_name(self) -> None:
        for renderer_class in (Canvas2DRenderer, SvgRenderer):
            renderer = renderer_class.__new__(renderer_class)
            renderer._handlers_by_type = {}
            drawn: List[str] = []
            renderer._render_drawable = lambda drawable, mapper, name=None, drawn=drawn: drawn.append(name)
            for class_name in optimized._HELPERS:
                drawable = SimpleNamespace(get_class_name=lambda name=class_name: name)
                self.assertTrue(renderer.render(drawable, CoordinateMapperStub()), class_name)
            self.assertFalse(
                renderer.render(SimpleNamespace(get_class_name=lambda: "Triangle"), CoordinateMapperStub())
            )
            self.assertEqual(drawn, list(optimized._HELPERS))

    def test_a_handler_registered_for_a_type_runs_before_the_plan_helper(self) -> None:
        for renderer_class in (Canvas2DRenderer, SvgRenderer):
            renderer = renderer_class.__new__(renderer_class)
            handled: List[Any] = []
            renderer._handlers_by_type = {SimpleNamespace: lambda drawable, mapper: handled.append(drawable)}
            renderer._render_drawable = lambda drawable, mapper, name=None: self.fail("plan helper used")
            drawable = SimpleNamespace(get_class_name=lambda: "Point")
            self.assertTrue(renderer.render(drawable, CoordinateMapperStub()))
            self.assertEqual(handled, [drawable])

    def test_style_manager_returns_independent_copy(self) -> None:
        style_a = style_manager.get_renderer_style()
        style_b = style_manager.get_renderer_style()

        style_a["point_color"] = "magenta"

        self.assertNotEqual(style_a["point_color"], style_b["point_color"])
        self.assertEqual(
            style_b["point_color"],
            style_manager.get_default_style_value("point_color"),
        )

    def test_map_state_equal_tolerance(self) -> None:
        base = {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}
        close = {"scale": 1.0 + 1e-7, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}
        far = {"scale": 1.0 + 1e-4, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}

        self.assertTrue(optimized._map_state_equal(base, close))
        self.assertFalse(optimized._map_state_equal(base, far))

    def test_render_point_helper_records_metadata(self) -> None:
        primitives = PrimitiveRecorder()
        mapper = CoordinateMapperStub()
        point = type("Point", (), {"x": 2.0, "y": -3.0, "name": "P", "color": "#123"})()
        style = {
            "point_radius": 4,
            "point_color": "#000",
            "point_label_font_size": 12,
        }

        shared.render_point_helper(primitives, point, mapper, style)

        draw_calls = [call for call in primitives.calls if call[0] == "draw_text"]
        self.assertTrue(draw_calls, "draw_text not emitted")
        _, _, _, metadata = draw_calls[0]
        self.assertIsNotNone(metadata)
        label_meta = metadata["point_label"]
        self.assertEqual(label_meta["math_position"], (point.x, point.y))
        self.assertEqual(label_meta["screen_offset"], (float(style["point_radius"]), float(-style["point_radius"])))

    def test_canvas2d_font_cache_quantizes_similar_sizes(self) -> None:
        canvas_el = SimpleNamespace(getContext=lambda _kind: SimpleNamespace())
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        adapter.MAX_FONT_CACHE_ENTRIES = 128

        f1 = FontStyle(family="Arial", size=9.60, weight=None)
        f2 = FontStyle(family="Arial", size=9.62, weight=None)
        s1 = adapter._resolve_font_string(f1)
        s2 = adapter._resolve_font_string(f2)

        self.assertEqual(s1, s2)
        self.assertLessEqual(len(adapter._font_cache), 1)

    def test_canvas2d_font_cache_is_bounded(self) -> None:
        canvas_el = SimpleNamespace(getContext=lambda _kind: SimpleNamespace())
        adapter = Canvas2DPrimitiveAdapter(canvas_el)
        adapter.MAX_FONT_CACHE_ENTRIES = 64

        for i in range(512):
            size = 10.0 + (i * 0.25)
            adapter._resolve_font_string(FontStyle(family="Arial", size=size, weight=None))

        self.assertLessEqual(len(adapter._font_cache), 64)

    def test_canvas2d_zoom_cycle_does_not_grow_font_cache_metrics(self) -> None:
        renderer = Canvas2DRenderer.__new__(Canvas2DRenderer)
        renderer._telemetry = SimpleNamespace(snapshot=lambda: {}, drain=lambda: {})
        renderer._plan_cache = {}

        canvas_el = SimpleNamespace(getContext=lambda _kind: SimpleNamespace())
        primitives = Canvas2DPrimitiveAdapter(canvas_el)
        primitives.MAX_FONT_CACHE_ENTRIES = 64
        renderer._shared_primitives = primitives

        base_size = 24.0
        scales = [max(0.01, 1.0 - (i / 100.0)) for i in range(100)]
        cycle = scales + list(reversed(scales))

        for scale in cycle:
            primitives._resolve_font_string(FontStyle(family="Arial", size=base_size * scale, weight=None))

        snap1 = renderer.peek_telemetry()
        self.assertIn("font_cache_entries", snap1)
        self.assertLessEqual(int(snap1.get("font_cache_entries", 0) or 0), 64)

        for scale in cycle:
            primitives._resolve_font_string(FontStyle(family="Arial", size=base_size * scale, weight=None))

        snap2 = renderer.peek_telemetry()
        self.assertIn("font_cache_entries", snap2)
        self.assertLessEqual(int(snap2.get("font_cache_entries", 0) or 0), 64)
        self.assertEqual(int(snap1.get("font_cache_entries", 0) or 0), int(snap2.get("font_cache_entries", 0) or 0))

    def test_canvas2d_signature_changes_when_circle_center_moves(self) -> None:
        from drawables.circle import Circle
        from drawables.point import Point

        renderer = Canvas2DRenderer.__new__(Canvas2DRenderer)
        center = Point(0, 0, name="A")
        circle = Circle(center, 2)
        before = renderer._compute_drawable_signature(circle)
        center.x = 3
        self.assertNotEqual(renderer._compute_drawable_signature(circle), before)
        self.assertNotIn("_dependent_coords", str(circle.get_state()))


__all__ = ["TestRendererLogic"]
