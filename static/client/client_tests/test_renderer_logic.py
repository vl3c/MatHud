from __future__ import annotations

import unittest
from types import SimpleNamespace

from drawables.bar import Bar
from rendering import cached_render_plan as optimized
from rendering import shared_drawable_renderers as shared
from rendering import style_manager
from rendering.canvas2d_renderer import Canvas2DRenderer
from rendering.canvas2d_primitive_adapter import Canvas2DPrimitiveAdapter
from rendering.primitives import FontStyle
from rendering.svg_renderer import SvgRenderer
from rendering.webgl_renderer import WebGLRenderer


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
    def test_renderers_register_bar_drawable(self) -> None:
        canvas2d = Canvas2DRenderer.__new__(Canvas2DRenderer)
        canvas2d._handlers_by_type = {}
        canvas2d.register_default_drawables()
        self.assertIn(Bar, canvas2d._handlers_by_type)

        svg = SvgRenderer.__new__(SvgRenderer)
        svg._handlers_by_type = {}
        svg.register_default_drawables()
        self.assertIn(Bar, svg._handlers_by_type)

        webgl = WebGLRenderer.__new__(WebGLRenderer)
        webgl._handlers_by_type = {}
        webgl.register_default_drawables()
        self.assertIn(Bar, webgl._handlers_by_type)

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
        self.assertEqual(label_meta["screen_offset"], (float(style["point_radius"]), float(-style["point_radius"])) )

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


__all__ = ["TestRendererLogic"]
