from __future__ import annotations

import unittest

from rendering import cached_render_plan as optimized
from rendering import shared_drawable_renderers as shared
from rendering import style_manager


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


__all__ = ["TestRendererLogic"]
