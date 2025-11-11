from __future__ import annotations

import sys
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "static" / "client"
CLIENT_ROOT_STR = str(CLIENT_ROOT)
if CLIENT_ROOT.exists() and CLIENT_ROOT_STR not in sys.path:
    sys.path.insert(0, CLIENT_ROOT_STR)

import unittest
from types import SimpleNamespace

from rendering import optimized_drawable_renderers as optimized
from rendering import shared_drawable_renderers as shared
from rendering import style_manager
from rendering.primitives import ClosedArea

from .renderer_fixtures import CoordinateMapperStub, PrimitiveRecorder


class TestRendererSupportPlan(unittest.TestCase):
    def test_render_point_helper_emits_label_metadata(self) -> None:
        primitives = PrimitiveRecorder()
        mapper = CoordinateMapperStub(scale_factor=1.0, origin=(0.0, 0.0), offset=(2.0, -3.0))
        point = SimpleNamespace(x=4.0, y=1.5, name="P", color="#123")
        style = {
            "point_radius": 6,
            "point_color": "#000",
            "point_label_font_size": 14,
        }

        shared.render_point_helper(primitives, point, mapper, style)

        draw_calls = [entry for entry in primitives.calls if entry[0] == "draw_text"]
        self.assertEqual(len(draw_calls), 1)
        metadata = draw_calls[0][2]["metadata"]
        self.assertIn("point_label", metadata)
        label_meta = metadata["point_label"]
        self.assertEqual(label_meta["math_position"], (point.x, point.y))
        self.assertEqual(label_meta["screen_offset"], (style["point_radius"], -style["point_radius"]))

    def test_map_state_equal_tolerance(self) -> None:
        baseline = {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}
        nearly_same = {"scale": 1.0 + 1e-7, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}
        different = {"scale": 1.0 + 1e-4, "offset_x": 0.0, "offset_y": 0.0, "origin_x": 0.0, "origin_y": 0.0}

        self.assertTrue(optimized._map_state_equal(baseline, nearly_same))
        self.assertFalse(optimized._map_state_equal(baseline, different))

    def test_get_renderer_style_returns_independent_copy(self) -> None:
        first = style_manager.get_renderer_style()
        second = style_manager.get_renderer_style()
        first["point_color"] = "magenta"

        self.assertNotEqual(first["point_color"], second["point_color"])
        self.assertEqual(second["point_color"], style_manager.get_default_style_value("point_color"))

    def test_closed_area_defaults_to_empty_lists(self) -> None:
        area = ClosedArea(forward_points=None, reverse_points=None, color="#000", opacity=0.5)

        self.assertEqual(area.forward_points, [])
        self.assertEqual(area.reverse_points, [])
        self.assertFalse(area.is_screen)
        self.assertEqual(area.color, "#000")
        self.assertEqual(area.opacity, 0.5)


__all__ = ["TestRendererSupportPlan"]
