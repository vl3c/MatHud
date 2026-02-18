from __future__ import annotations

import unittest

from drawables.bar import Bar
from rendering import shared_drawable_renderers as shared
from rendering.primitives import FillStyle, StrokeStyle, TextAlignment


class RecordingPrimitives(shared.RendererPrimitives):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def fill_polygon(self, points, fill, stroke=None, **kwargs):
        self.calls.append(("fill_polygon", (points, fill, stroke), dict(kwargs)))

    def draw_text(self, text, position, font, color, alignment, style_overrides=None, **kwargs):
        self.calls.append(("draw_text", (text, position, font, color, alignment, style_overrides), dict(kwargs)))

    # The remaining primitives are not needed for these tests.
    def stroke_line(self, *_args, **_kwargs):
        raise NotImplementedError

    def stroke_polyline(self, *_args, **_kwargs):
        raise NotImplementedError

    def stroke_circle(self, *_args, **_kwargs):
        raise NotImplementedError

    def fill_circle(self, *_args, **_kwargs):
        raise NotImplementedError

    def stroke_ellipse(self, *_args, **_kwargs):
        raise NotImplementedError

    def fill_joined_area(self, *_args, **_kwargs):
        raise NotImplementedError

    def stroke_arc(self, *_args, **_kwargs):
        raise NotImplementedError

    def clear_surface(self, *_args, **_kwargs):
        raise NotImplementedError

    def resize_surface(self, *_args, **_kwargs):
        raise NotImplementedError


class SimpleCoordinateMapper:
    def __init__(self, scale: float = 10.0) -> None:
        self.scale = float(scale)

    def math_to_screen(self, x: float, y: float):
        return (float(x) * self.scale, float(y) * self.scale)


class TestBarRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.primitives = RecordingPrimitives()
        self.mapper = SimpleCoordinateMapper(scale=10.0)
        self.style = {
            "default_area_fill_color": "#88aaff",
            "segment_width": 2,
            "bar_label_padding_px": 6,
            "label_color": "#000",
            "label_font_family": "Arial",
            "label_font_size": 12,
        }

    def test_renders_fill_polygon_with_defaults(self) -> None:
        bar = Bar(name="B", x_left=1.0, x_right=3.0, y_bottom=0.0, y_top=2.0)
        shared.render_bar_helper(self.primitives, bar, self.mapper, self.style)

        poly_calls = [c for c in self.primitives.calls if c[0] == "fill_polygon"]
        self.assertEqual(len(poly_calls), 1)

        _, args, kwargs = poly_calls[0]
        points, fill, stroke = args
        self.assertEqual(points, ((10.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0)))
        self.assertTrue(kwargs.get("screen_space", False))

        self.assertIsInstance(fill, FillStyle)
        self.assertEqual(fill.color, "#88aaff")
        self.assertIsNone(fill.opacity)

        self.assertIsNone(stroke)

    def test_renders_stroke_when_color_is_set(self) -> None:
        bar = Bar(
            name="B",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            stroke_color="#123",
        )
        shared.render_bar_helper(self.primitives, bar, self.mapper, self.style)

        poly_call = next(c for c in self.primitives.calls if c[0] == "fill_polygon")
        _, args, _ = poly_call
        _, _, stroke = args

        self.assertIsInstance(stroke, StrokeStyle)
        self.assertEqual(stroke.color, "#123")
        self.assertEqual(stroke.width, 2.0)

    def test_draws_labels_and_falls_back_to_label_text(self) -> None:
        bar = Bar(
            name="B",
            x_left=0.0,
            x_right=2.0,
            y_bottom=0.0,
            y_top=1.0,
            label_text="42",
            label_below_text="A",
        )
        shared.render_bar_helper(self.primitives, bar, self.mapper, self.style)

        text_calls = [c for c in self.primitives.calls if c[0] == "draw_text"]
        self.assertEqual(len(text_calls), 2)

        _, args, kwargs = text_calls[0]
        text, position, _font, _color, alignment, _overrides = args
        self.assertEqual(text, "42")
        self.assertEqual(position, (10.0, -6.0))
        self.assertIsInstance(alignment, TextAlignment)
        self.assertEqual(alignment.horizontal, "center")
        self.assertEqual(alignment.vertical, "bottom")
        self.assertTrue(kwargs.get("screen_space", False))

        _, args, kwargs = text_calls[1]
        text, position, _font, _color, alignment, _overrides = args
        self.assertEqual(text, "A")
        self.assertEqual(position, (10.0, 16.0))
        self.assertEqual(alignment.horizontal, "center")
        self.assertEqual(alignment.vertical, "top")
        self.assertTrue(kwargs.get("screen_space", False))

    def test_skips_blank_labels(self) -> None:
        bar = Bar(
            name="B",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            label_above_text="   ",
            label_below_text="",
        )
        shared.render_bar_helper(self.primitives, bar, self.mapper, self.style)
        text_calls = [c for c in self.primitives.calls if c[0] == "draw_text"]
        self.assertEqual(text_calls, [])

    def test_guard_clauses_skip_invalid_geometry(self) -> None:
        bar = Bar(name="B", x_left=1.0, x_right=1.0, y_bottom=0.0, y_top=1.0)
        shared.render_bar_helper(self.primitives, bar, self.mapper, self.style)
        self.assertEqual(self.primitives.calls, [])

        self.primitives.calls.clear()
        bar2 = Bar(name="B2", x_left=0.0, x_right=1.0, y_bottom=2.0, y_top=2.0)
        shared.render_bar_helper(self.primitives, bar2, self.mapper, self.style)
        self.assertEqual(self.primitives.calls, [])
