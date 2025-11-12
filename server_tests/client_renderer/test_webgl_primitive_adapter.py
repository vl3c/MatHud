from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from typing import List, Tuple

from rendering.shared_drawable_renderers import StrokeStyle, FillStyle


class MockWebGLRenderer:
    def __init__(self):
        self.draw_calls: List[Tuple[str, ...]] = []

    def _draw_lines(self, points: List[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        self.draw_calls.append(("draw_lines", points, color))

    def _draw_line_strip(self, points: List[Tuple[float, float]], color: Tuple[float, float, float, float]) -> None:
        self.draw_calls.append(("draw_line_strip", points, color))

    def _draw_points(self, points: List[Tuple[float, float]], color: Tuple[float, float, float, float], size: float) -> None:
        self.draw_calls.append(("draw_points", points, color, size))

    def _parse_color(self, color: str) -> Tuple[float, float, float, float]:
        if color.startswith("#"):
            if len(color) == 7:
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0
                return (r, g, b, 1.0)
        return (1.0, 1.0, 1.0, 1.0)


class TestWebGLPrimitiveAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        self.adapter = WebGLPrimitiveAdapter(self.renderer)

    def test_stroke_line_draws_lines(self) -> None:
        stroke = StrokeStyle(color="#FF0000", width=2.0)
        
        self.adapter.stroke_line((10.0, 20.0), (30.0, 40.0), stroke)
        
        self.assertEqual(len(self.renderer.draw_calls), 1)
        call_type, points, color = self.renderer.draw_calls[0]
        self.assertEqual(call_type, "draw_lines")
        self.assertEqual(len(points), 2)

    def test_stroke_polyline_draws_line_strip(self) -> None:
        points = [(10.0, 20.0), (30.0, 40.0), (50.0, 30.0)]
        stroke = StrokeStyle(color="#00FF00", width=1.5)
        
        self.adapter.stroke_polyline(points, stroke)
        
        self.assertEqual(len(self.renderer.draw_calls), 1)
        call_type, drawn_points, color = self.renderer.draw_calls[0]
        self.assertEqual(call_type, "draw_line_strip")
        self.assertEqual(len(drawn_points), 3)

    def test_fill_circle_draws_point(self) -> None:
        fill = FillStyle(color="#0000FF", opacity=0.5)
        
        self.adapter.fill_circle((50.0, 60.0), 20.0, fill)
        
        point_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_points"]
        self.assertEqual(len(point_calls), 1)

    def test_stroke_circle_approximates_with_line_strip(self) -> None:
        stroke = StrokeStyle(color="#FFFF00", width=1.0)
        
        self.adapter.stroke_circle((80.0, 90.0), 25.0, stroke)
        
        strip_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_line_strip"]
        self.assertEqual(len(strip_calls), 1)
        
        _, points, _ = strip_calls[0]
        self.assertGreater(len(points), 8)

    def test_stroke_ellipse_approximates_with_samples(self) -> None:
        stroke = StrokeStyle(color="#FF00FF", width=1.0)
        
        self.adapter.stroke_ellipse((100.0, 110.0), 30.0, 20.0, 0.5, stroke)
        
        strip_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_line_strip"]
        self.assertEqual(len(strip_calls), 1)

    def test_fill_polygon_uses_line_strip_fallback(self) -> None:
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        fill = FillStyle(color="#00FFFF")
        
        self.adapter.fill_polygon(points, fill)
        
        strip_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_line_strip"]
        self.assertEqual(len(strip_calls), 1)
        
        _, drawn_points, _ = strip_calls[0]
        self.assertEqual(drawn_points[0], drawn_points[-1])

    def test_fill_joined_area_closes_outline(self) -> None:
        forward = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
        reverse = [(10.0, 10.0), (0.0, 10.0), (0.0, 5.0)]
        fill = FillStyle(color="#FFAA00")
        
        self.adapter.fill_joined_area(forward, reverse, fill)
        
        strip_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_line_strip"]
        self.assertEqual(len(strip_calls), 1)
        
        _, drawn_points, _ = strip_calls[0]
        self.assertEqual(drawn_points[0], drawn_points[-1])

    def test_stroke_arc_approximates_with_samples(self) -> None:
        stroke = StrokeStyle(color="#AA00FF", width=1.0)
        
        self.adapter.stroke_arc((120.0, 130.0), 40.0, 0.0, 1.57, True, stroke)
        
        strip_calls = [call for call in self.renderer.draw_calls if call[0] == "draw_line_strip"]
        self.assertEqual(len(strip_calls), 1)

    def test_color_parsing_hex_format(self) -> None:
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        renderer = MockWebGLRenderer()
        adapter = WebGLPrimitiveAdapter(renderer)
        
        stroke = StrokeStyle(color="#FF8040", width=1.0)
        adapter.stroke_line((0.0, 0.0), (10.0, 10.0), stroke)
        
        self.assertEqual(len(renderer.draw_calls), 1)
        _, _, color = renderer.draw_calls[0]
        
        self.assertAlmostEqual(color[0], 1.0, places=2)
        self.assertAlmostEqual(color[1], 0x80 / 255.0, places=2)
        self.assertAlmostEqual(color[2], 0x40 / 255.0, places=2)


class TestWebGLPrimitiveAdapterSampling(unittest.TestCase):
    def test_circle_sampling_produces_closed_path(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        samples = adapter._sample_circle((50.0, 60.0), 20.0)
        
        self.assertGreater(len(samples), 8)
        
        first_point = samples[0]
        last_point = samples[-1]
        self.assertAlmostEqual(first_point[0], last_point[0], places=5)
        self.assertAlmostEqual(first_point[1], last_point[1], places=5)

    def test_ellipse_sampling_produces_closed_path(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        samples = adapter._sample_ellipse((70.0, 80.0), 30.0, 20.0, 0.0)
        
        self.assertGreater(len(samples), 8)
        
        first_point = samples[0]
        last_point = samples[-1]
        self.assertAlmostEqual(first_point[0], last_point[0], places=5)
        self.assertAlmostEqual(first_point[1], last_point[1], places=5)

    def test_arc_sampling_respects_angle_range(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        import math
        samples = adapter._sample_arc((100.0, 110.0), 50.0, 0.0, math.pi / 2, True)
        
        self.assertGreater(len(samples), 2)


class TestWebGLPrimitiveAdapterEdgeCases(unittest.TestCase):
    def test_empty_polyline_does_not_crash(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        stroke = StrokeStyle(color="#000000", width=1.0)
        
        try:
            adapter.stroke_polyline([], stroke)
        except Exception as e:
            self.fail(f"Empty polyline raised exception: {e}")

    def test_single_point_polyline_does_not_crash(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        stroke = StrokeStyle(color="#000000", width=1.0)
        
        try:
            adapter.stroke_polyline([(5.0, 5.0)], stroke)
        except Exception as e:
            self.fail(f"Single point polyline raised exception: {e}")

    def test_zero_radius_circle_handles_gracefully(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        fill = FillStyle(color="#FF0000")
        
        try:
            adapter.fill_circle((10.0, 10.0), 0.0, fill)
        except Exception as e:
            self.fail(f"Zero radius circle raised exception: {e}")

    def test_negative_radius_circle_handles_gracefully(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        fill = FillStyle(color="#FF0000")
        
        try:
            adapter.fill_circle((10.0, 10.0), -10.0, fill)
        except Exception as e:
            self.fail(f"Negative radius circle raised exception: {e}")

    def test_fill_polygon_with_single_point_handles_gracefully(self) -> None:
        renderer = MockWebGLRenderer()
        from rendering.webgl_primitive_adapter import WebGLPrimitiveAdapter
        adapter = WebGLPrimitiveAdapter(renderer)
        
        fill = FillStyle(color="#00FF00")
        
        try:
            adapter.fill_polygon([(5.0, 5.0)], fill)
        except Exception as e:
            self.fail(f"Single point polygon raised exception: {e}")


__all__ = [
    "TestWebGLPrimitiveAdapter",
    "TestWebGLPrimitiveAdapterSampling",
    "TestWebGLPrimitiveAdapterEdgeCases",
]

