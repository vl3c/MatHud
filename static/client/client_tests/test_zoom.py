from __future__ import annotations

import unittest

from canvas import Canvas
from .simple_mock import SimpleMock


class ZoomTestFixture:
    """Provides canvas instances with various dimensions for zoom testing."""
    
    @staticmethod
    def create_canvas(width: float, height: float) -> Canvas:
        """Create a canvas with specified dimensions and mocked draw."""
        canvas = Canvas(width, height, draw_enabled=False)
        canvas.draw = SimpleMock(return_value=None)
        return canvas


class TestZoomXAxisRange(unittest.TestCase):
    """Tests for zoom with range_axis='x'."""
    
    def test_zoom_x_axis_origin_center(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=0, center_y=0, range_val=5, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(left, -5, places=6)
        self.assertAlmostEqual(right, 5, places=6)
        expected_y_range = 5 * (480 / 640)
        self.assertAlmostEqual(top, expected_y_range, places=6)
        self.assertAlmostEqual(bottom, -expected_y_range, places=6)
    
    def test_zoom_x_axis_non_origin_center(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=10, center_y=20, range_val=3, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(left, 7, places=6)
        self.assertAlmostEqual(right, 13, places=6)
        expected_y_range = 3 * (480 / 640)
        self.assertAlmostEqual(top, 20 + expected_y_range, places=6)
        self.assertAlmostEqual(bottom, 20 - expected_y_range, places=6)


class TestZoomYAxisRange(unittest.TestCase):
    """Tests for zoom with range_axis='y'."""
    
    def test_zoom_y_axis_origin_center(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=0, center_y=0, range_val=5, range_axis="y")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(top, 5, places=6)
        self.assertAlmostEqual(bottom, -5, places=6)
        expected_x_range = 5 * (640 / 480)
        self.assertAlmostEqual(left, -expected_x_range, places=6)
        self.assertAlmostEqual(right, expected_x_range, places=6)
    
    def test_zoom_y_axis_non_origin_center(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=10, center_y=20, range_val=3, range_axis="y")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(top, 23, places=6)
        self.assertAlmostEqual(bottom, 17, places=6)
        expected_x_range = 3 * (640 / 480)
        self.assertAlmostEqual(left, 10 - expected_x_range, places=6)
        self.assertAlmostEqual(right, 10 + expected_x_range, places=6)


class TestZoomAspectRatios(unittest.TestCase):
    """Tests for zoom with different canvas aspect ratios."""
    
    def test_zoom_wide_canvas_x_axis(self) -> None:
        canvas = ZoomTestFixture.create_canvas(800, 400)
        canvas.zoom(center_x=0, center_y=0, range_val=10, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(left, -10, places=6)
        self.assertAlmostEqual(right, 10, places=6)
        expected_y_range = 10 * (400 / 800)
        self.assertAlmostEqual(top, expected_y_range, places=6)
        self.assertAlmostEqual(bottom, -expected_y_range, places=6)
    
    def test_zoom_wide_canvas_y_axis(self) -> None:
        canvas = ZoomTestFixture.create_canvas(800, 400)
        canvas.zoom(center_x=0, center_y=0, range_val=10, range_axis="y")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(top, 10, places=6)
        self.assertAlmostEqual(bottom, -10, places=6)
        expected_x_range = 10 * (800 / 400)
        self.assertAlmostEqual(left, -expected_x_range, places=6)
        self.assertAlmostEqual(right, expected_x_range, places=6)
    
    def test_zoom_tall_canvas_x_axis(self) -> None:
        canvas = ZoomTestFixture.create_canvas(400, 800)
        canvas.zoom(center_x=0, center_y=0, range_val=10, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(left, -10, places=6)
        self.assertAlmostEqual(right, 10, places=6)
        expected_y_range = 10 * (800 / 400)
        self.assertAlmostEqual(top, expected_y_range, places=6)
        self.assertAlmostEqual(bottom, -expected_y_range, places=6)
    
    def test_zoom_tall_canvas_y_axis(self) -> None:
        canvas = ZoomTestFixture.create_canvas(400, 800)
        canvas.zoom(center_x=0, center_y=0, range_val=10, range_axis="y")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(top, 10, places=6)
        self.assertAlmostEqual(bottom, -10, places=6)
        expected_x_range = 10 * (400 / 800)
        self.assertAlmostEqual(left, -expected_x_range, places=6)
        self.assertAlmostEqual(right, expected_x_range, places=6)
    
    def test_zoom_square_canvas(self) -> None:
        canvas = ZoomTestFixture.create_canvas(600, 600)
        canvas.zoom(center_x=5, center_y=5, range_val=10, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        top = canvas.coordinate_mapper.get_visible_top_bound()
        bottom = canvas.coordinate_mapper.get_visible_bottom_bound()
        
        self.assertAlmostEqual(left, -5, places=6)
        self.assertAlmostEqual(right, 15, places=6)
        self.assertAlmostEqual(top, 15, places=6)
        self.assertAlmostEqual(bottom, -5, places=6)


class TestZoomEdgeCases(unittest.TestCase):
    """Tests for zoom edge cases."""
    
    def test_zoom_small_range(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=0, center_y=0, range_val=0.1, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        
        self.assertAlmostEqual(left, -0.1, places=6)
        self.assertAlmostEqual(right, 0.1, places=6)
    
    def test_zoom_large_range(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=0, center_y=0, range_val=1000, range_axis="x")
        
        left = canvas.coordinate_mapper.get_visible_left_bound()
        right = canvas.coordinate_mapper.get_visible_right_bound()
        
        self.assertAlmostEqual(left, -1000, places=6)
        self.assertAlmostEqual(right, 1000, places=6)
    
    def test_zoom_calls_draw(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        canvas.zoom(center_x=0, center_y=0, range_val=5, range_axis="x")
        
        canvas.draw.assert_called_once()
        draw_args, draw_kwargs = canvas.draw.calls[0]
        self.assertEqual(draw_args, ())
        self.assertEqual(draw_kwargs, {'apply_zoom': True})
    
    def test_zoom_returns_true(self) -> None:
        canvas = ZoomTestFixture.create_canvas(640, 480)
        result = canvas.zoom(center_x=0, center_y=0, range_val=5, range_axis="x")
        self.assertTrue(result)

