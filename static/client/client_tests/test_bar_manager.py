from __future__ import annotations

import unittest

from canvas import Canvas


class TestBarManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.bar_manager = self.canvas.drawable_manager.bar_manager

    def test_create_bar_creates_drawable(self) -> None:
        bar = self.bar_manager.create_bar(
            name="TestBar",
            x_left=0.0,
            x_right=2.0,
            y_bottom=0.0,
            y_top=3.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            label_above_text="above",
            label_below_text="below",
            archive=False,
            redraw=False,
        )

        self.assertEqual(bar.name, "TestBar")
        self.assertEqual(bar.x_left, 0.0)
        self.assertEqual(bar.x_right, 2.0)
        self.assertEqual(bar.y_bottom, 0.0)
        self.assertEqual(bar.y_top, 3.0)
        self.assertEqual(bar.fill_color, "#222")
        self.assertEqual(bar.fill_opacity, 0.5)
        self.assertEqual(getattr(bar, "label_above_text", None), "above")
        self.assertEqual(getattr(bar, "label_below_text", None), "below")

        names = [d.name for d in self.canvas.get_drawables_by_class_name("Bar")]
        self.assertIn("TestBar", names)

    def test_create_bar_swaps_left_and_right(self) -> None:
        bar = self.bar_manager.create_bar(
            name="SwapBar",
            x_left=2.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.assertEqual(bar.x_left, 1.0)
        self.assertEqual(bar.x_right, 2.0)

    def test_create_bar_rejects_zero_width(self) -> None:
        with self.assertRaises(ValueError):
            self.bar_manager.create_bar(
                name="BadBar",
                x_left=1.0,
                x_right=1.0,
                y_bottom=0.0,
                y_top=1.0,
                archive=False,
                redraw=False,
            )

    def test_create_bar_rejects_zero_height(self) -> None:
        with self.assertRaises(ValueError):
            self.bar_manager.create_bar(
                name="BadBar",
                x_left=0.0,
                x_right=1.0,
                y_bottom=2.0,
                y_top=2.0,
                archive=False,
                redraw=False,
            )

    def test_delete_bar_removes_drawable(self) -> None:
        self.bar_manager.create_bar(
            name="DeleteMe",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )

        self.assertTrue(self.bar_manager.delete_bar("DeleteMe", archive=False, redraw=False))
        names = [d.name for d in self.canvas.get_drawables_by_class_name("Bar")]
        self.assertNotIn("DeleteMe", names)


