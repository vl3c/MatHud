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

    def test_create_bar_updates_existing_bar(self) -> None:
        bar1 = self.bar_manager.create_bar(
            name="UpdateMe",
            x_left=0.0,
            x_right=2.0,
            y_bottom=0.0,
            y_top=3.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.25,
            label_above_text="a",
            label_below_text="b",
            archive=False,
            redraw=False,
        )

        bar2 = self.bar_manager.create_bar(
            name="UpdateMe",
            x_left=10.0,
            x_right=12.0,
            y_bottom=-2.0,
            y_top=-1.0,
            stroke_color="#abc",
            fill_color="#def",
            fill_opacity=0.75,
            label_above_text="new_a",
            label_below_text="new_b",
            archive=False,
            redraw=False,
        )

        self.assertIs(bar2, bar1)
        self.assertEqual(bar2.name, "UpdateMe")
        self.assertEqual(bar2.x_left, 10.0)
        self.assertEqual(bar2.x_right, 12.0)
        self.assertEqual(bar2.y_bottom, -2.0)
        self.assertEqual(bar2.y_top, -1.0)
        self.assertEqual(bar2.color, "#abc")
        self.assertEqual(bar2.fill_color, "#def")
        self.assertEqual(bar2.fill_opacity, 0.75)
        self.assertEqual(getattr(bar2, "label_above_text", None), "new_a")
        self.assertEqual(getattr(bar2, "label_below_text", None), "new_b")

        self.assertEqual(self.canvas.get_drawables_by_class_name("Bar"), [bar1])

    def test_create_bar_generates_default_name_for_empty_input(self) -> None:
        bar1 = self.bar_manager.create_bar(
            name="   ",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.assertEqual(bar1.name, "bar")

        bar2 = self.bar_manager.create_bar(
            name="",
            x_left=2.0,
            x_right=3.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.assertEqual(bar2.name, "bar_1")

    def test_create_bar_rejects_non_finite_inputs(self) -> None:
        bad_values = [float("nan"), float("inf"), float("-inf")]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.bar_manager.create_bar(
                        name="Bad",
                        x_left=value,
                        x_right=1.0,
                        y_bottom=0.0,
                        y_top=1.0,
                        archive=False,
                        redraw=False,
                    )

                with self.assertRaises(ValueError):
                    self.bar_manager.create_bar(
                        name="Bad",
                        x_left=0.0,
                        x_right=value,
                        y_bottom=0.0,
                        y_top=1.0,
                        archive=False,
                        redraw=False,
                    )

                with self.assertRaises(ValueError):
                    self.bar_manager.create_bar(
                        name="Bad",
                        x_left=0.0,
                        x_right=1.0,
                        y_bottom=value,
                        y_top=1.0,
                        archive=False,
                        redraw=False,
                    )

                with self.assertRaises(ValueError):
                    self.bar_manager.create_bar(
                        name="Bad",
                        x_left=0.0,
                        x_right=1.0,
                        y_bottom=0.0,
                        y_top=value,
                        archive=False,
                        redraw=False,
                    )

    def test_create_bar_archive_flag_controls_canvas_archive(self) -> None:
        calls = []

        def fake_archive() -> None:
            calls.append("archive")

        self.canvas.archive = fake_archive  # type: ignore[method-assign]

        self.bar_manager.create_bar(
            name="Archived",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=True,
            redraw=False,
        )
        self.assertEqual(calls, ["archive"])

        calls.clear()
        self.bar_manager.create_bar(
            name="NotArchived",
            x_left=2.0,
            x_right=3.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.assertEqual(calls, [])

    def test_delete_bar_archive_flag_controls_canvas_archive(self) -> None:
        self.bar_manager.create_bar(
            name="ToDelete",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )

        calls = []

        def fake_archive() -> None:
            calls.append("archive")

        self.canvas.archive = fake_archive  # type: ignore[method-assign]

        self.assertTrue(self.bar_manager.delete_bar("ToDelete", archive=True, redraw=False))
        self.assertEqual(calls, ["archive"])

        self.bar_manager.create_bar(
            name="ToDelete2",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )

        calls.clear()
        self.assertTrue(self.bar_manager.delete_bar("ToDelete2", archive=False, redraw=False))
        self.assertEqual(calls, [])
