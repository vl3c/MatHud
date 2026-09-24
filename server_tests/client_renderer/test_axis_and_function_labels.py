from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple

from rendering.helpers import cartesian_renderer, function_renderer
from rendering.primitives import StrokeStyle


class _TextRecorder:
    """Records draw_text calls; every other primitive is a no-op."""

    def __init__(self) -> None:
        self.texts: List[Tuple[str, Tuple[float, float]]] = []

    def draw_text(self, text: str, position: Tuple[float, float], *_args: Any, **_kwargs: Any) -> None:
        self.texts.append((text, position))

    def stroke_line(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _MapperStub:
    def __init__(self, ox: float, oy: float, scale: float) -> None:
        self.ox = ox
        self.oy = oy
        self.scale_factor = scale

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return self.ox + x * self.scale_factor, self.oy - y * self.scale_factor


WIDTH = 640.0
HEIGHT = 480.0


def _render_grid(ox: float, oy: float, scale: float, tick_spacing: float) -> _TextRecorder:
    recorder = _TextRecorder()
    cartesian = SimpleNamespace(width=WIDTH, height=HEIGHT, current_tick_spacing=tick_spacing)
    cartesian_renderer.render_cartesian_helper(recorder, cartesian, _MapperStub(ox, oy, scale), {})
    return recorder


class TestTickLabelFormatting(unittest.TestCase):
    def test_adjacent_ticks_are_distinct_at_deep_zoom_away_from_origin(self) -> None:
        spacing = 2e-5
        precision = cartesian_renderer._calculate_tick_precision(spacing)
        labels = [cartesian_renderer._format_tick_value(1 + k * spacing, precision) for k in range(-5, 6)]
        self.assertEqual(len(set(labels)), len(labels), labels)

    def test_adjacent_ticks_are_distinct_for_large_values(self) -> None:
        precision = cartesian_renderer._calculate_tick_precision(1.0)
        labels = [cartesian_renderer._format_tick_value(1_000_000 + k, precision) for k in range(5)]
        self.assertEqual(len(set(labels)), len(labels), labels)

    def test_existing_formats_are_kept(self) -> None:
        fmt = cartesian_renderer._format_tick_value
        self.assertEqual(fmt(2.0, 0), "2")
        self.assertEqual(fmt(0.25, 2), "0.25")
        self.assertEqual(fmt(0.0001, 4), "1.0e-4")
        self.assertEqual(fmt(2_000_000, 0), "2.0e+6")

    def test_rendered_x_labels_are_distinct_at_deep_zoom(self) -> None:
        spacing = 2e-5
        scale = 100 / spacing
        recorder = _render_grid(WIDTH / 2 - 1.0 * scale, HEIGHT / 2, scale, spacing)
        x_labels = [text for text, pos in recorder.texts if abs(pos[1] - HEIGHT / 2) < 20]
        self.assertGreater(len(x_labels), 3)
        self.assertEqual(len(set(x_labels)), len(x_labels), x_labels)


class TestAxisLabelClamping(unittest.TestCase):
    def _x_axis_labels(self, oy: float) -> List[Tuple[str, Tuple[float, float]]]:
        recorder = _TextRecorder()
        styles = cartesian_renderer._get_cartesian_styles({})
        cartesian_renderer._draw_cartesian_ticks_x(
            recorder,
            WIDTH / 2,
            oy,
            WIDTH,
            50.0,
            50.0,
            styles["tick_size"],
            styles["mid_tick_size"],
            styles["tick_font_float"],
            styles["font"],
            styles["label_color"],
            styles["label_alignment"],
            styles["tick_stroke"],
            HEIGHT,
        )
        return recorder.texts

    def _y_axis_labels(self, ox: float) -> List[Tuple[str, Tuple[float, float]]]:
        recorder = _TextRecorder()
        styles = cartesian_renderer._get_cartesian_styles({})
        cartesian_renderer._draw_cartesian_ticks_y(
            recorder,
            ox,
            HEIGHT / 2,
            HEIGHT,
            50.0,
            50.0,
            styles["tick_size"],
            styles["mid_tick_size"],
            styles["font"],
            styles["label_color"],
            styles["label_alignment"],
            styles["tick_stroke"],
            WIDTH,
            styles["tick_font_float"],
        )
        return recorder.texts

    def test_x_axis_labels_stay_visible_when_axis_is_off_screen(self) -> None:
        for oy in (-300.0, HEIGHT + 300.0):
            labels = self._x_axis_labels(oy)
            self.assertTrue(labels)
            for text, (_x, y) in labels:
                self.assertGreaterEqual(y, 0, text)
                self.assertLessEqual(y, HEIGHT, text)

    def test_y_axis_labels_stay_visible_when_axis_is_off_screen(self) -> None:
        for ox in (-300.0, WIDTH + 300.0):
            labels = self._y_axis_labels(ox)
            self.assertTrue(labels)
            for text, (x, _y) in labels:
                self.assertGreaterEqual(x, 0, text)
                self.assertLessEqual(x, WIDTH - len(text) * 8 * 0.6, text)

    def test_on_screen_axis_labels_are_unchanged(self) -> None:
        oy = HEIGHT / 2
        styles = cartesian_renderer._get_cartesian_styles({})
        expected_y = oy + styles["tick_size"] + styles["tick_font_float"]
        for _text, (_x, y) in self._x_axis_labels(oy):
            self.assertEqual(y, expected_y)
        ox = WIDTH / 2
        for _text, (x, _y) in self._y_axis_labels(ox):
            self.assertEqual(x, ox + styles["tick_size"] + 2)


class TestFunctionLabelPlacement(unittest.TestCase):
    def _label_x(self, first_x: float, name: str = "f") -> float:
        recorder = _TextRecorder()
        func = SimpleNamespace(name=name)
        paths = [[(first_x, 100.0), (first_x + 10, 120.0)]]
        stroke = StrokeStyle(color="#000", width=1)
        function_renderer._render_function_label(recorder, func, paths, stroke, {}, WIDTH)
        self.assertEqual(len(recorder.texts), 1)
        return recorder.texts[0][1][0]

    def test_label_is_not_placed_left_of_the_canvas(self) -> None:
        self.assertGreaterEqual(self._label_x(0.0, "long_name"), 4)

    def test_label_is_not_placed_right_of_the_canvas(self) -> None:
        name = "g"
        self.assertLessEqual(self._label_x(WIDTH + 50, name), WIDTH - len(name) * 12 * 0.6 + 1e-9)


if __name__ == "__main__":
    unittest.main()
