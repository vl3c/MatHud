from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple

from rendering import cached_render_plan
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


def _map_state(offset_x: float = 0.0, offset_y: float = 0.0) -> dict:
    return {"scale": 1.0, "offset_x": offset_x, "offset_y": offset_y, "origin_x": 0.0, "origin_y": 0.0}


def _build_function_plan(paths: List[List[Tuple[float, float]]], name: str = "f") -> Any:
    """Record a function plan the way render_function_helper does, at map offset (0, 0)."""
    recorder = cached_render_plan._RecordingPrimitives(name)
    func = SimpleNamespace(name=name)
    stroke = StrokeStyle(color="#000", width=1)
    function_renderer._render_function_paths(recorder, paths, stroke, WIDTH, HEIGHT, WIDTH / 2)
    function_renderer._render_function_label(recorder, func, paths, stroke, {}, WIDTH, HEIGHT)
    mapper = SimpleNamespace(
        scale_factor=1.0, offset=SimpleNamespace(x=0.0, y=0.0), origin=SimpleNamespace(x=0.0, y=0.0)
    )
    return cached_render_plan._finish_plan(func, recorder, mapper, name, "Function")


def _label_positions(plan: Any) -> List[Tuple[float, float]]:
    return [command.args[1] for command in plan.commands if command.op == "draw_text"]


class TestFunctionLabelReprojection(unittest.TestCase):
    """Pans within a function plan's bucket reproject it; the label must stay anchored to the viewport."""

    def _line(self, start_x: float, end_x: float, y: float = 100.0) -> List[List[Tuple[float, float]]]:
        return [[(float(x), y) for x in range(int(start_x), int(end_x) + 1, 10)]]

    def test_label_stays_at_left_edge_when_panning_right(self) -> None:
        plan = _build_function_plan(self._line(-320, 960))
        built = _label_positions(plan)[0]

        plan.update_map_state(_map_state(offset_x=200.0))

        self.assertEqual(_label_positions(plan), [built])

    def test_label_stays_on_screen_when_panning_left(self) -> None:
        plan = _build_function_plan(self._line(-320, 960))
        built = _label_positions(plan)[0]

        plan.update_map_state(_map_state(offset_x=-200.0))

        self.assertEqual(_label_positions(plan), [built])

    def test_label_follows_a_curve_that_starts_on_screen(self) -> None:
        plan = _build_function_plan(self._line(300, 960))
        plan.update_map_state(_map_state(offset_x=-200.0))

        expected = _build_function_plan(self._line(100, 760))
        self.assertEqual(_label_positions(plan), _label_positions(expected))

    def test_label_matches_a_fresh_build_after_a_vertical_pan(self) -> None:
        plan = _build_function_plan(self._line(-320, 960, y=100.0))
        plan.update_map_state(_map_state(offset_y=150.0))

        expected = _build_function_plan(self._line(-320, 960, y=250.0))
        self.assertEqual(_label_positions(plan), _label_positions(expected))

    def test_plan_bounds_include_the_reanchored_label(self) -> None:
        # Built with the curve starting at the left edge, so the label is clamped to x=4.
        plan = _build_function_plan(self._line(0, 960, y=100.0))
        plan.update_map_state(_map_state(offset_x=200.0))

        label_x, _label_y = _label_positions(plan)[0]
        self.assertEqual(label_x, 200.0 - 12.0)
        min_x, max_x, min_y, max_y = plan.metadata["screen_bounds"]
        self.assertEqual((min_x, max_x, min_y, max_y), (label_x, 1160.0, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
