"""Histogram, box and scatter plots in the browser: rendering helpers, plans, the tool path and undo.

The binning and summary math, the drawables' state and the manager's results are tested
with pytest (``server_tests/test_data_plots.py``).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple

from canvas import Canvas
from constants import successful_call_message
from drawables.box_plot import BoxPlot
from drawables.histogram_plot import HistogramPlot
from drawables.scatter_plot import ScatterPlot
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from rendering import shared_drawable_renderers as shared
from rendering.cached_render_plan import build_plan_for_drawable, has_plan_helper
from scenario_hooks import content_extent
from utils.statistics.data_plots import Bins
from workspace_manager import WorkspaceManager


class _Recorder(shared.RendererPrimitives):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple, dict]] = []

    def fill_polygon(self, points, fill, stroke=None, **kwargs):
        self.calls.append(("fill_polygon", (points, fill, stroke), dict(kwargs)))

    def draw_text(self, text, position, font, color, alignment, style_overrides=None, **kwargs):
        self.calls.append(("draw_text", (text,), dict(kwargs)))

    def stroke_line(self, start, end, stroke, **kwargs):
        self.calls.append(("stroke_line", (start, end, stroke), dict(kwargs)))

    def fill_circle(self, center, radius, fill, stroke=None, **kwargs):
        self.calls.append(("fill_circle", (center, radius, fill), dict(kwargs)))

    def named(self, name: str) -> List[Tuple[str, tuple, dict]]:
        return [call for call in self.calls if call[0] == name]


class _Mapper:
    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return (float(x) * 10.0, -float(y) * 10.0)


STYLE: Dict[str, Any] = {
    "point_radius": 3,
    "point_color": "black",
    "segment_color": "black",
    "segment_stroke_width": 1,
    "segment_width": 1,
    "bar_label_padding_px": 6,
    "label_color": "#000",
    "label_font_family": "Arial",
    "label_font_size": 12,
}


class TestDataPlotRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = _Recorder()

    def test_histogram_draws_outlined_bars_for_non_empty_bins_with_counts(self) -> None:
        plot = HistogramPlot("H", data=[1, 1, 7], bins=Bins(0.0, 2.0, 4))
        shared.render_histogram_plot_helper(self.recorder, plot, _Mapper(), STYLE)
        bars = self.recorder.named("fill_polygon")
        self.assertEqual(len(bars), 2)  # bins [2, 4) and [4, 6) are empty
        points, fill, stroke = bars[0][1]
        self.assertEqual(points[0], (0.0, -0.0))
        self.assertEqual(points[2], (20.0, -20.0))
        self.assertEqual((fill.color, fill.opacity), ("lightblue", 0.6))
        self.assertIsNotNone(stroke)
        self.assertEqual([call[1][0] for call in self.recorder.named("draw_text")], ["2", "1"])

    def test_density_labels_and_unlabelled_many_bins(self) -> None:
        plot = HistogramPlot("H", data=[1, 1, 7], bins=Bins(0.0, 2.0, 4), normalize="density", color="red")
        shared.render_histogram_plot_helper(self.recorder, plot, _Mapper(), STYLE)
        self.assertEqual([call[1][0] for call in self.recorder.named("draw_text")], ["0.333", "0.167"])
        self.assertEqual(self.recorder.named("fill_polygon")[0][1][1].color, "red")
        many = _Recorder()
        values = [float(i) for i in range(40)]
        shared.render_histogram_plot_helper(
            many, HistogramPlot("M", data=values, bins=Bins(0.0, 1.0, 40)), _Mapper(), STYLE
        )
        self.assertEqual(len(many.named("fill_polygon")), 40)
        self.assertEqual(many.named("draw_text"), [])

    def test_box_plot_draws_box_median_whiskers_caps_and_outliers(self) -> None:
        plot = BoxPlot("B", data=[1, 2, 3, 4, 5, 6, 7, 30], y_position=-2.0, box_height=2.0, color="red")
        shared.render_box_plot_helper(self.recorder, plot, _Mapper(), STYLE)
        box = self.recorder.named("fill_polygon")
        self.assertEqual(len(box), 1)
        q1, q3 = plot.summary["q1"], plot.summary["q3"]
        self.assertEqual(box[0][1][0][0], (q1 * 10.0, 30.0))
        self.assertEqual(box[0][1][0][2], (q3 * 10.0, 10.0))
        self.assertEqual(box[0][1][2].color, "red")
        self.assertEqual(len(self.recorder.named("stroke_line")), 5)
        dots = self.recorder.named("fill_circle")
        self.assertEqual([call[1][0] for call in dots], [(300.0, 20.0)])
        self.assertTrue(dots[0][2].get("screen_space"))

    def test_box_plot_without_outliers_draws_no_dots(self) -> None:
        plot = BoxPlot("B", data=[1, 2, 3, 100], y_position=0.0, box_height=1.0, show_outliers=False)
        shared.render_box_plot_helper(self.recorder, plot, _Mapper(), STYLE)
        self.assertEqual(self.recorder.named("fill_circle"), [])
        whisker_ends = [call[1][1][0] for call in self.recorder.named("stroke_line")[1:3]]
        self.assertIn(1000.0, whisker_ends)

    def test_scatter_draws_one_dot_per_pair(self) -> None:
        plot = ScatterPlot("S", x_data=[1, 2, 3], y_data=[4, 5, 6], color="green")
        shared.render_scatter_plot_helper(self.recorder, plot, _Mapper(), STYLE)
        dots = self.recorder.named("fill_circle")
        self.assertEqual([call[1][0] for call in dots], [(10.0, -40.0), (20.0, -50.0), (30.0, -60.0)])
        self.assertEqual({call[1][2].color for call in dots}, {"green"})
        self.assertEqual([float(call[1][1]) for call in dots], [3.0, 3.0, 3.0])

    def test_every_kind_has_a_plan(self) -> None:
        canvas = Canvas(500, 500, draw_enabled=False)
        plots = [
            HistogramPlot("H", data=[1, 2], bins=Bins(0.0, 1.0, 2)),
            BoxPlot("B", data=[1, 2, 3], y_position=0.0, box_height=1.0),
            ScatterPlot("S", x_data=[1], y_data=[2]),
        ]
        for plot in plots:
            with self.subTest(plot=plot.get_class_name()):
                self.assertTrue(has_plan_helper(plot.get_class_name()))
                plan = build_plan_for_drawable(plot, canvas.coordinate_mapper, STYLE)
                self.assertIsNotNone(plan)


class TestDataPlotTools(unittest.TestCase):
    """The tools through the model's tool path on a real canvas."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.functions = FunctionRegistry.get_available_functions(self.canvas, self.workspace_manager)
        self.undoable = FunctionRegistry.get_undoable_functions()

    def run_batch(self, *calls: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch = [{"function_name": name, "arguments": dict(args)} for name, args in calls]
        _, traced = ProcessFunctionCalls.get_results_traced(batch, self.functions, self.undoable, self.canvas)
        return list(traced)

    def plot_names(self) -> List[str]:
        names: List[str] = []
        for class_name in ("HistogramPlot", "BoxPlot", "ScatterPlot"):
            names.extend(str(d.name) for d in self.canvas.drawable_manager.drawables.get_by_class_name(class_name))
        return names

    def test_results_reach_the_model_in_full(self) -> None:
        data = [float(i % 50) for i in range(300)]
        traced = self.run_batch(
            ("plot_histogram", {"data": data, "bin_start": 0, "bin_width": 1, "bin_count": None, "name": None}),
            ("plot_box", {"data": data}),
            ("plot_scatter", {"x_data": data, "y_data": data}),
        )
        for call in traced:
            with self.subTest(tool=call.get("function_name")):
                self.assertFalse(call["is_error"], call["result"])
                self.assertNotEqual(call["result"], successful_call_message)
                self.assertIsInstance(call["result"], dict)
        self.assertEqual(len(traced[0]["result"]["counts"]), 50)
        self.assertEqual(traced[1]["result"]["median"], 24.5)
        self.assertEqual(self.plot_names(), ["histogram", "box_plot", "scatter_plot"])

    def test_one_batch_is_one_undo_step_and_redo_restores_it(self) -> None:
        self.run_batch(("plot_scatter", {"x_data": [1, 2], "y_data": [3, 4]}), ("plot_box", {"data": [1, 2, 3]}))
        self.assertEqual(sorted(self.plot_names()), ["box_plot", "scatter_plot"])
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), 1)
        self.canvas.undo()
        self.assertEqual(self.plot_names(), [])
        self.canvas.redo()
        self.assertEqual(sorted(self.plot_names()), ["box_plot", "scatter_plot"])

    def test_a_lone_plot_call_is_undoable(self) -> None:
        self.run_batch(("plot_histogram", {"data": [1, 2, 3]}))
        self.assertEqual(len(self.canvas.undo_redo_manager.undo_stack), 1)
        self.canvas.undo()
        self.assertEqual(self.plot_names(), [])

    def test_errors_are_flagged_and_add_nothing(self) -> None:
        traced = self.run_batch(("plot_histogram", {"data": [1, 2], "bin_width": 0}))
        self.assertTrue(traced[0]["is_error"])
        self.assertIn("bin_width must be greater than 0", str(traced[0]["result"]))
        self.assertEqual(self.plot_names(), [])

    def test_delete_plot_and_transform_refusal(self) -> None:
        self.run_batch(("plot_scatter", {"x_data": [1, 2], "y_data": [3, 4], "name": "S"}))
        traced = self.run_batch(("translate_object", {"name": "S", "x_offset": 1, "y_offset": 0}))
        self.assertIn("plot_scatter from shifted data", str(traced[0]["result"]))
        self.run_batch(("delete_plot", {"name": "S"}))
        self.assertEqual(self.plot_names(), [])

    def test_content_extent_frames_the_plots(self) -> None:
        manager = self.canvas.drawable_manager.data_plot_manager
        manager.plot_histogram(data=[0, 1, 1, 2], bin_start=0, bin_width=1)
        manager.plot_scatter(x_data=[-5, 3], y_data=[1, 9])
        manager.plot_box(data=[1, 2, 3, 40], y_position=-3, box_height=2)
        self.assertEqual(content_extent(self.canvas), [-5.0, -4.0, 40.0, 9.0])

    def test_saved_workspace_restores_the_plots(self) -> None:
        manager = self.canvas.drawable_manager.data_plot_manager
        manager.plot_histogram(data=[1e-05, 0.5, 2.5e-07], name="H")
        manager.plot_box(data=[3, 1, 2], name="B")
        state = self.workspace_manager._snapshot_persistable_canvas_state()
        self.canvas.clear()
        self.workspace_manager._restore_workspace_state(state)
        self.assertEqual(sorted(self.plot_names()), ["B", "H"])
        restored = self.canvas.drawable_manager.drawables.get_by_class_name("HistogramPlot")[0]
        self.assertEqual(restored.data, (1e-05, 0.5, 2.5e-07))
