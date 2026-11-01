"""Histograms, box plots and scatter plots: binning, summaries, drawables, the manager and the canvas text.

The math (``utils/statistics/data_plots.py``), the drawables and ``DataPlotManager`` are pure
Python, so they run under CPython with a stub canvas; rendering is tested in the browser
(``client_tests/test_data_plots.py``).
"""

from __future__ import annotations

import copy
import json
import math
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from drawables.box_plot import BoxPlot
from drawables.histogram_plot import HistogramPlot
from drawables.scatter_plot import ScatterPlot
from managers.data_plot_manager import MAX_LISTED_OUTLIERS, DataPlotManager
from managers.drawables_container import DrawablesContainer
from managers.statistics_manager import StatisticsManager
from plot_kinds import build_saved_plot, plot_kind
from static.canvas_state_formatter import render_state
from utils.statistics.data_plots import (
    MAX_HISTOGRAM_BINS,
    MAX_PLOT_DATA_POINTS,
    Bins,
    bin_counts,
    box_summary,
    nice_width,
    resolve_bins,
    resolve_normalize,
    scatter_summary,
    sturges_bin_count,
    validate_plot_data,
    validate_scatter_data,
)
from utils.statistics.descriptive import compute_descriptive_statistics
from utils.statistics.two_variable import compute_two_variable_statistics


def _bins(data: List[float], **kwargs: Any) -> Bins:
    return resolve_bins([float(v) for v in data], **kwargs)


class TestResolveBins(unittest.TestCase):
    def test_default_bins_use_sturges_and_a_round_width(self) -> None:
        data = [12, 15, 17, 21, 22, 22, 25, 28, 30, 31, 35, 40, 41, 45, 60, 99]
        self.assertEqual(sturges_bin_count(16), 5)
        # span 87 / 5 bins = 17.4, rounded up to 20; start 0 is a multiple of 20 below 12
        self.assertEqual(_bins(data), Bins(0.0, 20.0, 5))

    def test_nice_widths(self) -> None:
        cases = [(17.4, 20.0), (0.3, 0.5), (2.2, 2.5), (1.0, 1.0), (0.050000000000000044, 0.05), (7.0, 10.0)]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(nice_width(raw), expected)

    def test_width_only_starts_on_a_multiple_of_the_width(self) -> None:
        self.assertEqual(_bins([0.3, 0.5, 0.9], bin_width=0.1), Bins(0.3, 0.1, 7))
        self.assertEqual(_bins([0.6, 0.9], bin_width=0.2), Bins(0.6, 0.2, 2))
        self.assertEqual(_bins([0.7, 0.9], bin_width=0.1), Bins(0.7, 0.1, 3))
        self.assertEqual(_bins([3, 8, 15], bin_width=5), Bins(0.0, 5.0, 4))
        self.assertEqual(_bins([-7, 2], bin_width=5), Bins(-10.0, 5.0, 3))

    def test_start_and_width_cover_the_largest_value(self) -> None:
        # (0.9 - 0.3) / 0.2 is 3.0000000000000004 in floats: 0.9 is on an edge and starts bin 4
        self.assertEqual(_bins([0.3, 0.9], bin_start=0.3, bin_width=0.2), Bins(0.3, 0.2, 4))
        self.assertEqual(_bins([0.3, 0.8], bin_start=0.3, bin_width=0.2), Bins(0.3, 0.2, 3))

    def test_integer_data_with_width_one_gets_a_bin_per_value(self) -> None:
        bins = _bins([1, 2, 2, 3, 4, 5, 6, 6], bin_start=1, bin_width=1)
        self.assertEqual(bins, Bins(1.0, 1.0, 6))
        self.assertEqual(bin_counts([1, 2, 2, 3, 4, 5, 6, 6], bins), ([1, 2, 1, 1, 1, 2], 0))

    def test_count_only_spans_the_data(self) -> None:
        self.assertEqual(_bins([1, 2, 3], bin_count=5), Bins(1.0, 0.4, 5))
        self.assertEqual(_bins([1, 2, 3], bin_count=5.0), Bins(1.0, 0.4, 5))

    def test_fixed_bins(self) -> None:
        self.assertEqual(_bins([1, 50], bin_width=10, bin_count=3), Bins(0.0, 10.0, 3))
        self.assertEqual(_bins([1, 50], bin_start=5, bin_width=10, bin_count=3), Bins(5.0, 10.0, 3))

    def test_a_zero_range_gives_width_one(self) -> None:
        self.assertEqual(_bins([3, 3, 3]), Bins(3.0, 1.0, 1))
        self.assertEqual(_bins([3, 3], bin_count=2), Bins(2.0, 1.0, 2))
        self.assertEqual(_bins([5, 9], bin_start=9), Bins(9.0, 1.0, 1))
        self.assertEqual(_bins([5, 9], bin_start=9, bin_count=3), Bins(9.0, 1.0, 3))

    def test_refusals(self) -> None:
        cases = [
            ({"bin_width": 0}, "bin_width must be greater than 0"),
            ({"bin_width": -1}, "bin_width must be greater than 0"),
            ({"bin_count": 0}, "bin_count must be a whole number"),
            ({"bin_count": 2.5}, "bin_count must be a whole number"),
            ({"bin_count": 201}, "bin_count must be a whole number"),
            ({"bin_count": True}, "bin_count must be a whole number"),
            ({"bin_start": 100}, "bin_start 100.0 is above every value"),
            ({"bin_width": float("nan")}, "bin_width must be a finite number"),
            ({"bin_width": 0.001}, "a histogram has at most 200 bins"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    _bins([1, 10], **kwargs)

    def test_overflowing_ranges_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "too large"):
            _bins([-1e308, 1e308])
        with self.assertRaisesRegex(ValueError, "more than 10\\^15"):
            _bins([0, 1e300], bin_width=1e-300)

    def test_the_bin_cap_names_a_width_that_fits(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 10000"):
            _bins([0, 1e6], bin_width=1)
        self.assertEqual(_bins([0, 1e6], bin_width=10000).count, 101)
        self.assertEqual(_bins([0, 999999], bin_width=5000).count, MAX_HISTOGRAM_BINS)


class TestBinCounts(unittest.TestCase):
    def test_bins_are_half_open_except_the_last(self) -> None:
        bins = Bins(0.0, 10.0, 3)
        self.assertEqual(bin_counts([0, 9.99, 10, 20, 29, 30], bins), ([2, 1, 3], 0))

    def test_decimal_values_on_edges_land_in_the_bin_they_start(self) -> None:
        bins = _bins([0.1, 0.9], bin_start=0.0, bin_width=0.1)
        counts, excluded = bin_counts([0.1, 0.2, 0.3, 0.6, 0.7, 0.9], bins)
        self.assertEqual(excluded, 0)
        self.assertEqual(counts, [0, 1, 1, 1, 0, 0, 1, 1, 0, 1])

    def test_values_outside_fixed_bins_are_excluded(self) -> None:
        bins = Bins(10.0, 5.0, 2)
        self.assertEqual(bin_counts([3, 10, 14, 15, 20, 21], bins), ([2, 2], 2))

    def test_edges_are_rounded_for_display(self) -> None:
        bins = Bins(0.0, 0.1, 3)
        self.assertEqual([bins.edge(i) for i in range(4)], [0.0, 0.1, 0.2, 0.3])


class TestNormalize(unittest.TestCase):
    def test_modes(self) -> None:
        self.assertEqual(resolve_normalize(None), "count")
        self.assertEqual(resolve_normalize(" Density "), "density")
        with self.assertRaisesRegex(ValueError, "normalize must be one of count, density"):
            resolve_normalize("relative")

    def test_density_bars_have_unit_area(self) -> None:
        plot = HistogramPlot("H", data=[1, 2, 2, 3, 7], bins=Bins(0.0, 2.0, 4), normalize="density")
        area = sum(height * 2.0 for height in plot.summary["heights"])
        self.assertAlmostEqual(area, 1.0)


class TestBoxSummary(unittest.TestCase):
    def test_quartiles_match_the_statistics_tool(self) -> None:
        for data in ([5.0], [1.0, 9.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [7.0, 1.0, 4.0, 9.0, 2.0, 8.0, 3.0]):
            with self.subTest(data=data):
                summary = box_summary(data)
                stats = compute_descriptive_statistics(data)
                for key in ("min", "q1", "median", "q3", "max", "iqr"):
                    self.assertEqual(summary[key], stats[key])  # type: ignore[literal-required]

    def test_outliers_and_whiskers(self) -> None:
        summary = box_summary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 30.0, -20.0])
        self.assertEqual(summary["outliers"], [-20.0, 30.0])
        self.assertEqual((summary["whisker_low"], summary["whisker_high"]), (1.0, 7.0))
        self.assertEqual(summary["lower_fence"], summary["q1"] - 1.5 * summary["iqr"])

    def test_values_on_a_fence_are_inside(self) -> None:
        # The fences are 0.8000000000000002 and 3.5999999999999996 in floats
        self.assertEqual(box_summary([0.8, 1.1, 1.1, 1.2, 1.3, 1.3, 1.4])["outliers"], [])
        self.assertEqual(box_summary([2.2, 2.6, 2.6, 2.8, 3.0, 3.0, 3.6])["outliers"], [])
        self.assertEqual(box_summary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.5])["outliers"], [])

    def test_without_outliers_the_whiskers_reach_the_extremes(self) -> None:
        summary = box_summary([1.0, 2.0, 3.0, 4.0, 100.0], show_outliers=False)
        self.assertEqual(summary["outliers"], [])
        self.assertIsNone(summary["lower_fence"])
        self.assertEqual((summary["whisker_low"], summary["whisker_high"]), (1.0, 100.0))

    def test_one_value(self) -> None:
        summary = box_summary([4.0])
        self.assertEqual((summary["q1"], summary["median"], summary["q3"]), (4.0, 4.0, 4.0))
        self.assertEqual((summary["whisker_low"], summary["whisker_high"]), (4.0, 4.0))


class TestScatterSummary(unittest.TestCase):
    def test_r_matches_the_statistics_tool(self) -> None:
        xs, ys = [1.0, 2.0, 3.0, 4.0], [52.0, 60.0, 71.0, 80.0]
        self.assertEqual(scatter_summary(xs, ys)["r"], compute_two_variable_statistics(xs, ys)["r"])

    def test_r_is_none_for_one_pair_or_a_constant_list(self) -> None:
        self.assertIsNone(scatter_summary([1.0], [2.0])["r"])
        self.assertIsNone(scatter_summary([1.0, 2.0], [3.0, 3.0])["r"])

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_scatter_data([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "x_data must not be empty"):
            validate_scatter_data([], [])


class TestValidation(unittest.TestCase):
    def test_data_rules(self) -> None:
        self.assertEqual(validate_plot_data([1, 2.5]), [1.0, 2.5])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_plot_data([])
        with self.assertRaisesRegex(TypeError, "is a bool"):
            validate_plot_data([1, True])
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_plot_data([1, math.inf])
        with self.assertRaisesRegex(ValueError, "at most 1000"):
            validate_plot_data([1.0] * (MAX_PLOT_DATA_POINTS + 1))


class TestDataPlotDrawables(unittest.TestCase):
    def plots(self) -> List[Any]:
        return [
            HistogramPlot("H", data=[0.1, 0.3, 0.3, 0.9], bins=Bins(0.0, 0.2, 5), color="red", fill_opacity=0.4),
            BoxPlot("B", data=[1, 2, 3, 4, 30], y_position=-1.5, box_height=2.0),
            ScatterPlot("S", x_data=[1, 2, 3], y_data=[2, 4, 7]),
        ]

    def test_saved_state_rebuilds_the_same_plot(self) -> None:
        for plot in self.plots():
            with self.subTest(plot=plot.name):
                state = json.loads(json.dumps(plot.get_state()))
                kind = plot_kind(plot.get_class_name())
                assert kind is not None
                rebuilt, _ = build_saved_plot(kind, state)
                self.assertEqual(rebuilt.get_state(), plot.get_state())

    def test_restore_recomputes_the_summary(self) -> None:
        state = self.plots()[1].get_state()
        state["args"]["summary"]["median"] = 999
        rebuilt, _ = build_saved_plot(plot_kind("BoxPlot"), state)  # type: ignore[arg-type]
        self.assertEqual(rebuilt.summary["median"], 3.0)

    def test_restore_refuses_bad_saved_data(self) -> None:
        cases = [
            ("HistogramPlot", {"data": [], "bin_start": 0, "bin_width": 1, "bin_count": 1}),
            ("BoxPlot", {"data": [1, 2], "y_position": 0, "box_height": 0}),
            ("BoxPlot", {"data": [1, 2], "y_position": float("nan"), "box_height": 1}),
            ("ScatterPlot", {"x_data": [1, 2], "y_data": [1]}),
        ]
        for class_name, args in cases:
            with self.subTest(case=class_name):
                with self.assertRaises((TypeError, ValueError)):
                    build_saved_plot(plot_kind(class_name), {"name": "N", "args": args})  # type: ignore[arg-type]

    def test_copies_share_data_and_summary_and_keep_their_state(self) -> None:
        for plot in self.plots():
            with self.subTest(plot=plot.name):
                copied = copy.deepcopy(plot)
                self.assertIsNot(copied, plot)
                self.assertIs(copied.summary, plot.summary)
                self.assertEqual(copied.get_state(), plot.get_state())
                self.assertEqual(copied.render_signature(), plot.render_signature())

    def test_render_signature_follows_the_name(self) -> None:
        plot = self.plots()[2]
        before = plot.render_signature()
        plot.name = "T"
        self.assertNotEqual(plot.render_signature(), before)
        self.assertEqual(plot.render_signature()[0], "T")

    def test_plots_are_renderable_and_keep_no_distribution_fields(self) -> None:
        for plot in self.plots():
            with self.subTest(plot=plot.name):
                self.assertTrue(plot.is_renderable)
                self.assertNotIn("distribution_type", plot.get_state()["args"])
                self.assertNotIn("bounds", plot.get_state()["args"])

    def test_extents(self) -> None:
        histogram, box, scatter = self.plots()
        self.assertEqual(histogram.extent_points(), [(0.0, 0.0), (1.0, 2.0)])
        self.assertEqual(box.extent_points(), [(1.0, -2.5), (30.0, -0.5)])
        self.assertEqual(scatter.extent_points(), [(1.0, 2.0), (3.0, 7.0)])


class _UndoRecorder:
    def __init__(self) -> None:
        self.archives = 0

    def archive(self) -> None:
        self.archives += 1


class _NameGenerator:
    def filter_string(self, value: str) -> str:
        return "".join(ch for ch in value if ch.isalnum() or ch == "_")


class TestDataPlotManager(unittest.TestCase):
    def setUp(self) -> None:
        self.drawables = DrawablesContainer()
        self.undo = _UndoRecorder()
        self.canvas = SimpleNamespace(draw_enabled=False, undo_redo_manager=self.undo)
        self.manager = DataPlotManager(self.canvas, self.drawables, _NameGenerator())  # type: ignore[arg-type]

    def test_histogram_result(self) -> None:
        result = self.manager.plot_histogram(data=[55, 62, 70, 71, 78, 85, 90], bin_start=50, bin_width=10)
        self.assertEqual(result["plot_name"], "histogram")
        self.assertEqual(result["edges"], [50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        self.assertEqual(result["counts"], [1, 1, 3, 1, 1])
        self.assertEqual((result["n"], result["bin_count"], result["normalize"]), (7, 5, "count"))
        self.assertNotIn("heights", result)
        self.assertNotIn("excluded_count", result)
        self.assertEqual(self.undo.archives, 1)
        self.assertEqual([d.name for d in self.drawables.get_by_class_name("HistogramPlot")], ["histogram"])

    def test_histogram_reports_excluded_values_and_density_heights(self) -> None:
        result = self.manager.plot_histogram(
            data=[1, 5, 6, 12], bin_start=5, bin_width=5, bin_count=1, normalize="density"
        )
        self.assertEqual(result["counts"], [2])
        self.assertEqual(result["excluded_count"], 2)
        self.assertIn("2 of the 4 values lie outside the bins", result["note"])
        self.assertEqual(result["heights"], [0.1])

    def test_errors_add_nothing(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.plot_histogram(data=[1, 2], bin_width=-1)
        with self.assertRaises(ValueError):
            self.manager.plot_box(data=[1, 2], fill_opacity=2)
        self.assertEqual(self.drawables.get_all(), [])
        self.assertEqual(self.undo.archives, 0)

    def test_box_result_and_default_size(self) -> None:
        result = self.manager.plot_box(data=[3, 7, 8, 5, 12, 14, 21, 13, 18, 60])
        for key, value in (("min", 3.0), ("q1", 7.0), ("median", 12.5), ("q3", 18.0), ("max", 60.0)):
            self.assertEqual(result[key], value)
        self.assertEqual((result["outlier_count"], result["outliers"]), (1, [60.0]))
        self.assertEqual(result["whisker_high"], 21.0)
        # 0.2 of the whisker span 3..21, centred below the axis with a gap of a quarter height
        self.assertAlmostEqual(result["box_height"], 3.6)
        self.assertAlmostEqual(result["y_position"], -0.25 * 3.6 - 1.8)

    def test_box_plots_stack_below_each_other(self) -> None:
        first = self.manager.plot_box(data=[1, 2, 3, 4, 5], box_height=1)
        second = self.manager.plot_box(data=[2, 3, 4], box_height=2)
        self.assertEqual(second["plot_name"], "box_plot_1")
        first_bottom = first["y_position"] - 0.5
        self.assertAlmostEqual(second["y_position"] + 1.0, first_bottom - 0.5)

    def test_box_without_outliers_and_given_position(self) -> None:
        result = self.manager.plot_box(data=[1, 2, 3, 100], show_outliers=False, y_position=4, box_height=1)
        self.assertEqual(result["y_position"], 4.0)
        self.assertNotIn("outliers", result)
        self.assertNotIn("lower_fence", result)
        self.assertEqual(result["whisker_high"], 100.0)

    def test_box_lists_only_the_first_outliers(self) -> None:
        data = [0.0] * 100 + [float(1000 + i) for i in range(30)]
        result = self.manager.plot_box(data=data)
        self.assertEqual(result["outlier_count"], 30)
        self.assertEqual(len(result["outliers"]), MAX_LISTED_OUTLIERS)
        self.assertIn("20 smallest of 30", result["note"])

    def test_scatter_result(self) -> None:
        result = self.manager.plot_scatter(x_data=[1, 2, 3, 4], y_data=[52, 60, 71, 80], name="study")
        self.assertEqual(result["plot_name"], "study")
        self.assertEqual((result["n"], result["x_min"], result["y_max"]), (4, 1.0, 80.0))
        self.assertGreater(result["r"], 0.99)
        single = self.manager.plot_scatter(x_data=[1], y_data=[2])
        self.assertIsNone(single["r"])
        self.assertIn("single pair", single["note"])

    def test_results_are_json(self) -> None:
        results = [
            self.manager.plot_histogram(data=[1.5, 2.5, 2.5]),
            self.manager.plot_box(data=[1.5, 2.5, 2.5]),
            self.manager.plot_scatter(x_data=[1.5, 2.5], y_data=[2.5, 1.0]),
        ]
        for result in results:
            self.assertEqual(json.loads(json.dumps(result)), result)

    def test_delete_plot_removes_a_data_plot(self) -> None:
        self.manager.plot_scatter(x_data=[1, 2], y_data=[3, 4], name="S")
        stats = StatisticsManager(
            self.canvas,
            self.drawables,
            _NameGenerator(),
            SimpleNamespace(remove_drawable=lambda d: None),
            None,
            None,  # type: ignore[arg-type]
        )
        self.assertTrue(stats.delete_plot("S"))
        self.assertEqual(self.drawables.get_all(), [])


class TestDataPlotCanvasText(unittest.TestCase):
    def state(self, *plots: Any) -> Dict[str, Any]:
        buckets: Dict[str, Any] = {}
        for plot in plots:
            buckets.setdefault(plot.get_class_name() + "s", []).append(plot.get_state())
        return buckets

    def test_text_lines(self) -> None:
        data = [12, 15, 17, 21, 22, 22, 25, 28, 30, 31, 35, 40, 41, 45, 60, 99]
        histogram = HistogramPlot("H", data=data, bins=resolve_bins([float(v) for v in data]))
        box = BoxPlot("B", data=data, y_position=-2, box_height=3, color="red")
        scatter = ScatterPlot("S", x_data=[1, 2, 3], y_data=[2, 4, 7])
        text = render_state(self.state(histogram, box, scatter), "text", None)
        self.assertIn(
            "H = Histogram(n 16; 5 bins of width 20 from 0; counts 3/8/3/1/1; data 12, 15, 17, 21, 22, 22, 25,", text
        )
        self.assertIn(
            "B = BoxPlot(n 16; min 12, Q1 21.5, median 29, Q3 40.5, max 99; whiskers 12..60; outliers 99; "
            "at y -2, height 3; data 12,",
            text,
        )
        self.assertIn(") color red", text)
        self.assertIn("S = ScatterPlot(n 3; x 1..3, y 2..7; r 0.993399; points (1, 2), (2, 4), (3, 7))", text)
        self.assertNotIn("summary", text)

    def test_long_data_is_summarized(self) -> None:
        values = [float(i % 17) for i in range(500)]
        histogram = HistogramPlot("H", data=values, bins=resolve_bins(values), normalize="density")
        scatter = ScatterPlot("S", x_data=values, y_data=values)
        text = render_state(self.state(histogram, scatter), "text", None)
        self.assertIn("density heights; 500 values, not listed)", text)
        self.assertIn("r 1; 500 pairs, not listed)", text)
        self.assertLess(len(text), 600)

    def test_min_json_shortens_long_data(self) -> None:
        values = [float(i) for i in range(1000)]
        state = self.state(ScatterPlot("S", x_data=values, y_data=values))
        state["Points"] = [{"name": "A", "args": {"position": {"x": 1, "y": 2}}}]
        rendered = json.loads(render_state(state, "min_json", 1500))
        self.assertEqual(rendered["ScatterPlots"][0]["x_data"], "1000 values, not listed")
        self.assertEqual([point["name"] for point in rendered["Points"]], ["A"])

    def test_many_bins_and_outliers_are_capped(self) -> None:
        values = [float(i) for i in range(200)]
        histogram = HistogramPlot("H", data=values, bins=resolve_bins(values, bin_width=1))
        box = BoxPlot("B", data=[0.0] * 50 + [float(100 + i) for i in range(15)], y_position=0, box_height=1)
        text = render_state(self.state(histogram, box), "text", None)
        self.assertIn("200 bins of width 1 from 0", text)
        self.assertIn("... (150 more)", text)
        self.assertIn("... (5 more)", text)


if __name__ == "__main__":
    unittest.main()
