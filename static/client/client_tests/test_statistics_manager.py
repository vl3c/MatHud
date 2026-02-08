from __future__ import annotations

import unittest
from typing import List

from constants import default_area_fill_color, default_area_opacity
from drawables.bars_plot import BarsPlot
from drawables.discrete_plot import DiscretePlot
from canvas import Canvas


class _LoggerSpy:
    def __init__(self) -> None:
        self.messages: List[str] = []

    def debug(self, message: str) -> None:
        self.messages.append(str(message))


class TestStatisticsManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _names_for_class(self, class_name: str) -> List[str]:
        return [d.name for d in self.canvas.get_drawables_by_class_name(class_name)]

    def test_plot_distribution_continuous_creates_components(self) -> None:
        result = self.canvas.plot_distribution(
            name="MyPlot",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds=None,
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )

        self.assertEqual(result["plot_name"], "MyPlot")
        self.assertEqual(result["representation"], "continuous")
        self.assertEqual(result["distribution_type"], "normal")
        self.assertIn("function_name", result)
        self.assertIn("fill_area_name", result)

        plot_names = self._names_for_class("ContinuousPlot")
        self.assertIn("MyPlot", plot_names)

        function_names = self._names_for_class("Function")
        self.assertIn(result["function_name"], function_names)

        area_names = self._names_for_class("FunctionsBoundedColoredArea")
        self.assertIn(result["fill_area_name"], area_names)

    def test_delete_plot_continuous_removes_components(self) -> None:
        result = self.canvas.plot_distribution(
            name="MyPlot",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds=None,
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )

        self.assertTrue(self.canvas.delete_plot("MyPlot"))

        self.assertNotIn("MyPlot", self._names_for_class("ContinuousPlot"))
        self.assertNotIn(result["function_name"], self._names_for_class("Function"))
        self.assertNotIn(result["fill_area_name"], self._names_for_class("FunctionsBoundedColoredArea"))

    def test_plot_distribution_rejects_invalid_sigma(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadSigmaPlot",
                representation="continuous",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 0.0},
                plot_bounds=None,
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

    def test_plot_distribution_generates_unique_plot_names(self) -> None:
        r1 = self.canvas.plot_distribution(
            name="MyPlot",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds=None,
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )
        r2 = self.canvas.plot_distribution(
            name="MyPlot",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds=None,
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )

        self.assertEqual(r1["plot_name"], "MyPlot")
        self.assertEqual(r2["plot_name"], "MyPlot_1")

        self.assertTrue(self.canvas.delete_plot("MyPlot"))
        self.assertTrue(self.canvas.delete_plot("MyPlot_1"))

    def test_plot_distribution_discrete_creates_bars(self) -> None:
        result = self.canvas.plot_distribution(
            name="MyBars",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -2.0, "right_bound": 2.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=5,
        )

        self.assertEqual(result["plot_name"], "MyBars")
        self.assertEqual(result["representation"], "discrete")
        self.assertEqual(result["bar_count"], 5)

        self.assertIn("MyBars", self._names_for_class("DiscretePlot"))

        bar_names = set(self._names_for_class("Bar"))
        for i in range(result["bar_count"]):
            self.assertIn(f"{result['plot_name']}_bar_{i}", bar_names)

    def test_delete_plot_discrete_removes_bars(self) -> None:
        result = self.canvas.plot_distribution(
            name="MyBars",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -2.0, "right_bound": 2.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=5,
        )
        expected_bar_names = [f"{result['plot_name']}_bar_{i}" for i in range(result["bar_count"])]

        self.assertTrue(self.canvas.delete_plot("MyBars"))
        self.assertNotIn("MyBars", self._names_for_class("DiscretePlot"))

        bar_names = set(self._names_for_class("Bar"))
        for bar_name in expected_bar_names:
            self.assertNotIn(bar_name, bar_names)

    def test_plot_bars_creates_components_and_labels(self) -> None:
        result = self.canvas.plot_bars(
            name="Poll",
            values=[10.0, 20.0, 5.0],
            labels_below=["A", "B", "C"],
            labels_above=["10", "20", "5"],
            bar_spacing=0.5,
            bar_width=2.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            x_start=1.0,
            y_base=0.0,
        )

        self.assertEqual(result["plot_name"], "Poll")
        self.assertEqual(result["plot_type"], "bars")
        self.assertEqual(result["bar_count"], 3)

        self.assertIn("Poll", self._names_for_class("BarsPlot"))
        self.assertEqual(self._names_for_class("BarsPlot"), ["Poll"])

        bars = self.canvas.get_drawables_by_class_name("Bar")
        by_name = {b.name: b for b in bars}
        self.assertIn("Poll_bar_0", by_name)
        self.assertIn("Poll_bar_1", by_name)
        self.assertIn("Poll_bar_2", by_name)

        b0 = by_name["Poll_bar_0"]
        self.assertEqual(getattr(b0, "label_below_text", None), "A")
        self.assertEqual(getattr(b0, "label_above_text", None), "10")

        b1 = by_name["Poll_bar_1"]
        self.assertEqual(getattr(b1, "label_below_text", None), "B")
        self.assertEqual(getattr(b1, "label_above_text", None), "20")

    def test_plot_bars_spacing_affects_positions(self) -> None:
        self.canvas.plot_bars(
            name="Spaced",
            values=[1.0, 1.0],
            labels_below=["L0", "L1"],
            labels_above=None,
            bar_spacing=0.25,
            bar_width=1.5,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            x_start=2.0,
            y_base=0.0,
        )

        bars = self.canvas.get_drawables_by_class_name("Bar")
        by_name = {b.name: b for b in bars}
        b0 = by_name["Spaced_bar_0"]
        b1 = by_name["Spaced_bar_1"]
        self.assertAlmostEqual(b0.x_left, 2.0)
        self.assertAlmostEqual(b0.x_right, 3.5)
        self.assertAlmostEqual(b1.x_left, 2.0 + 1.5 + 0.25)
        self.assertAlmostEqual(b1.x_right, 2.0 + 1.5 + 0.25 + 1.5)

    def test_delete_plot_bars_removes_components(self) -> None:
        self.canvas.plot_bars(
            name="ToDelete",
            values=[1.0, 2.0],
            labels_below=["A", "B"],
            labels_above=None,
            bar_spacing=0.2,
            bar_width=1.0,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            x_start=0.0,
            y_base=0.0,
        )

        self.assertTrue(self.canvas.delete_plot("ToDelete"))
        self.assertNotIn("ToDelete", self._names_for_class("BarsPlot"))
        self.assertNotIn("ToDelete_bar_0", self._names_for_class("Bar"))
        self.assertNotIn("ToDelete_bar_1", self._names_for_class("Bar"))

    def test_delete_plot_unknown_returns_false(self) -> None:
        self.assertFalse(self.canvas.delete_plot("DoesNotExist"))

    def test_plot_bars_emits_observability_start_and_end_logs(self) -> None:
        logger_spy = _LoggerSpy()
        self.canvas.logger = logger_spy

        self.canvas.plot_bars(
            name="ObservedBars",
            values=[1.0, 2.0],
            labels_below=["A", "B"],
            labels_above=None,
            bar_spacing=0.2,
            bar_width=1.0,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            x_start=0.0,
            y_base=0.0,
        )

        self.assertTrue(any("operation': 'plot_bars'" in msg and "stage': 'start'" in msg for msg in logger_spy.messages))
        self.assertTrue(any("operation': 'plot_bars'" in msg and "stage': 'end'" in msg for msg in logger_spy.messages))
        self.assertTrue(any("elapsed_ms" in msg for msg in logger_spy.messages))

    def test_plot_distribution_failure_emits_observability_failure_log(self) -> None:
        logger_spy = _LoggerSpy()
        self.canvas.logger = logger_spy

        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadRep",
                representation="invalid",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds=None,
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

        self.assertTrue(any("operation': 'plot_distribution'" in msg and "stage': 'failure'" in msg for msg in logger_spy.messages))

    def test_plot_distribution_rejects_invalid_representation(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadRep",
                representation="invalid",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds=None,
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

    def test_plot_distribution_rejects_invalid_distribution_type(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadDist",
                representation="continuous",
                distribution_type="poisson",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds=None,
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

    def test_plot_distribution_accepts_null_params_and_defaults(self) -> None:
        result = self.canvas.plot_distribution(
            name="Defaults",
            representation="continuous",
            distribution_type="normal",
            distribution_params=None,
            plot_bounds=None,
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )
        self.assertEqual(result["distribution_params"], {"mean": 0.0, "sigma": 1.0})
        self.assertEqual(result["bounds"], {"left": -4.0, "right": 4.0})
        self.assertEqual(result["shade_bounds"], {"left": -4.0, "right": 4.0})

    def test_plot_distribution_rejects_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadBounds",
                representation="continuous",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds={"left_bound": 1.0, "right_bound": 1.0},
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="BadBounds2",
                representation="continuous",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds={"left_bound": float("inf"), "right_bound": 1.0},
                shade_bounds=None,
                curve_color=None,
                fill_color=None,
                fill_opacity=None,
                bar_count=None,
            )

    def test_plot_distribution_shade_bounds_limits_fill_area(self) -> None:
        result = self.canvas.plot_distribution(
            name="Shade",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -4.0, "right_bound": 4.0},
            shade_bounds={"left_bound": -2.0, "right_bound": 2.0},
            curve_color=None,
            fill_color="lightblue",
            fill_opacity=0.5,
            bar_count=None,
        )

        self.assertEqual(result["bounds"], {"left": -4.0, "right": 4.0})
        self.assertEqual(result["shade_bounds"], {"left": -2.0, "right": 2.0})

        areas = self.canvas.get_drawables_by_class_name("FunctionsBoundedColoredArea")
        by_name = {a.name: a for a in areas}
        area = by_name[result["fill_area_name"]]
        self.assertAlmostEqual(float(getattr(area, "left_bound", 999.0)), -2.0)
        self.assertAlmostEqual(float(getattr(area, "right_bound", -999.0)), 2.0)

    def test_plot_distribution_shade_bounds_clamps_into_plot_bounds(self) -> None:
        result = self.canvas.plot_distribution(
            name="ShadeClamp",
            representation="continuous",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -4.0, "right_bound": 4.0},
            shade_bounds={"left_bound": -10.0, "right_bound": 10.0},
            curve_color=None,
            fill_color="lightblue",
            fill_opacity=0.5,
            bar_count=None,
        )
        self.assertEqual(result["bounds"], {"left": -4.0, "right": 4.0})
        self.assertEqual(result["shade_bounds"], {"left": -4.0, "right": 4.0})

    def test_plot_distribution_shade_bounds_invalid_after_clamp_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas.plot_distribution(
                name="ShadeBad",
                representation="continuous",
                distribution_type="normal",
                distribution_params={"mean": 0.0, "sigma": 1.0},
                plot_bounds={"left_bound": -4.0, "right_bound": 4.0},
                shade_bounds={"left_bound": 10.0, "right_bound": 11.0},
                curve_color=None,
                fill_color="lightblue",
                fill_opacity=0.5,
                bar_count=None,
            )

    def test_plot_distribution_discrete_default_bar_count(self) -> None:
        result = self.canvas.plot_distribution(
            name="DefaultBars",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=None,
        )
        self.assertEqual(result["bar_count"], 24)
        self.assertEqual(len(self._names_for_class("Bar")), 24)

    def test_plot_distribution_discrete_rejects_invalid_bar_count(self) -> None:
        for bar_count in (0, -1, 1.5, float("inf")):
            with self.subTest(bar_count=bar_count):
                with self.assertRaises(ValueError):
                    self.canvas.plot_distribution(
                        name="BadCount",
                        representation="discrete",
                        distribution_type="normal",
                        distribution_params={"mean": 0.0, "sigma": 1.0},
                        plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
                        shade_bounds=None,
                        curve_color=None,
                        fill_color=None,
                        fill_opacity=None,
                        bar_count=bar_count,
                    )

    def test_plot_distribution_discrete_defaults_fill_color_and_opacity(self) -> None:
        result = self.canvas.plot_distribution(
            name="FillDefaults",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color="  ",
            fill_opacity=None,
            bar_count=3,
        )
        bars = self.canvas.get_drawables_by_class_name("Bar")
        by_name = {b.name: b for b in bars}
        b0 = by_name[f"{result['plot_name']}_bar_0"]
        self.assertEqual(getattr(b0, "fill_color", None), default_area_fill_color)
        self.assertAlmostEqual(float(getattr(b0, "fill_opacity", -1.0)), float(default_area_opacity))

    def test_plot_distribution_discrete_clamps_fill_opacity(self) -> None:
        result = self.canvas.plot_distribution(
            name="Clamp",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=10.0,
            bar_count=1,
        )
        bars = self.canvas.get_drawables_by_class_name("Bar")
        by_name = {b.name: b for b in bars}
        bar = by_name[f"{result['plot_name']}_bar_0"]
        self.assertAlmostEqual(float(getattr(bar, "fill_opacity", -1.0)), 1.0)

        result2 = self.canvas.plot_distribution(
            name="Clamp2",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=-1.0,
            bar_count=1,
        )
        bars2 = self.canvas.get_drawables_by_class_name("Bar")
        by_name2 = {b.name: b for b in bars2}
        bar2 = by_name2[f"{result2['plot_name']}_bar_0"]
        self.assertAlmostEqual(float(getattr(bar2, "fill_opacity", -1.0)), 0.0)

        result3 = self.canvas.plot_distribution(
            name="Clamp3",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=float("inf"),
            bar_count=1,
        )
        bars3 = self.canvas.get_drawables_by_class_name("Bar")
        by_name3 = {b.name: b for b in bars3}
        bar3 = by_name3[f"{result3['plot_name']}_bar_0"]
        self.assertAlmostEqual(float(getattr(bar3, "fill_opacity", -1.0)), float(default_area_opacity))

    def test_materialize_discrete_plot_creates_bars_and_is_idempotent(self) -> None:
        plot = DiscretePlot(
            "RestoredDiscrete",
            plot_type="distribution",
            distribution_type="normal",
            bar_count=3,
            bar_labels=["a", "b", "c"],
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -1.0, "right": 1.0},
            metadata=None,
        )
        self.canvas.drawable_manager.drawables.add(plot)
        stats = self.canvas.drawable_manager.statistics_manager

        stats.materialize_discrete_plot(plot)
        self.assertIn("RestoredDiscrete_bar_0", self._names_for_class("Bar"))
        self.assertIn("RestoredDiscrete_bar_1", self._names_for_class("Bar"))
        self.assertIn("RestoredDiscrete_bar_2", self._names_for_class("Bar"))

        stats.materialize_discrete_plot(plot)
        self.assertEqual(len(self._names_for_class("Bar")), 3)

    def test_materialize_bars_plot_is_idempotent(self) -> None:
        plot = BarsPlot(
            "RestoredBars",
            plot_type="bars",
            values=[1.0, 2.0],
            labels_below=["A", "B"],
            labels_above=None,
            bar_spacing=0.2,
            bar_width=1.0,
            x_start=0.0,
            y_base=0.0,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            bounds=None,
            metadata=None,
        )
        self.canvas.drawable_manager.drawables.add(plot)
        stats = self.canvas.drawable_manager.statistics_manager
        stats.materialize_bars_plot(plot)
        self.assertIn("RestoredBars_bar_0", self._names_for_class("Bar"))
        self.assertIn("RestoredBars_bar_1", self._names_for_class("Bar"))

        stats.materialize_bars_plot(plot)
        self.assertEqual(len(self._names_for_class("Bar")), 2)

    def test_delete_plot_tolerates_missing_derived_bars(self) -> None:
        plot = DiscretePlot(
            "PartialDiscrete",
            plot_type="distribution",
            distribution_type="normal",
            bar_count=3,
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -1.0, "right": 1.0},
        )
        self.canvas.drawable_manager.drawables.add(plot)
        self.canvas.drawable_manager.bar_manager.create_bar(
            name="PartialDiscrete_bar_0",
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.canvas.drawable_manager.bar_manager.create_bar(
            name="PartialDiscrete_bar_2",
            x_left=2.0,
            x_right=3.0,
            y_bottom=0.0,
            y_top=1.0,
            archive=False,
            redraw=False,
        )
        self.assertTrue(self.canvas.delete_plot("PartialDiscrete"))
        self.assertNotIn("PartialDiscrete", self._names_for_class("DiscretePlot"))
        self.assertNotIn("PartialDiscrete_bar_0", self._names_for_class("Bar"))
        self.assertNotIn("PartialDiscrete_bar_1", self._names_for_class("Bar"))
        self.assertNotIn("PartialDiscrete_bar_2", self._names_for_class("Bar"))

