from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from canvas import Canvas


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
            left_bound=None,
            right_bound=None,
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
            left_bound=None,
            right_bound=None,
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
                left_bound=None,
                right_bound=None,
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
            left_bound=None,
            right_bound=None,
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
            left_bound=None,
            right_bound=None,
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
            left_bound=-2.0,
            right_bound=2.0,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=5,
        )

        self.assertEqual(result["plot_name"], "MyBars")
        self.assertEqual(result["representation"], "discrete")
        self.assertIn("bar_names", result)
        self.assertEqual(len(result["bar_names"]), 5)

        self.assertIn("MyBars", self._names_for_class("DiscretePlot"))

        bar_names = self._names_for_class("Bar")
        for bar_name in result["bar_names"]:
            self.assertIn(bar_name, bar_names)

    def test_delete_plot_discrete_removes_bars(self) -> None:
        result = self.canvas.plot_distribution(
            name="MyBars",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            left_bound=-2.0,
            right_bound=2.0,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=5,
        )

        self.assertTrue(self.canvas.delete_plot("MyBars"))
        self.assertNotIn("MyBars", self._names_for_class("DiscretePlot"))

        bar_names = self._names_for_class("Bar")
        for bar_name in result["bar_names"]:
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


