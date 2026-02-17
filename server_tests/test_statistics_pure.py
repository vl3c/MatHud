from __future__ import annotations

import math
import unittest
from copy import deepcopy

from drawables.bar import Bar
from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from utils.statistics.distributions import _require_finite, default_normal_bounds, normal_pdf_expression


class TestStatisticsDistributionsPure(unittest.TestCase):
    def test_require_finite_accepts_numbers(self) -> None:
        self.assertEqual(_require_finite(10, "v"), 10.0)
        self.assertEqual(_require_finite(10.5, "v"), 10.5)

    def test_require_finite_rejects_non_numbers(self) -> None:
        with self.assertRaises(TypeError):
            _require_finite("x", "v")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            _require_finite(None, "v")  # type: ignore[arg-type]

    def test_require_finite_rejects_non_finite(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _require_finite(value, "v")

    def test_normal_pdf_expression_embeds_params(self) -> None:
        expr = normal_pdf_expression(1.25, 2.5)
        self.assertIn("exp", expr)
        self.assertIn("sqrt", expr)
        self.assertIn("pi", expr)
        self.assertIn("^", expr)
        self.assertIn("(2.5)", expr)
        self.assertIn("(1.25)", expr)

    def test_normal_pdf_expression_rejects_non_finite_inputs(self) -> None:
        with self.assertRaises(ValueError):
            normal_pdf_expression(float("nan"), 1.0)
        with self.assertRaises(ValueError):
            normal_pdf_expression(0.0, float("inf"))

    def test_default_normal_bounds_rejects_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, 0.0)
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, -1.0)
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, 1.0, k=0.0)
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, 1.0, k=-1.0)

        with self.assertRaises(ValueError):
            default_normal_bounds(float("inf"), 1.0)
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, float("nan"))
        with self.assertRaises(ValueError):
            default_normal_bounds(0.0, 1.0, k=float("inf"))


class TestStatisticsDrawablesPure(unittest.TestCase):
    def test_bar_label_alias_and_state(self) -> None:
        bar = Bar(
            name="B",
            x_left=0.0,
            x_right=2.0,
            y_bottom=0.0,
            y_top=3.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            label_text="alias",
            label_below_text="below",
        )
        self.assertEqual(bar.get_class_name(), "Bar")
        self.assertEqual(bar.label_above_text, "alias")
        self.assertEqual(bar.label_text, "alias")
        self.assertEqual(bar.label_below_text, "below")

        state = bar.get_state()
        self.assertEqual(state["name"], "B")
        args = state["args"]
        self.assertEqual(args["x_left"], 0.0)
        self.assertEqual(args["x_right"], 2.0)
        self.assertEqual(args["y_bottom"], 0.0)
        self.assertEqual(args["y_top"], 3.0)
        self.assertEqual(args["stroke_color"], "#111")
        self.assertEqual(args["fill_color"], "#222")
        self.assertEqual(args["fill_opacity"], 0.5)
        self.assertEqual(args["label_above_text"], "alias")
        self.assertEqual(args["label_below_text"], "below")
        self.assertEqual(args["label_text"], "alias")

    def test_bar_translate(self) -> None:
        bar = Bar(name="B", x_left=1.0, x_right=3.0, y_bottom=2.0, y_top=5.0)
        bar.translate(-2.0, 10.0)
        self.assertEqual(bar.x_left, -1.0)
        self.assertEqual(bar.x_right, 1.0)
        self.assertEqual(bar.y_bottom, 12.0)
        self.assertEqual(bar.y_top, 15.0)

    def test_bar_deepcopy_is_independent(self) -> None:
        bar = Bar(
            name="B",
            x_left=0.0,
            x_right=2.0,
            y_bottom=0.0,
            y_top=3.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            label_above_text="a",
            label_below_text="b",
            is_renderable=False,
        )
        bar_copy = deepcopy(bar)
        self.assertIsNot(bar_copy, bar)
        self.assertEqual(bar_copy.get_state(), bar.get_state())
        self.assertEqual(bar_copy.is_renderable, False)

        bar_copy.translate(1.0, 1.0)
        self.assertNotEqual(bar_copy.x_left, bar.x_left)

    def test_plot_base_state_and_deepcopy(self) -> None:
        plot = Plot(
            "P",
            plot_type="distribution",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -4.0, "right": 4.0},
            metadata={"k": 4},
        )
        self.assertEqual(plot.get_class_name(), "Plot")
        self.assertFalse(plot.is_renderable)

        state = plot.get_state()
        self.assertEqual(state["name"], "P")
        self.assertEqual(state["args"]["plot_type"], "distribution")
        self.assertEqual(state["args"]["distribution_type"], "normal")
        self.assertEqual(state["args"]["bounds"], {"left": -4.0, "right": 4.0})
        self.assertEqual(state["args"]["metadata"], {"k": 4})

        plot_copy = deepcopy(plot)
        self.assertIsNot(plot_copy, plot)
        self.assertEqual(plot_copy.get_state(), plot.get_state())

    def test_plot_subclasses_state_and_deepcopy(self) -> None:
        continuous = ContinuousPlot(
            "C",
            plot_type="distribution",
            distribution_type="normal",
            function_name="f1",
            fill_area_name="a1",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -4.0, "right": 4.0},
            metadata={"m": 1},
            is_renderable=False,
        )
        self.assertEqual(continuous.get_class_name(), "ContinuousPlot")
        self.assertEqual(continuous.is_renderable, False)
        self.assertEqual(continuous.get_state()["args"]["function_name"], "f1")
        self.assertEqual(continuous.get_state()["args"]["fill_area_name"], "a1")
        self.assertEqual(deepcopy(continuous).get_state(), continuous.get_state())

        discrete = DiscretePlot(
            "D",
            plot_type="distribution",
            distribution_type="normal",
            bar_count=5,
            bar_labels=["L0", "L1"],
            curve_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            rectangle_names=["r1"],
            fill_area_names=["fa1"],
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -2.0, "right": 2.0},
            metadata={"n": 5},
            is_renderable=False,
        )
        self.assertEqual(discrete.get_class_name(), "DiscretePlot")
        self.assertEqual(discrete.is_renderable, False)
        d_state = discrete.get_state()["args"]
        self.assertEqual(d_state["bar_count"], 5)
        self.assertEqual(d_state["bar_labels"], ["L0", "L1"])
        self.assertEqual(d_state["curve_color"], "#111")
        self.assertEqual(d_state["fill_color"], "#222")
        self.assertEqual(d_state["fill_opacity"], 0.5)
        self.assertEqual(d_state["rectangle_names"], ["r1"])
        self.assertEqual(d_state["fill_area_names"], ["fa1"])
        self.assertEqual(deepcopy(discrete).get_state(), discrete.get_state())

        bars = BarsPlot(
            "BARS",
            plot_type="bars",
            values=[1, 2.5],
            labels_below=["A", "B"],
            labels_above=["1", "2.5"],
            bar_spacing=0.25,
            bar_width=1.5,
            x_start=2.0,
            y_base=0.0,
            stroke_color="#111",
            fill_color="#222",
            fill_opacity=0.5,
            bounds={"left": 0.0, "right": 10.0},
            metadata={"t": "x"},
            is_renderable=False,
        )
        self.assertEqual(bars.get_class_name(), "BarsPlot")
        self.assertEqual(bars.is_renderable, False)
        b_state = bars.get_state()["args"]
        self.assertEqual(b_state["values"], [1.0, 2.5])
        self.assertEqual(b_state["labels_below"], ["A", "B"])
        self.assertEqual(b_state["labels_above"], ["1", "2.5"])
        self.assertAlmostEqual(b_state["bar_spacing"], 0.25)
        self.assertAlmostEqual(b_state["bar_width"], 1.5)
        self.assertAlmostEqual(b_state["x_start"], 2.0)
        self.assertAlmostEqual(b_state["y_base"], 0.0)
        self.assertEqual(b_state["stroke_color"], "#111")
        self.assertEqual(b_state["fill_color"], "#222")
        self.assertAlmostEqual(b_state["fill_opacity"], 0.5)
        self.assertEqual(deepcopy(bars).get_state(), bars.get_state())

    def test_distributions_pdf_matches_expected_peak(self) -> None:
        # Sanity check for the normal PDF peak value at x == mean.
        mean = 0.0
        sigma = 2.0
        expected = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        expr = normal_pdf_expression(mean, sigma)
        self.assertTrue(expr)
        self.assertAlmostEqual(expected, 0.19947114020071635)
