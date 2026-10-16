"""StatisticsManager's per-kind dispatch, run under CPython with stub managers.

The canvas has no drawable manager, so bars are added to the container directly (the
``_add_bar`` fallback) and deleted from it by name.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from managers.drawables_container import DrawablesContainer
from managers.statistics_manager import StatisticsManager
from utils.statistics.distributions import normal_pdf, normal_pdf_expression


class _Part:
    def __init__(self, name: str) -> None:
        self.name = name


class _FunctionManager:
    def __init__(self) -> None:
        self.drawn: List[Tuple[str, Optional[str], float, float]] = []
        self.deleted: List[str] = []

    def draw_function(
        self, expression: str, *, name: str, left_bound: float, right_bound: float, color: Optional[str]
    ) -> _Part:
        self.drawn.append((expression, name, left_bound, right_bound))
        return _Part(name)

    def delete_function(self, name: str) -> None:
        self.deleted.append(name)


class _AreaManager:
    def __init__(self) -> None:
        self.created: List[Tuple[Any, ...]] = []
        self.deleted: List[str] = []

    def create_colored_area(self, **kwargs: Any) -> _Part:
        self.created.append(tuple(sorted(kwargs.items())))
        return _Part("area")

    def delete_colored_area(self, name: str) -> None:
        self.deleted.append(name)


class _DependencyManager:
    def __init__(self) -> None:
        self.removed: List[str] = []

    def analyze_drawable_for_dependencies(self, drawable: Any) -> None:
        return None

    def remove_drawable(self, drawable: Any) -> None:
        self.removed.append(drawable.name)


class _NameGenerator:
    def filter_string(self, value: str) -> str:
        return value

    def generate_function_name(self, preferred: str) -> str:
        return preferred


class StatisticsManagerHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.drawables = DrawablesContainer()
        self.functions = _FunctionManager()
        self.areas = _AreaManager()
        self.dependencies = _DependencyManager()
        self.canvas = SimpleNamespace(draw_enabled=False)
        self.stats = StatisticsManager(
            self.canvas,
            self.drawables,
            _NameGenerator(),
            self.dependencies,
            self.functions,
            self.areas,
        )

    def names(self, class_name: str) -> List[str]:
        return [str(d.name) for d in self.drawables.get_by_class_name(class_name)]

    def bar_states(self) -> List[Dict[str, Any]]:
        return [d.get_state() for d in self.drawables.get_by_class_name("Bar")]

    def plot_distribution(self, **overrides: Any) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "name": "D",
            "representation": "discrete",
            "distribution_type": "normal",
            "distribution_params": {"mean": 0.5, "sigma": 2.0},
            "plot_bounds": {"left_bound": -1.0, "right_bound": 2.0},
            "shade_bounds": None,
            "curve_color": "red",
            "fill_color": "orange",
            "fill_opacity": 0.5,
            "bar_count": 3,
        }
        args.update(overrides)
        return dict(self.stats.plot_distribution(**args))


class TestDeletePlotDispatch(StatisticsManagerHarness):
    def test_continuous_plot_deletes_its_function_and_area(self) -> None:
        plot = ContinuousPlot("C", plot_type="distribution", function_name="C_pdf", fill_area_name="C_fill")
        self.drawables.add(plot)
        self.assertTrue(self.stats.delete_plot("C"))
        self.assertEqual(self.functions.deleted, ["C_pdf"])
        self.assertEqual(self.areas.deleted, ["C_fill"])
        self.assertEqual(self.names("ContinuousPlot"), [])

    def test_discrete_plot_deletes_its_bars(self) -> None:
        self.stats.plot_distribution(
            name="D",
            representation="discrete",
            distribution_type="normal",
            distribution_params={"mean": 0.0, "sigma": 1.0},
            plot_bounds={"left_bound": -1.0, "right_bound": 1.0},
            shade_bounds=None,
            curve_color=None,
            fill_color=None,
            fill_opacity=None,
            bar_count=3,
        )
        self.assertEqual(self.names("Bar"), ["D_bar_0", "D_bar_1", "D_bar_2"])
        self.assertTrue(self.stats.delete_plot("D"))
        self.assertEqual(self.names("Bar"), [])
        self.assertEqual(self.names("DiscretePlot"), [])

    def test_bars_plot_deletes_its_bars(self) -> None:
        self.stats.plot_bars(
            name="B",
            values=[1.0, 2.0],
            labels_below=["a", "b"],
            labels_above=None,
            bar_spacing=None,
            bar_width=None,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            x_start=None,
            y_base=None,
        )
        self.assertEqual(self.names("Bar"), ["B_bar_0", "B_bar_1"])
        self.assertTrue(self.stats.delete_plot("B"))
        self.assertEqual(self.names("Bar"), [])
        self.assertEqual(self.names("BarsPlot"), [])

    def test_base_plot_removes_only_its_record(self) -> None:
        other = ContinuousPlot("C", plot_type="distribution", function_name="C_pdf", fill_area_name="C_fill")
        self.drawables.add(other)
        self.drawables.add(Plot("P", plot_type="distribution"))
        self.assertTrue(self.stats.delete_plot("P"))
        self.assertEqual(self.names("Plot"), [])
        self.assertEqual(self.names("ContinuousPlot"), ["C"])
        self.assertEqual(self.functions.deleted, [])
        self.assertEqual(self.areas.deleted, [])
        self.assertEqual(self.dependencies.removed, ["P"])

    def test_legacy_discrete_plot_keeps_its_rectangles(self) -> None:
        # A legacy item rebuilt as a DiscretePlot with no bars deletes no parts (as before).
        self.drawables.add(DiscretePlot("L", plot_type="distribution", rectangle_names=["R"], fill_area_names=["A"]))
        self.assertTrue(self.stats.delete_plot("L"))
        self.assertEqual(self.names("DiscretePlot"), [])
        self.assertEqual(self.areas.deleted, [])

    def test_unknown_name_returns_false(self) -> None:
        self.drawables.add(BarsPlot("B", plot_type="bars", values=[1.0], labels_below=["a"]))
        self.assertFalse(self.stats.delete_plot("missing"))
        self.assertEqual(self.names("BarsPlot"), ["B"])

    def test_lookup_finds_every_plot_class(self) -> None:
        plots = [
            ContinuousPlot("C", plot_type="distribution"),
            DiscretePlot("D", plot_type="distribution"),
            BarsPlot("B", plot_type="bars", values=[1.0], labels_below=["a"]),
            Plot("P", plot_type="distribution"),
        ]
        for plot in plots:
            self.drawables.add(plot)
        for plot in plots:
            with self.subTest(plot=plot.name):
                self.assertIs(self.stats._get_plot_by_name(plot.name), plot)


def _discrete_plot(distribution_type: Any = "normal", **overrides: Any) -> DiscretePlot:
    args: Dict[str, Any] = {
        "plot_type": "distribution",
        "distribution_type": distribution_type,
        "bar_count": 3,
        "bar_labels": ["a", "b", "c"],
        "curve_color": "red",
        "fill_color": "orange",
        "fill_opacity": 0.5,
        "distribution_params": {"mean": 0.5, "sigma": 2.0},
        "bounds": {"left": -1.0, "right": 2.0},
    }
    args.update(overrides)
    return DiscretePlot("R", **args)


class TestPlotDistributionPayloads(StatisticsManagerHarness):
    def test_continuous_payload(self) -> None:
        result = self.plot_distribution(
            name="C",
            representation="continuous",
            distribution_type=" Normal ",
            distribution_params={"sigma": 2, "mean": 1},
            plot_bounds=None,
            shade_bounds={"left_bound": 0.0, "right_bound": None},
        )
        expected = {
            "plot_name": "C",
            "representation": "continuous",
            "distribution_type": "normal",
            "distribution_params": {"mean": 1.0, "sigma": 2.0},
            "bounds": {"left": -7.0, "right": 9.0},
            "shade_bounds": {"left": 0.0, "right": 9.0},
            "function_name": "C_pdf",
            "fill_area_name": "C_fill",
        }
        self.assertEqual(result, expected)
        self.assertEqual(list(result), list(expected))
        self.assertEqual(list(result["distribution_params"]), ["mean", "sigma"])
        self.assertEqual(self.functions.drawn, [(normal_pdf_expression(1.0, 2.0), "C_pdf", -7.0, 9.0)])
        plot = self.drawables.get_by_class_name("ContinuousPlot")[0]
        self.assertEqual(
            plot.get_state()["args"],
            {
                "plot_type": "distribution",
                "distribution_type": "normal",
                "distribution_params": {"mean": 1.0, "sigma": 2.0},
                "bounds": {"left": -7.0, "right": 9.0},
                "metadata": {},
                "function_name": "C_pdf",
                "fill_area_name": "C_fill",
            },
        )

    def test_discrete_payload_and_bars(self) -> None:
        result = self.plot_distribution()
        expected = {
            "plot_name": "D",
            "representation": "discrete",
            "distribution_type": "normal",
            "distribution_params": {"mean": 0.5, "sigma": 2.0},
            "bounds": {"left": -1.0, "right": 2.0},
            "bar_count": 3,
        }
        self.assertEqual(result, expected)
        self.assertEqual(list(result), list(expected))
        self.assertEqual(list(result["distribution_params"]), ["mean", "sigma"])
        bars = self.bar_states()
        self.assertEqual([bar["name"] for bar in bars], ["D_bar_0", "D_bar_1", "D_bar_2"])
        for i, bar in enumerate(bars):
            with self.subTest(bar=bar["name"]):
                self.assertEqual(bar["args"]["x_left"], -1.0 + i)
                self.assertEqual(bar["args"]["y_top"], normal_pdf(-0.5 + i, 0.5, 2.0))
                self.assertEqual(bar["args"]["stroke_color"], "red")
                self.assertEqual(bar["args"]["fill_color"], "orange")
                self.assertEqual(bar["args"]["fill_opacity"], 0.5)
                self.assertIsNone(bar["args"]["label_above_text"])

    def test_default_name_and_bounds(self) -> None:
        result = self.plot_distribution(name=None, distribution_params=None, plot_bounds=None, bar_count=None)
        self.assertEqual(result["plot_name"], "normal_plot")
        self.assertEqual(result["distribution_params"], {"mean": 0.0, "sigma": 1.0})
        self.assertEqual(result["bounds"], {"left": -4.0, "right": 4.0})
        self.assertEqual(result["bar_count"], 24)

    def test_tool_errors_are_unchanged(self) -> None:
        cases = [
            ({"distribution_type": None}, "Unsupported distribution_type 'None'"),
            ({"distribution_type": "poisson"}, "Unsupported distribution_type 'poisson'"),
            ({"representation": "pmf"}, "representation must be 'continuous' or 'discrete'"),
            ({"distribution_params": {"sigma": 0}}, "sigma must be > 0"),
            ({"distribution_params": {"mean": float("nan")}}, "mean must be finite"),
            ({"plot_bounds": {"left_bound": 2.0, "right_bound": 1.0}}, "left_bound must be less than right_bound"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError) as caught:
                    self.plot_distribution(**overrides)
                self.assertEqual(str(caught.exception), message)
        self.assertEqual(self.names("DiscretePlot"), [])
        self.assertEqual(self.names("Bar"), [])

    def test_plot_bars_payload(self) -> None:
        result = self.stats.plot_bars(
            name="B",
            values=[1, 2],
            labels_below=["a", "b"],
            labels_above=None,
            bar_spacing=None,
            bar_width=None,
            stroke_color=None,
            fill_color=None,
            fill_opacity=None,
            x_start=None,
            y_base=None,
        )
        self.assertEqual(result, {"plot_name": "B", "plot_type": "bars", "bar_count": 2})
        self.assertEqual(list(result), ["plot_name", "plot_type", "bar_count"])


class TestMaterializeDiscretePlot(StatisticsManagerHarness):
    def materialize(self, plot: DiscretePlot) -> Tuple[List[Dict[str, Any]], str]:
        self.drawables.add(plot)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.stats.materialize_discrete_plot(plot)
        return self.bar_states(), output.getvalue()

    def test_untyped_plots_materialize_as_normal(self) -> None:
        self.plot_distribution(name="R")
        drawn = self.bar_states()
        for distribution_type in ("normal", None, "", " NORMAL "):
            with self.subTest(distribution_type=distribution_type):
                self.setUp()
                bars, output = self.materialize(_discrete_plot(distribution_type, bar_labels=None))
                self.assertEqual(bars, drawn)
                self.assertEqual(output, "")

    def test_materialized_bars_carry_labels_and_colours(self) -> None:
        bars, _ = self.materialize(_discrete_plot())
        self.assertEqual([bar["args"]["label_above_text"] for bar in bars], ["a", "b", "c"])
        self.assertEqual({bar["args"]["stroke_color"] for bar in bars}, {"red"})

    def test_unknown_type_makes_no_bars_and_warns(self) -> None:
        bars, output = self.materialize(_discrete_plot("poisson"))
        self.assertEqual(bars, [])
        self.assertIn("Unsupported distribution_type 'poisson'", output)
        self.assertIn("'R'", output)

    def test_plots_without_bars_never_warn(self) -> None:
        for overrides in ({"bar_count": None}, {"bar_count": 0}, {"bounds": {}}, {"bounds": {"left": 1, "right": 0}}):
            with self.subTest(overrides=overrides):
                self.setUp()
                bars, output = self.materialize(_discrete_plot("poisson", **overrides))
                self.assertEqual((bars, output), ([], ""))

    def test_saved_param_edge_cases_keep_their_outcome(self) -> None:
        normal_bars = [normal_pdf(x, 0.0, 1.0) for x in (-0.5, 0.5, 1.5)]
        cases: List[Tuple[Any, List[float]]] = [
            ({"mean": None, "sigma": None}, []),
            ({"mean": None}, []),
            ({"mean": 0.0, "sigma": None}, []),
            ({}, normal_bars),
            (None, normal_bars),
            ([1, 2], normal_bars),
            ("text", normal_bars),
            ({"mean": "0", "sigma": "1"}, normal_bars),
            ({"mean": 0, "sigma": 1, "extra": None}, normal_bars),
            ({"sigma": 0}, []),
            ({"sigma": -1.0}, []),
            ({"sigma": float("nan")}, []),
            ({"mean": float("inf")}, []),
            ({"mean": "abc"}, []),
            ({"mean": [1]}, []),
        ]
        for params, heights in cases:
            with self.subTest(params=params):
                self.setUp()
                plot = _discrete_plot()
                plot.distribution_params = params  # type: ignore[assignment]
                bars, output = self.materialize(plot)
                self.assertEqual([bar["args"]["y_top"] for bar in bars], heights)
                self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
