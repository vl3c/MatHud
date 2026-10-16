"""StatisticsManager's per-kind dispatch, run under CPython with stub managers.

The canvas has no drawable manager, so bars are added to the container directly (the
``_add_bar`` fallback) and deleted from it by name.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from managers.drawables_container import DrawablesContainer
from managers.statistics_manager import StatisticsManager


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


if __name__ == "__main__":
    unittest.main()
