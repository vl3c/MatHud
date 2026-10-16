"""The plot-kind table (static/client/plot_kinds.py) covers every plot class and stays consistent."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from managers.statistics_manager import StatisticsManager
from plot_kinds import (
    PLOT_CLASS_NAMES,
    PLOT_KINDS,
    PlotKind,
    build_saved_plot,
    derived_bar_name,
    derived_bar_prefix,
    is_legacy_continuous_args,
    is_legacy_discrete_args,
    plot_kind,
)
from server_tests.test_drawable_wiring import _drawable_buckets, _literal


def _kind(class_name: str) -> PlotKind:
    kind = plot_kind(class_name)
    assert kind is not None, class_name
    return kind


def _sample_plots() -> List[Plot]:
    return [
        ContinuousPlot(
            "C",
            plot_type="distribution",
            distribution_type="normal",
            function_name="C_pdf",
            fill_area_name="C_fill",
            distribution_params={"mean": 0.5, "sigma": 2.0},
            bounds={"left": -7.5, "right": 8.5},
            metadata={"note": "x"},
        ),
        DiscretePlot(
            "D",
            plot_type="distribution",
            distribution_type=None,
            bar_count=3,
            bar_labels=["a", "b", "c"],
            curve_color="red",
            fill_color="orange",
            fill_opacity=0.5,
            distribution_params={"mean": 0.0, "sigma": 1.0},
            bounds={"left": -1.5, "right": 1.5},
            metadata={"bar_count": 3},
        ),
        DiscretePlot("Legacy", plot_type="distribution", rectangle_names=["R"], fill_area_names=["A"]),
        BarsPlot(
            "B",
            plot_type="bars",
            values=[1.0, 2.5],
            labels_below=["a", "b"],
            labels_above=["1", "2.5"],
            bar_spacing=0.2,
            bar_width=1.0,
            x_start=3.0,
            y_base=-1.0,
            stroke_color="black",
            fill_color="blue",
            fill_opacity=0.4,
        ),
        BarsPlot("Pruned", plot_type="bars", values=[1.0], labels_below=["a"]),
        Plot("P", plot_type="distribution", distribution_params={"mean": 1, "sigma": 2}),
    ]


class TestPlotKindRows(unittest.TestCase):
    def test_each_bucket_is_the_class_name_plus_s(self) -> None:
        for kind in PLOT_KINDS:
            with self.subTest(kind=kind.class_name):
                self.assertEqual(kind.bucket, kind.class_name + "s")

    def test_every_plot_class_has_a_row(self) -> None:
        plot_buckets = {bucket for bucket in _drawable_buckets() if bucket.endswith("Plots")}
        self.assertEqual(plot_buckets, {kind.bucket for kind in PLOT_KINDS})

    def test_every_bucket_is_restored(self) -> None:
        restored = set(_literal("workspace_manager.py", "RESTORED_BUCKETS"))
        self.assertEqual(sorted(kind.bucket for kind in PLOT_KINDS if kind.bucket not in restored), [])

    def test_named_methods_exist_on_the_statistics_manager(self) -> None:
        for kind in PLOT_KINDS:
            for method_name in (kind.materialize, kind.delete_parts):
                if method_name is None:
                    continue
                with self.subTest(kind=kind.class_name, method=method_name):
                    self.assertTrue(callable(getattr(StatisticsManager, method_name, None)))

    def test_rows_keep_the_restore_order(self) -> None:
        self.assertEqual(PLOT_CLASS_NAMES, ("ContinuousPlot", "DiscretePlot", "BarsPlot", "Plot"))
        self.assertEqual([kind.class_name for kind in PLOT_KINDS if kind.materialize], ["DiscretePlot", "BarsPlot"])
        self.assertEqual([kind.class_name for kind in PLOT_KINDS if kind.derived_bars], ["DiscretePlot", "BarsPlot"])

    def test_plot_kind_looks_up_rows_by_class_name(self) -> None:
        self.assertIs(plot_kind("BarsPlot"), PLOT_KINDS[2])
        self.assertIsNone(plot_kind("Bar"))
        self.assertIsNone(plot_kind(""))


class TestPlotKindBuilders(unittest.TestCase):
    def test_saved_plots_rebuild_to_the_same_state(self) -> None:
        for plot in _sample_plots():
            with self.subTest(plot=plot.name):
                state = plot.get_state()
                rebuilt, name = build_saved_plot(_kind(plot.get_class_name()), state)
                self.assertEqual(name, plot.name)
                self.assertEqual(rebuilt.get_class_name(), plot.get_class_name())
                self.assertEqual(rebuilt.get_state(), state)

    def test_plot_type_defaults_to_distribution(self) -> None:
        for kind in PLOT_KINDS:
            with self.subTest(kind=kind.class_name):
                plot, _ = build_saved_plot(kind, {"name": "N", "args": {"plot_type": ""}})
                self.assertEqual(plot.plot_type, "distribution")

    def test_bars_ignore_saved_distribution_fields(self) -> None:
        args: Dict[str, Any] = {
            "plot_type": "bars",
            "distribution_type": "normal",
            "distribution_params": {"mean": 0},
            "values": [1],
            "labels_below": ["a"],
        }
        plot, _ = build_saved_plot(_kind("BarsPlot"), {"name": "B", "args": args})
        self.assertIsNone(plot.distribution_type)
        self.assertEqual(plot.distribution_params, {})

    def test_items_without_dict_args_build_from_empty_args(self) -> None:
        plot, name = build_saved_plot(_kind("DiscretePlot"), {"name": None, "args": ["x"]})
        self.assertEqual(name, "")
        self.assertEqual(plot.get_state()["args"]["bar_count"], None)

    def test_legacy_plots_rebuild_as_the_subclass_their_parts_call_for(self) -> None:
        legacy = _kind("Plot")
        cases = [
            ({"rectangle_names": ["R"]}, "DiscretePlot"),
            ({"fill_area_names": ["A"]}, "DiscretePlot"),
            ({"rectangle_names": ["R"], "function_name": "f"}, "DiscretePlot"),
            ({"function_name": "f"}, "ContinuousPlot"),
            ({"fill_area_name": "A"}, "ContinuousPlot"),
            ({"rectangle_names": [], "function_name": None}, "Plot"),
            ({}, "Plot"),
        ]
        for args, class_name in cases:
            with self.subTest(args=args):
                plot, _ = build_saved_plot(legacy, {"name": "L", "args": args})
                self.assertEqual(plot.get_class_name(), class_name)

    def test_legacy_detectors(self) -> None:
        self.assertTrue(is_legacy_discrete_args({"rectangle_names": ["r1"]}))
        self.assertTrue(is_legacy_discrete_args({"fill_area_names": ["a1"]}))
        self.assertFalse(is_legacy_discrete_args({}))
        self.assertTrue(is_legacy_continuous_args({"function_name": "f"}))
        self.assertTrue(is_legacy_continuous_args({"fill_area_name": "fa"}))
        self.assertFalse(is_legacy_continuous_args({}))


class TestDerivedBarNames(unittest.TestCase):
    def test_names_and_prefix(self) -> None:
        self.assertEqual(derived_bar_prefix("Poll"), "Poll_bar_")
        self.assertEqual(derived_bar_name("Poll", 0), "Poll_bar_0")
        self.assertEqual(derived_bar_name("Poll", 12), "Poll_bar_12")
        self.assertTrue(derived_bar_name("Poll", 3).startswith(derived_bar_prefix("Poll")))


if __name__ == "__main__":
    unittest.main()
