"""plot_distribution for the distributions beyond normal: support clipping, one bar per value,
bar shading, the shaded probability and the notes in the result."""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List

from drawables.discrete_plot import DiscretePlot
from managers.statistics_manager import UNSHADED_BAR_FILL_COLOR
from plot_kinds import plot_kind
from server_tests.test_statistics_manager_dispatch import StatisticsManagerHarness
from utils.statistics.distribution_plot import MAX_INTEGER_BARS
from utils.statistics.distributions import DISTRIBUTIONS


def _call(**overrides: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "name": "P",
        "representation": "discrete",
        "distribution_type": "binomial",
        "distribution_params": {"n": 10, "p": 0.3},
        "plot_bounds": None,
        "shade_bounds": None,
        "curve_color": None,
        "fill_color": "orange",
        "fill_opacity": 0.5,
        "bar_count": None,
    }
    args.update(overrides)
    return args


class TestIntegerBars(StatisticsManagerHarness):
    def plot(self, **overrides: Any) -> Dict[str, Any]:
        return dict(self.stats.plot_distribution(**_call(**overrides)))

    def bars(self) -> List[Dict[str, Any]]:
        return [state["args"] for state in self.bar_states()]

    def test_binomial_draws_one_bar_per_value(self) -> None:
        result = self.plot()
        self.assertEqual(result["bar_count"], 11)
        self.assertEqual(result["values"], "0..10")
        self.assertEqual(result["bounds"], {"left": -0.5, "right": 10.5})
        spec = DISTRIBUTIONS["binomial"]
        params = spec.parse_params({"n": 10, "p": 0.3})
        for k, bar in enumerate(self.bars()):
            with self.subTest(k=k):
                self.assertEqual((bar["x_left"], bar["x_right"]), (k - 0.5, k + 0.5))
                self.assertEqual(bar["y_top"], spec.pdf(float(k), params))
                self.assertEqual(bar["fill_color"], "orange")
        self.assertAlmostEqual(sum(bar["y_top"] for bar in self.bars()), 1.0, places=12)
        self.assertNotIn("shaded_probability", result)
        self.assertNotIn("note", result)

    def test_shading_highlights_the_values_and_reports_their_probability(self) -> None:
        result = self.plot(shade_bounds={"left_bound": 2, "right_bound": 5})
        self.assertEqual(result["shaded_range"], "P(2 <= X <= 5)")
        self.assertAlmostEqual(result["shaded_probability"], 0.8033426667, places=9)
        self.assertEqual(result["shade_bounds"], {"left": 2.0, "right": 5.0})
        colours = [bar["fill_color"] for bar in self.bars()]
        self.assertEqual(colours, [UNSHADED_BAR_FILL_COLOR] * 2 + ["orange"] * 4 + [UNSHADED_BAR_FILL_COLOR] * 5)
        self.assertEqual({bar["fill_opacity"] for bar in self.bars()}, {0.5})
        plot = self.drawables.get_by_class_name("DiscretePlot")[0]
        self.assertEqual(plot.get_state()["args"]["shade_bounds"], {"left": 2.0, "right": 5.0})

    def test_an_open_shade_side_runs_to_the_end_of_the_support(self) -> None:
        result = self.plot(shade_bounds={"left_bound": 8, "right_bound": None})
        self.assertEqual(result["shaded_range"], "P(8 <= X <= 10)")
        spec = DISTRIBUTIONS["binomial"]
        params = spec.parse_params({"n": 10, "p": 0.3})
        self.assertAlmostEqual(result["shaded_probability"], spec.sf(7.0, params), places=12)
        self.assertEqual(result["shade_bounds"], {"left": 8.0, "right": 10.0})

    def test_poisson_upper_tail_is_open(self) -> None:
        result = self.plot(
            distribution_type="poisson",
            distribution_params={"rate": 4},
            shade_bounds={"left_bound": 3, "right_bound": None},
        )
        self.assertEqual(result["shaded_range"], "P(X >= 3)")
        spec = DISTRIBUTIONS["poisson"]
        self.assertAlmostEqual(result["shaded_probability"], spec.sf(2.0, {"rate": 4.0}), places=12)
        self.assertEqual(result["shade_bounds"], {"left": 3.0, "right": None})
        self.assertEqual(result["values"], "0..14")

    def test_a_shade_beyond_the_default_bars_widens_them(self) -> None:
        result = self.plot(
            distribution_type="poisson",
            distribution_params={"rate": 2},
            shade_bounds={"left_bound": 20, "right_bound": 25},
        )
        self.assertEqual(result["values"], "0..25")
        self.assertEqual(result["shaded_range"], "P(20 <= X <= 25)")

    def test_a_shade_past_given_bounds_is_counted_in_full_with_a_note(self) -> None:
        result = self.plot(
            plot_bounds={"left_bound": 0, "right_bound": 5}, shade_bounds={"left_bound": 4, "right_bound": None}
        )
        self.assertEqual(result["values"], "0..5")
        self.assertEqual(result["shaded_range"], "P(4 <= X <= 10)")
        self.assertIn("reaches past the drawn bars", result["note"])

    def test_given_bounds_round_inward_and_clip_to_the_support(self) -> None:
        result = self.plot(plot_bounds={"left_bound": -3.2, "right_bound": 4.7})
        self.assertEqual(result["values"], "0..4")
        self.assertIn("plot_bounds clipped to the support of binomial, [0, 10]", result["note"])

    def test_continuous_representation_and_bar_count_are_noted(self) -> None:
        result = self.plot(representation="continuous", bar_count=24)
        self.assertEqual(result["representation"], "discrete")
        self.assertIn("binomial is discrete: drawn as bars, one per value", result["note"])
        self.assertIn("bar_count is ignored", result["note"])
        self.assertEqual(result["bar_count"], 11)

    def test_other_distributions_params_are_ignored_and_named(self) -> None:
        result = self.plot(distribution_params={"n": 10, "p": 0.3, "mean": 0, "sigma": 1, "rate": None})
        self.assertEqual(result["ignored_params"], ["mean", "sigma"])
        self.assertEqual(result["distribution_params"], {"n": 10.0, "p": 0.3})

    def test_a_geometric_near_certain_success_still_draws(self) -> None:
        result = self.plot(distribution_type="geometric", distribution_params={"p": 0.999})
        self.assertEqual(result["values"], "1..2")

    def test_wide_default_bounds_are_trimmed_around_the_mode(self) -> None:
        result = self.plot(distribution_type="geometric", distribution_params={"p": 0.001})
        self.assertEqual(result["bar_count"], MAX_INTEGER_BARS)
        self.assertEqual(result["values"], f"1..{MAX_INTEGER_BARS}")
        self.assertIn("around the most likely value", result["note"])
        binomial = self.plot(name="Q", distribution_params={"n": 1000000, "p": 0.5})
        self.assertEqual(binomial["values"], "499750..500249")

    def test_wide_given_bounds_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.plot(
                distribution_type="poisson",
                distribution_params={"rate": 4},
                plot_bounds={"left_bound": 0, "right_bound": 600},
            )
        self.assertIn("at most 500 bars", str(caught.exception))
        self.assertEqual(self.names("DiscretePlot"), [])

    def test_bounds_that_miss_the_support_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.plot(plot_bounds={"left_bound": 11, "right_bound": 20})
        self.assertEqual(str(caught.exception), "plot_bounds [11, 20] miss the support of binomial, [0, 10]")

    def test_needed_params_are_named(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.plot(distribution_type="poisson", distribution_params={"mean": 4})
        self.assertEqual(str(caught.exception), "poisson needs rate (λ)")

    def test_restore_rebuilds_the_same_bars_and_shading(self) -> None:
        self.plot(shade_bounds={"left_bound": 2, "right_bound": 5})
        drawn = self.bars()
        state = self.drawables.get_by_class_name("DiscretePlot")[0].get_state()
        self.setUp()
        rebuilt = plot_kind("DiscretePlot").build(state["name"], state["args"])  # type: ignore[union-attr]
        self.drawables.add(rebuilt)
        self.stats.materialize_discrete_plot(rebuilt)
        self.assertEqual(self.bars(), drawn)

    def test_unshaded_plots_save_no_shade_bounds(self) -> None:
        self.plot()
        state = self.drawables.get_by_class_name("DiscretePlot")[0].get_state()
        self.assertNotIn("shade_bounds", state["args"])
        plot = DiscretePlot("X", plot_type="distribution", shade_bounds={"left": 1.0, "right": None})
        from copy import deepcopy

        self.assertEqual(deepcopy(plot).shade_bounds, {"left": 1.0, "right": None})


class TestContinuousDistributions(StatisticsManagerHarness):
    def plot(self, **overrides: Any) -> Dict[str, Any]:
        args = _call(representation="continuous", **overrides)
        return dict(self.stats.plot_distribution(**args))

    def test_each_continuous_type_draws_its_density_over_its_support(self) -> None:
        cases = [
            ("uniform", {"lower": -1, "upper": 3}, (-1.0, 3.0)),
            ("exponential", {"rate": 2}, (0.0, -math.log(0.001) / 2)),
            ("t", {"df": 1}, (-10.0, 10.0)),
            ("chi_squared", {"df": 1}, (0.0, None)),
            ("f", {"df": 3, "df2": 12}, (0.0, None)),
        ]
        for name, params, (left, right) in cases:
            with self.subTest(distribution=name):
                self.setUp()
                result = self.plot(distribution_type=name, distribution_params=params)
                self.assertEqual(result["bounds"]["left"], left)
                if right is not None:
                    self.assertAlmostEqual(result["bounds"]["right"], right, places=9)
                expression, _, drawn_left, drawn_right = self.functions.drawn[0]
                self.assertEqual((drawn_left, drawn_right), (result["bounds"]["left"], result["bounds"]["right"]))
                self.assertEqual(
                    expression, DISTRIBUTIONS[name].pdf_expression(DISTRIBUTIONS[name].parse_params(params))
                )  # type: ignore[misc]

    def test_bounds_left_of_the_support_are_clipped(self) -> None:
        result = self.plot(
            distribution_type="exponential",
            distribution_params={"rate": 1},
            plot_bounds={"left_bound": -2, "right_bound": 5},
        )
        self.assertEqual(result["bounds"], {"left": 0.0, "right": 5.0})
        self.assertIn("plot_bounds clipped to the support of exponential, [0, inf)", result["note"])

    def test_one_sided_shading_counts_the_whole_tail(self) -> None:
        result = self.plot(
            distribution_type="t", distribution_params={"df": 1}, shade_bounds={"left_bound": 2, "right_bound": None}
        )
        self.assertAlmostEqual(result["shaded_probability"], 0.5 - math.atan(2.0) / math.pi, places=11)
        self.assertEqual(result["shaded_range"], "P(X >= 2)")
        self.assertEqual(result["shade_bounds"], {"left": 2.0, "right": 10.0})

    def test_a_shade_beyond_the_default_range_widens_it(self) -> None:
        result = self.plot(
            distribution_type="normal",
            distribution_params={"mean": 0, "sigma": 1},
            shade_bounds={"left_bound": 5, "right_bound": 6},
        )
        self.assertEqual(result["bounds"], {"left": -4.0, "right": 6.0})
        self.assertEqual(result["shade_bounds"], {"left": 5.0, "right": 6.0})

    def test_no_shade_shades_the_drawn_range(self) -> None:
        result = self.plot(distribution_type="uniform", distribution_params={"lower": 0, "upper": 4})
        self.assertEqual(result["shaded_range"], "P(0 <= X <= 4)")
        self.assertEqual(result["shaded_probability"], 1.0)

    def test_bars_of_a_continuous_distribution_highlight_by_midpoint(self) -> None:
        result = self.stats.plot_distribution(
            **_call(
                distribution_type="normal",
                distribution_params={"mean": 0, "sigma": 1},
                plot_bounds={"left_bound": -2, "right_bound": 2},
                bar_count=4,
                shade_bounds={"left_bound": 0, "right_bound": None},
            )
        )
        colours = [state["args"]["fill_color"] for state in self.bar_states()]
        self.assertEqual(colours, [UNSHADED_BAR_FILL_COLOR] * 2 + ["orange"] * 2)
        self.assertEqual(result["shaded_range"], "P(X >= 0)")
        self.assertEqual(result["shaded_probability"], 0.5)

    def test_one_given_bound_past_the_other_default_side_keeps_the_default_width(self) -> None:
        result = self.plot(
            distribution_type="normal", distribution_params=None, plot_bounds={"left_bound": 10, "right_bound": None}
        )
        self.assertEqual(result["bounds"], {"left": 10.0, "right": 18.0})
        result = self.plot(
            distribution_type="normal", distribution_params=None, plot_bounds={"left_bound": None, "right_bound": -5}
        )
        self.assertEqual(result["bounds"], {"left": -13.0, "right": -5.0})

    def test_a_refused_shade_draws_nothing(self) -> None:
        with self.assertRaises(ValueError):
            self.plot(
                distribution_type="normal",
                distribution_params=None,
                plot_bounds={"left_bound": 0, "right_bound": 2},
                shade_bounds={"left_bound": 3, "right_bound": 4},
            )
        self.assertEqual(self.functions.drawn, [])

    def test_df_2_densities_are_defined_at_zero(self) -> None:
        chi = DISTRIBUTIONS["chi_squared"].pdf_expression({"df": 2.0})  # type: ignore[misc]
        f = DISTRIBUTIONS["f"].pdf_expression({"df": 2.0, "df2": 10.0})  # type: ignore[misc]
        self.assertNotIn("ln(x)", chi)
        self.assertNotIn("*ln(x)", f)
        self.assertIn("*ln(x)", DISTRIBUTIONS["chi_squared"].pdf_expression({"df": 3.0}))  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
