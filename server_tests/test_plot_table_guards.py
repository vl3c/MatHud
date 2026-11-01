"""Lists kept by hand elsewhere follow the plot-kind and distribution tables.

The ``plot_distribution`` schema is model-facing and written out in full, and the scenario
grader's ``Plot`` selector family is a literal, so neither reads the tables; these tests make
a new row fail until they are updated with it.
"""

from __future__ import annotations

import unittest

from cli.scenarios.geometry import PLOT_TYPES, TYPE_FAMILIES
from plot_kinds import PLOT_CLASS_NAMES
from server_tests.test_plot_tool_schemas import _find_tool
from static.canvas_state_formatter import DISCRETE_DISTRIBUTION_TYPES
from utils.statistics.distributions import DISTRIBUTIONS, SHARED_PARAM_NAMES


class TestPlotTableGuards(unittest.TestCase):
    def test_plot_distribution_offers_every_distribution(self) -> None:
        props = _find_tool("plot_distribution")["function"]["parameters"]["properties"]
        self.assertEqual(props["distribution_type"]["enum"], sorted(DISTRIBUTIONS))

    def test_the_distribution_tools_share_the_type_enum_and_params(self) -> None:
        plot = _find_tool("plot_distribution")["function"]["parameters"]["properties"]
        compute = _find_tool("compute_distribution")["function"]["parameters"]["properties"]
        self.assertEqual(compute["distribution_type"]["enum"], plot["distribution_type"]["enum"])
        self.assertEqual(compute["distribution_params"], plot["distribution_params"])
        self.assertIsNot(compute["distribution_params"], plot["distribution_params"])

    def test_the_params_object_is_the_shared_param_names(self) -> None:
        props = _find_tool("plot_distribution")["function"]["parameters"]["properties"]
        self.assertEqual(props["distribution_params"]["required"], list(SHARED_PARAM_NAMES))

    def test_the_formatter_knows_the_discrete_distributions(self) -> None:
        discrete = {name for name, spec in DISTRIBUTIONS.items() if spec.discrete}
        self.assertEqual(set(DISCRETE_DISTRIBUTION_TYPES), discrete)

    def test_the_scenario_plot_family_is_every_plot_kind_but_the_legacy_base(self) -> None:
        self.assertEqual(set(PLOT_TYPES), set(PLOT_CLASS_NAMES) - {"Plot"})
        self.assertIs(TYPE_FAMILIES["Plot"], PLOT_TYPES)


if __name__ == "__main__":
    unittest.main()
