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
from utils.statistics.distributions import DISTRIBUTIONS


class TestPlotTableGuards(unittest.TestCase):
    def test_plot_distribution_offers_every_distribution(self) -> None:
        props = _find_tool("plot_distribution")["function"]["parameters"]["properties"]
        self.assertEqual(props["distribution_type"]["enum"], sorted(DISTRIBUTIONS))

    def test_the_scenario_plot_family_is_every_plot_kind_but_the_legacy_base(self) -> None:
        self.assertEqual(set(PLOT_TYPES), set(PLOT_CLASS_NAMES) - {"Plot"})
        self.assertIs(TYPE_FAMILIES["Plot"], PLOT_TYPES)


if __name__ == "__main__":
    unittest.main()
