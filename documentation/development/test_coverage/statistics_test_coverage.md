# Statistics test coverage matrix

This document tracks test coverage for the statistics plotting stack (distributions, plots, bars).
It is intended to make it easy to see what is covered today and what branches still need tests.

## Scope

1. AI tool schemas: `plot_distribution`, `plot_bars`, `delete_plot`
2. Client dispatch: tool name -> `Canvas` -> `DrawableManager` -> `StatisticsManager`
3. Managers: `StatisticsManager`, `BarManager`
4. Drawables: `Bar`, `Plot` and subclasses (`ContinuousPlot`, `DiscretePlot`, `BarsPlot`)
5. Workspace restore: plot composites and derived components
6. Rendering: `Bar` rendering helper

## Coverage matrix

| Area | Entry point / behavior | Code | Tests | Notes |
| --- | --- | --- | --- | --- |
| Tool schema | Plot tool schemas exist and remain stable | `static/functions_definitions.py` | `server_tests/test_plot_tool_schemas.py` | Covered (schema drift guard) |
| Client dispatch | Tool mapping includes plot tools | `static/client/function_registry.py` | Indirect via plot tests | Gap: add a direct tool-call integration test using `ProcessFunctionCalls.get_results` |
| Plot: continuous | Creates `ContinuousPlot` + `Function` + `FunctionsBoundedColoredArea` | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_distribution_continuous_creates_components` | Covered with `draw_enabled=False` |
| Plot: continuous | Deletes plot and its components | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_delete_plot_continuous_removes_components` | Covered |
| Plot: continuous | Rejects invalid sigma | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_distribution_rejects_invalid_sigma` | Gap: add more validation branches (representation/type/bounds) |
| Plot: continuous | Generates unique plot names | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_distribution_generates_unique_plot_names` | Covered |
| Plot: discrete | Creates `DiscretePlot` + `Bar` drawables | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_distribution_discrete_creates_bars` | Covered |
| Plot: discrete | Deletes plot and derived bars | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_delete_plot_discrete_removes_bars` | Covered |
| Bars plot | Creates `BarsPlot` + derived `Bar` drawables | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_bars_creates_components_and_labels` | Covered (labels and naming) |
| Bars plot | Bar spacing affects derived bar positions | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_plot_bars_spacing_affects_positions` | Covered |
| Bars plot | Deletes plot and derived bars | `static/client/managers/statistics_manager.py` | `static/client/client_tests/test_statistics_manager.py::test_delete_plot_bars_removes_components` | Covered |
| BarManager | Creates bars, swaps x bounds, rejects zero width/height, deletes bars | `static/client/managers/bar_manager.py` | `static/client/client_tests/test_bar_manager.py` | Gap: update-existing-bar path, nan/inf validation |
| Distributions utils | `normal_pdf_expression` tokens and sigma validation | `static/client/utils/statistics/distributions.py` | `static/client/client_tests/test_statistics_distributions.py` | Gap: non-finite inputs, default bounds validation branches |
| Drawables: state | `Bar` state, deepcopy, translation | `static/client/drawables/bar.py` | (none) | Gap: add pure-python unit tests for drawables state/deepcopy |
| Drawables: state | Plot composite states and deepcopy | `static/client/drawables/*_plot.py` | (none) | Gap: add pure-python unit tests for plot composite serialization |
| Workspace restore | Restores plot composites from workspace and materializes derived bars | `static/client/workspace_manager.py` and `StatisticsManager.materialize_*` | (none) | Gap: add workspace restore tests for plots and post-restore deletion |
| Rendering | Bar renders fill polygon and labels correctly | `static/client/rendering/helpers/bar_renderer.py` | (none) | Gap: add renderer helper tests (primitives + label behavior + guard clauses) |


