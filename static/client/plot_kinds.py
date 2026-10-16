"""The plot kinds: one row per plot class, read by every site that handles plots by kind.

A plot is a non-renderable composite that names or derives the drawables it is drawn with.
Each row says how a plot of that class is saved and rebuilt:

- ``bucket``: the saved-state list it is stored in (its class name plus "s").
- ``build``: makes the plot from a saved item's name and args.
- ``materialize``: the ``StatisticsManager`` method that rebuilds its derived parts after a
  restore (they are not saved), or None.
- ``delete_parts``: the ``StatisticsManager`` method that deletes the drawables it is drawn
  with, or None.
- ``derived_bars``: whether it owns Bars named ``derived_bar_name(name, i)``, which saves
  and canvas states leave out because the plot rebuilds them.

The methods are named, not referenced, because ``StatisticsManager`` imports this table
(``server_tests/test_plot_kinds.py`` checks the names). Rows are in restore order:
materialization and Bar pruning follow it too. The base ``Plot`` row reads the ``Plots``
list that workspaces saved before the plot classes were split; its builder returns the
subclass the saved parts call for.

Model-facing text (``static/canvas_state_formatter.py``, ``static/canvas_state_summarizer.py``)
describes each kind by hand and does not read this table. A new plot kind needs a row here,
its bucket in ``WorkspaceManager.RESTORED_BUCKETS`` and a formatter spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot


@dataclass(frozen=True)
class PlotKind:
    """How one plot class is saved, rebuilt, materialized and deleted."""

    class_name: str
    bucket: str
    build: Callable[[str, Dict[str, Any]], Plot]
    materialize: Optional[str]
    delete_parts: Optional[str]
    derived_bars: bool


def derived_bar_prefix(plot_name: str) -> str:
    """The name prefix of every Bar a plot derives."""
    return f"{plot_name}_bar_"


def derived_bar_name(plot_name: str, index: int) -> str:
    """The name of a plot's derived Bar number ``index``."""
    return f"{derived_bar_prefix(plot_name)}{index}"


def is_legacy_discrete_args(args: Dict[str, Any]) -> bool:
    """Whether a legacy ``Plots`` item names the rectangles or areas of a discrete plot."""
    return bool(args.get("rectangle_names") or args.get("fill_area_names"))


def is_legacy_continuous_args(args: Dict[str, Any]) -> bool:
    """Whether a legacy ``Plots`` item names the function or area of a continuous plot."""
    return bool(args.get("function_name") or args.get("fill_area_name"))


def _plot_type(args: Dict[str, Any]) -> str:
    return str(args.get("plot_type", "") or "distribution")


def _build_continuous(name: str, args: Dict[str, Any]) -> Plot:
    return ContinuousPlot(
        name,
        plot_type=_plot_type(args),
        distribution_type=args.get("distribution_type", None),
        function_name=args.get("function_name"),
        fill_area_name=args.get("fill_area_name"),
        distribution_params=args.get("distribution_params", None),
        bounds=args.get("bounds", None),
        metadata=args.get("metadata", None),
    )


def _build_discrete(name: str, args: Dict[str, Any]) -> Plot:
    return DiscretePlot(
        name,
        plot_type=_plot_type(args),
        distribution_type=args.get("distribution_type", None),
        bar_count=args.get("bar_count"),
        bar_labels=args.get("bar_labels"),
        curve_color=args.get("curve_color"),
        fill_color=args.get("fill_color"),
        fill_opacity=args.get("fill_opacity"),
        rectangle_names=args.get("rectangle_names"),
        fill_area_names=args.get("fill_area_names"),
        distribution_params=args.get("distribution_params", None),
        bounds=args.get("bounds", None),
        metadata=args.get("metadata", None),
    )


def _build_bars(name: str, args: Dict[str, Any]) -> Plot:
    # A bar chart has no distribution: saved distribution fields are ignored.
    return BarsPlot(
        name,
        plot_type=_plot_type(args),
        values=args.get("values") or [],
        labels_below=args.get("labels_below") or [],
        labels_above=args.get("labels_above"),
        bar_spacing=args.get("bar_spacing"),
        bar_width=args.get("bar_width"),
        x_start=args.get("x_start"),
        y_base=args.get("y_base"),
        stroke_color=args.get("stroke_color"),
        fill_color=args.get("fill_color"),
        fill_opacity=args.get("fill_opacity"),
        bounds=args.get("bounds", None),
        metadata=args.get("metadata", None),
    )


def _build_legacy(name: str, args: Dict[str, Any]) -> Plot:
    # A Plot saved before the subclasses existed: rebuild the subclass its parts call for.
    if is_legacy_discrete_args(args):
        return _build_discrete(name, args)
    if is_legacy_continuous_args(args):
        return _build_continuous(name, args)
    return Plot(
        name,
        plot_type=_plot_type(args),
        distribution_type=args.get("distribution_type", None),
        distribution_params=args.get("distribution_params", None),
        bounds=args.get("bounds", None),
        metadata=args.get("metadata", None),
    )


PLOT_KINDS: Tuple[PlotKind, ...] = (
    PlotKind("ContinuousPlot", "ContinuousPlots", _build_continuous, None, "_delete_continuous_plot", False),
    PlotKind(
        "DiscretePlot",
        "DiscretePlots",
        _build_discrete,
        "materialize_discrete_plot",
        "_delete_discrete_plot",
        True,
    ),
    PlotKind("BarsPlot", "BarsPlots", _build_bars, "materialize_bars_plot", "_delete_bars_plot", True),
    PlotKind("Plot", "Plots", _build_legacy, None, None, False),
)

PLOT_CLASS_NAMES: Tuple[str, ...] = tuple(kind.class_name for kind in PLOT_KINDS)


def plot_kind(class_name: str) -> Optional[PlotKind]:
    """The row for a plot class name, or None when it is not a plot class."""
    for kind in PLOT_KINDS:
        if kind.class_name == class_name:
            return kind
    return None


def build_saved_plot(kind: PlotKind, item_state: Dict[str, Any]) -> Tuple[Plot, str]:
    """Build the plot a saved item of ``kind``'s bucket describes; return it and its name."""
    raw_args = item_state.get("args", {})
    args = raw_args if isinstance(raw_args, dict) else {}
    name = str(item_state.get("name", "") or "")
    return kind.build(name, args), name
