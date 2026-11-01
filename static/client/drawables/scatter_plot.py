"""Scatter plot: (x, y) data pairs drawn as dots, not as named points.

``summary`` holds the ranges and Pearson's r (``utils.statistics.data_plots.scatter_summary``).
A trend line is a separate ``fit_regression`` call with ``show_points`` false.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from drawables.data_plot import DataPlot
from utils.statistics.data_plots import scatter_summary


class ScatterPlot(DataPlot):
    """Dots at (``x_data[i]``, ``y_data[i]``); the lists have equal length.

    Attributes:
        x_data: The x values (a tuple of floats).
        y_data: The y values (a tuple of floats).
        summary: The ``ScatterSummary``.
    """

    def __init__(
        self,
        name: str,
        *,
        x_data: Iterable[float],
        y_data: Iterable[float],
        color: Optional[str] = None,
    ) -> None:
        super().__init__(name, plot_type="scatter", color=color)
        self.x_data = tuple(float(v) for v in x_data)
        self.y_data = tuple(float(v) for v in y_data)
        if len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must have the same length")
        self.summary: Dict[str, Any] = dict(scatter_summary(list(self.x_data), list(self.y_data)))

    def get_class_name(self) -> str:
        return "ScatterPlot"

    def extent_points(self) -> List[Tuple[float, float]]:
        summary = self.summary
        return [(summary["x_min"], summary["y_min"]), (summary["x_max"], summary["y_max"])]

    def _plot_args(self) -> Dict[str, Any]:
        return {"x_data": list(self.x_data), "y_data": list(self.y_data), "summary": dict(self.summary)}
