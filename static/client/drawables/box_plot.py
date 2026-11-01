"""Box plot: a horizontal box-and-whisker plot of a data list.

The quartiles are those of ``compute_descriptive_statistics`` (median of halves); with
``show_outliers`` the whiskers stop at the 1.5 IQR fences and the values beyond are drawn
as dots (``utils.statistics.data_plots.box_summary``). The box spans Q1..Q3 in x and
``y_position`` +/- ``box_height`` / 2 in y.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from drawables.data_plot import DataPlot
from utils.statistics.data_plots import box_summary


class BoxPlot(DataPlot):
    """A box plot of ``data`` centred on the line y = ``y_position``.

    Attributes:
        data: The values, as given (a tuple of floats).
        y_position: The y of the box's centre line.
        box_height: The box's height in y units.
        show_outliers: Whether values beyond the 1.5 IQR fences are drawn as outliers.
        fill_opacity: Box fill opacity, or None for the default.
        summary: The ``BoxSummary`` (outliers as a tuple).
    """

    def __init__(
        self,
        name: str,
        *,
        data: Iterable[float],
        y_position: float,
        box_height: float,
        show_outliers: bool = True,
        color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
    ) -> None:
        super().__init__(name, plot_type="box", color=color)
        self.data = tuple(float(v) for v in data)
        self.y_position = float(y_position)
        self.box_height = float(box_height)
        self.show_outliers = bool(show_outliers)
        self.fill_opacity: Optional[float] = None if fill_opacity is None else float(fill_opacity)
        summary: Dict[str, Any] = dict(box_summary(list(self.data), self.show_outliers))
        summary["outliers"] = tuple(summary["outliers"])
        self.summary = summary

    def get_class_name(self) -> str:
        return "BoxPlot"

    @property
    def y_bottom(self) -> float:
        return self.y_position - self.box_height / 2.0

    @property
    def y_top(self) -> float:
        return self.y_position + self.box_height / 2.0

    def extent_points(self) -> List[Tuple[float, float]]:
        low = min([self.summary["whisker_low"], *self.summary["outliers"]])
        high = max([self.summary["whisker_high"], *self.summary["outliers"]])
        return [(low, self.y_bottom), (high, self.y_top)]

    def _plot_args(self) -> Dict[str, Any]:
        summary = dict(self.summary)
        summary["outliers"] = list(summary["outliers"])
        return {
            "data": list(self.data),
            "y_position": self.y_position,
            "box_height": self.box_height,
            "show_outliers": self.show_outliers,
            "fill_opacity": self.fill_opacity,
            "summary": summary,
        }
