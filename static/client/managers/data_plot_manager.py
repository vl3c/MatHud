"""The data-plot tools: plot_histogram, plot_box and plot_scatter.

Each tool validates its data, builds one self-drawing plot (``drawables/data_plot.py``),
adds it as one undo step and returns the numbers the plot shows. The tools are value tools
(``tool_kinds.VALUE_TOOLS``), so their results reach the model in full, and they archive
their own undo step, as ``fit_regression`` does. The function registry binds them to this
manager directly; ``delete_plot`` (``StatisticsManager``) deletes the plots, which have no
parts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.box_plot import BoxPlot
from drawables.data_plot import DataPlot
from drawables.histogram_plot import HistogramPlot
from drawables.scatter_plot import ScatterPlot
from utils.statistics.data_plots import (
    box_summary,
    resolve_bins,
    resolve_normalize,
    round_significant,
    validate_plot_data,
    validate_scatter_data,
)

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from name_generator.drawable import DrawableNameGenerator

# Outliers listed in a box plot's result; the count says how many there are in all.
MAX_LISTED_OUTLIERS = 20
# A default box is this share of its whisker span tall, and stacks this share of its
# height below the axis and below the lowest existing box.
BOX_HEIGHT_SHARE = 0.2
BOX_GAP_SHARE = 0.25


class DataPlotManager:
    """Creates histograms, box plots and scatter plots (see the module docstring)."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.name_generator = name_generator

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def plot_histogram(
        self,
        *,
        data: List[float],
        name: Optional[str] = None,
        bin_start: Optional[float] = None,
        bin_width: Optional[float] = None,
        bin_count: Optional[float] = None,
        normalize: Optional[str] = None,
        color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Bin ``data`` and draw a histogram; return the bins, counts and any values left out."""
        values = validate_plot_data(data)
        bins = resolve_bins(values, bin_start, bin_width, bin_count)
        plot = HistogramPlot(
            self._plot_name(name, "histogram"),
            data=values,
            bins=bins,
            normalize=resolve_normalize(normalize),
            color=_optional_text(color),
            fill_opacity=_opacity(fill_opacity),
        )
        self._add(plot)
        return self._histogram_result(plot)

    def plot_box(
        self,
        *,
        data: List[float],
        name: Optional[str] = None,
        y_position: Optional[float] = None,
        box_height: Optional[float] = None,
        show_outliers: Optional[bool] = None,
        color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Draw a horizontal box plot of ``data``; return its five-number summary and outliers."""
        values = validate_plot_data(data)
        outliers_shown = True if show_outliers is None else bool(show_outliers)
        height = self._box_height(values, outliers_shown, box_height)
        position = self._box_position(height, y_position)
        plot = BoxPlot(
            self._plot_name(name, "box_plot"),
            data=values,
            y_position=position,
            box_height=height,
            show_outliers=outliers_shown,
            color=_optional_text(color),
            fill_opacity=_opacity(fill_opacity),
        )
        self._add(plot)
        return self._box_result(plot)

    def plot_scatter(
        self,
        *,
        x_data: List[float],
        y_data: List[float],
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draw the (x, y) pairs as dots; return their ranges and Pearson's r."""
        xs, ys = validate_scatter_data(x_data, y_data)
        plot = ScatterPlot(self._plot_name(name, "scatter_plot"), x_data=xs, y_data=ys, color=_optional_text(color))
        self._add(plot)
        result: Dict[str, Any] = {"plot_name": plot.name, "plot_type": "scatter"}
        result.update(plot.summary)
        if plot.summary["r"] is None:
            result["note"] = (
                "r is undefined for a single pair" if plot.summary["n"] < 2 else "r is undefined: x or y is constant"
            )
        return result

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @staticmethod
    def _histogram_result(plot: HistogramPlot) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "plot_name": plot.name,
            "plot_type": "histogram",
            "n": len(plot.data),
            "bin_start": plot.bins.edge(0),
            "bin_width": round_significant(plot.bins.width),
            "bin_count": plot.bins.count,
            "normalize": plot.normalize,
            "edges": plot.edges(),
            "counts": list(plot.summary["counts"]),
        }
        if plot.normalize == "density":
            result["heights"] = list(plot.summary["heights"])
        excluded = plot.summary["excluded_count"]
        if excluded:
            result["excluded_count"] = excluded
            result["note"] = (
                f"{excluded} of the {len(plot.data)} values lie outside the bins and are not counted; "
                "move bin_start or give more bins to include them"
            )
        return result

    @staticmethod
    def _box_result(plot: BoxPlot) -> Dict[str, Any]:
        summary = plot.summary
        result: Dict[str, Any] = {"plot_name": plot.name, "plot_type": "box"}
        for key in ("n", "min", "q1", "median", "q3", "max", "iqr", "whisker_low", "whisker_high"):
            result[key] = summary[key]
        result["y_position"] = plot.y_position
        result["box_height"] = plot.box_height
        if plot.show_outliers:
            outliers = list(summary["outliers"])
            result["lower_fence"] = summary["lower_fence"]
            result["upper_fence"] = summary["upper_fence"]
            result["outlier_count"] = len(outliers)
            result["outliers"] = outliers[:MAX_LISTED_OUTLIERS]
            if len(outliers) > MAX_LISTED_OUTLIERS:
                result["note"] = f"the {MAX_LISTED_OUTLIERS} smallest of {len(outliers)} outliers are listed"
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _box_height(values: List[float], show_outliers: bool, box_height: Optional[float]) -> float:
        """The given height, or a share of the whisker span (1 when the span is 0)."""
        if box_height is not None:
            height = _finite(box_height, "box_height")
            if height <= 0:
                raise ValueError(f"box_height must be greater than 0, got {box_height}")
            return height
        summary = box_summary(values, show_outliers)
        span = summary["whisker_high"] - summary["whisker_low"]
        return BOX_HEIGHT_SHARE * span if span > 0 and math.isfinite(span) else 1.0

    def _box_position(self, height: float, y_position: Optional[float]) -> float:
        """The given centre line, or one just below the x axis and below every existing box."""
        if y_position is not None:
            return _finite(y_position, "y_position")
        lowest = 0.0
        for box in self.drawables.get_by_class_name("BoxPlot"):
            lowest = min(lowest, float(getattr(box, "y_bottom", 0.0)))
        return lowest - BOX_GAP_SHARE * height - height / 2.0

    def _plot_name(self, name: Optional[str], default: str) -> str:
        base = self.name_generator.filter_string(name or "") or default
        return str(self.drawables.unique_name(base))

    def _add(self, plot: DataPlot) -> None:
        """Add the plot as one undo step and redraw."""
        self.canvas.undo_redo_manager.archive()
        self.drawables.add(plot)
        if getattr(self.canvas, "draw_enabled", False):
            self.canvas.draw()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opacity(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    opacity = _finite(value, "fill_opacity")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"fill_opacity must be from 0 to 1, got {value}")
    return opacity
