"""Histogram: a data list counted into equal-width bins, drawn as touching bars.

The bins and counts come from ``utils.statistics.data_plots`` (bins ``[a, b)``). The plot saves its data and the resolved bins, so a restore draws the same
bars; ``summary`` (counts, heights, values outside the bins) is recomputed on load.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from drawables.data_plot import DataPlot
from utils.statistics.data_plots import Bins, bin_counts, bin_heights


class HistogramPlot(DataPlot):
    """A histogram of ``data`` over ``bin_count`` bins of ``bin_width`` from ``bin_start``.

    Attributes:
        data: The values, as given (a tuple of floats).
        bins: The resolved bins.
        normalize: ``count`` (bar height = count) or ``density`` (count / (n * width)).
        fill_opacity: Bar fill opacity, or None for the default.
        summary: ``counts``, ``heights`` and ``excluded_count`` (values outside every bin).
    """

    _SHARED_ATTRIBUTES = ("summary", "bins")

    def __init__(
        self,
        name: str,
        *,
        data: Iterable[float],
        bins: Bins,
        normalize: str = "count",
        color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
    ) -> None:
        super().__init__(name, plot_type="histogram", color=color)
        self.data = tuple(float(v) for v in data)
        self.bins = bins
        self.normalize = str(normalize)
        self.fill_opacity: Optional[float] = None if fill_opacity is None else float(fill_opacity)
        counts, excluded = bin_counts(list(self.data), bins)
        self.summary: Dict[str, Any] = {
            "counts": tuple(counts),
            "heights": tuple(bin_heights(counts, len(self.data), bins.width, self.normalize)),
            "excluded_count": excluded,
        }

    def get_class_name(self) -> str:
        return "HistogramPlot"

    def edges(self) -> List[float]:
        """The bin edges, ``bin_count + 1`` of them, rounded for display."""
        return [self.bins.edge(i) for i in range(self.bins.count + 1)]

    def extent_points(self) -> List[Tuple[float, float]]:
        top = max(self.summary["heights"]) if self.summary["heights"] else 0.0
        return [(self.bins.edge(0), 0.0), (self.bins.edge(self.bins.count), max(top, 0.0))]

    def _plot_args(self) -> Dict[str, Any]:
        return {
            "data": list(self.data),
            "bin_start": self.bins.start,
            "bin_width": self.bins.width,
            "bin_count": self.bins.count,
            "normalize": self.normalize,
            "fill_opacity": self.fill_opacity,
            "summary": {
                "counts": list(self.summary["counts"]),
                "heights": list(self.summary["heights"]),
                "excluded_count": self.summary["excluded_count"],
            },
        }
