"""The math behind the data plots: histogram bins, box-plot summaries and scatter summaries.

Pure Python like ``descriptive``: the drawables in ``drawables/histogram_plot.py``,
``box_plot.py`` and ``scatter_plot.py`` call these when built, and pytest tests them
directly.

- Histogram bins are half-open ``[a, b)``, as on the TI-84, so integer data with width 1
  gets one bin per value. Bins the caller did not fix always reach past the largest value
  (a value on an edge starts a new bin); only when the caller fixes the bin count is the
  largest value on the right end of the last bin, which then counts it. ``resolve_bins``
  fills in what the caller left out (Sturges' bin count and a "nice" width by default) and
  ``bin_counts`` counts the values in each bin.
- Box plots take their quartiles from ``descriptive`` (median of halves, the TI-84
  method), so a box plot and ``compute_descriptive_statistics`` agree. With outliers
  shown, the fences are 1.5 IQR beyond the quartiles.
- Scatter summaries take Pearson's r from ``two_variable.paired_sums``, as the statistics
  tool and the linear fit do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypedDict

from utils.statistics.descriptive import _compute_median, _compute_quartiles, _validate_data
from utils.statistics.two_variable import paired_sums

MAX_PLOT_DATA_POINTS = 1000
MAX_HISTOGRAM_BINS = 200
NORMALIZE_MODES: Tuple[str, ...] = ("count", "density")
DEFAULT_NORMALIZE = "count"
# A value this close to a bin edge (as a share of the bin width) is on the edge: decimal
# data such as 0.3 with width 0.1 would otherwise fall one bin low through rounding.
_EDGE_TOLERANCE = 1e-9
_NICE_FACTORS: Tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)
_SIGNIFICANT_DIGITS = 12


def round_significant(value: float, digits: int = _SIGNIFICANT_DIGITS) -> float:
    """``value`` rounded to ``digits`` significant digits (0.30000000000000004 -> 0.3)."""
    if value == 0.0 or not math.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def validate_plot_data(data: Any, name: str = "data") -> List[float]:
    """Return ``data`` as floats: a non-empty list of finite numbers, at most 1000 of them."""
    _validate_data(data, name)
    if len(data) > MAX_PLOT_DATA_POINTS:
        raise ValueError(f"{name} has {len(data)} values; a plot takes at most {MAX_PLOT_DATA_POINTS}")
    return [float(v) for v in data]


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bins:
    """Equal-width bins: ``count`` of them from ``start``, each ``width`` wide."""

    start: float
    width: float
    count: int

    def edge(self, index: int) -> float:
        """The left edge of bin ``index`` (``index == count`` gives the right end)."""
        return round_significant(self.start + index * self.width)

    def index_of(self, value: float) -> Optional[int]:
        """The bin holding ``value``, or None when it lies outside every bin."""
        position = _snap((value - self.start) / self.width)
        index = int(math.floor(position))
        if index == self.count and position == float(self.count):
            return self.count - 1  # the right end of the last bin belongs to it
        if index < 0 or index >= self.count:
            return None
        return index


class HistogramBin(TypedDict):
    """One bin of a histogram result."""

    start: float
    end: float
    count: int
    height: float


def _snap(position: float) -> float:
    """``position`` rounded to the nearest whole number when within the edge tolerance of it."""
    if not math.isfinite(position):
        return position
    nearest = round(position)
    if abs(position - nearest) <= _EDGE_TOLERANCE:
        return float(nearest)
    return position


def _whole_bin_count(bin_count: Any) -> int:
    if isinstance(bin_count, bool) or not isinstance(bin_count, (int, float)):
        raise ValueError("bin_count must be a whole number from 1 to 200")
    if not math.isfinite(bin_count) or bin_count != int(bin_count) or not 1 <= bin_count <= MAX_HISTOGRAM_BINS:
        raise ValueError(f"bin_count must be a whole number from 1 to {MAX_HISTOGRAM_BINS}, got {bin_count}")
    return int(bin_count)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def nice_width(raw: float) -> float:
    """The smallest of 1, 2, 2.5 and 5 times a power of ten that is at least ``raw`` (> 0).

    A mantissa within the tolerance of a factor is that factor, so a raw width of
    0.050000000000000044 gives 0.05, not 0.1.
    """
    power = 10.0 ** math.floor(math.log10(raw))
    mantissa = raw / power
    for factor in _NICE_FACTORS:
        if mantissa <= factor * (1.0 + _EDGE_TOLERANCE):
            return round_significant(factor * power)
    return round_significant(10.0 * power)


def sturges_bin_count(n: int) -> int:
    """Sturges' rule: ceil(log2(n)) + 1 bins for n values."""
    return int(math.ceil(math.log2(n))) + 1 if n > 1 else 1


def _aligned_start(low: float, width: float) -> float:
    """The largest multiple of ``width`` at or below ``low`` (``low`` when that overflows)."""
    position = _snap(low / width)
    if not math.isfinite(position):
        return low
    return round_significant(math.floor(position) * width)


def _count_to_cover(start: float, high: float, width: float) -> int:
    """Bins of ``width`` from ``start`` whose half-open span holds ``high``; checks the cap.

    A ``high`` on an edge starts the next bin: [1, 2, ..., 6] with width 1 from 1 gives six
    bins, not five with 5 and 6 sharing the last.
    """
    position = _snap((high - start) / width)
    needed = math.floor(position) + 1.0 if math.isfinite(position) else math.inf
    if needed > MAX_HISTOGRAM_BINS:
        smallest = nice_width((high - start) / (MAX_HISTOGRAM_BINS - 1))
        raise ValueError(
            f"these bins would number {_count_text(needed)}; a histogram has at most {MAX_HISTOGRAM_BINS} bins, "
            f"so give a bin_width of at least {smallest}"
        )
    return max(1, int(needed))


def _count_text(count: float) -> str:
    return f"{count:.0f}" if count < 1e15 else "more than 10^15"


def resolve_bins(
    data: List[float],
    bin_start: Optional[float] = None,
    bin_width: Optional[float] = None,
    bin_count: Optional[Any] = None,
) -> Bins:
    """The bins for ``data`` (validated) from whichever of start, width and count are given.

    Missing parts come from the data: Sturges' count and a nice width, a start aligned to
    the width, and a count that covers the largest value. When the range to cover is 0
    (all values equal, or a start equal to the largest value) the width is 1.
    """
    low, high = min(data), max(data)
    start = None if bin_start is None else _finite_number(bin_start, "bin_start")
    width = None if bin_width is None else _finite_number(bin_width, "bin_width")
    count = None if bin_count is None else _whole_bin_count(bin_count)
    if width is not None and width <= 0:
        raise ValueError(f"bin_width must be greater than 0, got {width}")
    if start is not None and start > high:
        raise ValueError(f"bin_start {start} is above every value (the largest is {high}); give a smaller bin_start")
    lower = low if start is None else start
    span = high - lower
    if not math.isfinite(span):
        raise ValueError("the data's range is too large for a histogram (it overflows)")

    if width is None:
        width = _default_width(span, count, len(data))
        if start is None and count is not None:
            start = low if span > 0 else low - count / 2.0
    if start is None:
        start = _aligned_start(low, width)
    if count is None:
        count = _count_to_cover(start, high, width)
    return Bins(start=float(start), width=float(width), count=int(count))


def _default_width(span: float, count: Optional[int], n: int) -> float:
    """The width when none is given: the span over the count, or a nice Sturges width."""
    if span <= 0:
        return 1.0
    if count is not None:
        return span / count
    return nice_width(span / sturges_bin_count(n))


def bin_counts(data: List[float], bins: Bins) -> Tuple[List[int], int]:
    """(the count of values in each bin, the number of values outside every bin)."""
    counts = [0] * bins.count
    excluded = 0
    for value in data:
        index = bins.index_of(value)
        if index is None:
            excluded += 1
        else:
            counts[index] += 1
    return counts, excluded


def bin_heights(counts: List[int], n: int, width: float, normalize: str) -> List[float]:
    """Bar heights: counts, or densities (share of all ``n`` values per unit of x)."""
    if normalize == "density":
        return [count / (n * width) for count in counts]
    return [float(count) for count in counts]


def resolve_normalize(normalize: Any) -> str:
    """The normalize mode, ``count`` when null; refuses other words."""
    if normalize is None:
        return DEFAULT_NORMALIZE
    mode = str(normalize).strip().lower()
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize must be one of {', '.join(NORMALIZE_MODES)} or null, got {normalize!r}")
    return mode


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------


class BoxSummary(TypedDict):
    """The numbers a box plot draws. Fences are None when outliers are not shown."""

    n: int
    min: float
    q1: float
    median: float
    q3: float
    max: float
    iqr: float
    lower_fence: Optional[float]
    upper_fence: Optional[float]
    whisker_low: float
    whisker_high: float
    outliers: List[float]


def box_summary(data: List[float], show_outliers: bool = True) -> BoxSummary:
    """Five-number summary, whisker ends and outliers of ``data`` (validated).

    With ``show_outliers``, values more than 1.5 IQR beyond a quartile are outliers and
    the whiskers reach the most extreme values inside the fences (TI-84 ModBoxplot);
    otherwise the whiskers reach the minimum and maximum (Boxplot).
    """
    ordered = sorted(data)
    q1, q3 = _compute_quartiles(ordered)
    iqr = q3 - q1
    low, high = ordered[0], ordered[-1]
    lower_fence: Optional[float] = None
    upper_fence: Optional[float] = None
    outliers: List[float] = []
    whisker_low, whisker_high = low, high
    if show_outliers:
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # A value on a fence is inside it, also when the fence is a rounding away from it
        slack = _EDGE_TOLERANCE * max(abs(q1), abs(q3), iqr)
        inside = [v for v in ordered if lower_fence - slack <= v <= upper_fence + slack]
        outliers = [v for v in ordered if v < lower_fence - slack or v > upper_fence + slack]
        whisker_low, whisker_high = inside[0], inside[-1]
    return BoxSummary(
        n=len(ordered),
        min=low,
        q1=q1,
        median=_compute_median(ordered),
        q3=q3,
        max=high,
        iqr=iqr,
        lower_fence=lower_fence,
        upper_fence=upper_fence,
        whisker_low=whisker_low,
        whisker_high=whisker_high,
        outliers=outliers,
    )


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------


class ScatterSummary(TypedDict):
    """Ranges of a scatter plot's data and Pearson's r (None for one pair or a constant list)."""

    n: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    r: Optional[float]


def validate_scatter_data(x_data: Any, y_data: Any) -> Tuple[List[float], List[float]]:
    """Both lists validated as plot data, of equal length."""
    xs = validate_plot_data(x_data, "x_data")
    ys = validate_plot_data(y_data, "y_data")
    if len(xs) != len(ys):
        raise ValueError(f"x_data and y_data must have the same length ({len(xs)} and {len(ys)})")
    return xs, ys


def scatter_summary(x_data: List[float], y_data: List[float]) -> ScatterSummary:
    """Ranges and Pearson's r of validated, equally long lists."""
    r = paired_sums(x_data, y_data).r if len(x_data) >= 2 else None
    return ScatterSummary(
        n=len(x_data),
        x_min=min(x_data),
        x_max=max(x_data),
        y_min=min(y_data),
        y_max=max(y_data),
        r=r,
    )
