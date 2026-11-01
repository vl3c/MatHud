"""Rendering helpers for the data plots that draw themselves: histograms, box plots, scatter plots.

- A histogram draws each non-empty bin as a bar through ``render_bar_helper``, outlined so
  touching bins stay apart, with its count (or density) above when there are at most
  ``MAX_LABELLED_BINS`` bins. An empty bin draws nothing.
- A box plot draws the box from Q1 to Q3, the median line, the whiskers with end caps
  half the box's height, and each outlier as a dot of the point radius.
- A scatter plot draws each pair as a dot of the point radius.

Dots are screen-space circles, so they keep their size when the view zooms.
"""

from __future__ import annotations

from rendering.helpers.bar_renderer import render_bar_helper
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FillStyle, StrokeStyle

MAX_LABELLED_BINS = 30
_HISTOGRAM_OUTLINE = "#333333"
_DEFAULT_FILL = "lightblue"
_HISTOGRAM_OPACITY = 0.6
_BOX_OPACITY = 0.3


class _BinBar:
    """The attributes ``render_bar_helper`` reads, for one histogram bin."""

    def __init__(self, x_left, x_right, height, fill_color, fill_opacity, label):
        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = 0.0
        self.y_top = height
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.color = _HISTOGRAM_OUTLINE
        self.label_above_text = label
        self.label_text = label
        self.label_below_text = None


def _format_height(value, normalize):
    if normalize == "density":
        return f"{value:.3g}"
    return str(int(round(value)))


def render_histogram_plot_helper(primitives, plot, coordinate_mapper, style):
    """Draw a histogram plot's bins as outlined bars with optional value labels."""
    try:
        bins = plot.bins
        heights = plot.summary["heights"]
        counts = plot.summary["counts"]
    except Exception:
        return
    fill_color = getattr(plot, "plot_color", None) or _DEFAULT_FILL
    opacity = getattr(plot, "fill_opacity", None)
    if opacity is None:
        opacity = _HISTOGRAM_OPACITY
    labelled = bins.count <= MAX_LABELLED_BINS
    normalize = getattr(plot, "normalize", "count")
    for index in range(bins.count):
        if counts[index] == 0:
            continue
        label = _format_height(heights[index], normalize) if labelled else None
        bar = _BinBar(bins.edge(index), bins.edge(index + 1), heights[index], fill_color, opacity, label)
        render_bar_helper(primitives, bar, coordinate_mapper, style)


def _stroke(plot, style):
    color = getattr(plot, "plot_color", None) or style.get("segment_color", "#000")
    return StrokeStyle(color=str(color), width=float(style.get("segment_stroke_width", 1) or 1))


def _dot_radius(style):
    try:
        return float(style.get("point_radius", 0) or 0)
    except Exception:
        return 0.0


@_manages_shape
def _draw_lines(primitives, lines, stroke):
    for start, end in lines:
        primitives.stroke_line(start, end, stroke, include_width=False)


@_manages_shape
def _draw_dots(primitives, centers, radius, fill):
    for center in centers:
        primitives.fill_circle(center, radius, fill, screen_space=True)


def render_box_plot_helper(primitives, plot, coordinate_mapper, style):
    """Draw a box plot: box, median, whiskers with caps, and outlier dots."""
    try:
        summary = plot.summary
        y_mid = float(plot.y_position)
        y_low = float(plot.y_bottom)
        y_high = float(plot.y_top)
        q1, median, q3 = float(summary["q1"]), float(summary["median"]), float(summary["q3"])
        w_low, w_high = float(summary["whisker_low"]), float(summary["whisker_high"])
        cap_low = y_mid - (y_high - y_low) / 4.0
        cap_high = y_mid + (y_high - y_low) / 4.0
        to_screen = coordinate_mapper.math_to_screen
        corners = [to_screen(q1, y_low), to_screen(q3, y_low), to_screen(q3, y_high), to_screen(q1, y_high)]
        lines = [
            (to_screen(median, y_low), to_screen(median, y_high)),
            (to_screen(w_low, y_mid), to_screen(q1, y_mid)),
            (to_screen(q3, y_mid), to_screen(w_high, y_mid)),
            (to_screen(w_low, cap_low), to_screen(w_low, cap_high)),
            (to_screen(w_high, cap_low), to_screen(w_high, cap_high)),
        ]
        outliers = [to_screen(float(value), y_mid) for value in summary["outliers"]]
    except Exception:
        return
    stroke = _stroke(plot, style)
    opacity = getattr(plot, "fill_opacity", None)
    fill_color = getattr(plot, "plot_color", None) or _DEFAULT_FILL
    fill = FillStyle(color=str(fill_color), opacity=_BOX_OPACITY if opacity is None else opacity)
    primitives.fill_polygon(corners, fill, stroke, screen_space=True)
    _draw_lines(primitives, lines, stroke)
    radius = _dot_radius(style)
    if outliers and radius > 0:
        _draw_dots(primitives, outliers, radius, FillStyle(color=stroke.color, opacity=None))


def render_scatter_plot_helper(primitives, plot, coordinate_mapper, style):
    """Draw a scatter plot's pairs as dots."""
    radius = _dot_radius(style)
    if radius <= 0:
        return
    try:
        to_screen = coordinate_mapper.math_to_screen
        centers = [to_screen(x, y) for x, y in zip(plot.x_data, plot.y_data)]
    except Exception:
        return
    color = getattr(plot, "plot_color", None) or style.get("point_color", "#000")
    _draw_dots(primitives, centers, radius, FillStyle(color=str(color), opacity=None))
