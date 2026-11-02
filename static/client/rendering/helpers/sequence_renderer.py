"""Rendering helpers for sequence plots and cobweb diagrams, which draw themselves.

- A sequence plot draws a dot of the point radius at each (n, u(n)).
- A cobweb draws y = g(x) over its x range (through the parametric curve helper, so the
  curve is sampled adaptively and breaks where g is undefined), the diagonal y = x over the
  same range, the web as one polyline, and a dot where the web starts, at (x0, 0).

Dots are screen-space circles, so they keep their size when the view zooms.
"""

from __future__ import annotations

from rendering.helpers.parametric_function_renderer import render_parametric_function_helper
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FillStyle, StrokeStyle

_DEFAULT_WEB_COLOR = "#d62728"
_DIAGONAL_COLOR = "#888888"


def _dot_radius(style):
    try:
        return float(style.get("point_radius", 0) or 0)
    except Exception:
        return 0.0


@_manages_shape
def _draw_dots(primitives, centers, radius, fill):
    for center in centers:
        primitives.fill_circle(center, radius, fill, screen_space=True)


@_manages_shape
def _draw_polyline(primitives, points, stroke):
    primitives.stroke_polyline(points, stroke)


def render_sequence_plot_helper(primitives, plot, coordinate_mapper, style):
    """Draw a sequence plot's terms as dots at (n, u(n))."""
    radius = _dot_radius(style)
    if radius <= 0:
        return
    try:
        to_screen = coordinate_mapper.math_to_screen
        centers = [to_screen(n, value) for n, value in plot.points()]
    except Exception:
        return
    color = getattr(plot, "plot_color", None) or style.get("point_color", "#000")
    _draw_dots(primitives, centers, radius, FillStyle(color=str(color), opacity=None))


def render_cobweb_plot_helper(primitives, plot, coordinate_mapper, style):
    """Draw a cobweb: the curve y = g(x), the diagonal y = x, the web and its starting dot."""
    try:
        x_min, x_max = plot.x_range
        to_screen = coordinate_mapper.math_to_screen
        diagonal = [to_screen(x_min, x_min), to_screen(x_max, x_max)]
        web = [to_screen(x, y) for x, y in plot.web_points()]
    except Exception:
        return
    curve = plot.curve
    curve.color = str(style.get("function_color", "#000"))
    render_parametric_function_helper(primitives, curve, coordinate_mapper, style)
    width = float(style.get("segment_stroke_width", 1) or 1)
    _draw_polyline(primitives, diagonal, StrokeStyle(color=_DIAGONAL_COLOR, width=width))
    color = str(getattr(plot, "plot_color", None) or _DEFAULT_WEB_COLOR)
    if len(web) >= 2:
        _draw_polyline(primitives, web, StrokeStyle(color=color, width=width, line_join="round"))
    radius = _dot_radius(style)
    if radius > 0:
        _draw_dots(primitives, web[:1], radius, FillStyle(color=color, opacity=None))
