"""
Implicit Curve and Inequality Region Rendering Helpers for MatHud

render_implicit_curve_helper strokes the curve of an equation in x and y;
render_inequality_area_helper fills an inequality region and strokes its boundary,
dashed where the inequality is strict (<, >) and solid where it is not (<=, >=).

Both plot the view plus the renderer's function_view_margin of a screen on each side
(ImplicitRenderable). A region is filled as one polygon: its rings, outer ones
counter-clockwise and holes clockwise, joined by bridges the path walks back along, so
each bridge cancels out of the non-zero winding rule and translucent fills show no seams.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

from constants import default_font_family
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FillStyle, FontStyle, StrokeStyle, TextAlignment
from rendering.renderables.implicit_renderable import ImplicitRenderable

Point = Tuple[float, float]

# Dash pattern of a strict boundary, in screen pixels
DASH_ON: float = 6.0
DASH_OFF: float = 4.0


def _view_margin(style: Any) -> float:
    try:
        margin = float(style.get("function_view_margin", 0.0) or 0.0)
    except Exception:
        return 0.0
    return margin if math.isfinite(margin) and margin > 0 else 0.0


def _renderable(model: Any, coordinate_mapper: Any) -> ImplicitRenderable:
    renderable = getattr(model, "_renderable", None)
    if not isinstance(renderable, ImplicitRenderable):
        renderable = ImplicitRenderable(model, coordinate_mapper)
        try:
            model._renderable = renderable
        except Exception:
            pass
    else:
        renderable.mapper = coordinate_mapper
    return renderable


def _stroke(model: Any, style: Any) -> StrokeStyle:
    return StrokeStyle(
        color=str(getattr(model, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
        line_join="round",
    )


def dashed(path: Sequence[Point], on: float = DASH_ON, off: float = DASH_OFF) -> List[List[Point]]:
    """The dashes of a polyline: pieces `on` long with gaps `off` long, measured along it."""
    pieces: List[List[Point]] = []
    current: List[Point] = []
    drawing = True
    left = on
    for index in range(1, len(path)):
        x0, y0 = path[index - 1]
        x1, y1 = path[index]
        length = math.hypot(x1 - x0, y1 - y0)
        start = 0.0
        if drawing and not current:
            current.append((x0, y0))
        while length - start > left:
            start += left
            point = (x0 + (x1 - x0) * start / length, y0 + (y1 - y0) * start / length)
            if drawing:
                current.append(point)
                pieces.append(current)
                current = []
            else:
                current = [point]
            drawing = not drawing
            left = on if drawing else off
        left -= length - start
        if drawing:
            current.append((x1, y1))
    if drawing and len(current) >= 2:
        pieces.append(current)
    return pieces


def bridged_polygon(rings: Sequence[Sequence[Point]]) -> List[Point]:
    """One polygon tracing every ring: R1, R2, ..., Rk, then back through Rk-1[0], ..., R1[0].

    Each ring is closed back to its first point before the next starts, and the walk back
    retraces the bridges between first points, so every bridge is crossed once each way
    and adds nothing to the winding number.
    """
    usable = [list(ring) for ring in rings if len(ring) >= 3]
    polygon: List[Point] = []
    for ring in usable:
        polygon.extend(ring)
        polygon.append(ring[0])
    for ring in reversed(usable[:-1]):
        polygon.append(ring[0])
    return polygon


@_manages_shape
def _stroke_paths(primitives: Any, paths: Sequence[Sequence[Point]], stroke: StrokeStyle) -> None:
    for path in paths:
        if len(path) >= 2:
            primitives.stroke_polyline(list(path), stroke)


def _render_label(primitives: Any, model: Any, paths: Sequence[Sequence[Point]], color: str, style: Any) -> None:
    """The name near the leftmost point of the curve that is on screen."""
    name = getattr(model, "name", "")
    if not name:
        return
    font_size = style.get("function_label_font_size", 12)
    try:
        font_size = float(font_size)
    except Exception:
        font_size = 12.0
    if font_size.is_integer():
        font_size = int(font_size)
    canvas = style.get("_canvas_size")
    # The leftmost point on screen: it stays put while panning, unlike the first point of a path
    best: Any = None
    for path in paths:
        for x, y in path:
            if (canvas is None or (0 <= x <= canvas[0] and 0 <= y <= canvas[1])) and (best is None or x < best[0]):
                best = (x, y)
    if best is None:
        return
    family = style.get("function_label_font_family", style.get("font_family", default_font_family))
    primitives.draw_text(
        name,
        (best[0] + 4, max(best[1] - 4, font_size)),
        FontStyle(family=family, size=font_size),
        color,
        TextAlignment(horizontal="left", vertical="alphabetic"),
    )


def render_implicit_curve_helper(primitives: Any, curve: Any, coordinate_mapper: Any, style: Any) -> None:
    """Stroke an implicit curve over the view (and the margin), with its name."""
    try:
        paths, _ = _renderable(curve, coordinate_mapper).screen_paths(_view_margin(style), fill=False)
    except Exception as exc:
        print(f"[render_implicit_curve_helper] Error plotting {getattr(curve, 'name', 'unknown')}: {exc}")
        return
    if not paths:
        return
    stroke = _stroke(curve, style)
    _stroke_paths(primitives, paths, stroke)
    style_with_canvas = dict(style)
    style_with_canvas["_canvas_size"] = (
        float(getattr(coordinate_mapper, "canvas_width", 0) or 0) or float("inf"),
        float(getattr(coordinate_mapper, "canvas_height", 0) or 0) or float("inf"),
    )
    _render_label(primitives, curve, paths, stroke.color, style_with_canvas)


def render_inequality_area_helper(primitives: Any, area: Any, coordinate_mapper: Any, style: Any) -> None:
    """Fill an inequality region over the view (and the margin) and stroke its boundary."""
    margin = _view_margin(style)
    try:
        renderable = _renderable(area, coordinate_mapper)
        rings = renderable.screen_rings(margin)
        paths, atoms = renderable.screen_paths(margin, fill=True)
    except Exception as exc:
        print(f"[render_inequality_area_helper] Error plotting {getattr(area, 'name', 'unknown')}: {exc}")
        return
    # Fill only a complete ring set: open chains (a plot cut short) could shade wrong areas
    polygon = bridged_polygon(rings) if renderable.math_result(margin, fill=True).complete else []
    if len(polygon) >= 3:
        fill = FillStyle(color=str(getattr(area, "color", "lightblue")), opacity=getattr(area, "opacity", 0.3))
        primitives.fill_polygon(polygon, fill, None, screen_space=False)
    if not paths:
        return
    relation_atoms = area.relation.atoms
    solid: List[Sequence[Point]] = []
    dashes: List[Sequence[Point]] = []
    for path, atom in zip(paths, atoms):
        strict = relation_atoms[atom].strict if 0 <= atom < len(relation_atoms) else True
        if strict:
            dashes.extend(dashed(path))
        else:
            solid.append(path)
    stroke = _stroke(area, style)
    _stroke_paths(primitives, solid + dashes, stroke)
