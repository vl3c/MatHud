"""What an implicit curve or an inequality region shows in a view, for the tools' results.

``curve_in_view`` plots the curve over the view on a coarse grid and reports whether it
has points there, how many pieces, and their x and y range; ``region_in_view`` samples
the view and reports the share of it the region covers. Each adds a note when nothing
is found (or everything is shaded), so the model can tell the user what the drawing
shows. Both are estimates on a grid (6 px for curves, 40 × 25 samples for regions).

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.implicit_plotter import PlotResult, plot, ring_area

Bounds = Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

# Grid cell of the curve check, in pixels, and its budget
SUMMARY_CELL_PIXELS = 6.0
SUMMARY_MAX_EVALS = 15000
# Samples of the region check across and up the view
REGION_SAMPLES = (40, 25)


# A power of x, y or a bracket whose exponent is a fraction in brackets or a decimal:
# x^(2/3), (x - 1)^(1/3), y^0.5 (complex for a negative base); not x^2/9, which is x^2 over 9
_FRACTIONAL_POWER = re.compile(
    r"(?:(?<![a-z_])[xy]|\))\s*(?:\^|\*\*)\s*(?:\(\s*-?\s*\d+\s*/\s*\d+\s*\)|\(?\s*-?\s*\d*\.\d*[1-9]\d*)"
)
_FRACTIONAL_POWER_NOTE = (
    "A fractional power of a negative number is not real here (x^(2/3) for x < 0), so that part "
    "is not drawn; for the real cube root write cbrt(x), as in cbrt(x)^2 + cbrt(y)^2 = 1."
)


# A bracket that is never negative: abs(...), or sums of even powers and positive numbers
# ((x^2 + y^2)^(1/2))
_EVEN_POWER_SUM = re.compile(r"\s*(?:\d*\.?\d+\s*\*?\s*)?(?:[xy]\s*(?:\^|\*\*)\s*\d*[02468]|\d*\.?\d+)\s*")


def fractional_power_note(text: str) -> Optional[str]:
    """The note on fractional powers when the text has one on a base that can be negative,
    else None."""
    for match in _FRACTIONAL_POWER.finditer(text):
        if not (text[match.start()] == ")" and _never_negative_bracket(text, match.start())):
            return _FRACTIONAL_POWER_NOTE
    return None


def _never_negative_bracket(text: str, close: int) -> bool:
    """Whether the bracket closing at ``close`` is abs(...) or a sum of even powers."""
    depth = 0
    for index in range(close, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                if text[:index].rstrip().endswith("abs"):
                    return True
                terms = text[index + 1 : close].split("+")
                return all(_EVEN_POWER_SUM.fullmatch(term) for term in terms)
    return False


def _add_note(summary: Dict[str, Any], note: Optional[str]) -> None:
    if note:
        summary["note"] = f"{summary['note']} {note}" if summary.get("note") else note


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def _view_text(bounds: Bounds) -> str:
    x_min, y_min, x_max, y_max = bounds
    return f"x in [{_round(x_min):g}, {_round(x_max):g}], y in [{_round(y_min):g}, {_round(y_max):g}]"


def curve_in_view(
    equation: str,
    g: Callable[[float, float], float],
    factors: Sequence[Callable[..., float]],
    bounds: Bounds,
    scale: float,
    drawn: Optional[PlotResult] = None,
) -> Dict[str, Any]:
    """{"points_found", "pieces", "x_range", "y_range"} of the curve within bounds, plus a
    "note" when no point was found, the plot was cut short or drawn coarser, or the
    equation has a fractional power.

    drawn: The plot the renderer made (it covers the view), used instead of plotting again.
    """
    x_min, y_min, x_max, y_max = bounds
    summary: Dict[str, Any] = {"view": _view_text(bounds)}
    result = drawn
    if result is None:
        try:
            result = plot(
                g,
                bounds,
                SUMMARY_CELL_PIXELS / scale,
                factors=list(factors),
                max_evals=SUMMARY_MAX_EVALS,
            )
        except ValueError:
            summary["points_found"] = False
            return summary
    xs: List[float] = []
    ys: List[float] = []
    pieces = 0
    for path in result.curves:
        inside = [(x, y) for x, y in path if x_min <= x <= x_max and y_min <= y <= y_max]
        if inside:
            pieces += 1
            xs.extend(x for x, _ in inside)
            ys.extend(y for _, y in inside)
    summary["points_found"] = bool(xs)
    if xs:
        summary["pieces"] = pieces
        summary["x_range"] = [_round(min(xs)), _round(max(xs))]
        summary["y_range"] = [_round(min(ys)), _round(max(ys))]
    else:
        summary["note"] = (
            f"No point of {equation} was found in the view ({_view_text(bounds)}). It may lie outside "
            "the view, be smaller than a few pixels at this zoom (zoom in to see it), have no real points "
            "(x^2 + y^2 = -1), or be a single point or touch zero without crossing it (x^2 + y^2 = 0), which "
            "is not drawn."
        )
    if not result.complete:
        _add_note(summary, "The curve is intricate in this view and was only partly drawn.")
    elif drawn is not None and result.cell > 1.01 * result.asked_cell:
        _add_note(
            summary,
            f"To stay within the plotting budget it was drawn on a coarser grid ({result.cell * scale:.0f} px cells "
            f"instead of {result.asked_cell * scale:.0f} px), so fine detail may be smoothed.",
        )
    _add_note(summary, fractional_power_note(equation))
    return summary


def _clip(ring: Sequence[Tuple[float, float]], bounds: Bounds) -> List[Tuple[float, float]]:
    """The ring clipped to the bounds (Sutherland-Hodgman, one side of the box at a time)."""
    points = list(ring)
    for axis, limit, keep_above in (
        (0, bounds[0], True),
        (0, bounds[2], False),
        (1, bounds[1], True),
        (1, bounds[3], False),
    ):
        if not points:
            break
        clipped: List[Tuple[float, float]] = []
        for k, current in enumerate(points):
            previous = points[k - 1]
            current_in = (current[axis] >= limit) == keep_above
            previous_in = (previous[axis] >= limit) == keep_above
            if current_in != previous_in:
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(
                    (previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1]))
                )
            if current_in:
                clipped.append(current)
        points = clipped
    return points


def _ring_area_in_view(rings: Sequence[Sequence[Tuple[float, float]]], bounds: Bounds) -> float:
    """The area of the region the rings bound (outer rings counter-clockwise, holes
    clockwise) within bounds."""
    return float(abs(sum(ring_area(_clip(ring, bounds)) for ring in rings)))


def region_in_view(
    inequality: str,
    contains: Callable[[float, float], bool],
    bounds: Bounds,
    samples: Optional[Tuple[int, int]] = None,
    drawn: Optional[PlotResult] = None,
) -> Dict[str, Any]:
    """{"shaded_share"} of the view (sampled), plus a "note" when none or all of it is shaded.

    drawn: The plot the renderer made; when no sample falls in the region but the drawn
        region reaches the view, the share is its area over the view's and a note says the
        region is small in this view.
    """
    x_min, y_min, x_max, y_max = bounds
    across, up = samples or REGION_SAMPLES
    inside = 0
    for i in range(across):
        x = x_min + (x_max - x_min) * (i + 0.5) / across
        for j in range(up):
            y = y_min + (y_max - y_min) * (j + 0.5) / up
            try:
                if contains(x, y):
                    inside += 1
            except Exception:
                continue
    share = inside / float(across * up)
    summary: Dict[str, Any] = {"view": _view_text(bounds), "shaded_share": round(share, 3)}
    small_area = _ring_area_in_view(drawn.rings, bounds) if inside == 0 and drawn is not None else 0.0
    if small_area > 0:
        view_area = (x_max - x_min) * (y_max - y_min)
        summary["shaded_share"] = float(f"{small_area / view_area:.2g}")
        area = float(f"{small_area:.2g}")
        summary["note"] = (
            f"The region is small in this view ({_view_text(bounds)}): about {area:g} square units; "
            "zooming in shows it larger."
        )
    elif inside == 0:
        summary["note"] = (
            f"No sample point of the view ({_view_text(bounds)}) satisfies {inequality}. The region "
            "may lie outside the view, be thinner than the samples, or be empty (x^2 + y^2 < 0)."
        )
    elif inside == across * up:
        summary["note"] = f"The whole view ({_view_text(bounds)}) satisfies {inequality}; its boundary is outside it."
    if drawn is not None and not drawn.complete:
        _add_note(
            summary,
            "The region is too intricate in this view to fill: only parts of its boundary are drawn, without "
            "shading. Zooming in shows it filled.",
        )
    _add_note(summary, fractional_power_note(inequality))
    return summary
