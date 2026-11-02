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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from utils.implicit_plotter import PlotResult, plot

Bounds = Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

# Grid cell of the curve check, in pixels, and its budget
SUMMARY_CELL_PIXELS = 6.0
SUMMARY_MAX_EVALS = 15000
# Samples of the region check across and up the view
REGION_SAMPLES = (40, 25)


# A power whose exponent is a fraction or a decimal: x^(2/3), x^0.5 (complex for x < 0)
_FRACTIONAL_POWER = re.compile(r"\^\s*\(?\s*-?\s*(?:\d+\s*/\s*\d+|\d*\.\d+)")
_FRACTIONAL_POWER_NOTE = (
    "A fractional power of a negative number is not real here (x^(2/3) for x < 0), so that part "
    "is not drawn; for the real cube root write cbrt(x), as in cbrt(x)^2 + cbrt(y)^2 = 1."
)


def fractional_power_note(text: str) -> Optional[str]:
    """The note on fractional powers when the text has one, else None."""
    return _FRACTIONAL_POWER_NOTE if _FRACTIONAL_POWER.search(text) else None


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
    xs, ys, pieces = [], [], 0
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
            "the view, or have no real points (x^2 + y^2 = -1), or be a single point or touch zero without "
            "crossing it (x^2 + y^2 = 0), which is not drawn."
        )
    if not result.complete:
        _add_note(summary, "The curve is intricate in this view and was only partly drawn.")
    elif drawn is not None and result.cell > 1.01 * result.asked_cell:
        _add_note(
            summary,
            f"The curve is dense in this view, so it was drawn on a coarser grid ({result.cell * scale:.0f} px); "
            "zooming in shows it in full detail.",
        )
    _add_note(summary, fractional_power_note(equation))
    return summary


def region_in_view(
    inequality: str, contains: Callable[[float, float], bool], bounds: Bounds, samples: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """{"shaded_share"} of the view (sampled), plus a "note" when none or all of it is shaded."""
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
    if inside == 0:
        summary["note"] = (
            f"No sample point of the view ({_view_text(bounds)}) satisfies {inequality}. The region "
            "may lie outside the view, be thinner than the samples, or be empty (x^2 + y^2 < 0)."
        )
    elif inside == across * up:
        summary["note"] = f"The whole view ({_view_text(bounds)}) satisfies {inequality}; its boundary is outside it."
    _add_note(summary, fractional_power_note(inequality))
    return summary
