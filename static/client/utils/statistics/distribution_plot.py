"""Where a distribution plot is drawn and what its shaded range holds (pure, no canvas).

``StatisticsManager.plot_distribution`` asks this module for:
    - the drawn range: the plot bounds, widened to take in the shade range when the plot
      bounds are defaults, then clipped to the distribution's support (``drawn_range``);
    - for a discrete distribution, the integers to draw a bar for (``integer_bars``), at
      most ``MAX_INTEGER_BARS``: default bounds are trimmed to that many around the mode;
    - the probability of the shade range asked for, clipped to the support but not to the
      drawn range, and its text, "P(2 <= X <= 5)" (``shaded_probability``).
Notes for the tool result explain each widening, trimming or clipping.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from utils.statistics.distribution_functions import floor_integer
from utils.statistics.distribution_tool import interval_probability, round_significant
from utils.statistics.distributions import DistributionSpec

MAX_INTEGER_BARS = 500

Range = Tuple[float, float]


def support_text(spec: DistributionSpec, params: Dict[str, float]) -> str:
    lo, hi = spec.support(params)
    left = "(-inf" if math.isinf(lo) else f"[{_plain(lo)}"
    right = "inf)" if math.isinf(hi) else f"{_plain(hi)}]"
    return f"{left}, {right}"


def _plain(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    rounded = round_significant(value, 10)
    assert rounded is not None
    return str(int(rounded)) if rounded == math.floor(rounded) and abs(rounded) < 1e15 else repr(rounded)


def clip_to_support(spec: DistributionSpec, params: Dict[str, float], span: Range) -> Range:
    """``span`` intersected with the support (an infinite end may stay infinite)."""
    lo, hi = spec.support(params)
    return max(span[0], lo), min(span[1], hi)


def drawn_range(
    spec: DistributionSpec,
    params: Dict[str, float],
    bounds: Range,
    *,
    bounds_given: Tuple[bool, bool],
    shade: Optional[Range],
    notes: List[str],
) -> Range:
    """The range the curve or bars cover; raises ValueError when it misses the support.

    A default side widens to a finite shade end beyond it, so the shaded region is drawn.
    A given side outside the support is clipped to it, with a note.
    """
    left, right = bounds
    if shade is not None:
        margin = 0.1 * (right - left)
        if not bounds_given[0]:
            # A shade range beyond the left end (finite) or wholly left of the plot is drawn too.
            for end in shade:
                if math.isfinite(end) and end < left:
                    left = end - (margin if end == shade[1] else 0.0)
        if not bounds_given[1]:
            for end in shade:
                if math.isfinite(end) and end > right:
                    right = end + (margin if end == shade[0] else 0.0)
    clipped_left, clipped_right = clip_to_support(spec, params, (left, right))
    if clipped_left > clipped_right or (clipped_left == clipped_right and not spec.discrete):
        raise ValueError(
            f"plot_bounds [{_plain(left)}, {_plain(right)}] miss the support of {spec.name}, "
            f"{support_text(spec, params)}"
        )
    if (bounds_given[0] and clipped_left > left) or (bounds_given[1] and clipped_right < right):
        notes.append(f"plot_bounds clipped to the support of {spec.name}, {support_text(spec, params)}")
    return clipped_left, clipped_right


def integer_bars(
    spec: DistributionSpec,
    params: Dict[str, float],
    span: Range,
    *,
    bounds_given: bool,
    shade: Optional[Range],
    notes: List[str],
) -> Tuple[int, int]:
    """The first and last integer to draw a bar for, inside the drawn ``span``.

    More than ``MAX_INTEGER_BARS`` integers: given bounds are refused, default ones keep the
    integers around the mode. A note says when the shade range reaches past the bars.
    """
    first = -floor_integer(-span[0])
    last = floor_integer(span[1])
    if first > last:
        raise ValueError(f"plot_bounds [{_plain(span[0])}, {_plain(span[1])}] hold no value of {spec.name}")
    if last - first + 1 > MAX_INTEGER_BARS:
        if bounds_given:
            raise ValueError(
                f"plot_bounds hold {last - first + 1} values of {spec.name}; at most {MAX_INTEGER_BARS} "
                "bars can be drawn, so narrow plot_bounds"
            )
        first, last = _trim_around_mode(spec, params, first, last)
        notes.append(f"drawn: the {MAX_INTEGER_BARS} values {first}..{last} around the most likely value")
    if shade is not None and (shade[0] < first - 0.5 or shade[1] > last + 0.5):
        shaded = clip_to_support(spec, params, shade)
        if shaded[0] < first - 0.5 or shaded[1] > last + 0.5:
            notes.append("the shaded range reaches past the drawn bars; shaded_probability counts all of it")
    return first, last


def _trim_around_mode(spec: DistributionSpec, params: Dict[str, float], first: int, last: int) -> Tuple[int, int]:
    mode = spec.mode(params) if spec.mode is not None else 0.5 * (first + last)
    start = int(round(mode)) - MAX_INTEGER_BARS // 2
    start = max(first, min(start, last - MAX_INTEGER_BARS + 1))
    return start, start + MAX_INTEGER_BARS - 1


def values_text(first: int, last: int) -> str:
    """The values a discrete plot draws a bar for: "0..10"."""
    return f"{first}..{last}"


def shaded_probability(spec: DistributionSpec, params: Dict[str, float], shade: Range) -> Dict[str, object]:
    """``shaded_probability`` (12 significant digits) and ``shaded_range`` of a shade range.

    The range is clipped to the support; an end may be infinite (an open side). For a
    discrete distribution it is the integers inside it ("P(2 <= X <= 5)", "P(X >= 3)"),
    and a range holding none has probability 0.
    """
    left, right = clip_to_support(spec, params, shade)
    if not spec.discrete:
        return {
            "shaded_probability": round_significant(interval_probability(spec, params, left, right)),
            "shaded_range": _continuous_text(left, right),
        }
    first = -floor_integer(-left)
    if math.isinf(right):
        return {
            "shaded_probability": round_significant(interval_probability(spec, params, first, right)),
            "shaded_range": f"P(X >= {first})",
        }
    last = floor_integer(right)
    if first > last:
        return {"shaded_probability": 0.0, "shaded_range": f"no value of X in [{_plain(left)}, {_plain(right)}]"}
    text = f"P(X = {first})" if first == last else f"P({first} <= X <= {last})"
    return {
        "shaded_probability": round_significant(interval_probability(spec, params, first, last)),
        "shaded_range": text,
    }


def _continuous_text(left: float, right: float) -> str:
    if math.isinf(left) and math.isinf(right):
        return "P(-inf < X < inf)"
    if math.isinf(left):
        return f"P(X <= {_plain(right)})"
    if math.isinf(right):
        return f"P(X >= {_plain(left)})"
    return f"P({_plain(left)} <= X <= {_plain(right)})"
