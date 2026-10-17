"""Whether two expressions are equivalent, by comparing their values at sample points.

This is the numeric half of the step checker (check_steps), stricter than
SymbolicChecks.equivalent (7 points, 2 agreeing, one variable): every variable varies at
once, over four tiers of |values| up to 1, 4, 40 and 400 with 24 points each, and
"equivalent" needs no mismatch and at least MIN_AGREEING points where both sides are
defined. A difference beyond |x| = 4 (abs(x - 5) against 5 - x) is seen, because every
tier is sampled, and a domain such as asin's [-1, 1] still has enough points.

The points come from a Kronecker sequence, frac(k * alpha_j + beta_j) with irrational
alpha_j, so they are deterministic and never land exactly on 0, +-1 or another integer,
where removable holes ((x^2 - 1)/(x - 1) at 1) usually sit. A consequence: removed holes
are not reported.

Values are compared with the tolerance REL * max(|a|, |b|) + ABS, plus each side's
rounding noise, so an expanded polynomial near a root, whose computed values cancel
terms far larger than the result, is not called different. The noise is how much a
side's value scatters when the variables are nudged by a few units in the last place,
plus SCALE_ULPS units in the last place of its scale (the sum of |term| over its terms,
when the caller gives one): x^8 + 1 - x^8 is 0 in floating point at x = 300 whatever the
nudge, but its terms are near 10^20.

Evaluators take a dict of variable values and return a float, or None where the value is
not real or cannot be computed. An infinite or NaN value (an overflow) is unknown: it
neither agrees nor differs. Points where exactly one side is defined are counted
(one_sided): log(x^2) against 2*log(x) agree wherever both are defined, and the report
names a point where only one is.

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Point = Dict[str, float]
Evaluator = Callable[[Point], Optional[float]]

EQUIVALENT = "equivalent"
NOT_EQUIVALENT = "not_equivalent"
UNDECIDED = "undecided"

# Values agree when they differ by at most REL * max(|a|, |b|) + ABS (plus rounding noise).
REL = 1e-8
ABS = 1e-12
# The sample tiers: the largest |value| of a variable, and the points in each tier.
TIERS = (1.0, 4.0, 40.0, 400.0)
POINTS_PER_TIER = 24
# "equivalent" needs this many points where both sides are defined (1 without variables).
MIN_AGREEING = 8
# Two mismatches, or one larger than this (relative), make the expressions not equivalent.
MIN_MISMATCHES = 2
CLEAR_MISMATCH = 1e-4
# This many points where exactly one side is defined give the domain note.
MIN_ONE_SIDED = 2

# Rounding noise is estimated from f with every variable scaled by (1 + k * 2^-52) for these
# k, and scaled up by NOISE_FACTOR, since a few nudges only sample it.
NOISE_NUDGES = (-8, -4, -2, -1, 1, 2, 4, 8)
NOISE_FACTOR = 4.0
# A side with a scale (the sum of its terms' sizes) may be off by this many units in the
# last place of the scale.
SCALE_ULPS = 16
EPSILON = 2.0**-52
SMALLEST = 2.0**-1074

# Readable points tried before a sample point, nearest 0 first: integers, then halves.
_READABLE_INTEGERS = (1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0)
_READABLE_HALVES = (0.5, -0.5, 1.5, -1.5, 2.5, -2.5)
_READABLE_LARGE = tuple(float(sign * value) for value in range(6, 11) for sign in (1, -1))
# With several variables, readable points combine this many values per variable.
_READABLE_PER_VARIABLE = (1.0, 2.0, -1.0, 3.0, 0.5, -2.0)
_MAX_READABLE_POINTS = 400
# Significant digits a sample point is rounded to, in turn, before it is given as it is.
_ROUNDING_DIGITS = (1, 2, 3, 4)


class EquivalenceReport:
    """What compare found.

    Attributes:
        verdict: EQUIVALENT, NOT_EQUIVALENT or UNDECIDED.
        agreeing: points where both sides are defined and agree.
        mismatches: points where both are defined and differ.
        one_sided: points where exactly one side is defined.
        counterexample: for NOT_EQUIVALENT, a readable point where they differ, with
            first_value and second_value there.
        undefined_point: when one_sided >= MIN_ONE_SIDED, a readable point where only one
            side is defined, and undefined_side (1 for the first, 2 for the second) the
            side that is not.
        reason: why the verdict is UNDECIDED (empty otherwise).
    """

    __slots__ = (
        "verdict",
        "agreeing",
        "mismatches",
        "one_sided",
        "counterexample",
        "first_value",
        "second_value",
        "undefined_point",
        "undefined_side",
        "reason",
    )

    def __init__(self, verdict: str, agreeing: int, mismatches: int, one_sided: int, reason: str = "") -> None:
        self.verdict = verdict
        self.agreeing = agreeing
        self.mismatches = mismatches
        self.one_sided = one_sided
        self.counterexample: Optional[Point] = None
        self.first_value: Optional[float] = None
        self.second_value: Optional[float] = None
        self.undefined_point: Optional[Point] = None
        self.undefined_side: Optional[int] = None
        self.reason = reason

    @property
    def domain_note(self) -> bool:
        """True when one side is undefined at points where the other is defined."""
        return self.undefined_point is not None


def compare(
    first: Evaluator,
    second: Evaluator,
    variables: Sequence[str],
    first_scale: Optional[Evaluator] = None,
    second_scale: Optional[Evaluator] = None,
) -> EquivalenceReport:
    """Compare two expressions' values at the sample points of every tier.

    first_scale and second_scale, when given, are each side's scale: the sum of |term| over
    the terms of a sum (see SCALE_ULPS), None where it cannot be computed.
    """
    names = list(variables)
    sides = _Sides(first, second, first_scale, second_scale)
    agreeing = mismatches = one_sided = 0
    first_defined = second_defined = False
    largest_mismatch = 0.0
    mismatch_point: Optional[Point] = None
    one_sided_point: Optional[Point] = None
    one_sided_side = 0
    for point in sample_points(names):
        a, b = _value(first, point), _value(second, point)
        first_defined = first_defined or a is not None
        second_defined = second_defined or b is not None
        if a is None and b is None:
            continue
        if a is None or b is None:
            one_sided += 1
            if one_sided_point is None:
                one_sided_point, one_sided_side = point, 1 if a is None else 2
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue  # an overflow: unknown
        size = sides.mismatch(point, a, b)
        if size is None:
            agreeing += 1
            continue
        mismatches += 1
        if mismatch_point is None or size > largest_mismatch:
            mismatch_point, largest_mismatch = point, size
    if not names:
        one_sided = 0  # one point, evaluated once: nothing to sample
    report = _verdict(names, agreeing, mismatches, one_sided, largest_mismatch, first_defined, second_defined)
    if report.verdict == NOT_EQUIVALENT and mismatch_point is not None:
        _add_counterexample(report, sides, names, mismatch_point)
    elif report.verdict != NOT_EQUIVALENT and one_sided >= MIN_ONE_SIDED and one_sided_point is not None:
        point = _readable_point(names, one_sided_point, lambda p: _defined_side(first, second, p) == one_sided_side)
        report.undefined_point = point
        report.undefined_side = one_sided_side
    return report


def _verdict(
    names: Sequence[str],
    agreeing: int,
    mismatches: int,
    one_sided: int,
    largest_mismatch: float,
    first_defined: bool,
    second_defined: bool,
) -> EquivalenceReport:
    if mismatches >= MIN_MISMATCHES or (mismatches == 1 and largest_mismatch > CLEAR_MISMATCH):
        return EquivalenceReport(NOT_EQUIVALENT, agreeing, mismatches, one_sided)
    needed = MIN_AGREEING if names else 1
    if mismatches == 0 and agreeing >= needed:
        return EquivalenceReport(EQUIVALENT, agreeing, mismatches, one_sided)
    if not first_defined and not second_defined:
        reason = "neither expression has a real value at any sample point"
    elif not first_defined or not second_defined:
        side = "first" if not first_defined else "second"
        reason = f"the {side} expression has no real value at any sample point"
    elif mismatches:
        reason = "the values differ slightly at one sample point only, which may be rounding"
    elif agreeing == 0:
        reason = "the expressions are never both defined at a sample point"
    else:
        reason = f"the expressions are both defined at only {agreeing} sample point{'s' if agreeing != 1 else ''}"
    return EquivalenceReport(UNDECIDED, agreeing, mismatches, one_sided, reason)


# ----- sample points ----------------------------------------------------------------------


def sample_points(variables: Sequence[str]) -> List[Point]:
    """The points of every tier: POINTS_PER_TIER per tier, |values| up to TIERS[i].

    Without variables there is one point, the empty one.
    """
    names = list(variables)
    if not names:
        return [{}]
    steps = _irrational_steps(len(names))
    points: List[Point] = []
    index = 0
    for reach in TIERS:
        for _ in range(POINTS_PER_TIER):
            index += 1
            points.append(
                {
                    name: _off_integer(reach * (2.0 * ((index * alpha + beta) % 1.0) - 1.0))
                    for name, (alpha, beta) in zip(names, steps)
                }
            )
    return points


def _irrational_steps(count: int) -> List[Tuple[float, float]]:
    """(alpha_j, beta_j) for each variable: fractional parts of square roots of primes, and offsets."""
    steps: List[Tuple[float, float]] = []
    candidate = 2
    while len(steps) < count:
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1)):
            root = math.sqrt(candidate)
            alpha = root - math.floor(root)
            beta = (0.5 + 0.6180339887498949 * (len(steps) + 1)) % 1.0
            steps.append((alpha, beta))
        candidate += 1
    return steps


def _off_integer(value: float) -> float:
    """value, nudged when it is an integer (where holes sit), which the sequence makes very unlikely."""
    return value + 1e-3 if value == math.floor(value) else value


# ----- comparing values -------------------------------------------------------------------


def _value(function: Evaluator, point: Point) -> Optional[float]:
    """function at point as a float (inf and nan included), None where it is undefined."""
    try:
        value = function(point)
    except (ArithmeticError, ValueError, TypeError):
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Sides:
    """The two expressions and their scales."""

    __slots__ = ("first", "second", "first_scale", "second_scale")

    def __init__(
        self, first: Evaluator, second: Evaluator, first_scale: Optional[Evaluator], second_scale: Optional[Evaluator]
    ) -> None:
        self.first = first
        self.second = second
        self.first_scale = first_scale
        self.second_scale = second_scale

    def mismatch(self, point: Point, a: float, b: float) -> Optional[float]:
        """None when a and b agree within the tolerance and rounding noise, else |a - b| relative to the larger."""
        difference = abs(a - b)
        tolerance = REL * max(abs(a), abs(b)) + ABS
        if difference <= tolerance:
            return None
        tolerance += self._scale_noise(point)
        if difference <= tolerance:
            return None
        if difference <= tolerance + rounding_noise(self.first, point) + rounding_noise(self.second, point):
            return None
        return difference / max(abs(a), abs(b), ABS)

    def differs(self, point: Point) -> bool:
        """Whether both sides are defined and finite at point and differ there."""
        a, b = _value(self.first, point), _value(self.second, point)
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            return False
        return self.mismatch(point, a, b) is not None

    def _scale_noise(self, point: Point) -> float:
        total = 0.0
        for scale in (self.first_scale, self.second_scale):
            size = _value(scale, point) if scale is not None else None
            if size is not None and math.isfinite(size):
                total += abs(size)
        return SCALE_ULPS * EPSILON * total


def rounding_noise(function: Evaluator, point: Point) -> float:
    """How much function's floating-point value at point is off by rounding, estimated from its scatter.

    Every variable is scaled by (1 + k * 2^-52) for k in NOISE_NUDGES (a variable at 0 is
    set to k * 2^-1074). An expanded polynomial near a root ((x-3)^15 expanded, at 2.2)
    cancels terms much larger than its value, so its computed values scatter far beyond a
    relative tolerance on the value. inf when f is not finite at point or at a nudge.
    """
    centre = _value(function, point)
    if centre is None or not math.isfinite(centre):
        return math.inf
    largest = 0.0
    for nudge in NOISE_NUDGES:
        nudged = {
            name: value * (1 + nudge * EPSILON) if value != 0 else nudge * SMALLEST for name, value in point.items()
        }
        value = _value(function, nudged)
        if value is None or not math.isfinite(value):
            return math.inf
        largest = max(largest, abs(value - centre))
    return NOISE_FACTOR * largest


def _defined_side(first: Evaluator, second: Evaluator, point: Point) -> int:
    """0 when both sides or neither is defined at point, else the undefined side (1 or 2)."""
    a, b = _value(first, point), _value(second, point)
    if (a is None) == (b is None):
        return 0
    return 1 if a is None else 2


# ----- readable points --------------------------------------------------------------------


def _add_counterexample(report: EquivalenceReport, sides: _Sides, names: Sequence[str], found: Point) -> None:
    point = _readable_point(names, found, sides.differs)
    report.counterexample = point
    report.first_value = _value(sides.first, point)
    report.second_value = _value(sides.second, point)


def _readable_point(names: Sequence[str], found: Point, accept: Callable[[Point], bool]) -> Point:
    """A point accept() takes: small integers or halves first, then found rounded to 1 to 4
    significant digits, else found itself."""
    for candidate in _readable_candidates(names):
        if accept(candidate):
            return candidate
    for digits in _ROUNDING_DIGITS:
        rounded = {name: _rounded(value, digits) for name, value in found.items()}
        if accept(rounded):
            return rounded
    return dict(found)


def _readable_candidates(names: Sequence[str]) -> List[Point]:
    if not names:
        return []
    if len(names) == 1:
        values = _READABLE_INTEGERS + _READABLE_HALVES + _READABLE_LARGE
        return [{names[0]: value} for value in values]
    points: List[Point] = [{}]
    for name in names:
        points = [dict(point, **{name: value}) for point in points for value in _READABLE_PER_VARIABLE]
        if len(points) > _MAX_READABLE_POINTS:
            points = points[:_MAX_READABLE_POINTS]
    return points


def _rounded(value: float, digits: int) -> float:
    if value == 0 or not math.isfinite(value):
        return value
    return float(format(value, "." + str(digits) + "g"))


# ----- text -------------------------------------------------------------------------------


def number_text(value: float, digits: int = 12) -> str:
    """real_sets.number_text, the one number-to-text helper of the CAS tools (imported when
    first needed: SymbolicChecks imports this module, and most of its users never print)."""
    from utils.real_sets import number_text as real_sets_number_text

    return str(real_sets_number_text(value, digits))


def value_text(value: float) -> str:
    """A value as the step checker writes it: an integer, else up to 12 significant digits."""
    return str(number_text(value, 12))


def distinct_value_texts(first: float, second: float) -> Tuple[str, str]:
    """first and second with as few significant digits (6 to 12) as tell them apart."""
    for digits in (6, 9, 12, 15, 17):
        texts = (str(number_text(first, digits)), str(number_text(second, digits)))
        if texts[0] != texts[1]:
            return texts
    return str(number_text(first, 17)), str(number_text(second, 17))


def point_text(point: Point) -> str:
    """A point as text: x = 1, or x = 1, y = -0.5."""
    return ", ".join(f"{name} = {value_text(value)}" for name, value in point.items())


def json_number(value: float) -> object:
    """value for a JSON result: an int when it is one, else a float of 12 significant digits."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return str(number_text(value))
    if value == int(value) and abs(value) < 1e15:
        return int(value)
    return float(number_text(value, 12))
