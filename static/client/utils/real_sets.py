"""Sets of real numbers for the solve tool's inequalities: checking, finding and writing them.

A solution set is a sorted list of ``Piece`` objects (intervals, an isolated point being an
interval with equal closed ends). The inequality itself is a list of ``Relation`` objects
(two for a chain such as 1 < 2x+1 <= 5), each a function giving lhs - rhs and a sign.

- ``verify_solution_set`` checks a set (SymPy's) against the inequality on a dense grid,
  at every finite end and just beside it: "confirmed", "refuted" or "unchecked".
- ``sign_chart`` finds the set numerically when SymPy is unavailable or wrong: states on
  the grid, boundaries by bisection, touching roots by golden-section search, ends snapped
  to p/q or p*pi/q (``nice_number``).
- ``format_solution`` writes a set as inequalities and in interval notation.

Pure Python (no browser imports), so server-side pytest covers it; the browser supplies
math.js evaluators. Every number becomes text through ``number_text``, so CPython and
Brython print the same digits.
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

RealFunction = Callable[[float], Optional[float]]
# (point, lhs - rhs there) -> max(|lhs|, |rhs|) there
Magnitude = Callable[[float, float], Optional[float]]

# The grid: GRID_POINTS points spaced by asinh on [-GRID_LIMIT, GRID_LIMIT] (dense near 0),
# then geometric tail samples (TAIL_POINTS of them out to TAIL_LIMIT) on each side, reaching
# further when a seed (a polynomial root) lies beyond: to SEED_REACH times the largest one.
# A finite window gets GRID_POINTS evenly spaced points instead, plus the asinh grid's points
# inside it when it is wider than WIDE_WINDOW.
GRID_POINTS = 4001
GRID_LIMIT = 1000.0
TAIL_LIMIT = 1e6
TAIL_POINTS = 60
SEED_REACH = 100.0
MAX_REACH = 1e300
WIDE_WINDOW = 200.0
# The outer intervals reach Infinity only when this many outermost samples agree.
TAIL_AGREEMENT = 10
# A value below ROOT_TOLERANCE * max(1, |lhs|, |rhs|) may be rounding: its noise is measured.
ROOT_TOLERANCE = 1e-9
# Rounding noise. lhs - rhs within NOISE_ULPS ulps of max(|lhs|, |rhs|) of 0, or (for a value
# below ROOT_TOLERANCE * max(1, |lhs|, |rhs|)) within NOISE_FACTOR times its scatter over
# NOISE_NUDGES ulps of the point, is rounding noise: sin(x)^2 + cos(x)^2 against 1, or exp(-x^2)
# underflowed to 0. Such a sample says nothing about the sign.
EPSILON = 2.0**-52
SMALLEST = 5e-324
NOISE_ULPS = 16
NOISE_NUDGES = (-4, -1, 1, 4)
NOISE_FACTOR = 4.0
# Below |x| = 1 the noise is also measured over these ulps of 1 (_curvature), around 0 below
# CURVATURE_POINT; a second difference below CURVATURE_FLOOR there is no rounding of an inner
# value near 1.
CURVATURE_STEPS = (0.5, 1.5)
CURVATURE_POINT = 1e-12
CURVATURE_FLOOR = EPSILON / 16
# A run of noisy samples next to a value below UNDERFLOW_SIZE is underflow (x*exp(x) far left):
# it stays unknown. Any other run of at least ZERO_RUN noisy samples is a stretch where
# lhs = rhs (an identity such as (x+1)^2 >= x^2 + 2x + 1).
UNDERFLOW_SIZE = 1e-250
ZERO_RUN = 3
# An end undefined only by rounding (sqrt(2 - x^2) at the float nearest sqrt(2)) is read from
# its values these many ulps away.
END_NUDGES = (-1, 1, -2, 2, -4, 4, -8, 8, -16, 16)
# A boundary moves to the nearest nice number or seed within LOOSE_TOLERANCE (relative to
# max(1, |x|)) when lhs - rhs is within SNAP_FACTOR times its rounding noise of 0 at all
# SNAP_SAMPLES points between them; a touching point does when the state changes there too.
LOOSE_TOLERANCE = 1e-3
SNAP_SAMPLES = 8
SNAP_FACTOR = 4.0
# Beyond this every float is an integer, so no number is nicer than another.
NICE_LIMIT = 1e15
# Two points found numerically within this (relative) of each other are the same point
SAME_POINT = 1e-12
# A value this much larger than its neighbours at +-NEIGHBOUR_STEP (relative) is a pole that
# math.js reaches as a huge finite number (tan(pi/2) is 1.6e16).
POLE_RATIO = 1e4
POLE_MINIMUM = 1e6
NEIGHBOUR_STEP = 1e-7
# An infinite value whose neighbours are infinite or beyond this is an overflow (e^1000), not a pole.
OVERFLOW_SIZE = 1e100
# Ends are written as p/q or p*pi/q (q <= NICE_DENOMINATOR) when that close (relative); a
# value is written as 0 only within ZERO_SIZE of it (the rounding of 0 next to sides near 1).
NICE_DENOMINATOR = 12
NICE_TOLERANCE = 1e-10
ZERO_SIZE = 1e-13
# Numeric ends are written with this many significant digits.
DIGITS = 12
# The sign chart gives up beyond this many intervals.
MAX_PIECES = 40
# At most this many dips of |lhs - rhs| are searched for touching roots.
MAX_TOUCH_SEARCHES = 400

INFINITY_TEXT = "Infinity"


class Relation:
    """lhs - rhs as a function, the sign ("<", "<=", ">" or ">="), and optionally max(|lhs|, |rhs|).

    magnitude(point, lhs - rhs at point) gives max(|lhs|, |rhs|), the size the rounding of
    lhs - rhs is measured against; without it, |lhs - rhs| stands in.
    """

    def __init__(self, difference: RealFunction, sign: str, magnitude: Optional[Magnitude] = None) -> None:
        self.difference = difference
        self.sign = sign
        self.magnitude = magnitude

    @property
    def strict(self) -> bool:
        return self.sign in ("<", ">")


class Piece:
    """An interval of the solution set; an isolated point has lower == upper, both closed."""

    def __init__(
        self,
        lower: float,
        upper: float,
        lower_closed: bool,
        upper_closed: bool,
        lower_text: Optional[str] = None,
        upper_text: Optional[str] = None,
    ) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.lower_closed = lower_closed and math.isfinite(self.lower)
        self.upper_closed = upper_closed and math.isfinite(self.upper)
        self.lower_text = lower_text if lower_text is not None else end_text(self.lower)
        self.upper_text = upper_text if upper_text is not None else end_text(self.upper)

    @property
    def is_point(self) -> bool:
        return self.lower == self.upper

    def contains(self, value: float) -> bool:
        if value < self.lower or value > self.upper:
            return False
        if value == self.lower and not self.lower_closed:
            return False
        if value == self.upper and not self.upper_closed:
            return False
        return True

    def __repr__(self) -> str:
        return interval_notation([self])


class Window:
    """The interval an answer is restricted to ([0, 2*pi), ...), with its ends as text."""

    def __init__(
        self, lower: float, upper: float, lower_closed: bool, upper_closed: bool, lower_text: str, upper_text: str
    ) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.lower_closed = lower_closed and math.isfinite(self.lower)
        self.upper_closed = upper_closed and math.isfinite(self.upper)
        self.lower_text = lower_text
        self.upper_text = upper_text

    def contains(self, value: float) -> bool:
        return Piece(self.lower, self.upper, self.lower_closed, self.upper_closed, "", "").contains(value)

    def text(self) -> str:
        left = "[" if self.lower_closed else "("
        right = "]" if self.upper_closed else ")"
        return f"{left}{self.lower_text}, {self.upper_text}{right}"


# ----- numbers as text --------------------------------------------------------------------


def number_text(value: float, digits: int = DIGITS) -> str:
    """A float as text: an integer when it is one, else `digits` significant digits; +-Infinity."""
    value = float(value)
    if math.isinf(value):
        return INFINITY_TEXT if value > 0 else f"-{INFINITY_TEXT}"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return format(value, "." + str(digits) + "g")


def nice_number(value: float, tolerance: float = NICE_TOLERANCE) -> Optional[Tuple[str, float]]:
    """(text, exact value) of p/q or p*pi/q (q <= NICE_DENOMINATOR) within tolerance of value, else None."""
    value = float(value)
    if not math.isfinite(value):
        return None
    allowed = tolerance * max(1.0, abs(value))
    for denominator in range(1, NICE_DENOMINATOR + 1):
        numerator = round(value * denominator)
        if numerator == 0 and abs(value) > ZERO_SIZE:
            continue  # 3e-11 is not 0
        if abs(value - numerator / denominator) <= allowed:
            return _fraction_text(numerator, denominator, ""), numerator / denominator
    for denominator in range(1, NICE_DENOMINATOR + 1):
        numerator = round(value * denominator / math.pi)
        if numerator != 0 and abs(value - numerator * math.pi / denominator) <= allowed:
            return _fraction_text(numerator, denominator, "pi"), numerator * math.pi / denominator
    return None


def _fraction_text(numerator: int, denominator: int, factor: str) -> str:
    """3/4, -2, pi/6, -5*pi/6, 2*pi."""
    divisor = math.gcd(abs(numerator), denominator)
    numerator, denominator = numerator // divisor, denominator // divisor
    if factor:
        top = factor if numerator == 1 else f"-{factor}" if numerator == -1 else f"{numerator}*{factor}"
    else:
        top = str(numerator)
    return top if denominator == 1 else f"{top}/{denominator}"


def end_text(value: float, tolerance: float = NICE_TOLERANCE) -> str:
    """An end found numerically, as nice_number's text or number_text."""
    nice = nice_number(value, tolerance)
    return nice[0] if nice is not None else number_text(value)


def is_plain_number(text: str) -> bool:
    """True for an integer, a decimal, a fraction p/q or +-Infinity: text that needs no approximation."""
    body = text[1:] if text[:1] == "-" else text
    if body == INFINITY_TEXT:
        return True
    parts = body.split("/")
    if len(parts) > 2:
        return False
    for part in parts:
        try:
            float(part)
        except (ValueError, TypeError):  # Brython raises TypeError
            return False
        if not all(char.isdigit() or char in ".eE+-" for char in part):
            return False  # float() also reads "inf" and "nan"
    return True


# ----- evaluating the inequality ----------------------------------------------------------


def _value(function: RealFunction, point: float) -> Optional[float]:
    """function(point) as a float; None where it is undefined (not real, 0/0 or a pole), +-inf for an overflow."""
    value = function(point)
    if value is None or value != value:  # None, or NaN
        return None
    if (value == math.inf or value == -math.inf) and not _overflows(function, point):
        return None
    return float(value)


def _overflows(function: RealFunction, point: float) -> bool:
    """True when function's infinite value at point comes from an overflow: its neighbours are huge too."""
    step = NEIGHBOUR_STEP * max(1.0, abs(point))
    for neighbour in (point - step, point + step):
        value = function(neighbour)
        if value is None or math.isnan(value) or (math.isfinite(value) and abs(value) < OVERFLOW_SIZE):
            return False
    return True


def _compare(value: float, sign: str) -> bool:
    if sign == "<":
        return value < 0
    if sign == "<=":
        return value <= 0
    if sign == ">":
        return value > 0
    return value >= 0


def holds(relations: Sequence[Relation], point: float) -> Optional[bool]:
    """Whether every relation holds at point, comparing the computed values as they are; None where one is undefined."""
    result = True
    for relation in relations:
        value = _value(relation.difference, point)
        if value is None:
            return None
        result = result and _compare(value, relation.sign)
    return result


def _magnitude(relation: Relation, point: float, value: float) -> float:
    """max(|lhs|, |rhs|) at point, value being lhs - rhs there (|value| when the relation cannot say)."""
    size = relation.magnitude(point, value) if relation.magnitude is not None else None
    if size is None or not math.isfinite(size):
        return abs(value)
    return max(abs(size), abs(value) / 2)


def _noise(relation: Relation, point: float, value: float) -> float:
    """How far value (lhs - rhs at point) can be off by rounding.

    NOISE_ULPS ulps of max(|lhs|, |rhs|) (sin(x)^2 + cos(x)^2 against 1); for a value small
    next to the sides, also the scatter of lhs - rhs over a few ulps of the point, which shows
    cancellation inside one side (sin(x)^2 + cos(x)^2 - 1 against 0).
    """
    return _noise_and_suspicion(relation, point, value)[0]


def _noise_and_suspicion(
    relation: Relation, point: float, value: float, size: Optional[float] = None
) -> Tuple[float, bool]:
    """(_noise, whether value is below ROOT_TOLERANCE * max(1, |lhs|, |rhs|): small enough to be
    rounding); size is _magnitude there when the caller has it."""
    if size is None:
        size = _magnitude(relation, point, value)
    noise = NOISE_ULPS * EPSILON * size
    suspicious = abs(value) <= ROOT_TOLERANCE * max(1.0, size)
    if abs(value) <= noise or not suspicious:
        return noise, suspicious
    spread = _scatter(relation.difference, point, value, NOISE_NUDGES)
    if abs(point) < 1.0:
        curvature = _curvature(relation.difference, point, value)
        if curvature <= NOISE_FACTOR * NOISE_ULPS * EPSILON * max(1.0, size):
            spread = max(spread, curvature)
    return max(noise, spread), True


def _scatter(function: RealFunction, point: float, value: float, nudges: Sequence[int]) -> float:
    """NOISE_FACTOR times the largest change of function from value over the given ulps of point."""
    largest = 0.0
    for nudge in nudges:
        nudged = point * (1 + nudge * EPSILON) if point != 0 else nudge * SMALLEST
        other = _value(function, nudged)
        if other is not None and math.isfinite(other):
            largest = max(largest, abs(other - value))
    return NOISE_FACTOR * largest


def _curvature(function: RealFunction, point: float, value: float) -> float:
    """NOISE_FACTOR times the largest second difference of function over CURVATURE_STEPS ulps
    of 1 around point, where it shows the rounding of an inner value near 1; else 0.

    For |point| < 1 an inner 1 + x rounds by up to half an ulp of 1, which nudging point by
    its own ulps never shows (ln(1 + x) - x near 0); that rounding error is a sawtooth with
    a period of one ulp, so half ulps show it. The second difference cancels the slope, so a
    genuine small value (x itself) is not read as noise. Below CURVATURE_POINT, x plus an ulp
    of 1 loses x itself, so the reading is taken around 0, which rounds the same way. A
    reading far below an ulp of 1 is the function's shape, not rounding (x^2 near 1e-15); the
    caller ignores one above the rounding of an inner value near 1 (1/x, floor(x)).
    """
    if abs(point) < CURVATURE_POINT:
        point, value_at = 0.0, _value(function, 0.0)
        if value_at is None or not math.isfinite(value_at):
            return 0.0
        value = value_at
    largest = 0.0
    for nudge in CURVATURE_STEPS:
        step = nudge * EPSILON
        above, below = _value(function, point + step), _value(function, point - step)
        if above is not None and below is not None and math.isfinite(above) and math.isfinite(below):
            largest = max(largest, abs(above + below - 2 * value))
    return NOISE_FACTOR * largest if largest >= CURVATURE_FLOOR else 0.0


# The sign of lhs - rhs at a sample: None where undefined, -1 or 1, 0 on a stretch where
# lhs = rhs, _NEAR within rounding of 0 (unknown).
_NEAR = 2


def _code(relation: Relation, point: float, value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    if math.isfinite(value):
        size = _magnitude(relation, point, value)
        # Most samples are far from 0 next to the sides, and need no noise estimate
        if abs(value) <= ROOT_TOLERANCE * max(1.0, size):
            if abs(value) <= _noise_and_suspicion(relation, point, value, size)[0]:
                return _NEAR
    return 1 if value > 0 else -1


def _holds_code(code: int, sign: str) -> bool:
    if code == 0:
        return sign in ("<=", ">=")
    return sign in (">", ">=") if code > 0 else sign in ("<", "<=")


def _combine(relations: Sequence[Relation], codes: Sequence[Optional[int]]) -> Tuple[bool, Optional[bool]]:
    """(settled?, whether the inequality holds: None where undefined) from each relation's code.

    A relation within rounding of 0 leaves the answer unsettled, unless another one fails.
    """
    if any(code is None for code in codes):
        return True, None
    near = False
    for relation, code in zip(relations, codes):
        if code == _NEAR:
            near = True
        elif code is not None and not _holds_code(code, relation.sign):
            return True, False
    return (False, None) if near else (True, True)


def _point_state(relations: Sequence[Relation], point: float) -> Tuple[bool, Optional[bool]]:
    """(settled?, holds?) at one point: unsettled where lhs - rhs is within rounding of 0."""
    if len(relations) == 1:  # the common case, without the lists
        relation = relations[0]
        code = _code(relation, point, _value(relation.difference, point))
        if code is None:
            return True, None
        if code == _NEAR:
            return False, None
        return True, _holds_code(code, relation.sign)
    codes = [_code(relation, point, _value(relation.difference, point)) for relation in relations]
    return _combine(relations, codes)


def holds_at_end(relations: Sequence[Relation], point: float) -> Optional[bool]:
    """holds() at a candidate end, reading a value within rounding of 0 as a root (which a
    non-strict sign includes) and a huge value beside much smaller neighbours as a pole."""
    result = True
    for relation in relations:
        reading = _end_reading(relation, point)
        if reading is None:
            return None
        value, tolerance = reading
        if abs(value) <= tolerance:
            result = result and not relation.strict
        else:
            result = result and _compare(value, relation.sign)
    return result


def _end_reading(relation: Relation, point: float) -> Optional[Tuple[float, float]]:
    """(lhs - rhs at point, how close to 0 is a root there), or None where it is undefined or a pole.

    Where it is undefined only by rounding at the edge of its domain (sqrt(2 - x^2) at the
    float nearest sqrt(2), whose square is just above 2), its values a few ulps away on the
    side where it is defined decide: within their spread of 0 the end is a root (sqrt(2 -
    x^2) = 0 there); otherwise it stays undefined. A hole, defined on both sides, stays
    undefined.
    """
    value = _value(relation.difference, point)
    if value is not None:
        if _is_pole(relation.difference, point, value):
            return None
        if not math.isfinite(value):
            return value, 0.0  # an overflow (e^1000)
        return value, max(_root_tolerance(relation, point, value), _noise(relation, point, value))
    if point == 0 or not math.isfinite(point):
        return None
    sides: Dict[bool, List[float]] = {True: [], False: []}  # the defined values above and below
    for nudge in END_NUDGES:
        nudged = point * (1 + nudge * EPSILON)
        other = _value(relation.difference, nudged)
        if other is not None and math.isfinite(other) and not _is_pole(relation.difference, nudged, other):
            sides[nudge > 0].append(other)
    if bool(sides[True]) == bool(sides[False]):
        return None  # a hole ((x^2 - 1)/(x - 1) at 1, defined on both sides), or undefined around it
    nearby = sides[True] or sides[False]
    if abs(nearby[0]) > NOISE_FACTOR * (max(nearby) - min(nearby)):
        return None
    return 0.0, 0.0


def _is_pole(function: RealFunction, point: float, value: float) -> bool:
    if not math.isfinite(value) or abs(value) < POLE_MINIMUM:
        return False
    step = NEIGHBOUR_STEP * max(1.0, abs(point))
    neighbours = [_value(function, point - step), _value(function, point + step)]
    if any(neighbour is None for neighbour in neighbours):
        return True
    largest = max(1.0, *(abs(neighbour) for neighbour in neighbours if neighbour is not None))
    return abs(value) > POLE_RATIO * largest


def _root_tolerance(relation: Relation, point: float, value: float) -> float:
    """The change of lhs - rhs over the rounding of point (a printed end such as 0.69314718056).

    A value below it, or below its rounding noise (_noise), is a root; nothing else is:
    1e-10*(x - 2) at 1, (x - 1)^2 + 1e-10 at 1 and x^2 + 1.0000000001 against 2x at 1 are no
    roots, nor x at 1e-300.
    """
    step = NEIGHBOUR_STEP * max(1.0, abs(point))
    below, above = _value(relation.difference, point - step), _value(relation.difference, point + step)
    slope = 0.0
    if below is not None and above is not None and math.isfinite(below) and math.isfinite(above):
        slope = abs(above - below) / (2 * step)
    return 1e-12 * slope * abs(point)


# ----- the grid ---------------------------------------------------------------------------


def grid_reach(seeds: Sequence[float] = ()) -> float:
    """How far out the grid of an unbounded inequality samples: TAIL_LIMIT, or a power of ten
    at least SEED_REACH times the largest seed."""
    largest = max([abs(seed) for seed in seeds if math.isfinite(seed)] + [0.0])
    reach = TAIL_LIMIT
    while reach < SEED_REACH * largest and reach < MAX_REACH:
        reach *= 10
    return reach


def reach_text(reach: float) -> str:
    """10^6 for 1e6; number_text for anything else."""
    exponent = round(math.log10(reach))
    return f"10^{exponent}" if abs(reach - 10.0**exponent) <= 1e-9 * reach else number_text(reach)


def sample_grid(window: Optional[Window], seeds: Sequence[float] = ()) -> List[float]:
    """The points both the check and the sign chart sample, sorted.

    The seeds inside the window are added, with the midpoint of each two neighbouring seeds
    (x^2 - 5x + 6 < 0 between 2 and 3 on [-10000, 10000], where even samples are 5 apart).
    """
    top = math.asinh(GRID_LIMIT)
    half = (GRID_POINTS - 1) // 2
    inner = [math.sinh(top * index / half) for index in range(-half, half + 1)]
    ratio = (TAIL_LIMIT / GRID_LIMIT) ** (1.0 / TAIL_POINTS)
    count = int(math.ceil(math.log(grid_reach(seeds) / GRID_LIMIT) / math.log(ratio) - 1e-9))
    tail = [GRID_LIMIT * ratio**index for index in range(1, count + 1)]
    base = [-value for value in reversed(tail)] + inner + tail
    if window is not None and math.isfinite(window.lower) and math.isfinite(window.upper):
        step = (window.upper - window.lower) / (GRID_POINTS - 1)
        points = [window.lower + index * step for index in range(GRID_POINTS - 1)] + [window.upper]
        if window.upper - window.lower > WIDE_WINDOW:
            points += [point for point in base if window.lower < point < window.upper]
    else:
        points = base
        if window is not None:
            points = [point for point in points if window.contains(point)]
            points += [end for end in (window.lower, window.upper) if math.isfinite(end) and window.contains(end)]
    inside = sorted({seed for seed in seeds if math.isfinite(seed) and (window is None or window.contains(seed))})
    points += inside + [(first + second) / 2 for first, second in zip(inside, inside[1:])]
    return sorted(set(points))


def _contained(pieces: Sequence[Piece], point: float) -> bool:
    return any(piece.contains(point) for piece in pieces)


def _membership(pieces: Sequence[Piece], points: Sequence[float]) -> List[bool]:
    """_contained for each of the sorted points, in one sweep over the sorted pieces."""
    result: List[bool] = []
    index = 0
    count = len(pieces)
    for point in points:
        while index < count and pieces[index].upper < point:
            index += 1
        inside = index < count and pieces[index].contains(point)
        if not inside and index + 1 < count:
            inside = pieces[index + 1].contains(point)  # [0, 1) and [1, 2] share the point 1
        result.append(inside)
    return result


def _codes(relation: Relation, points: Sequence[float]) -> Tuple[List[Optional[float]], List[Optional[int]]]:
    """(lhs - rhs at each point, its code there), with the runs of noisy samples settled.

    The loop is written out (no helper calls for a sample far from 0): the browser runs it
    for every grid point, and a Brython call costs microseconds.
    """
    difference = relation.difference
    magnitude = relation.magnitude
    values: List[Optional[float]] = []
    codes: List[Optional[int]] = []
    small: List[bool] = []
    for point in points:
        value = _value(difference, point)
        values.append(value)
        if value is None:
            codes.append(None)
            small.append(False)
            continue
        size = value if value >= 0 else -value
        if size > ROOT_TOLERANCE and size != math.inf:
            # Far from 0 unless the sides are much larger (cancellation): no noise estimate needed
            reported = magnitude(point, value) if magnitude is not None else None
            if reported is None or reported != reported or reported == math.inf or reported == -math.inf:
                far = True
            else:
                far = size > ROOT_TOLERANCE * (reported if reported >= 0 else -reported)
            if far:
                codes.append(1 if value > 0 else -1)
                small.append(False)
                continue
        elif size == math.inf:
            codes.append(1 if value > 0 else -1)
            small.append(False)
            continue
        noise, suspicious = _noise_and_suspicion(relation, point, value)
        codes.append(_NEAR if size <= noise else (1 if value > 0 else -1))
        small.append(suspicious)
    _settle_noise_runs(values, codes, small)
    return values, codes


def _settle_noise_runs(values: Sequence[Optional[float]], codes: List[Optional[int]], small: Sequence[bool]) -> None:
    """Read each run of ZERO_RUN or more noisy samples as a stretch where lhs = rhs (code 0),
    unless a value next to it is below UNDERFLOW_SIZE: then it is underflow and stays unknown.

    A few small values between two such stretches are rounding the noise estimate missed
    ((x+1)^2 against x^2 + 2x + 1 near -1, where both sides are tiny next to x^2 and 2x):
    they join them.
    """
    count = len(codes)
    start = 0
    while start < count:
        if codes[start] != _NEAR:
            start += 1
            continue
        end = start
        while end < count and codes[end] == _NEAR:
            end += 1
        border = [values[index] for index in (start - 1, end) if 0 <= index < count and codes[index] in (-1, 1)]
        underflow = any(value is not None and abs(value) < UNDERFLOW_SIZE for value in border)
        if end - start >= ZERO_RUN and not underflow:
            for index in range(start, end):
                codes[index] = 0
        start = end
    start = 1
    while start < count:
        if codes[start] == 0 or codes[start - 1] != 0:
            start += 1
            continue
        end = start
        while end < count and end - start < ZERO_RUN and codes[end] != 0 and small[end]:
            end += 1
        if end < count and codes[end] == 0:
            for index in range(start, end):
                codes[index] = 0
        start = end + 1


def _grid_states(
    relations: Sequence[Relation], points: Sequence[float]
) -> Tuple[List[List[Optional[float]]], List[Optional[bool]], List[bool]]:
    """(each relation's values at the points, whether the inequality holds there (None where
    undefined), whether that is settled: False where a sample is within rounding of 0)."""
    columns: List[List[Optional[float]]] = []
    code_columns: List[List[Optional[int]]] = []
    for relation in relations:
        values, codes = _codes(relation, points)
        columns.append(values)
        code_columns.append(codes)
    if len(relations) == 1:  # the common case, by table lookup
        sign = relations[0].sign
        table: Dict[Optional[int], Tuple[bool, Optional[bool]]] = {
            None: (True, None),
            _NEAR: (False, None),
            0: (True, _holds_code(0, sign)),
            1: (True, _holds_code(1, sign)),
            -1: (True, _holds_code(-1, sign)),
        }
        pairs = [table[code] for code in code_columns[0]]
        return columns, [pair[1] for pair in pairs], [pair[0] for pair in pairs]
    states: List[Optional[bool]] = []
    settled: List[bool] = []
    for index in range(len(points)):
        is_settled, state = _combine(relations, [codes[index] for codes in code_columns])
        settled.append(is_settled)
        states.append(state)
    return columns, states, settled


def _near_end(ends: Sequence[float], point: float) -> bool:
    for end in ends:
        if abs(point - end) <= 2 * NEIGHBOUR_STEP * max(1.0, abs(end)):
            return True
    return False


# ----- checking a set ---------------------------------------------------------------------


def verify_solution_set(
    relations: Sequence[Relation],
    pieces: Sequence[Piece],
    window: Optional[Window] = None,
    seeds: Sequence[float] = (),
) -> str:
    """ "confirmed", "refuted" or "unchecked": whether pieces is the inequality's solution set in window.

    Every grid point away from the ends must hold exactly where the set contains it (a point
    of the set where the inequality is undefined refutes it); a sample within rounding of a
    root, or underflowed to 0, neither confirms nor refutes. Each finite end and window end
    in the window, and each seed (a polynomial root, say), must be in the set exactly when
    the inequality holds there (a root belongs to it for <= and >=), and the points just
    beside each end must agree with the set too. "unchecked" when the inequality could not
    be evaluated anywhere.
    """
    ends = [end for piece in pieces for end in (piece.lower, piece.upper) if math.isfinite(end)]
    window_ends = [end for end in (window.lower, window.upper) if math.isfinite(end)] if window is not None else []
    skipped = ends + window_ends
    points = [point for point in sample_grid(window, seeds) if not _near_end(skipped, point)]
    _, states, settled = _grid_states(relations, points)
    checked = 0
    for state, is_settled, inside in zip(states, settled, _membership(pieces, points)):
        if not is_settled:
            continue
        if state is None:
            if inside:
                return "refuted"
            continue
        checked += 1
        if state != inside:
            return "refuted"
    if checked == 0:
        return "unchecked"
    for point in _end_candidates(ends, window, seeds):
        expected = holds_at_end(relations, point)
        if bool(expected) != _contained(pieces, point):
            return "refuted"
    for end in ends:
        step = NEIGHBOUR_STEP * max(1.0, abs(end))
        for point in (end - step, end + step):
            if window is not None and not window.contains(point):
                continue
            is_settled, state = _point_state(relations, point)
            if not is_settled:
                continue
            if state is None and _contained(pieces, point):
                return "refuted"
            if state is not None and state != _contained(pieces, point):
                return "refuted"
    return "confirmed"


def _end_candidates(ends: Sequence[float], window: Optional[Window], seeds: Sequence[float]) -> List[float]:
    """The ends, window ends and seeds in the window (an open window leaves its ends out); a
    seed is a numeric root, off by rounding, so one beside an end is that end."""
    candidates = [end for end in ends if window is None or window.contains(end)]
    candidates += [
        point for point in seeds if (window is None or window.contains(point)) and not _near_end(ends, point)
    ]
    if window is not None:
        candidates += [end for end in (window.lower, window.upper) if math.isfinite(end) and window.contains(end)]
    return candidates


# ----- the sign chart ---------------------------------------------------------------------


class SignChartFailure(ValueError):
    """The sign chart could not find a reliable answer (the message says why)."""


def sign_chart(
    relations: Sequence[Relation], window: Optional[Window] = None, seeds: Sequence[float] = ()
) -> List[Piece]:
    """The inequality's solution set found numerically on the grid (and seeds, e.g. polynomial roots).

    Samples within rounding of a root (or underflowed to 0) are left out; a long run of them
    is a stretch where both sides are equal. Boundaries are found by bisection between
    samples, touching roots by golden-section search, and ends are snapped to seeds and nice
    numbers.

    Raises:
        SignChartFailure: when the inequality cannot be evaluated, its sign keeps changing far
            out (no reliable outer interval) or the set has more than MAX_PIECES intervals.
    """
    points = sample_grid(window, seeds)
    values, states, settled = _grid_states(relations, points)
    kept = [index for index in range(len(points)) if settled[index]]
    if all(states[index] is None for index in kept):
        raise SignChartFailure("the inequality cannot be evaluated")
    if window is None or not (math.isfinite(window.lower) and math.isfinite(window.upper)):
        _check_tails([states[index] for index in kept])
    _check_piece_count([states[index] for index in kept])
    touching = _touching_roots(relations, points, values, states, settled, seeds)
    chart_points = [points[index] for index in kept]
    chart_states = [states[index] for index in kept]
    if touching:
        merged = [item for item in zip(chart_points, chart_states) if item[0] not in touching]
        merged += [(point, holds_at_end(relations, point)) for point in touching]
        merged.sort(key=lambda item: item[0])
        chart_points = [point for point, _ in merged]
        chart_states = [state for _, state in merged]
    sampled = [points[index] for index in kept]  # the grid samples, judged as runs (no touching points)
    boundaries = _boundaries(relations, chart_points, chart_states, sampled, seeds)
    pieces = _pieces(relations, boundaries, chart_points, chart_states, sampled, window)
    if len(pieces) > MAX_PIECES:
        raise SignChartFailure(f"the solution set has more than {MAX_PIECES} intervals; give a smaller interval")
    return pieces


def _check_piece_count(states: List[Optional[bool]]) -> None:
    """Fail before any bisection when the samples alone hold in more than MAX_PIECES separate
    runs (sin(x) > 0 on [0, 10^9]): each run lies in its own interval of the set."""
    runs = sum(1 for index, state in enumerate(states) if state and (index == 0 or not states[index - 1]))
    if runs > MAX_PIECES:
        raise SignChartFailure(f"the solution set has more than {MAX_PIECES} intervals; give a smaller interval")


def _check_tails(states: List[Optional[bool]]) -> None:
    for side in (states[:TAIL_AGREEMENT], states[-TAIL_AGREEMENT:]):
        if len(set(side)) > 1:
            raise SignChartFailure("its sign keeps changing far from 0")


def _touching_roots(
    relations: Sequence[Relation],
    points: List[float],
    values: List[List[Optional[float]]],
    states: List[Optional[bool]],
    settled: List[bool],
    seeds: Sequence[float] = (),
) -> List[float]:
    """Where |lhs - rhs| dips to 0 without a sign change ((x-1)^2 > 0 at 1), found by
    golden-section search: around each dip between samples and each run of unsettled samples
    whose settled neighbours agree. A seed or nice number near the minimum where the state
    changes too is taken instead (exp(x) > x + 1 at 0, not 8.5e-10)."""
    count = len(points)
    before = [-1] * count  # the nearest settled sample before each index
    last = -1
    for index in range(count):
        before[index] = last
        if settled[index]:
            last = index
    after = [count] * count  # the nearest settled sample after each index
    last = count
    for index in range(count - 1, -1, -1):
        after[index] = last
        if settled[index]:
            last = index
    regions: List[Tuple[int, int]] = []
    for column in values:
        sizes = [_size(value) for value in column]
        for index in range(1, count - 1):
            if sizes[index] < sizes[index - 1] and sizes[index] < sizes[index + 1]:
                regions.append((index - 1, index + 1))
    for index in range(1, count - 1):
        if not settled[index] and settled[index - 1]:  # the start of an unsettled run
            regions.append((index - 1, after[index]))
    found: List[float] = []
    searched: Set[Tuple[int, int]] = set()
    for low, high in regions:
        if len(searched) >= MAX_TOUCH_SEARCHES or (low, high) in searched:
            continue
        left = low if settled[low] else before[low]
        right = high if high >= count or settled[high] else after[high]
        if left < 0 or right >= count or states[left] is None or states[left] != states[right]:
            continue  # a sign change (left to bisection), or no settled sample on one side
        searched.add((low, high))
        bracket = (points[low], points[min(high, count - 1)])
        best = _touching_candidate(relations, bracket[0], bracket[1])
        if best is None:
            continue
        # A seed or a nice number beside the minimum is preferred where it is a touching point itself
        for candidate in _snap_candidates(best, seeds, LOOSE_TOLERANCE) + [best]:
            if not bracket[0] <= candidate <= bracket[1]:
                continue
            if holds_at_end(relations, candidate) != states[left]:
                if not any(_same_point(candidate, other) for other in found):
                    found.append(candidate)
                break
    return sorted(found)


def _same_point(first: float, second: float) -> bool:
    """Whether two points found numerically are one: within 1e-12 relative. Relative also near 0,
    where -1e-15 and 1e-15 are the two ends of x^2 <= 1e-30."""
    return abs(first - second) <= SAME_POINT * max(abs(first), abs(second))


def _size(value: Optional[float]) -> float:
    """|value|, or inf where it is undefined or infinite."""
    if value is None or value != value:
        return math.inf
    return value if value >= 0 else -value


def _touching_candidate(relations: Sequence[Relation], low: float, high: float) -> Optional[float]:
    """The point of [low, high] where some |lhs - rhs| is smallest."""
    best: Optional[float] = None
    best_size = math.inf
    for relation in relations:
        candidate = _golden_minimum(relation.difference, low, high)
        if candidate is None:
            continue
        value = _value(relation.difference, candidate)
        size = abs(value) if value is not None and math.isfinite(value) else math.inf
        if size < best_size:
            best, best_size = candidate, size
    return best


_GOLDEN = (math.sqrt(5) - 1) / 2


def _golden_minimum(function: RealFunction, low: float, high: float) -> Optional[float]:
    """The point of [low, high] where |function| is smallest, by golden-section search."""

    def size(point: float) -> float:
        value = _value(function, point)
        return abs(value) if value is not None and math.isfinite(value) else math.inf

    a, b = low, high
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    size_c, size_d = size(c), size(d)
    for _ in range(100):
        if b - a <= 1e-15 * max(1.0, abs(a), abs(b)):
            break
        if size_c <= size_d:
            b, d, size_d = d, c, size_c
            c = b - _GOLDEN * (b - a)
            size_c = size(c)
        else:
            a, c, size_c = c, d, size_d
            d = a + _GOLDEN * (b - a)
            size_d = size(d)
    middle = (a + b) / 2
    return middle if math.isfinite(size(middle)) else None


# A boundary found by bisection: (value, low, high) with the state changing in [low, high]
_Bracket = Tuple[float, float, float]


class _Boundary:
    """A point where the inequality's state changes, its end text and whether it belongs to the set."""

    def __init__(self, value: float, text: str, state: Optional[bool]) -> None:
        self.value = value
        self.text = text
        self.state = state


def _boundaries(
    relations: Sequence[Relation],
    points: List[float],
    states: List[Optional[bool]],
    sampled: Sequence[float],
    seeds: Sequence[float] = (),
) -> List[_Boundary]:
    """Every state change between neighbouring chart points, located by bisection and snapped
    to a seed or a nice value; sampled are the grid samples (the chart points but touching ones)."""
    raw: List[_Bracket] = []
    for index in range(len(points) - 1):
        if states[index] != states[index + 1]:
            raw += _transitions(relations, points[index], states[index], points[index + 1], states[index + 1], 0)
    boundaries: List[_Boundary] = []
    for value, low, high in _collapse_noise_gaps(relations, sorted(raw), sampled):
        exact = _snapped(relations, value, low, high, seeds)
        if boundaries and _same_point(boundaries[-1].value, exact):
            continue
        nice = nice_number(exact)
        text = nice[0] if nice is not None and nice[1] == exact else number_text(exact)
        boundaries.append(_Boundary(exact, text, holds_at_end(relations, exact)))
    return boundaries


def _snapped(relations: Sequence[Relation], value: float, low: float, high: float, seeds: Sequence[float]) -> float:
    """value moved to a seed or a nice number in its bracket [low, high] or with only rounding
    noise between them, else value.

    Bisection stops at a computed sign change, which within the noise around a root can be
    anywhere in it: sin(x) < x at 1.5e-7 (the root is 0), the expanded (x - 1)^3 at 0.99999
    (1), x^2 > 1e-12 just beside 1e-06; at a pole it stops where the values overflow (1/sin(x)
    at 5.6e-309, the pole is 0). A candidate within NICE_TOLERANCE (relative) is taken as it
    is: the printed digits cannot tell them apart, and math.js's floor and ceil read a number
    within 1e-12 of an integer as that integer (floor(x) > 1 changes at 1.999999999998).
    """
    for candidate in _snap_candidates(value, seeds, LOOSE_TOLERANCE):
        if low <= candidate <= high or abs(candidate - value) <= NICE_TOLERANCE * abs(value):
            return candidate
        if _noise_between(relations, value, candidate) or _pole_between(relations, value, candidate):
            return candidate
    return value


def _snap_candidates(value: float, seeds: Sequence[float], tolerance: float) -> List[float]:
    """The nearest nice number (p/q or p*pi/q in lowest terms, 0 included; none beyond
    NICE_LIMIT) and the nearest seed (a numeric root, off by rounding at a multiple root)
    within tolerance (relative to max(1, |value|)) of value, in that order."""
    allowed = tolerance * max(1.0, abs(value))
    nice: Optional[float] = None
    if abs(value) < NICE_LIMIT:
        for denominator in range(1, NICE_DENOMINATOR + 1):
            for factor in (1.0, math.pi):
                numerator = round(value * denominator / factor)
                divisor = math.gcd(numerator, denominator)
                candidate = (numerator // divisor) * factor / (denominator // divisor)
                if abs(candidate - value) < allowed and (nice is None or abs(candidate - value) < abs(nice - value)):
                    nice = candidate
    near = [seed for seed in seeds if math.isfinite(seed) and abs(seed - value) <= allowed]
    seed = min(near, key=lambda item: abs(item - value)) if near else None
    candidates = [nice] if nice is not None else []
    if seed is not None and (nice is None or not _same_point(seed, nice)):
        candidates.append(seed)
    return candidates


def _noise_between(relations: Sequence[Relation], start: float, end: float) -> bool:
    """Whether some lhs - rhs is within SNAP_FACTOR times its rounding noise of 0 at all of
    SNAP_SAMPLES points strictly between start and end.

    The noise is judged over the run, not point by point: it is the largest estimate at any
    of the points, because the estimate at a single point can fall short (the scatter of the
    expanded (x - 1)^3 over a few ulps of 1.00001 happens to be 0).
    """
    low, high = min(start, end), max(start, end)
    points = [low + (high - low) * index / (SNAP_SAMPLES + 1) for index in range(1, SNAP_SAMPLES + 1)]
    points = [point for point in points if low < point < high]
    for relation in relations:
        values = [_value(relation.difference, point) for point in points]
        if any(value is None or not math.isfinite(value) for value in values):
            continue
        sizes = [abs(value) for value in values if value is not None]
        noise = max(
            (_noise(relation, point, value) for point, value in zip(points, values) if value is not None), default=0.0
        )
        if all(size <= SNAP_FACTOR * noise for size in sizes):
            return True
    return False


def _pole_between(relations: Sequence[Relation], start: float, end: float) -> bool:
    """Whether some lhs - rhs is undefined at end (a pole) and infinite all the way from start:
    1/sin(x) overflows to inf below 5.6e-309, so bisection stops there, not at the pole 0."""
    low, high = min(start, end), max(start, end)
    points = [low + (high - low) * index / (SNAP_SAMPLES + 1) for index in range(1, SNAP_SAMPLES + 1)]
    for relation in relations:
        function = relation.difference
        if _value(function, end) is not None:
            continue
        values = [function(point) for point in points if low < point < high]
        if all(value is not None and math.isinf(value) for value in values):
            return True
    return False


def _sampled_between(points: Sequence[float], start: float, end: float) -> bool:
    """Whether one of the sorted points lies strictly between start and end (start < end)."""
    index = bisect.bisect_right(points, start)
    return index < len(points) and points[index] < end


def _collapse_noise_gaps(relations: Sequence[Relation], raw: List[_Bracket], points: Sequence[float]) -> List[_Bracket]:
    """Two boundaries with only rounding noise between them as one point between them.

    Near a touching root lhs - rhs can round to 0 over a short stretch (2*sin(x)^2 - sin(x) -
    1 within 1e-8 of pi/2), so bisection finds where that stretch starts and where it ends.
    A gap the grid sampled is no noise: its samples were judged as runs, and exact zeros over
    a run are a stretch where lhs = rhs (2 <= floor(x) < 3).
    """
    collapsed: List[_Bracket] = []
    for bracket in raw:
        value = bracket[0]
        if collapsed:
            previous = collapsed[-1]
            duplicate = _same_point(previous[0], value)
            if duplicate or (
                not _sampled_between(points, previous[0], value) and _noise_between(relations, previous[0], value)
            ):
                collapsed[-1] = ((previous[0] + value) / 2, previous[1], bracket[2])
                continue
        collapsed.append(bracket)
    return collapsed


def _transitions(
    relations: Sequence[Relation],
    low: float,
    low_state: Optional[bool],
    high: float,
    high_state: Optional[bool],
    depth: int,
) -> List[_Bracket]:
    """The points between low and high where the state changes (two when a third state lies
    between), each with the last bracket around it. Each midpoint is judged by its computed
    values as they are: that ends bisection at the edge of a stretch where lhs = rhs exactly
    (floor(x) = 2 up to 3) and at the computed crossing of a simple root, not at the edge of a
    noise estimate (1 - cos(x) < 1e-6). Within the noise around a root it ends anywhere;
    _collapse_noise_gaps and _snapped judge that noise over runs of points."""
    for _ in range(200):
        middle = _middle(low, high)
        if middle <= low or middle >= high:
            break
        state = holds(relations, middle)
        if state == low_state:
            low = middle
        elif state == high_state:
            high = middle
        elif depth < 4:
            return _transitions(relations, low, low_state, middle, state, depth + 1) + _transitions(
                relations, middle, state, high, high_state, depth + 1
            )
        else:
            high = middle
    return [((low + high) / 2, low, high)]


def _middle(low: float, high: float) -> float:
    """The point bisection tries next in [low, high]: 0 when they have opposite signs, the
    geometric mean when one is more than twice the other (0 standing for the smallest double),
    else the arithmetic mean. Halving the exponent first reaches a boundary as close to 0 as
    1e-300 (1e300*x > 1) in about 70 steps; halving the bracket would need over 1000."""
    if low < 0 < high:
        return 0.0
    sign = 1.0 if high > 0 else -1.0
    near, far = (low, high) if sign > 0 else (-high, -low)
    near = max(near, SMALLEST)
    if far > 2 * near:
        return sign * math.sqrt(near) * math.sqrt(far)
    return (low + high) / 2


def _pieces(
    relations: Sequence[Relation],
    boundaries: List[_Boundary],
    points: List[float],
    states: List[Optional[bool]],
    sampled: Sequence[float],
    window: Optional[Window],
) -> List[Piece]:
    """The set from the boundaries: alternating open stretches and boundary points, each in or out."""
    lower = window.lower if window is not None else -math.inf
    upper = window.upper if window is not None else math.inf
    inner = [boundary for boundary in boundaries if lower < boundary.value < upper]
    # A boundary with only rounding noise between it and a window end is that end (cos(x) >= 1 at 2*pi)
    if inner and math.isfinite(lower) and _noise_gap(relations, sampled, lower, inner[0].value):
        inner = inner[1:]
    if inner and math.isfinite(upper) and _noise_gap(relations, sampled, inner[-1].value, upper):
        inner = inner[:-1]
    # atoms: (start, end, start text, end text, in the set?) -- a point has start == end
    atoms: List[Tuple[float, float, str, str, bool]] = []
    if window is not None and window.lower_closed:
        atoms.append((lower, lower, window.lower_text, window.lower_text, bool(holds_at_end(relations, lower))))
    edges = [(lower, window.lower_text if window is not None else f"-{INFINITY_TEXT}")]
    edges += [(boundary.value, boundary.text) for boundary in inner]
    edges.append((upper, window.upper_text if window is not None else INFINITY_TEXT))
    for index in range(len(edges) - 1):
        (start, start_text), (end, end_text_) = edges[index], edges[index + 1]
        atoms.append((start, end, start_text, end_text_, _stretch_state(relations, start, end, points, states)))
        if index < len(inner):
            boundary = inner[index]
            atoms.append((boundary.value, boundary.value, boundary.text, boundary.text, bool(boundary.state)))
    if window is not None and window.upper_closed:
        atoms.append((upper, upper, window.upper_text, window.upper_text, bool(holds_at_end(relations, upper))))
    return _merge_atoms(atoms)


def _noise_gap(relations: Sequence[Relation], points: Sequence[float], start: float, end: float) -> bool:
    """Whether (start, end) holds no grid sample and only rounding noise."""
    return not _sampled_between(points, start, end) and _noise_between(relations, start, end)


def _stretch_state(
    relations: Sequence[Relation], start: float, end: float, points: List[float], states: List[Optional[bool]]
) -> bool:
    """Whether the open stretch (start, end) between two boundaries is in the set."""
    if math.isinf(start) or math.isinf(end):
        # The outermost sample stands for the stretch out to Infinity
        index = 0 if math.isinf(start) else len(points) - 1
        if math.isinf(start) and math.isinf(end):
            index = len(points) // 2
        return bool(states[index])
    is_settled, state = _point_state(relations, (start + end) / 2)
    if is_settled:
        return bool(state)
    # The middle is within rounding of 0: a sample inside the stretch says
    index = bisect.bisect_right(points, start)
    if index < len(points) and points[index] < end:
        return bool(states[index])
    return False


def _merge_atoms(atoms: List[Tuple[float, float, str, str, bool]]) -> List[Piece]:
    pieces: List[Piece] = []
    run: List[Tuple[float, float, str, str, bool]] = []
    for atom in atoms + [(0.0, 0.0, "", "", False)]:
        if atom[4]:
            run.append(atom)
            continue
        if run:
            first, last = run[0], run[-1]
            pieces.append(Piece(first[0], last[1], first[0] == first[1], last[0] == last[1], first[2], last[3]))
            run = []
    return pieces


# ----- writing a set ----------------------------------------------------------------------


def interval_notation(pieces: Sequence[Piece]) -> str:
    """(2, 3), (-Infinity, -1] ∪ [3, Infinity), {0}."""
    parts = []
    for piece in pieces:
        if piece.is_point:
            parts.append(f"{{{piece.lower_text}}}")
        else:
            left = "[" if piece.lower_closed else "("
            right = "]" if piece.upper_closed else ")"
            parts.append(f"{left}{piece.lower_text}, {piece.upper_text}{right}")
    return " ∪ ".join(parts)


def _piece_inequality(piece: Piece, variable: str) -> str:
    if piece.is_point:
        return f"{variable} = {piece.lower_text}"
    low_sign = "<=" if piece.lower_closed else "<"
    high_sign = "<=" if piece.upper_closed else "<"
    if math.isinf(piece.lower):
        return f"{variable} {high_sign} {piece.upper_text}"
    if math.isinf(piece.upper):
        return f"{variable} {'>=' if piece.lower_closed else '>'} {piece.lower_text}"
    return f"{piece.lower_text} {low_sign} {variable} {high_sign} {piece.upper_text}"


def _missing_points(pieces: Sequence[Piece], window: Optional[Window]) -> Optional[List[str]]:
    """The points the set leaves out when it covers the whole domain but them, else None."""
    lower = window.lower if window is not None else -math.inf
    upper = window.upper if window is not None else math.inf
    lower_closed = window.lower_closed if window is not None else False
    upper_closed = window.upper_closed if window is not None else False
    intervals = [piece for piece in pieces if not piece.is_point]
    if not intervals or intervals[0].lower != lower or intervals[-1].upper != upper:
        return None
    if (lower_closed and not intervals[0].lower_closed) or (upper_closed and not intervals[-1].upper_closed):
        return None
    missing: List[str] = []
    for before, after in zip(intervals, intervals[1:]):
        if before.upper != after.lower:
            return None
        if not (before.upper_closed or after.lower_closed or _contained(pieces, before.upper)):
            missing.append(before.upper_text)
    return missing or None


def format_solution(
    variable: str,
    pieces: Sequence[Piece],
    relation_text: str,
    window: Optional[Window] = None,
    period: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """The solution set as the model sees it.

    "2 < x < 3; interval notation: (2, 3)", "x <= -1 or x >= 3; interval notation: ...",
    "x = 0; interval notation: {0}", "No real x satisfies x^2 + 1 < 0 (the solution set is
    empty).", "Every real x satisfies x^2 + 1 > 0; interval notation: (-Infinity, Infinity)".
    A window adds "(on [0, 2*pi])", and with a period "(on [-2*pi, 2*pi]; the pattern
    repeats every 2*pi)"; an empty or whole set names the window already ("No x in [0, 1]
    satisfies ..."), so it keeps only the period. Ends that are not plain numbers get
    decimals: "; endpoints ≈ ...". A note (how the set was found) ends the text in
    parentheses, inside the empty set's.
    """
    where = f"{variable} in {window.text()}" if window is not None else f"real {variable}"
    repeats = f"the pattern repeats every {period}" if window is not None and period else ""
    suffix = f" ({note})" if note else ""
    if not pieces:
        remarks = "; ".join(part for part in ("the solution set is empty", repeats, note or "") if part)
        return f"No {where} satisfies {relation_text} ({remarks})."
    domain = Piece(
        window.lower if window else -math.inf,
        window.upper if window else math.inf,
        window.lower_closed if window else False,
        window.upper_closed if window else False,
    )
    if len(pieces) == 1 and _same_piece(pieces[0], domain):
        scope = f" ({repeats})" if repeats else ""
        return f"Every {where} satisfies {relation_text}{scope}; interval notation: {interval_notation(pieces)}{suffix}"
    scope = ""
    if window is not None:
        scope = f" (on {window.text()}; {repeats})" if repeats else f" (on {window.text()})"
    text = " or ".join(_piece_inequality(piece, variable) for piece in pieces)
    missing = _missing_points(pieces, window)
    if missing:
        text += f" (every {where} except {', '.join(missing)})"
    text += f"{scope}; interval notation: {interval_notation(pieces)}"
    return text + _approximations(pieces) + suffix


def _same_piece(piece: Piece, other: Piece) -> bool:
    return (piece.lower, piece.upper, piece.lower_closed, piece.upper_closed) == (
        other.lower,
        other.upper,
        other.lower_closed,
        other.upper_closed,
    )


def _approximations(pieces: Sequence[Piece]) -> str:
    """ "; endpoints ≈ 0.523598775598, 2.61799387799" for the ends that are not plain numbers."""
    approximations: List[str] = []
    seen: List[str] = []
    for piece in pieces:
        for text, value in ((piece.lower_text, piece.lower), (piece.upper_text, piece.upper)):
            if is_plain_number(text) or text in seen or not math.isfinite(value):
                continue
            seen.append(text)
            approximations.append(number_text(value))
    return f"; endpoints ≈ {', '.join(approximations)}" if approximations else ""
