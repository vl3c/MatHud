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

import math
from typing import Callable, List, Optional, Sequence, Tuple

RealFunction = Callable[[float], Optional[float]]

# The grid: GRID_POINTS points spaced by asinh on [-GRID_LIMIT, GRID_LIMIT] (dense near 0),
# then TAIL_POINTS geometric samples on each side out to TAIL_LIMIT. A finite window gets
# GRID_POINTS evenly spaced points instead.
GRID_POINTS = 4001
GRID_LIMIT = 1000.0
TAIL_LIMIT = 1e6
TAIL_POINTS = 60
# The outer intervals reach Infinity only when this many outermost samples agree.
TAIL_AGREEMENT = 10
# A value within ROOT_TOLERANCE * max(1, |lhs|, |rhs|) of 0 (plus the rounding of its point)
# is a root.
ROOT_TOLERANCE = 1e-9
# A value this much larger than its neighbours at +-NEIGHBOUR_STEP (relative) is a pole that
# math.js reaches as a huge finite number (tan(pi/2) is 1.6e16).
POLE_RATIO = 1e4
POLE_MINIMUM = 1e6
NEIGHBOUR_STEP = 1e-7
# An infinite value whose neighbours are infinite or beyond this is an overflow (e^1000), not a pole.
OVERFLOW_SIZE = 1e100
# Ends are written as p/q or p*pi/q (q <= NICE_DENOMINATOR) when that close (relative).
NICE_DENOMINATOR = 12
NICE_TOLERANCE = 1e-10
TOUCH_NICE_TOLERANCE = 1e-7
# Numeric ends are written with this many significant digits.
DIGITS = 12
# The sign chart gives up beyond this many intervals.
MAX_PIECES = 40
# At most this many dips of |lhs - rhs| are searched for touching roots.
MAX_TOUCH_SEARCHES = 400

INFINITY_TEXT = "Infinity"


class Relation:
    """lhs - rhs as a function, the sign ("<", "<=", ">" or ">="), and optionally max(1, |lhs|, |rhs|)."""

    def __init__(self, difference: RealFunction, sign: str, scale: Optional[RealFunction] = None) -> None:
        self.difference = difference
        self.sign = sign
        self.scale = scale

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
    """Whether every relation holds at point; None where one is undefined."""
    result = True
    for relation in relations:
        value = _value(relation.difference, point)
        if value is None:
            return None
        result = result and _compare(value, relation.sign)
    return result


def holds_at_end(relations: Sequence[Relation], point: float) -> Optional[bool]:
    """holds() at a candidate end, reading a value within rounding of 0 as a root (which a
    non-strict sign includes) and a huge value beside much smaller neighbours as a pole."""
    result = True
    for relation in relations:
        value = _value(relation.difference, point)
        if value is None or _is_pole(relation.difference, point, value):
            return None
        if abs(value) <= _root_tolerance(relation, point):
            result = result and not relation.strict
        else:
            result = result and _compare(value, relation.sign)
    return result


def _is_pole(function: RealFunction, point: float, value: float) -> bool:
    if not math.isfinite(value) or abs(value) < POLE_MINIMUM:
        return False
    step = NEIGHBOUR_STEP * max(1.0, abs(point))
    neighbours = [_value(function, point - step), _value(function, point + step)]
    if any(neighbour is None for neighbour in neighbours):
        return True
    largest = max(1.0, *(abs(neighbour) for neighbour in neighbours if neighbour is not None))
    return abs(value) > POLE_RATIO * largest


def _root_tolerance(relation: Relation, point: float) -> float:
    """ROOT_TOLERANCE times the size of the sides, plus the change of lhs - rhs over the rounding of point."""
    scale = 1.0
    if relation.scale is not None:
        size = relation.scale(point)
        if size is not None and math.isfinite(size):
            scale = max(1.0, size)
    step = NEIGHBOUR_STEP * max(1.0, abs(point))
    below, above = _value(relation.difference, point - step), _value(relation.difference, point + step)
    slope = 0.0
    if below is not None and above is not None and math.isfinite(below) and math.isfinite(above):
        slope = abs(above - below) / (2 * step)
    return ROOT_TOLERANCE * scale + 1e-12 * slope * max(1.0, abs(point))


# ----- the grid ---------------------------------------------------------------------------


def sample_grid(window: Optional[Window]) -> List[float]:
    """The points both the check and the sign chart sample, sorted."""
    if window is not None and math.isfinite(window.lower) and math.isfinite(window.upper):
        step = (window.upper - window.lower) / (GRID_POINTS - 1)
        return [window.lower + index * step for index in range(GRID_POINTS - 1)] + [window.upper]
    top = math.asinh(GRID_LIMIT)
    half = (GRID_POINTS - 1) // 2
    inner = [math.sinh(top * index / half) for index in range(-half, half + 1)]
    ratio = (TAIL_LIMIT / GRID_LIMIT) ** (1.0 / TAIL_POINTS)
    tail = [GRID_LIMIT * ratio**index for index in range(1, TAIL_POINTS + 1)]
    points = [-value for value in reversed(tail)] + inner + tail
    if window is not None:
        points = [point for point in points if window.contains(point)]
        points += [end for end in (window.lower, window.upper) if math.isfinite(end) and window.contains(end)]
    return sorted(points)


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


def _grid_states(
    relations: Sequence[Relation], points: Sequence[float]
) -> Tuple[List[List[Optional[float]]], List[Optional[bool]]]:
    """(each relation's values at the points, whether the inequality holds there: None where undefined)."""
    columns = [[_value(relation.difference, point) for point in points] for relation in relations]
    if len(relations) == 1:
        column, sign = columns[0], relations[0].sign
        return columns, [None if value is None else _compare(value, sign) for value in column]
    states: List[Optional[bool]] = []
    for index in range(len(points)):
        state: Optional[bool] = True
        for relation, column in zip(relations, columns):
            value = column[index]
            if value is None:
                state = None
                break
            if state and not _compare(value, relation.sign):
                state = False
        states.append(state)
    return columns, states


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
    of the set where the inequality is undefined refutes it). Each finite end, each window
    end and each seed (a polynomial root, say) must be in the set exactly when the
    inequality holds there (a root belongs to it for <= and >=), and the points just beside
    each end must agree with the set too. "unchecked" when the inequality could not be
    evaluated anywhere.
    """
    ends = [end for piece in pieces for end in (piece.lower, piece.upper) if math.isfinite(end)]
    window_ends = [end for end in (window.lower, window.upper) if math.isfinite(end)] if window is not None else []
    skipped = ends + window_ends
    points = [point for point in sample_grid(window) if not _near_end(skipped, point)]
    states = _grid_states(relations, points)[1]
    checked = 0
    for state, inside in zip(states, _membership(pieces, points)):
        if state is None:
            if inside:
                return "refuted"
            continue
        checked += 1
        if state != inside:
            return "refuted"
    if checked == 0:
        return "unchecked"
    # A seed is a numeric root, off by rounding: one beside an end is the end
    candidates = list(ends) + [
        point for point in seeds if (window is None or window.contains(point)) and not _near_end(ends, point)
    ]
    if window is not None:
        candidates += [end for end in (window.lower, window.upper) if math.isfinite(end) and window.contains(end)]
    for point in candidates:
        expected = holds_at_end(relations, point)
        if bool(expected) != _contained(pieces, point):
            return "refuted"
    for end in ends:
        step = NEIGHBOUR_STEP * max(1.0, abs(end))
        for point in (end - step, end + step):
            if window is not None and not window.contains(point):
                continue
            state = holds(relations, point)
            if state is None and _contained(pieces, point):
                return "refuted"
            if state is not None and state != _contained(pieces, point):
                return "refuted"
    return "confirmed"


# ----- the sign chart ---------------------------------------------------------------------


class SignChartFailure(ValueError):
    """The sign chart could not find a reliable answer (the message says why)."""


def sign_chart(
    relations: Sequence[Relation], window: Optional[Window] = None, seeds: Sequence[float] = ()
) -> List[Piece]:
    """The inequality's solution set found numerically on the grid (and seeds, e.g. polynomial roots).

    Raises:
        SignChartFailure: when the inequality cannot be evaluated, its sign keeps changing far
            out (no reliable outer interval) or the set has more than MAX_PIECES intervals.
    """
    points = sorted(set(sample_grid(window)) | {seed for seed in seeds if window is None or window.contains(seed)})
    values, states = _grid_states(relations, points)
    if all(state is None for state in states):
        raise SignChartFailure("the inequality cannot be evaluated")
    if window is None or not (math.isfinite(window.lower) and math.isfinite(window.upper)):
        _check_tails(states)
    points, states, touching = _with_touching_points(relations, points, states, values)
    boundaries = _boundaries(relations, points, states, touching)
    pieces = _pieces(relations, boundaries, points, states, window)
    if len(pieces) > MAX_PIECES:
        raise SignChartFailure(f"the solution set has more than {MAX_PIECES} intervals; give a smaller interval")
    return pieces


def _check_tails(states: List[Optional[bool]]) -> None:
    for side in (states[:TAIL_AGREEMENT], states[-TAIL_AGREEMENT:]):
        if len(set(side)) > 1:
            raise SignChartFailure("its sign keeps changing far from 0")


def _with_touching_points(
    relations: Sequence[Relation],
    points: List[float],
    states: List[Optional[bool]],
    values: List[List[Optional[float]]],
) -> Tuple[List[float], List[Optional[bool]], List[float]]:
    """points and states with the touching roots added: where |lhs - rhs| dips to 0 without a sign
    change ((x-1)^2 > 0 at 1), found by golden-section search between samples."""
    found: List[float] = []
    searches = 0
    for relation, column in zip(relations, values):
        sizes = [math.inf if value is None else abs(value) for value in column]
        for index in range(1, len(points) - 1):
            middle = sizes[index]
            if not (middle < sizes[index - 1] and middle < sizes[index + 1]):
                continue  # not a dip (an undefined or infinite neighbour counts as infinitely large)
            if sizes[index - 1] == math.inf or sizes[index + 1] == math.inf or states[index - 1] != states[index + 1]:
                continue
            searches += 1
            if searches > MAX_TOUCH_SEARCHES:
                break
            candidate = _golden_minimum(relation.difference, points[index - 1], points[index + 1])
            if candidate is None:
                continue
            nice = nice_number(candidate, TOUCH_NICE_TOLERANCE)
            if nice is not None:
                candidate = nice[1]
            if holds_at_end(relations, candidate) != states[index]:
                found.append(candidate)
    if not found:
        return points, states, []
    merged = sorted(set(points) | set(found))
    touching = set(found)
    merged_states = [
        holds_at_end(relations, point) if point in touching else holds(relations, point) for point in merged
    ]
    return merged, merged_states, sorted(touching)


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


class _Boundary:
    """A point where the inequality's state changes, its end text and whether it belongs to the set."""

    def __init__(self, value: float, text: str, state: Optional[bool]) -> None:
        self.value = value
        self.text = text
        self.state = state


def _boundaries(
    relations: Sequence[Relation], points: List[float], states: List[Optional[bool]], touching: List[float]
) -> List[_Boundary]:
    """Every state change between neighbouring samples, located by bisection and snapped to a nice value."""
    raw: List[Tuple[float, float]] = []  # (value, nice tolerance)
    for index in range(len(points) - 1):
        if states[index] != states[index + 1]:
            for value in _transitions(relations, points[index], states[index], points[index + 1], states[index + 1], 0):
                near_touch = any(abs(value - point) <= 1e-12 * max(1.0, abs(point)) for point in touching)
                raw.append((value, TOUCH_NICE_TOLERANCE if near_touch else NICE_TOLERANCE))
    boundaries: List[_Boundary] = []
    for value, tolerance in _collapse_narrow_gaps(sorted(raw)):
        nice = nice_number(value, tolerance)
        exact = nice[1] if nice is not None else value
        if boundaries and abs(boundaries[-1].value - exact) <= 1e-12 * max(1.0, abs(exact)):
            continue
        text = nice[0] if nice is not None else number_text(value)
        boundaries.append(_Boundary(exact, text, holds_at_end(relations, exact)))
    return boundaries


def _collapse_narrow_gaps(raw: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Two boundaries closer than NEIGHBOUR_STEP (relative) as one point between them.

    Near a touching root lhs - rhs can round to exactly 0 over a short stretch (2*sin(x)^2 -
    sin(x) - 1 within 1e-8 of pi/2), which bisection reports as two boundaries.
    """
    collapsed: List[Tuple[float, float]] = []
    for value, tolerance in raw:
        if collapsed and value - collapsed[-1][0] <= NEIGHBOUR_STEP * max(1.0, abs(value)):
            previous = collapsed.pop()
            collapsed.append(((previous[0] + value) / 2, TOUCH_NICE_TOLERANCE))
            continue
        collapsed.append((value, tolerance))
    return collapsed


def _transitions(
    relations: Sequence[Relation],
    low: float,
    low_state: Optional[bool],
    high: float,
    high_state: Optional[bool],
    depth: int,
) -> List[float]:
    """The points between low and high where the state changes (two when a third state lies between)."""
    for _ in range(200):
        middle = (low + high) / 2
        if middle <= low or middle >= high or high - low <= 1e-15 * max(1.0, abs(low), abs(high)):
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
    return [(low + high) / 2]


def _pieces(
    relations: Sequence[Relation],
    boundaries: List[_Boundary],
    points: List[float],
    states: List[Optional[bool]],
    window: Optional[Window],
) -> List[Piece]:
    """The set from the boundaries: alternating open stretches and boundary points, each in or out."""
    lower = window.lower if window is not None else -math.inf
    upper = window.upper if window is not None else math.inf
    inner = [boundary for boundary in boundaries if lower < boundary.value < upper]
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
    return bool(holds(relations, (start + end) / 2))


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
) -> str:
    """The solution set as the model sees it.

    "2 < x < 3; interval notation: (2, 3)", "x <= -1 or x >= 3; interval notation: ...",
    "x = 0; interval notation: {0}", "No real x satisfies x^2 + 1 < 0 (the solution set is
    empty).", "Every real x satisfies x^2 + 1 > 0; interval notation: (-Infinity, Infinity)".
    A window adds "(on [0, 2*pi])", and with a period "(on [-2*pi, 2*pi]; the pattern
    repeats every 2*pi)". Ends that are not plain numbers get decimals: "; endpoints ≈ ...".
    """
    where = f"{variable} in {window.text()}" if window is not None else f"real {variable}"
    scope = ""
    if window is not None:
        scope = f" (on {window.text()}; the pattern repeats every {period})" if period else f" (on {window.text()})"
    if not pieces:
        return f"No {where} satisfies {relation_text} (the solution set is empty){scope}."
    domain = Piece(
        window.lower if window else -math.inf,
        window.upper if window else math.inf,
        window.lower_closed if window else False,
        window.upper_closed if window else False,
    )
    if len(pieces) == 1 and _same_piece(pieces[0], domain):
        return f"Every {where} satisfies {relation_text}{scope}; interval notation: {interval_notation(pieces)}"
    text = " or ".join(_piece_inequality(piece, variable) for piece in pieces)
    missing = _missing_points(pieces, window)
    if missing:
        text += f" (every {where} except {', '.join(missing)})"
    text += f"{scope}; interval notation: {interval_notation(pieces)}"
    return text + _approximations(pieces)


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
