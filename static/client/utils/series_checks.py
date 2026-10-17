"""Numeric checks of series results: is a polynomial the Taylor polynomial of a function,
and is a closed form the sum of its terms (see the sums section below)?

taylor_matches compares f and P on each side of the centre a at the steps h = START_STEP,
START_STEP/2, ... (HALVINGS of them). When P is f's Taylor polynomial of order n, the error
E(h) = |f(a + h) - P(a + h)| shrinks at least like h^(n+1), so each halving divides it by
2^(n+1) or more; a wrong coefficient of degree k <= n leaves an error like h^k, which a
halving divides by 2^k only. The ratios are read at the smallest steps where the error is
still well above rounding noise: closer to a the error is rounding, farther away the higher
terms (or a small radius of convergence, 1/(1+25x^2) at 0) still blur the ratio. Errors
that break off from far above rounding to rounding in one halving tell nothing: no power
of h falls that fast, so the function changes form beyond the centre (abs(x) at 1 is
1 + (x - 1) right up to x = 0, where it turns). And once the errors have settled to
rounding, signal closer to a is cancellation in f, not the remainder, which only shrinks
toward a: (e^x - 1)/x rounds e^h to 1 + h at small h, an error like h that halving
divides by 2 only. The errors are read up to that point. They are also read only up to
where f stops resolving the step (the same value at two halvings, as cos(h) rounds to
exactly 1 - h^2/2 in (1 - cos(x))/x^2), and up to a small error that grows toward a,
which is rounding in f ((2^x - 1)/x); after such growth the check never refutes, since
the errors left may already carry that rounding.

Evaluators take a float (an int index for sums) and return a float, or None where the
function is not real or cannot be computed. Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple

from utils.equivalence import rounding_noise

RealFunction = Callable[[float], Optional[float]]
# (halving, |f - P|, that error over the rounding noise, that error over max(1, |f|, |P|), f)
ErrorRow = Tuple[int, float, float, float, float]

# The steps from the centre: START_STEP * 2^-k for k = 0 .. HALVINGS - 1.
START_STEP = 8.0
HALVINGS = 48
# An error counts when it is this many times the rounding noise of the values.
SIGNAL = 100.0
# An error this many times the rounding noise cannot reach rounding in one halving: a power
# of h up to h^11 divides it by 2048 at most.
BREAK_OFF = SIGNAL**3
# Errors this many halvings before one that grows are left out too: the rounding in f
# that grows is already part of them.
GROWTH_MARGIN = 2
# An error that grows counts as rounding in f only below this relative size; a larger one
# is the error of a wrong polynomial past a zero (|0.1 - x^2/2| beyond x = 0.45).
ROUNDING_GROWTH = 1e-7
# The errors have settled when this many halvings in a row are rounding.
SETTLED_STEPS = 2
# The quick noise estimate: this many units in the last place of the largest value.
NOISE_ULPS = 64
EPSILON = 2.0**-52
# Pass: the last two ratios are at most PASS_SLACK * 2^-(n+1). Fail: both above
# FAIL_SLACK * 2^-n, the ratio of an error like h^n. In between, the check cannot tell.
PASS_SLACK = 1.5
FAIL_SLACK = 0.85
# An exact match needs this many steps where both are defined and agree to rounding.
MIN_EXACT_STEPS = 8


def taylor_matches(function: RealFunction, polynomial: RealFunction, center: float, order: int) -> Optional[bool]:
    """True when polynomial agrees with function to order + 1 at center on each side where
    both are defined; False when it agrees to order at most (a coefficient of degree <= order
    is wrong) on one side; None when the errors do not tell (the function is undefined
    beside the centre, or the errors stay too close to rounding)."""
    verdicts = [_side_verdict(function, polynomial, center, order, side) for side in (1.0, -1.0)]
    if False in verdicts:
        return False
    return True if True in verdicts else None


def _side_verdict(
    function: RealFunction, polynomial: RealFunction, center: float, order: int, side: float
) -> Optional[bool]:
    errors = _settled(_errors(function, polynomial, center, side))
    stuck = _stuck(errors)
    if stuck is not None:
        errors = errors[: max(stuck - GROWTH_MARGIN, 0)]
    growth = _growth(errors)
    if growth is not None:
        # f's own rounding grows toward the centre: read the errors before it, and never refute
        verdict = _ratio_verdict(errors[: max(growth - GROWTH_MARGIN, 0)], order)
        return True if verdict else None
    return _ratio_verdict(errors, order)


def _ratio_verdict(errors: List[ErrorRow], order: int) -> Optional[bool]:
    if not errors:
        return None
    run = _last_signal_run(errors)
    if not run:
        exact = sum(1 for row in errors if row[2] <= SIGNAL)
        return True if exact >= MIN_EXACT_STEPS else None
    if len(run) < 3 or _breaks_off(errors):
        return None
    ratios = [run[index + 1] / run[index] for index in range(len(run) - 3, len(run) - 1)]
    if all(ratio <= PASS_SLACK * 2.0 ** -(order + 1) for ratio in ratios):
        return True
    if all(ratio > FAIL_SLACK * 2.0**-order for ratio in ratios):
        return False
    return None


def _growth(errors: List[ErrorRow]) -> Optional[int]:
    """The index of the first signal error that is larger than the error one halving
    farther out while still a rounding-sized error (ROUNDING_GROWTH), or None: the remainder
    only shrinks toward the centre, so such an error is rounding in f, as in (2^x - 1)/x."""
    for index in range(1, len(errors)):
        before, error_before, margin_before, _, _ = errors[index - 1]
        halving, error, margin, relative, _ = errors[index]
        if halving != before + 1 or margin_before <= SIGNAL or margin <= SIGNAL:
            continue
        if error > error_before and relative < ROUNDING_GROWTH:
            return index
    return None


def _stuck(errors: List[ErrorRow]) -> Optional[int]:
    """The index of the first signal error where f has the same value as one halving
    farther out, or None: f no longer resolves the step there ((1 - cos(x))/x^2 is exactly
    1/2 once cos(h) rounds to 1 - h^2/2), and the errors from there on are its rounding."""
    for index in range(1, len(errors)):
        before, after = errors[index - 1], errors[index]
        if after[0] == before[0] + 1 and after[2] > SIGNAL and after[4] == before[4]:
            return index
    return None


def _settled(errors: List[ErrorRow]) -> List[ErrorRow]:
    """The errors up to the first signal after SETTLED_STEPS halvings of rounding in a row."""
    quiet = 0
    for index, row in enumerate(errors):
        if row[2] <= SIGNAL:
            quiet += 1
        elif quiet >= SETTLED_STEPS:
            return errors[:index]
        else:
            quiet = 0
    return errors


def _breaks_off(errors: List[ErrorRow]) -> bool:
    """Whether a run of signal errors ends at an error BREAK_OFF times the rounding noise,
    and the next halving's error is rounding."""
    last: Optional[ErrorRow] = None
    for row in errors:
        if last is not None and last[2] > SIGNAL and row[2] <= SIGNAL:
            if row[0] == last[0] + 1 and last[2] > BREAK_OFF:
                return True
        last = row
    return False


def _errors(function: RealFunction, polynomial: RealFunction, center: float, side: float) -> List[ErrorRow]:
    """An ErrorRow at each step where both are defined and finite."""
    errors: List[ErrorRow] = []
    for halving in range(HALVINGS):
        point = center + side * START_STEP * 2.0**-halving
        if point == center:
            break
        value, approximation = _value(function, point), _value(polynomial, point)
        if value is None or approximation is None:
            continue
        error = abs(value - approximation)
        if not math.isfinite(error):
            continue
        margin = _margin(function, polynomial, point, value, approximation, error)
        errors.append((halving, error, margin, error / max(1.0, abs(value), abs(approximation)), value))
    return errors


def _margin(
    function: RealFunction, polynomial: RealFunction, point: float, value: float, approximation: float, error: float
) -> float:
    """error over the rounding noise: first NOISE_ULPS units in the last place of the values,
    then, when the error is SIGNAL times that, the larger of it and how much each side
    scatters when the point is nudged (equivalence.rounding_noise), which a polynomial with
    large cancelling terms needs."""
    floor = NOISE_ULPS * EPSILON * max(1.0, abs(value), abs(approximation))
    if error <= SIGNAL * floor:
        return error / floor

    def first(values: dict) -> Optional[float]:
        return function(values["x"])

    def second(values: dict) -> Optional[float]:
        return polynomial(values["x"])

    noise = rounding_noise(first, {"x": point}) + rounding_noise(second, {"x": point})
    return float(error / max(floor, noise))


def _last_signal_run(errors: List[ErrorRow]) -> List[float]:
    """The errors of the last run of consecutive halvings whose errors are signal."""
    run: List[float] = []
    previous: Optional[int] = None
    for halving, error, margin, _, _ in errors:
        if margin <= SIGNAL:
            continue
        if previous is None or halving != previous + 1:
            run = []
        run.append(error)
        previous = halving
    return run


def _value(function: RealFunction, point: float) -> Optional[float]:
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


# ----- sums --------------------------------------------------------------------------------
#
# A closed form of a sum is checked against the sum of its terms: numeric bounds term by
# term, a symbolic bound n at several n, an infinite bound against partial sums extrapolated
# with Aitken's delta-squared process. Term evaluators take the index (and n) and return a
# float, or None where the term is not a real number or cannot be computed.

TermFunction = Callable[[int], Optional[float]]

# Sums agree when they differ by at most this, relative to max(1, |sum|), plus rounding.
SUM_AGREEMENT = 1e-9
# Rounding allowance per term, relative to the largest term.
TERM_ROUNDING = 4 * EPSILON
# A symbolic bound n is checked at the first SYMBOLIC_POINTS values of n = 0, 1, ... where the
# bounds are integers and the range has at most MAX_RANGE_TERMS terms, plus one with about
# LONG_RANGE_TERMS terms, and needs MIN_SYMBOLIC_POINTS comparisons.
SYMBOLIC_POINTS = 6
MAX_RANGE_TERMS = 200
LONG_RANGE_TERMS = 50
MIN_SYMBOLIC_POINTS = 3
MAX_BOUND_SYMBOL = 400
# Partial sums of an infinite series are taken after these numbers of terms (16000 terms
# take about 60 ms in math.js). Aitken's process applied twice to the doublings leaves an
# error of 1e-12 on 1/k^2 (once: 1.5e-6, too much to refute pi^2/6 + 1e-4).
PARTIAL_COUNTS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)
# A term that is not finite ends the partial sums when the last TAIL_TERMS terms were all
# below TAIL_SIZE times the sum (x^k/k! once k! overflows), and spoils them otherwise.
TAIL_TERMS = 20
TAIL_SIZE = 1e-17
# An infinite sum matches its estimate within SERIES_PASS, relative, or PASS_ERRORS times the
# estimate's error, if that is at most SERIES_REFUTE, relative (the estimate of 1/k^1.1 is
# off by 2e-5 and would accept anything near it); it is refuted beyond SERIES_REFUTE,
# relative, and REFUTE_ERRORS times the error. In between the check cannot tell (a slow
# series such as 1/(k*log(k)^2), or sin(k)/k^2, whose sums do not settle geometrically).
SERIES_PASS = 1e-9
PASS_ERRORS = 10.0
SERIES_REFUTE = 1e-6
REFUTE_ERRORS = 100.0
# A series plainly converges when its terms have shrunk below CONVERGED_TERM times the sum
# and each doubling of the terms adds at most SHRINKING times what the one before added.
CONVERGED_TERM = 1e-6
SHRINKING = 0.6


class SeriesEstimate:
    """Partial sums of an infinite series and their extrapolated value.

    Attributes:
        sums: The partial sums after PARTIAL_COUNTS terms (averaged with the next one for
            an alternating series).
        raw: The same without the averaging.
        value: Aitken's extrapolation applied twice, to the last five sums.
        error: How much that differs from the same for the five before the last.
        last_term: The size of the last term summed.
    """

    __slots__ = ("sums", "raw", "value", "error", "last_term")

    def __init__(self, sums: List[float], raw: List[float], value: float, error: float, last_term: float) -> None:
        self.sums = sums
        self.raw = raw
        self.value = value
        self.error = error
        self.last_term = last_term


def finite_sum(term: TermFunction, low: int, high: int) -> Optional[Tuple[float, float]]:
    """(sum of term(k) for k = low .. high, the largest |term|), or None when a term is not a
    finite real number. An empty range (high < low) sums to 0."""
    values: List[float] = []
    for index in range(low, high + 1):
        value = _value_at(term, index)
        if value is None or not math.isfinite(value):
            return None
        values.append(value)
    return math.fsum(values), max((abs(value) for value in values), default=0.0)


def sums_agree(value: float, total: float, largest: float, count: int) -> bool:
    """Whether a closed form's value matches a sum of count terms (largest the largest |term|)."""
    allowance = SUM_AGREEMENT * max(1.0, abs(total)) + count * TERM_ROUNDING * largest
    return abs(value - total) <= allowance


def numeric_sum_verdict(value: float, term: TermFunction, low: int, high: int) -> Optional[bool]:
    """Whether value is the sum of term for k = low .. high; None when a term cannot be computed."""
    summed = finite_sum(term, low, high)
    if summed is None:
        return None
    return sums_agree(value, summed[0], summed[1], high - low + 1)


def symbolic_sum_verdict(
    closed: Callable[[int], Optional[float]],
    term: Callable[[int, int], Optional[float]],
    lower: Callable[[int], Optional[float]],
    upper: Callable[[int], Optional[float]],
) -> Optional[bool]:
    """Whether closed(n) is the sum of term(k, n) for k = lower(n) .. upper(n), at the n where
    both bounds are integers; None with fewer than MIN_SYMBOLIC_POINTS comparisons."""
    compared = 0
    for n in _bound_samples(lower, upper):
        low, high = int(_value_at(lower, n) or 0), int(_value_at(upper, n) or 0)
        summed = finite_sum(lambda k: term(k, n), low, high)
        value = _value_at(closed, n)
        if summed is None or value is None or not math.isfinite(value):
            continue
        if not sums_agree(value, summed[0], summed[1], high - low + 1):
            return False
        compared += 1
    return True if compared >= MIN_SYMBOLIC_POINTS else None


def _bound_samples(lower: Callable[[int], Optional[float]], upper: Callable[[int], Optional[float]]) -> List[int]:
    """The n to compare at: the first SYMBOLIC_POINTS with a range of 1 to MAX_RANGE_TERMS
    terms, and the first with at least LONG_RANGE_TERMS."""
    first: List[int] = []
    long: List[int] = []
    for n in range(MAX_BOUND_SYMBOL + 1):
        low, high = _value_at(lower, n), _value_at(upper, n)
        if low is None or high is None or not (float(low).is_integer() and float(high).is_integer()):
            continue
        count = high - low + 1
        if count < 1 or count > MAX_RANGE_TERMS:
            continue  # a closed form need not hold for an empty range: k/n from 1 to n is (n+1)/2, not 0 at n = 0
        if len(first) < SYMBOLIC_POINTS:
            first.append(n)
        elif count >= LONG_RANGE_TERMS:
            long.append(n)
            break
    return first + long


def series_estimate(term: TermFunction, low: int) -> Optional[SeriesEstimate]:
    """The partial sums of term for k = low, low + 1, ... and their extrapolated limit, or
    None when a term cannot be computed (or stops being finite while still significant)."""
    last = PARTIAL_COUNTS[-1]
    values: List[float] = []
    raw: List[float] = []
    sums: List[float] = []
    for offset in range(last + 1):
        value = _value_at(term, low + offset)
        if value is None:
            return None
        if not math.isfinite(value):
            total = math.fsum(values)
            if len(values) < TAIL_TERMS or any(abs(v) > TAIL_SIZE * abs(total) for v in values[-TAIL_TERMS:]):
                return None
            return SeriesEstimate([total] * len(PARTIAL_COUNTS), [total] * len(PARTIAL_COUNTS), total, 0.0, 0.0)
        values.append(value)
        if offset + 1 in PARTIAL_COUNTS:
            raw.append(math.fsum(values))
    # An alternating series is read at the mean of S_N and S_(N+1), whose error shrinks like
    # the change in its terms rather than like the terms ((-1)^k/k: 1/N^2, not 1/N).
    alternating = all(values[index] * values[index + 1] < 0 for index in range(last - 4, last))
    for index, count in enumerate(PARTIAL_COUNTS):
        sums.append(raw[index] + values[count] / 2 if alternating else raw[index])
    row = sums
    for _ in range(2):
        row = [_aitken(row[index], row[index + 1], row[index + 2]) for index in range(len(row) - 2)]
    return SeriesEstimate(sums, raw, row[-1], abs(row[-1] - row[-2]), abs(values[last - 1]))


def _aitken(first: float, second: float, third: float) -> float:
    """Aitken's delta-squared extrapolation of three partial sums (exact when they approach
    their limit geometrically, as c/N does at doublings of N)."""
    step, previous = third - second, second - first
    denominator = step - previous
    if denominator == 0 or not math.isfinite(denominator):
        return third
    return third - step * step / denominator


def series_verdict(value: float, estimate: SeriesEstimate) -> Optional[bool]:
    """Whether value is the limit the partial sums estimate; None when they cannot tell."""
    difference = abs(value - estimate.value)
    scale = max(1.0, abs(value))
    reliable = PASS_ERRORS * estimate.error <= SERIES_REFUTE * scale
    if reliable and difference <= max(SERIES_PASS * scale, PASS_ERRORS * estimate.error):
        return True
    if difference > max(SERIES_REFUTE * scale, REFUTE_ERRORS * estimate.error):
        return False
    return None


def divergence_verdict(limit: Optional[str], estimate: SeriesEstimate) -> Optional[bool]:
    """False when the partial sums contradict a claim that the series diverges (to limit,
    "Infinity" or "-Infinity", or without one): its terms have shrunk to nothing and each
    doubling adds less than SHRINKING times the one before, or the sums move the other way.
    None otherwise: a slowly diverging series (1/k) and a slowly converging one look alike.
    The sums are the raw ones: the mean of S_N and S_(N+1) is constant for (-1)^k, which
    diverges."""
    steps = [estimate.raw[index + 1] - estimate.raw[index] for index in range(len(estimate.raw) - 1)]
    if limit == "Infinity" and all(step < 0 for step in steps):
        return False
    if limit == "-Infinity" and all(step > 0 for step in steps):
        return False
    small = estimate.last_term <= CONVERGED_TERM * max(1.0, abs(estimate.value))
    shrinking = all(0 < abs(steps[index + 1]) <= SHRINKING * abs(steps[index]) for index in range(len(steps) - 1))
    if small and shrinking:
        return False
    return None


def _value_at(function: Callable[..., Optional[float]], *arguments: Any) -> Optional[float]:
    try:
        value = function(*arguments)
    except (ArithmeticError, ValueError, TypeError):
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
