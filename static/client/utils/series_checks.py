"""Numeric checks of series results: is a polynomial the Taylor polynomial of a function?

taylor_matches compares f and P on each side of the centre a at the steps h = START_STEP,
START_STEP/2, ... (HALVINGS of them). When P is f's Taylor polynomial of order n, the error
E(h) = |f(a + h) - P(a + h)| shrinks at least like h^(n+1), so each halving divides it by
2^(n+1) or more; a wrong coefficient of degree k <= n leaves an error like h^k, which a
halving divides by 2^k only. The ratios are read at the smallest steps where the error is
still well above rounding noise: closer to a the error is rounding, farther away the higher
terms (or a small radius of convergence, 1/(1+25x^2) at 0) still blur the ratio. Errors
that break off from far above rounding to rounding in one halving tell nothing: no power
of h falls that fast, so the function changes form beyond the centre (abs(x) at 1 is
1 + (x - 1) right up to x = 0, where it turns).

Evaluators take a float and return a float, or None where the function is not real or
cannot be computed. Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

from utils.equivalence import rounding_noise

RealFunction = Callable[[float], Optional[float]]

# The steps from the centre: START_STEP * 2^-k for k = 0 .. HALVINGS - 1.
START_STEP = 8.0
HALVINGS = 48
# An error counts when it is this many times the rounding noise of the values.
SIGNAL = 100.0
# An error this many times the rounding noise cannot reach rounding in one halving: a power
# of h up to h^11 divides it by 2048 at most.
BREAK_OFF = SIGNAL**3
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
    errors = _errors(function, polynomial, center, side)
    if not errors:
        return None
    run = _last_signal_run(errors)
    if not run:
        exact = sum(1 for _, _, margin in errors if margin <= SIGNAL)
        return True if exact >= MIN_EXACT_STEPS else None
    if len(run) < 3 or _breaks_off(errors):
        return None
    ratios = [run[index + 1] / run[index] for index in range(len(run) - 3, len(run) - 1)]
    if all(ratio <= PASS_SLACK * 2.0 ** -(order + 1) for ratio in ratios):
        return True
    if all(ratio > FAIL_SLACK * 2.0**-order for ratio in ratios):
        return False
    return None


def _breaks_off(errors: List[Tuple[int, float, float]]) -> bool:
    """Whether a run of signal errors ends at an error BREAK_OFF times the rounding noise,
    and the next halving's error is rounding."""
    last: Optional[Tuple[int, float, float]] = None
    for halving, error, margin in errors:
        if last is not None and last[2] > SIGNAL and margin <= SIGNAL:
            if halving == last[0] + 1 and last[2] > BREAK_OFF:
                return True
        last = (halving, error, margin)
    return False


def _errors(
    function: RealFunction, polynomial: RealFunction, center: float, side: float
) -> List[Tuple[int, float, float]]:
    """(halving, |f - P|, that error over the rounding noise) at each step where both are
    defined and finite."""
    errors: List[Tuple[int, float, float]] = []
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
        errors.append((halving, error, _margin(function, polynomial, point, value, approximation, error)))
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


def _last_signal_run(errors: List[Tuple[int, float, float]]) -> List[float]:
    """The errors of the last run of consecutive halvings whose errors are signal."""
    run: List[float] = []
    previous: Optional[int] = None
    for halving, error, margin in errors:
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
