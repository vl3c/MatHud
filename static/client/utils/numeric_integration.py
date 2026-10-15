"""Pure-Python numeric integration helpers.

This module intentionally has no browser/Brython dependencies so it can be
validated via server-side pytest suites.

Before integrating, the integrand is scanned for interior singularities (points
where it is undefined or grows without bound); integrating across one would
return a meaningless finite number, so ``integrate`` raises a ValueError instead.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple, TypedDict

DEFAULT_METHOD = "simpson"
DEFAULT_STEPS = 200

# An integrand growing by more than this factor between half a step and
# ENDPOINT_PROBE_FRACTION of a step from an endpoint is treated as singular there
ENDPOINT_PROBE_FRACTION = 1e-4
ENDPOINT_GROWTH_LIMIT = 10.0

# Singularity scan: at least this many grid intervals; only local maxima of |f| above
# SINGULARITY_PEAK_FACTOR times its median are examined (at most SINGULARITY_MAX_PEAKS);
# a peak whose zoomed-in maximum exceeds BLOW_UP_FACTOR times its grid value is a pole
SINGULARITY_SCAN_MIN_INTERVALS = 200
SINGULARITY_PEAK_FACTOR = 5.0
SINGULARITY_MAX_PEAKS = 5
BLOW_UP_FACTOR = 1e6
# Removable points (0/0 such as sin(x)/x at 0) are probed this fraction of the interval away
REMOVABLE_PROBE_FRACTION = 1e-7

EvalFn = Callable[[float], float]


class _NumericIntegrationResultBase(TypedDict):
    method: str
    steps: int
    value: float
    error_estimate: float


class NumericIntegrationResult(_NumericIntegrationResultBase, total=False):
    """Result payload for numeric integration.

    ``warning`` is present when the value or error estimate is likely unreliable
    (for example an endpoint singularity, or the midpoint fallback used for it).
    """

    warning: str


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _safe_eval(eval_fn: EvalFn, x: float) -> float:
    y = eval_fn(float(x))
    if isinstance(y, bool) or not isinstance(y, (int, float)):
        raise TypeError("integrand must evaluate to a number")
    y = float(y)
    if not math.isfinite(y):
        raise ValueError("integrand produced a non-finite value")
    return y


def _try_eval(eval_fn: EvalFn, x: float) -> Optional[float]:
    """Return f(x), or None where the integrand fails, is not a real number or is not finite."""
    try:
        return _safe_eval(eval_fn, x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Undefined points: removable or singular
# ---------------------------------------------------------------------------


def _probe_step(x: float, width: float) -> float:
    return max(REMOVABLE_PROBE_FRACTION * width, 1e-12 * max(1.0, abs(x)))


def _removable_limit(eval_fn: EvalFn, x: float, width: float, typical: float) -> Optional[float]:
    """Return the limit of f at an interior x where f is undefined but continuous around it, else None.

    f is probed at x +/- h and x +/- 2h: a removable point gives four nearly equal finite
    values, while a pole or a jump makes them differ markedly.
    """
    h = _probe_step(x, width)
    probes = [_try_eval(eval_fn, x + offset) for offset in (-2 * h, -h, h, 2 * h)]
    finite = [p for p in probes if p is not None]
    if len(finite) != len(probes):
        return None
    if max(finite) - min(finite) > 1e-3 * max(typical, max(abs(p) for p in finite)):
        return None
    return (finite[1] + finite[2]) / 2


def _one_sided_limit(eval_fn: EvalFn, x: float, direction: float, width: float, typical: float) -> Optional[float]:
    """Return the limit of f at an endpoint x where f is undefined, approached from inside, else None."""
    h = _probe_step(x, width)
    near = _try_eval(eval_fn, x + direction * h)
    far = _try_eval(eval_fn, x + direction * 2 * h)
    if near is None or far is None:
        return None
    if abs(near - far) > 1e-3 * max(typical, abs(near), abs(far)):
        return None
    return 2 * near - far


def _typical_magnitude(values: List[Optional[float]]) -> float:
    finite = sorted(abs(v) for v in values if v is not None)
    return finite[len(finite) // 2] if finite else 0.0


def _sample_grid(eval_fn: EvalFn, a: float, b: float, intervals: int) -> Tuple[List[float], List[Optional[float]]]:
    """Sample f on a uniform grid of the interval; repair removable points with their limits.

    Interior values stay None where f is undefined and not removable; an endpoint
    value stays None where f has no finite one-sided limit there.
    """
    h = (b - a) / intervals
    xs = [a + i * h for i in range(intervals)] + [b]
    values = [_try_eval(eval_fn, x) for x in xs]
    if all(v is not None for v in values):
        return xs, values
    typical = _typical_magnitude(values)
    width = b - a
    for i in range(1, intervals):
        if values[i] is None:
            values[i] = _removable_limit(eval_fn, xs[i], width, typical)
    if values[0] is None:
        values[0] = _one_sided_limit(eval_fn, a, 1.0, width, typical)
    if values[-1] is None:
        values[-1] = _one_sided_limit(eval_fn, b, -1.0, width, typical)
    return xs, values


def _refine_blow_up(
    magnitude: Callable[[float], Optional[float]], lo: float, hi: float, grid_value: float
) -> Optional[float]:
    """Ternary-search the maximum of |f| on [lo, hi]; return its location if it grows without bound.

    magnitude(x) returns |f(x)|, or None where f is undefined.
    """
    for _ in range(100):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        v1 = magnitude(m1)
        if v1 is None:
            return m1
        v2 = magnitude(m2)
        if v2 is None:
            return m2
        if v1 < v2:
            lo = m1
        else:
            hi = m2
    peak_location = (lo + hi) / 2
    peak = magnitude(peak_location)
    if peak is None or peak > BLOW_UP_FACTOR * max(1.0, grid_value):
        return peak_location
    return None


def _locate_singularity(eval_fn: EvalFn, xs: List[float], values: List[Optional[float]]) -> Optional[float]:
    """Return the location of an interior point where f is undefined or blows up, or None."""
    intervals = len(xs) - 1
    for k in range(1, intervals):
        if values[k] is None:
            return xs[k]

    a, b = xs[0], xs[-1]
    width = b - a
    typical = _typical_magnitude(values)
    mags = [math.inf if v is None else abs(v) for v in values]

    def magnitude(x: float) -> Optional[float]:
        value = _try_eval(eval_fn, x)
        if value is None:
            value = _removable_limit(eval_fn, x, width, typical)
        return None if value is None else abs(value)

    # Zoom into the largest local maxima of |f|: a pole keeps growing, a smooth peak does not
    peaks = [
        k
        for k in range(1, intervals)
        if mags[k] >= mags[k - 1] and mags[k] >= mags[k + 1] and mags[k] > SINGULARITY_PEAK_FACTOR * typical
    ]
    peaks.sort(key=lambda k: mags[k], reverse=True)
    step = width / intervals
    for k in peaks[:SINGULARITY_MAX_PEAKS]:
        location = _refine_blow_up(magnitude, xs[k - 1], xs[k + 1], mags[k])
        if location is not None and min(location - a, b - location) > step * 1e-6:
            return location
    return None


def find_interior_singularity(
    eval_fn: EvalFn, lower_bound: float, upper_bound: float, intervals: int = 2000
) -> Optional[float]:
    """Return the approximate location of an interior point where eval_fn is undefined or blows up.

    Scans a uniform grid of ``intervals`` intervals, skips removable points such as
    sin(x)/x at 0, and zooms into peaks of |f| to tell poles (tan(x) at pi/2) from
    smooth maxima. Returns None when the integrand looks finite inside the interval;
    points exactly at a bound are ignored.
    """
    a, b = sorted((float(lower_bound), float(upper_bound)))
    if a == b:
        return None
    xs, values = _sample_grid(eval_fn, a, b, intervals)
    return _locate_singularity(eval_fn, xs, values)


# ---------------------------------------------------------------------------
# Quadrature rules
# ---------------------------------------------------------------------------


def _value_or_limit(eval_fn: EvalFn, x: float, width: float) -> float:
    value = _try_eval(eval_fn, x)
    if value is None:
        value = _removable_limit(eval_fn, x, width, 0.0)
    if value is None:
        raise ValueError(f"integrand is undefined or not finite at {x:.6g}")
    return value


def _midpoint(eval_fn: EvalFn, a: float, b: float, steps: int) -> float:
    h = (b - a) / steps
    return math.fsum(_value_or_limit(eval_fn, a + (i + 0.5) * h, b - a) for i in range(steps)) * h


def _trapezoid_from_nodes(nodes: List[float], a: float, b: float, stride: int) -> float:
    picked = nodes[::stride]
    h = (b - a) / (len(picked) - 1)
    return (0.5 * (picked[0] + picked[-1]) + math.fsum(picked[1:-1])) * h


def _simpson_from_nodes(nodes: List[float], a: float, b: float, stride: int) -> float:
    picked = nodes[::stride]
    h = (b - a) / (len(picked) - 1)
    odd = math.fsum(picked[1:-1:2])
    even = math.fsum(picked[2:-1:2])
    return (picked[0] + picked[-1] + 4 * odd + 2 * even) * h / 3.0


def _endpoint_singularity_warning(
    eval_fn: EvalFn, a: float, b: float, h: float, value: float, labels: Tuple[str, str]
) -> Optional[str]:
    """Detect an integrand that blows up (or fails) near an endpoint.

    The coarse/fine error estimate assumes a smooth integrand and badly
    underestimates the error near an integrable singularity such as 1/sqrt(x).
    """
    # Average magnitude floor keeps zero crossings near an endpoint from being flagged
    typical = abs(value) / (b - a)
    for endpoint, direction, label in ((a, 1.0, labels[0]), (b, -1.0, labels[1])):
        try:
            near = abs(_safe_eval(eval_fn, endpoint + direction * 0.5 * h))
            closer = abs(_safe_eval(eval_fn, endpoint + direction * ENDPOINT_PROBE_FRACTION * h))
        except Exception:
            closer, near = math.inf, 0.0
        if closer > ENDPOINT_GROWTH_LIMIT * max(near, typical):
            return (
                f"The integrand is very large or not finite near the {label} bound "
                "(possible singularity); the value and error estimate may be inaccurate."
            )
    return None


def _interior_singularity_error(location: float, a: float, b: float) -> ValueError:
    if abs(location) <= 1e-9 * (b - a):
        location = 0.0  # a pole at 0 is located only to within round-off
    return ValueError(
        f"The integrand is singular or undefined near {location:.6g}, inside the integration interval "
        f"[{a:.6g}, {b:.6g}], so this is an improper integral that may diverge and no finite value is "
        "returned. Integrate up to that point from each side separately to check whether the parts converge."
    )


def _integrate_ordered(
    eval_fn: EvalFn, a: float, b: float, method: str, steps: int, labels: Tuple[str, str]
) -> NumericIntegrationResult:
    """Integrate over [a, b] with a < b; labels name the bounds at a and b for warnings."""
    coarse_steps = steps + steps % 2 if method == "simpson" else steps
    fine_steps = 2 * coarse_steps

    xs, nodes = _sample_grid(eval_fn, a, b, fine_steps)
    if all(value is None for value in nodes):
        # Undefined everywhere (e.g. an unknown function name): raise the integrand's own error
        _safe_eval(eval_fn, (a + b) / 2)
    if fine_steps >= SINGULARITY_SCAN_MIN_INTERVALS:
        location = _locate_singularity(eval_fn, xs, nodes)
    else:
        location = find_interior_singularity(eval_fn, a, b, SINGULARITY_SCAN_MIN_INTERVALS)
    if location is not None:
        raise _interior_singularity_error(location, a, b)

    undefined_bounds = [label for label, value in zip(labels, (nodes[0], nodes[-1])) if value is None]
    if method == "midpoint" or undefined_bounds:
        coarse = _midpoint(eval_fn, a, b, steps)
        fine = _midpoint(eval_fn, a, b, 2 * steps)
        result = NumericIntegrationResult(
            method="midpoint", steps=2 * steps, value=fine, error_estimate=abs(fine - coarse) / 3.0
        )
        if undefined_bounds:
            result["warning"] = (
                f"The integrand is undefined at the {' and '.join(undefined_bounds)} bound, so the midpoint rule "
                f"(which never evaluates the bounds) was used instead of {method}; the value is approximate, "
                "and the integral may be improper or divergent."
            )
            return result
    else:
        # No value is None here: interior points passed the scan and both bounds are defined
        defined = [v for v in nodes if v is not None]
        if method == "trapezoid":
            coarse = _trapezoid_from_nodes(defined, a, b, 2)
            fine = _trapezoid_from_nodes(defined, a, b, 1)
            err = abs(fine - coarse) / 3.0
        else:
            coarse = _simpson_from_nodes(defined, a, b, 2)
            fine = _simpson_from_nodes(defined, a, b, 1)
            err = abs(fine - coarse) / 15.0
        result = NumericIntegrationResult(method=method, steps=fine_steps, value=fine, error_estimate=err)

    warning = _endpoint_singularity_warning(eval_fn, a, b, (b - a) / result["steps"], result["value"], labels)
    if warning:
        result["warning"] = warning
    return result


def integrate(
    eval_fn: EvalFn,
    lower_bound: float,
    upper_bound: float,
    method: Optional[str] = DEFAULT_METHOD,
    steps: Optional[int] = DEFAULT_STEPS,
) -> NumericIntegrationResult:
    """Numerically integrate an integrand function over a finite interval.

    ``method`` (trapezoid, midpoint or simpson) and ``steps`` default when None.
    Equal bounds give 0 and reversed bounds the negated value. Raises ValueError
    when the integrand is singular or undefined inside the interval. Where it is
    undefined at a bound, trapezoid and simpson fall back to the midpoint rule and
    the result carries a warning.
    """
    if not callable(eval_fn):
        raise TypeError("eval_fn must be callable")

    a = _require_finite(lower_bound, "lower_bound")
    b = _require_finite(upper_bound, "upper_bound")

    steps = _require_positive_int(DEFAULT_STEPS if steps is None else steps, "steps")
    method = str(DEFAULT_METHOD if method is None else method).strip().lower()
    if method not in ("trapezoid", "midpoint", "simpson"):
        raise ValueError("method must be one of: trapezoid, midpoint, simpson")

    if a == b:
        return NumericIntegrationResult(method=method, steps=0, value=0.0, error_estimate=0.0)
    if a > b:
        result = _integrate_ordered(eval_fn, b, a, method, steps, ("upper", "lower"))
        result["value"] = -result["value"]
        return result
    return _integrate_ordered(eval_fn, a, b, method, steps, ("lower", "upper"))
