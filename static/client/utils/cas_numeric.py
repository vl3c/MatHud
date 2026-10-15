"""Numeric algorithms that check and complete results from nerdamer, the symbolic engine.

- adaptive_integral: adaptive Gauss-Kronrod (7/15) quadrature, used to check a definite
  integral computed from an antiderivative and to stand in when there is no antiderivative.
- polynomial_roots: every complex root of a polynomial (Aberth-Ehrlich iteration), used
  when nerdamer's closed forms for a cubic or higher lose roots.
- format_number / format_complex / distinct_values: tidy numeric roots for display.

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import cmath
import heapq
import math
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple, TypedDict

# Gauss-Kronrod 15-point nodes (positive half, the last is 0) and weights, and the
# weights of the embedded 7-point Gauss rule on nodes 1, 3, 5 and 7.
_KRONROD_NODES = (
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144845693013,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.0,
)
_KRONROD_WEIGHTS = (
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
)
_GAUSS_WEIGHTS = (
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
)

# Quadrature stops when the error estimate is below this (relative to the value, and
# absolutely) or after this many interval splits.
INTEGRAL_TOLERANCE = 1e-10
MAX_INTEGRAL_SPLITS = 400


class IntegralEstimate(TypedDict):
    value: float
    error: float
    converged: bool


def _kronrod_interval(f: Callable[[float], Optional[float]], a: float, b: float) -> Optional[Tuple[float, float]]:
    """(Kronrod value, |Kronrod - Gauss|) on [a, b]; None where f is undefined or not finite at a node."""
    center = (a + b) / 2
    half = (b - a) / 2
    kronrod = 0.0
    gauss = 0.0
    for index, node in enumerate(_KRONROD_NODES):
        points = (center,) if node == 0.0 else (center - half * node, center + half * node)
        for point in points:
            value = f(point)
            if value is not None and math.isnan(value):
                value = _removable_value(f, point, 1e-6 * half)
            if value is None or not math.isfinite(value):
                return None
            kronrod += _KRONROD_WEIGHTS[index] * value
            if index % 2 == 1:
                gauss += _GAUSS_WEIGHTS[index // 2] * value
    return kronrod * half, abs(kronrod - gauss) * half


def _removable_value(f: Callable[[float], Optional[float]], point: float, step: float) -> Optional[float]:
    """f at a 0/0 point ((e^x-1)/x at 0, often a node of a symmetric interval): the mean of its neighbours."""
    below, above = f(point - step), f(point + step)
    if below is None or above is None or not (math.isfinite(below) and math.isfinite(above)):
        return None
    return (below + above) / 2


def adaptive_integral(
    f: Callable[[float], Optional[float]],
    a: float,
    b: float,
    tolerance: float = INTEGRAL_TOLERANCE,
    max_splits: int = MAX_INTEGRAL_SPLITS,
) -> Optional[IntegralEstimate]:
    """Integrate f over [a, b] by adaptive Gauss-Kronrod quadrature.

    f returns a float, or None where it is undefined or not real. The nodes never touch
    a or b, so an integrable singularity at an endpoint (1/sqrt(x) at 0) is fine; it just
    needs more splits. Returns None when f is undefined or infinite at a node, and
    converged False when the error estimate is still above the tolerance after
    max_splits splits (a divergent or badly behaved integral).
    """
    if a == b:
        return {"value": 0.0, "error": 0.0, "converged": True}
    first = _kronrod_interval(f, a, b)
    if first is None:
        return None
    # Max-heap on the error estimate: (-error, value, a, b)
    heap = [(-first[1], first[0], a, b)]
    total, total_error = first
    splits = 0
    while total_error > tolerance * max(1.0, abs(total)) and splits < max_splits:
        negative_error, value, left, right = heapq.heappop(heap)
        middle = (left + right) / 2
        if middle in (left, right):
            heapq.heappush(heap, (negative_error, value, left, right))
            break  # the interval cannot be split further in floating point
        halves = [_kronrod_interval(f, left, middle), _kronrod_interval(f, middle, right)]
        if halves[0] is None or halves[1] is None:
            return None
        total += halves[0][0] + halves[1][0] - value
        total_error += halves[0][1] + halves[1][1] + negative_error
        heapq.heappush(heap, (-halves[0][1], halves[0][0], left, middle))
        heapq.heappush(heap, (-halves[1][1], halves[1][0], middle, right))
        splits += 1
    # Recompute from the pieces so the running sums carry no accumulated rounding.
    total = math.fsum(item[1] for item in heap)
    total_error = math.fsum(-item[0] for item in heap)
    converged = total_error <= tolerance * max(1.0, abs(total)) * 10
    return {"value": total, "error": total_error, "converged": converged}


# Aberth-Ehrlich iteration: at most this many sweeps. Copies of a root of multiplicity m
# converge only to about machine epsilon^(1/m) (7e-4 for m = 5), so roots closer than
# _ROOT_CLUSTER (relative) are taken as one multiple root and refined, and kept as one only
# when the refined value is a root to _MULTIPLE_ROOT_RESIDUAL (relative to the terms' size).
_ABERTH_SWEEPS = 500
_ROOT_CLUSTER = 1e-2
_MULTIPLE_ROOT_RESIDUAL = 1e-12
# A root whose imaginary part is below this (relative) is real.
REAL_ROOT_TOLERANCE = 1e-9


def _evaluate_polynomial(coefficients: Sequence[complex], z: complex) -> Tuple[complex, complex]:
    """(p(z), p'(z)) by Horner's rule; coefficients run from the highest power down."""
    value = 0j
    derivative = 0j
    for coefficient in coefficients:
        derivative = derivative * z + value
        value = value * z + coefficient
    return value, derivative


# Up to this degree a polynomial is split into square-free factors in exact rational
# arithmetic first, so each factor has only simple roots (repeated roots are what the
# iteration finds least accurately). Above it the work grows too fast for the browser.
MAX_EXACT_SPLIT_DEGREE = 12
# Coefficients are read as the nearest fraction with at most this denominator.
_EXACT_DENOMINATOR = 10**12


def polynomial_roots(coefficients_low_to_high: Sequence[float]) -> List[complex]:
    """All complex roots of a polynomial, with multiplicity, real ones made exactly real.

    coefficients_low_to_high[k] is the coefficient of x^k. Trailing zero coefficients
    (above the degree) are ignored; a constant polynomial has no roots.
    """
    coefficients = [float(c) for c in coefficients_low_to_high]
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    degree = len(coefficients) - 1
    if degree < 1:
        return []
    zero_roots = 0
    while coefficients[0] == 0:
        coefficients.pop(0)
        zero_roots += 1
    roots: List[complex] = []
    high_to_low = [complex(c / coefficients[-1]) for c in reversed(coefficients)]
    if len(coefficients) - 1 <= MAX_EXACT_SPLIT_DEGREE:
        # Decimal coefficients (0.3) are not exact in binary: read them as the fractions they
        # stand for, or (x-0.3)^3 looks square-free and its triple root scatters
        exact = [Fraction(c).limit_denominator(_EXACT_DENOMINATOR) for c in coefficients]
        for factor, multiplicity in square_free_factors(exact):
            roots.extend(root for root in _simple_roots(factor) for _ in range(multiplicity))
    elif len(coefficients) > 1:
        roots = _aberth(high_to_low)
    if len(coefficients) > 1:
        roots = _merge_multiple_roots(high_to_low, roots)
    return [_realify(root) for root in roots] + [0j] * zero_roots


def _simple_roots(factor: List[Fraction]) -> List[complex]:
    """The roots of a polynomial without repeated roots (coefficients low to high)."""
    if len(factor) < 2:
        return []
    high_to_low = [complex(float(c / factor[-1])) for c in reversed(factor)]
    return _aberth(high_to_low)


# ----- exact square-free factorisation (Yun's algorithm over the rationals) ---------------

Polynomial = List[Fraction]  # coefficients from x^0 up, no trailing zeros


def _trimmed(p: Sequence[Fraction]) -> Polynomial:
    result = list(p)
    while result and result[-1] == 0:
        result.pop()
    return result


def _monic(p: Polynomial) -> Polynomial:
    return [c / p[-1] for c in p] if p else p


def _derivative(p: Polynomial) -> Polynomial:
    return _trimmed([c * k for k, c in enumerate(p)][1:])


def _subtract(p: Polynomial, q: Polynomial) -> Polynomial:
    size = max(len(p), len(q))
    return _trimmed(
        [(p[k] if k < len(p) else Fraction(0)) - (q[k] if k < len(q) else Fraction(0)) for k in range(size)]
    )


def _divmod(p: Polynomial, q: Polynomial) -> Tuple[Polynomial, Polynomial]:
    remainder = list(p)
    quotient = [Fraction(0)] * max(0, len(p) - len(q) + 1)
    while len(remainder) >= len(q) and remainder:
        factor = remainder[-1] / q[-1]
        shift = len(remainder) - len(q)
        quotient[shift] = factor
        for k, c in enumerate(q):
            remainder[shift + k] -= factor * c
        remainder = _trimmed(remainder[:-1] + [Fraction(0)]) if remainder[-1] == 0 else _trimmed(remainder)
    return _trimmed(quotient), remainder


def _gcd(p: Polynomial, q: Polynomial) -> Polynomial:
    a, b = _monic(p), _monic(q)
    while b:
        a, b = b, _monic(_divmod(a, b)[1])
    return a


def square_free_factors(p: Sequence[Fraction]) -> List[Tuple[Polynomial, int]]:
    """[(factor, multiplicity)] with p = c * product(factor^multiplicity), each factor square-free."""
    f = _monic(_trimmed(p))
    if len(f) < 2:
        return []
    factors: List[Tuple[Polynomial, int]] = []
    derivative = _derivative(f)
    common = _gcd(f, derivative)
    b = _divmod(f, common)[0]
    c = _divmod(derivative, common)[0]
    d = _subtract(c, _derivative(b))
    multiplicity = 1
    while len(b) > 1:
        a = _gcd(b, d) if d else _monic(b)
        if len(a) > 1:
            factors.append((a, multiplicity))
        b = _divmod(b, a)[0]
        c = _divmod(d, a)[0]
        d = _subtract(c, _derivative(b))
        multiplicity += 1
    return factors


def _aberth(monic: Sequence[complex]) -> List[complex]:
    degree = len(monic) - 1
    # Fujiwara bound: every root lies within this radius (no overflow for large coefficients,
    # unlike 1 + max |c|).
    radius = 2 * max(abs(c) ** (1.0 / k) for k, c in enumerate(monic[1:], start=1))
    radius = radius if radius > 0 else 1.0
    roots = [radius * 0.5 * cmath.exp(1j * (2 * math.pi * k / degree + 0.4)) for k in range(degree)]
    for _ in range(_ABERTH_SWEEPS):
        largest_step = 0.0
        for index, root in enumerate(roots):
            value, derivative = _evaluate_polynomial(monic, root)
            if value == 0:
                continue
            ratio = value / derivative if derivative != 0 else value
            repulsion = sum(1 / (root - other) for j, other in enumerate(roots) if j != index and root != other)
            denominator = 1 - ratio * repulsion
            step = ratio / denominator if denominator != 0 else ratio
            roots[index] = root - step
            largest_step = max(largest_step, abs(step) / max(1.0, abs(root)))
        if largest_step < 1e-15:
            break
    return roots


def _derivative_coefficients(high_to_low: Sequence[complex]) -> List[complex]:
    degree = len(high_to_low) - 1
    return [coefficient * (degree - index) for index, coefficient in enumerate(high_to_low[:-1])]


def _merge_multiple_roots(monic: Sequence[complex], roots: List[complex]) -> List[complex]:
    """Replace each cluster of nearly equal roots (a root of multiplicity m) by m copies of one value.

    The value is the cluster's mean refined by Newton's method on the (m-1)th derivative of
    the polynomial, where the multiple root is a simple root. A cluster holding a root and
    its near-conjugate is a real multiple root (the coefficients are real), so it is made real.
    """
    remaining = list(roots)
    merged: List[complex] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        grown = True
        while grown:  # transitive: a root near any member joins
            grown = False
            for other in list(remaining):
                if any(abs(other - member) <= _ROOT_CLUSTER * max(1.0, abs(member)) for member in cluster):
                    cluster.append(other)
                    remaining.remove(other)
                    grown = True
        if len(cluster) == 1:
            merged.append(seed)
            continue
        root = _refined_multiple_root(monic, cluster)
        merged.extend([root] * len(cluster) if root is not None else cluster)
    return merged


def _refined_multiple_root(monic: Sequence[complex], cluster: List[complex]) -> Optional[complex]:
    """The multiple root a cluster approximates, or None when the cluster is distinct close roots."""
    root = sum(cluster) / len(cluster)
    spread = max(abs(member - root) for member in cluster)
    if abs(root.imag) <= spread + REAL_ROOT_TOLERANCE * max(1.0, abs(root.real)):
        root = complex(root.real, 0.0)
    derivative = list(monic)
    for _ in range(len(cluster) - 1):
        derivative = _derivative_coefficients(derivative)
    for _ in range(30):
        value, slope = _evaluate_polynomial(derivative, root)
        if slope == 0:
            break
        step = value / slope
        root -= step
        if abs(step) <= 1e-16 * max(1.0, abs(root)):
            break
    residual, _ = _evaluate_polynomial(monic, root)
    size = sum(abs(coefficient) * abs(root) ** (len(monic) - 1 - index) for index, coefficient in enumerate(monic))
    return root if abs(residual) <= _MULTIPLE_ROOT_RESIDUAL * size else None


def _realify(root: complex) -> complex:
    if abs(root.imag) <= REAL_ROOT_TOLERANCE * max(1.0, abs(root.real)):
        return complex(root.real, 0.0)
    return root


def snap(value: float) -> float:
    """value rounded to the nearest integer when within 1e-12 (relative) of it, -0.0 as 0.0."""
    nearest = round(value)
    if abs(value - nearest) <= 1e-12 * max(1.0, abs(value)):
        return float(nearest)
    return value + 0.0


def format_number(value: float) -> str:
    """A real root as text: an integer when it is one (to 1e-12), else 15 significant digits."""
    value = snap(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.15g}"


def format_complex(value: complex) -> str:
    """A complex root as nerdamer-style text: 0.5+1.2*i, -0.5-i, 2*i, or a real number."""
    real, imag = snap(value.real), snap(value.imag)
    if imag == 0:
        return format_number(real)
    magnitude = format_number(abs(imag))
    imaginary = "i" if magnitude == "1" else f"{magnitude}*i"
    if real == 0:
        return imaginary if imag > 0 else f"-{imaginary}"
    return f"{format_number(real)}{'+' if imag > 0 else '-'}{imaginary}"


def distinct_values(values: Sequence[float], tolerance: float = 1e-9) -> List[float]:
    """Sorted values with near-duplicates (within tolerance, relative) removed."""
    result: List[float] = []
    for value in sorted(values):
        if result and abs(value - result[-1]) <= tolerance * max(1.0, abs(value)):
            continue
        result.append(value)
    return result
