"""Numeric checks on nerdamer's symbolic results, and numeric answers where it has none.

nerdamer (the symbolic engine) is often confidently wrong: it reads unknown function names
as variables (ln(x) -> ln*x), returns limits such as lim sqrt(x^2+x)-x = -Infinity, loses
real roots of cubics, and hands back garbage antiderivatives. Every CAS tool therefore
checks its result with math.js before returning it:

- derivatives against central differences, antiderivatives by differentiating them back,
  simplify/expand/factor by comparing values at sample points;
- definite integrals against adaptive Gauss-Kronrod quadrature (utils.cas_numeric);
- limits against samples on each side (MathUtils' asymptote probes).

A check that cannot run (a parameter without a value, a function math.js lacks) neither
confirms nor rejects a result.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from browser import window

from utils.cas_numeric import IntegralEstimate, adaptive_integral
from utils.cas_text import rewrite_for_nerdamer, unknown_function_names

RealFunction = Callable[[float], Optional[float]]

# Points where results are compared: away from 0, 1 and other spots where functions
# typically have singularities or removable holes.
SAMPLE_POINTS = (0.37, 0.81, 1.43, 2.2, -0.7, -1.6, 3.1)
# Values given to the other free variables of a multi-variable expression.
PARAMETER_VALUES = (0.53, 1.27, 0.71, 1.9, 0.29, 2.4)
# Results agree when they differ by at most this (relative to max(1, |value|)).
AGREEMENT = 1e-6
# A comparison needs at least this many sample points where both sides are real and finite.
MIN_CHECKED_POINTS = 2
# Numeric limit estimates are rounded to the digits their samples support, so two of them (or
# one and a symbolic value) agree when they differ by at most this (relative to max(1, |value|)).
LIMIT_AGREEMENT = 1e-4


class SymbolicChecks:
    """Numeric checks on symbolic results; every method is static."""

    _nerdamer_functions: Optional[List[str]] = None

    @staticmethod
    def nerdamer_input(expression: str) -> str:
        """Rewrite an expression for nerdamer and refuse function names it would misread.

        Raises:
            ValueError: if the expression calls a function nerdamer does not define (it
                would read gamma(x) as gamma*x and answer for that instead), or holds % (it
                reads 7%-3 as a percentage and x%3 as its own mod, which truncates).
        """
        if "%" in expression:
            raise ValueError("The symbolic engine cannot read the remainder operator %.")
        rewritten = rewrite_for_nerdamer(expression)
        unknown = unknown_function_names(rewritten, SymbolicChecks._known_functions())
        if unknown:
            names = ", ".join(f"{name}()" for name in unknown)
            raise ValueError(f"The symbolic engine has no function named {names}.")
        return rewritten

    @staticmethod
    def _known_functions() -> List[str]:
        if SymbolicChecks._nerdamer_functions is None:
            functions = window.nerdamer.getCore().PARSER.functions
            SymbolicChecks._nerdamer_functions = [str(name) for name in window.Object.keys(functions)]
        return SymbolicChecks._nerdamer_functions

    @staticmethod
    def free_variables(expression: str) -> List[str]:
        """nerdamer's free variables of an expression (constants e and pi excluded); [] on error."""
        try:
            return [str(name) for name in window.nerdamer(expression).variables()]
        except Exception:
            return []

    @staticmethod
    def real_function(expression: str, variable: str, others: Optional[Sequence[str]] = None) -> Optional[RealFunction]:
        """The expression as f(variable) evaluated by math.js, the other variables fixed.

        Returns None when math.js cannot compile it. f returns a float (inf or nan where the
        expression is infinite or 0/0), or None where the value is not real or math.js cannot
        evaluate it (an unknown function such as Si).
        """
        from utils.math_utils import MathUtils

        try:
            compiled = MathUtils._compiled_evaluator(str(expression).replace("**", "^"))
        except Exception:
            return None
        names = [name for name in (others or []) if name != variable]
        fixed: Dict[str, Any] = {
            name: PARAMETER_VALUES[index % len(PARAMETER_VALUES)] for index, name in enumerate(names)
        }

        def evaluate(point: float) -> Optional[float]:
            scope = dict(fixed)
            scope[variable] = point
            try:
                value = compiled(scope)
            except Exception:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        return evaluate

    @staticmethod
    def _function_pair(first: str, second: str, variable: str) -> Optional[List[RealFunction]]:
        others = sorted(set(SymbolicChecks.free_variables(first)) | set(SymbolicChecks.free_variables(second)))
        functions = [SymbolicChecks.real_function(text, variable, others) for text in (first, second)]
        if functions[0] is None or functions[1] is None:
            return None
        return [functions[0], functions[1]]

    @staticmethod
    def derivative_matches(function_text: str, derivative_text: str, variable: str) -> Optional[bool]:
        """Whether derivative_text is the derivative of function_text, by central differences.

        False also when one text is infinite or undefined (0/0) at MIN_CHECKED_POINTS points where
        the other is finite. Returns None when fewer than MIN_CHECKED_POINTS points could be
        checked (the function is not smooth there, or math.js cannot evaluate one of the texts).
        """
        pair = SymbolicChecks._function_pair(function_text, derivative_text, variable)
        if pair is None:
            return None
        function, derivative = pair
        checked = 0
        broken = 0
        for point in SAMPLE_POINTS:
            expected, centre = derivative(point), function(point)
            if expected is None or centre is None:
                continue
            if math.isfinite(expected) != math.isfinite(centre):
                broken += 1
                continue
            estimate = SymbolicChecks._difference_quotient(function, point)
            if not math.isfinite(expected) or estimate is None:
                continue
            value, tolerance = estimate
            tolerance += AGREEMENT * max(1.0, abs(expected))
            if abs(value - expected) > tolerance:
                step = 1e-4 * max(1.0, abs(point))
                noise = SymbolicChecks.rounding_noise(function, point) / step + SymbolicChecks.rounding_noise(
                    derivative, point
                )
                if abs(value - expected) > tolerance + noise:
                    return False
            checked += 1
        if broken >= MIN_CHECKED_POINTS:
            return False
        return True if checked >= MIN_CHECKED_POINTS else None

    @staticmethod
    def antiderivative_matches(antiderivative_text: str, integrand_text: str, variable: str) -> Optional[bool]:
        """Whether antiderivative_text is an antiderivative of integrand_text.

        Differentiates it back (derivative_matches) and, when that cannot decide, compares
        F(b) - F(a) with quadrature of the integrand between neighbouring sample points: an
        antiderivative made of rounding noise (a denominator that is 0 up to rounding) has no
        smooth difference quotient, but its differences are still far from the integral.
        """
        verdict = SymbolicChecks.derivative_matches(antiderivative_text, integrand_text, variable)
        if verdict is not None:
            return verdict
        pair = SymbolicChecks._function_pair(antiderivative_text, integrand_text, variable)
        if pair is None:
            return None
        antiderivative, integrand = pair
        points = sorted(SAMPLE_POINTS)
        matches = 0
        mismatches = 0
        for a, b in zip(points, points[1:]):
            left, right = antiderivative(a), antiderivative(b)
            if left is None or right is None or not (math.isfinite(left) and math.isfinite(right)):
                continue
            estimate = adaptive_integral(integrand, a, b)
            if estimate is None or not estimate["converged"]:
                continue
            mismatch = abs(right - left - estimate["value"])
            tolerance = AGREEMENT * max(1.0, abs(estimate["value"]))
            if mismatch > tolerance:
                # Rounding noise excuses a mismatch only while it is small next to F itself: an
                # antiderivative that is all noise (a denominator 0 up to rounding) is wrong.
                noise = sum(SymbolicChecks.rounding_noise(antiderivative, end) for end in (a, b))
                if noise <= 1e-3 * max(1.0, abs(left), abs(right)):
                    tolerance += noise
            if mismatch > tolerance:
                mismatches += 1
            else:
                matches += 1
        if mismatches >= MIN_CHECKED_POINTS:
            return False
        return True if matches >= MIN_CHECKED_POINTS and mismatches == 0 else None

    # Rounding noise is estimated from f at point * (1 + k * 2^-52) for these k, and scaled up
    # by _NOISE_FACTOR, since a few nudges only sample it.
    _NOISE_NUDGES = (-8, -4, -2, -1, 1, 2, 4, 8)
    _NOISE_FACTOR = 4.0

    @staticmethod
    def rounding_noise(function: RealFunction, point: float) -> float:
        """How much f's floating-point value at point is off by rounding, estimated from its scatter.

        An expanded polynomial near a root ((x-3)^15 expanded, at 2.2) cancels terms much larger
        than its value, so its computed values scatter far beyond a relative tolerance on the value.
        """
        centre = function(point)
        if centre is None or not math.isfinite(centre):
            return math.inf
        largest = 0.0
        for nudge in SymbolicChecks._NOISE_NUDGES:
            value = function(point * (1 + nudge * 2.0**-52) if point != 0 else nudge * 2.0**-1074)
            if value is None or not math.isfinite(value):
                return math.inf
            largest = max(largest, abs(value - centre))
        return SymbolicChecks._NOISE_FACTOR * largest

    @staticmethod
    def _difference_quotient(function: RealFunction, point: float) -> Optional[List[float]]:
        """[f'(point) by Richardson-extrapolated central differences, its rounding error]; None if not smooth there."""
        step = 1e-4 * max(1.0, abs(point))
        values = [function(point + offset) for offset in (-step, step, -step / 4, step / 4, 0.0)]
        if any(value is None or not math.isfinite(value) for value in values):
            return None
        below, above, near_below, near_above, centre = [float(value) for value in values if value is not None]
        coarse = (above - below) / (2 * step)
        fine = (near_above - near_below) / (step / 2)
        # Very different quotients mean a kink or jump near the point: not a fair test there.
        if abs(coarse - fine) > 1e-3 * max(1.0, abs(fine)):
            return None
        rounding = 1e-13 * max(1.0, abs(centre), abs(above), abs(below)) / step
        return [fine + (fine - coarse) / 15, rounding]

    @staticmethod
    def equivalent(first: str, second: str) -> Optional[bool]:
        """Whether two expressions take the same values at the sample points (None if not checkable).

        False also when one is infinite or undefined (0/0) at MIN_CHECKED_POINTS points where
        the other is finite.
        """
        variables = sorted(set(SymbolicChecks.free_variables(first)) | set(SymbolicChecks.free_variables(second)))
        variable = variables[0] if variables else "x"
        pair = SymbolicChecks._function_pair(first, second, variable)
        if pair is None:
            return None
        checked = 0
        broken = 0
        for point in SAMPLE_POINTS:
            a, b = pair[0](point), pair[1](point)
            if a is None or b is None:
                continue
            if math.isfinite(a) != math.isfinite(b):
                broken += 1
            elif math.isfinite(a):
                tolerance = AGREEMENT * max(1.0, abs(a), abs(b))
                if abs(a - b) > tolerance:
                    noise = sum(SymbolicChecks.rounding_noise(side, point) for side in pair)
                    if abs(a - b) > tolerance + noise:
                        return False
                checked += 1
        if broken >= MIN_CHECKED_POINTS:
            return False
        return True if checked >= MIN_CHECKED_POINTS else None

    @staticmethod
    def numeric_integral(expression: str, variable: str, lower: float, upper: float) -> Optional[IntegralEstimate]:
        """Adaptive quadrature of the expression over [lower, upper]; None where it is not real at a node."""
        if SymbolicChecks.free_variables(expression) not in ([], [variable]):
            return None
        function = SymbolicChecks.real_function(expression, variable)
        if function is None:
            return None
        if lower > upper:
            estimate = adaptive_integral(function, upper, lower)
            if estimate is not None:
                estimate["value"] = -estimate["value"]
            return estimate
        return adaptive_integral(function, lower, upper)

    @staticmethod
    def numeric_value(text: Any) -> Optional[float]:
        """text evaluated by math.js as a finite real number, else None."""
        from utils.math_utils import MathUtils

        try:
            value = MathUtils._compiled_evaluator(str(text).replace("**", "^"))({})
        except Exception:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    # ----- limits -----------------------------------------------------------------------

    # Bounded oscillation test: f sampled at _OSCILLATION_SAMPLES points on each of three
    # ever-smaller scales beside the target (offsets h * u for u in (0.1, 1]); a range that
    # does not shrink means f keeps oscillating, one that collapses means f settles.
    _OSCILLATION_SCALES = (1e-2, 1e-5, 1e-8)
    _OSCILLATION_SAMPLES = 23
    _PERSISTENT_RANGE = 0.1
    _COLLAPSED_RANGE = 1e-7

    @staticmethod
    def limit_verdict(expression: str, variable: str, target: str) -> Dict[str, Any]:
        """What the expression's values say about its limit as variable -> target.

        target is "Infinity", "-Infinity" or a finite value (math.js syntax). Returns a dict
        with "kind": "value" (with "value"), "+inf", "-inf", "dne" (with "reason"), or
        "unclear" when the samples settle on nothing. "one_sided" is True when f is not real
        on one side of a finite target, so the answer is the limit from the other side.
        """
        from utils.math_utils import MathUtils

        evaluate = MathUtils._mathjs_probe_evaluator(expression, variable)
        if evaluate is None or [name for name in SymbolicChecks.free_variables(expression) if name != variable]:
            return {"kind": "unclear"}
        if target in ("Infinity", "-Infinity"):
            return SymbolicChecks._verdict_at_infinity(evaluate, -1.0 if target.startswith("-") else 1.0)
        x0 = SymbolicChecks.numeric_value(target)
        if x0 is None:
            return {"kind": "unclear"}
        start = MathUtils._PROBE_START * max(1.0, abs(x0))
        sides = [SymbolicChecks._side_verdict(evaluate, x0, side, start) for side in (-1.0, 1.0)]
        return SymbolicChecks._combine_sides(sides[0], sides[1])

    @staticmethod
    def _side_verdict(evaluate: RealFunction, x0: float, side: float, start: float) -> Dict[str, Any]:
        """One side's behaviour: kind "value", "+inf", "-inf", "undefined", "oscillates" or "unclear"."""
        from utils.math_utils import MathUtils

        values = MathUtils._probe_side(evaluate, x0, side, start)
        verdict = MathUtils._probe_side_verdict(values)
        if verdict == "undefined" or values is None:
            return {"kind": "undefined"}
        if verdict == "grow":
            infinite = SymbolicChecks._infinite_kind(values)
            if infinite is not None:
                return {"kind": infinite}
            if not SymbolicChecks._grows_in_magnitude(values):
                # A slow monotone approach (1/log(x) at Infinity, x^0.05 at 0) also reads as "grow"
                return {"kind": "unclear"}
            return {"kind": "oscillates"}
        if verdict == "converge":
            settled = MathUtils._settled_value(values)
            if settled is None:
                settled = SymbolicChecks._extrapolated(values)
            if settled is not None:
                return {"kind": "value", "value": settled}
        scale = max(1.0, abs(x0))
        return SymbolicChecks._range_verdict(
            lambda u: evaluate(x0 + side * u * scale), SymbolicChecks._OSCILLATION_SCALES
        )

    # |f| must reach this, rising over the last _GROWTH_TAIL samples, for a side to be infinite.
    _INFINITE_MAGNITUDE = 10.0
    _GROWTH_TAIL = 4

    @staticmethod
    def _infinite_kind(values: List[float]) -> Optional[str]:
        """Return "+inf" or "-inf" when the last samples keep one sign and rise past _INFINITE_MAGNITUDE."""
        tail = values[max(0, len(values) - SymbolicChecks._GROWTH_TAIL) :]
        if len(tail) < 3 or not SymbolicChecks._grows_in_magnitude(tail):
            return None
        if abs(tail[-1]) < SymbolicChecks._INFINITE_MAGNITUDE:
            return None
        if all(value > 0 for value in tail):
            return "+inf"
        if all(value < 0 for value in tail):
            return "-inf"
        return None

    @staticmethod
    def _grows_in_magnitude(values: List[float]) -> bool:
        tail = values[max(0, len(values) - SymbolicChecks._GROWTH_TAIL) :]
        return all(abs(later) > abs(earlier) for earlier, later in zip(tail, tail[1:]))

    @staticmethod
    def _extrapolated(values: List[float]) -> Optional[float]:
        """The limit of converging samples too few for MathUtils._settled_value (float noise cut them short).

        Aitken's extrapolation of the last three, rounded to the digits the last step supports.
        (1-cos(x))/x^2 near 0 gives three clean samples before cancellation takes over.
        """
        from utils.math_utils import MathUtils

        if len(values) < 3:
            return None
        estimate = MathUtils._aitken(values[-3], values[-2], values[-1])
        if estimate is None or not math.isfinite(estimate):
            return None
        tolerance = max(abs(values[-1] - values[-2]), 1e-12 * max(1.0, abs(estimate)))
        digits = max(0, min(12, int(math.floor(-math.log10(tolerance)))))
        rounded = round(estimate, digits)
        return 0.0 if rounded == 0 else float(rounded)

    @staticmethod
    def _range_verdict(at_offset: Callable[[float], Optional[float]], scales: Sequence[float]) -> Dict[str, Any]:
        """Settle or oscillation from the range of f over shrinking (or, at infinity, growing) scales.

        A scale where f is rounding noise (Taylor-type cancellation such as (e^x-e^-x-2x)/x^3
        near 0: nudging x by 1e-10 of itself changes f by a good part of its range) ends the
        test; the scales before it decide, and fewer than two of them decide nothing.
        """
        count = SymbolicChecks._OSCILLATION_SAMPLES
        ranges: List[List[float]] = []
        for scale in scales:
            offsets = [scale * (0.1 + 0.9 * ((k * 0.618034) % 1.0)) for k in range(1, count + 1)]
            samples = [at_offset(offset) for offset in offsets]
            finite = [value for value in samples if value is not None and math.isfinite(value)]
            if len(finite) < count:
                return {"kind": "unclear"}
            low, high = min(finite), max(finite)
            if SymbolicChecks._noise(at_offset, offsets[:5], finite[:5]) > SymbolicChecks._NOISY_SHARE * (high - low):
                break
            ranges.append([low, high])
        if len(ranges) < 2:
            return {"kind": "unclear"}
        first_range = ranges[0][1] - ranges[0][0]
        last_low, last_high = ranges[-1]
        middle = (last_low + last_high) / 2
        # Settling needs each range inside the one before (with a little slack): float noise at
        # the smallest scale ((1 - cos(x))/x^2 is 0 there) collapses the range somewhere else.
        nested = all(
            later[0] >= earlier[0] - SymbolicChecks._COLLAPSED_RANGE * max(1.0, abs(earlier[0]))
            and later[1] <= earlier[1] + SymbolicChecks._COLLAPSED_RANGE * max(1.0, abs(earlier[1]))
            for earlier, later in zip(ranges, ranges[1:])
        )
        if nested and last_high - last_low <= SymbolicChecks._COLLAPSED_RANGE * max(1.0, abs(middle)):
            digits = max(0, min(12, int(-math.log10(max(last_high - last_low, 1e-12)))))
            value = round(middle, digits)
            return {"kind": "value", "value": 0.0 if value == 0 else value}
        if first_range > 0 and last_high - last_low >= SymbolicChecks._PERSISTENT_RANGE * first_range:
            return {"kind": "oscillates", "low": last_low, "high": last_high}
        return {"kind": "unclear"}

    # A range is rounding noise when nudging x by _JITTER of its offset changes f by more than
    # this share of the range.
    _JITTER = 1e-10
    _NOISY_SHARE = 0.1

    @staticmethod
    def _noise(at_offset: Callable[[float], Optional[float]], offsets: List[float], values: List[float]) -> float:
        largest = 0.0
        for offset, value in zip(offsets, values):
            nudged = at_offset(offset * (1 + SymbolicChecks._JITTER))
            if nudged is None or not math.isfinite(nudged):
                return math.inf
            largest = max(largest, abs(nudged - value))
        return largest

    @staticmethod
    def _verdict_at_infinity(evaluate: RealFunction, sign: float) -> Dict[str, Any]:
        from utils.math_utils import MathUtils

        value = MathUtils._limit_at_infinity(evaluate, sign)
        if value is not None:
            return {"kind": "value", "value": value}
        # f(sign / t) as t -> 0+ is f as x -> sign * infinity: reuse the one-sided tests.
        side = SymbolicChecks._side_verdict(lambda t: evaluate(sign / t), 0.0, 1.0, MathUtils._PROBE_START)
        if side["kind"] in ("+inf", "-inf", "value"):
            return side
        if side["kind"] == "oscillates":
            return {"kind": "dne", "reason": SymbolicChecks._oscillation_reason(side)}
        return {"kind": "unclear"}

    @staticmethod
    def _oscillation_reason(verdict: Dict[str, Any]) -> str:
        if "low" in verdict:
            return f"it keeps oscillating (between about {verdict['low']:.6g} and {verdict['high']:.6g})"
        return "it keeps oscillating without settling"

    @staticmethod
    def _combine_sides(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        """The two-sided verdict from the left and right ones."""
        kinds = (left["kind"], right["kind"])
        if "unclear" in kinds:
            return {"kind": "unclear"}
        if kinds == ("undefined", "undefined"):
            return {"kind": "unclear"}
        if "undefined" in kinds:
            defined = right if left["kind"] == "undefined" else left
            if defined["kind"] == "oscillates":
                return {"kind": "dne", "reason": SymbolicChecks._oscillation_reason(defined)}
            return dict(defined, one_sided=True, side="right" if left["kind"] == "undefined" else "left")
        if "oscillates" in kinds:
            oscillating = left if left["kind"] == "oscillates" else right
            return {"kind": "dne", "reason": SymbolicChecks._oscillation_reason(oscillating)}
        if kinds == ("value", "value"):
            a, b = left["value"], right["value"]
            if abs(a - b) <= LIMIT_AGREEMENT * max(1.0, abs(a), abs(b)):
                return {"kind": "value", "value": b}
        if left["kind"] == right["kind"] and left["kind"] in ("+inf", "-inf"):
            return {"kind": left["kind"]}
        return {
            "kind": "dne",
            "reason": f"it {SymbolicChecks._describe_side(left)} from the left and {SymbolicChecks._describe_side(right)} from the right",
        }

    @staticmethod
    def _describe_side(verdict: Dict[str, Any]) -> str:
        if verdict["kind"] == "value":
            return f"approaches {verdict['value']:.12g}"
        return "tends to Infinity" if verdict["kind"] == "+inf" else "tends to -Infinity"
