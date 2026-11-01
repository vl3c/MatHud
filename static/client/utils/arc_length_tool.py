"""The arc_length tool: reads the curve and range from the tool's arguments and measures it.

A curve comes as exactly one of:
    - function_string: y = f(x), in x (read as draw_function reads it);
    - x_expression and y_expression: a parametric curve (x(t), y(t)), in t (as
      draw_parametric_function reads them);
    - r_expression: a polar curve r(θ), in θ or the word theta (PolarFunction's parser);
    - curve_name: a plotted curve, which the caller's resolver turns into a NamedCurve.

Bounds are numbers or constant expressions ("pi", "2*pi", "pi/3"). The length comes from
utils.arc_length (numeric, with an error estimate) and is rounded to the digits the estimate
supports. A parametric or polar range that traces its curve several times (r = cos(θ) on
[0, 2π]) is reported with traced_times and the length of one trace.

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import compiled_expression
from utils.arc_length import TOLERANCE, Point, arc_length_in_pieces, traced_times

# Leading "y =", "f(x) =" or "y(x) =" (one reading: the right-hand side follows)
_FUNCTION_PREFIX = re.compile(r"^\s*(?:[yY]|[A-Za-z]\w*\s*\(\s*x\s*\))\s*=(?!=)\s*")
# A plotted curve's name written as a call, as the canvas text shows functions: "f(x)", "r(θ)"
_CALLED_NAME = re.compile(r"^(\w+)\s*\(\s*(?:x|t|θ|theta)\s*\)$")
# Leading "x =" / "x(t) =" or "y =" / "y(t) =" of a parametric coordinate
_COORDINATE_PREFIX = re.compile(r"^\s*([xy])\s*(?:\(\s*t\s*\))?\s*=(?!=)\s*")
# A name glued to the variable, such as sinx or cost
_GLUED_FUNCTION = re.compile(
    r"^(sin|cos|tan|sec|csc|cot|asin|acos|atan|sinh|cosh|tanh|ln|log|exp|sqrt|abs)([a-zθ]\w*)$"
)
_DEGREE_MARK = re.compile(r"°|\bdeg(?:rees?)?\b", re.IGNORECASE)

_SYNTAX_HINT = (
    "Write every function with parentheses and every product with *: sin(x), 2*x, x*e^x, e^x or exp(x), |x| or abs(x)."
)
_ROOT_HINT = (
    " A fractional power of a negative number, such as x^(1/3) for x < 0, is not real here: "
    "write cbrt(x) for the real cube root (cbrt(x)^2 for x^(2/3))."
)

POLAR_DEFAULT_RANGE = (0.0, 2 * math.pi)
PARAMETRIC_DEFAULT_RANGE = (0.0, 2 * math.pi)
# One unit in the last place of a double, relative; and the error at which arc_length accepts a length
_ONE_ULP = 2.0**-52
_ACCEPTED = 20.0 * TOLERANCE


class NamedCurve(NamedTuple):
    """A plotted curve, as the resolver gives it.

    kind is "function", "parametric" or "polar"; point maps the parameter to (x, y), or None
    where the curve has no point; own_range is the curve's plotted range (None for an unbounded
    side); breakpoints are parameters where the curve may change formula or have a hole
    (pieces are measured separately and must meet); asymptotes are parameters where it goes
    off to infinity; description is how the result names the curve.
    """

    kind: str
    point: Callable[[float], Optional[Point]]
    own_range: Tuple[Optional[float], Optional[float]]
    breakpoints: Sequence[float]
    asymptotes: Sequence[float]
    description: str


Resolver = Callable[[str], NamedCurve]

_VARIABLE = {"function": "x", "parametric": "t", "polar": "θ"}


def arc_length_tool(
    curve_name: Optional[str] = None,
    function_string: Optional[str] = None,
    x_expression: Optional[str] = None,
    y_expression: Optional[str] = None,
    r_expression: Optional[str] = None,
    lower_bound: Any = None,
    upper_bound: Any = None,
    resolve: Optional[Resolver] = None,
) -> Dict[str, Any]:
    """The length of a curve over a range; see the module docstring.

    Raises:
        ValueError: With a hint, when the curve or the bounds cannot be read, the curve is
            undefined or breaks on the range, or the length does not converge.
    """
    curve = _read_curve(curve_name, function_string, x_expression, y_expression, r_expression, resolve)
    lower, upper = _read_range(curve, lower_bound, upper_bound)
    _refuse_asymptotes(curve, lower, upper)
    variable = _VARIABLE[curve.kind]
    try:
        measured = arc_length_in_pieces(curve.point, lower, upper, curve.breakpoints, variable)
    except ValueError as exc:
        message = str(exc)
        if "no real point" in message and _has_fractional_power(
            function_string, x_expression, y_expression, r_expression
        ):
            message += _ROOT_HINT
        raise ValueError(message) from exc

    length, error = measured["length"], measured["error_estimate"]
    bound_error = _bound_rounding_error(curve, lower, upper, lower_bound, upper_bound)
    error += bound_error
    shown_length = _rounded(length, error)
    result: Dict[str, Any] = {
        "length": shown_length,
        # The estimate covers the rounding of the length shown too
        "error_estimate": float(f"{error + abs(shown_length - length):.2g}"),
        "curve_type": curve.kind,
        "curve": curve.description,
        "variable": variable,
        "lower_bound": lower,
        "upper_bound": upper,
    }
    notes: List[str] = []
    if "warning" in measured:
        result["warning"] = measured["warning"]
    elif bound_error > 0.0 and error > _ACCEPTED * length:
        result["warning"] = (
            f"The length is accurate only to about {error:.1g}: the curve moves so fast at a bound that "
            "rounding the bound to the nearest floating-point number moves the curve's end by about "
            f"{bound_error:.1g}."
        )
    if curve.kind != "function" and length > 0.0:
        times = traced_times(curve.point, lower, upper)
        if times > 1:
            result["traced_times"] = times
            result["length_of_one_trace"] = _rounded(length / times, error / times)
            notes.append(
                f"This range traces the curve {times} times, so the length is {times} times the curve's own "
                f"length: one trace (the curve's perimeter) is {result['length_of_one_trace']}; the full "
                "length is the distance travelled along it."
            )
    # Only a plotted curve has a range of its own; an expression's default range is not one
    outside = _outside_own_range(curve, lower, upper) if _filled(curve_name) else None
    if outside:
        notes.append(outside)
    if notes:
        result["note"] = " ".join(notes)
    return result


# ---------------------------------------------------------------------------
# Reading the curve
# ---------------------------------------------------------------------------


def _read_curve(
    curve_name: Optional[str],
    function_string: Optional[str],
    x_expression: Optional[str],
    y_expression: Optional[str],
    r_expression: Optional[str],
    resolve: Optional[Resolver],
) -> NamedCurve:
    given = [
        label
        for label, value in (
            ("curve_name", curve_name),
            ("function_string", function_string),
            ("x_expression/y_expression", x_expression or y_expression),
            ("r_expression", r_expression),
        )
        if _filled(value)
    ]
    if len(given) != 1:
        found = ", ".join(given) if given else "none"
        raise ValueError(
            "Give exactly one curve: curve_name (a plotted curve), function_string (y = f(x)), "
            f"x_expression and y_expression (a parametric curve in t) or r_expression (a polar curve in θ); got {found}."
        )
    if _filled(curve_name):
        if resolve is None:
            raise ValueError("Curves can be measured by name only on the canvas.")
        return _resolve_name(str(curve_name).strip(), resolve)
    if _filled(function_string):
        return _function_curve(str(function_string))
    if _filled(r_expression):
        return _polar_curve(str(r_expression))
    if not (_filled(x_expression) and _filled(y_expression)):
        raise ValueError(
            "A parametric curve needs both x_expression and y_expression, in t, such as cos(t) and sin(t)."
        )
    return _parametric_curve(str(x_expression), str(y_expression))


def _filled(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _function_curve(text: str) -> NamedCurve:
    source = _FUNCTION_PREFIX.sub("", text.strip(), count=1)
    f = _compile(source, "x", compiled_expression.FUNCTIONS_ALL, "f(x)", text)

    def point(x: float) -> Optional[Point]:
        return (x, f(x))

    return NamedCurve("function", point, (None, None), (), (), f"y = {source.strip()}")


def _parametric_curve(x_text: str, y_text: str) -> NamedCurve:
    x_source = _strip_coordinate(x_text, "x")
    y_source = _strip_coordinate(y_text, "y")
    fx = _compile(x_source, "t", compiled_expression.FUNCTIONS_CURVE, "x(t)", x_text)
    fy = _compile(y_source, "t", compiled_expression.FUNCTIONS_CURVE, "y(t)", y_text)

    def point(t: float) -> Optional[Point]:
        return (fx(t), fy(t))

    return NamedCurve(
        "parametric",
        point,
        PARAMETRIC_DEFAULT_RANGE,
        (),
        (),
        f"(x(t), y(t)) = ({x_source.strip()}, {y_source.strip()})",
    )


def _resolve_name(name: str, resolve: Resolver) -> NamedCurve:
    """The plotted curve called name; "f(x)", as the canvas text shows a function, is f."""
    try:
        return resolve(name)
    except ValueError:
        called = _CALLED_NAME.match(name)
        if called is None:
            raise
        return resolve(called.group(1))


def _strip_coordinate(text: str, coordinate: str) -> str:
    match = _COORDINATE_PREFIX.match(text)
    if match and match.group(1) == coordinate:
        return text[match.end() :]
    return text


def _polar_curve(text: str) -> NamedCurve:
    from drawables.polar_function import PolarFunction

    stored, r = PolarFunction.parse_r_expression(text)

    def point(theta: float) -> Optional[Point]:
        radius = r(theta)
        return (radius * math.cos(theta), radius * math.sin(theta))

    return NamedCurve("polar", point, POLAR_DEFAULT_RANGE, (), (), f"r = {stored}")


def _compile(source: str, variable: str, functions: str, what: str, original: str) -> Callable[[float], float]:
    if "=" in source.replace("==", "").replace("<=", "").replace(">=", "").replace("!=", ""):
        raise ValueError(f"{what} = '{original}' has an '='. Give only the right-hand side, such as x^2 + 1.")
    try:
        compiled = compiled_expression.compile_expression(source, (variable,), functions)
    except Exception as exc:
        raise ValueError(f"Could not read {what} = '{original}': {exc}. {_SYNTAX_HINT}") from exc
    free = compiled.free_names
    if free:
        raise ValueError(_free_names_message(what, original, variable, free))
    return compiled.bind()  # type: ignore[no-any-return]


def _free_names_message(what: str, original: str, variable: str, free: Sequence[str]) -> str:
    names = ", ".join(f"'{name}'" for name in free)
    message = f"{what} = '{original}' may use only {variable}; {names} is not defined."
    for name in free:
        glued = _GLUED_FUNCTION.match(name)
        if glued:
            return f"{message} Write {glued.group(1)}({glued.group(2)}), with parentheses."
    return f"{message} {_SYNTAX_HINT}"


def _has_fractional_power(*texts: Optional[str]) -> bool:
    return any(text and re.search(r"\^\s*\(\s*\d+\s*/\s*\d+\s*\)", text) for text in texts)


# ---------------------------------------------------------------------------
# Reading the range
# ---------------------------------------------------------------------------


def _read_range(curve: NamedCurve, lower_bound: Any, upper_bound: Any) -> Tuple[float, float]:
    variable = _VARIABLE[curve.kind]
    lower = _read_bound(lower_bound, "lower_bound", variable)
    upper = _read_bound(upper_bound, "upper_bound", variable)
    own_lower, own_upper = curve.own_range
    if lower is None:
        lower = own_lower
    if upper is None:
        upper = own_upper
    if lower is None or upper is None:
        raise ValueError(
            f"Give lower_bound and upper_bound, the {variable} range to measure "
            f"(the curve has no {'lower' if lower is None else 'upper'} bound of its own)."
        )
    if lower > upper:
        lower, upper = upper, lower
    if curve.kind == "polar":
        from drawables.polar_function import PolarFunction

        if upper - lower > PolarFunction.MAX_SPAN:
            raise ValueError("The θ range may be at most 50π long (25 turns).")
    return lower, upper


def _read_bound(value: Any, name: str, variable: str) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or a constant expression such as pi/2.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if _DEGREE_MARK.search(text):
            raise ValueError(f"{name} = '{text}' is in degrees; give the {variable} bound in radians, such as pi/2.")
        try:
            compiled = compiled_expression.compile_expression(text, (), compiled_expression.FUNCTIONS_ALL)
            if compiled.free_names:
                raise ValueError(f"it reads {', '.join(compiled.free_names)}")
            number = float(compiled.bind()())
        except Exception as exc:
            raise ValueError(
                f"{name} = '{text}' is not a number or a constant expression such as pi, 2*pi or pi/3 ({exc})."
            ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _refuse_asymptotes(curve: NamedCurve, lower: float, upper: float) -> None:
    for location in curve.asymptotes:
        if lower < location < upper:
            raise ValueError(
                f"{curve.description} has a vertical asymptote at {_VARIABLE[curve.kind]} = {location:.6g}, inside "
                f"[{lower:.6g}, {upper:.6g}], so its length there is infinite. Measure each side separately."
            )


def _bound_rounding_error(curve: NamedCurve, lower: float, upper: float, lower_bound: Any, upper_bound: Any) -> float:
    """How far the curve's ends move when a rounded bound (pi/4, a default 2π) moves by one unit
    in its last place: next to zero for most curves, but r = sqrt(cos(2θ)) at θ = pi/4 ends
    about 1e-8 away from the pole it reaches at the exact π/4."""
    given = [_number_as_given(lower_bound), _number_as_given(upper_bound)]
    total = 0.0
    for bound, inward in ((lower, 1.0), (upper, -1.0)):
        if bound == math.floor(bound) or bound in given:
            continue  # a whole number, or a number as the model gave it, is the bound meant
        here, there = _point_or_none(curve, bound), _point_or_none(curve, bound + inward * abs(bound) * _ONE_ULP)
        if here is not None and there is not None:
            total += math.hypot(here[0] - there[0], here[1] - there[1])
    return total if math.isfinite(total) else 0.0


def _number_as_given(value: Any) -> Optional[float]:
    """The bound when the model gave it as a number (or a number's text), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _point_or_none(curve: NamedCurve, parameter: float) -> Optional[Point]:
    try:
        return curve.point(parameter)
    except Exception:
        return None


def _outside_own_range(curve: NamedCurve, lower: float, upper: float) -> Optional[str]:
    own_lower, own_upper = curve.own_range
    if curve.kind == "function" and own_lower is None and own_upper is None:
        return None
    if (own_lower is not None and lower < own_lower - 1e-12) or (own_upper is not None and upper > own_upper + 1e-12):
        shown = f"[{'-inf' if own_lower is None else f'{own_lower:.6g}'}, {'inf' if own_upper is None else f'{own_upper:.6g}'}]"
        return f"The range measured goes beyond the curve's plotted range {shown}; its formula was measured there too."
    return None


def _rounded(value: float, error: float) -> float:
    """value to the significant digits error supports (3 to 12): rounding moves it by at most error."""
    if value == 0.0 or not math.isfinite(value):
        return value
    if error <= 0.0:
        digits = 12
    else:
        # The last digit kept is worth at most 2 * error, so rounding moves value by at most error
        digits = math.floor(math.log10(abs(value))) + 1 - math.floor(math.log10(2.0 * error))
        digits = max(3, min(12, digits))
    return float(f"{value:.{digits}g}")
