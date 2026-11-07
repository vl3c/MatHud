"""A named real number with a slider range: the parameters of functions, curves, points and circles.

A parameter such as ``a = 1.5`` on [-5, 5] with step 0.1 may be read by the expressions of
function plots, piecewise pieces, parametric and polar curves, point coordinates and
circle radii (``a*sin(b*x)``, the point ``(a, a^2)``). It is never drawn on the canvas:
the slider panel shows it, and the canvas text lists it.

Math only (no browser imports): the name rules, the range rules and the saved state.
ParameterManager owns the namespace the expressions read (compiled_expression).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import expression_functions as ef
from constants import default_color
from drawables.drawable import Drawable
from expression_validator import ExpressionValidator

DEFAULT_MIN = -10.0
DEFAULT_MAX = 10.0
DEFAULT_STEP = 0.1
# The variables of plots and polar curves never name a parameter. t and r may: in a
# parametric curve t is always the curve variable, and polar's r is not an expression name.
CURVE_VARIABLES = frozenset(("x", "y", "θ", "theta"))
DEFAULT_VALUE = 1.0
ANIMATION_REPEATS = ("once", "loop", "bounce")


def check_parameter_name(name: Any, names_in_use: Iterable[str] = ()) -> str:
    """The name, normalised (ϑ -> θ), if it can name a parameter.

    Raises:
        ValueError: With a hint, for an empty or non-identifier name, a name starting with
            "_", a reserved name (functions, constants such as pi or π, i, j, keywords), a
            curve variable (x, y, θ, theta) or a name another object already uses.
    """
    text = str(name if name is not None else "").strip()
    normalised = ExpressionValidator.normalize_unicode_name(text)
    if not normalised or not normalised.isidentifier() or not normalised[0].isalpha():
        raise ValueError(
            f"'{text}' cannot name a parameter: use letters, digits and '_', starting with a letter, "
            f"such as a, b2, k_1 or ω."
        )
    if normalised in ef.RESERVED_NAMES or normalised == "π":
        raise ValueError(
            f"'{text}' cannot name a parameter: it names a function, a constant or the imaginary unit. "
            f"Choose another name, such as a or k."
        )
    if normalised in CURVE_VARIABLES:
        raise ValueError(
            f"'{text}' cannot name a parameter: x, y and θ (theta) are the variables of plots and curves. "
            f"Choose another name, such as a or k."
        )
    if normalised in set(names_in_use):
        raise ValueError(f"'{text}' cannot name a parameter: another object on the canvas is named '{normalised}'.")
    return normalised


def snap_to_step(value: float, low: float, high: float, step: float) -> float:
    """A slider position snapped to min + k*step, clamped into [low, high] (max stays reachable).

    The slider element takes any value (step="any"), so snapping happens here: the
    browser's own step rounding differs between browsers and with floating steps.
    """
    number = min(max(float(value), low), high)
    if step <= 0:
        return number
    if high - number < step / 2.0:
        return high
    snapped = low + round((number - low) / step) * step
    return min(max(round(snapped, 12), low), high)


def format_value(value: float, step: float) -> str:
    """The value as the slider panel shows it: the step's decimals, at most 6 significant digits."""
    decimals = 0
    if step > 0:
        step_text = f"{step:.6f}".rstrip("0")  # 0.25 -> "0.25": two decimals, 2.5 -> one
        decimals = len(step_text.split(".")[1]) if "." in step_text else 0
    text = f"{value:.{decimals}f}"
    if len(text.lstrip("-").replace(".", "")) > 6:
        text = f"{value:.6g}"
    return "0" if text in ("-0", "-0.0") or float(text) == 0 else text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}.")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return number


def resolve_range(
    value: Optional[float],
    minimum: Optional[float],
    maximum: Optional[float],
    step: Optional[float],
    current: Optional["Parameter"] = None,
) -> Tuple[float, float, float, float, List[str]]:
    """The (value, min, max, step) a set_parameter call gives, and notes on widened ranges.

    Nulls keep the current parameter's values; for a new parameter they default to
    min = min(-10, value), max = max(10, value), step 0.1 and value 1 clamped into the
    range. A value outside the min or max given in the same call is refused; a value
    outside the stored (or default) range widens it (noted). A stored value outside a new
    range is clamped into it (noted). min must be below max and step positive; a step
    longer than the range becomes the range.

    Raises:
        ValueError: A value is not a finite number, lies outside the min/max given with it,
            min >= max, or step <= 0.
    """
    notes: List[str] = []
    new_value = _finite(value, "value") if value is not None else None
    low = _finite(minimum, "min") if minimum is not None else None
    high = _finite(maximum, "max") if maximum is not None else None
    new_step = _finite(step, "step") if step is not None else None
    if low is not None and high is not None and low >= high:
        raise ValueError(f"min ({minimum}) must be less than max ({maximum}).")
    if new_step is not None and new_step <= 0:
        raise ValueError(f"step must be positive, got {step}.")
    _refuse_outside_given(new_value, low, high)
    value_given = new_value is not None
    if current is not None:
        low = current.min if low is None else low
        high = current.max if high is None else high
        new_step = current.step if new_step is None else new_step
        if new_value is None:
            new_value = current.value
    else:
        low, high = _default_range(low, high, new_value if new_value is not None else DEFAULT_VALUE)
        new_step = DEFAULT_STEP if new_step is None else new_step
        if new_value is None:
            new_value = min(max(DEFAULT_VALUE, low), high)
    if low >= high:
        raise ValueError(f"min ({_text(low)}) must be less than max ({_text(high)}).")
    if not value_given and not low <= new_value <= high:
        new_value = min(max(new_value, low), high)
        notes.append(f"value clamped to {_text(new_value)} to stay in [{_text(low)}, {_text(high)}]")
    elif new_value < low:
        low = new_value
        notes.append(f"range widened to [{_text(low)}, {_text(high)}] to include {_text(new_value)}")
    elif new_value > high:
        high = new_value
        notes.append(f"range widened to [{_text(low)}, {_text(high)}] to include {_text(new_value)}")
    if new_step > high - low:
        new_step = high - low
    return new_value, low, high, new_step, notes


def range_around(value: float) -> Tuple[float, float, float]:
    """A slider (min, max, step) for a computed value, such as a fitted constant.

    The range is value ± |value| (± 1 for 0), rounded outward to the step, and the step
    is a 1, 2 or 5 times a power of ten near a hundredth of the range. The value itself
    is left as it is (not snapped to the step).
    """
    half = abs(value) if value != 0.0 else 1.0
    if half > 1e300:
        # value ± |value| would overflow: run from 0 to the value
        return min(0.0, value), max(0.0, value), _round_text(half / 100.0)
    raw_step = 2.0 * half / 100.0
    power = 10.0 ** math.floor(math.log10(raw_step))
    step = next(m * power for m in (1.0, 2.0, 5.0, 10.0) if m * power >= raw_step * (1.0 - 1e-12))
    low = _round_text(math.floor((value - half) / step + 1e-9) * step)
    high = _round_text(math.ceil((value + half) / step - 1e-9) * step)
    return min(low, value), max(high, value), _round_text(step)


def _round_text(number: float) -> float:
    """The number to 12 significant digits (drops 0.30000000000000004 noise)."""
    return float(f"{number:.12g}")


def _refuse_outside_given(value: Optional[float], low: Optional[float], high: Optional[float]) -> None:
    """Refuse a value outside the min or max given in the same call (the caller chose both)."""
    if value is None:
        return
    if low is not None and value < low:
        raise ValueError(
            f"value {_text(value)} is below the min {_text(low)} given with it; give a value in the range."
        )
    if high is not None and value > high:
        raise ValueError(
            f"value {_text(value)} is above the max {_text(high)} given with it; give a value in the range."
        )


def _default_range(low: Optional[float], high: Optional[float], anchor: float) -> Tuple[float, float]:
    """A new parameter's range: the given ends, the others from -10..10 widened to the anchor.

    A default end that would not lie beyond the given one is put 20 (the default width)
    beyond it, so min = 20 alone gives [20, 40].
    """
    width = DEFAULT_MAX - DEFAULT_MIN
    if low is None:
        low = min(DEFAULT_MIN, anchor)
        if high is not None and low >= high:
            low = high - width
    if high is None:
        high = max(DEFAULT_MAX, anchor)
        if high <= low:
            high = low + width
    return low, high


def _text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else repr(float(value))


class Parameter(Drawable):
    """A named real number with a range and a step, read by the expressions on the canvas.

    Attributes:
        value: The current value, always within [min, max].
        min, max: The slider range (min < max).
        step: The slider step (0 < step <= max - min); the AI may set any value.
        animation: The running animation's description (start, end, duration, repeat),
            or None. Shown to the model; a restored workspace never starts animating.
    """

    def __init__(
        self,
        name: str,
        value: float,
        min_value: float = DEFAULT_MIN,
        max_value: float = DEFAULT_MAX,
        step: float = DEFAULT_STEP,
        color: str = default_color,
    ) -> None:
        value_number = _finite(value, "value")
        low, high, step_number = _finite(min_value, "min"), _finite(max_value, "max"), _finite(step, "step")
        if low >= high:
            raise ValueError(f"min ({min_value}) must be less than max ({max_value}).")
        if step_number <= 0:
            raise ValueError(f"step must be positive, got {step}.")
        self.value: float = min(max(value_number, low), high)
        self.min: float = low
        self.max: float = high
        self.step: float = min(step_number, high - low)
        self.animation: Optional[Dict[str, Any]] = None
        super().__init__(name=name, color=color, is_renderable=False)

    def get_class_name(self) -> str:
        return "Parameter"

    def get_state(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {"value": self.value, "min": self.min, "max": self.max, "step": self.step}
        if self.animation:
            args["animation"] = dict(self.animation)
        state: Dict[str, Any] = {"name": self.name, "args": args}
        self._add_color_to_state(state)
        return state

    def set_value(self, value: float) -> None:
        """Set the value, clamped into [min, max] (the slider and animation frames call this)."""
        self.value = min(max(float(value), self.min), self.max)

    def redefine(self, value: float, min_value: float, max_value: float, step: float) -> None:
        """Replace value, range and step (already resolved by resolve_range)."""
        self.min, self.max = float(min_value), float(max_value)
        self.step = min(float(step), self.max - self.min)
        self.set_value(value)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Parameter":
        if id(self) in memo:
            return cast(Parameter, memo[id(self)])
        copy = Parameter(self.name, self.value, self.min, self.max, self.step, color=self.color)
        copy.animation = dict(self.animation) if self.animation else None
        memo[id(self)] = copy
        return copy

    def translate(self, x_offset: float, y_offset: float) -> None:
        """A parameter has no position."""
        return None

    def rotate(self, angle: float) -> None:
        """A parameter has no position."""
        return None
