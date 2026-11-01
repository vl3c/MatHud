"""
Polar Function Drawable for MatHud

Represents a polar curve r = f(θ) over a range of θ, plotted at (r·cos θ, r·sin θ), as in a
graphing calculator's polar mode: cardioids, roses, limaçons, spirals, lemniscates.

Key Features:
    - One expression r(θ), in θ (the ASCII word theta is read as θ)
    - Negative r is plotted on the opposite ray, the calculator convention
    - The same evaluation interface as ParametricFunction (evaluate, evaluate_x, evaluate_y,
      t_min, t_max), so the parametric renderer, intersections and tangents draw and use it
    - No translate or rotate: a moved r(θ) is no longer a polar curve about the origin
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Optional, Tuple, cast

import compiled_expression
from constants import default_color
from drawables.drawable import Drawable
from expression_validator import ExpressionValidator

# A leading "r =", "r(θ) =" or "r(theta) =" has one reading: the right-hand side follows
_R_PREFIX = re.compile(r"^\s*r\s*(?:\(\s*(?:θ|ϑ|theta)\s*\)\s*)?=(?!=)\s*")
# "r^2 = ..." or "r² = ...": the equation of r^2, refused with the sqrt form as the hint
_R_SQUARED_PREFIX = re.compile(r"^\s*r\s*(?:\^\s*2|\*\*\s*2|²)\s*=")
# Glued function names before theta ("costheta"), refused with a hint
_GLUED_THETA = re.compile(r"^([A-Za-z]+)theta$")
_EXAMPLE = "Example: 1 + cos(θ) or 2*sin(3*θ)."


class PolarFunction(Drawable):
    """
    A polar curve r = f(θ) for θ from theta_min to theta_max.

    Attributes:
        r_expression: The expression for r in θ, in stored form (e.g. "1+cos(θ)")
        theta_min: Start of the θ range in radians (default 0)
        theta_max: End of the θ range in radians (default 2π)
        name: Identifier for the curve
        color: Display color for rendering
    """

    DEFAULT_THETA_MIN: float = 0.0
    DEFAULT_THETA_MAX: float = 2 * math.pi
    # At most 25 turns: enough for calculator spirals, and sampled densely enough (min_samples)
    MAX_SPAN: float = 50 * math.pi
    # Samples per full turn of the range, and their cap (25 turns * 240)
    SAMPLES_PER_TURN: int = 240
    MAX_SAMPLES: int = 6000
    # Points the "has a real value" check samples over the range
    REAL_CHECK_SAMPLES: int = 64

    def __init__(
        self,
        r_expression: str,
        name: Optional[str] = None,
        theta_min: float = DEFAULT_THETA_MIN,
        theta_max: float = DEFAULT_THETA_MAX,
        color: str = default_color,
    ) -> None:
        """
        Initialize a polar curve.

        Args:
            r_expression: r as an expression in θ (or theta), such as "1 + cos(θ)"; a leading
                "r =" or "r(θ) =" is dropped
            name: Optional name for the curve
            theta_min: Start of the θ range (radians)
            theta_max: End of the θ range (radians)
            color: Display color

        Raises:
            ValueError: If the expression cannot be parsed or reads other names, the range is
                empty, not finite or longer than MAX_SPAN, or r has no real value on the range
        """
        self.theta_min: float
        self.theta_max: float
        self.theta_min, self.theta_max = self.ordered_theta_range(theta_min, theta_max)
        self.r_expression: str
        self._r_function: Callable[[float], float]
        self.r_expression, self._r_function = self.parse_r_expression(r_expression)
        self.check_has_real_value(self.r_expression, self._r_function, self.theta_min, self.theta_max)
        super().__init__(name=name or "r", color=color)

    # ------------------------------------------------------------------
    # Parsing and checks
    # ------------------------------------------------------------------

    @staticmethod
    def parse_r_expression(text: str) -> Tuple[str, Callable[[float], float]]:
        """The stored form of r(θ) and its compiled function of θ.

        The whole word theta becomes θ before fix_math_expression, so "3theta" and
        "cos 2theta" read as "3θ" and "cos 2θ" do.

        Raises:
            ValueError: If the text is empty, has an "=" other than a leading "r =", does not
                parse, or reads a name other than θ
        """
        source = str(text or "").strip()
        if _R_SQUARED_PREFIX.match(source):
            raise ValueError(
                f"r(θ) = '{text}' gives r^2, not r. For r^2 = f(θ) (a lemniscate), give sqrt(f(θ)), "
                f"such as sqrt(4*cos(2*θ)) for r^2 = 4cos(2θ). That is the whole curve when f(θ + π) = f(θ), "
                f"as for cos(2θ) or sin(2θ); otherwise also draw -sqrt(f(θ)) for the other half."
            )
        source = _R_PREFIX.sub("", source, count=1)
        if not source:
            raise ValueError(f"r(θ) is empty. Give the right-hand side of r = ..., in θ. {_EXAMPLE}")
        if "=" in source:
            raise ValueError(
                f"r(θ) = '{text}' has an '='. Give only the right-hand side of r = ..., "
                f"such as 1 + cos(θ), not an equation."
            )
        rewritten = compiled_expression.substitute_variable(source, "theta", "θ")
        try:
            stored = ExpressionValidator.fix_math_expression(rewritten)
            compiled = compiled_expression.compile_expression(stored, ("θ",), compiled_expression.FUNCTIONS_CURVE)
        except Exception as exc:
            raise ValueError(
                f"Could not read r(θ) = '{text}': {exc}. Write every function with parentheses and "
                f"products explicitly: sin(θ), 2*θ, e^(0.1*θ). {_EXAMPLE}"
            ) from exc
        free = compiled.free_names
        if free:
            raise ValueError(PolarFunction._free_names_message(text, free))
        return stored, cast(Callable[[float], float], compiled.bind())

    @staticmethod
    def _free_names_message(text: str, free: Tuple[str, ...]) -> str:
        names = ", ".join(f"'{name}'" for name in free)
        message = f"r(θ) = '{text}' may use only θ (or theta); {names} is not defined."
        for name in free:
            glued = _GLUED_THETA.match(name)
            if glued:
                return f"{message} Write {glued.group(1)}(theta), with parentheses."
        return f"{message} {_EXAMPLE}"

    @classmethod
    def ordered_theta_range(cls, theta_min: Any, theta_max: Any) -> Tuple[float, float]:
        """The θ range in increasing order (a reversed range traces the same curve).

        Raises:
            ValueError: If an end is not a finite number, the ends are equal, or the range is
                longer than MAX_SPAN
        """
        try:
            low, high = float(theta_min), float(theta_max)
        except (TypeError, ValueError):
            raise ValueError(f"theta_min and theta_max must be numbers, got {theta_min} and {theta_max}.")
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"theta_min and theta_max must be finite numbers, got {theta_min} and {theta_max}.")
        if low == high:
            raise ValueError(f"theta_min and theta_max are both {low}; theta_min must be less than theta_max.")
        if low > high:
            low, high = high, low
        if high - low > cls.MAX_SPAN:
            raise ValueError(
                f"The θ range [{low:g}, {high:g}] is longer than 50π (~157.08, 25 turns); use a shorter range. "
                f"θ is in radians: 360° is 2π (~6.283)."
            )
        return low, high

    @classmethod
    def check_has_real_value(
        cls, r_expression: str, r_function: Callable[[float], float], theta_min: float, theta_max: float
    ) -> None:
        """Refuse a curve with no real, finite r at any of REAL_CHECK_SAMPLES points of the range.

        Raises:
            ValueError: Quoting the first error when no sample could be evaluated at all (an
                expression such as sin^2θ that fails everywhere), else "no real value"
        """
        steps = cls.REAL_CHECK_SAMPLES - 1
        first_error: Optional[Exception] = None
        evaluated = False
        for index in range(steps + 1):
            theta = theta_min + (theta_max - theta_min) * index / steps
            try:
                value = r_function(theta)
            except (TypeError, NameError, AttributeError) as exc:
                # A complex r (sqrt of a negative) fails float() with a TypeError naming complex
                if isinstance(exc, TypeError) and "complex" in str(exc):
                    evaluated = True
                else:
                    first_error = first_error or exc
                continue
            except Exception:
                evaluated = True
                continue
            evaluated = True
            if isinstance(value, (int, float)) and math.isfinite(value):
                return
        range_text = f"[{theta_min:g}, {theta_max:g}]"
        if not evaluated and first_error is not None:
            raise ValueError(
                f"r(θ) = {r_expression} could not be evaluated for θ in {range_text}: {first_error}. "
                f"Write powers of functions as sin(θ)^2, not sin^2θ. {_EXAMPLE}"
            )
        raise ValueError(f"r(θ) = {r_expression} has no real value for θ in {range_text}.")

    # ------------------------------------------------------------------
    # Evaluation (the ParametricFunction interface)
    # ------------------------------------------------------------------

    @property
    def t_min(self) -> float:
        """θ_min, under the parameter name the parametric renderer and intersections read."""
        return self.theta_min

    @property
    def t_max(self) -> float:
        """θ_max, under the parameter name the parametric renderer and intersections read."""
        return self.theta_max

    @property
    def min_samples(self) -> int:
        """Uniform samples the renderer takes at least: SAMPLES_PER_TURN per turn of the range."""
        turns = (self.theta_max - self.theta_min) / (2 * math.pi)
        return min(self.MAX_SAMPLES, int(math.ceil(turns * self.SAMPLES_PER_TURN)))

    def radius(self, theta: float) -> float:
        """r(θ), which may be negative.

        Raises:
            ValueError: If r is undefined, not real or not finite at θ
        """
        try:
            value = self._r_function(theta)
        except Exception as exc:
            raise ValueError(f"r(θ) is undefined or not real at θ={theta:g}") from exc
        if not math.isfinite(value):
            raise ValueError(f"r(θ) is not finite at θ={theta:g}")
        return value

    def evaluate(self, theta: float) -> Tuple[float, float]:
        """The point (r·cos θ, r·sin θ); a negative r lies on the opposite ray.

        Raises:
            ValueError: If r is undefined, not real or not finite at θ
        """
        r = self.radius(theta)
        return (r * math.cos(theta), r * math.sin(theta))

    def evaluate_x(self, theta: float) -> float:
        """x = r(θ)·cos θ."""
        return self.radius(theta) * math.cos(theta)

    def evaluate_y(self, theta: float) -> float:
        """y = r(θ)·sin θ."""
        return self.radius(theta) * math.sin(theta)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_class_name(self) -> str:
        """Return the class name for serialization and type identification."""
        return "PolarFunction"

    def get_state(self) -> Dict[str, Any]:
        """Serialize the curve for workspace persistence."""
        return {
            "name": self.name,
            "args": {
                "r_expression": self.r_expression,
                "theta_min": self.theta_min,
                "theta_max": self.theta_max,
                "color": self.color,
            },
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "PolarFunction":
        """A copy rebuilt from the stored expression and range."""
        if id(self) in memo:
            return cast("PolarFunction", memo[id(self)])
        copy = PolarFunction(
            r_expression=self.r_expression,
            name=self.name,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            color=self.color,
        )
        memo[id(self)] = copy
        return copy

    def update_color(self, color: str) -> None:
        """Update the display color of the curve."""
        self.color = str(color)

    def update_theta_range(self, theta_min: float, theta_max: float) -> None:
        """Set both ends of the θ range (ordered and checked, see ordered_theta_range).

        Raises:
            ValueError: If the range is invalid or r has no real value on it; the curve is
                then unchanged
        """
        low, high = self.ordered_theta_range(theta_min, theta_max)
        self.check_has_real_value(self.r_expression, self._r_function, low, high)
        self.theta_min, self.theta_max = low, high
