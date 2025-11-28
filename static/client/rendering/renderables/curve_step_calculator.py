"""
Step calculators for curve rendering.

Provides algorithms to determine the optimal sampling step size for rendering
mathematical functions and parametric curves as polylines.
"""

from __future__ import annotations

import math as _m
import re
from typing import Any, Callable, Optional, Tuple

MAX_POINTS: float = 160.0
BASE_SAMPLES: float = 200.0
TRIG_POINTS_TARGET: float = 320.0
QUADRATIC_STEP_MULTIPLIER: float = 2.0
SLOPE_TARGET_PIXELS: float = 2.0

# Step calculator selection: "heuristic", "pixel", or "adaptive"
STEP_CALCULATOR_MODE: str = "pixel"

ADAPTIVE_MAX_PIXEL_DISTANCE: float = 10.0
ADAPTIVE_MAX_ANGLE_DEG: float = 10.0
ADAPTIVE_MAX_BEND: float = _m.tan(_m.radians(ADAPTIVE_MAX_ANGLE_DEG))
ADAPTIVE_MAX_DEPTH: int = 8
ADAPTIVE_MIN_SAMPLES: int = 200
ADAPTIVE_PROBE_COUNT: int = 64


class StepCalculator:
    """Calculates sampling step size using heuristics based on function type."""

    @staticmethod
    def calculate(
        left_bound: float,
        right_bound: float,
        function_string: str,
        eval_func: Callable[[float], Any],
        scale_factor: float = 1.0,
    ) -> float:
        range_width = right_bound - left_bound
        if range_width <= 0:
            return 1.0
        step = StepCalculator._base_step(range_width, function_string)
        step = StepCalculator._apply_cap(step, range_width)
        step = StepCalculator._adjust_for_slope(step, left_bound, right_bound, eval_func, scale_factor)
        return step

    @staticmethod
    def _base_step(range_width: float, function_string: str) -> float:
        base = range_width / BASE_SAMPLES
        if 'sin' in function_string or 'cos' in function_string:
            return StepCalculator._trig_step(range_width, base, function_string)
        if any(p in function_string for p in ['x**2', 'x^2']):
            return base * QUADRATIC_STEP_MULTIPLIER
        return base

    @staticmethod
    def _trig_step(range_width: float, base_step: float, function_string: str) -> float:
        matches = re.findall(r'(?:sin|cos)\((\d+(?:\.\d+)?)\*?x\)', function_string)
        freq_multiplier = float(matches[0]) if matches else 1.0
        period = 2 * _m.pi / freq_multiplier
        visible_periods = range_width / period if period != 0 else 1.0
        if visible_periods <= 1:
            return base_step
        points_per_period = TRIG_POINTS_TARGET / max(visible_periods, 1e-9)
        points_per_period = min(MAX_POINTS, points_per_period)
        step_trig = period / points_per_period if points_per_period > 0 else base_step
        cap_step = range_width / MAX_POINTS
        return max(cap_step, step_trig)

    @staticmethod
    def _apply_cap(step: float, range_width: float) -> float:
        if step <= 0:
            return step
        cap_step = range_width / MAX_POINTS
        return max(cap_step, step)

    @staticmethod
    def _adjust_for_slope(
        step: float,
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        scale_factor: float,
    ) -> float:
        try:
            eps = max(1e-6, (right_bound - left_bound) / 1000.0)
            cx = (left_bound + right_bound) / 2.0
            y1 = eval_func(cx - eps)
            y2 = eval_func(cx + eps)
            if isinstance(y1, (int, float)) and isinstance(y2, (int, float)):
                slope_abs = abs(y2 - y1) / (2.0 * eps) if eps > 0 else 0.0
                if slope_abs > 0:
                    scale = scale_factor or 1.0
                    desired_step = (SLOPE_TARGET_PIXELS / scale) / slope_abs
                    cap_step = (right_bound - left_bound) / MAX_POINTS
                    step = min(step, max(cap_step, desired_step))
        except Exception:
            pass
        return step


class PixelStepCalculator:
    """
    Simple step calculator based on target pixel distance between points.
    
    Computes step size to maintain approximately 3 pixels between rendered points,
    clamped to produce between 50-200 points total.
    """

    @staticmethod
    def calculate(
        left_bound: float,
        right_bound: float,
        _eval_func: Callable[[float], Any],  # unused, kept for API compatibility
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> float:
        range_width = right_bound - left_bound
        if range_width <= 0:
            return 1.0

        # Compute scale factor from math_to_screen
        try:
            screen_left = math_to_screen(left_bound, 0)
            screen_right = math_to_screen(right_bound, 0)
            screen_width = abs(screen_right[0] - screen_left[0])
            pixels_per_unit = screen_width / range_width if range_width > 0 else 1.0
        except Exception:
            pixels_per_unit = 32.0  # fallback

        # Target ~5 pixels between points for smooth curves
        target_pixel_step = 5.0
        step = target_pixel_step / pixels_per_unit if pixels_per_unit > 0 else range_width / 200

        # Ensure reasonable bounds: at least 50 points, at most 200 points
        min_step = range_width / 200
        max_step = range_width / 50
        
        return max(min_step, min(step, max_step))


class AdaptiveStepCalculator:
    """
    Calculates step size using adaptive bisection inspired by GeoGebra.

    Instead of using fixed formulas, this samples the function at probe points
    and bisects until segments meet pixel distance and angle criteria.
    """

    @staticmethod
    def calculate(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> float:
        range_width = right_bound - left_bound
        if range_width <= 0:
            return 1.0

        max_step = range_width / ADAPTIVE_MIN_SAMPLES
        min_step = range_width / (2 ** ADAPTIVE_MAX_DEPTH)

        worst_step = max_step
        probe_step = range_width / ADAPTIVE_PROBE_COUNT

        x = left_bound
        prev_screen = None
        prev_diff = None

        while x < right_bound:
            try:
                y = eval_func(x)
                if AdaptiveStepCalculator._is_valid(y):
                    screen = math_to_screen(x, y)
                    if prev_screen is not None:
                        diff = (screen[0] - prev_screen[0], screen[1] - prev_screen[1])
                        required = AdaptiveStepCalculator._find_required_step(
                            x - probe_step, x, prev_screen, screen, diff, prev_diff,
                            eval_func, math_to_screen, probe_step, 0
                        )
                        worst_step = min(worst_step, required)
                        prev_diff = diff
                    else:
                        prev_diff = None
                    prev_screen = screen
                else:
                    prev_screen = None
                    prev_diff = None
            except Exception:
                prev_screen = None
                prev_diff = None
            x += probe_step

        return max(min_step, worst_step)

    @staticmethod
    def _is_valid(y: Any) -> bool:
        if y is None:
            return False
        if isinstance(y, float) and (y != y or abs(y) == float('inf')):
            return False
        return True

    @staticmethod
    def _is_distance_ok(diff: Tuple[float, float]) -> bool:
        return abs(diff[0]) <= ADAPTIVE_MAX_PIXEL_DISTANCE and abs(diff[1]) <= ADAPTIVE_MAX_PIXEL_DISTANCE

    @staticmethod
    def _is_angle_ok(prev_diff: Optional[Tuple[float, float]], diff: Tuple[float, float]) -> bool:
        if prev_diff is None:
            return True
        dot = prev_diff[0] * diff[0] + prev_diff[1] * diff[1]
        if dot <= 0:
            return False
        det = abs(prev_diff[0] * diff[1] - prev_diff[1] * diff[0])
        return det < ADAPTIVE_MAX_BEND * dot

    @staticmethod
    def _find_required_step(
        x_left: float,
        x_right: float,
        screen_left: Tuple[float, float],
        screen_right: Tuple[float, float],
        diff: Tuple[float, float],
        prev_diff: Optional[Tuple[float, float]],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        current_step: float,
        depth: int,
    ) -> float:
        if AdaptiveStepCalculator._is_distance_ok(diff) and AdaptiveStepCalculator._is_angle_ok(prev_diff, diff):
            return current_step

        if depth >= ADAPTIVE_MAX_DEPTH:
            return current_step / 2

        mid_x = (x_left + x_right) / 2
        half_step = current_step / 2

        try:
            mid_y = eval_func(mid_x)
            if not AdaptiveStepCalculator._is_valid(mid_y):
                return half_step
            screen_mid = math_to_screen(mid_x, mid_y)
        except Exception:
            return half_step

        left_diff = (screen_mid[0] - screen_left[0], screen_mid[1] - screen_left[1])
        right_diff = (screen_right[0] - screen_mid[0], screen_right[1] - screen_mid[1])

        left_step = AdaptiveStepCalculator._find_required_step(
            x_left, mid_x, screen_left, screen_mid, left_diff, prev_diff,
            eval_func, math_to_screen, half_step, depth + 1
        )
        right_step = AdaptiveStepCalculator._find_required_step(
            mid_x, x_right, screen_mid, screen_right, right_diff, left_diff,
            eval_func, math_to_screen, half_step, depth + 1
        )

        return min(left_step, right_step)

