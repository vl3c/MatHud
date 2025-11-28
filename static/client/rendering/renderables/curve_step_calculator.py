"""
Step calculator for curve rendering.

Determines optimal sampling step size for rendering mathematical functions
as polylines, adjusting for amplitude and oscillation frequency.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Tuple

PROBE_COUNT: int = 10
MIN_POINTS: int = 50
MAX_POINTS: int = 200
MAX_POINTS_HIGH_DETAIL: int = 1500
TARGET_PIXEL_STEP: float = 5.0
AMPLITUDE_EXPONENT: float = 1.5


class PixelStepCalculator:
    """
    Step calculator based on pixel distance and function characteristics.
    
    Adjusts sampling density based on:
    - Screen pixel density
    - Function amplitude relative to screen height
    - Oscillation frequency (detected via direction changes)
    """

    @staticmethod
    def calculate(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        screen_height: float = 480.0,
    ) -> float:
        range_width = right_bound - left_bound
        if range_width <= 0:
            return 1.0

        base_step, _ = PixelStepCalculator._compute_base_step(
            left_bound, right_bound, range_width, math_to_screen
        )

        return PixelStepCalculator._amplitude_adjusted_step(
            range_width, eval_func, left_bound, right_bound,
            base_step, math_to_screen, screen_height
        )

    @staticmethod
    def _compute_base_step(
        left_bound: float,
        right_bound: float,
        range_width: float,
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> Tuple[float, float]:
        try:
            screen_left = math_to_screen(left_bound, 0)
            screen_right = math_to_screen(right_bound, 0)
            screen_width = abs(screen_right[0] - screen_left[0])
            pixels_per_unit = screen_width / range_width if range_width > 0 else 1.0
        except Exception:
            pixels_per_unit = 32.0

        step = TARGET_PIXEL_STEP / pixels_per_unit if pixels_per_unit > 0 else range_width / MAX_POINTS
        return step, pixels_per_unit

    @staticmethod
    def _clamp_step(step: float, range_width: float, max_points: int) -> float:
        min_step = range_width / max_points
        max_step = range_width / MIN_POINTS
        return max(min_step, min(step, max_step))

    @staticmethod
    def _probe_y_values(
        eval_func: Callable[[float], Any],
        left_bound: float,
        right_bound: float,
    ) -> List[float]:
        """Sample function at evenly spaced points."""
        values: List[float] = []
        probe_step = (right_bound - left_bound) / (PROBE_COUNT + 1)
        for i in range(1, PROBE_COUNT + 1):
            x = left_bound + i * probe_step
            try:
                y = eval_func(x)
                if isinstance(y, (int, float)) and math.isfinite(y):
                    values.append(float(y))
            except Exception:
                pass
        return values

    @staticmethod
    def _count_sign_changes(values: List[float]) -> int:
        """Count direction changes in a sequence (indicates oscillation frequency)."""
        if len(values) < 3:
            return 0
        changes = 0
        for i in range(1, len(values) - 1):
            diff_before = values[i] - values[i - 1]
            diff_after = values[i + 1] - values[i]
            if diff_before * diff_after < 0:
                changes += 1
        return changes

    @staticmethod
    def _amplitude_adjusted_step(
        range_width: float,
        eval_func: Callable[[float], Any],
        left_bound: float,
        right_bound: float,
        base_step: float,
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        screen_height: float,
    ) -> float:
        """
        Adjust step based on function amplitude and oscillation frequency.
        
        High amplitude functions get more samples to ensure smooth peaks.
        High frequency oscillations also get more samples.
        """
        y_values = PixelStepCalculator._probe_y_values(eval_func, left_bound, right_bound)
        
        if len(y_values) < 2:
            return PixelStepCalculator._clamp_step(base_step, range_width, MAX_POINTS)

        y_min = min(y_values)
        y_max = max(y_values)

        try:
            _, screen_y_min = math_to_screen(0, y_min)
            _, screen_y_max = math_to_screen(0, y_max)
            screen_amplitude = abs(screen_y_max - screen_y_min)
        except Exception:
            screen_amplitude = abs(y_max - y_min)

        amplitude_ratio = screen_amplitude / screen_height if screen_height > 0 else 0
        amplitude_factor = 1.0 + amplitude_ratio * 4.0

        sign_changes = PixelStepCalculator._count_sign_changes(y_values)
        frequency_factor = 1.0 + sign_changes * 0.5

        combined_factor = amplitude_factor * frequency_factor
        adjusted_step = base_step / combined_factor
        
        needs_high_detail = amplitude_ratio > 0.1 or sign_changes > 1
        max_points = MAX_POINTS_HIGH_DETAIL if needs_high_detail else MAX_POINTS
        
        return PixelStepCalculator._clamp_step(adjusted_step, range_width, max_points)
