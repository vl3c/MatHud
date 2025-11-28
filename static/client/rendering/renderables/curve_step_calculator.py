"""
Step calculator for curve rendering.

Provides algorithm to determine the optimal sampling step size for rendering
mathematical functions and parametric curves as polylines.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple


class PixelStepCalculator:
    """
    Step calculator based on target pixel distance between points.
    
    Computes step size to maintain approximately 5 pixels between rendered points,
    clamped to produce between 50-200 points total.
    """

    @staticmethod
    def calculate(
        left_bound: float,
        right_bound: float,
        _eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> float:
        range_width = right_bound - left_bound
        if range_width <= 0:
            return 1.0

        try:
            screen_left = math_to_screen(left_bound, 0)
            screen_right = math_to_screen(right_bound, 0)
            screen_width = abs(screen_right[0] - screen_left[0])
            pixels_per_unit = screen_width / range_width if range_width > 0 else 1.0
        except Exception:
            pixels_per_unit = 32.0

        target_pixel_step = 5.0
        step = target_pixel_step / pixels_per_unit if pixels_per_unit > 0 else range_width / 200

        min_step = range_width / 200
        max_step = range_width / 50
        
        return max(min_step, min(step, max_step))
