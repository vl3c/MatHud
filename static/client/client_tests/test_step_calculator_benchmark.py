"""
Benchmark and quality tests for step calculator.
"""

from __future__ import annotations

import math
import time
import unittest
from typing import Callable, Tuple

from coordinate_mapper import CoordinateMapper
from drawables.function import Function
from rendering.renderables import FunctionRenderable
from rendering.renderables.curve_step_calculator import PixelStepCalculator


class TestStepCalculatorPerformance(unittest.TestCase):
    """Performance benchmarks for step calculator."""

    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.iterations = 100
        self.test_functions = [
            ("x", "linear"),
            ("sin(x)", "sin"),
            ("100*sin(x)", "high_amplitude_sin"),
            ("sin(100*x)", "high_frequency_sin"),
            ("x^2", "quadratic"),
        ]

    def _create_eval_func(self, func_str: str) -> Callable[[float], float]:
        func = Function(func_str, name="test")
        return func.function

    def test_step_calculator_performance(self) -> None:
        """Benchmark step calculator across various function types."""
        left, right = -10.0, 10.0
        print("\n### === STEP CALCULATOR PERFORMANCE ===")

        total_time = 0.0
        for func_str, name in self.test_functions:
            eval_func = self._create_eval_func(func_str)

            start = time.time()
            for _ in range(self.iterations):
                PixelStepCalculator.calculate(
                    left, right, eval_func,
                    self.mapper.math_to_screen, 480.0
                )
            elapsed = time.time() - start
            avg_ms = elapsed * 1000 / self.iterations
            total_time += avg_ms

            step = PixelStepCalculator.calculate(
                left, right, eval_func,
                self.mapper.math_to_screen, 480.0
            )
            num_points = int((right - left) / step) if step > 0 else 0
            print(f"###   {name}: {avg_ms:.3f}ms, step={step:.4f}, points={num_points}")

        avg_total = total_time / len(self.test_functions)
        print(f"### Average: {avg_total:.3f}ms")
        self.assertLess(avg_total, 10.0, "Step calculation should be under 10ms on average")


class TestStepCalculatorQuality(unittest.TestCase):
    """Quality tests for rendered curves."""

    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)

    def _count_sharp_angles(self, func_str: str, min_angle_deg: float = 30) -> Tuple[int, int]:
        """Count angles below threshold. Returns (violations, total_checked)."""
        func = Function(func_str, name="test", left_bound=-100, right_bound=100)
        renderable = FunctionRenderable(func, self.mapper)
        result = renderable.build_screen_paths()

        min_angle_rad = math.radians(min_angle_deg)
        violations = 0
        total = 0

        for path in result.paths:
            if len(path) < 3:
                continue
            for i in range(1, len(path) - 1):
                angle = renderable._compute_angle(path[i - 1], path[i], path[i + 1])
                total += 1
                if angle < min_angle_rad:
                    violations += 1

        return violations, total

    def test_high_amplitude_sin_quality(self) -> None:
        """Verify 100*sin(x) has acceptable smoothness."""
        violations, total = self._count_sharp_angles("100*sin(x)")
        rate = violations / total if total > 0 else 0
        print(f"\n### 100*sin(x): {violations} violations out of {total} angles ({rate:.1%})")
        self.assertLess(rate, 0.30, f"Should have < 30% sharp angles, got {rate:.1%}")

    def test_high_frequency_sin_quality(self) -> None:
        """Verify sin(100*x) has acceptable smoothness."""
        violations, total = self._count_sharp_angles("sin(100*x)")
        rate = violations / total if total > 0 else 0
        print(f"\n### sin(100*x): {violations} violations out of {total} angles ({rate:.1%})")
        self.assertLess(rate, 0.30, f"Should have < 30% sharp angles, got {rate:.1%}")


__all__ = [
    "TestStepCalculatorPerformance",
    "TestStepCalculatorQuality",
]
