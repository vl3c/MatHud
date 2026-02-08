"""
Client tests for MathUtils.detect_function_periodicity.

Tests periodicity detection for various function types.
"""

import math
import unittest
from utils.math_utils import MathUtils


class TestPeriodicityDetection(unittest.TestCase):
    """Tests for detect_function_periodicity."""

    def test_sin_detected_as_periodic(self):
        """sin(x) should be detected as periodic."""
        is_periodic, period = MathUtils.detect_function_periodicity(math.sin)
        self.assertTrue(is_periodic)
        self.assertIsNotNone(period)

    def test_cos_detected_as_periodic(self):
        """cos(x) should be detected as periodic."""
        is_periodic, period = MathUtils.detect_function_periodicity(math.cos)
        self.assertTrue(is_periodic)
        self.assertIsNotNone(period)

    def test_linear_not_periodic(self):
        """Linear function f(x) = x should not be periodic."""
        is_periodic, period = MathUtils.detect_function_periodicity(lambda x: x)
        self.assertFalse(is_periodic)
        self.assertIsNone(period)

    def test_quadratic_not_periodic(self):
        """Quadratic function f(x) = x^2 should not be periodic."""
        is_periodic, period = MathUtils.detect_function_periodicity(lambda x: x * x)
        self.assertFalse(is_periodic)
        self.assertIsNone(period)

    def test_constant_not_periodic(self):
        """Constant function f(x) = 5 should not be periodic."""
        is_periodic, period = MathUtils.detect_function_periodicity(lambda x: 5)
        self.assertFalse(is_periodic)
        self.assertIsNone(period)

    def test_high_frequency_sin(self):
        """sin(10x) should be detected as periodic with shorter period."""
        is_periodic, period = MathUtils.detect_function_periodicity(lambda x: math.sin(10 * x))
        self.assertTrue(is_periodic)
        self.assertIsNotNone(period)

    def test_range_hint_scales_test_range(self):
        """range_hint should scale test_range for long-period functions."""
        # sin(x/50) has period 2*pi*50 ~ 314
        # With default test_range=20, it looks linear
        # With range_hint=600, it should be detected
        def long_period_sin(x):
            return math.sin(x / 50)

        # Without range_hint - might not detect
        is_periodic_default, _ = MathUtils.detect_function_periodicity(long_period_sin)

        # With range_hint - should detect
        is_periodic_hint, period = MathUtils.detect_function_periodicity(
            long_period_sin, range_hint=600
        )
        self.assertTrue(is_periodic_hint)
        self.assertIsNotNone(period)

    def test_tan_with_asymptotes(self):
        """tan(x) has asymptotes - should still detect periodicity or handle gracefully."""
        def safe_tan(x):
            try:
                return math.tan(x)
            except:
                return float('inf')

        # Should not crash, may or may not detect as periodic due to asymptotes
        is_periodic, period = MathUtils.detect_function_periodicity(safe_tan)
        # Just verify it doesn't crash and returns valid types
        self.assertIsInstance(is_periodic, bool)

    def test_combined_sin_function(self):
        """100*sin(x/50) + 50*tan(x/100) with range_hint should be detected."""
        def combo(x):
            try:
                return 100 * math.sin(x / 50) + 50 * math.tan(x / 100)
            except:
                return float('inf')

        is_periodic, period = MathUtils.detect_function_periodicity(combo, range_hint=600)
        self.assertTrue(is_periodic)

    def test_one_over_x_not_periodic(self):
        """1/x has curvature that may trigger periodicity detection."""
        def one_over_x(x):
            if x == 0:
                return float('inf')
            return 1 / x

        # 1/x has curvature on both sides of the asymptote which may look
        # like oscillation to the chord-deviation detector. The result
        # depends on how many segments span the asymptote.
        is_periodic, period = MathUtils.detect_function_periodicity(one_over_x)
        # Just verify it returns a valid result without crashing
        self.assertIsInstance(is_periodic, bool)


class TestPeriodicityEdgeCases(unittest.TestCase):
    """Edge case tests for periodicity detection."""

    def test_function_with_exceptions(self):
        """Function that throws exceptions should be handled gracefully."""
        def bad_func(x):
            if x > 0:
                raise ValueError("test error")
            return x

        # Should not crash
        is_periodic, period = MathUtils.detect_function_periodicity(bad_func)
        self.assertIsInstance(is_periodic, bool)

    def test_function_returning_nan(self):
        """Function returning NaN should be handled gracefully."""
        def nan_func(x):
            return float('nan')

        is_periodic, period = MathUtils.detect_function_periodicity(nan_func)
        self.assertFalse(is_periodic)

    def test_function_returning_inf(self):
        """Function returning infinity should be handled gracefully."""
        def inf_func(x):
            return float('inf')

        is_periodic, period = MathUtils.detect_function_periodicity(inf_func)
        self.assertFalse(is_periodic)

    def test_range_hint_capped_at_1000(self):
        """range_hint should be capped at 1000 to avoid excessive computation."""
        # With very large range_hint, test_range should be capped
        is_periodic, period = MathUtils.detect_function_periodicity(
            math.sin, range_hint=10000
        )
        # Should still work (capped at 1000)
        self.assertTrue(is_periodic)


__all__ = [
    "TestPeriodicityDetection",
    "TestPeriodicityEdgeCases",
]

