"""
Server tests for AdaptiveSampler.

Tests the recursive subdivision algorithm for generating adaptive sample points.
"""

import math
import sys
import time
import unittest
from typing import Tuple

sys.path.insert(0, 'static/client/rendering/renderables')

from adaptive_sampler import (
    AdaptiveSampler,
    MAX_DEPTH,
    PIXEL_TOLERANCE,
)


def identity_transform(x: float, y: float) -> Tuple[float, float]:
    """Identity transform for testing (math coords = screen coords)."""
    return (x, y)


def scaled_transform(x: float, y: float) -> Tuple[float, float]:
    """Scaled transform: 1 math unit = 32 pixels, centered at (320, 240)."""
    return (320 + x * 32, 240 - y * 32)


def get_samples(*args, **kwargs) -> list:
    """Helper to extract just samples from generate_samples result."""
    result = AdaptiveSampler.generate_samples(*args, **kwargs)
    return result[0] if isinstance(result, tuple) else result


class TestAdaptiveSamplerBasic(unittest.TestCase):
    """Basic functionality tests."""

    def test_empty_range(self) -> None:
        samples = get_samples(10, 10, lambda x: x, identity_transform)
        self.assertEqual(samples, [])

    def test_inverted_range(self) -> None:
        samples = get_samples(10, 0, lambda x: x, identity_transform)
        self.assertEqual(samples, [])

    def test_includes_endpoints(self) -> None:
        samples = get_samples(-5, 5, lambda x: x, identity_transform)
        self.assertIn(-5, samples)
        self.assertIn(5, samples)

    def test_samples_sorted(self) -> None:
        samples = get_samples(-10, 10, lambda x: math.sin(x), scaled_transform)
        self.assertEqual(samples, sorted(samples))

    def test_linear_covers_full_range_after_pan(self) -> None:
        """Test that linear function samples cover exact bounds when panning."""
        # Simulate different pan positions
        pan_positions = [
            (-10.0, 10.0),
            (-15.5, 4.5),
            (0.0, 20.0),
            (-100.0, -80.0),
            (50.0, 70.0),
            (-3.14159, 3.14159),
        ]
        for left, right in pan_positions:
            samples = get_samples(left, right, lambda x: x, scaled_transform)
            self.assertAlmostEqual(
                samples[0], left, places=10,
                msg=f"First sample {samples[0]} != left bound {left}"
            )
            self.assertAlmostEqual(
                samples[-1], right, places=10,
                msg=f"Last sample {samples[-1]} != right bound {right}"
            )

    def test_linear_no_gaps_in_coverage(self) -> None:
        """Test that samples cover the full range with no gaps for linear functions."""
        left, right = -10.0, 10.0
        samples = get_samples(left, right, lambda x: x, scaled_transform)
        
        # First sample should be at left bound
        self.assertEqual(samples[0], left)
        # Last sample should be at right bound
        self.assertEqual(samples[-1], right)
        # For linear function with INITIAL_SEGMENTS=2, we should have exactly 3 points
        # (left, middle, right) since no subdivision is needed
        self.assertGreaterEqual(len(samples), 3)

    def test_linear_wide_range_includes_all_points(self) -> None:
        """
        Test that y=x over a wide range still includes first point.
        
        This catches the bug where panning vertically causes the lower portion
        of y=x to disappear because the first off-screen point was lost.
        """
        # Large range - with only 3 sample points, multiple points might be "off-screen"
        # in screen coordinate terms depending on the transform
        left, right = -100.0, 100.0
        samples = get_samples(left, right, lambda x: x, scaled_transform)
        
        self.assertEqual(samples[0], left, "First sample must be at left bound")
        self.assertEqual(samples[-1], right, "Last sample must be at right bound")
        
        # Verify midpoint is included
        mid = (left + right) / 2
        self.assertIn(mid, samples, "Midpoint should be in samples")


class TestAdaptiveSamplerLinear(unittest.TestCase):
    """Tests for linear functions (should produce few samples)."""

    def test_linear_minimal_samples(self) -> None:
        """Linear function y=x should only need initial samples plus midpoints."""
        samples = get_samples(-10, 10, lambda x: x, scaled_transform)
        self.assertLessEqual(len(samples), 20)

    def test_linear_with_offset(self) -> None:
        """Linear function y=2x+5 should also need few samples."""
        samples = get_samples(-10, 10, lambda x: 2 * x + 5, scaled_transform)
        self.assertLessEqual(len(samples), 20)

    def test_constant_function(self) -> None:
        """Constant function y=5 should need minimal samples."""
        samples = get_samples(-10, 10, lambda x: 5, scaled_transform)
        self.assertLessEqual(len(samples), 20)


def high_amplitude_transform(x: float, y: float) -> Tuple[float, float]:
    """Transform with higher amplitude to make curves more visible."""
    return (320 + x * 32, 240 - y * 100)


class TestAdaptiveSamplerCurved(unittest.TestCase):
    """Tests for curved functions (should produce more samples)."""

    def test_quadratic_more_samples_than_linear(self) -> None:
        """x^2 should need more samples than y=x."""
        linear_samples = get_samples(-10, 10, lambda x: x, scaled_transform)
        quad_samples = get_samples(-10, 10, lambda x: x * x, scaled_transform)
        self.assertGreater(len(quad_samples), len(linear_samples))

    def test_cubic_asymmetric_range(self) -> None:
        """x^3 over asymmetric range should need more samples than linear."""
        linear_samples = get_samples(0, 5, lambda x: x, high_amplitude_transform)
        cubic_samples = get_samples(0, 5, lambda x: x * x * x, high_amplitude_transform)
        self.assertGreater(len(cubic_samples), len(linear_samples))

    def test_exponential_needs_samples(self) -> None:
        """Exponential function should produce multiple samples."""
        samples = get_samples(0, 5, lambda x: math.exp(x), high_amplitude_transform)
        self.assertGreater(len(samples), 5)


class TestAdaptiveSamplerMaxDepth(unittest.TestCase):
    """Tests for max depth limiting."""

    def test_respects_max_depth(self) -> None:
        """Should not exceed 2^MAX_DEPTH + 1 samples."""
        max_possible = (2 ** MAX_DEPTH) + 1
        samples = get_samples(-10, 10, lambda x: math.sin(100 * x), scaled_transform)
        self.assertLessEqual(len(samples), max_possible)


class TestAdaptiveSamplerInvalidValues(unittest.TestCase):
    """Tests for handling invalid function values."""

    def test_handles_nan(self) -> None:
        """Should handle NaN values gracefully."""
        def func_with_nan(x: float) -> float:
            if x == 0:
                return float('nan')
            return x

        samples = get_samples(-10, 10, func_with_nan, identity_transform)
        self.assertIn(-10, samples)
        self.assertIn(10, samples)

    def test_handles_inf(self) -> None:
        """Should handle infinity values gracefully."""
        def func_with_inf(x: float) -> float:
            if abs(x) < 0.01:
                return float('inf')
            return 1 / x

        samples = get_samples(-10, 10, func_with_inf, identity_transform)
        self.assertIn(-10, samples)
        self.assertIn(10, samples)

    def test_handles_exception(self) -> None:
        """Should handle function exceptions gracefully."""
        def func_with_exception(x: float) -> float:
            if x == 0:
                raise ValueError("Division by zero")
            return 1 / x

        samples = get_samples(-10, 10, func_with_exception, identity_transform)
        self.assertIn(-10, samples)
        self.assertIn(10, samples)


class TestIsStraight(unittest.TestCase):
    """Tests for the _is_straight helper."""

    def test_collinear_points(self) -> None:
        """Collinear points should be straight."""
        p_left = (0.0, 0.0)
        p_mid = (5.0, 5.0)
        p_right = (10.0, 10.0)
        self.assertTrue(AdaptiveSampler._is_straight(p_left, p_mid, p_right))

    def test_points_with_small_deviation(self) -> None:
        """Points with deviation < PIXEL_TOLERANCE should be straight."""
        p_left = (0.0, 0.0)
        p_mid = (5.0, 0.2)  # Small deviation (< 0.5 pixel tolerance)
        p_right = (10.0, 0.0)
        self.assertTrue(AdaptiveSampler._is_straight(p_left, p_mid, p_right))

    def test_points_with_large_deviation(self) -> None:
        """Points with deviation > PIXEL_TOLERANCE should not be straight."""
        p_left = (0.0, 0.0)
        p_mid = (5.0, 10.0)  # Large deviation
        p_right = (10.0, 0.0)
        self.assertFalse(AdaptiveSampler._is_straight(p_left, p_mid, p_right))

    def test_coincident_endpoints(self) -> None:
        """Coincident endpoints should be considered straight."""
        p_left = (5.0, 5.0)
        p_mid = (5.0, 5.0)
        p_right = (5.0, 5.0)
        self.assertTrue(AdaptiveSampler._is_straight(p_left, p_mid, p_right))


class TestAdaptiveSamplerPerformance(unittest.TestCase):
    """Performance comparison tests."""

    def test_linear_vs_curved_ratio(self) -> None:
        """Curved functions should use more samples than linear."""
        linear_samples = get_samples(-10, 10, lambda x: x, scaled_transform)
        curved_samples = get_samples(-10, 10, lambda x: x * x, scaled_transform)
        
        self.assertGreater(len(curved_samples), len(linear_samples),
            f"Curved ({len(curved_samples)}) should use more samples than linear ({len(linear_samples)})")


class TestAdaptiveSamplerBenchmarks(unittest.TestCase):
    """Benchmark tests for adaptive sampling."""

    ITERATIONS = 100
    LEFT = -10.0
    RIGHT = 10.0

    def _time_adaptive(self, eval_func) -> float:
        """Time adaptive sample generation."""
        start = time.perf_counter()
        for _ in range(self.ITERATIONS):
            AdaptiveSampler.generate_samples(
                self.LEFT, self.RIGHT, eval_func, scaled_transform
            )
        return (time.perf_counter() - start) * 1000 / self.ITERATIONS

    def test_linear_benchmark(self) -> None:
        """Benchmark linear function y=x."""
        eval_func = lambda x: x
        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))
        
        print(f"\n### Linear (y=x): {adaptive_ms:.3f}ms, {adaptive_count} samples")
        self.assertLessEqual(adaptive_count, 20, "Linear should use minimal samples")

    def test_quadratic_benchmark(self) -> None:
        """Benchmark quadratic function y=x^2."""
        eval_func = lambda x: x * x
        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))
        
        print(f"\n### Quadratic (y=x^2): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_sin_benchmark(self) -> None:
        """Benchmark sin function."""
        eval_func = lambda x: math.sin(x)
        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))
        
        print(f"\n### Sin (y=sin(x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_high_amplitude_sin_benchmark(self) -> None:
        """Benchmark high amplitude sin function."""
        eval_func = lambda x: math.sin(x) * 100
        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))
        
        print(f"\n### High Amplitude Sin (y=100*sin(x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_high_frequency_sin_benchmark(self) -> None:
        """Benchmark high frequency sin(10x)."""
        eval_func = lambda x: 10 * math.sin(10 * x)
        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))
        
        print(f"\n### High Freq Sin (y=10*sin(10x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")
        self.assertGreater(adaptive_count, 20, 
            f"High frequency sin should produce >20 samples, got {adaptive_count}")


if __name__ == '__main__':
    unittest.main()

