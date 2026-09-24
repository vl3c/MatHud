"""
Server tests for AdaptiveSampler.

Tests the recursive subdivision algorithm for generating adaptive sample points.
"""

import math
import sys
import time
import unittest
from typing import Tuple

sys.path.insert(0, "static/client/rendering/renderables")

from adaptive_sampler import (
    AdaptiveSampler,
    MAX_INITIAL_SEGMENTS,
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
            self.assertAlmostEqual(samples[0], left, places=10, msg=f"First sample {samples[0]} != left bound {left}")
            self.assertAlmostEqual(
                samples[-1], right, places=10, msg=f"Last sample {samples[-1]} != right bound {right}"
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
        """Should not exceed the default sample cap even for dense oscillation."""
        max_possible = MAX_INITIAL_SEGMENTS * 8
        samples = get_samples(-10, 10, lambda x: math.sin(100 * x), scaled_transform)
        self.assertLessEqual(len(samples), max_possible)


class TestAdaptiveSamplerInvalidValues(unittest.TestCase):
    """Tests for handling invalid function values."""

    def test_handles_nan(self) -> None:
        """Should handle NaN values gracefully."""

        def func_with_nan(x: float) -> float:
            if x == 0:
                return float("nan")
            return x

        samples = get_samples(-10, 10, func_with_nan, identity_transform)
        self.assertIn(-10, samples)
        self.assertIn(10, samples)

    def test_handles_inf(self) -> None:
        """Should handle infinity values gracefully."""

        def func_with_inf(x: float) -> float:
            if abs(x) < 0.01:
                return float("inf")
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


def _safe_sqrt(value: float) -> float:
    return math.sqrt(value) if value >= 0 else float("nan")


def _valid_samples(samples, func) -> list:
    valid = []
    for x in samples:
        try:
            y = func(x)
        except Exception:
            continue
        if isinstance(y, float) and math.isfinite(y):
            valid.append(x)
    return valid


class TestAdaptiveSamplerDomainEdges(unittest.TestCase):
    """Samples should reach the edges of a restricted domain."""

    def test_semicircle_is_sampled_in_default_view(self) -> None:
        def func(x: float) -> float:
            return _safe_sqrt(1 - x * x)

        samples = get_samples(-4.7, 4.7, func, scaled_transform)
        valid = _valid_samples(samples, func)
        self.assertGreater(len(valid), 5)
        self.assertLess(min(valid), -0.999)
        self.assertGreater(max(valid), 0.999)

    def test_sqrt_starts_at_domain_edge(self) -> None:
        samples = get_samples(-4.7, 4.7, _safe_sqrt, scaled_transform)
        valid = _valid_samples(samples, _safe_sqrt)
        self.assertLess(min(valid), 1e-4)

    def test_hole_inside_interval_is_bounded_on_both_sides(self) -> None:
        def func(x: float) -> float:
            return _safe_sqrt(x * x - 1)

        samples = get_samples(-3, 3, func, scaled_transform)
        valid = _valid_samples(samples, func)
        inner_left = max(x for x in valid if x < 0)
        inner_right = min(x for x in valid if x > 0)
        self.assertGreater(inner_left, -1.001)
        self.assertLess(inner_right, 1.001)


class TestAdaptiveSamplerViewportCulling(unittest.TestCase):
    """Refinement should not be spent on curve parts outside the viewport."""

    def test_offscreen_parabola_is_not_refined(self) -> None:
        # Visible y-range is [-7.5, 7.5], so |x| > ~2.8 lies above the viewport.
        samples = get_samples(-50, 50, lambda x: x * x, scaled_transform, viewport_band=(0, 480))
        offscreen = [x for x in samples if abs(x) > 5]
        onscreen = [x for x in samples if abs(x) < 2.7]
        self.assertLessEqual(len(offscreen), 20)
        self.assertGreater(len(onscreen), 10)

    def test_narrow_peak_rising_into_view_is_sampled(self) -> None:
        # Base of each parabola is far below the viewport; only a narrow tip reaches y = 3.
        for center in (1.1, 0.37, 5.05):
            for steepness in (1000, 3000):

                def func(x: float, k: float = steepness, c: float = center) -> float:
                    return -k * (x - c) ** 2 + 3

                samples = get_samples(-20, 20, func, scaled_transform, viewport_band=(0, 480))
                visible = [x for x in samples if func(x) > -7.5]
                self.assertGreaterEqual(len(visible), 5, (center, steepness))
                self.assertGreater(max(func(x) for x in samples), 2.5, (center, steepness))


class TestAdaptiveSamplerProbes(unittest.TestCase):
    """Oscillation detection should be deterministic and robust."""

    def test_periodicity_probe_is_deterministic(self) -> None:
        import random

        results = []
        for seed in range(10):
            random.seed(seed)
            results.append(get_samples(-8, 8, lambda x: math.sin(math.pi * x), scaled_transform))
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_symmetric_oscillation_under_flat_midpoint_is_refined(self) -> None:
        # Initial nodes (step 8) and interval midpoints of [0, 32] fall on zeros of
        # sin(pi*x/4) (one full period per interval); the parabola on the left
        # keeps the periodicity probe off.
        def func(x: float) -> float:
            return x * x if x < 0 else 3 * math.sin(math.pi * x / 4)

        samples = get_samples(-32, 32, func, scaled_transform)
        positive = [x for x in samples if 0 < x < 32]
        self.assertGreater(len(positive), 16)


class TestAdaptiveSamplerSubrangeBudget(unittest.TestCase):
    """Sub-ranges between asymptotes share the sample budget instead of each getting all of it."""

    def test_tan_total_samples_stay_within_budget(self) -> None:
        def tan_value(x: float) -> float:
            return math.tan(x)

        asymptotes = [math.pi / 2 + k * math.pi for k in range(-8, 8)]
        budget = 1000
        subranges = AdaptiveSampler.generate_samples_with_asymptotes(
            -20, 20, tan_value, scaled_transform, asymptotes, initial_segments=51, max_samples=budget
        )

        self.assertGreater(len(subranges), 10)
        total = sum(len(samples) for samples in subranges)
        self.assertLessEqual(total, budget * 1.1)
        for samples in subranges:
            self.assertGreaterEqual(len(samples), 3)


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

        self.assertGreater(
            len(curved_samples),
            len(linear_samples),
            f"Curved ({len(curved_samples)}) should use more samples than linear ({len(linear_samples)})",
        )


class TestAdaptiveSamplerBenchmarks(unittest.TestCase):
    """Benchmark tests for adaptive sampling."""

    ITERATIONS = 100
    LEFT = -10.0
    RIGHT = 10.0

    def _time_adaptive(self, eval_func) -> float:
        """Time adaptive sample generation."""
        start = time.perf_counter()
        for _ in range(self.ITERATIONS):
            AdaptiveSampler.generate_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform)
        return (time.perf_counter() - start) * 1000 / self.ITERATIONS

    def test_linear_benchmark(self) -> None:
        """Benchmark linear function y=x."""

        def eval_func(x):
            return x

        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))

        print(f"\n### Linear (y=x): {adaptive_ms:.3f}ms, {adaptive_count} samples")
        self.assertLessEqual(adaptive_count, 20, "Linear should use minimal samples")

    def test_quadratic_benchmark(self) -> None:
        """Benchmark quadratic function y=x^2."""

        def eval_func(x):
            return x * x

        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))

        print(f"\n### Quadratic (y=x^2): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_sin_benchmark(self) -> None:
        """Benchmark sin function."""

        def eval_func(x):
            return math.sin(x)

        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))

        print(f"\n### Sin (y=sin(x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_high_amplitude_sin_benchmark(self) -> None:
        """Benchmark high amplitude sin function."""

        def eval_func(x):
            return math.sin(x) * 100

        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))

        print(f"\n### High Amplitude Sin (y=100*sin(x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")

    def test_high_frequency_sin_benchmark(self) -> None:
        """Benchmark high frequency sin(10x)."""

        def eval_func(x):
            return 10 * math.sin(10 * x)

        adaptive_ms = self._time_adaptive(eval_func)
        adaptive_count = len(get_samples(self.LEFT, self.RIGHT, eval_func, scaled_transform))

        print(f"\n### High Freq Sin (y=10*sin(10x)): {adaptive_ms:.3f}ms, {adaptive_count} samples")
        self.assertGreater(adaptive_count, 20, f"High frequency sin should produce >20 samples, got {adaptive_count}")


if __name__ == "__main__":
    unittest.main()
