"""
Adaptive sampler for curve rendering.

Uses recursive binary subdivision to generate sample x-values,
concentrating samples in curved regions and using fewer in straight sections.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Set, Tuple, Optional

MAX_DEPTH: int = 8
PIXEL_TOLERANCE: float = 0.5
INITIAL_SEGMENTS: int = 8
RANDOM_PROBE_COUNT: int = 10
MAX_INITIAL_SEGMENTS: int = 64
# Smallest sample budget a sub-range between asymptotes gets when the overall
# budget is shared out in proportion to sub-range width.
MIN_SUBRANGE_SAMPLES: int = 64
# Bisection steps used to locate the edge of a function's domain (e.g. x = 1
# for sqrt(1 - x^2)); 28 halvings shrink the bracket by ~2.7e8.
DOMAIN_EDGE_BISECTIONS: int = 28
# Quarter-point probes guard against symmetric oscillation fooling the
# midpoint flatness test; only applied to coarse intervals to keep it cheap.
QUARTER_PROBE_MAX_DEPTH: int = 3
# Golden-ratio stride gives well-spread, deterministic probe offsets.
_PROBE_OFFSET_STRIDE: float = 0.6180339887498949


class AdaptiveSampler:
    """
    Generates sample x-values using recursive subdivision.

    Starts with INITIAL_SEGMENTS evenly spaced points to catch periodic
    oscillations, then recursively subdivides intervals where the midpoint
    deviates from the chord by more than PIXEL_TOLERANCE pixels.
    """

    @staticmethod
    def generate_samples_with_asymptotes(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        asymptotes: List[float],
        initial_segments: Optional[int] = None,
        max_samples: Optional[int] = None,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> List[List[float]]:
        """
        Generate adaptive sample x-values, splitting at asymptotes.

        Each sub-range between asymptotes is sampled independently, ensuring
        the function is properly sampled up to the screen boundary near each asymptote.

        Args:
            left_bound: Left x boundary
            right_bound: Right x boundary
            eval_func: Function to evaluate y = f(x)
            math_to_screen: Converts (x, y) math coords to screen coords
            asymptotes: List of x-values where vertical asymptotes occur
            initial_segments: Override for initial segment count over the whole range;
                each sub-range gets its proportional share
            max_samples: Maximum number of samples over the whole range; each
                sub-range gets its proportional share (at least MIN_SUBRANGE_SAMPLES)
            viewport_band: (top, bottom) screen-y band; when given, parts of the
                curve entirely above or below it are not refined

        Returns:
            List of sample lists, one for each sub-range between asymptotes
        """
        if right_bound <= left_bound:
            return []

        # Filter asymptotes to those strictly within bounds and sort
        valid_asymptotes = sorted([a for a in asymptotes if left_bound < a < right_bound])

        # Cap number of sub-ranges to prevent performance issues with many asymptotes
        MAX_SUBRANGES = 20
        if len(valid_asymptotes) > MAX_SUBRANGES - 1:
            # Too many asymptotes - just sample without splitting
            samples, _ = AdaptiveSampler.generate_samples(
                left_bound, right_bound, eval_func, math_to_screen, initial_segments, max_samples, viewport_band
            )
            return [samples] if samples else []

        # Create sub-ranges: [left_bound, asym1], [asym1, asym2], ..., [asymN, right_bound]
        boundaries = [left_bound] + valid_asymptotes + [right_bound]
        total_width = right_bound - left_bound
        total_budget = max_samples if max_samples is not None else MAX_INITIAL_SEGMENTS * 8

        all_samples: List[List[float]] = []

        for i in range(len(boundaries) - 1):
            sub_left = boundaries[i]
            sub_right = boundaries[i + 1]

            # Skip asymptote points - use larger epsilon for numerical stability
            epsilon = 1e-6
            if i > 0:  # Not the first sub-range, so sub_left is an asymptote
                sub_left = sub_left + epsilon
            if i < len(boundaries) - 2:  # Not the last sub-range, so sub_right is an asymptote
                sub_right = sub_right - epsilon

            if sub_right <= sub_left:
                continue

            # Share the budget by width so e.g. tan(x) with many branches keeps the
            # overall sampling density instead of giving every branch the full budget.
            share = (sub_right - sub_left) / total_width
            sub_segments = None
            if initial_segments is not None:
                sub_segments = max(2, int(math.ceil(initial_segments * share)))
            sub_budget = max(MIN_SUBRANGE_SAMPLES, int(total_budget * share))

            samples, _ = AdaptiveSampler.generate_samples(
                sub_left, sub_right, eval_func, math_to_screen, sub_segments, sub_budget, viewport_band
            )

            if samples:
                all_samples.append(samples)
        return all_samples

    @staticmethod
    def generate_samples(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        initial_segments: Optional[int] = None,
        max_samples: Optional[int] = None,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[float], Optional[float]]:
        """
        Generate adaptive sample x-values for the given range.

        Args:
            left_bound: Left x boundary
            right_bound: Right x boundary
            eval_func: Function to evaluate y = f(x)
            math_to_screen: Converts (x, y) math coords to screen coords
            initial_segments: Override for initial segment count (for periodic functions)
            max_samples: Maximum number of samples (typically canvas width in pixels)
            viewport_band: (top, bottom) screen-y band; when given, parts of the
                curve entirely above or below it are not refined

        Returns:
            Tuple of (sorted list of x-values, estimated_period or None)
        """
        if right_bound <= left_bound:
            return [], None

        effective_max = max_samples if max_samples is not None else MAX_INITIAL_SEGMENTS * 8
        segments = initial_segments if initial_segments is not None else INITIAL_SEGMENTS
        segments = min(segments, effective_max)

        results = AdaptiveSampler._generate_with_segments(
            left_bound, right_bound, eval_func, math_to_screen, segments, effective_max, viewport_band
        )

        estimated_period: Optional[float] = None
        min_expected_samples = 20
        if len(results) < min_expected_samples and initial_segments is None:
            estimated_period = AdaptiveSampler._detect_periodicity(
                left_bound, right_bound, results, eval_func, math_to_screen
            )
            if estimated_period is not None:
                range_width = right_bound - left_bound
                num_periods = range_width / estimated_period
                new_segments = min(effective_max, max(INITIAL_SEGMENTS, int(num_periods * 4)))
                results = AdaptiveSampler._generate_with_segments(
                    left_bound, right_bound, eval_func, math_to_screen, new_segments, effective_max, viewport_band
                )

        return sorted(results), estimated_period

    @staticmethod
    def _generate_with_segments(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        segments: int,
        max_samples: int = 512,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> Set[float]:
        """Generate samples using specified number of initial segments."""
        results: Set[float] = set()

        initial_step = (right_bound - left_bound) / segments
        initial_points: List[Tuple[float, Optional[Tuple[float, float]]]] = []

        for i in range(segments + 1):
            x = left_bound + i * initial_step
            results.add(x)
            p = AdaptiveSampler._eval_screen_point(x, eval_func, math_to_screen)
            initial_points.append((x, p))

        for i in range(len(initial_points) - 1):
            if len(results) >= max_samples:
                break
            x_left, p_left = initial_points[i]
            x_right, p_right = initial_points[i + 1]

            if p_left is not None and p_right is not None:
                AdaptiveSampler._subdivide(
                    x_left,
                    x_right,
                    p_left,
                    p_right,
                    eval_func,
                    math_to_screen,
                    0,
                    results,
                    max_samples,
                    viewport_band,
                )
            elif p_left is not None:
                AdaptiveSampler._sample_toward_domain_edge(
                    x_left, p_left, x_right, eval_func, math_to_screen, 0, results, max_samples, viewport_band
                )
            elif p_right is not None:
                AdaptiveSampler._sample_toward_domain_edge(
                    x_right, p_right, x_left, eval_func, math_to_screen, 0, results, max_samples, viewport_band
                )

        return results

    @staticmethod
    def _detect_periodicity(
        left_bound: float,
        right_bound: float,
        current_samples: Set[float],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> Optional[float]:
        """
        Probe inside each stratum to detect missed high-frequency oscillations.

        Divides range into RANDOM_PROBE_COUNT segments and probes one point per
        segment at a deterministic, well-spread offset (golden-ratio sequence),
        so the same view always produces the same samples.

        Does NOT modify current_samples - only detects if there's periodicity.

        Returns:
            Estimated period if periodicity detected, None otherwise.
        """
        range_width = right_bound - left_bound
        segment_width = range_width / RANDOM_PROBE_COUNT

        sorted_samples = sorted(current_samples)
        deviation_count = 0

        for i in range(RANDOM_PROBE_COUNT):
            segment_left = left_bound + i * segment_width
            offset_fraction = (0.5 + i * _PROBE_OFFSET_STRIDE) % 1.0
            probe_x = segment_left + offset_fraction * segment_width

            if probe_x in current_samples:
                continue

            probe_p = AdaptiveSampler._eval_screen_point(probe_x, eval_func, math_to_screen)
            if probe_p is None:
                continue

            neighbor_left_x = None
            neighbor_right_x = None
            for x in sorted_samples:
                if x < probe_x:
                    neighbor_left_x = x
                elif x > probe_x and neighbor_right_x is None:
                    neighbor_right_x = x
                    break

            if neighbor_left_x is None or neighbor_right_x is None:
                continue

            p_left = AdaptiveSampler._eval_screen_point(neighbor_left_x, eval_func, math_to_screen)
            p_right = AdaptiveSampler._eval_screen_point(neighbor_right_x, eval_func, math_to_screen)

            if p_left is None or p_right is None:
                continue

            if not AdaptiveSampler._is_straight(p_left, probe_p, p_right):
                deviation_count += 1

        if deviation_count > 0:
            estimated_periods = deviation_count * 2
            estimated_period = range_width / estimated_periods
            return estimated_period

        return None

    @staticmethod
    def _eval_screen_point(
        x: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Evaluate function at x and convert to screen coordinates."""
        try:
            y = eval_func(x)
            if not isinstance(y, (int, float)) or not math.isfinite(y):
                return None
            return math_to_screen(x, y)
        except Exception:
            return None

    @staticmethod
    def _subdivide(
        x_left: float,
        x_right: float,
        p_left: Tuple[float, float],
        p_right: Tuple[float, float],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        depth: int,
        results: Set[float],
        max_samples: int = 512,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Recursively subdivide interval if not straight enough.

        Adds midpoint to results and recurses on both halves if the
        midpoint deviates from the chord by more than PIXEL_TOLERANCE.
        If the midpoint is outside the function's domain, samples toward the
        domain edge from both valid endpoints instead.
        """
        if depth >= MAX_DEPTH or len(results) >= max_samples:
            return

        x_mid = (x_left + x_right) / 2.0
        p_mid = AdaptiveSampler._eval_screen_point(x_mid, eval_func, math_to_screen)

        results.add(x_mid)

        if p_mid is None:
            AdaptiveSampler._sample_toward_domain_edge(
                x_left, p_left, x_mid, eval_func, math_to_screen, depth + 1, results, max_samples, viewport_band
            )
            AdaptiveSampler._sample_toward_domain_edge(
                x_right, p_right, x_mid, eval_func, math_to_screen, depth + 1, results, max_samples, viewport_band
            )
            return

        if AdaptiveSampler._is_outside_viewport(viewport_band, p_left, p_mid, p_right):
            return

        if AdaptiveSampler._is_straight(p_left, p_mid, p_right) and AdaptiveSampler._quarters_are_straight(
            x_left, x_mid, x_right, p_left, p_mid, p_right, eval_func, math_to_screen, depth
        ):
            return

        AdaptiveSampler._subdivide(
            x_left, x_mid, p_left, p_mid, eval_func, math_to_screen, depth + 1, results, max_samples, viewport_band
        )
        AdaptiveSampler._subdivide(
            x_mid, x_right, p_mid, p_right, eval_func, math_to_screen, depth + 1, results, max_samples, viewport_band
        )

    @staticmethod
    def _quarters_are_straight(
        x_left: float,
        x_mid: float,
        x_right: float,
        p_left: Tuple[float, float],
        p_mid: Tuple[float, float],
        p_right: Tuple[float, float],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        depth: int,
    ) -> bool:
        """Second flatness probe at the 1/4 and 3/4 points of coarse intervals.

        A symmetric oscillation (e.g. a full sine period) can put the midpoint
        exactly on the chord; the quarter points expose it.
        """
        if depth >= QUARTER_PROBE_MAX_DEPTH:
            return True
        p_q1 = AdaptiveSampler._eval_screen_point((x_left + x_mid) / 2.0, eval_func, math_to_screen)
        if p_q1 is None or not AdaptiveSampler._is_straight(p_left, p_q1, p_mid):
            return False
        p_q3 = AdaptiveSampler._eval_screen_point((x_mid + x_right) / 2.0, eval_func, math_to_screen)
        if p_q3 is None or not AdaptiveSampler._is_straight(p_mid, p_q3, p_right):
            return False
        return True

    @staticmethod
    def _is_outside_viewport(
        viewport_band: Optional[Tuple[float, float]],
        p_left: Tuple[float, float],
        p_mid: Tuple[float, float],
        p_right: Tuple[float, float],
    ) -> bool:
        """True when the curve over this interval stays above, or stays below, the viewport band.

        All three samples must be on the same side, and so must the extremum of the parabola
        through them; otherwise a narrow peak between the samples could reach into view.
        """
        if viewport_band is None:
            return False
        top, bottom = viewport_band
        if bottom <= top:
            return False
        ys = (p_left[1], p_mid[1], p_right[1])
        extreme = AdaptiveSampler._interpolated_extremum(*ys)
        if extreme is not None:
            ys = ys + (extreme,)
        return all(y < top for y in ys) or all(y > bottom for y in ys)

    @staticmethod
    def _interpolated_extremum(y_left: float, y_mid: float, y_right: float) -> Optional[float]:
        """Extremum of the parabola through three equally spaced samples, if inside the interval."""
        curvature = (y_left + y_right) / 2.0 - y_mid
        if curvature == 0:
            return None
        slope = (y_right - y_left) / 2.0
        t_vertex = -slope / (2.0 * curvature)
        if abs(t_vertex) > 1.0:
            return None
        return y_mid - slope * slope / (4.0 * curvature)

    @staticmethod
    def _sample_toward_domain_edge(
        x_valid: float,
        p_valid: Tuple[float, float],
        x_invalid: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        depth: int,
        results: Set[float],
        max_samples: int = 512,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Locate the domain edge between a valid and an invalid x and sample up to it.

        Bisects toward the invalid side, adds the last valid x found (the curve's
        true end, e.g. x = 1 for sqrt(1 - x^2)) and refines the stretch between it
        and the valid endpoint.
        """
        if len(results) >= max_samples:
            return
        edge_x = x_valid
        edge_p: Optional[Tuple[float, float]] = None
        bad_x = x_invalid
        for _ in range(DOMAIN_EDGE_BISECTIONS):
            probe_x = (edge_x + bad_x) / 2.0
            probe_p = AdaptiveSampler._eval_screen_point(probe_x, eval_func, math_to_screen)
            if probe_p is None:
                bad_x = probe_x
            else:
                edge_x, edge_p = probe_x, probe_p
        if edge_p is None:
            return
        results.add(edge_x)
        if x_valid < edge_x:
            AdaptiveSampler._subdivide(
                x_valid,
                edge_x,
                p_valid,
                edge_p,
                eval_func,
                math_to_screen,
                depth,
                results,
                max_samples,
                viewport_band,
            )
        else:
            AdaptiveSampler._subdivide(
                edge_x,
                x_valid,
                edge_p,
                p_valid,
                eval_func,
                math_to_screen,
                depth,
                results,
                max_samples,
                viewport_band,
            )

    @staticmethod
    def _is_straight(
        p_left: Tuple[float, float],
        p_mid: Tuple[float, float],
        p_right: Tuple[float, float],
    ) -> bool:
        """
        Check if midpoint deviation from chord is within tolerance.

        Calculates perpendicular distance from p_mid to the line segment
        from p_left to p_right. Returns True if distance < PIXEL_TOLERANCE.
        """
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-12:
            return True

        cross = abs((p_mid[0] - p_left[0]) * dy - (p_mid[1] - p_left[1]) * dx)
        distance = cross / math.sqrt(length_sq)

        return distance < PIXEL_TOLERANCE
