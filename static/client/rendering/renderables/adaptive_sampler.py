"""
Adaptive sampler for curve rendering.

Uses recursive binary subdivision to generate sample x-values,
concentrating samples in curved regions and using fewer in straight sections.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

MAX_DEPTH: int = 8
PIXEL_TOLERANCE: float = 0.5
INITIAL_SEGMENTS: int = 8
RANDOM_PROBE_COUNT: int = 10
MAX_INITIAL_SEGMENTS: int = 64
# With a pixel budget (max_samples) the initial grid gets one segment per this many
# pixels (bounded by INITIAL_SEGMENTS and MAX_INITIAL_SEGMENTS), so a narrow spike or
# dip between two grid nodes is less likely to be stepped over as a straight line.
PIXELS_PER_INITIAL_SEGMENT: int = 16
# Oscillation probes run when refinement added fewer samples than this to the
# initial grid (an apparently flat curve may be aliased by the grid).
MIN_EXPECTED_REFINEMENTS: int = 11
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
# Split points (asymptotes and discontinuities) are honoured as sub-ranges while there is
# at least this much budget per split; beyond, the range is sampled as one (the path
# builder still breaks at listed asymptotes, and the jump test runs).
SAMPLES_PER_SPLIT: int = 8
# Fewest samples a sub-range between asymptotes gets when many share the budget.
MIN_BRANCH_SAMPLES: int = 4
# Jump test between consecutive samples: a pair whose screen y differ by more than
# JUMP_SUSPECT_PX, with a slope more than JUMP_NEIGHBOUR_RATIO times its neighbours',
# is bisected towards its larger half (JUMP_BISECTIONS halvings at most). It is
# continuous once the gap is under JUMP_CONTINUOUS_PX; a break (a jump, a pole or an
# undefined point) once the bracket is narrower than JUMP_NARROW_PX and the gap has
# shrunk by less than JUMP_SHRINK_RATIO over the last JUMP_STEADY_HALVINGS halvings. A jump
# tends to ratio 1 and a pole's gap grows; a vertical tangent of x^(1/n) shrinks by
# 2^(-8/n) over 8 halvings (0.33 for n = 5, 0.89 for n = 50), so it stays continuous.
JUMP_SUSPECT_PX: float = 8.0
JUMP_NEIGHBOUR_RATIO: float = 2.0
JUMP_CONTINUOUS_PX: float = 1.0
JUMP_BISECTIONS: int = 40
JUMP_NARROW_PX: float = 1e-3
JUMP_SHRINK_RATIO: float = 0.9
JUMP_STEADY_HALVINGS: int = 8
# ... and continuous as soon as the gap has shrunk to at most JUMP_SMOOTH_RATIO of itself for
# JUMP_SMOOTH_HALVINGS halvings in a row (a jump J keeps the gap above J, so this needs the
# smooth part to be over 4 J; after the halvings J is under 1/32 of the pair's step).
JUMP_SMOOTH_RATIO: float = 0.6
JUMP_SMOOTH_HALVINGS: int = 3
# Pairs tested per sampled range at most, largest steps first.
MAX_JUMP_TESTS: int = 200
# Samples allowed beyond the budget beside each break found.
BREAK_SAMPLES: int = 16
# Intervals wider than this (screen px) also get a probe at the golden-ratio point.
GOLDEN_PROBE_MIN_PX: float = 2.0


# A queued interval: (x_left, x_right, screen point left, screen point right, depth, the
# deviation of its parent's midpoint from the chord in px, used to take the worst first)
_Interval = Tuple[float, float, Tuple[float, float], Tuple[float, float], int, float]
# The samples of a range: x -> its screen point (None where the function is undefined)
_Samples = Dict[float, Optional[Tuple[float, float]]]


class AdaptiveSampler:
    """
    Generates sample x-values using recursive subdivision.

    Starts with evenly spaced points (INITIAL_SEGMENTS, or one segment per
    PIXELS_PER_INITIAL_SEGMENT pixels of the sample budget) to catch periodic
    oscillations and narrow features, then recursively subdivides intervals where
    the midpoint deviates from the chord by more than PIXEL_TOLERANCE pixels.
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
        breaks: Optional[List[Tuple[float, float]]] = None,
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
                sub-range gets its proportional share (at least MIN_SUBRANGE_SAMPLES, or
                less when many sub-ranges share the budget)
            viewport_band: (top, bottom) screen-y band; when given, parts of the
                curve entirely above or below it are not refined
            breaks: When given, the jump test runs and each break found is appended as
                (x_left, x_right): consecutive samples the curve must not be drawn between

        Returns:
            List of sample lists, one for each sub-range between asymptotes
        """
        if right_bound <= left_bound:
            return []

        # Filter asymptotes to those strictly within bounds and sort
        valid_asymptotes = sorted([a for a in asymptotes if left_bound < a < right_bound])

        total_budget = max_samples if max_samples is not None else MAX_INITIAL_SEGMENTS * 8
        if len(valid_asymptotes) > total_budget // SAMPLES_PER_SPLIT:
            samples, _ = AdaptiveSampler.generate_samples(
                left_bound, right_bound, eval_func, math_to_screen, initial_segments, max_samples, viewport_band, breaks
            )
            return [samples] if samples else []

        # Create sub-ranges: [left_bound, asym1], [asym1, asym2], ..., [asymN, right_bound]
        boundaries = [left_bound] + valid_asymptotes + [right_bound]
        total_width = right_bound - left_bound
        subrange_count = len(boundaries) - 1
        # Many branches (tan(10x) has 64 in a 20-unit view) share the budget; each keeps at
        # least a few samples so it reaches out towards both of its asymptotes.
        min_samples = max(MIN_BRANCH_SAMPLES, min(MIN_SUBRANGE_SAMPLES, total_budget // subrange_count))

        all_samples: List[List[float]] = []

        for i in range(subrange_count):
            sub_left = boundaries[i]
            sub_right = boundaries[i + 1]

            # Step off the asymptotes by a tiny fraction of the sub-range (at least a few
            # float steps of x), so a branch is sampled as close to its poles as the view needs
            epsilon = AdaptiveSampler._asymptote_offset(sub_left, sub_right)
            if i > 0:  # Not the first sub-range, so sub_left is an asymptote
                sub_left = sub_left + epsilon
            if i < subrange_count - 1:  # Not the last sub-range, so sub_right is an asymptote
                sub_right = sub_right - epsilon

            if sub_right <= sub_left:
                continue

            # Share the budget by width so e.g. tan(x) with many branches keeps the
            # overall sampling density instead of giving every branch the full budget.
            share = (sub_right - sub_left) / total_width
            sub_segments = None
            sub_budget = max(min_samples, int(total_budget * share))
            if initial_segments is not None:
                sub_segments = max(2, int(math.ceil(initial_segments * share)))
            elif subrange_count > 1:
                # Sub-ranges get a grid of at most half their budget and no periodicity probe
                # (with many narrow branches its probes and regeneration cost more than a
                # branch's whole budget; the golden-ratio probe catches aliasing instead)
                sub_segments = max(2, min(AdaptiveSampler.default_initial_segments(sub_budget), sub_budget // 2))

            samples, _ = AdaptiveSampler.generate_samples(
                sub_left, sub_right, eval_func, math_to_screen, sub_segments, sub_budget, viewport_band, breaks
            )

            if samples:
                all_samples.append(samples)
        return all_samples

    @staticmethod
    def _asymptote_offset(sub_left: float, sub_right: float) -> float:
        """Distance kept from an asymptote at a sub-range end: 1e-9 of the sub-range, at least
        a few float steps of the larger |x| (1e-6 absolute was a visible gap when zoomed in,
        and below the float spacing of x far from 0)."""
        scale = max(abs(sub_left), abs(sub_right))
        return max((sub_right - sub_left) * 1e-9, scale * 1e-15, 1e-300)

    @staticmethod
    def generate_samples(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        initial_segments: Optional[int] = None,
        max_samples: Optional[int] = None,
        viewport_band: Optional[Tuple[float, float]] = None,
        breaks: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[float], Optional[float]]:
        """
        Generate adaptive sample x-values for the given range.

        Args:
            left_bound: Left x boundary
            right_bound: Right x boundary
            eval_func: Function to evaluate y = f(x)
            math_to_screen: Converts (x, y) math coords to screen coords
            initial_segments: Override for initial segment count (for periodic functions);
                by default it follows max_samples (see default_initial_segments)
            max_samples: Maximum number of samples (typically canvas width in pixels)
            viewport_band: (top, bottom) screen-y band; when given, parts of the
                curve entirely above or below it are not refined
            breaks: When given, the jump test runs (see find_breaks) and each break found
                is appended as (x_left, x_right), both of them samples

        Returns:
            Tuple of (sorted list of x-values, estimated_period or None)
        """
        if right_bound <= left_bound:
            return [], None

        effective_max = max_samples if max_samples is not None else MAX_INITIAL_SEGMENTS * 8
        segments = (
            initial_segments if initial_segments is not None else AdaptiveSampler.default_initial_segments(max_samples)
        )
        segments = max(1, min(segments, effective_max))

        results = AdaptiveSampler._generate_with_segments(
            left_bound, right_bound, eval_func, math_to_screen, segments, effective_max, viewport_band
        )

        estimated_period: Optional[float] = None
        refinements = len(results) - (segments + 1)
        if refinements < MIN_EXPECTED_REFINEMENTS and initial_segments is None:
            estimated_period = AdaptiveSampler._detect_periodicity(
                left_bound, right_bound, results, eval_func, math_to_screen
            )
            if estimated_period is not None:
                range_width = right_bound - left_bound
                num_periods = range_width / estimated_period
                new_segments = min(effective_max, max(segments, int(num_periods * 4)))
                results = AdaptiveSampler._generate_with_segments(
                    left_bound, right_bound, eval_func, math_to_screen, new_segments, effective_max, viewport_band
                )

        xs = sorted(results)
        if breaks is not None:
            found = AdaptiveSampler.find_breaks(results, eval_func, math_to_screen, effective_max, viewport_band, xs)
            if found:
                breaks.extend(found)
                # Nearly sorted already: the samples beside the breaks were added at the end
                xs = sorted(results)

        return xs, estimated_period

    @staticmethod
    def default_initial_segments(max_samples: Optional[int]) -> int:
        """Initial grid size for a sample budget of max_samples (about one per pixel).

        One segment per PIXELS_PER_INITIAL_SEGMENT pixels, between INITIAL_SEGMENTS and
        MAX_INITIAL_SEGMENTS; INITIAL_SEGMENTS without a budget.
        """
        if max_samples is None:
            return INITIAL_SEGMENTS
        return max(INITIAL_SEGMENTS, min(MAX_INITIAL_SEGMENTS, int(max_samples) // PIXELS_PER_INITIAL_SEGMENT))

    @staticmethod
    def _generate_with_segments(
        left_bound: float,
        right_bound: float,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        segments: int,
        max_samples: int = 512,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> _Samples:
        """Generate samples using specified number of initial segments.

        The intervals are refined breadth first (every interval of one depth before any of
        the next), so when the budget runs out the whole range has been refined alike; depth
        first, the left of a fast oscillation took the budget and the right was drawn as a
        slower false wave.
        """
        results: _Samples = {}

        initial_step = (right_bound - left_bound) / segments
        initial_points: List[Tuple[float, Optional[Tuple[float, float]]]] = []

        for i in range(segments + 1):
            x = left_bound + i * initial_step
            p = AdaptiveSampler._eval_screen_point(x, eval_func, math_to_screen)
            results[x] = p
            initial_points.append((x, p))

        queue: List[_Interval] = []
        for i in range(len(initial_points) - 1):
            x_left, p_left = initial_points[i]
            x_right, p_right = initial_points[i + 1]

            if p_left is not None and p_right is not None:
                queue.append((x_left, x_right, p_left, p_right, 0, math.inf))
            elif p_left is not None:
                AdaptiveSampler._sample_toward_domain_edge(
                    x_left, p_left, x_right, eval_func, math_to_screen, 0, results, queue
                )
            elif p_right is not None:
                AdaptiveSampler._sample_toward_domain_edge(
                    x_right, p_right, x_left, eval_func, math_to_screen, 0, results, queue
                )

        AdaptiveSampler._refine(queue, eval_func, math_to_screen, results, max_samples, viewport_band)
        return results

    @staticmethod
    def _refine(
        queue: List[_Interval],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        results: _Samples,
        max_samples: int,
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Refine the queued intervals breadth first until straight, at MAX_DEPTH or out of budget.

        An interval whose midpoint deviates from the chord by more than PIXEL_TOLERANCE (or
        whose quarter or golden-ratio probes do, while coarse) is split in two, its halves
        joining the next level. If the midpoint is outside the function's domain, each valid
        end is sampled toward the domain edge instead. A level that does not fit in the budget
        left is taken worst first (see _worst_first), so the budget's end is not spent left to
        right. Levels are plain lists: Brython's deque is many times slower.
        """
        level = queue
        while level and len(results) < max_samples:
            if len(results) + len(level) > max_samples:
                level = AdaptiveSampler._worst_first(level, max_samples - len(results))
            next_level: List[_Interval] = []
            for x_left, x_right, p_left, p_right, depth, _ in level:
                if len(results) >= max_samples:
                    break
                if depth >= MAX_DEPTH:
                    continue
                AdaptiveSampler._split_if_curved(
                    x_left,
                    x_right,
                    p_left,
                    p_right,
                    depth,
                    eval_func,
                    math_to_screen,
                    results,
                    next_level,
                    viewport_band,
                )
            level = next_level

    @staticmethod
    def _split_if_curved(
        x_left: float,
        x_right: float,
        p_left: Tuple[float, float],
        p_right: Tuple[float, float],
        depth: int,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        results: _Samples,
        next_level: List[_Interval],
        viewport_band: Optional[Tuple[float, float]],
    ) -> None:
        """Sample the interval's midpoint and queue both halves on next_level unless it is straight."""
        x_mid = (x_left + x_right) / 2.0
        p_mid = AdaptiveSampler._eval_screen_point(x_mid, eval_func, math_to_screen)
        results[x_mid] = p_mid

        if p_mid is None:
            AdaptiveSampler._sample_toward_domain_edge(
                x_left, p_left, x_mid, eval_func, math_to_screen, depth + 1, results, next_level
            )
            AdaptiveSampler._sample_toward_domain_edge(
                x_right, p_right, x_mid, eval_func, math_to_screen, depth + 1, results, next_level
            )
            return

        if AdaptiveSampler._is_outside_viewport(viewport_band, p_left, p_mid, p_right):
            return

        deviation = AdaptiveSampler._deviation(p_left, p_mid, p_right)
        # The probes' cheap guards (depth, width) are tested here to spare Brython the calls
        if (
            deviation < PIXEL_TOLERANCE
            and (
                depth >= QUARTER_PROBE_MAX_DEPTH
                or AdaptiveSampler._quarters_are_straight(
                    x_left, x_mid, x_right, p_left, p_mid, p_right, eval_func, math_to_screen, depth
                )
            )
            and (
                abs(p_right[0] - p_left[0]) <= GOLDEN_PROBE_MIN_PX
                or AdaptiveSampler._golden_probe_is_straight(
                    x_left, x_mid, x_right, p_left, p_mid, p_right, eval_func, math_to_screen
                )
            )
        ):
            return

        next_level.append((x_left, x_mid, p_left, p_mid, depth + 1, deviation))
        next_level.append((x_mid, x_right, p_mid, p_right, depth + 1, deviation))

    @staticmethod
    def _worst_first(level: List[_Interval], room: int) -> List[_Interval]:
        """The intervals of a level to refine with room samples left, worst first.

        Intervals are grouped by their parent's deviation within a factor of two (no sort:
        Brython's is slow) and taken from the worst group down; the group that does not fit
        is thinned evenly across it, so equally bad intervals are not taken left to right.
        """
        groups: Dict[int, List[_Interval]] = {}
        for item in level:
            deviation = item[5]
            if deviation == math.inf:
                key = 1 << 20
            elif deviation > 0:
                key = math.frexp(deviation)[1]
            else:
                key = -(1 << 20)
            group = groups.get(key)
            if group is None:
                groups[key] = [item]
            else:
                group.append(item)
        chosen: List[_Interval] = []
        for key in sorted(groups, reverse=True):
            group = groups[key]
            left = room - len(chosen)
            if left <= 0:
                break
            if len(group) <= left:
                chosen.extend(group)
            else:
                count = len(group)
                chosen.extend(group[(j * count) // left] for j in range(left))
        return chosen

    @staticmethod
    def _golden_probe_is_straight(
        x_left: float,
        x_mid: float,
        x_right: float,
        p_left: Tuple[float, float],
        p_mid: Tuple[float, float],
        p_right: Tuple[float, float],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
    ) -> bool:
        """A probe at the golden-ratio point of intervals wider than GOLDEN_PROBE_MIN_PX.

        Midpoints and quarter points are dyadic: on a grid whose step is close to a multiple
        of an oscillation's period (sin(200x) on the periodic grid) they all lie on a slow
        false wave, and a symmetric oscillation (sin(1/x)) can put them all on the chord. The
        golden-ratio point is not dyadic, so it lands elsewhere in the oscillation.
        """
        if abs(p_right[0] - p_left[0]) <= GOLDEN_PROBE_MIN_PX:
            return True
        x_probe = x_left + (x_right - x_left) * _PROBE_OFFSET_STRIDE
        p_probe = AdaptiveSampler._eval_screen_point(x_probe, eval_func, math_to_screen)
        if p_probe is None:
            return False
        return AdaptiveSampler._is_straight(p_mid, p_probe, p_right)

    @staticmethod
    def _detect_periodicity(
        left_bound: float,
        right_bound: float,
        current_samples: _Samples,
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
        results: _Samples,
        queue: List[_Interval],
    ) -> None:
        """Locate the domain edge between a valid and an invalid x and queue the stretch up to it.

        Bisects toward the invalid side, adds the last valid x found (the curve's
        true end, e.g. x = 1 for sqrt(1 - x^2)) and queues the stretch between it
        and the valid endpoint for refinement.
        """
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
        results[edge_x] = edge_p
        if x_valid < edge_x:
            queue.append((x_valid, edge_x, p_valid, edge_p, depth, math.inf))
        else:
            queue.append((edge_x, x_valid, edge_p, p_valid, depth, math.inf))

    @staticmethod
    def find_breaks(
        results: _Samples,
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        max_samples: int,
        viewport_band: Optional[Tuple[float, float]] = None,
        sorted_xs: Optional[List[float]] = None,
    ) -> List[Tuple[float, float]]:
        """Find where the curve breaks between consecutive samples (jumps and poles the
        expression analysis did not list: floor, round, mod, sign, ...).

        A pair is suspect when its screen-y step exceeds JUMP_SUSPECT_PX and it stands out from
        its neighbours (see _stands_out: not a steep stretch, the approach to a pole, or a fast
        oscillation), has an end beyond the viewport band, or steps against both neighbours'
        direction (a pole between two samples, see _against_neighbours), unless both samples
        are above, or both below, the band, or it leaves the band the way the curve already heads.
        Each suspect (the MAX_JUMP_TESTS largest) is bisected toward its larger half (see
        _locate_break): a break is a gap that stops shrinking once the bracket is far below a
        pixel, or a probe where the function is undefined. The bracket ends are added to results and
        the stretches beside them refined, so a branch is drawn up to the break. sorted_xs, when
        given, is sorted(results) already.

        Returns:
            The breaks as (x_left, x_right), consecutive samples once added
        """
        xs = sorted_xs if sorted_xs is not None else sorted(results)
        points = [results[x] for x in xs]
        # Signed screen-y step of each consecutive pair and its slope's size (step per screen-x
        # pixel); None and -1 where a sample is undefined or both are beyond the band on one
        # side (never drawn, so neither a suspect nor a neighbour)
        steps: List[Optional[float]] = []
        mags: List[float] = []
        top, bottom = viewport_band if viewport_band is not None else (-math.inf, math.inf)
        for i in range(len(xs) - 1):
            p_a, p_b = points[i], points[i + 1]
            if p_a is None or p_b is None or (p_a[1] < top and p_b[1] < top) or (p_a[1] > bottom and p_b[1] > bottom):
                steps.append(None)
                mags.append(-1.0)
            else:
                step = p_b[1] - p_a[1]
                steps.append(step)
                mags.append(abs(step) / max(abs(p_b[0] - p_a[0]), 1e-12))

        suspects: List[Tuple[float, int]] = []
        for i, step in enumerate(steps):
            # The size test first, inline: most pairs are small steps
            if step is None or -JUMP_SUSPECT_PX <= step <= JUMP_SUSPECT_PX:
                continue
            # A pair with an end beyond the band is tested whatever its neighbours: next to a pole
            # its neighbour is as steep, and drawn it would be a stroke across the view
            crosses = AdaptiveSampler._has_end_beyond(top, bottom, points[i], points[i + 1])
            # So is a step against both neighbours' direction: the curve runs off one way and
            # comes back from the other, through a pole between the two samples
            flips = AdaptiveSampler._against_neighbours(steps, i)
            if (crosses or flips or AdaptiveSampler._stands_out(steps, mags, i)) and not (
                AdaptiveSampler._leaves_band_heading_on(steps, points, i, viewport_band)
            ):
                suspects.append((abs(step), i))
        suspects.sort(reverse=True)

        breaks: List[Tuple[float, float]] = []
        queue: List[_Interval] = []
        for _, i in suspects[:MAX_JUMP_TESTS]:
            bracket = AdaptiveSampler._locate_break(
                xs[i], points[i], xs[i + 1], points[i + 1], eval_func, math_to_screen, viewport_band
            )
            if bracket is None:
                continue
            (x_l, p_l), (x_r, p_r) = bracket
            breaks.append((x_l, x_r))
            results[x_l] = p_l
            results[x_r] = p_r
            p_a, p_b = points[i], points[i + 1]
            if p_a is not None and p_l is not None and xs[i] < x_l:
                queue.append((xs[i], x_l, p_a, p_l, 0, math.inf))
            if p_b is not None and p_r is not None and x_r < xs[i + 1]:
                queue.append((x_r, xs[i + 1], p_r, p_b, 0, math.inf))
        # The stretches beside each break get BREAK_SAMPLES beyond the budget, so a branch is
        # drawn out to its pole even when the budget is spent
        budget = max(len(results), max_samples) + BREAK_SAMPLES * len(breaks)
        AdaptiveSampler._refine(queue, eval_func, math_to_screen, results, budget, viewport_band)
        return breaks

    @staticmethod
    def _has_end_beyond(
        top: float, bottom: float, p_a: Optional[Tuple[float, float]], p_b: Optional[Tuple[float, float]]
    ) -> bool:
        """True when one of the two points (not both on one side) is above or below the band."""
        if p_a is None or p_b is None:
            return False
        return not (top <= p_a[1] <= bottom and top <= p_b[1] <= bottom)

    @staticmethod
    def _against_neighbours(steps: List[Optional[float]], i: int) -> bool:
        """True when step i goes the other way from the drawn steps on both sides of it, and the
        curve keeps its direction for two steps on at least one side: a branch running into a pole
        and the next coming out of it, not a zigzag (an oscillation faster than the samples)."""
        if i == 0 or i + 1 >= len(steps):
            return False
        step, before, after = steps[i], steps[i - 1], steps[i + 1]
        if step is None or before is None or after is None:
            return False
        rising = step > 0
        if (before > 0) == rising or (after > 0) == rising:
            return False
        earlier = steps[i - 2] if i >= 2 else None
        later = steps[i + 2] if i + 2 < len(steps) else None
        return (earlier is not None and (earlier > 0) != rising) or (later is not None and (later > 0) != rising)

    @staticmethod
    def _stands_out(steps: List[Optional[float]], mags: List[float], i: int) -> bool:
        """Whether the slope of pair i (a step over JUMP_SUSPECT_PX) stands out from its neighbours'.

        It does when it exceeds JUMP_NEIGHBOUR_RATIO times each neighbouring pair's slope, or
        when it and a neighbour of about its slope in the same direction exceed their outer
        neighbours that way (a sample on a jump with a value between its sides, as sign(0) = 0).
        Slopes, not steps, since neighbouring pairs can differ in width. Steep stretches, the
        approach to a pole and fast oscillations have neighbours of their slope. A missing or
        undrawn neighbour (mag -1) does not count against it. Written out inline: it runs for
        every large step, and Brython function calls are slow.
        """
        last = len(mags) - 1
        mag = mags[i]
        left = mags[i - 1] if i > 0 else -1.0
        right = mags[i + 1] if i < last else -1.0
        if left * JUMP_NEIGHBOUR_RATIO < mag and right * JUMP_NEIGHBOUR_RATIO < mag:
            return True
        positive = cast(float, steps[i]) > 0
        for j, other in ((i - 1, left), (i + 1, right)):
            other_step = steps[j] if 0 <= j <= last else None
            if other_step is None or (other_step > 0) != positive:
                continue
            low, high = (other, mag) if other < mag else (mag, other)
            if high > JUMP_NEIGHBOUR_RATIO * low:
                continue
            first, final = (j, i) if j < i else (i, j)
            outer_left = mags[first - 1] if first > 0 else -1.0
            outer_right = mags[final + 1] if final < last else -1.0
            if outer_left * JUMP_NEIGHBOUR_RATIO < low and outer_right * JUMP_NEIGHBOUR_RATIO < low:
                return True
        return False

    @staticmethod
    def _leaves_band_heading_on(
        steps: List[Optional[float]],
        points: List[Optional[Tuple[float, float]]],
        i: int,
        viewport_band: Optional[Tuple[float, float]],
    ) -> bool:
        """True when pair i has one sample in the band and one beyond it, and the step on the
        in-band side goes the same way (the curve heads off screen, as near a pole outside the pair)."""
        if viewport_band is None:
            return False
        p_a, p_b = points[i], points[i + 1]
        step = steps[i]
        if p_a is None or p_b is None or step is None:
            return False
        a_in = AdaptiveSampler._in_band(viewport_band, p_a)
        b_in = AdaptiveSampler._in_band(viewport_band, p_b)
        if a_in == b_in:
            return False
        inner = steps[i - 1] if a_in and i > 0 else (steps[i + 1] if b_in and i + 1 < len(steps) else None)
        return inner is not None and inner != 0 and (inner > 0) == (step > 0)

    @staticmethod
    def _in_band(viewport_band: Tuple[float, float], p: Tuple[float, float]) -> bool:
        return viewport_band[0] <= p[1] <= viewport_band[1]

    @staticmethod
    def _both_beyond_band(
        viewport_band: Optional[Tuple[float, float]],
        p_a: Optional[Tuple[float, float]],
        p_b: Optional[Tuple[float, float]],
    ) -> bool:
        """True when both points are above the band, or both below it (nothing is drawn between)."""
        if viewport_band is None or p_a is None or p_b is None:
            return False
        top, bottom = viewport_band
        return (p_a[1] < top and p_b[1] < top) or (p_a[1] > bottom and p_b[1] > bottom)

    @staticmethod
    def _locate_break(
        x_a: float,
        p_a: Optional[Tuple[float, float]],
        x_b: float,
        p_b: Optional[Tuple[float, float]],
        eval_func: Callable[[float], Any],
        math_to_screen: Callable[[float, float], Tuple[float, float]],
        viewport_band: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[float, Optional[Tuple[float, float]]], Tuple[float, Optional[Tuple[float, float]]]]]:
        """Bisect a suspect pair toward its larger step: the bracket of the break, or None if continuous.

        None too when the bracket closes in on a stretch beyond one side of the band, where
        nothing is drawn (a branch heading off screen towards a pole outside the pair).
        The bracket ends come with their screen points (None where the function is undefined).
        """
        if p_a is None or p_b is None:
            return None
        gap = abs(p_b[1] - p_a[1])
        narrow_gaps: List[float] = []
        halving = 0
        for _ in range(JUMP_BISECTIONS):
            x_m = (x_a + x_b) / 2.0
            if not x_a < x_m < x_b:
                # Down to the float resolution of x with a gap left: a break
                return (x_a, p_a), (x_b, p_b)
            p_m = AdaptiveSampler._eval_screen_point(x_m, eval_func, math_to_screen)
            if p_m is None:
                # Undefined inside the step: a break
                return (x_a, p_a), (x_b, p_b)
            left_step = abs(p_m[1] - p_a[1])
            right_step = abs(p_b[1] - p_m[1])
            if left_step >= right_step:
                x_b, p_b = x_m, p_m
            else:
                x_a, p_a = x_m, p_m
            new_gap = max(left_step, right_step)
            if new_gap < JUMP_CONTINUOUS_PX:
                return None
            # A smooth stretch's gap halves with the bracket; a jump's stays above its size, so a
            # gap that has kept halving is a smooth curve (or holds a jump of under a few percent
            # of the pair's step, which the drawn chord hides)
            halving = halving + 1 if new_gap <= JUMP_SMOOTH_RATIO * gap else 0
            if halving >= JUMP_SMOOTH_HALVINGS:
                return None
            if AdaptiveSampler._both_beyond_band(viewport_band, p_a, p_b):
                return None
            # A jump keeps its gap and a pole's grows, while a continuous curve's shrinks (by
            # half where it is smooth, by 2^(-1/n) per halving at x^(1/n)'s vertical tangent,
            # unevenly while the tangent's point moves within the bracket: judged over a window)
            if abs(p_b[0] - p_a[0]) < JUMP_NARROW_PX:
                narrow_gaps.append(new_gap)
                window = len(narrow_gaps) - 1 - JUMP_STEADY_HALVINGS
                if window >= 0 and new_gap > JUMP_SHRINK_RATIO * narrow_gaps[window]:
                    return (x_a, p_a), (x_b, p_b)
            gap = new_gap
        # Still shrinking after every halving (a vertical tangent): continuous
        return None

    @staticmethod
    def _deviation(
        p_left: Tuple[float, float],
        p_mid: Tuple[float, float],
        p_right: Tuple[float, float],
    ) -> float:
        """Distance in px of p_mid from the line through p_left and p_right (0 when they coincide)."""
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return 0.0
        return abs((p_mid[0] - p_left[0]) * dy - (p_mid[1] - p_left[1]) * dx) / math.sqrt(length_sq)

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
