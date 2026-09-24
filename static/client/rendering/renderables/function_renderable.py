"""
Function renderable: computes function polylines in math or screen space.

This class extracts the sampling, discontinuity handling, and asymptote logic
from the math model (`drawables.function.Function`) so that the model remains
math-only and the renderer consumes a clean representation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Set, Tuple, cast

from rendering.primitives import MathPolyline, ScreenPolyline
from rendering.renderables.adaptive_sampler import AdaptiveSampler


class FunctionRenderable:
    def __init__(self, function_model: Any, coordinate_mapper: Any) -> None:
        self.func: Any = function_model
        self.mapper: Any = coordinate_mapper
        self._cached_screen_paths: Optional[ScreenPolyline] = None
        self._cache_valid: bool = False
        self._last_scale: Optional[float] = None
        self._last_bounds: Optional[Tuple[float, float]] = None
        self._last_vertical_bounds: Optional[Tuple[float, float]] = None
        self._last_screen_bounds: Optional[Tuple[int, int]] = None
        # Signature of the function model the cached paths were built from
        # (maintained by the function render helper).
        self._model_signature: Optional[str] = None
        # Per-build memo of f(x) so the sampler and the path builder evaluate
        # each x once, and the point discontinuities as a set.
        self._eval_cache: Optional[Dict[float, Any]] = None
        self._discontinuity_set: Optional[Set[float]] = None

    def invalidate_cache(self) -> None:
        self._cached_screen_paths = None
        self._cache_valid = False
        self._last_scale = None
        self._last_bounds = None
        self._last_vertical_bounds = None
        self._last_screen_bounds = None

    def _compute_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> float:
        """Compute angle at p2 formed by points p1-p2-p3, in radians."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return math.pi
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_angle)

    def _get_visible_bounds(self) -> Tuple[float, float]:
        try:
            left: float = self.mapper.get_visible_left_bound()
            right: float = self.mapper.get_visible_right_bound()
            return left, right
        except Exception:
            return -10, 10

    def _get_vertical_bounds(self) -> Optional[Tuple[float, float]]:
        try:
            return (self.mapper.get_visible_top_bound(), self.mapper.get_visible_bottom_bound())
        except Exception:
            return None

    def _get_screen_signature(self) -> Tuple[int, int]:
        screen_width = getattr(self.mapper, "canvas_width", None)
        screen_height = getattr(self.mapper, "canvas_height", None)
        return (int(screen_width or 0), int(screen_height or 0))

    def _update_cache_state(
        self,
        scale: Optional[float],
        bounds: Tuple[float, float],
        screen_sig: Tuple[int, int],
        vertical_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._last_scale = scale
        self._last_bounds = bounds
        self._last_vertical_bounds = vertical_bounds
        self._last_screen_bounds = screen_sig

    def _should_regenerate(self) -> bool:
        current_scale: Optional[float] = getattr(self.mapper, "scale_factor", None)
        current_bounds: Tuple[float, float] = self._get_visible_bounds()
        vertical_bounds = self._get_vertical_bounds()
        screen_signature = self._get_screen_signature()
        if self._cached_screen_paths is None or not self._cache_valid:
            self._update_cache_state(current_scale, current_bounds, screen_signature, vertical_bounds)
            return True
        if self._last_bounds != current_bounds or self._last_vertical_bounds != vertical_bounds:
            self._update_cache_state(current_scale, current_bounds, screen_signature, vertical_bounds)
            return True
        if self._last_scale != current_scale:
            self._update_cache_state(current_scale, current_bounds, screen_signature, vertical_bounds)
            return True
        if self._last_screen_bounds != screen_signature:
            self._update_cache_state(current_scale, current_bounds, screen_signature, vertical_bounds)
            return True
        return False

    def _resolve_bounds(self, left_bound: Optional[float], right_bound: Optional[float]) -> Tuple[float, float]:
        if left_bound is None or right_bound is None:
            v_left, v_right = self._get_visible_bounds()
            left_bound = v_left if left_bound is None else left_bound
            right_bound = right_bound if right_bound is not None else v_right
        if self.func.left_bound is not None:
            left_bound = max(left_bound, self.func.left_bound)
        if self.func.right_bound is not None:
            right_bound = min(right_bound, self.func.right_bound)
        return cast(float, left_bound), cast(float, right_bound)

    def _is_discontinuity(self, x: float) -> bool:
        if self._discontinuity_set is not None:
            return x in self._discontinuity_set
        try:
            if getattr(self.func, "point_discontinuities", None) and x in self.func.point_discontinuities:
                return True
        except Exception:
            pass
        return False

    def _get_asymptote_between(self, x1: float, x2: float) -> Optional[float]:
        if not hasattr(self.func, "get_vertical_asymptote_between_x"):
            return None
        try:
            return cast(Optional[float], self.func.get_vertical_asymptote_between_x(x1, x2))
        except Exception:
            return None

    def _evaluate_function(self, x: float) -> Optional[float]:
        try:
            return cast(Optional[float], self.func.function(x))
        except Exception:
            return None

    def _evaluate_cached(self, x: float) -> Any:
        """Evaluate f(x) once per build; failures are cached as None."""
        cache = self._eval_cache
        if cache is not None and x in cache:
            return cache[x]
        try:
            y = self.func.function(x)
        except Exception:
            y = None
        if cache is not None:
            cache[x] = y
        return y

    def _is_invalid_y(self, y: Optional[float]) -> bool:
        if y is None:
            return True
        if isinstance(y, float) and (y != y or abs(y) == float("inf")):
            return True
        return False

    def build_math_paths(self, left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> MathPolyline:
        left_bound, right_bound = self._resolve_bounds(left_bound, right_bound)
        assert left_bound is not None and right_bound is not None
        if right_bound <= left_bound:
            return MathPolyline([])

        step: float = (right_bound - left_bound) / 200.0
        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        x: float = left_bound
        expect_asymptote_behind: bool = False

        while x < right_bound - 1e-12:
            if self._is_discontinuity(x):
                x += step
                continue

            y: Optional[float] = self._evaluate_function(x)
            asymptote_x = self._get_asymptote_between(x, x + step)

            if asymptote_x is not None:
                expect_asymptote_behind = True
                y = self._evaluate_function(asymptote_x - min(1e-3, step / 10))
                x = asymptote_x - min(1e-3, step / 10)

            if expect_asymptote_behind:
                if current_path:
                    paths.append(current_path)
                current_path = []
                expect_asymptote_behind = False

            if self._is_invalid_y(y):
                if current_path:
                    paths.append(current_path)
                    current_path = []
                x += step
                continue

            current_path.append((x, cast(float, y)))
            x += step

        if current_path:
            paths.append(current_path)

        return MathPolyline(paths)

    def build_screen_paths(self) -> ScreenPolyline:
        if self._should_regenerate():
            screen_paths: list[list[tuple[float, float]]] = self._build_screen_paths_adaptive()
            self._cached_screen_paths = ScreenPolyline(screen_paths)
            self._cache_valid = True
        return self._cached_screen_paths or ScreenPolyline([])

    def _get_effective_bounds(self) -> Tuple[float, float]:
        visible_left, visible_right = self._get_visible_bounds()
        base_left: Optional[float] = getattr(self.func, "left_bound", None)
        base_right: Optional[float] = getattr(self.func, "right_bound", None)
        # Use visible bounds when no explicit function bounds are set
        if base_left is None:
            base_left = visible_left
        if base_right is None:
            base_right = visible_right
        return max(visible_left, base_left), min(visible_right, base_right)

    def _get_screen_dimensions(self) -> Tuple[float, float]:
        width: float = getattr(self.mapper, "canvas_width", 0) or 0
        height: float = getattr(self.mapper, "canvas_height", 0) or 0
        return width, height

    def _calculate_sample_points_by_subrange(self, left_bound: float, right_bound: float) -> list[list[float]]:
        """
        Calculate sample points, splitting at asymptotes and discontinuities into separate sub-ranges.
        Returns a list of sample lists, one per continuous sub-range.
        """
        canvas_width = int(getattr(self.mapper, "canvas_width", 800) or 800)
        viewport_height = getattr(self.mapper, "canvas_height", None) or None

        initial_segments = None
        if getattr(self.func, "is_periodic", False) and getattr(self.func, "estimated_period", None):
            range_width = right_bound - left_bound
            num_periods = range_width / self.func.estimated_period
            initial_segments = min(canvas_width, max(8, int(num_periods * 4)))

        # Get asymptotes AND point discontinuities - both require splitting
        asymptotes = getattr(self.func, "vertical_asymptotes", []) or []
        point_discontinuities = getattr(self.func, "point_discontinuities", []) or []

        # Combine all split points (asymptotes and discontinuities)
        all_split_points = sorted(set(asymptotes + point_discontinuities))

        if all_split_points:
            return cast(
                list[list[float]],
                AdaptiveSampler.generate_samples_with_asymptotes(
                    left_bound,
                    right_bound,
                    self._evaluate_cached,
                    self.mapper.math_to_screen,
                    all_split_points,
                    initial_segments,
                    max_samples=canvas_width,
                    viewport_height=viewport_height,
                ),
            )
        else:
            # No split points - single range
            samples, _ = AdaptiveSampler.generate_samples(
                left_bound,
                right_bound,
                self._evaluate_cached,
                self.mapper.math_to_screen,
                initial_segments,
                max_samples=canvas_width,
                viewport_height=viewport_height,
            )
            return [samples] if samples else []

    def _eval_scaled_point(self, x_val: float) -> Tuple[Tuple[Optional[float], Optional[float]], Any]:
        y_val: Any = self._evaluate_cached(x_val)
        # Undefined values (NaN, inf, non-numeric) break the path instead of
        # letting it bridge an undefined interval.
        if not isinstance(y_val, (int, float)) or self._is_invalid_y(y_val):
            return (None, None), y_val
        try:
            sx, sy = self.mapper.math_to_screen(x_val, y_val)
            return (sx, sy), y_val
        except Exception:
            return (None, None), None

    def _is_large_jump(self, prev_sy: float, sy: float, height: float) -> bool:
        return abs(prev_sy - sy) > height * 2

    def _finalize_path(self, current_path: list[tuple[float, float]], paths: list[list[tuple[float, float]]]) -> None:
        """Add current path to paths list if non-empty."""
        if current_path:
            paths.append(current_path)

    def _build_screen_paths_adaptive(self) -> list[list[tuple[float, float]]]:
        """
        Adaptive path building using sample points split by asymptotes.
        Each sub-range between asymptotes is sampled independently.
        """
        left_bound, right_bound = self._get_effective_bounds()
        width, height = self._get_screen_dimensions()
        self._eval_cache = {}
        self._discontinuity_set = set(getattr(self.func, "point_discontinuities", None) or [])
        try:
            # Get samples split by asymptotes - each sub-list is a continuous range
            sample_subranges = self._calculate_sample_points_by_subrange(left_bound, right_bound)

            all_paths: list[list[tuple[float, float]]] = []

            for sample_points in sample_subranges:
                paths = self._build_path_from_samples(sample_points, height)
                all_paths.extend(paths)
            return all_paths
        finally:
            self._eval_cache = None
            self._discontinuity_set = None

    def _interpolate_boundary_crossing(
        self, sx1: float, sy1: float, sx2: float, sy2: float, height: float
    ) -> Optional[tuple[float, float]]:
        """Calculate intersection point where line crosses screen boundary (y=0 or y=height)."""
        if sy1 == sy2:
            return None
        # Check if crossing top boundary (y=0)
        if (sy1 < 0 <= sy2) or (sy2 < 0 <= sy1):
            t = (0 - sy1) / (sy2 - sy1)
            return (sx1 + t * (sx2 - sx1), 0.0)
        # Check if crossing bottom boundary (y=height)
        if (sy1 <= height < sy2) or (sy2 <= height < sy1):
            t = (height - sy1) / (sy2 - sy1)
            return (sx1 + t * (sx2 - sx1), height)
        return None

    def _is_on_screen(self, sy: float, height: float) -> bool:
        """Check if y coordinate is within screen bounds."""
        return 0 <= sy <= height

    def _build_path_from_samples(self, sample_points: list[float], height: float) -> list[list[tuple[float, float]]]:
        """
        Build screen paths from a list of sample x-values (within a single sub-range).
        Path breaks on: failed evaluation, large y-jumps, discontinuities, or asymptotes.
        """
        if len(sample_points) < 2:
            return []

        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        prev_sy: Optional[float] = None
        prev_sx: Optional[float] = None
        prev_x: Optional[float] = None

        for x in sample_points:
            if self._is_discontinuity(x):
                self._finalize_path(current_path, paths)
                current_path = []
                prev_sy = None
                prev_sx = None
                prev_x = None
                continue

            if prev_x is not None and self._get_asymptote_between(prev_x, x) is not None:
                self._finalize_path(current_path, paths)
                current_path = []
                prev_sy = None
                prev_sx = None

            scaled_point, _ = self._eval_scaled_point(x)
            if scaled_point[0] is None:
                self._finalize_path(current_path, paths)
                current_path = []
                prev_sy = None
                prev_sx = None
                prev_x = x
                continue

            sx, sy = scaled_point[0], cast(float, scaled_point[1])
            on_screen = self._is_on_screen(sy, height)
            prev_on_screen = prev_sy is not None and self._is_on_screen(prev_sy, height)

            # Handle boundary crossings BEFORE large jump check
            if prev_sx is not None and prev_sy is not None:
                crossing = self._interpolate_boundary_crossing(prev_sx, prev_sy, sx, sy, height)
                if crossing:
                    if prev_on_screen and not on_screen:
                        current_path.append(crossing)
                        self._finalize_path(current_path, paths)
                        current_path = []
                    elif not prev_on_screen and on_screen:
                        current_path.append(crossing)

            # Large jump without crossing = discontinuity, break path
            if prev_sy is not None and self._is_large_jump(prev_sy, sy, height) and on_screen and prev_on_screen:
                self._finalize_path(current_path, paths)
                current_path = []

            if on_screen:
                current_path.append((sx, sy))

            prev_sy = sy
            prev_sx = sx
            prev_x = x

        self._finalize_path(current_path, paths)
        return paths

    def _is_valid_extension(self, orig_y: float, ext_y: float, height: float) -> bool:
        """
        Check if extension point is valid (same side of screen, not crossing to different branch).
        - If original is in top half and extension goes further up (or to top boundary): valid
        - If original is in bottom half and extension goes further down (or to bottom boundary): valid
        - If extension crosses from top half to bottom half or vice versa: invalid
        """
        mid = height / 2
        orig_in_top_half = orig_y < mid
        ext_in_top_half = ext_y < mid
        # Extension is valid if both are in same half, or extension is at boundary
        if ext_y <= 0 or ext_y >= height:
            # Extension is at boundary - valid only if it's the "correct" boundary
            if orig_in_top_half and ext_y <= 0:
                return True  # Top half extending to top boundary
            if not orig_in_top_half and ext_y >= height:
                return True  # Bottom half extending to bottom boundary
            return False  # Wrong boundary
        return orig_in_top_half == ext_in_top_half

    def _is_inside_screen(self, sx: float, sy: float, width: float, height: float) -> bool:
        """Check if point is strictly inside visible screen bounds."""
        return 0 < sy < height and 0 < sx < width

    def _is_outside_screen_y(self, sy: float, height: float) -> bool:
        """Check if point is outside screen in Y direction."""
        return sy <= 0 or sy >= height

    def _clamp_to_boundary(self, x1: float, y1: float, x2: float, y2: float, height: float) -> tuple[float, float]:
        """
        If (x2,y2) is outside screen, return intersection of line (x1,y1)→(x2,y2)
        with screen boundary. Uses linear interpolation: t = (target_y - y1) / (y2 - y1)
        """
        if 0 <= y2 <= height:
            return (x2, y2)
        if abs(y2 - y1) < 1e-9:
            return (x2, max(0.0, min(y2, height)))
        if y2 < 0:
            t = -y1 / (y2 - y1)
            return (x1 + t * (x2 - x1), 0.0)
        t = (height - y1) / (y2 - y1)
        return (x1 + t * (x2 - x1), height)
