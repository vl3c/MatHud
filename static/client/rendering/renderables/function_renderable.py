"""
Function renderable: computes function polylines in math or screen space.

This class extracts the sampling, discontinuity handling, and asymptote logic
from the math model (`drawables.function.Function`) so that the model remains
math-only and the renderer consumes a clean representation.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from rendering.primitives import MathPolyline, ScreenPolyline
from rendering.renderables.curve_step_calculator import (
    MAX_POINTS,
    AdaptiveStepCalculator,
    StepCalculator,
    USE_ADAPTIVE_STEP_CALCULATOR,
)

MAX_POINTS_VISIBLE_FRACTION: float = 0.65
MIN_POINTS_PER_FUNCTION: int = 100
SCREEN_MARGIN: float = 16.0
SCALE_CHANGE_THRESHOLD_LOW: float = 0.8
SCALE_CHANGE_THRESHOLD_HIGH: float = 1.2


class FunctionRenderable:
    def __init__(self, function_model: Any, coordinate_mapper: Any, cartesian2axis: Optional[Any] = None) -> None:
        self.func: Any = function_model
        self.mapper: Any = coordinate_mapper
        self.cartesian2axis: Optional[Any] = cartesian2axis
        self._cached_screen_paths: Optional[ScreenPolyline] = None
        self._cache_valid: bool = False
        self._last_scale: Optional[float] = None
        self._last_bounds: Optional[Tuple[float, float]] = None
        self._last_screen_bounds: Optional[Tuple[int, int]] = None

    def invalidate_cache(self) -> None:
        self._cached_screen_paths = None
        self._cache_valid = False
        self._last_scale = None
        self._last_bounds = None
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
            if self.cartesian2axis:
                return (
                    self.cartesian2axis.get_visible_left_bound(),
                    self.cartesian2axis.get_visible_right_bound(),
                )
            left: float = self.mapper.get_visible_left_bound()
            right: float = self.mapper.get_visible_right_bound()
            return left, right
        except Exception:
            return -10, 10

    def _get_screen_signature(self) -> Tuple[int, int]:
        screen_width = getattr(self.mapper, "canvas_width", None) or getattr(self.cartesian2axis, "width", None)
        screen_height = getattr(self.mapper, "canvas_height", None) or getattr(self.cartesian2axis, "height", None)
        return (int(screen_width or 0), int(screen_height or 0))

    def _update_cache_state(self, scale: Optional[float], bounds: Tuple[float, float], screen_sig: Tuple[int, int]) -> None:
        self._last_scale = scale
        self._last_bounds = bounds
        self._last_screen_bounds = screen_sig

    def _should_regenerate(self) -> bool:
        current_scale: Optional[float] = getattr(self.mapper, 'scale_factor', None)
        current_bounds: Tuple[float, float] = self._get_visible_bounds()
        screen_signature = self._get_screen_signature()
        if self._cached_screen_paths is None or not self._cache_valid or self._last_bounds != current_bounds:
            self._update_cache_state(current_scale, current_bounds, screen_signature)
            return True
        if self._last_scale is not None and current_scale is not None:
            ratio: float = current_scale / self._last_scale if self._last_scale else 1
            if ratio < SCALE_CHANGE_THRESHOLD_LOW or ratio > SCALE_CHANGE_THRESHOLD_HIGH:
                self._update_cache_state(current_scale, current_bounds, screen_signature)
                return True
        if self._last_screen_bounds != screen_signature:
            self._last_screen_bounds = screen_signature
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
        return left_bound, right_bound

    def _is_discontinuity(self, x: float) -> bool:
        try:
            if getattr(self.func, 'point_discontinuities', None) and x in self.func.point_discontinuities:
                return True
        except Exception:
            pass
        return False

    def _get_asymptote_between(self, x1: float, x2: float) -> Optional[float]:
        if not hasattr(self.func, 'get_vertical_asymptote_between_x'):
            return None
        try:
            return self.func.get_vertical_asymptote_between_x(x1, x2)
        except Exception:
            return None

    def _evaluate_function(self, x: float) -> Optional[float]:
        try:
            return self.func.function(x)
        except Exception:
            return None

    def _is_invalid_y(self, y: Optional[float]) -> bool:
        if y is None:
            return True
        if isinstance(y, float) and (y != y or abs(y) == float('inf')):
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

            current_path.append((x, y))
            x += step

        if current_path:
            paths.append(current_path)

        return MathPolyline(paths)

    def build_screen_paths(self) -> ScreenPolyline:
        if self._should_regenerate():
            screen_paths: list[list[tuple[float, float]]] = self._build_screen_paths_equivalent()
            self._cached_screen_paths = ScreenPolyline(screen_paths)
            self._cache_valid = True
        return self._cached_screen_paths or ScreenPolyline([])

    def _get_effective_bounds(self) -> Tuple[float, float]:
        visible_left, visible_right = self._get_visible_bounds()
        base_left: Optional[float] = getattr(self.func, 'left_bound', None)
        base_right: Optional[float] = getattr(self.func, 'right_bound', None)
        if base_left is None:
            base_left = -10
        if base_right is None:
            base_right = 10
        return max(visible_left, base_left), min(visible_right, base_right)

    def _get_screen_dimensions(self) -> Tuple[float, float]:
        height: float = getattr(self.cartesian2axis, 'height', None) or 0
        if not height:
            height = getattr(self.mapper, 'canvas_height', 0) or 0
        width: float = getattr(self.cartesian2axis, 'width', None) or 0
        if not width:
            width = getattr(self.mapper, 'canvas_width', 0) or 0
        return width, height

    def _calculate_step(self, left_bound: float, right_bound: float) -> float:
        if USE_ADAPTIVE_STEP_CALCULATOR:
            return self._calculate_step_adaptive(left_bound, right_bound)
        return self._calculate_step_heuristic(left_bound, right_bound)

    def _calculate_step_heuristic(self, left_bound: float, right_bound: float) -> float:
        function_string = getattr(self.func, 'function_string', '')
        scale_factor = getattr(self.mapper, 'scale_factor', 1.0) or 1.0
        return StepCalculator.calculate(
            left_bound, right_bound, function_string, self.func.function, scale_factor
        )

    def _calculate_step_adaptive(self, left_bound: float, right_bound: float) -> float:
        return AdaptiveStepCalculator.calculate(
            left_bound, right_bound, self.func.function, self.mapper.math_to_screen
        )

    def _eval_scaled_point(self, x_val: float) -> Tuple[Tuple[Optional[float], Optional[float]], Any]:
        try:
            y_val: Any = self.func.function(x_val)
            sx, sy = self.mapper.math_to_screen(x_val, y_val)
            return (sx, sy), y_val
        except Exception:
            return (None, None), None

    def _downsample_paths(self, paths: list, samples_emitted: int, samples_visible: int) -> list:
        total_points_threshold = max(MIN_POINTS_PER_FUNCTION, int(MAX_POINTS * MAX_POINTS_VISIBLE_FRACTION))
        if samples_emitted <= total_points_threshold or samples_visible <= total_points_threshold:
            return paths
        original_paths = paths
        downsampled_paths: list[list[tuple[float, float]]] = []
        decimation_factor = int(max(2, round(samples_emitted / total_points_threshold)))
        for path in paths:
            if len(path) <= decimation_factor:
                downsampled_paths.append(path)
                continue
            reduced: list[tuple[float, float]] = []
            for index, point in enumerate(path):
                if index % decimation_factor == 0 or index == len(path) - 1:
                    reduced.append(point)
            downsampled_paths.append(reduced)
        total_after = sum(len(path) for path in downsampled_paths)
        if total_after >= MIN_POINTS_PER_FUNCTION:
            return downsampled_paths
        return original_paths

    def _adjust_point_for_asymptote_ahead(
        self, x: float, step: float, scaled_point: Tuple, y_val: Any
    ) -> Tuple[Tuple, Any, bool]:
        asymptote_x = self._get_asymptote_between(x, x + step)
        if asymptote_x is None:
            return scaled_point, y_val, False
        new_scaled_point, new_y = self._eval_scaled_point(asymptote_x - min(1e-3, step / 10))
        if new_scaled_point[0] is not None:
            return new_scaled_point, new_y, True
        return scaled_point, y_val, True

    def _get_neighbor_prev_point(
        self, x: float, step: float, had_asymptote_behind: bool
    ) -> Tuple[Tuple[Optional[float], Optional[float]], Any]:
        if had_asymptote_behind:
            asymptote_x_prev = self._get_asymptote_between(x - step, x)
            if asymptote_x_prev is not None:
                return self._eval_scaled_point(asymptote_x_prev + min(1e-3, step / 10))
        return self._eval_scaled_point(x - step)

    def _is_large_jump(self, prev_sy: float, sy: float, height: float) -> bool:
        return abs(prev_sy - sy) > height * 2

    def _is_point_visible(
        self, sx: float, sy: float, width: float, height: float,
        visible_min_x: float, visible_max_x: float
    ) -> bool:
        if sy >= height or sy <= 0:
            return False
        if width > 0 and not (visible_min_x <= sx <= visible_max_x):
            return False
        return True

    def _should_emit_point(self, scaled_point: Tuple, current_path: list) -> bool:
        if not current_path:
            return True
        lastx, lasty = current_path[-1]
        return abs(scaled_point[0] - lastx) > 1 or abs(scaled_point[1] - lasty) > 1

    def _build_screen_paths_equivalent(self) -> list[list[tuple[float, float]]]:
        left_bound, right_bound = self._get_effective_bounds()
        width, height = self._get_screen_dimensions()
        step = self._calculate_step(left_bound, right_bound)
        visible_min_x: float = -SCREEN_MARGIN
        visible_max_x: float = width + SCREEN_MARGIN

        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        expect_asymptote_behind: bool = False
        samples_emitted = 0
        samples_visible = 0

        x: float = left_bound
        while x < right_bound - 1e-12:
            if self._is_discontinuity(x):
                x += step
                continue

            scaled_point, y_val = self._eval_scaled_point(x)
            if scaled_point[0] is None:
                x += step
                continue

            scaled_point, y_val, has_asymptote_ahead = self._adjust_point_for_asymptote_ahead(
                x, step, scaled_point, y_val
            )
            had_asymptote_behind = expect_asymptote_behind
            if has_asymptote_ahead:
                expect_asymptote_behind = True

            neighbor_prev, _ = self._get_neighbor_prev_point(x, step, had_asymptote_behind)
            if had_asymptote_behind:
                expect_asymptote_behind = False

            if neighbor_prev[0] is None:
                x += step
                continue

            prev_sx, prev_sy = neighbor_prev[0], neighbor_prev[1]
            sx_val, sy = scaled_point[0], scaled_point[1]

            if self._is_large_jump(prev_sy, sy, height):
                if current_path:
                    paths.append(current_path)
                    current_path = []
                x += step
                continue

            boundary_result = self._handle_boundary_crossing(
                x, left_bound, right_bound, width, height,
                prev_sx, prev_sy, sx_val, sy,
                neighbor_prev, scaled_point, current_path, paths
            )
            if boundary_result is not None:
                current_path, paths, should_continue = boundary_result
                if should_continue:
                    x += step
                    continue

            if x > right_bound:
                break

            if not self._is_point_visible(sx_val, sy, width, height, visible_min_x, visible_max_x):
                x += step
                continue
            samples_visible += 1

            if self._should_emit_point(scaled_point, current_path):
                current_path.append((scaled_point[0], scaled_point[1]))
                samples_emitted += 1

            x += step

        if current_path:
            paths.append(current_path)

        return self._downsample_paths(paths, samples_emitted, samples_visible)

    def _handle_boundary_crossing(
        self, x: float, left_bound: float, right_bound: float, width: float, height: float,
        prev_sx: float, prev_sy: float, sx_val: float, sy: float,
        neighbor_prev_scaled_point: Tuple, scaled_point: Tuple,
        current_path: list, paths: list
    ) -> Optional[Tuple[list, list, bool]]:
        if x <= left_bound:
            return None

        top_bound: float = 0
        bottom_bound: float = height
        crosses_top_bound_upward: bool = prev_sy >= top_bound and sy < top_bound
        crosses_top_bound_downward: bool = prev_sy <= top_bound and sy > top_bound
        crosses_bottom_bound_downward: bool = prev_sy <= bottom_bound and sy > bottom_bound
        crosses_bottom_bound_upward: bool = prev_sy >= bottom_bound and sy < bottom_bound

        crossed_bound_onto_screen: bool = crosses_top_bound_downward or crosses_bottom_bound_upward
        crossed_bound_off_screen: bool = crosses_top_bound_upward or crosses_bottom_bound_downward

        if crossed_bound_onto_screen:
            current_path.append((neighbor_prev_scaled_point[0], neighbor_prev_scaled_point[1]))
            current_path.append((scaled_point[0], scaled_point[1]))
            return current_path, paths, True

        if crossed_bound_off_screen:
            current_path.append((scaled_point[0], scaled_point[1]))
            if current_path:
                paths.append(current_path)
                current_path = []
            return current_path, paths, True

        if width > 0:
            left_bound_screen: float = -SCREEN_MARGIN
            right_bound_screen: float = width + SCREEN_MARGIN
            crosses_left_enter: bool = prev_sx <= left_bound_screen and sx_val > left_bound_screen
            crosses_left_exit: bool = prev_sx >= left_bound_screen and sx_val < left_bound_screen
            crosses_right_enter: bool = prev_sx >= right_bound_screen and sx_val < right_bound_screen
            crosses_right_exit: bool = prev_sx <= right_bound_screen and sx_val > right_bound_screen

            if crosses_left_enter or crosses_right_enter:
                current_path.append((neighbor_prev_scaled_point[0], neighbor_prev_scaled_point[1]))
                current_path.append((scaled_point[0], scaled_point[1]))
                return current_path, paths, True

            if crosses_left_exit or crosses_right_exit:
                current_path.append((scaled_point[0], scaled_point[1]))
                if current_path:
                    paths.append(current_path)
                    current_path = []
                return current_path, paths, True

        return None

