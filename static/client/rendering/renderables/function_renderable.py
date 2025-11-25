"""
Function renderable: computes function polylines in math or screen space.

This class extracts the sampling, discontinuity handling, and asymptote logic
from the math model (`drawables.function.Function`) so that the model remains
math-only and the renderer consumes a clean representation.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from rendering.primitives import MathPolyline, ScreenPolyline


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

    def _should_regenerate(self) -> bool:
        current_scale: Optional[float] = getattr(self.mapper, 'scale_factor', None)
        left: float
        right: float
        left, right = self._get_visible_bounds()
        current_bounds: Tuple[float, float] = (left, right)
        screen_width = getattr(self.mapper, "canvas_width", None) or getattr(self.cartesian2axis, "width", None)
        screen_height = getattr(self.mapper, "canvas_height", None) or getattr(self.cartesian2axis, "height", None)
        screen_signature = (int(screen_width or 0), int(screen_height or 0))
        if (self._cached_screen_paths is None or not self._cache_valid or self._last_bounds != current_bounds):
            self._last_scale = current_scale
            self._last_bounds = current_bounds
            self._last_screen_bounds = screen_signature
            return True
        if self._last_scale is not None and current_scale is not None:
            ratio: float = current_scale / self._last_scale if self._last_scale else 1
            if ratio < 0.8 or ratio > 1.2:
                self._last_scale = current_scale
                self._last_bounds = current_bounds
                self._last_screen_bounds = screen_signature
                return True
        if self._last_screen_bounds != screen_signature:
            self._last_screen_bounds = screen_signature
            return True
        return False

    def build_math_paths(self, left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> MathPolyline:
        if left_bound is None or right_bound is None:
            v_left: float
            v_right: float
            v_left, v_right = self._get_visible_bounds()
            left_bound = v_left if left_bound is None else left_bound
            right_bound = right_bound if right_bound is not None else v_right

        if self.func.left_bound is not None:
            left_bound = max(left_bound, self.func.left_bound)
        if self.func.right_bound is not None:
            right_bound = min(right_bound, self.func.right_bound)

        assert left_bound is not None and right_bound is not None

        if right_bound <= left_bound:
            return MathPolyline([])

        range_width: float = right_bound - left_bound
        step: float = range_width / 200.0

        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        x: float = left_bound
        expect_asymptote_behind: bool = False

        while x < right_bound - 1e-12:
            try:
                if hasattr(self.func, 'point_discontinuities') and self.func.point_discontinuities and x in self.func.point_discontinuities:
                    x += step
                    continue
            except Exception:
                pass

            try:
                y: Optional[float] = self.func.function(x)
            except Exception:
                y = None

            asymptote_x: Optional[float] = None
            if hasattr(self.func, 'get_vertical_asymptote_between_x'):
                try:
                    asymptote_x = self.func.get_vertical_asymptote_between_x(x, x + step)
                except Exception:
                    asymptote_x = None
            has_vertical_asymptote_in_front: bool = asymptote_x is not None

            if has_vertical_asymptote_in_front:
                expect_asymptote_behind = True
                try:
                    assert asymptote_x is not None
                    y = self.func.function(asymptote_x - min(1e-3, step / 10))
                    x = asymptote_x - min(1e-3, step / 10)
                except Exception:
                    y = None

            if expect_asymptote_behind:
                if current_path:
                    paths.append(current_path)
                current_path = []
                expect_asymptote_behind = False

            if y is None or (isinstance(y, float) and (y != y or abs(y) == float('inf'))):
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

    def _build_screen_paths_equivalent(self) -> list[list[tuple[float, float]]]:
        visible_left: float
        visible_right: float
        visible_left, visible_right = self._get_visible_bounds()
        base_left: Optional[float] = getattr(self.func, 'left_bound', None)
        base_right: Optional[float] = getattr(self.func, 'right_bound', None)
        if base_left is None or base_right is None:
            base_left = -10 if base_left is None else base_left
            base_right = 10 if base_right is None else base_right
        left_bound: float = max(visible_left, base_left)
        right_bound: float = min(visible_right, base_right)

        MAX_POINTS: float = 160.0
        MAX_POINTS_VISIBLE_FRACTION: float = 0.65
        MIN_POINTS_PER_FUNCTION: int = 100

        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        expect_asymptote_behind: bool = False

        def calculate_step_size() -> float:
            import re
            import math as _m
            range_width: float = right_bound - left_bound
            if range_width <= 0:
                return 1.0
            base_step: float = range_width / 200.0
            fs: str = getattr(self.func, 'function_string', '')
            if 'sin' in fs or 'cos' in fs:
                matches: list[str] = re.findall(r'(?:sin|cos)\((\d+(?:\.\d+)?)\*?x\)', fs)
                freq_multiplier: float = float(matches[0]) if matches else 1.0
                period: float = 2 * _m.pi / freq_multiplier
                visible_periods: float = range_width / period if period != 0 else 1.0
                if visible_periods <= 1:
                    return base_step
                points_per_period: float = 320.0 / max(visible_periods, 1e-9)
                points_per_period = min(MAX_POINTS, points_per_period)
                step_trig: float = period / points_per_period if points_per_period > 0 else base_step
                cap_step: float = range_width / MAX_POINTS
                return max(cap_step, step_trig)
            if any(pattern in fs for pattern in ['x**2', 'x^2']):
                return base_step * 2.0
            return base_step

        step: float = calculate_step_size()
        if step > 0:
            cap_step: float = (right_bound - left_bound) / MAX_POINTS
            step = max(cap_step, step)

        try:
            eps: float = max(1e-6, (right_bound - left_bound) / 1000.0)
            cx: float = (left_bound + right_bound) / 2.0
            y1: Any = self.func.function(cx - eps)
            y2: Any = self.func.function(cx + eps)
            if isinstance(y1, (int, float)) and isinstance(y2, (int, float)):
                slope_abs: float = abs(y2 - y1) / (2.0 * eps) if eps > 0 else 0.0
                if slope_abs > 0:
                    scale: float = getattr(self.mapper, 'scale_factor', 1.0) or 1.0
                    target_pixel: float = 2.0
                    desired_step: float = (target_pixel / scale) / slope_abs
                    step = min(step, max((right_bound - left_bound) / MAX_POINTS, desired_step))
        except Exception:
            pass

        height: float = getattr(self.cartesian2axis, 'height', None) or 0
        if not height:
            height = getattr(self.mapper, 'canvas_height', 0) or 0
        width: float = getattr(self.cartesian2axis, 'width', None) or 0
        if not width:
            width = getattr(self.mapper, 'canvas_width', 0) or 0
        screen_margin: float = 16.0
        visible_min_x: float = -screen_margin
        visible_max_x: float = width + screen_margin
        samples_emitted = 0
        samples_visible = 0

        def eval_scaled_point(x_val: float) -> tuple[tuple[float, float] | tuple[None, None], Any]:
            try:
                y_val: Any = self.func.function(x_val)
                sx: float
                sy: float
                sx, sy = self.mapper.math_to_screen(x_val, y_val)
                return (sx, sy), y_val
            except Exception:
                return (None, None), None

        x: float = left_bound
        while x < right_bound - 1e-12:
            try:
                if getattr(self.func, 'point_discontinuities', None) and x in self.func.point_discontinuities:
                    x += step
                    continue
            except Exception:
                pass

            scaled_point: tuple[float, float] | tuple[None, None]
            y_val: Any
            scaled_point, y_val = eval_scaled_point(x)
            if not scaled_point[0]:
                x += step
                continue

            asymptote_x: Optional[float] = None
            if hasattr(self.func, 'get_vertical_asymptote_between_x'):
                try:
                    asymptote_x = self.func.get_vertical_asymptote_between_x(x, x + step)
                except Exception:
                    asymptote_x = None
            has_vertical_asymptote_in_front: bool = asymptote_x is not None
            has_vertical_asymptote_behind: bool = expect_asymptote_behind

            if has_vertical_asymptote_in_front:
                expect_asymptote_behind = True
                new_scaled_point: tuple[float, float] | tuple[None, None]
                new_y: Any
                assert asymptote_x is not None
                new_scaled_point, new_y = eval_scaled_point(asymptote_x - min(1e-3, step / 10))
                if new_scaled_point[0] is not None:
                    scaled_point, y_val = new_scaled_point, new_y

            neighbor_prev_scaled_point: tuple[float, float] | tuple[None, None]
            if has_vertical_asymptote_behind:
                try:
                    asymptote_x_prev: Optional[float] = self.func.get_vertical_asymptote_between_x(x - step, x)
                except Exception:
                    asymptote_x_prev = None
                if asymptote_x_prev is not None:
                    neighbor_prev_scaled_point, _ = eval_scaled_point(asymptote_x_prev + min(1e-3, step / 10))
                expect_asymptote_behind = False
            else:
                neighbor_prev_scaled_point, _ = eval_scaled_point(x - step)

            if not neighbor_prev_scaled_point[0]:
                x += step
                continue

            prev_sy: float = neighbor_prev_scaled_point[1]
            sy: float = scaled_point[1]
            prev_sx: float = neighbor_prev_scaled_point[0]
            sx_val: float = scaled_point[0]

            if abs(prev_sy - sy) > height * 2:
                if current_path:
                    paths.append(current_path)
                    current_path = []
                x += step
                continue

            top_bound: float = 0
            bottom_bound: float = height
            crosses_top_bound_upward: bool = prev_sy >= top_bound and sy < top_bound
            crosses_top_bound_downward: bool = prev_sy <= top_bound and sy > top_bound
            crosses_bottom_bound_downward: bool = prev_sy <= bottom_bound and sy > bottom_bound
            crosses_bottom_bound_upward: bool = prev_sy >= bottom_bound and sy < bottom_bound
            left_bound_screen: float = -screen_margin
            right_bound_screen: float = width + screen_margin
            crosses_left_enter: bool = prev_sx <= left_bound_screen and sx_val > left_bound_screen
            crosses_left_exit: bool = prev_sx >= left_bound_screen and sx_val < left_bound_screen
            crosses_right_enter: bool = prev_sx >= right_bound_screen and sx_val < right_bound_screen
            crosses_right_exit: bool = prev_sx <= right_bound_screen and sx_val > right_bound_screen

            crossed_bound_onto_screen: bool = crosses_top_bound_downward or crosses_bottom_bound_upward
            crossed_bound_off_screen: bool = crosses_top_bound_upward or crosses_bottom_bound_downward

            if x > left_bound:
                if crossed_bound_onto_screen:
                    current_path.append((neighbor_prev_scaled_point[0], neighbor_prev_scaled_point[1]))
                    current_path.append((scaled_point[0], scaled_point[1]))
                    x += step
                    continue
                if crossed_bound_off_screen:
                    current_path.append((scaled_point[0], scaled_point[1]))
                    if current_path:
                        paths.append(current_path)
                        current_path = []
                    x += step
                    continue
                if width > 0:
                    if crosses_left_enter or crosses_right_enter:
                        current_path.append((neighbor_prev_scaled_point[0], neighbor_prev_scaled_point[1]))
                        current_path.append((scaled_point[0], scaled_point[1]))
                        x += step
                        continue
                    if crosses_left_exit or crosses_right_exit:
                        current_path.append((scaled_point[0], scaled_point[1]))
                        if current_path:
                            paths.append(current_path)
                            current_path = []
                        x += step
                        continue

            if x > right_bound:
                break

            if sy >= bottom_bound or sy <= top_bound:
                x += step
                continue
            if width > 0 and not (visible_min_x <= sx_val <= visible_max_x):
                x += step
                continue
            samples_visible += 1

            if not current_path:
                current_path = [(scaled_point[0], scaled_point[1])]
                samples_emitted += 1
                x += step
                continue

            pixel_diff: float = 1
            lastx: float
            lasty: float
            lastx, lasty = current_path[-1]
            visible_x_diff: bool = abs(scaled_point[0] - lastx) > pixel_diff
            visible_y_diff: bool = abs(scaled_point[1] - lasty) > pixel_diff
            if visible_x_diff or visible_y_diff:
                current_path.append((scaled_point[0], scaled_point[1]))
                samples_emitted += 1

            x += step

        if current_path:
            paths.append(current_path)

        total_points_threshold = max(MIN_POINTS_PER_FUNCTION, int(MAX_POINTS * MAX_POINTS_VISIBLE_FRACTION))
        if samples_emitted > total_points_threshold and samples_visible > total_points_threshold:
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
                paths = downsampled_paths
            else:
                paths = original_paths

        return paths

