"""
Function renderable: computes function polylines in math or screen space.

This class extracts the sampling, discontinuity handling, and asymptote logic
from the math model (`drawables.function.Function`) so that the model remains
math-only and the renderer consumes a clean representation.
"""

from rendering.primitives import MathPolyline, ScreenPolyline


class FunctionRenderable:
    def __init__(self, function_model, coordinate_mapper, cartesian2axis=None):
        self.func = function_model
        self.mapper = coordinate_mapper
        # Prefer explicitly provided cartesian2axis; otherwise, do not rely on model.canvas
        self.cartesian2axis = cartesian2axis

        # Simple cache
        self._cached_screen_paths = None
        self._cache_valid = False
        self._last_scale = None
        self._last_bounds = None

    def invalidate_cache(self):
        self._cached_screen_paths = None
        self._cache_valid = False
        self._last_scale = None
        self._last_bounds = None

    def _get_visible_bounds(self):
        # Use cartesian2axis if available (legacy path), otherwise rely on mapper
        try:
            if self.cartesian2axis:
                return (
                    self.cartesian2axis.get_visible_left_bound(),
                    self.cartesian2axis.get_visible_right_bound(),
                )
            # Mapper-based bounds
            left = self.mapper.get_visible_left_bound()
            right = self.mapper.get_visible_right_bound()
            return left, right
        except Exception:
            # Fallback safe default
            return -10, 10

    def _should_regenerate(self):
        current_scale = getattr(self.mapper, 'scale_factor', None)
        left, right = self._get_visible_bounds()
        current_bounds = (left, right)
        if (self._cached_screen_paths is None or not self._cache_valid or self._last_bounds != current_bounds):
            self._last_scale = current_scale
            self._last_bounds = current_bounds
            return True
        if self._last_scale is not None and current_scale is not None:
            ratio = current_scale / self._last_scale if self._last_scale else 1
            if ratio < 0.8 or ratio > 1.2:
                self._last_scale = current_scale
                self._last_bounds = current_bounds
                return True
        return False

    def build_math_paths(self, left_bound=None, right_bound=None):
        # Determine bounds
        if left_bound is None or right_bound is None:
            v_left, v_right = self._get_visible_bounds()
            left_bound = v_left if left_bound is None else left_bound
            right_bound = right_bound if right_bound is not None else v_right

        # Clamp to model bounds if present
        if self.func.left_bound is not None:
            left_bound = max(left_bound, self.func.left_bound)
        if self.func.right_bound is not None:
            right_bound = min(right_bound, self.func.right_bound)

        if right_bound <= left_bound:
            return MathPolyline([])

        # Step heuristic similar to original, capped to at most 200 samples across range
        range_width = right_bound - left_bound
        step = range_width / 200.0

        paths = []
        current_path = []
        x = left_bound
        expect_asymptote_behind = False

        # Use a strict upper bound to avoid exceeding 200 points due to inclusive end
        while x < right_bound - 1e-12:
            # Skip discontinuities
            try:
                if hasattr(self.func, 'point_discontinuities') and self.func.point_discontinuities and x in self.func.point_discontinuities:
                    x += step
                    continue
            except Exception:
                pass

            # Evaluate function
            try:
                y = self.func.function(x)
            except Exception:
                y = None

            # Detect asymptotes via model
            asymptote_x = None
            if hasattr(self.func, 'get_vertical_asymptote_between_x'):
                try:
                    asymptote_x = self.func.get_vertical_asymptote_between_x(x, x + step)
                except Exception:
                    asymptote_x = None
            has_vertical_asymptote_in_front = asymptote_x is not None

            if has_vertical_asymptote_in_front:
                expect_asymptote_behind = True
                try:
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

    def build_screen_paths(self):
        if self._should_regenerate():
            screen_paths = self._build_screen_paths_equivalent()
            self._cached_screen_paths = ScreenPolyline(screen_paths)
            self._cache_valid = True
        return self._cached_screen_paths or ScreenPolyline([])

    # --- Equivalent of original Function._generate_paths but producing (x,y) tuples ---
    def _build_screen_paths_equivalent(self):
        # Determine base bounds (model-provided or default math window) and intersect with visible
        visible_left, visible_right = self._get_visible_bounds()
        base_left = getattr(self.func, 'left_bound', None)
        base_right = getattr(self.func, 'right_bound', None)
        if base_left is None or base_right is None:
            base_left = -10 if base_left is None else base_left
            base_right = 10 if base_right is None else base_right
        left_bound = max(visible_left, base_left)
        right_bound = min(visible_right, base_right)

        paths = []
        current_path = []
        expect_asymptote_behind = False

        def calculate_step_size():
            import re, math as _m
            range_width = right_bound - left_bound
            if range_width <= 0:
                return 1.0
            base_step = range_width / 200.0
            # Trig-specific adaptation similar to original
            fs = getattr(self.func, 'function_string', '')
            if 'sin' in fs or 'cos' in fs:
                matches = re.findall(r'(?:sin|cos)\((\d+(?:\.\d+)?)\*?x\)', fs)
                freq_multiplier = float(matches[0]) if matches else 1.0
                period = 2 * _m.pi / freq_multiplier
                visible_periods = range_width / period if period != 0 else 1.0
                if visible_periods <= 1:
                    return base_step
                points_per_period = 400.0 / max(visible_periods, 1e-9)
                points_per_period = min(200.0, points_per_period)
                step_trig = period / points_per_period if points_per_period > 0 else base_step
                # Enforce a global cap by adjusting to at most 200 points across the range
                max_points = 200.0
                cap_step = range_width / max_points
                return max(cap_step, step_trig)
            if any(pattern in fs for pattern in ['x**2', 'x^2']):
                return base_step * 2.0
            # Enforce global cap for non-trig as well
            return base_step

        step = calculate_step_size()
        # Global cap: ensure total samples across range does not exceed 200
        if step > 0:
            max_points = 200.0
            cap_step = (right_bound - left_bound) / max_points
            step = max(cap_step, step)

        # Adaptive step by slope magnitude to densify steep functions
        try:
            eps = max(1e-6, (right_bound - left_bound) / 1000.0)
            cx = (left_bound + right_bound) / 2.0
            y1 = self.func.function(cx - eps)
            y2 = self.func.function(cx + eps)
            if isinstance(y1, (int, float)) and isinstance(y2, (int, float)):
                slope_abs = abs(y2 - y1) / (2.0 * eps) if eps > 0 else 0.0
                if slope_abs > 0:
                    scale = getattr(self.mapper, 'scale_factor', 1.0) or 1.0
                    target_pixel = 2.0
                    desired_step = (target_pixel / scale) / slope_abs
                    # Respect global cap while allowing densification for steep regions
                    step = min(step, max((right_bound - left_bound) / 200.0, desired_step))
        except Exception:
            pass

        # Screen height for bound checks (prefer cartesian2axis.height, fallback to mapper canvas height)
        height = getattr(self.cartesian2axis, 'height', None)
        if not height:
            height = getattr(self.mapper, 'canvas_height', 0) or 0

        def eval_scaled_point(x_val):
            try:
                y_val = self.func.function(x_val)
                sx, sy = self.mapper.math_to_screen(x_val, y_val)
                return (sx, sy), y_val
            except Exception:
                return (None, None)

        x = left_bound
        # Use strict right bound to avoid oversampling
        while x < right_bound - 1e-12:
            # Skip exact point discontinuities
            try:
                if getattr(self.func, 'point_discontinuities', None) and x in self.func.point_discontinuities:
                    x += step
                    continue
            except Exception:
                pass

            scaled_point, y_val = eval_scaled_point(x)
            if not scaled_point[0]:
                x += step
                continue

            # Asymptote ahead
            asymptote_x = None
            if hasattr(self.func, 'get_vertical_asymptote_between_x'):
                try:
                    asymptote_x = self.func.get_vertical_asymptote_between_x(x, x + step)
                except Exception:
                    asymptote_x = None
            has_vertical_asymptote_in_front = asymptote_x is not None
            has_vertical_asymptote_behind = expect_asymptote_behind

            if has_vertical_asymptote_in_front:
                expect_asymptote_behind = True
                new_scaled_point, new_y = eval_scaled_point(asymptote_x - min(1e-3, step / 10))
                if new_scaled_point[0] is not None:
                    scaled_point, y_val = new_scaled_point, new_y

            # Determine neighbor previous scaled point
            if has_vertical_asymptote_behind:
                try:
                    asymptote_x_prev = self.func.get_vertical_asymptote_between_x(x - step, x)
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

            prev_y = neighbor_prev_scaled_point[1] if len(neighbor_prev_scaled_point) > 1 else neighbor_prev_scaled_point[1] if isinstance(neighbor_prev_scaled_point, tuple) else None
            # Use screen y for bound checks
            prev_sy = neighbor_prev_scaled_point[1]
            sy = scaled_point[1]

            # Large pixel jump -> break path
            if abs(prev_sy - sy) > height * 2:
                if current_path:
                    paths.append(current_path)
                    current_path = []
                x += step
                continue

            # Bound crossing detection (using screen coordinates)
            top_bound = 0
            bottom_bound = height
            crosses_top_bound_upward = prev_sy >= top_bound and sy < top_bound
            crosses_top_bound_downward = prev_sy <= top_bound and sy > top_bound
            crosses_bottom_bound_downward = prev_sy <= bottom_bound and sy > bottom_bound
            crosses_bottom_bound_upward = prev_sy >= bottom_bound and sy < bottom_bound

            crossed_bound_onto_screen = crosses_top_bound_downward or crosses_bottom_bound_upward
            crossed_bound_off_screen = crosses_top_bound_upward or crosses_bottom_bound_downward

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

            if x > right_bound:
                break

            # Skip off-screen points
            if sy >= bottom_bound or sy <= top_bound:
                x += step
                continue

            # Add visibly different points
            if not current_path:
                current_path = [(scaled_point[0], scaled_point[1])]
                x += step
                continue

            pixel_diff = 1
            lastx, lasty = current_path[-1]
            visible_x_diff = abs(scaled_point[0] - lastx) > pixel_diff
            visible_y_diff = abs(scaled_point[1] - lasty) > pixel_diff
            if visible_x_diff or visible_y_diff:
                current_path.append((scaled_point[0], scaled_point[1]))

            x += step

        if current_path:
            paths.append(current_path)

        return paths


