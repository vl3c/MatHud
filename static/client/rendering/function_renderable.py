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
        self.cartesian2axis = cartesian2axis or getattr(function_model.canvas, 'cartesian2axis', None)

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
        if not self.cartesian2axis:
            return -10, 10
        return (
            self.cartesian2axis.get_visible_left_bound(),
            self.cartesian2axis.get_visible_right_bound(),
        )

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

        # Step heuristic similar to original
        range_width = right_bound - left_bound
        step = range_width / 200.0

        paths = []
        current_path = []
        x = left_bound
        expect_asymptote_behind = False

        while x <= right_bound + step:
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
            math_poly = self.build_math_paths()
            screen_paths = []
            for path in math_poly.paths:
                if len(path) < 2:
                    continue
                sp = []
                for mx, my in path:
                    sx, sy = self.mapper.math_to_screen(mx, my)
                    sp.append((sx, sy))
                if len(sp) >= 2:
                    screen_paths.append(sp)
            self._cached_screen_paths = ScreenPolyline(screen_paths)
            self._cache_valid = True
        return self._cached_screen_paths or ScreenPolyline([])


