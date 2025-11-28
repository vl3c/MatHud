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
from rendering.renderables.curve_step_calculator import PixelStepCalculator

SCREEN_MARGIN: float = 16.0


class FunctionRenderable:
    def __init__(self, function_model: Any, coordinate_mapper: Any) -> None:
        self.func: Any = function_model
        self.mapper: Any = coordinate_mapper
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
            left: float = self.mapper.get_visible_left_bound()
            right: float = self.mapper.get_visible_right_bound()
            return left, right
        except Exception:
            return -10, 10

    def _get_screen_signature(self) -> Tuple[int, int]:
        screen_width = getattr(self.mapper, "canvas_width", None)
        screen_height = getattr(self.mapper, "canvas_height", None)
        return (int(screen_width or 0), int(screen_height or 0))

    def _update_cache_state(self, scale: Optional[float], bounds: Tuple[float, float], screen_sig: Tuple[int, int]) -> None:
        self._last_scale = scale
        self._last_bounds = bounds
        self._last_screen_bounds = screen_sig

    def _should_regenerate(self) -> bool:
        current_scale: Optional[float] = getattr(self.mapper, 'scale_factor', None)
        current_bounds: Tuple[float, float] = self._get_visible_bounds()
        screen_signature = self._get_screen_signature()
        if self._cached_screen_paths is None or not self._cache_valid:
            self._update_cache_state(current_scale, current_bounds, screen_signature)
            return True
        if self._last_bounds != current_bounds:
            self._update_cache_state(current_scale, current_bounds, screen_signature)
            return True
        if self._last_scale != current_scale:
            self._update_cache_state(current_scale, current_bounds, screen_signature)
            return True
        if self._last_screen_bounds != screen_signature:
            self._update_cache_state(current_scale, current_bounds, screen_signature)
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
        width: float = getattr(self.mapper, 'canvas_width', 0) or 0
        height: float = getattr(self.mapper, 'canvas_height', 0) or 0
        return width, height

    def _calculate_step(self, left_bound: float, right_bound: float) -> float:
        _, screen_height = self._get_screen_dimensions()
        return PixelStepCalculator.calculate(
            left_bound, right_bound, self.func.function,
            self.mapper.math_to_screen, screen_height
        )

    def _eval_scaled_point(self, x_val: float) -> Tuple[Tuple[Optional[float], Optional[float]], Any]:
        try:
            y_val: Any = self.func.function(x_val)
            sx, sy = self.mapper.math_to_screen(x_val, y_val)
            return (sx, sy), y_val
        except Exception:
            return (None, None), None


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


    def _build_screen_paths_equivalent(self) -> list[list[tuple[float, float]]]:
        left_bound, right_bound = self._get_effective_bounds()
        width, height = self._get_screen_dimensions()
        step = self._calculate_step(left_bound, right_bound)
        visible_min_x: float = -SCREEN_MARGIN
        visible_max_x: float = width + SCREEN_MARGIN

        paths: list[list[tuple[float, float]]] = []
        current_path: list[tuple[float, float]] = []
        
        # Cache previous point to avoid redundant function evaluation
        prev_scaled_point: Optional[Tuple[Optional[float], Optional[float]]] = None

        x: float = left_bound
        while x < right_bound - 1e-12:
            if self._is_discontinuity(x):
                prev_scaled_point = None
                x += step
                continue

            scaled_point, y_val = self._eval_scaled_point(x)
            if scaled_point[0] is None:
                prev_scaled_point = None
                x += step
                continue

            # Check for asymptote ahead
            asymptote_x = self._get_asymptote_between(x, x + step)
            if asymptote_x is not None:
                new_scaled_point, new_y = self._eval_scaled_point(asymptote_x - min(1e-3, step / 10))
                if new_scaled_point[0] is not None:
                    scaled_point, y_val = new_scaled_point, new_y
                prev_scaled_point = None

            sx_val, sy = scaled_point[0], scaled_point[1]

            # Use cached previous point instead of re-evaluating
            if prev_scaled_point is not None and prev_scaled_point[0] is not None:
                prev_sx, prev_sy = prev_scaled_point[0], prev_scaled_point[1]

                if self._is_large_jump(prev_sy, sy, height):
                    if current_path:
                        paths.append(current_path)
                        current_path = []
                    prev_scaled_point = scaled_point
                    x += step
                    continue

                boundary_result = self._handle_boundary_crossing(
                    x, left_bound, right_bound, width, height,
                    prev_sx, prev_sy, sx_val, sy,
                    prev_scaled_point, scaled_point, current_path, paths
                )
                if boundary_result is not None:
                    current_path, paths, should_continue = boundary_result
                    if should_continue:
                        prev_scaled_point = scaled_point
                        x += step
                        continue

            if x > right_bound:
                break

            if not self._is_point_visible(sx_val, sy, width, height, visible_min_x, visible_max_x):
                if current_path:
                    paths.append(current_path)
                    current_path = []
                prev_scaled_point = scaled_point
                x += step
                continue

            current_path.append((scaled_point[0], scaled_point[1]))
            prev_scaled_point = scaled_point

            x += step

        if current_path:
            paths.append(current_path)

        self._extend_paths_to_boundaries(paths, width, height, step, left_bound, right_bound)
        return paths

    def _extend_paths_to_boundaries(
        self, paths: list[list[tuple[float, float]]], width: float, height: float, step: float,
        left_bound: float, right_bound: float
    ) -> None:
        """
        Ensures each sub-path extends to screen boundaries for complete rendering.
        """
        for path in paths:
            if len(path) < 2:
                continue
            self._extend_path_start(path, width, height, step, left_bound)
            self._extend_path_end(path, width, height, step, right_bound)

    def _extend_path_start(
        self, path: list[tuple[float, float]], width: float, height: float, step: float, left_bound: float
    ) -> None:
        """Extend or clamp the start of a path to reach the screen boundary."""
        sx, sy = path[0]
        math_x, _ = self.mapper.screen_to_math(sx, sy)
        if abs(math_x - left_bound) < 0.01:
            return
        
        if self._is_inside_screen(sx, sy, width, height):
            ext_pt = self._sample_extension_backward(sx, sy, step, left_bound, height)
            if self._is_usable_sample(ext_pt, sy, height):
                path.insert(0, ext_pt)
            else:
                ext_pt = self._get_extrapolated_start(path, height, left_bound)
                if ext_pt is not None:
                    path.insert(0, ext_pt)
        elif self._is_outside_screen_y(sy, height) and len(path) >= 2:
            x2, y2 = path[1]
            path[0] = self._clamp_to_boundary(x2, y2, sx, sy, height)

    def _extend_path_end(
        self, path: list[tuple[float, float]], width: float, height: float, step: float, right_bound: float
    ) -> None:
        """Extend or clamp the end of a path to reach the screen boundary."""
        sx, sy = path[-1]
        math_x, _ = self.mapper.screen_to_math(sx, sy)
        if abs(math_x - right_bound) < 0.01:
            return
        
        if self._is_inside_screen(sx, sy, width, height):
            ext_pt = self._sample_extension_forward(sx, sy, step, right_bound, height)
            if self._is_usable_sample(ext_pt, sy, height):
                path.append(ext_pt)
            else:
                ext_pt = self._get_extrapolated_end(path, height, right_bound)
                if ext_pt is not None:
                    path.append(ext_pt)
        elif self._is_outside_screen_y(sy, height) and len(path) >= 2:
            x1, y1 = path[-2]
            path[-1] = self._clamp_to_boundary(x1, y1, sx, sy, height)

    def _is_usable_sample(self, ext_pt: Optional[tuple[float, float]], origin_y: float, height: float) -> bool:
        """Check if sampled extension point is valid and reaches boundary."""
        if ext_pt is None:
            return False
        return self._is_valid_extension(origin_y, ext_pt[1], height) and self._is_at_boundary(ext_pt[1], height)

    def _get_extrapolated_start(
        self, path: list[tuple[float, float]], height: float, left_bound: float
    ) -> Optional[tuple[float, float]]:
        """Extrapolate to boundary and clamp to left bound."""
        inner_pt = path[1] if len(path) > 1 else path[0]
        ext_pt = self._extrapolate_to_boundary(path[0], inner_pt, height, backward=True)
        if ext_pt is not None:
            ext_pt = self._clamp_to_left_bound(ext_pt, left_bound)
        return ext_pt

    def _get_extrapolated_end(
        self, path: list[tuple[float, float]], height: float, right_bound: float
    ) -> Optional[tuple[float, float]]:
        """Extrapolate to boundary and clamp to right bound."""
        inner_pt = path[-2] if len(path) > 1 else path[-1]
        ext_pt = self._extrapolate_to_boundary(path[-1], inner_pt, height, backward=False)
        if ext_pt is not None:
            ext_pt = self._clamp_to_right_bound(ext_pt, right_bound)
        return ext_pt

    def _clamp_to_left_bound(self, pt: tuple[float, float], left_bound: float) -> Optional[tuple[float, float]]:
        """Clamp point to left bound if it extends past it."""
        math_x, _ = self.mapper.screen_to_math(pt[0], pt[1])
        if math_x >= left_bound:
            return pt  # Within bounds
        # Clamp to left_bound - evaluate function at left_bound
        clamped_pt, _ = self._eval_scaled_point(left_bound)
        if clamped_pt[0] is not None:
            return (clamped_pt[0], clamped_pt[1])
        return None

    def _clamp_to_right_bound(self, pt: tuple[float, float], right_bound: float) -> Optional[tuple[float, float]]:
        """Clamp point to right bound if it extends past it."""
        math_x, _ = self.mapper.screen_to_math(pt[0], pt[1])
        if math_x <= right_bound:
            return pt  # Within bounds
        # Clamp to right_bound - evaluate function at right_bound
        clamped_pt, _ = self._eval_scaled_point(right_bound)
        if clamped_pt[0] is not None:
            return (clamped_pt[0], clamped_pt[1])
        return None

    def _is_at_boundary(self, y: float, height: float) -> bool:
        """Check if y is at or beyond screen boundary."""
        return y <= 0 or y >= height

    def _extrapolate_to_boundary(
        self, edge_pt: tuple[float, float], inner_pt: tuple[float, float], height: float, backward: bool
    ) -> Optional[tuple[float, float]]:
        """
        Extrapolate from path direction to hit the appropriate boundary.
        edge_pt: the endpoint we're extending from
        inner_pt: the next point in the path (gives us direction)
        backward: True if extending start (go opposite direction), False if extending end
        """
        x1, y1 = edge_pt
        x2, y2 = inner_pt
        
        # Direction from inner to edge (the way the path is going)
        dx = x1 - x2
        dy = y1 - y2
        
        if abs(dy) < 1e-9:
            return None  # Horizontal line, no vertical boundary to hit
        
        # Determine which boundary to hit based on direction
        if backward:
            # Extending start: continue in the direction the path came from
            # If dy > 0, path is going down (y increasing), so continue down to y=height
            # If dy < 0, path is going up (y decreasing), so continue up to y=0
            target_y = height if dy > 0 else 0.0
        else:
            # Extending end: continue in the direction the path is going
            target_y = height if dy > 0 else 0.0
        
        # Calculate intersection
        t = (target_y - y1) / dy
        if t < 0:
            return None  # Boundary is behind us
        
        new_x = x1 + t * dx
        return (new_x, target_y)

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

    def _sample_extension_backward(
        self, sx: float, sy: float, step: float, left_bound: float, height: float
    ) -> Optional[tuple[float, float]]:
        """Sample one step backward, respecting bounds and asymptotes."""
        math_x, _ = self.mapper.screen_to_math(sx, sy)
        # Check for asymptote within a few steps (to extend toward it)
        search_left = max(left_bound, math_x - 3 * step)
        asymptote_x = self._get_asymptote_between(search_left, math_x)
        if asymptote_x is not None:
            # Sample just past the asymptote on our side
            sample_x = asymptote_x + min(1e-3, step / 10)
        else:
            sample_x = max(left_bound, math_x - step)
        if sample_x >= math_x:
            return None
        prev_pt, _ = self._eval_scaled_point(sample_x)
        if prev_pt[0] is None:
            return None
        return self._clamp_to_boundary(sx, sy, prev_pt[0], prev_pt[1], height)

    def _sample_extension_forward(
        self, sx: float, sy: float, step: float, right_bound: float, height: float
    ) -> Optional[tuple[float, float]]:
        """Sample one step forward, respecting bounds and asymptotes."""
        math_x, _ = self.mapper.screen_to_math(sx, sy)
        # Check for asymptote within a few steps (to extend toward it)
        search_right = min(right_bound, math_x + 3 * step)
        asymptote_x = self._get_asymptote_between(math_x, search_right)
        if asymptote_x is not None:
            # Sample just before the asymptote on our side
            sample_x = asymptote_x - min(1e-3, step / 10)
        else:
            sample_x = min(right_bound, math_x + step)
        if sample_x <= math_x:
            return None
        next_pt, _ = self._eval_scaled_point(sample_x)
        if next_pt[0] is None:
            return None
        return self._clamp_to_boundary(sx, sy, next_pt[0], next_pt[1], height)

    def _clamp_to_boundary(
        self, x1: float, y1: float, x2: float, y2: float, height: float
    ) -> tuple[float, float]:
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

