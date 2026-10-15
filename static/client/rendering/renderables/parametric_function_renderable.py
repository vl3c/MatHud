"""
Parametric Function Renderable for MatHud

Computes screen paths for parametric curves by sampling the parameter t uniformly
and converting math coordinates to screen coordinates. Handles discontinuities
by breaking paths at NaN/Inf values and at poles or jumps (a long step whose middle
point is not between its ends, e.g. tan(t) passing pi/2 off screen).

Unlike regular function renderables that sample x uniformly, parametric curves
sample the parameter t uniformly, which naturally handles curves that loop back
or have multiple y values for a single x.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from rendering.primitives import ScreenPolyline


class ParametricFunctionRenderable:
    """
    Renderable for parametric curves that handles path computation and caching.

    Samples the parameter t uniformly from t_min to t_max, evaluates x(t) and y(t),
    and builds screen-space polylines for rendering. Caches results and invalidates
    when the coordinate mapping changes.
    """

    # Default number of sample points for smooth curves
    DEFAULT_SAMPLES: int = 400
    # Pixels beyond a screen edge that still count as on screen
    OFF_SCREEN_MARGIN: float = 50.0

    def __init__(self, parametric_model: Any, coordinate_mapper: Any) -> None:
        """
        Initialize the renderable with a parametric function model.

        Args:
            parametric_model: ParametricFunction drawable with x/y expressions
            coordinate_mapper: CoordinateMapper for math-to-screen conversion
        """
        self.func: Any = parametric_model
        self.mapper: Any = coordinate_mapper
        self._cached_screen_paths: Optional[ScreenPolyline] = None
        self._cache_valid: bool = False
        self._last_scale: Optional[float] = None
        self._last_screen_bounds: Optional[Tuple[int, int]] = None
        self._last_t_range: Optional[Tuple[float, float]] = None

    def invalidate_cache(self) -> None:
        """Invalidate the cached screen paths, forcing regeneration on next build."""
        self._cached_screen_paths = None
        self._cache_valid = False
        self._last_scale = None
        self._last_screen_bounds = None
        self._last_t_range = None

    def _get_screen_signature(self) -> Tuple[int, int]:
        """Get current screen dimensions for cache comparison."""
        screen_width = getattr(self.mapper, "canvas_width", None)
        screen_height = getattr(self.mapper, "canvas_height", None)
        return (int(screen_width or 0), int(screen_height or 0))

    def _get_t_range(self) -> Tuple[float, float]:
        """Get the current t_min and t_max from the parametric function."""
        t_min = getattr(self.func, "t_min", 0.0)
        t_max = getattr(self.func, "t_max", 2 * math.pi)
        return (float(t_min), float(t_max))

    def _should_regenerate(self) -> bool:
        """Determine if cached paths need to be regenerated."""
        current_scale: Optional[float] = getattr(self.mapper, "scale_factor", None)
        screen_signature = self._get_screen_signature()
        t_range = self._get_t_range()

        if self._cached_screen_paths is None or not self._cache_valid:
            self._update_cache_state(current_scale, screen_signature, t_range)
            return True

        if self._last_scale != current_scale:
            self._update_cache_state(current_scale, screen_signature, t_range)
            return True

        if self._last_screen_bounds != screen_signature:
            self._update_cache_state(current_scale, screen_signature, t_range)
            return True

        if self._last_t_range != t_range:
            self._update_cache_state(current_scale, screen_signature, t_range)
            return True

        return False

    def _update_cache_state(
        self,
        scale: Optional[float],
        screen_sig: Tuple[int, int],
        t_range: Tuple[float, float],
    ) -> None:
        """Update cached state for comparison."""
        self._last_scale = scale
        self._last_screen_bounds = screen_sig
        self._last_t_range = t_range

    def _evaluate_point(self, t: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate the parametric function at t and convert to screen coordinates.

        Returns (None, None) if evaluation fails or produces invalid values.
        """
        try:
            x_val = self.func.evaluate_x(t)
            y_val = self.func.evaluate_y(t)

            # Check for invalid values
            if not self._is_valid_number(x_val) or not self._is_valid_number(y_val):
                return (None, None)

            # Convert to screen coordinates
            sx, sy = self.mapper.math_to_screen(x_val, y_val)
            return (sx, sy)
        except Exception:
            return (None, None)

    def _is_valid_number(self, value: Any) -> bool:
        """Check if a value is a valid finite number."""
        if value is None:
            return False
        try:
            fval = float(value)
            return math.isfinite(fval)
        except (TypeError, ValueError):
            return False

    def _generate_sample_points(self) -> List[float]:
        """Generate uniform sample points for the parameter t."""
        t_min, t_max = self._get_t_range()

        if t_max <= t_min:
            return []

        # Calculate number of samples based on screen width for smooth curves
        screen_width, screen_height = self._get_screen_signature()
        num_samples = max(self.DEFAULT_SAMPLES, min(screen_width, 800))

        # Generate uniform samples
        step = (t_max - t_min) / num_samples
        samples: List[float] = []
        t = t_min
        while t <= t_max:
            samples.append(t)
            t += step

        # Ensure we include the endpoint
        if samples and samples[-1] < t_max:
            samples.append(t_max)

        return samples

    def _is_large_jump(self, prev_sy: float, sy: float, height: float) -> bool:
        """Whether a step in y is long enough (over a screen height) to check for a break."""
        return height > 0 and abs(prev_sy - sy) > height

    def _is_large_horizontal_jump(self, prev_sx: float, sx: float, width: float) -> bool:
        """Whether a step in x is long enough (over a screen width) to check for a break."""
        return width > 0 and abs(prev_sx - sx) > width

    def _is_off_screen_on_opposite_sides(
        self, prev: Tuple[float, float], point: Tuple[float, float], width: float, height: float
    ) -> bool:
        """Whether two points lie beyond opposite edges of the screen (above/below or left/right)."""
        margin = self.OFF_SCREEN_MARGIN
        if height > 0 and self._beyond_opposite_ends(prev[1], point[1], -margin, height + margin):
            return True
        return width > 0 and self._beyond_opposite_ends(prev[0], point[0], -margin, width + margin)

    @staticmethod
    def _beyond_opposite_ends(a: float, b: float, low: float, high: float) -> bool:
        return (a < low and b > high) or (b < low and a > high)

    @staticmethod
    def _follows(mid: float, a: float, b: float) -> bool:
        """Whether a middle value lies between two ends (or all three are equal)."""
        return min(a, b) < mid < max(a, b) or a == b == mid

    def _is_break_between(
        self,
        prev_t: float,
        prev: Tuple[float, float],
        t: float,
        point: Tuple[float, float],
        width: float,
        height: float,
    ) -> bool:
        """Detect a discontinuity (a pole such as tan(t) at pi/2, or a jump) between two samples.

        A step is only suspicious when it is long (over a screen width or height) or runs
        between opposite off-screen sides. Such a step is kept when the curve at the middle
        parameter lies between its ends (a steep but continuous crossing); a pole or a jump
        puts the middle point beyond one end, or leaves it undefined.
        """
        suspicious = (
            self._is_large_jump(prev[1], point[1], height)
            or self._is_large_horizontal_jump(prev[0], point[0], width)
            or self._is_off_screen_on_opposite_sides(prev, point, width, height)
        )
        if not suspicious:
            return False
        mid_x, mid_y = self._evaluate_point((prev_t + t) / 2.0)
        if mid_x is None or mid_y is None:
            return True
        return not (self._follows(mid_x, prev[0], point[0]) and self._follows(mid_y, prev[1], point[1]))

    def build_screen_paths(self) -> ScreenPolyline:
        """
        Build screen-space paths for rendering the parametric curve.

        Samples the parameter t uniformly, evaluates x(t) and y(t), converts
        to screen coordinates, and breaks paths at discontinuities: NaN/Inf values, and
        long steps whose middle point is not between their ends (poles and jumps).

        Returns:
            ScreenPolyline containing one or more continuous path segments
        """
        if not self._should_regenerate():
            return self._cached_screen_paths or ScreenPolyline([])

        sample_points = self._generate_sample_points()
        if not sample_points:
            self._cached_screen_paths = ScreenPolyline([])
            self._cache_valid = True
            return self._cached_screen_paths

        width, height = self._get_screen_signature()
        paths: List[List[Tuple[float, float]]] = []
        current_path: List[Tuple[float, float]] = []
        prev_t: Optional[float] = None
        prev_point: Optional[Tuple[float, float]] = None

        for t in sample_points:
            sx, sy = self._evaluate_point(t)

            if sx is None or sy is None:
                # Invalid point - break the path
                if current_path:
                    paths.append(current_path)
                    current_path = []
                prev_t = None
                prev_point = None
                continue

            point = (sx, sy)
            if prev_t is not None and prev_point is not None:
                if self._is_break_between(prev_t, prev_point, t, point, width, height):
                    if current_path:
                        paths.append(current_path)
                        current_path = []

            current_path.append(point)
            prev_t = t
            prev_point = point

        # Add the final path segment
        if current_path:
            paths.append(current_path)

        self._cached_screen_paths = ScreenPolyline(paths)
        self._cache_valid = True
        return self._cached_screen_paths

    def build_math_paths(self) -> List[List[Tuple[float, float]]]:
        """
        Build math-space paths for the parametric curve.

        Returns:
            List of path segments in math coordinates (x, y)
        """
        sample_points = self._generate_sample_points()
        if not sample_points:
            return []

        paths: List[List[Tuple[float, float]]] = []
        current_path: List[Tuple[float, float]] = []

        for t in sample_points:
            try:
                x_val = self.func.evaluate_x(t)
                y_val = self.func.evaluate_y(t)

                if not self._is_valid_number(x_val) or not self._is_valid_number(y_val):
                    if current_path:
                        paths.append(current_path)
                        current_path = []
                    continue

                current_path.append((x_val, y_val))
            except Exception:
                if current_path:
                    paths.append(current_path)
                    current_path = []

        if current_path:
            paths.append(current_path)

        return paths
