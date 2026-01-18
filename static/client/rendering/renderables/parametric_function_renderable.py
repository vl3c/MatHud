"""
Parametric Function Renderable for MatHud

Computes screen paths for parametric curves by sampling the parameter t uniformly
and converting math coordinates to screen coordinates. Handles discontinuities
by breaking paths at NaN/Inf values.

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

    def _is_on_screen(self, sy: float, height: float) -> bool:
        """Check if y coordinate is within screen bounds with margin."""
        margin = 50
        return -margin <= sy <= height + margin

    def _is_large_jump(self, prev_sy: float, sy: float, height: float) -> bool:
        """Detect discontinuities by checking for large jumps between points."""
        return abs(prev_sy - sy) > height * 2

    def build_screen_paths(self) -> ScreenPolyline:
        """
        Build screen-space paths for rendering the parametric curve.

        Samples the parameter t uniformly, evaluates x(t) and y(t), converts
        to screen coordinates, and breaks paths at discontinuities (NaN/Inf values
        or large jumps).

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

        _, height = self._get_screen_signature()
        paths: List[List[Tuple[float, float]]] = []
        current_path: List[Tuple[float, float]] = []
        prev_sy: Optional[float] = None

        for t in sample_points:
            point = self._evaluate_point(t)

            if point[0] is None:
                # Invalid point - break the path
                if current_path:
                    paths.append(current_path)
                    current_path = []
                prev_sy = None
                continue

            sx, sy = point

            # Check for large jumps (discontinuity detection)
            if prev_sy is not None and self._is_large_jump(prev_sy, sy, height):
                if current_path:
                    paths.append(current_path)
                    current_path = []

            current_path.append((sx, sy))
            prev_sy = sy

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
