"""
Renderable for FunctionsBoundedColoredArea producing a screen-space ClosedArea.
"""

from rendering.primitives import ClosedArea


class FunctionsBoundedAreaRenderable:
    def __init__(self, area_model, coordinate_mapper):
        self.area = area_model
        self.mapper = coordinate_mapper

    def _is_function_like(self, f):
        return hasattr(f, 'function')

    def _eval_y_math(self, f, x_math):
        if f is None:
            return 0.0
        if isinstance(f, (int, float)):
            return float(f)
        if self._is_function_like(f):
            try:
                return f.function(x_math)
            except Exception:
                return None
        return None

    def _get_bounds(self):
        # Start from model math bounds
        try:
            left, right = self.area._get_bounds()
        except Exception:
            left, right = -10, 10
        # Apply function bounds if present
        for f in (self.area.func1, self.area.func2):
            if hasattr(f, 'left_bound') and hasattr(f, 'right_bound') and f.left_bound is not None and f.right_bound is not None:
                left = max(left, f.left_bound)
                right = min(right, f.right_bound)
        # Apply user bounds
        if getattr(self.area, 'left_bound', None) is not None:
            left = max(left, self.area.left_bound)
        if getattr(self.area, 'right_bound', None) is not None:
            right = min(right, self.area.right_bound)
        # Finally, intersect with visible bounds for rendering
        try:
            vis_left = self.mapper.get_visible_left_bound()
            vis_right = self.mapper.get_visible_right_bound()
            left = max(left, vis_left)
            right = min(right, vis_right)
        except Exception:
            pass
        if left >= right:
            c = (left + right) / 2.0
            left, right = c - 0.1, c + 0.1
        return left, right

    def _generate_path_screen(self, f, left, right, num_points, reverse=False):
        if num_points < 2:
            num_points = 2
        dx = (right - left) / (num_points - 1) if num_points > 1 else 1.0
        idxs = range(num_points - 1, -1, -1) if reverse else range(num_points)
        pts = []
        for i in idxs:
            x_m = left + i * dx
            y_m = self._eval_y_math(f, x_m)
            if y_m is None:
                continue
            sx, sy = self.mapper.math_to_screen(x_m, y_m)
            pts.append((sx, sy))
        return pts

    def build_screen_area(self, num_points=None):
        left, right = self._get_bounds()
        n = num_points if num_points is not None else getattr(self.area, 'num_sample_points', 100)
        fwd = self._generate_path_screen(self.area.func1, left, right, n, reverse=False)
        rev = self._generate_path_screen(self.area.func2, left, right, n, reverse=True)
        if not fwd or not rev:
            return None
        return ClosedArea(fwd, rev, is_screen=True)


