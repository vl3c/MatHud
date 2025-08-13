"""
Renderable for FunctionSegmentBoundedColoredArea producing a screen-space ClosedArea.
"""

from rendering.primitives import ClosedArea


class FunctionSegmentAreaRenderable:
    def __init__(self, area_model, coordinate_mapper):
        self.area = area_model
        self.mapper = coordinate_mapper

    def _get_bounds(self):
        seg_left, seg_right = self.area._get_segment_bounds()
        func = self.area.func
        if hasattr(func, 'left_bound') and hasattr(func, 'right_bound'):
            return max(seg_left, func.left_bound), min(seg_right, func.right_bound)
        return seg_left, seg_right

    def _eval_function(self, x_math):
        func = self.area.func
        if func is None:
            return 0.0
        if isinstance(func, (int, float)):
            return float(func)
        if hasattr(func, 'function'):
            try:
                return func.function(x_math)
            except Exception:
                return None
        return None

    def _generate_function_points_screen(self, left_bound, right_bound, num_points):
        if num_points < 2:
            num_points = 2
        dx = (right_bound - left_bound) / (num_points - 1) if num_points > 1 else 1.0
        pts = []
        for i in range(num_points):
            x_m = left_bound + i * dx
            y_m = self._eval_function(x_m)
            if y_m is None:
                continue
            sx, sy = self.mapper.math_to_screen(x_m, y_m)
            pts.append((sx, sy))
        return pts

    def _segment_reverse_points_screen(self):
        p1 = self.area.segment.point1
        p2 = self.area.segment.point2
        # Prefer screen fields if present
        x1 = getattr(p1, 'screen_x', getattr(p1, 'x', None))
        y1 = getattr(p1, 'screen_y', getattr(p1, 'y', None))
        x2 = getattr(p2, 'screen_x', getattr(p2, 'x', None))
        y2 = getattr(p2, 'screen_y', getattr(p2, 'y', None))
        return [(x2, y2), (x1, y1)]

    def build_screen_area(self, num_points=100):
        left_bound, right_bound = self._get_bounds()
        forward = self._generate_function_points_screen(left_bound, right_bound, num_points)
        reverse_points = self._segment_reverse_points_screen()
        if not forward or not reverse_points:
            return None
        return ClosedArea(forward, reverse_points, is_screen=True)


