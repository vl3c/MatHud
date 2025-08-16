"""
Renderable for SegmentsBoundedColoredArea producing a screen-space ClosedArea.

This keeps math models pure and provides renderer-agnostic geometry.
"""

from rendering.primitives import ClosedArea


class SegmentsBoundedAreaRenderable:
    def __init__(self, area_model, coordinate_mapper):
        self.area = area_model
        self.mapper = coordinate_mapper

    def _screen_xy(self, point):
        # Convert math coordinates to screen using the mapper exclusively
        if point is None or not hasattr(point, 'x') or not hasattr(point, 'y'):
            return None, None
        return self.mapper.math_to_screen(point.x, point.y)

    def _get_y_at_x_screen(self, segment, x):
        x1, y1 = self._screen_xy(segment.point1)
        x2, y2 = self._screen_xy(segment.point2)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return None
        if x2 == x1:
            return y1
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def build_screen_area(self):
        if not getattr(self.area, 'segment2', None):
            p1 = self._screen_xy(self.area.segment1.point1)
            p2 = self._screen_xy(self.area.segment1.point2)
            if None in p1 or None in p2:
                return None
            # Use mapper to compute x-axis (y=0) at the same screen x positions
            xaxis_screen_at_p2 = self.mapper.math_to_screen(self.mapper.screen_to_math(p2[0], p2[1])[0], 0)[1]
            xaxis_screen_at_p1 = self.mapper.math_to_screen(self.mapper.screen_to_math(p1[0], p1[1])[0], 0)[1]
            reverse_points = [(p2[0], xaxis_screen_at_p2), (p1[0], xaxis_screen_at_p1)]
            return ClosedArea([p1, p2], reverse_points, is_screen=True)

        x11, _ = self._screen_xy(self.area.segment1.point1)
        x12, _ = self._screen_xy(self.area.segment1.point2)
        x21, _ = self._screen_xy(self.area.segment2.point1)
        x22, _ = self._screen_xy(self.area.segment2.point2)
        if None in (x11, x12, x21, x22):
            return None
        x1_min, x1_max = min(x11, x12), max(x11, x12)
        x2_min, x2_max = min(x21, x22), max(x21, x22)
        overlap_min = max(x1_min, x2_min)
        overlap_max = min(x1_max, x2_max)
        if overlap_max <= overlap_min:
            return None
        y1_start = self._get_y_at_x_screen(self.area.segment1, overlap_min)
        y1_end = self._get_y_at_x_screen(self.area.segment1, overlap_max)
        y2_start = self._get_y_at_x_screen(self.area.segment2, overlap_min)
        y2_end = self._get_y_at_x_screen(self.area.segment2, overlap_max)
        if None in (y1_start, y1_end, y2_start, y2_end):
            return None
        points = [(overlap_min, y1_start), (overlap_max, y1_end)]
        reverse_points = [(overlap_max, y2_end), (overlap_min, y2_start)]
        return ClosedArea(points, reverse_points, is_screen=True)


