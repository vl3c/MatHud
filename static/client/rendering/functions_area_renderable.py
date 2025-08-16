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
                y = f.function(x_math)
                # Filter out NaN and infinite values
                if y is None:
                    return None
                if not isinstance(y, (int, float)):
                    return None
                if isinstance(y, float) and (y != y or abs(y) == float('inf')):
                    return None
                return y
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

    def _generate_pair_paths_screen(self, f1, f2, left, right, num_points):
        if num_points < 2:
            num_points = 2
        dx = (right - left) / (num_points - 1) if num_points > 1 else 1.0
        pairs = []
        for i in range(num_points):
            x_m = left + i * dx
            y1 = self._eval_y_math(f1, x_m)
            y2 = self._eval_y_math(f2, x_m)
            if y1 is None or y2 is None:
                # Keep alignment: insert a break marker so ends do not transpose
                pairs.append((None, None))
                continue
            s1 = self.mapper.math_to_screen(x_m, y1)
            s2 = self.mapper.math_to_screen(x_m, y2)
            pairs.append((s1, s2))
        if not pairs:
            return [], []
        # Trim leading/trailing invalids and split on gaps to avoid transposed joins
        def split_valid(seq):
            chunks = []
            cur = []
            for p in seq:
                if p is None:
                    if cur:
                        chunks.append(cur)
                        cur = []
                else:
                    cur.append(p)
            if cur:
                chunks.append(cur)
            return chunks
        f_seq = [p[0] if p[0] is not None else None for p in pairs]
        g_seq = [p[1] if p[1] is not None else None for p in pairs]
        f_chunks = split_valid(f_seq)
        g_chunks = split_valid(g_seq)
        if not f_chunks or not g_chunks:
            return [], []
        # Use the longest aligned chunk to build area
        idx = max(range(min(len(f_chunks), len(g_chunks))), key=lambda i: min(len(f_chunks[i]), len(g_chunks[i])))
        forward = f_chunks[idx]
        reverse = list(reversed(g_chunks[idx]))
        return forward, reverse

    def build_screen_area(self, num_points=None):
        left, right = self._get_bounds()
        n = num_points if num_points is not None else getattr(self.area, 'num_sample_points', 100)
        fwd, rev = self._generate_pair_paths_screen(self.area.func1, self.area.func2, left, right, n)
        if not fwd or not rev:
            return None
        return ClosedArea(fwd, rev, is_screen=True)


