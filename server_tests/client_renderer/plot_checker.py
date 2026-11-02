"""Plot correctness checker for function curves (server-side, pure Python).

Draws a function with the client's FunctionRenderable on a real CoordinateMapper and
compares the drawn polylines with the true curve, both in screen pixels:

- spurious: length of drawn line farther than TOLERANCE_PX from the true curve (a
  connector across a jump or a pole, a wave that is not there);
- missing: length of the true curve farther than TOLERANCE_PX from anything drawn
  (a branch cut short, a peak flattened).

The true curve is traced with steps small enough that consecutive points differ by at
most 1 px vertically; where that cannot be reached by bisection the curve breaks.
"""

from __future__ import annotations

from server_tests import client_renderer  # noqa: F401  (stubs the browser module)

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from coordinate_mapper import CoordinateMapper
from rendering.renderables.function_renderable import FunctionRenderable

WIDTH = 800
HEIGHT = 600
TOLERANCE_PX = 2.0
_CELL = 4.0
_TRUTH_BUDGET = 400_000

Point = Tuple[float, float]
View = Tuple[float, float, float, float]  # left, right, top, bottom


@dataclass
class CurveModel:
    """A function as the renderer sees it, with the analysis lists the client gives."""

    function: Callable[[float], float]
    vertical_asymptotes: List[float] = field(default_factory=list)
    point_discontinuities: List[float] = field(default_factory=list)
    corner_points: List[float] = field(default_factory=list)
    is_periodic: bool = False
    estimated_period: Optional[float] = None
    left_bound: Optional[float] = None
    right_bound: Optional[float] = None

    def get_vertical_asymptote_between_x(self, x1: float, x2: float) -> Optional[float]:
        low, high = min(x1, x2), max(x1, x2)
        for x in self.vertical_asymptotes:
            if low <= x < high:
                return x
        return None


def safe(f: Callable[[float], float]) -> Callable[[float], float]:
    """f with math errors and non-finite values turned into nan (as the client evaluates)."""

    def evaluate(x: float) -> float:
        try:
            y = f(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan
        if isinstance(y, complex) or not math.isfinite(y):
            return math.nan
        return float(y)

    return evaluate


def periodic_asymptotes(offset: float, period: float, limit: float = 1000.0) -> List[float]:
    """offset + k * period within [-limit, limit], as the client lists tan's poles."""
    first = math.ceil((-limit - offset) / period)
    last = math.floor((limit - offset) / period)
    return [offset + k * period for k in range(first, last + 1)]


def draw(model: CurveModel, view: View, view_margin: float = 0.5) -> List[List[Point]]:
    """The screen polylines FunctionRenderable draws for the model in the view (800 x 600)."""
    mapper = CoordinateMapper(WIDTH, HEIGHT)
    mapper.set_visible_bounds(*view)
    renderable = FunctionRenderable(model, mapper)
    renderable.view_margin = view_margin
    return [[(float(x), float(y)) for x, y in path] for path in renderable.build_screen_paths().paths]


def plot_errors(model: CurveModel, f: Callable[[float], float], view: View) -> Tuple[float, float]:
    """(spurious px, missing px) of the model's drawing in the view against the true f."""
    drawn = draw(model, view)
    truth = _truth_runs(safe(f), view)
    drawn_index = _index_segments(drawn)
    truth_index = _index_segments(truth)
    return _length_beyond(drawn, truth_index), _length_beyond(truth, drawn_index)


def _truth_runs(f: Callable[[float], float], view: View) -> List[List[Point]]:
    left, right, top, bottom = view

    def screen_y(sx: float) -> Optional[float]:
        y = f(left + sx / WIDTH * (right - left))
        return None if math.isnan(y) else (top - y) / (top - bottom) * HEIGHT

    budget = [_TRUTH_BUDGET]
    runs: List[List[Point]] = []
    current: List[Point] = []

    def end_run() -> None:
        nonlocal current
        if len(current) > 1:
            runs.append(current)
        current = []

    def add(sx: float, sy: Optional[float]) -> None:
        if sy is None:
            end_run()
        else:
            current.append((sx, sy))

    def visible(sy: Optional[float]) -> bool:
        return sy is not None and -2 <= sy <= HEIGHT + 2

    def refine(a: float, ya: Optional[float], c: float, yc: Optional[float], depth: int) -> None:
        """Add the truth from a (exclusive) to c (inclusive)."""
        if budget[0] <= 0 or (ya is None and yc is None):
            add(c, yc)
            return
        if ya is not None and yc is not None:
            same_side_off = not visible(ya) and not visible(yc) and (ya < 0) == (yc < 0)
            if abs(ya - yc) <= 1.0 or same_side_off:
                add(c, yc)
                return
        if depth >= 40 or c - a < 1e-9:
            if not _shrinks_when_narrowed(screen_y, a, ya, c, yc):
                end_run()
            add(c, yc)
            return
        m = (a + c) / 2
        ym = screen_y(m)
        budget[0] -= 1
        refine(a, ya, m, ym, depth + 1)
        refine(m, ym, c, yc, depth + 1)

    sx, y0 = 0.0, screen_y(0.0)
    add(sx, y0)
    while sx < WIDTH:
        nx = min(float(WIDTH), sx + 0.25)
        refine(sx, current[-1][1] if current and current[-1][0] == sx else screen_y(sx), nx, screen_y(nx), 0)
        sx = nx
    end_run()
    return runs


def _shrinks_when_narrowed(
    screen_y: Callable[[float], Optional[float]], a: float, ya: Optional[float], c: float, yc: Optional[float]
) -> bool:
    """Whether a step left at the bisection limit is a steep continuous curve, not a jump.

    Across a jump or a pole the step stays as tall however narrow the interval; across
    a steep continuous curve (x^(1/7) at 0) it keeps shrinking, so compare it with the
    step across an interval 2^20 times wider around the same point.
    """
    if ya is None or yc is None:
        return False
    half_wide = (c - a) * 2.0**19
    wide_a, wide_c = screen_y(a - half_wide), screen_y(c + half_wide)
    if wide_a is None or wide_c is None:
        return False
    return abs(yc - ya) < 0.5 * abs(wide_c - wide_a)


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    u = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(ax + u * dx - px, ay + u * dy - py)


_Index = Tuple[Dict[Tuple[int, int], List[int]], List[Tuple[float, float, float, float]]]


def _index_segments(polylines: Sequence[Sequence[Point]]) -> _Index:
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    segments: List[Tuple[float, float, float, float]] = []
    for line in polylines:
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            clipped = _clip_to_view(ax, ay, bx, by, margin=10.0)
            if clipped is None:
                continue
            index = len(segments)
            segments.append((ax, ay, bx, by))
            cax, cay, cbx, cby = clipped
            x0, x1 = sorted((cax, cbx))
            y0, y1 = sorted((cay, cby))
            for gx in range(int(math.floor(x0 / _CELL)) - 1, int(math.floor(x1 / _CELL)) + 2):
                for gy in range(int(math.floor(y0 / _CELL)) - 1, int(math.floor(y1 / _CELL)) + 2):
                    grid[(gx, gy)].append(index)
    return grid, segments


def _clip_to_view(
    ax: float, ay: float, bx: float, by: float, margin: float = 0.0
) -> Optional[Tuple[float, float, float, float]]:
    top, bottom = -margin, HEIGHT + margin
    if (ay < top and by < top) or (ay > bottom and by > bottom):
        return None

    def x_at(y: float) -> float:
        return ax + (bx - ax) * (y - ay) / (by - ay)

    if ay < top:
        ax, ay = x_at(top), top
    elif ay > bottom:
        ax, ay = x_at(bottom), bottom
    if by < top:
        bx, by = x_at(top), top
    elif by > bottom:
        bx, by = x_at(bottom), bottom
    return ax, ay, bx, by


def _length_beyond(polylines: Sequence[Sequence[Point]], index: _Index, step_px: float = 0.5) -> float:
    """Length (px) of the polylines, within the view, farther than TOLERANCE_PX from the index."""
    grid, segments = index
    total = 0.0
    for line in polylines:
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            clipped = _clip_to_view(ax, ay, bx, by)
            if clipped is None:
                continue
            ax, ay, bx, by = clipped
            length = math.hypot(bx - ax, by - ay)
            count = max(1, int(length / step_px))
            for k in range(count):
                u = (k + 0.5) / count
                px, py = ax + (bx - ax) * u, ay + (by - ay) * u
                if not (0 <= px <= WIDTH and 0 <= py <= HEIGHT):
                    continue
                cell = (int(math.floor(px / _CELL)), int(math.floor(py / _CELL)))
                nearest = min((_segment_distance(px, py, *segments[i]) for i in grid.get(cell, ())), default=math.inf)
                if nearest > TOLERANCE_PX:
                    total += length / count
    return round(total, 1)
