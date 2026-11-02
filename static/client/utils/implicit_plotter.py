"""Plot the zero set of g(x, y), or the region g(x, y) < 0, by marching squares.

``plot(g, rect, cell)`` lays a uniform grid of square cells over ``rect`` (math
coordinates) and evaluates g only where its zero set can be:

1. Coarse pass: g at the corners and centre of every block of ``block`` × ``block`` cells.
   A block is searched when its samples change sign (a NaN counts as a sign of its own),
   and, after those (of every factor), while the budget lasts, when g comes closer to zero
   than half its variation over the block (a tangency or a thin loop between samples).
2. Fine pass: every node of a searched block, and marching squares on its cells. A
   crossing on an edge is placed by linear interpolation and one regula falsi step. It
   is a pole, not a zero (y = 1/x at x = 0), when |g| at the interpolated point is larger
   than at an end of the edge and the regula falsi step does not bring it closer to zero. A saddle cell (corners alternating in sign) is decided by its centre.
3. Following: when a cell's contour leaves its block, the neighbouring block is searched
   too, so the curve is followed into blocks whose samples did not show it.

Every cell the curve passes through has the same size, so each crossing is shared by
the two cells beside its edge and segments are linked by edge, never by comparing
coordinates.

With ``fill=True`` the result also has the boundary rings of {g < 0} within ``rect``:
segments are oriented with the region on their left, the border of ``rect`` is walked
counter-clockwise where it is inside, and the chains close into rings (outer rings
counter-clockwise, holes clockwise). A point where g is undefined (NaN) is outside the
region; a boundary between the region and such points is found by bisection on the
edge and is not stroked, as are poles and the border of ``rect``.

Budget: at most ``max_evals`` evaluations and, with a ``clock``, ``deadline`` seconds.
A curve whose sign-changing blocks would cost more than the budget is plotted on a
coarser grid (up to 8 times coarser). When the budget runs out before every
sign-changing block is searched, the search stops and the result has ``complete`` False;
running out among the blocks a zero may only be near does not count.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

NAN = float("nan")
INF = float("inf")

# Crossing kinds: a zero of g (stroked), a pole of g, the edge of g's domain, the border of rect
ZERO = 0
POLE = 1
DOMAIN = 2
BORDER = 3

DEFAULT_BLOCK = 16
DEFAULT_MAX_EVALS = 60000
# Bisection steps placing the edge of g's domain on a cell edge (to 1/64 of a cell)
_DOMAIN_BISECTIONS = 6
# Halvings that tell a pole from a zero next to a kink or a node
_POLE_BISECTIONS = 6
# A bracket end that moved and grew more than this past its edge end is near a pole (a
# simple pole at least doubles it: each move halves the end's distance to the pole)
_POLE_GROWTH = 1.5
# Moves of the compass search for a small loop inside a maybe block
_DESCENT_MOVES = 24
# Cells per side above which the grid is refused (the caller picks a coarser cell)
MAX_CELLS_PER_SIDE = 4000

# Fraction of a cell the grid is shifted by, so nodes rarely land exactly on x = 0 or
# y = 0, where many functions are undefined (1/x) or singular (sin(1/x))
_PHASE = 0.3183098861837907


class PlotResult:
    """What one plot gives.

    Attributes:
        curves: Polylines of the zero set (with fill, its stroked stretches), in math
            coordinates; a closed one ends where it starts.
        curve_atoms: For each polyline, the value ``atom_of`` gave at its first point
            (with fill and an ``atom_of``; else 0), for the boundary style.
        rings: With fill, the boundary rings of {g < 0} within rect, region on the left.
        complete: False when the budget stopped the search.
        evaluations: Calls of g (and of atom_of) made.
        cell: The cell size used: the one asked for, or up to 8 times coarser when the
            curve was too dense for the budget at that size.
        asked_cell: The cell size asked for.
    """

    __slots__ = ("curves", "curve_atoms", "rings", "complete", "evaluations", "cell", "asked_cell")

    def __init__(self) -> None:
        self.curves: List[List[Point]] = []
        self.curve_atoms: List[int] = []
        self.rings: List[List[Point]] = []
        self.complete: bool = True
        self.evaluations: int = 0
        self.cell: float = 0.0
        self.asked_cell: float = 0.0


class _Budget:
    """Evaluations left and the deadline, shared by the factors of one plot."""

    __slots__ = ("max_evals", "evals_left", "clock", "deadline", "spent", "timed_out")

    def __init__(self, max_evals: int, clock: Optional[Callable[[], float]], deadline: Optional[float]) -> None:
        self.max_evals = max_evals
        self.evals_left = max_evals
        self.clock = clock
        self.deadline = None if clock is None or deadline is None else clock() + deadline
        self.spent = False
        self.timed_out = False

    def exhausted(self) -> bool:
        if self.spent:
            return True
        if self.evals_left <= 0:
            self.spent = True
        elif self.deadline is not None and self.clock is not None and self.clock() > self.deadline:
            self.spent = self.timed_out = True
        return self.spent

    def retry(self, evals: int) -> None:
        """More evaluations, for a coarser try after the evaluations ran out."""
        self.evals_left = evals
        self.spent = False


def grid_origin(low: float, cell: float, block: int, phase: Optional[float] = None) -> Tuple[int, float]:
    """The first block-aligned node index at or below ``low`` and the grid's phase.

    Nodes are at phase + k * cell for whole k, with block corners at multiples of block,
    so two plots of the same function at the same cell size share their nodes.
    """
    if phase is None:
        phase = _PHASE * cell
    index = int(math.floor((low - phase) / cell))
    index -= index % block
    return index, phase


def plot(
    g: Callable[[float, float], float],
    rect: Tuple[float, float, float, float],
    cell: float,
    *,
    fill: bool = False,
    block: int = DEFAULT_BLOCK,
    max_evals: int = DEFAULT_MAX_EVALS,
    clock: Optional[Callable[[], float]] = None,
    deadline: Optional[float] = None,
    atom_of: Optional[Callable[[float, float], int]] = None,
    factors: Optional[Sequence[Callable[[float, float], float]]] = None,
) -> PlotResult:
    """Plot g = 0 (and with fill, the region g < 0) over rect = (x_min, y_min, x_max, y_max).

    Args:
        g: The function; it may raise or return NaN where undefined.
        rect: The area to plot, in math coordinates.
        cell: The side of a grid cell, in math units.
        fill: Also return the boundary rings of {g < 0}.
        block: Cells per side of a coarse block (even, at least 2).
        max_evals: Evaluation budget, shared by all factors.
        clock: Seconds now, for the deadline (None: no deadline).
        deadline: Seconds the plot may take.
        atom_of: With fill, the index of the part of g that gives its value at (x, y)
            (for and/or regions: the inequality whose boundary it is).
        factors: Without fill, plot these functions' zero sets instead of g's (the
            factors of a product; g is then unused).

    Raises:
        ValueError: If rect is empty, cell is not positive, or the grid is too fine.
    """
    x_min, y_min, x_max, y_max = (float(value) for value in rect)
    if not (cell > 0 and math.isfinite(cell)) or not (x_min < x_max and y_min < y_max):
        raise ValueError("plot needs a non-empty rectangle and a positive cell size.")
    if block < 2 or block % 2:
        raise ValueError("block must be an even number of cells, at least 2.")
    if _layout(x_min, y_min, x_max, y_max, cell, block) is None:
        raise ValueError("The grid is too fine for the rectangle; use a larger cell.")
    budget = _Budget(max_evals, clock, deadline)
    result = PlotResult()
    result.cell = result.asked_cell = cell
    functions = list(factors) if (factors and not fill) else [g]
    # Every function's sure blocks first; the blocks where a zero may be share what is left
    grids: List[_Grid] = []
    for function in functions:
        whole = g if len(functions) > 1 else None
        grid = _search_function(function, (x_min, y_min, x_max, y_max), cell, block, budget, fill, atom_of, whole)
        grids.append(grid)
        result.cell = max(result.cell, grid.cell)
        if budget.spent:
            result.complete = False
            break
    if result.complete:
        for grid in grids:
            grid.search_maybe()
    for grid in grids:
        if fill:
            grid.add_region(result)
        else:
            grid.add_curves(result)
        result.evaluations += grid.evaluations
    return result


# Evaluations a sure block costs per cell of its side, roughly (its perimeter, and the
# cells and crossings the curve passes through): about 5 to 6 measured on conics and
# periodic curves, with some room; an estimate that falls short gets one coarser retry
_COST_PER_BLOCK_CELL = 8
# A curve too dense for the budget at the asked cell size is plotted at most this much coarser
_MAX_COARSENING = 8


def _search_function(
    function: Callable[[float, float], float],
    rect: Tuple[float, float, float, float],
    cell: float,
    block: int,
    budget: _Budget,
    fill: bool,
    atom_of: Optional[Callable[[float, float], int]],
    whole: Optional[Callable[[float, float], float]] = None,
) -> "_Grid":
    """Search one function, at a coarser cell when the curve is too dense for the budget
    (sin(x*y) = 0.5 over a wide view): a complete coarser plot is better than parts missing.

    The blocks are classified once, on the asked grid. A grid 2, 4 or 8 times coarser has
    the same blocks (fewer cells each, same phase), so it reuses that classification: a
    coarser classification would miss periodic curves whose period is near a block's size
    (sin(y) = 0.5), and cost a second coarse pass.
    """
    phase = _PHASE * cell

    def make_grid(factor: int) -> "_Grid":
        block_cells = block // factor
        layout = _layout(rect[0], rect[1], rect[2], rect[3], cell * factor, block_cells, phase)
        assert layout is not None
        grid = _Grid(function, layout, cell * factor, block_cells, budget)
        grid.atom_of = atom_of if fill else None
        grid.whole = whole
        grid.fill = fill
        return grid

    grid = make_grid(1)
    sure, maybe = grid.coarse(fill)
    if budget.spent:
        return grid
    # A sure block costs about its cells per side times the constant: 1 / factor of it
    needed = len(sure) * block * _COST_PER_BLOCK_CELL
    factor = 1
    while needed > factor * budget.evals_left and _can_coarsen(block, factor):
        factor *= 2
    if factor > 1:
        grid = make_grid(factor)
    grid.search(sure, maybe)
    if budget.spent and not budget.timed_out and _can_coarsen(block, factor):
        # The estimate was short: one try twice as coarse, with half the budget again (the
        # deadline bounds the time either way)
        budget.retry(budget.max_evals // 2)
        grid = make_grid(2 * factor)
        grid.search(sure, maybe)
    return grid


def _can_coarsen(block: int, factor: int) -> bool:
    """Whether a grid twice as coarse is allowed: at most _MAX_COARSENING times the asked
    cell, with blocks of an even number of cells, at least 2."""
    coarser = 2 * factor
    cells = block // coarser
    return coarser <= _MAX_COARSENING and block % coarser == 0 and cells >= 2 and cells % 2 == 0


def _layout(
    x_min: float, y_min: float, x_max: float, y_max: float, cell: float, block: int, phase: Optional[float] = None
) -> Optional[Tuple[int, int, float, int, int]]:
    """(first node index along x, along y, phase, blocks along x, along y) of the
    block-aligned grid covering the rect, or None when it would have more than
    MAX_CELLS_PER_SIDE cells on a side. Node (i, j) is at phase + (i0 + i) * cell."""
    i0, phase = grid_origin(x_min, cell, block, phase)
    j0, _ = grid_origin(y_min, cell, block, phase)
    blocks_i = max(1, int(math.ceil((x_max - (phase + i0 * cell)) / (cell * block))))
    blocks_j = max(1, int(math.ceil((y_max - (phase + j0 * cell)) / (cell * block))))
    if max(blocks_i, blocks_j) * block > MAX_CELLS_PER_SIDE:
        return None
    return i0, j0, phase, blocks_i, blocks_j


class _Grid:
    """One function on one grid: node values, crossings and segments."""

    def __init__(
        self,
        g: Callable[[float, float], float],
        layout: Tuple[int, int, float, int, int],
        cell: float,
        block: int,
        budget: _Budget,
    ) -> None:
        i0, j0, phase, blocks_i, blocks_j = layout
        self.g = g
        self.i0 = i0
        self.j0 = j0
        self.phase = phase
        self.cell = cell
        self.block = block
        self.blocks_i = blocks_i
        self.blocks_j = blocks_j
        self.ni = blocks_i * block  # cells per side; nodes 0..ni
        self.nj = blocks_j * block
        self.stride = self.nj + 1
        self.values: List[Optional[float]] = [None] * ((self.ni + 1) * self.stride)
        self.budget = budget
        self.evaluations = 0
        self.atom_of: Optional[Callable[[float, float], int]] = None
        # For a factor of a product: the product, which must be defined where the factor's
        # zero is drawn (x*sqrt(y) = 0 is not x = 0 for y < 0)
        self.whole: Optional[Callable[[float, float], float]] = None
        # Crossings by edge key (2 * node + 0 for the edge to i + 1, + 1 for the edge to j + 1)
        self.cross_x: Dict[int, float] = {}
        self.cross_y: Dict[int, float] = {}
        self.cross_kind: Dict[int, int] = {}
        self.cross_atom: Dict[int, int] = {}
        # Directed segments (fill) or adjacency (curves), between crossing keys
        self.next_key: Dict[int, int] = {}
        self.links: Dict[int, List[int]] = {}
        self.searched: Dict[int, bool] = {}
        self.maybe: List[int] = []
        self.visited: Optional[List[bool]] = None
        self.perimeter: Optional[Tuple[List[int], List[int], List[int]]] = None
        self.fill = False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _x(self, i: float) -> float:
        """The x of node column i, from its index in the plane (the same in every plot)."""
        return self.phase + (self.i0 + i) * self.cell

    def _y(self, j: float) -> float:
        return self.phase + (self.j0 + j) * self.cell

    def _evaluate(self, x: float, y: float) -> float:
        self.evaluations += 1
        self.budget.evals_left -= 1
        try:
            value = float(self.g(x, y))
        except Exception:
            return NAN
        if value - value != 0:  # NaN or infinite
            return NAN
        return value

    def value(self, i: int, j: int) -> float:
        index = i * self.stride + j
        value = self.values[index]
        if value is None:
            value = self._evaluate(self._x(i), self._y(j))
            self.values[index] = value
        return value

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def coarse(self, fill: bool) -> Tuple[List[int], List[int]]:
        """The coarse pass: blocks whose samples change sign, and blocks a zero may be near."""
        self.fill = fill
        return self._coarse_pass()

    def search(self, sure: List[int], maybe: List[int]) -> None:
        """The fine pass from the sure blocks (and, for a region, the border of rect),
        tracing the contour cell by cell; the other blocks wait for search_maybe."""
        stack: List[int] = []
        for key in sure:
            self._seed_block(key, stack, False)
        if self.fill:
            self._seed_border(stack)
        self._trace(stack)
        self.maybe = maybe

    def search_maybe(self) -> None:
        """Search the blocks a zero may be near, while the budget lasts (best effort: an
        unsearched one rarely hides more than a touching point)."""
        for key in self.maybe:
            if self.budget.spent:
                return
            stack: List[int] = []
            self._seed_block(key, stack, True)
            self._trace(stack)

    def _coarse_pass(self) -> Tuple[List[int], List[int]]:
        """Blocks whose samples change sign, then blocks where a zero may be within reach."""
        block, half = self.block, self.block // 2
        self._evaluate_lattice(0, self.ni, 0, self.nj, block)
        if self.budget.spent:
            return [], []
        values, stride, fill = self.values, self.stride, self.fill
        across, up = block * stride, block
        sure: List[int] = []
        maybe: List[int] = []
        for bi in range(self.blocks_i):
            self._evaluate_lattice(bi * block + half, bi * block + half, half, self.nj - half, block)
            if self.budget.exhausted():
                return sure, maybe
            row = bi * block * stride
            key = bi * self.blocks_j
            for bj in range(self.blocks_j):
                node = row + bj * block
                s0, s1 = values[node], values[node + across]
                s2, s3 = values[node + across + up], values[node + up]
                s4 = values[node + half * stride + half]
                # Fast path for the usual block, all samples of one sign (NaN fails both tests)
                if s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0:  # type: ignore[operator]
                    kind = 1 if 3 * min(s0, s1, s2, s3, s4) < max(s0, s1, s2, s3, s4) else 0  # type: ignore[type-var]
                elif s0 < 0 and s1 < 0 and s2 < 0 and s3 < 0 and s4 < 0:  # type: ignore[operator]
                    kind = 1 if min(s0, s1, s2, s3, s4) < 3 * max(s0, s1, s2, s3, s4) else 0  # type: ignore[type-var]
                else:
                    kind = _block_kind(s0, s1, s2, s3, s4, fill)  # type: ignore[arg-type]
                if kind == 2:
                    sure.append(key + bj)
                elif kind == 1:
                    maybe.append(key + bj)
        return sure, maybe

    def _seed_block(self, key: int, stack: List[int], maybe: bool) -> None:
        """Push the block's cells where the contour enters it, found on its perimeter.

        A contour inside a block that never reaches its perimeter (a small loop) shows no
        crossing there: a sure block is then marched cell by cell, a maybe block only when
        a 3 × 3 lattice inside it changes sign.
        """
        if key in self.searched:
            return
        self.searched[key] = True
        block, nj = self.block, self.nj
        bi, bj = divmod(key, self.blocks_j)
        i_s, j_s = bi * block, bj * block
        i_e, j_e = i_s + block, j_s + block
        self._evaluate_lattice(i_s, i_e, j_s, j_s, 1)
        self._evaluate_lattice(i_s, i_e, j_e, j_e, 1)
        self._evaluate_lattice(i_s, i_s, j_s, j_e, 1)
        self._evaluate_lattice(i_e, i_e, j_s, j_e, 1)
        if self.budget.spent:
            return
        values, fill = self.values, self.fill
        found = len(stack)
        node_base = i_s * self.stride + j_s
        cell_base = i_s * nj + j_s
        node_a, node_b, cells = self._perimeter()
        for k in range(len(cells)):
            a = values[node_base + node_a[k]]
            b = values[node_base + node_b[k]]
            if (a < 0) != (b < 0) and (fill or (a == a and b == b)):  # type: ignore[operator]
                stack.append(cell_base + cells[k])
        if len(stack) > found:
            return
        if maybe:
            quarter = block // 4 or 1
            self._evaluate_lattice(i_s, i_e, j_s, j_e, quarter)
            if self.budget.spent:
                return
            if not self._lattice_changes(i_s, i_e, j_s, j_e, quarter):
                # A small loop between the lattice's nodes: walk down from its lowest node
                seed = self._descend(i_s, i_e, j_s, j_e, quarter)
                if seed is not None:
                    stack.append(seed)
                return
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                stack.append(i * nj + j)

    def _descend(self, i_s: int, i_e: int, j_s: int, j_e: int, step: int) -> Optional[int]:
        """A cell where g changes sign near the lattice node closest to zero, found by a
        compass search (steps of step, halved down to one cell) within the block, or None.

        A loop a few cells across between the lattice's nodes (a circle 8 px across) leaves
        every sample on one side of zero; walking towards zero from the lowest one finds it.
        """
        values, stride, fill = self.values, self.stride, self.fill
        sign = 0.0
        best_i = best_j = -1
        best = INF
        for i in range(i_s, i_e + 1, step):
            for j in range(j_s, j_e + 1, step):
                value = values[i * stride + j]
                if value is None or value != value:
                    continue
                if sign == 0.0:
                    sign = 1.0 if value >= 0 else -1.0
                if sign * value < best:
                    best, best_i, best_j = sign * value, i, j
        if best_i < 0:
            return None
        i, j = best_i, best_j
        start = values[i * stride + j]
        moves = 0
        while step >= 1 and moves < _DESCENT_MOVES and not self.budget.spent:
            moves += 1
            target_i, target_j = i, j
            for di, dj in ((step, 0), (-step, 0), (0, step), (0, -step)):
                ni, nj = i + di, j + dj
                if ni < i_s or ni > i_e or nj < j_s or nj > j_e:
                    continue
                self._evaluate_lattice(ni, ni, nj, nj, 1)
                value = values[ni * stride + nj]
                if value is None or value != value:
                    continue
                if _changes(start, value, fill):
                    return self._cell_on_change(i, j, ni, nj)
                if sign * value < best:
                    best, target_i, target_j = sign * value, ni, nj
            if (target_i, target_j) == (i, j):
                step //= 2
            else:
                i, j = target_i, target_j
        return None

    def _cell_on_change(self, i: int, j: int, i_to: int, j_to: int) -> Optional[int]:
        """The cell beside the first one-cell edge, walking from node (i, j) to (i_to, j_to)
        along a row or column, where g changes sign."""
        values, stride, fill = self.values, self.stride, self.fill
        di = (i_to > i) - (i_to < i)
        dj = (j_to > j) - (j_to < j)
        start = values[i * stride + j]
        while (i, j) != (i_to, j_to):
            ni, nj = i + di, j + dj
            self._evaluate_lattice(ni, ni, nj, nj, 1)
            value = values[ni * stride + nj]
            if value is not None and _changes(start, value, fill):
                # The cell on the edge from (i, j) to (ni, nj), kept inside the grid
                ci = min(i, ni) if di else min(i, self.ni - 1)
                cj = min(j, nj) if dj else min(j, self.nj - 1)
                return ci * self.nj + cj
            i, j = ni, nj
        return None

    def _perimeter(self) -> Tuple[List[int], List[int], List[int]]:
        """A block's perimeter edges, relative to its first node and cell: the two nodes of
        each edge and the block's cell beside it (the same for every block of the grid)."""
        if self.perimeter is None:
            block, stride, nj = self.block, self.stride, self.nj
            node_a: List[int] = []
            node_b: List[int] = []
            cells: List[int] = []
            for k in range(block):
                # bottom, top, left and right sides
                node_a.extend((k * stride, k * stride + block, k, block * stride + k))
                node_b.extend(((k + 1) * stride, (k + 1) * stride + block, k + 1, block * stride + k + 1))
                cells.extend((k * nj, k * nj + block - 1, k, (block - 1) * nj + k))
            self.perimeter = (node_a, node_b, cells)
        return self.perimeter

    def _lattice_changes(self, i_s: int, i_e: int, j_s: int, j_e: int, step: int) -> bool:
        values, stride, fill = self.values, self.stride, self.fill
        first = values[i_s * stride + j_s]
        for i in range(i_s, i_e + 1, step):
            for j in range(j_s, j_e + 1, step):
                if _changes(first, values[i * stride + j], fill):
                    return True
        return False

    def _seed_border(self, stack: List[int]) -> None:
        """For a region: the cells along the border of rect where the region begins or ends."""
        ni, nj, stride = self.ni, self.nj, self.stride
        self._evaluate_lattice(0, ni, 0, 0, 1)
        self._evaluate_lattice(0, ni, nj, nj, 1)
        self._evaluate_lattice(0, 0, 0, nj, 1)
        self._evaluate_lattice(ni, ni, 0, nj, 1)
        if self.budget.spent:
            return
        values = self.values
        for i in range(ni):
            if _changes(values[i * stride], values[(i + 1) * stride], True):
                stack.append(i * nj)
            if _changes(values[i * stride + nj], values[(i + 1) * stride + nj], True):
                stack.append(i * nj + nj - 1)
        for j in range(nj):
            if _changes(values[j], values[j + 1], True):
                stack.append(j)
            if _changes(values[ni * stride + j], values[ni * stride + j + 1], True):
                stack.append((ni - 1) * nj + j)

    def _trace(self, stack: List[int]) -> None:
        """Follow the contour from the cells on the stack into every cell it crosses.

        Each cell is visited once: its corners are evaluated, its segments added, and the
        neighbours across its sides with a crossing pushed (not across a pole of a curve,
        so an asymptote is not walked).
        """
        if self.visited is None:
            self.visited = [False] * (self.ni * self.nj)
        visited, values, stride, nj, ni = self.visited, self.values, self.stride, self.nj, self.ni
        g, phase, cell, i0, j0 = self.g, self.phase, self.cell, self.i0, self.j0
        fill, kinds, budget = self.fill, self.cross_kind, self.budget
        count = steps = 0
        while stack:
            c = stack.pop()
            if visited[c]:
                continue
            if count >= budget.evals_left:
                budget.spent = True
                break
            steps += 1
            if steps % 256 == 0 and budget.exhausted():
                break  # the deadline
            visited[c] = True
            i = c // nj
            j = c - i * nj
            n00 = i * stride + j
            v00, v01, v10, v11 = values[n00], values[n00 + 1], values[n00 + stride], values[n00 + stride + 1]
            if v00 is None or v01 is None or v10 is None or v11 is None:
                x_a, x_b = phase + (i0 + i) * cell, phase + (i0 + i + 1) * cell
                y_a, y_b = phase + (j0 + j) * cell, phase + (j0 + j + 1) * cell
                if v00 is None:
                    v00 = values[n00] = _value_at(g, x_a, y_a)
                    count += 1
                if v01 is None:
                    v01 = values[n00 + 1] = _value_at(g, x_a, y_b)
                    count += 1
                if v10 is None:
                    v10 = values[n00 + stride] = _value_at(g, x_b, y_a)
                    count += 1
                if v11 is None:
                    v11 = values[n00 + stride + 1] = _value_at(g, x_b, y_b)
                    count += 1
            if fill:
                a = v00 < 0  # type: ignore[operator]
                b = v10 < 0  # type: ignore[operator]
                cc = v11 < 0  # type: ignore[operator]
                d = v01 < 0  # type: ignore[operator]
            else:
                if v00 != v00 or v01 != v01 or v10 != v10 or v11 != v11:
                    continue  # no curve in a cell with an undefined corner
                a, b, cc, d = v00 < 0, v10 < 0, v11 < 0, v01 < 0  # type: ignore[operator]
            if a == b == cc == d:
                continue
            # The cell's crossings count their evaluations on the grid and budget; move them
            # into count, which both get at the end
            before, left = self.evaluations, budget.evals_left
            self._march_cell(i, j, n00, a, b, cc, d)
            count += self.evaluations - before
            self.evaluations, budget.evals_left = before, left
            # Neighbours across the sides that change: bottom, right, top, left
            if a != b and j > 0 and (fill or kinds[2 * n00] != POLE):
                stack.append(c - 1)
            if b != cc and i + 1 < ni and (fill or kinds[2 * (n00 + stride) + 1] != POLE):
                stack.append(c + nj)
            if cc != d and j + 1 < nj and (fill or kinds[2 * (n00 + 1)] != POLE):
                stack.append(c + 1)
            if d != a and i > 0 and (fill or kinds[2 * n00 + 1] != POLE):
                stack.append(c - nj)
        self.evaluations += count
        budget.evals_left -= count
        if budget.evals_left <= 0:
            budget.spent = True

    def _evaluate_lattice(self, i_first: int, i_last: int, j_first: int, j_last: int, step: int) -> None:
        """Evaluate the nodes i_first..i_last by j_first..j_last, every step-th, not yet known.

        The hot loop of the plotter: locals only, no method calls but g's (Brython makes
        each attribute lookup and call cost about as much as evaluating g).
        """
        g, values, stride = self.g, self.values, self.stride
        phase, cell, i0, j0 = self.phase, self.cell, self.i0, self.j0
        budget = self.budget
        count = 0
        for i in range(i_first, i_last + 1, step):
            x = phase + (i0 + i) * cell
            base = i * stride
            for j in range(j_first, j_last + 1, step):
                index = base + j
                if values[index] is not None:
                    continue
                try:
                    value = g(x, phase + (j0 + j) * cell)
                except Exception:
                    value = NAN
                if value - value != 0:  # NaN or infinite
                    value = NAN
                values[index] = value
                count += 1
            if count > budget.evals_left:
                break
        self.evaluations += count
        budget.evals_left -= count
        if budget.evals_left <= 0:
            budget.spent = True

    # ------------------------------------------------------------------
    # One cell
    # ------------------------------------------------------------------

    def _march_cell(self, i: int, j: int, n00: int, a: bool, b: bool, c: bool, d: bool) -> None:
        """Add the segments of cell (i, j), whose corners (counter-clockwise from n00) are
        inside the region g < 0 as a, b, c, d, not all alike."""
        stride, kinds = self.stride, self.cross_kind
        n10 = n00 + stride
        n01 = n00 + 1
        # Sides counter-clockwise (math y up): bottom, right, top, left, with the corner
        # each one leads to (an entry into the region when that corner is inside)
        keys: List[int] = []
        enters: List[bool] = []
        if a != b:
            key = 2 * n00
            if key not in kinds:
                self._locate(key, i, j, 1, 0, n00, n10)
            keys.append(key)
            enters.append(b)
        if b != c:
            key = 2 * n10 + 1
            if key not in kinds:
                self._locate(key, i + 1, j, 0, 1, n10, n10 + 1)
            keys.append(key)
            enters.append(c)
        if c != d:
            key = 2 * n01
            if key not in kinds:
                self._locate(key, i, j + 1, 1, 0, n01, n01 + stride)
            keys.append(key)
            enters.append(d)
        if d != a:
            key = 2 * n00 + 1
            if key not in kinds:
                self._locate(key, i, j, 0, 1, n00, n01)
            keys.append(key)
            enters.append(a)
        if len(keys) == 2:
            if enters[0]:
                self._add_segment(keys[1], keys[0])
            else:
                self._add_segment(keys[0], keys[1])
            return
        # A saddle: the centre says whether the region joins the two inside corners
        centre = self._evaluate(self._x(i + 0.5), self._y(j + 0.5))
        self._add_saddle_segments(keys, enters, centre < 0)

    def _add_saddle_segments(self, keys: List[int], enters: List[bool], centre_inside: bool) -> None:
        """Pair each exit from the region with an entry (region on the segment's left).

        Going counter-clockwise round the cell, the region's part of the boundary runs
        from an entry to an exit; the segment closing it runs from the exit back to the
        entry. When the region joins two opposite corners through the centre, each exit
        is closed to the next entry instead.
        """
        for index in range(4):
            if enters[index]:
                continue
            partner = keys[(index + 1) % 4] if centre_inside else keys[(index - 1) % 4]
            self._add_segment(keys[index], partner)

    def _add_segment(self, start: int, end: int) -> None:
        kinds = self.cross_kind
        if self.fill:
            self.next_key[start] = end
            return
        if kinds[start] != ZERO or kinds[end] != ZERO:
            return  # a pole: no curve there
        # Not setdefault: Brython's setdefault on an empty dict stores an int key as a str
        links = self.links
        if start in links:
            links[start].append(end)
        else:
            links[start] = [end]
        if end in links:
            links[end].append(start)
        else:
            links[end] = [start]

    def _crossing(self, key: int, i: int, j: int, side: int) -> None:
        """Locate the crossing on side 0 (bottom) or 3 (left) of cell (i, j), once."""
        if key in self.cross_kind:
            return
        node = i * self.stride + j
        if side == 0:
            self._locate(key, i, j, 1, 0, node, node + self.stride)
        else:
            self._locate(key, i, j, 0, 1, node, node + 1)

    def _locate(self, key: int, i: int, j: int, di: int, dj: int, node_a: int, node_b: int) -> None:
        """Place the crossing on the edge from node (i, j) to (i + di, j + dj).

        Points on the edge are at a fraction t of the way: linear interpolation, then
        one regula falsi step; a bisection when g is undefined at an end (the edge of its
        domain) or the chord point is farther from zero than an end (a pole, or a kink).
        """
        values = self.values
        va, vb = values[node_a], values[node_b]
        cell, phase = self.cell, self.phase
        x0, y0 = phase + (self.i0 + i) * cell, phase + (self.j0 + j) * cell
        dx, dy = di * cell, dj * cell
        if va != va or vb != vb:  # type: ignore[comparison-overlap]
            # The edge of g's domain (only regions reach here), found by bisection
            t_ok, t_bad = (1.0, 0.0) if va != va else (0.0, 1.0)  # type: ignore[comparison-overlap]
            for _ in range(_DOMAIN_BISECTIONS):
                t = 0.5 * (t_ok + t_bad)
                vm = self._evaluate(x0 + t * dx, y0 + t * dy)
                if vm == vm:
                    t_ok = t
                else:
                    t_bad = t
            t = 0.5 * (t_ok + t_bad)
            self._store(key, x0 + t * dx, y0 + t * dy, DOMAIN)
            return
        assert va is not None and vb is not None
        t = va / (va - vb)
        vm = _value_at(self.g, x0 + t * dx, y0 + t * dy)
        self.evaluations += 1
        self.budget.evals_left -= 1
        if vm != vm:
            self._store(key, x0 + t * dx, y0 + t * dy, POLE)
            return
        if vm == 0:
            self._store(key, x0 + t * dx, y0 + t * dy, ZERO)
            return
        if abs(vm) > min(abs(va), abs(vb)):
            # Farther from zero than an end: a pole or a kink (and, or, |x|) next to the zero
            kind, t = self._bisect(x0, y0, dx, dy, va, vb, t, vm)
            self._store(key, x0 + t * dx, y0 + t * dy, kind)
            return
        # One regula falsi step inside the half that still changes sign
        if (vm < 0) == (va < 0):
            t = t + vm / (vm - vb) * (1.0 - t)
        else:
            t = t - vm / (vm - va) * t
        self._store(key, x0 + t * dx, y0 + t * dy, ZERO)

    def _bisect(
        self, x0: float, y0: float, dx: float, dy: float, va: float, vb: float, t: float, vm: float
    ) -> Tuple[int, float]:
        """Halve the bracket that changes sign: (POLE, t) when its ends move away from
        zero, else (ZERO, t) at the bracket's chord point.

        A zero's bracket ends shrink towards 0; a pole's grow. Each end that moved is
        compared with the edge's end on its side (an end that never moved, when the pole
        is within 1/64 of the edge from it, says nothing). The max and min of and/or (and
        |x|) shrink towards their zero too, so kinks stay zeros.
        """
        start_a, start_b = abs(va), abs(vb)
        t_a, t_b = 0.0, 1.0
        # Start from the half of the edge that changes sign around the chord point
        if (vm < 0) == (va < 0):
            t_a, va = t, vm
        else:
            t_b, vb = t, vm
        for _ in range(_POLE_BISECTIONS):
            t = 0.5 * (t_a + t_b)
            value = self._evaluate(x0 + t * dx, y0 + t * dy)
            if value != value:
                return POLE, t
            if value == 0:
                return ZERO, t
            if (value < 0) == (va < 0):
                t_a, va = t, value
            else:
                t_b, vb = t, value
        grew_a = t_a > 0.0 and abs(va) > _POLE_GROWTH * start_a
        grew_b = t_b < 1.0 and abs(vb) > _POLE_GROWTH * start_b
        if (grew_a or t_a == 0.0) and (grew_b or t_b == 1.0):
            return POLE, 0.5 * (t_a + t_b)
        return ZERO, t_a + va / (va - vb) * (t_b - t_a)

    def _store(self, key: int, x: float, y: float, kind: int) -> None:
        if kind == ZERO and self.whole is not None:
            self.evaluations += 1
            self.budget.evals_left -= 1
            whole = _value_at(self.whole, x, y)
            if whole != whole:
                kind = DOMAIN
        self.cross_x[key] = x
        self.cross_y[key] = y
        self.cross_kind[key] = kind
        if self.atom_of is not None and kind == ZERO:
            self.evaluations += 1
            self.budget.evals_left -= 1
            try:
                self.cross_atom[key] = int(self.atom_of(x, y))
            except Exception:
                self.cross_atom[key] = 0

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _point(self, key: int) -> Point:
        return (self.cross_x[key], self.cross_y[key])

    def add_curves(self, result: PlotResult) -> None:
        """Link the undirected segments into polylines: open chains first, then loops."""
        links = self.links
        used: Dict[Tuple[int, int], bool] = {}
        starts = [key for key, ends in links.items() if len(ends) != 2] + list(links)
        for start in starts:
            for first in links[start]:
                if (start, first) in used:
                    continue
                chain = self._follow(start, first, used)
                if len(chain) >= 2:
                    result.curves.append([self._point(key) for key in chain])
                    result.curve_atoms.append(0)

    def _follow(self, start: int, first: int, used: Dict[Tuple[int, int], bool]) -> List[int]:
        chain = [start]
        previous, current = start, first
        while True:
            used[(previous, current)] = True
            used[(current, previous)] = True
            chain.append(current)
            if current == start:
                return chain
            following = None
            for candidate in self.links[current]:
                if (current, candidate) not in used:
                    following = candidate
                    break
            if following is None:
                return chain
            previous, current = current, following

    def add_region(self, result: PlotResult) -> None:
        """The region's rings (with the border of rect) and its stroked boundary stretches."""
        self._add_border()
        next_key = self.next_key
        visited: Dict[int, bool] = {}
        for start in list(next_key):
            if start in visited:
                continue
            ring = [start]
            visited[start] = True
            current = next_key[start]
            closed = False
            while True:
                if current == start:
                    closed = True
                    break
                if current in visited or current not in next_key:
                    break
                visited[current] = True
                ring.append(current)
                current = next_key[current]
            if closed and len(ring) >= 3:
                points = [self._point(key) for key in ring]
                result.rings.append(points)
                self._add_strokes(ring, result)
            elif not closed:
                result.complete = False

    def _add_strokes(self, ring: List[int], result: PlotResult) -> None:
        """The stretches of a ring between zeros of g (not poles, domain edges or the border)."""
        kinds, atoms = self.cross_kind, self.cross_atom
        count = len(ring)
        stroked = [kinds[ring[index]] == ZERO and kinds[ring[(index + 1) % count]] == ZERO for index in range(count)]
        if all(stroked):
            points = [self._point(key) for key in ring]
            points.append(points[0])
            self._append_runs(points, [atoms.get(key, 0) for key in ring] + [atoms.get(ring[0], 0)], result)
            return
        # Start right after an unstroked segment, so no run is cut in two
        offset = stroked.index(False) + 1
        run: List[int] = []
        for step in range(count + 1):
            index = (offset + step) % count
            if step < count and stroked[index]:
                if not run:
                    run.append(ring[index])
                run.append(ring[(index + 1) % count])
            elif run:
                self._append_runs([self._point(key) for key in run], [atoms.get(key, 0) for key in run], result)
                run = []

    @staticmethod
    def _append_runs(points: List[Point], atoms: List[int], result: PlotResult) -> None:
        """Split a stroked stretch where the atom (and so the line style) changes."""
        start = 0
        for index in range(1, len(points)):
            if atoms[index] != atoms[start] and index - start >= 1:
                result.curves.append(points[start : index + 1])
                result.curve_atoms.append(atoms[start])
                start = index
        if len(points) - start >= 2:
            result.curves.append(points[start:])
            result.curve_atoms.append(atoms[start])

    def _add_border(self) -> None:
        """Walk the border of rect counter-clockwise, adding the stretches inside the region."""
        nodes = self._border_nodes()
        count = len(nodes)
        inside = [_inside(self.value(i, j)) for i, j in nodes]
        if not any(inside):
            return
        corner_keys = self._corner_keys()
        if all(inside):
            keys = [corner_keys[k] for k in range(4)]
            for index in range(4):
                self.next_key[keys[index]] = keys[(index + 1) % 4]
            return
        # Crossing keys on the border edges, and the corners between them, in order
        sequence: List[Tuple[int, int]] = []  # (key, 1 entry / -1 exit / 0 corner)
        corner_at = {0: 0, self.ni: 1, self.ni + self.nj: 2, 2 * self.ni + self.nj: 3}
        for index in range(count):
            if index in corner_at and inside[index]:
                sequence.append((corner_keys[corner_at[index]], 0))
            following = (index + 1) % count
            if inside[index] != inside[following]:
                key = self._border_edge_key(nodes[index], nodes[following])
                if key not in self.cross_kind:
                    i_a, j_a = nodes[index]
                    i_b, j_b = nodes[following]
                    self._store_border_crossing(key, i_a, j_a, i_b, j_b)
                sequence.append((key, 1 if inside[following] else -1))
        # Link from each entry through the corners to the next exit
        length = len(sequence)
        for index in range(length):
            key, role = sequence[index]
            if role != 1:
                continue
            current = key
            step = index
            while True:
                step = (step + 1) % length
                following_key, following_role = sequence[step]
                self.next_key[current] = following_key
                current = following_key
                if following_role == -1:
                    break

    def _border_nodes(self) -> List[Tuple[int, int]]:
        """Border nodes counter-clockwise from (0, 0): bottom, right, top, left."""
        ni, nj = self.ni, self.nj
        nodes = [(i, 0) for i in range(ni)]
        nodes += [(ni, j) for j in range(nj)]
        nodes += [(i, nj) for i in range(ni, 0, -1)]
        nodes += [(0, j) for j in range(nj, 0, -1)]
        return nodes

    def _corner_keys(self) -> List[int]:
        """Keys for the four corners of rect (negative, so never an edge key)."""
        corners = ((0, 0), (self.ni, 0), (self.ni, self.nj), (0, self.nj))
        keys = []
        for number, (i, j) in enumerate(corners):
            key = -(number + 1)
            self.cross_x[key] = self._x(i)
            self.cross_y[key] = self._y(j)
            self.cross_kind[key] = BORDER
            keys.append(key)
        return keys

    def _border_edge_key(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        (ia, ja), (ib, jb) = a, b
        i, j = min(ia, ib), min(ja, jb)
        node = i * self.stride + j
        return 2 * node + (1 if ia == ib else 0)

    def _store_border_crossing(self, key: int, ia: int, ja: int, ib: int, jb: int) -> None:
        """A crossing on the border that no searched cell located (the budget ran out)."""
        side = 0 if ja == jb else 3
        self._crossing(key, min(ia, ib), min(ja, jb), side)


def _inside(value: Optional[float]) -> bool:
    """Whether a node is in the region g < 0 (NaN and None are outside)."""
    return value is not None and value < 0


def _value_at(g: Callable[[float, float], float], x: float, y: float) -> float:
    """g(x, y), NaN where it raises or is not finite."""
    try:
        value = g(x, y)
    except Exception:
        return NAN
    if value - value != 0:  # NaN or infinite
        return NAN
    return value


def _changes(a: Optional[float], b: Optional[float], fill: bool) -> bool:
    """Whether an edge between two known nodes has a crossing (of a region: NaN is outside;
    of a curve: both ends defined and of opposite signs)."""
    if fill:
        return (a < 0) != (b < 0)  # type: ignore[operator]
    return a == a and b == b and (a < 0) != (b < 0)  # type: ignore[operator]


def _block_kind(s0: float, s1: float, s2: float, s3: float, s4: float, fill: bool) -> int:
    """2: the samples change sign; 1: a zero may be near; 0: neither.

    For a region a NaN sample is outside (a sign of its own when it meets an inside one);
    for a curve only the defined samples count. A zero may be near when the defined
    samples come closer to zero than half their spread.
    """
    negative = 0
    finite = 0
    low = high = 0.0
    for value in (s0, s1, s2, s3, s4):
        if value != value:
            continue
        if finite == 0:
            low = high = value
        elif value < low:
            low = value
        elif value > high:
            high = value
        finite += 1
        if value < 0:
            negative += 1
    if finite == 0:
        return 0
    if negative and (negative < finite or (fill and finite < 5)):
        return 2
    nearest = -high if negative else low
    if nearest <= 0 or nearest < 0.5 * (high - low):
        return 1
    return 0


def ring_area(ring: Sequence[Point]) -> float:
    """Signed area of a ring (positive when counter-clockwise)."""
    total = 0.0
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total
