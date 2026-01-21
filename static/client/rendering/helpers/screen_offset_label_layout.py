"""Screen-offset label layout solver for resolving label overlaps.

This module provides a sophisticated layout solver that adjusts label
positions to prevent overlapping text in dense point clusters.

Key Features:
    - Spatial hash acceleration for overlap detection
    - Vertical displacement with configurable step sizes
    - Greedy relaxation toward original positions
    - Group-based label tracking for multi-line labels
    - Configurable iteration limits for performance
    - Hide mode for labels that cannot be placed cleanly
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

Rect = Tuple[float, float, float, float]


def _coerce_float(value: Any, default: float) -> float:
    """Safely convert a value to float with fallback.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails or result is NaN.

    Returns:
        Float value or default.
    """
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return out


def _coerce_int(value: Any, default: int) -> int:
    """Safely convert a value to int with fallback.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Integer value or default.
    """
    try:
        return int(value)
    except Exception:
        return default


def shift_rect_y(rect: Rect, dy: float) -> Rect:
    """Shift a rectangle vertically by a given amount.

    Args:
        rect: Tuple of (min_x, max_x, min_y, max_y).
        dy: Vertical displacement in pixels.

    Returns:
        New rectangle tuple with shifted y bounds.
    """
    shift = _coerce_float(dy, 0.0)
    if shift == 0.0:
        return rect
    min_x, max_x, min_y, max_y = rect
    return (min_x, max_x, min_y + shift, max_y + shift)


def rects_intersect(a: Rect, b: Rect) -> bool:
    """Check if two rectangles overlap.

    Touching edges are treated as non-overlapping to allow flush stacking.

    Args:
        a: First rectangle (min_x, max_x, min_y, max_y).
        b: Second rectangle (min_x, max_x, min_y, max_y).

    Returns:
        True if rectangles overlap (not just touch).
    """
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


class LabelTextCall:
    """Data class for a pending label text draw call.

    Stores all information needed to render a label and resolve
    its layout position within a group of potentially overlapping labels.

    Attributes:
        group: Identifier for grouping multi-line labels.
        order: Draw order for tie-breaking during layout.
        text: Label text content.
        position: Screen position tuple (x, y).
        anchor_screen: Anchor point before offset.
        font: FontStyle for rendering.
        color: Text color string.
        alignment: TextAlignment for positioning.
        style_overrides: Optional CSS style overrides.
        metadata: Metadata dict for reprojection.
        line_index: Index within multi-line label.
        line_count: Total lines in this label group.
        max_line_len: Maximum line length for width estimation.
        font_size: Font size in pixels.
        line_height: Line spacing in pixels.
    """
    __slots__ = (
        "group",
        "order",
        "text",
        "position",
        "anchor_screen",
        "font",
        "color",
        "alignment",
        "style_overrides",
        "metadata",
        "line_index",
        "line_count",
        "max_line_len",
        "font_size",
        "line_height",
    )

    def __init__(
        self,
        *,
        group: Any,
        order: int,
        text: str,
        position: Tuple[float, float],
        anchor_screen: Tuple[float, float],
        font: Any,
        color: str,
        alignment: Any,
        style_overrides: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        line_index: int,
        line_count: int,
        max_line_len: int,
        font_size: float,
        line_height: float,
    ) -> None:
        self.group = group
        self.order = order
        self.text = text
        self.position = position
        self.anchor_screen = anchor_screen
        self.font = font
        self.color = color
        self.alignment = alignment
        self.style_overrides = style_overrides
        self.metadata = metadata
        self.line_index = line_index
        self.line_count = line_count
        self.max_line_len = max_line_len
        self.font_size = font_size
        self.line_height = line_height


class LabelBlock:
    """Simplified block representation for layout solving.

    Attributes:
        group: Identifier for the label group.
        order: Draw order for tie-breaking.
        base_rect: Bounding rectangle at dy=0.
        step: Vertical step size for displacement.
    """
    __slots__ = ("group", "order", "base_rect", "step")

    def __init__(self, *, group: Any, order: int, base_rect: Rect, step: float) -> None:
        self.group = group
        self.order = int(order)
        self.base_rect = base_rect
        self.step = _coerce_float(step, 1.0)


class SpatialHash2D:
    """Spatial hash for efficient rectangle overlap queries.

    Uses a 2D grid to accelerate collision detection between
    label bounding boxes during layout solving.

    Attributes:
        _cell_size: Size of grid cells in pixels.
        _cells: Dict mapping cell coords to groups in that cell.
        _rects: Dict mapping groups to their bounding rectangles.
        _group_cells: Dict mapping groups to their occupied cells.
    """
    __slots__ = ("_cell_size", "_cells", "_rects", "_group_cells")

    def __init__(self, *, cell_size: float = 32.0) -> None:
        size = _coerce_float(cell_size, 32.0)
        if size <= 0:
            size = 32.0
        self._cell_size = size
        self._cells: Dict[Tuple[int, int], Set[Any]] = {}
        self._rects: Dict[Any, Rect] = {}
        self._group_cells: Dict[Any, List[Tuple[int, int]]] = {}

    def _rect_cells(self, rect: Rect) -> List[Tuple[int, int]]:
        cell_size = self._cell_size
        min_x, max_x, min_y, max_y = rect
        cx0 = int(min_x // cell_size)
        cx1 = int(max_x // cell_size)
        cy0 = int(min_y // cell_size)
        cy1 = int(max_y // cell_size)
        cells: List[Tuple[int, int]] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                cells.append((cx, cy))
        return cells

    def get_rect(self, group: Any) -> Optional[Rect]:
        return self._rects.get(group)

    def add(self, group: Any, rect: Rect) -> None:
        self._rects[group] = rect
        cells = self._rect_cells(rect)
        self._group_cells[group] = cells
        for cell in cells:
            bucket = self._cells.get(cell)
            if bucket is None:
                bucket = set()
                self._cells[cell] = bucket
            bucket.add(group)

    def remove(self, group: Any) -> None:
        cells = self._group_cells.pop(group, None)
        if cells:
            for cell in cells:
                bucket = self._cells.get(cell)
                if not bucket:
                    continue
                try:
                    bucket.remove(group)
                except Exception:
                    pass
                if not bucket:
                    self._cells.pop(cell, None)
        self._rects.pop(group, None)

    def update(self, group: Any, rect: Rect) -> None:
        if group not in self._rects:
            self.add(group, rect)
            return
        old_cells = self._group_cells.get(group, [])
        new_cells = self._rect_cells(rect)
        if old_cells != new_cells:
            self.remove(group)
            self.add(group, rect)
            return
        self._rects[group] = rect

    def query(self, rect: Rect) -> Set[Any]:
        out: Set[Any] = set()
        for cell in self._rect_cells(rect):
            bucket = self._cells.get(cell)
            if bucket:
                out.update(bucket)
        return out


def _safe_hashable_group(candidate: Any, *, fallback: Any) -> Any:
    if candidate is None:
        return fallback
    try:
        hash(candidate)
    except Exception:
        return fallback
    return candidate


def make_label_text_call(
    *,
    order: int,
    text: str,
    position: Tuple[float, float],
    font: Any,
    color: str,
    alignment: Any,
    style_overrides: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Optional[LabelTextCall]:
    if not isinstance(metadata, dict):
        return None
    point_meta = metadata.get("point_label")
    if not isinstance(point_meta, dict):
        return None

    font_size_raw = getattr(font, "size", None)
    font_size = _coerce_float(font_size_raw, 0.0)
    if font_size <= 0:
        return None

    line_height = font_size + 2.0
    if line_height < 1.0:
        line_height = 1.0

    line_index = _coerce_int(point_meta.get("layout_line_index", 0), 0)
    line_count = _coerce_int(point_meta.get("layout_line_count", 1), 1)
    max_line_len = _coerce_int(point_meta.get("layout_max_line_len", len(text)), len(text))
    if line_count <= 0:
        line_count = 1
    if max_line_len < 0:
        max_line_len = 0

    offset_raw = point_meta.get("screen_offset", (0.0, 0.0))
    if isinstance(offset_raw, (list, tuple)) and len(offset_raw) == 2:
        offset_x = _coerce_float(offset_raw[0], 0.0)
        offset_y = _coerce_float(offset_raw[1], 0.0)
    else:
        offset_x = 0.0
        offset_y = 0.0

    pos_x = float(position[0])
    pos_y = float(position[1])
    anchor_screen = (pos_x - offset_x, pos_y - offset_y)

    fallback_group = (str(text), float(position[0]), float(position[1]))
    group = _safe_hashable_group(point_meta.get("layout_group"), fallback=fallback_group)

    return LabelTextCall(
        group=group,
        order=int(order),
        text=str(text),
        position=(pos_x, pos_y),
        anchor_screen=anchor_screen,
        font=font,
        color=str(color),
        alignment=alignment,
        style_overrides=style_overrides,
        metadata=metadata,
        line_index=int(line_index),
        line_count=int(line_count),
        max_line_len=int(max_line_len),
        font_size=float(font_size),
        line_height=float(line_height),
    )


def compute_block_rect(call: LabelTextCall) -> Rect:
    font_size = float(call.font_size)
    line_height = float(call.line_height)
    baseline_y0 = float(call.position[1]) - float(call.line_index) * line_height
    block_top = baseline_y0 - font_size
    block_bottom = block_top + float(call.line_count) * line_height
    approx_width = 0.6 * font_size * float(call.max_line_len)
    if approx_width < 1.0:
        approx_width = 1.0
    left = float(call.position[0])
    return (left, left + approx_width, block_top, block_bottom)


def _count_overlaps(grid: SpatialHash2D, rect: Rect, *, ignore: Any = None) -> Tuple[int, Set[Any]]:
    candidates = grid.query(rect)
    if ignore is not None and ignore in candidates:
        candidates.discard(ignore)
    overlaps = 0
    overlapping_groups: Set[Any] = set()
    for other in candidates:
        other_rect = grid.get_rect(other)
        if other_rect is None:
            continue
        if rects_intersect(rect, other_rect):
            overlaps += 1
            overlapping_groups.add(other)
    return overlaps, overlapping_groups


def _mover_key(group: Any, dy: Dict[Any, float], order: Dict[Any, int]) -> Tuple[float, int]:
    # Move the smallest |dy|, tie-break by moving later draw-order first.
    return (abs(dy.get(group, 0.0)), -int(order.get(group, 0)))


class _ScreenOffsetLabelLayoutSolver:
    __slots__ = ("_max_steps", "_cap", "_grid", "_base_rect", "_step", "_order", "_dy", "_prefer_positive")

    def __init__(self, *, max_steps: int, iteration_cap: int, cell_size: float) -> None:
        max_steps_i = _coerce_int(max_steps, 10)
        if max_steps_i < 0:
            max_steps_i = 0
        cap = _coerce_int(iteration_cap, 5000)
        if cap <= 0:
            cap = 1
        self._max_steps = max_steps_i
        self._cap = cap
        self._grid = SpatialHash2D(cell_size=cell_size)
        self._base_rect: Dict[Any, Rect] = {}
        self._step: Dict[Any, float] = {}
        self._order: Dict[Any, int] = {}
        self._dy: Dict[Any, float] = {}
        self._prefer_positive: Dict[Any, bool] = {}

    def _prefer_positive_direction(self, group: Any) -> bool:
        """Return True to prefer +dy (down), False to prefer -dy (up).

        This is derived from the text box position relative to its anchor:
        if the anchor point is in the lower half of the block, prefer moving down;
        if it is in the upper half, prefer moving up.

        For our screen_offset labels, the anchor is baseline-left (alphabetic),
        and the block rect is computed from that baseline using an approximate font box.
        """
        rect = self._base_rect.get(group)
        step_px = _coerce_float(self._step.get(group, 0.0), 0.0)
        if rect is None or step_px <= 0:
            return True
        # In our model line_height = font_size + 2, so font_size ~= step - 2.
        font_size = step_px - 2.0
        if font_size <= 0:
            return True
        anchor_y = float(rect[2]) + float(font_size)
        center_y = (float(rect[2]) + float(rect[3])) * 0.5
        return anchor_y >= center_y

    def _init_blocks(self, blocks: List[LabelBlock]) -> List[Any]:
        base_rect: Dict[Any, Rect] = {}
        step: Dict[Any, float] = {}
        order: Dict[Any, int] = {}
        prefer_positive: Dict[Any, bool] = {}
        groups: List[Any] = []
        for block in blocks:
            g = block.group
            base_rect[g] = block.base_rect
            step_value = _coerce_float(block.step, 1.0)
            if step_value <= 0:
                step_value = 1.0
            step[g] = step_value
            order[g] = int(block.order)
            groups.append(g)
        self._base_rect = base_rect
        self._step = step
        self._order = order
        self._dy = {g: 0.0 for g in groups}
        self._prefer_positive = prefer_positive
        grid = self._grid
        for g in groups:
            grid.add(g, base_rect[g])
        for g in groups:
            prefer_positive[g] = self._prefer_positive_direction(g)
        return groups

    def _pick_best_dy(self, mover: Any, *, ignore: Any = None) -> float:
        base = self._base_rect[mover]
        step_px = self._step[mover]
        max_steps = self._max_steps
        grid = self._grid
        prefer_positive = bool(self._prefer_positive.get(mover, True))

        def count_for(candidate_dy: float) -> int:
            candidate_rect = shift_rect_y(base, candidate_dy)
            count, _ = _count_overlaps(grid, candidate_rect, ignore=ignore)
            return count

        def is_preferred(candidate_dy: float) -> bool:
            if candidate_dy == 0.0:
                return True
            return candidate_dy > 0.0 if prefer_positive else candidate_dy < 0.0

        def better_dy(a: float, b: float, *, overlaps_a: int, overlaps_b: int) -> float:
            if overlaps_a != overlaps_b:
                return a if overlaps_a < overlaps_b else b
            abs_a = abs(a)
            abs_b = abs(b)
            if abs_a != abs_b:
                return a if abs_a < abs_b else b
            a_pref = is_preferred(a)
            b_pref = is_preferred(b)
            if a_pref != b_pref:
                return a if a_pref else b
            return a

        best_dy = self._dy.get(mover, 0.0)
        best_overlaps = 10 ** 9

        for k in range(0, max_steps + 1):
            if k == 0:
                candidates = (0.0,)
            else:
                first = k * step_px if prefer_positive else -k * step_px
                candidates = (first, -first)
            best_zero: Optional[float] = None
            for candidate_dy in candidates:
                overlaps = count_for(candidate_dy)
                if overlaps == 0:
                    if best_zero is None:
                        best_zero = candidate_dy
                    else:
                        best_zero = better_dy(best_zero, candidate_dy, overlaps_a=0, overlaps_b=0)
                    continue
                if best_overlaps == 10 ** 9:
                    best_overlaps = overlaps
                    best_dy = candidate_dy
                    continue
                choice = better_dy(best_dy, candidate_dy, overlaps_a=best_overlaps, overlaps_b=overlaps)
                if choice != best_dy:
                    best_dy = choice
                    best_overlaps = overlaps if choice == candidate_dy else best_overlaps
            if best_zero is not None:
                return float(best_zero)

        return float(best_dy)

    def _choose_mover_with_lookahead(self, collision_set: Set[Any]) -> Tuple[Any, float]:
        """Choose which label in a collision set to move, minimizing final abs(dy).

        Decision tuple (lexicographic):
        - overlaps after moving (prefer fewer)
        - abs(best_dy) (prefer smaller total displacement from dy=0)
        - abs(best_dy - current_dy) (prefer smaller change)
        - -order (prefer later labels on ties for stability)
        """
        grid = self._grid
        dy = self._dy
        order = self._order

        best_group: Any = None
        best_dy: float = 0.0
        best_key: Optional[Tuple[int, float, float, int]] = None

        for candidate in collision_set:
            current_dy = float(dy.get(candidate, 0.0) or 0.0)
            if grid.get_rect(candidate) is None:
                continue

            candidate_best_dy = float(self._pick_best_dy(candidate, ignore=candidate))
            candidate_rect = shift_rect_y(self._base_rect[candidate], candidate_best_dy)
            overlaps, _ = _count_overlaps(grid, candidate_rect, ignore=candidate)

            key = (
                int(overlaps),
                float(abs(candidate_best_dy)),
                float(abs(candidate_best_dy - current_dy)),
                -int(order.get(candidate, 0)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_group = candidate
                best_dy = candidate_best_dy

        if best_group is None:
            # Fallback to the old behavior.
            mover = min(collision_set, key=lambda item: _mover_key(item, dy, order))
            return mover, float(dy.get(mover, 0.0) or 0.0)

        return best_group, float(best_dy)

    def _resolve(self, groups: List[Any]) -> None:
        grid = self._grid
        dy = self._dy
        order = self._order

        # Process later labels first. Use a stack (pop from end) to keep operations O(1).
        groups_sorted = sorted(groups, key=lambda g: int(order.get(g, 0)))
        stack: List[Any] = list(groups_sorted)
        in_stack: Set[Any] = set(stack)

        iterations = 0
        while stack and iterations < self._cap:
            g = stack.pop()
            in_stack.discard(g)
            current_rect = grid.get_rect(g)
            if current_rect is None:
                iterations += 1
                continue
            overlaps, overlapping_groups = _count_overlaps(grid, current_rect, ignore=g)
            if overlaps == 0:
                iterations += 1
                continue

            # Reuse the returned set to keep allocations down.
            collision_set = overlapping_groups
            collision_set.add(g)
            mover, best_dy = self._choose_mover_with_lookahead(collision_set)

            if grid.get_rect(mover) is None:
                iterations += 1
                continue

            dy[mover] = float(best_dy)
            new_rect = shift_rect_y(self._base_rect[mover], dy[mover])
            grid.update(mover, new_rect)

            # Re-enqueue potentially affected labels.
            _, new_overlapping = _count_overlaps(grid, new_rect, ignore=mover)
            affected = collision_set
            affected.update(new_overlapping)
            affected.add(mover)
            for item in affected:
                if item not in in_stack:
                    stack.append(item)
                    in_stack.add(item)

            iterations += 1

    def _relax_toward_zero(self, groups: List[Any]) -> None:
        # Relaxation pass: pull labels back toward dy=0 when safe.
        grid = self._grid
        dy = self._dy
        base_rect = self._base_rect
        step = self._step
        prefer_positive = self._prefer_positive

        groups_by_displacement = sorted(groups, key=lambda g: abs(dy.get(g, 0.0)), reverse=True)
        for g in groups_by_displacement:
            current = dy.get(g, 0.0)
            if current == 0.0:
                continue
            grid.remove(g)
            base = base_rect[g]
            step_px = step[g]
            target_abs = abs(current)

            chosen = current
            pref_pos = bool(prefer_positive.get(g, True))
            eps = 1e-9

            # Try candidates with strictly smaller abs(dy) first.
            if step_px > 0:
                max_k = int(target_abs // step_px) + 1
            else:
                max_k = 0
            for k in range(0, max_k + 1):
                candidate_abs = float(k) * float(step_px)
                if candidate_abs > 0.0 and candidate_abs + eps >= target_abs:
                    continue
                if candidate_abs == 0.0:
                    candidates = (0.0,)
                else:
                    first = candidate_abs if pref_pos else -candidate_abs
                    candidates = (first, -first)
                for cand in candidates:
                    if abs(cand) + eps >= target_abs and cand != 0.0:
                        continue
                    cand_rect = shift_rect_y(base, cand)
                    count, _ = _count_overlaps(grid, cand_rect)
                    if count == 0:
                        chosen = cand
                        break
                if chosen != current:
                    break

            # If we could not reduce abs(dy), allow flipping sign at equal magnitude
            # when it is collision-free and matches the preferred direction.
            if chosen == current:
                flip = -float(current)
                if flip != current and abs(flip) == target_abs:
                    preferred_flip = (flip > 0.0) if pref_pos else (flip < 0.0)
                    current_preferred = (current > 0.0) if pref_pos else (current < 0.0)
                    if preferred_flip and not current_preferred:
                        flip_rect = shift_rect_y(base, flip)
                        count, _ = _count_overlaps(grid, flip_rect)
                        if count == 0:
                            chosen = flip

            dy[g] = float(chosen)
            grid.add(g, shift_rect_y(base, dy[g]))

    def solve(self, blocks: List[LabelBlock]) -> Dict[Any, float]:
        groups = self._init_blocks(blocks)
        self._resolve(groups)
        self._relax_toward_zero(groups)
        return dict(self._dy)


def solve_dy(
    blocks: List[LabelBlock],
    *,
    max_steps: int = 10,
    iteration_cap: int = 5000,
    cell_size: float = 32.0,
) -> Dict[Any, float]:
    """Solve vertical displacements to resolve label overlaps.

    Args:
        blocks: List of LabelBlock objects to position.
        max_steps: Maximum step multiples to try in each direction.
        iteration_cap: Maximum solver iterations.
        cell_size: Spatial hash cell size in pixels.

    Returns:
        Dict mapping group to vertical displacement (dy).
    """
    if not blocks:
        return {}
    solver = _ScreenOffsetLabelLayoutSolver(max_steps=max_steps, iteration_cap=iteration_cap, cell_size=cell_size)
    return solver.solve(blocks)


def solve_dy_for_text_calls(
    calls: List[LabelTextCall],
    *,
    max_steps: int = 10,
    iteration_cap: int = 5000,
    cell_size: float = 32.0,
) -> Dict[Any, float]:
    """Solve vertical displacements for label text calls.

    Converts LabelTextCall objects to blocks and solves layout.

    Args:
        calls: List of LabelTextCall objects to position.
        max_steps: Maximum step multiples to try in each direction.
        iteration_cap: Maximum solver iterations.
        cell_size: Spatial hash cell size in pixels.

    Returns:
        Dict mapping group to vertical displacement (dy).
    """
    if not calls:
        return {}
    by_group: Dict[Any, LabelTextCall] = {}
    for call in calls:
        existing = by_group.get(call.group)
        if existing is None:
            by_group[call.group] = call
            continue
        # Prefer the earliest line (smaller line_index) and later order on ties.
        if call.line_index < existing.line_index:
            by_group[call.group] = call
        elif call.line_index == existing.line_index and call.order > existing.order:
            by_group[call.group] = call
    blocks: List[LabelBlock] = []
    for group, call in by_group.items():
        blocks.append(
            LabelBlock(
                group=group,
                order=call.order,
                base_rect=compute_block_rect(call),
                step=call.line_height,
            )
        )
    return solve_dy(blocks, max_steps=max_steps, iteration_cap=iteration_cap, cell_size=cell_size)


def solve_dy_with_hide_for_text_calls(
    calls: List[LabelTextCall],
    *,
    max_abs_dy_factor: float = 3.0,
    max_passes: int = 2,
    max_steps: int = 10,
    iteration_cap: int = 5000,
    cell_size: float = 32.0,
) -> Tuple[Dict[Any, float], Set[Any]]:
    """Solve dy for screen_offset labels and hide labels under hard constraints.

    This is performance-oriented:
    - Treat max abs(dy) as a hard visibility bound (do not search beyond it).
    - If a label cannot be placed collision-free within the bound, hide it immediately.

    max_passes is accepted for backward compatibility but is not used; the algorithm
    performs a single constrained solve and returns (dy_by_group, hidden_groups).
    """
    if not calls:
        return {}, set()

    factor = _coerce_float(max_abs_dy_factor, 3.0)
    if factor <= 0:
        factor = 3.0
    cap_steps = _coerce_int(max_steps, 10)
    if cap_steps < 0:
        cap_steps = 0
    cap_iters = _coerce_int(iteration_cap, 5000)
    if cap_iters <= 0:
        cap_iters = 1

    # Pick a representative call per group (same as solve_dy_for_text_calls) so we can
    # compute group-level constraints cheaply.
    by_group: Dict[Any, LabelTextCall] = {}
    for call in calls:
        existing = by_group.get(call.group)
        if existing is None:
            by_group[call.group] = call
            continue
        if call.line_index < existing.line_index:
            by_group[call.group] = call
        elif call.line_index == existing.line_index and call.order > existing.order:
            by_group[call.group] = call

    base_rect: Dict[Any, Rect] = {}
    step_px: Dict[Any, float] = {}
    font_size_px: Dict[Any, float] = {}
    k_max: Dict[Any, int] = {}
    order: Dict[Any, int] = {}
    anchor: Dict[Any, Tuple[float, float]] = {}
    width: Dict[Any, float] = {}
    prefer_positive: Dict[Any, bool] = {}

    for group, rep in by_group.items():
        rect = compute_block_rect(rep)
        base_rect[group] = rect
        w = float(rect[1]) - float(rect[0])
        if w < 0:
            w = 0.0
        width[group] = w

        order[group] = int(rep.order)
        anchor[group] = (float(rep.anchor_screen[0]), float(rep.anchor_screen[1]))

        fs = _coerce_float(getattr(rep, "font_size", 0.0), 0.0)
        lh = _coerce_float(getattr(rep, "line_height", 0.0), 0.0)
        if fs <= 0 or lh <= 0:
            fs = max(fs, 0.0)
            lh = max(lh, 0.0)
        font_size_px[group] = fs
        step_px[group] = lh

        max_abs_dy = factor * fs if fs > 0 else 0.0
        if lh > 0 and max_abs_dy > 0:
            km = int(max_abs_dy // lh)
        else:
            km = 0
        if km < 0:
            km = 0
        if cap_steps > 0:
            km = min(km, cap_steps)
        k_max[group] = km

        # Direction preference: baseline anchor sits in lower half -> prefer +dy (down).
        center_y = (float(rect[2]) + float(rect[3])) * 0.5
        baseline_y = float(rect[2]) + fs
        prefer_positive[group] = bool(baseline_y >= center_y)

    # Spatial hash seeded at dy=0 for all groups.
    grid = SpatialHash2D(cell_size=cell_size)
    dy: Dict[Any, float] = {}
    hidden: Set[Any] = set()
    for g in base_rect:
        dy[g] = 0.0
        grid.add(g, base_rect[g])

    def hide_group(g: Any) -> None:
        if g in hidden:
            return
        hidden.add(g)
        try:
            grid.remove(g)
        except Exception:
            pass

    def anchor_too_close(a: Any, b: Any) -> bool:
        ax, ay = anchor.get(a, (0.0, 0.0))
        bx, by = anchor.get(b, (0.0, 0.0))
        dx = ax - bx
        dyv = ay - by
        dist2 = dx * dx + dyv * dyv
        # threshold = 0.5 * ((widthA + widthB) / 2) = (widthA + widthB) / 4
        thresh = (float(width.get(a, 0.0)) + float(width.get(b, 0.0))) * 0.25
        if thresh <= 0:
            return False
        return dist2 < (thresh * thresh)

    def pick_best_dy_for(g: Any, *, ignore: Any) -> Tuple[float, int]:
        rect0 = base_rect[g]
        step = float(step_px.get(g, 0.0) or 0.0)
        max_k = int(k_max.get(g, 0) or 0)
        pref_pos = bool(prefer_positive.get(g, True))

        if step <= 0 or max_k < 0:
            max_k = 0
            step = 0.0

        def is_preferred(candidate: float) -> bool:
            if candidate == 0.0:
                return True
            return candidate > 0.0 if pref_pos else candidate < 0.0

        def better(a: float, b: float, oa: int, ob: int) -> float:
            if oa != ob:
                return a if oa < ob else b
            abs_a = abs(a)
            abs_b = abs(b)
            if abs_a != abs_b:
                return a if abs_a < abs_b else b
            ap = is_preferred(a)
            bp = is_preferred(b)
            if ap != bp:
                return a if ap else b
            return a

        best_dy = dy.get(g, 0.0) or 0.0
        best_overlaps = 10 ** 9
        for k in range(0, max_k + 1):
            if k == 0:
                candidates = (0.0,)
            else:
                first = k * step if pref_pos else -k * step
                candidates = (first, -first)
            for cand in candidates:
                cand_rect = shift_rect_y(rect0, cand)
                overlaps, _ = _count_overlaps(grid, cand_rect, ignore=ignore)
                if best_overlaps == 10 ** 9:
                    best_overlaps = overlaps
                    best_dy = cand
                    if overlaps == 0:
                        return float(best_dy), 0
                    continue
                choice = better(best_dy, cand, best_overlaps, overlaps)
                if choice != best_dy:
                    best_dy = choice
                    best_overlaps = overlaps if choice == cand else best_overlaps
                if best_overlaps == 0 and k == 0:
                    return float(best_dy), 0
            if best_overlaps == 0:
                return float(best_dy), 0
        return float(best_dy), int(best_overlaps)

    # Process later labels first.
    stack = sorted(list(base_rect.keys()), key=lambda g: int(order.get(g, 0)))
    in_stack: Set[Any] = set(stack)

    iterations = 0
    while stack and iterations < cap_iters:
        g = stack.pop()
        in_stack.discard(g)
        if g in hidden:
            iterations += 1
            continue
        current_rect = grid.get_rect(g)
        if current_rect is None:
            iterations += 1
            continue

        overlaps, overlapping = _count_overlaps(grid, current_rect, ignore=g)
        if overlaps == 0:
            iterations += 1
            continue

        # Proximity hide rule for tiny-anchor-distance collisions.
        to_hide: List[Any] = []
        for other in list(overlapping):
            if other in hidden:
                overlapping.discard(other)
                continue
            if anchor_too_close(g, other):
                later = g if int(order.get(g, 0)) >= int(order.get(other, 0)) else other
                to_hide.append(later)
        if to_hide:
            for h in to_hide:
                hide_group(h)
            iterations += 1
            # Re-enqueue neighbors because the grid changed.
            for item in overlapping:
                if item not in hidden and item not in in_stack:
                    stack.append(item)
                    in_stack.add(item)
            continue

        # Recompute overlaps after any proximity filtering.
        current_rect = grid.get_rect(g)
        if current_rect is None:
            iterations += 1
            continue
        overlaps, overlapping = _count_overlaps(grid, current_rect, ignore=g)
        if overlaps == 0:
            iterations += 1
            continue

        collision_set = overlapping
        collision_set.add(g)

        # Lookahead mover selection under the hard dy bound.
        best_group: Any = None
        best_dy: float = 0.0
        best_overlaps: int = 10 ** 9
        best_key: Optional[Tuple[int, float, float, int]] = None
        for cand in collision_set:
            if cand in hidden or grid.get_rect(cand) is None:
                continue
            cand_best_dy, cand_overlaps = pick_best_dy_for(cand, ignore=cand)
            key = (
                int(cand_overlaps),
                float(abs(cand_best_dy)),
                float(abs(cand_best_dy - float(dy.get(cand, 0.0) or 0.0))),
                -int(order.get(cand, 0)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_group = cand
                best_dy = float(cand_best_dy)
                best_overlaps = int(cand_overlaps)

        if best_group is None:
            iterations += 1
            continue

        if best_overlaps != 0:
            # Nothing can be placed without overlap inside the bound; hide the later label.
            later = max(collision_set, key=lambda x: int(order.get(x, 0)))
            hide_group(later)
            iterations += 1
            continue

        dy[best_group] = float(best_dy)
        grid.update(best_group, shift_rect_y(base_rect[best_group], dy[best_group]))

        # Re-enqueue affected labels.
        affected = collision_set
        for item in affected:
            if item not in hidden and item not in in_stack:
                stack.append(item)
                in_stack.add(item)

        iterations += 1

    # Relaxation within bounds: pull labels toward dy=0 when safe.
    groups_by_disp = sorted([g for g in dy.keys() if g not in hidden], key=lambda x: abs(dy.get(x, 0.0)), reverse=True)
    for g in groups_by_disp:
        current = float(dy.get(g, 0.0) or 0.0)
        if current == 0.0:
            continue
        # Temporarily remove g for overlap checking.
        grid.remove(g)
        rect0 = base_rect[g]
        step = float(step_px.get(g, 0.0) or 0.0)
        max_k = int(k_max.get(g, 0) or 0)
        pref_pos = bool(prefer_positive.get(g, True))
        if step <= 0 or max_k < 0:
            max_k = 0
            step = 0.0
        current_k = int(abs(current) // step) if step > 0 else 0
        limit_k = min(current_k, max_k)
        chosen = current
        for k in range(0, limit_k + 1):
            cand_abs = float(k) * step
            if cand_abs >= abs(current) and cand_abs != 0.0:
                continue
            if cand_abs == 0.0:
                candidates = (0.0,)
            else:
                first = cand_abs if pref_pos else -cand_abs
                candidates = (first, -first)
            found = False
            for cand in candidates:
                cand_rect = shift_rect_y(rect0, cand)
                overlaps, _ = _count_overlaps(grid, cand_rect, ignore=g)
                if overlaps == 0:
                    chosen = cand
                    found = True
                    break
            if found:
                break
        dy[g] = float(chosen)
        grid.add(g, shift_rect_y(rect0, dy[g]))

    # Remove hidden groups from dy.
    for h in hidden:
        dy.pop(h, None)

    return dict(dy), hidden


