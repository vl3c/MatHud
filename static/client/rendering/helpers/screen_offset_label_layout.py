from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

Rect = Tuple[float, float, float, float]


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return out


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def shift_rect_y(rect: Rect, dy: float) -> Rect:
    shift = _coerce_float(dy, 0.0)
    if shift == 0.0:
        return rect
    min_x, max_x, min_y, max_y = rect
    return (min_x, max_x, min_y + shift, max_y + shift)


def rects_intersect(a: Rect, b: Rect) -> bool:
    # Touching edges are treated as non-overlapping so blocks can stack flush.
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


class LabelTextCall:
    __slots__ = (
        "group",
        "order",
        "text",
        "position",
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
    __slots__ = ("group", "order", "base_rect", "step")

    def __init__(self, *, group: Any, order: int, base_rect: Rect, step: float) -> None:
        self.group = group
        self.order = int(order)
        self.base_rect = base_rect
        self.step = _coerce_float(step, 1.0)


class SpatialHash2D:
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

    fallback_group = (str(text), float(position[0]), float(position[1]))
    group = _safe_hashable_group(point_meta.get("layout_group"), fallback=fallback_group)

    return LabelTextCall(
        group=group,
        order=int(order),
        text=str(text),
        position=(float(position[0]), float(position[1])),
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

    def _pick_best_dy(self, mover: Any) -> float:
        base = self._base_rect[mover]
        step_px = self._step[mover]
        max_steps = self._max_steps
        grid = self._grid
        prefer_positive = bool(self._prefer_positive.get(mover, True))

        def count_for(candidate_dy: float) -> int:
            candidate_rect = shift_rect_y(base, candidate_dy)
            count, _ = _count_overlaps(grid, candidate_rect)
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

            collision_set = set(overlapping_groups)
            collision_set.add(g)
            mover = min(collision_set, key=lambda item: _mover_key(item, dy, order))

            if grid.get_rect(mover) is None:
                iterations += 1
                continue

            grid.remove(mover)
            dy[mover] = self._pick_best_dy(mover)
            new_rect = shift_rect_y(self._base_rect[mover], dy[mover])
            grid.add(mover, new_rect)

            # Re-enqueue potentially affected labels.
            _, new_overlapping = _count_overlaps(grid, new_rect, ignore=mover)
            affected = set(overlapping_groups)
            affected.update(new_overlapping)
            affected.add(mover)
            affected.add(g)
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


