from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from utils.graph_utils import Edge, GraphUtils


def _default_box(box: Optional[Dict[str, float]], width: float, height: float) -> Dict[str, float]:
    if not box:
        return {"x": 0.0, "y": 0.0, "width": width, "height": height}
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "width": float(box.get("width", width)),
        "height": float(box.get("height", height)),
    }


def _center(box: Dict[str, float]) -> Tuple[float, float]:
    return box["x"] + box["width"] / 2.0, box["y"] + box["height"] / 2.0


def layout_vertices(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    *,
    layout: Optional[str],
    placement_box: Optional[Dict[str, float]],
    canvas_width: float,
    canvas_height: float,
    root_id: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    if not vertex_ids:
        return {}

    box = _default_box(placement_box, canvas_width, canvas_height)
    strategy = (layout or "circular").lower()
    if strategy == "grid":
        return _grid_layout(vertex_ids, box)
    if strategy in ("hierarchical", "tree") and root_id is not None:
        return _hierarchical_layout(vertex_ids, edges, box, root_id)
    if strategy == "radial" and root_id is not None:
        return _radial_layout(vertex_ids, edges, box, root_id)
    return _circular_layout(vertex_ids, box)


def _circular_layout(vertex_ids: List[str], box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    cx, cy = _center(box)
    radius = 0.4 * min(box["width"], box["height"])
    if radius <= 0:
        radius = 1.0
    n = len(vertex_ids)
    positions: Dict[str, Tuple[float, float]] = {}
    for i, vid in enumerate(sorted(vertex_ids)):
        angle = 2 * math.pi * i / max(n, 1)
        positions[vid] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def _grid_layout(vertex_ids: List[str], box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    n = len(vertex_ids)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    step_x = box["width"] / max(cols, 1)
    step_y = box["height"] / max(rows, 1)
    positions: Dict[str, Tuple[float, float]] = {}
    for idx, vid in enumerate(vertex_ids):
        row = idx // cols
        col = idx % cols
        x = box["x"] + (col + 0.5) * step_x
        y = box["y"] + (row + 0.5) * step_y
        positions[vid] = (x, y)
    return positions


def _hierarchical_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    root_id: str,
) -> Dict[str, Tuple[float, float]]:
    adjacency = GraphUtils.build_adjacency_map(edges)
    levels = GraphUtils.tree_levels(root_id, adjacency) or []
    depth_count = max(len(levels), 1)
    layer_height = box["height"] / depth_count
    positions: Dict[str, Tuple[float, float]] = {}
    for depth, level_vertices in enumerate(levels):
        count = max(len(level_vertices), 1)
        step_x = box["width"] / count
        y = box["y"] + (depth + 0.5) * layer_height
        for i, vid in enumerate(level_vertices):
            x = box["x"] + (i + 0.5) * step_x
            positions[vid] = (x, y)
    # fallback for any disconnected vertices
    missing = [v for v in vertex_ids if v not in positions]
    if missing:
        extra = _circular_layout(missing, box)
        positions.update(extra)
    return positions


def _radial_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    root_id: str,
) -> Dict[str, Tuple[float, float]]:
    adjacency = GraphUtils.build_adjacency_map(edges)
    levels = GraphUtils.tree_levels(root_id, adjacency) or []
    cx, cy = _center(box)
    max_radius = 0.45 * min(box["width"], box["height"])
    depth_count = max(len(levels), 1)
    positions: Dict[str, Tuple[float, float]] = {}
    for depth, level_vertices in enumerate(levels):
        radius = max_radius * (depth / max(depth_count - 1, 1))
        count = max(len(level_vertices), 1)
        for i, vid in enumerate(level_vertices):
            angle = 2 * math.pi * i / count
            positions[vid] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    missing = [v for v in vertex_ids if v not in positions]
    if missing:
        extra = _circular_layout(missing, box)
        positions.update(extra)
    return positions



