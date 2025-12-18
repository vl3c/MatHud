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
    strategy = (layout or "").lower()
    
    if strategy == "grid":
        return _grid_layout(vertex_ids, box)
    if strategy == "circular":
        return _circular_layout(vertex_ids, box)
    if strategy == "radial" and root_id is not None:
        return _radial_layout(vertex_ids, edges, box, root_id)
    if strategy in ("hierarchical", "tree") and root_id is not None:
        return _tree_layout(vertex_ids, edges, box, root_id)
    if strategy == "force":
        return _force_directed_layout(vertex_ids, edges, box)
    
    if root_id is not None:
        return _tree_layout(vertex_ids, edges, box, root_id)
    if edges:
        return _force_directed_layout(vertex_ids, edges, box)
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


def _tree_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    root_id: str,
) -> Dict[str, Tuple[float, float]]:
    """Proper tree layout with children centered under parents.
    
    Uses a simplified Reingold-Tilford approach:
    1. Build tree structure (parent/children)
    2. Compute subtree widths bottom-up
    3. Assign x-positions top-down with centering
    4. Root at top, equal layer heights
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    rooted = GraphUtils.root_tree(adjacency, root_id)
    if rooted is None:
        return _circular_layout(vertex_ids, box)
    
    parent, children = rooted
    depths = GraphUtils.node_depths(root_id, adjacency) or {}
    max_depth = max(depths.values()) if depths else 0
    
    subtree_widths: Dict[str, float] = {}
    min_node_spacing = 1.0
    
    def compute_subtree_width(node: str) -> float:
        node_children = children.get(node, [])
        if not node_children:
            subtree_widths[node] = min_node_spacing
            return min_node_spacing
        total = sum(compute_subtree_width(c) for c in node_children)
        subtree_widths[node] = total
        return total
    
    compute_subtree_width(root_id)
    
    total_width = subtree_widths.get(root_id, min_node_spacing)
    scale_x = box["width"] / total_width if total_width > 0 else 1.0
    layer_count = max_depth + 1
    layer_height = box["height"] / max(layer_count, 1)
    
    positions: Dict[str, Tuple[float, float]] = {}
    
    def assign_positions(node: str, left_x: float, depth: int) -> None:
        node_width = subtree_widths.get(node, min_node_spacing)
        center_x = left_x + (node_width * scale_x) / 2.0
        y = box["y"] + box["height"] - (depth + 0.5) * layer_height
        positions[node] = (center_x, y)
        
        node_children = children.get(node, [])
        child_left = left_x
        for child in node_children:
            child_width = subtree_widths.get(child, min_node_spacing)
            assign_positions(child, child_left, depth + 1)
            child_left += child_width * scale_x
    
    assign_positions(root_id, box["x"], 0)
    
    missing = [v for v in vertex_ids if v not in positions]
    if missing:
        extra = _circular_layout(missing, box)
        positions.update(extra)
    
    return positions


def _force_directed_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    iterations: int = 50,
) -> Dict[str, Tuple[float, float]]:
    """Force-directed layout that places neighbors close and minimizes edge crossings.
    
    Uses spring-electrical model:
    - All nodes repel each other (prevents overlap)
    - Connected nodes attract each other (keeps neighbors close)
    - Iterates until positions stabilize
    """
    n = len(vertex_ids)
    if n == 0:
        return {}
    if n == 1:
        cx, cy = _center(box)
        return {vertex_ids[0]: (cx, cy)}
    
    positions = _circular_layout(vertex_ids, box)
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    area = box["width"] * box["height"]
    k = math.sqrt(area / n)
    
    temp = box["width"] / 10.0
    cooling = temp / (iterations + 1)
    
    for _ in range(iterations):
        displacement: Dict[str, Tuple[float, float]] = {vid: (0.0, 0.0) for vid in vertex_ids}
        
        for i, v1 in enumerate(vertex_ids):
            x1, y1 = positions[v1]
            for v2 in vertex_ids[i + 1:]:
                x2, y2 = positions[v2]
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 0.01:
                    dist = 0.01
                    dx = 0.01
                repulsion = (k * k) / dist
                fx = (dx / dist) * repulsion
                fy = (dy / dist) * repulsion
                d1x, d1y = displacement[v1]
                d2x, d2y = displacement[v2]
                displacement[v1] = (d1x + fx, d1y + fy)
                displacement[v2] = (d2x - fx, d2y - fy)
        
        for edge in edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            x1, y1 = positions[edge.source]
            x2, y2 = positions[edge.target]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                continue
            attraction = (dist * dist) / k
            fx = (dx / dist) * attraction
            fy = (dy / dist) * attraction
            d1x, d1y = displacement[edge.source]
            d2x, d2y = displacement[edge.target]
            displacement[edge.source] = (d1x + fx, d1y + fy)
            displacement[edge.target] = (d2x - fx, d2y - fy)
        
        for vid in vertex_ids:
            dx, dy = displacement[vid]
            disp_len = math.sqrt(dx * dx + dy * dy)
            if disp_len > 0:
                capped = min(disp_len, temp)
                dx = (dx / disp_len) * capped
                dy = (dy / disp_len) * capped
            x, y = positions[vid]
            new_x = max(box["x"], min(box["x"] + box["width"], x + dx))
            new_y = max(box["y"], min(box["y"] + box["height"], y + dy))
            positions[vid] = (new_x, new_y)
        
        temp -= cooling
        if temp < 0:
            temp = 0.01
    
    return positions



