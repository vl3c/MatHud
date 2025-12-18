"""
Graph Layout Algorithms

Provides vertex positioning strategies for graph visualization:
- Circular: Equal angular spacing around a circle
- Grid: Regular row/column arrangement
- Radial: Concentric rings by BFS depth from root
- Tree: Reingold-Tilford style with children centered under parents
- Force-directed: Spring-electrical simulation for general graphs

All layouts fit vertices within a bounding box and return {vertex_id: (x, y)}.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from utils.graph_utils import Edge, GraphUtils


# =============================================================================
# Box Utilities
# =============================================================================

def _default_box(box: Optional[Dict[str, float]], width: float, height: float) -> Dict[str, float]:
    """Normalize placement box, using canvas dimensions as fallback."""
    if not box:
        return {"x": 0.0, "y": 0.0, "width": width, "height": height}
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "width": float(box.get("width", width)),
        "height": float(box.get("height", height)),
    }


def _center(box: Dict[str, float]) -> Tuple[float, float]:
    """Return the center point of a box."""
    return box["x"] + box["width"] / 2.0, box["y"] + box["height"] / 2.0


def _add_missing_vertices(
    positions: Dict[str, Tuple[float, float]],
    vertex_ids: List[str],
    box: Dict[str, float],
) -> None:
    """Place any disconnected vertices using circular layout."""
    missing = [v for v in vertex_ids if v not in positions]
    if missing:
        extra = _circular_layout(missing, box)
        positions.update(extra)


# =============================================================================
# Layout Selector
# =============================================================================

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
    """
    Compute vertex positions using the specified layout strategy.
    
    Selection priority:
    1. Explicit layout name (grid, circular, radial, tree, force)
    2. If root provided → tree layout
    3. If edges exist → force-directed layout
    4. Otherwise → circular layout
    """
    if not vertex_ids:
        return {}

    box = _default_box(placement_box, canvas_width, canvas_height)
    strategy = (layout or "").lower()
    
    # Explicit layout selection
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
    
    # Auto-selection based on graph structure
    if root_id is not None:
        return _tree_layout(vertex_ids, edges, box, root_id)
    if edges:
        return _force_directed_layout(vertex_ids, edges, box)
    return _circular_layout(vertex_ids, box)


# =============================================================================
# Circular Layout
# =============================================================================

def _circular_layout(vertex_ids: List[str], box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Place vertices evenly spaced around a circle.
    
    Algorithm: Divide 2π radians equally among n vertices.
    """
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


# =============================================================================
# Grid Layout
# =============================================================================

def _grid_layout(vertex_ids: List[str], box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Arrange vertices in a rectangular grid.
    
    Algorithm: Compute near-square grid dimensions, center each cell.
    """
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


# =============================================================================
# Radial Layout
# =============================================================================

def _radial_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    root_id: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Place vertices in concentric rings by BFS depth from root.
    
    Algorithm:
    1. Compute BFS levels from root
    2. Root at center, each level on progressively larger circles
    3. Vertices at same level spaced evenly on their ring
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    levels = GraphUtils.tree_levels(root_id, adjacency) or []
    
    cx, cy = _center(box)
    max_radius = 0.45 * min(box["width"], box["height"])
    depth_count = max(len(levels), 1)
    
    positions: Dict[str, Tuple[float, float]] = {}
    for depth, level_vertices in enumerate(levels):
        radius = max_radius * (depth / max(depth_count - 1, 1)) if depth_count > 1 else 0
        count = max(len(level_vertices), 1)
        for i, vid in enumerate(level_vertices):
            angle = 2 * math.pi * i / count
            positions[vid] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    
    _add_missing_vertices(positions, vertex_ids, box)
    return positions


# =============================================================================
# Tree Layout (Reingold-Tilford Style)
# =============================================================================

def _tree_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    root_id: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Hierarchical tree layout with children centered under parents.
    
    Algorithm (simplified Reingold-Tilford):
    1. Build parent/children structure from root
    2. Bottom-up: compute subtree widths (leaves = 1, internal = sum of children)
    3. Top-down: assign x by centering node over its subtree's allocated width
    4. Y determined by depth (root at top = highest y in math coords)
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    rooted = GraphUtils.root_tree(adjacency, root_id)
    if rooted is None:
        return _circular_layout(vertex_ids, box)
    
    parent, children = rooted
    depths = GraphUtils.node_depths(root_id, adjacency) or {}
    max_depth = max(depths.values()) if depths else 0
    
    # Phase 1: Compute subtree widths bottom-up
    subtree_widths = _compute_subtree_widths(root_id, children)
    
    # Phase 2: Compute layout parameters
    total_width = subtree_widths.get(root_id, 1.0)
    scale_x = box["width"] / total_width if total_width > 0 else 1.0
    layer_count = max_depth + 1
    layer_height = box["height"] / max(layer_count, 1)
    
    # Phase 3: Assign positions top-down
    positions = _assign_tree_positions(
        root_id, children, subtree_widths,
        box["x"], box["y"], box["height"],
        scale_x, layer_height
    )
    
    _add_missing_vertices(positions, vertex_ids, box)
    return positions


def _compute_subtree_widths(root: str, children: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute width of each subtree (number of leaves in subtree).
    
    Leaf nodes have width 1. Internal nodes sum their children's widths.
    """
    widths: Dict[str, float] = {}
    
    def compute(node: str) -> float:
        node_children = children.get(node, [])
        if not node_children:
            widths[node] = 1.0
            return 1.0
        total = sum(compute(c) for c in node_children)
        widths[node] = total
        return total
    
    compute(root)
    return widths


def _assign_tree_positions(
    root: str,
    children: Dict[str, List[str]],
    subtree_widths: Dict[str, float],
    box_x: float,
    box_y: float,
    box_height: float,
    scale_x: float,
    layer_height: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Assign (x, y) positions top-down with children centered under parent.
    
    Each node is centered within its allocated horizontal span.
    Y decreases with depth (root at top in math coordinates).
    """
    positions: Dict[str, Tuple[float, float]] = {}
    
    def assign(node: str, left_x: float, depth: int) -> None:
        node_width = subtree_widths.get(node, 1.0)
        center_x = left_x + (node_width * scale_x) / 2.0
        y = box_y + box_height - (depth + 0.5) * layer_height
        positions[node] = (center_x, y)
        
        # Recursively position children left-to-right
        child_left = left_x
        for child in children.get(node, []):
            child_width = subtree_widths.get(child, 1.0)
            assign(child, child_left, depth + 1)
            child_left += child_width * scale_x
    
    assign(root, box_x, 0)
    return positions


# =============================================================================
# Force-Directed Layout (Fruchterman-Reingold Style)
# =============================================================================

def _force_directed_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
    iterations: int = 50,
) -> Dict[str, Tuple[float, float]]:
    """
    Spring-electrical model for general graph layout.
    
    Algorithm (Fruchterman-Reingold):
    1. Initialize positions (circular)
    2. Each iteration:
       a. Compute repulsion between all pairs (Coulomb's law: k²/d)
       b. Compute attraction along edges (Hooke's law: d²/k)
       c. Apply displacement capped by temperature
       d. Cool temperature (simulated annealing)
    3. Clamp positions to bounding box
    """
    n = len(vertex_ids)
    if n == 0:
        return {}
    if n == 1:
        cx, cy = _center(box)
        return {vertex_ids[0]: (cx, cy)}
    
    # Initialize with circular layout for determinism
    positions = _circular_layout(vertex_ids, box)
    
    # Optimal distance between nodes
    area = box["width"] * box["height"]
    k = math.sqrt(area / n)
    
    # Simulated annealing parameters
    temp = box["width"] / 10.0
    cooling = temp / (iterations + 1)
    
    for _ in range(iterations):
        displacement = _compute_forces(vertex_ids, edges, positions, k)
        positions = _apply_displacement(vertex_ids, positions, displacement, temp, box)
        temp = max(temp - cooling, 0.01)
    
    return positions


def _compute_forces(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    positions: Dict[str, Tuple[float, float]],
    k: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute net force on each vertex from repulsion and attraction.
    
    Repulsion: All pairs repel with force k²/distance (Coulomb-like)
    Attraction: Connected pairs attract with force distance²/k (spring-like)
    """
    displacement: Dict[str, Tuple[float, float]] = {vid: (0.0, 0.0) for vid in vertex_ids}
    
    # Repulsion between all pairs
    for i, v1 in enumerate(vertex_ids):
        x1, y1 = positions[v1]
        for v2 in vertex_ids[i + 1:]:
            x2, y2 = positions[v2]
            dx, dy, dist = _vector_between(x1, y1, x2, y2)
            
            repulsion = (k * k) / dist
            fx = (dx / dist) * repulsion
            fy = (dy / dist) * repulsion
            
            d1x, d1y = displacement[v1]
            d2x, d2y = displacement[v2]
            displacement[v1] = (d1x + fx, d1y + fy)
            displacement[v2] = (d2x - fx, d2y - fy)
    
    # Attraction along edges
    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        dx, dy, dist = _vector_between(x2, y2, x1, y1)  # Note: reversed for attraction
        
        if dist < 0.01:
            continue
        
        attraction = (dist * dist) / k
        fx = (dx / dist) * attraction
        fy = (dy / dist) * attraction
        
        d1x, d1y = displacement[edge.source]
        d2x, d2y = displacement[edge.target]
        displacement[edge.source] = (d1x + fx, d1y + fy)
        displacement[edge.target] = (d2x - fx, d2y - fy)
    
    return displacement


def _vector_between(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    """Compute (dx, dy, distance) from point 2 to point 1, with minimum distance clamping."""
    dx = x1 - x2
    dy = y1 - y2
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        dist = 0.01
        dx = 0.01
    return dx, dy, dist


def _apply_displacement(
    vertex_ids: List[str],
    positions: Dict[str, Tuple[float, float]],
    displacement: Dict[str, Tuple[float, float]],
    temp: float,
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Apply displacement to positions, capped by temperature and clamped to box.
    
    Temperature limits maximum movement per iteration (simulated annealing).
    """
    new_positions: Dict[str, Tuple[float, float]] = {}
    
    for vid in vertex_ids:
        dx, dy = displacement[vid]
        disp_len = math.sqrt(dx * dx + dy * dy)
        
        # Cap displacement by temperature
        if disp_len > 0:
            capped = min(disp_len, temp)
            dx = (dx / disp_len) * capped
            dy = (dy / disp_len) * capped
        
        # Apply and clamp to box
        x, y = positions[vid]
        new_x = max(box["x"], min(box["x"] + box["width"], x + dx))
        new_y = max(box["y"], min(box["y"] + box["height"], y + dy))
        new_positions[vid] = (new_x, new_y)
    
    return new_positions
