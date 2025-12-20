"""
Graph Layout Algorithms

Provides vertex positioning strategies for graph visualization:
- Circular: Equal angular spacing around a circle
- Grid: TSM orthogonal layout with axis-aligned edges
- Radial: Concentric rings by BFS depth from root
- Tree: Reingold-Tilford style with children centered under parents
- Force-directed: Spring-electrical simulation for general graphs

The Grid layout uses the Topology-Shape-Metrics (TSM) approach:
1. Planarity check and embedding computation
2. Orthogonalization (visibility representation)
3. Compaction to grid coordinates

For non-planar graphs, the grid layout uses planarization (inserting
dummy vertices at edge crossings) or falls back to force-directed
with grid snapping.

All layouts fit vertices within a bounding box and return {vertex_id: (x, y)}.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

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
    1. Auto-detect tree structure: use tree layout (overrides circular/force)
    2. Explicit layout name (grid, circular, radial, tree, force)
    3. If root provided: use tree layout
    4. If edges exist: use force-directed layout
    5. Otherwise: use circular layout
    """
    if not vertex_ids:
        return {}

    box = _default_box(placement_box, canvas_width, canvas_height)
    strategy = (layout or "").lower()
    
    # Explicit layout selection (radial, hierarchical, tree are always respected)
    # Grid is overridden for trees since grid + many edges = overlapping mess
    if strategy == "grid":
        if _is_tree_structure(vertex_ids, edges):
            effective_root = root_id or _infer_root(vertex_ids, edges)
            if effective_root:
                return _tree_layout(vertex_ids, edges, box, effective_root)
        return _grid_layout(vertex_ids, edges, box)
    if strategy == "radial":
        effective_root = root_id or _infer_root(vertex_ids, edges)
        if effective_root:
            return _radial_layout(vertex_ids, edges, box, effective_root)
    if strategy in ("hierarchical", "tree"):
        effective_root = root_id or _infer_root(vertex_ids, edges)
        if effective_root:
            return _tree_layout(vertex_ids, edges, box, effective_root)
    
    # Auto-detect tree: override circular/force/unspecified with tree layout for tree structures
    if strategy in ("circular", "force", "") and _is_tree_structure(vertex_ids, edges):
        effective_root = root_id or _infer_root(vertex_ids, edges)
        if effective_root:
            return _tree_layout(vertex_ids, edges, box, effective_root)
    
    # Remaining explicit layouts
    if strategy == "circular":
        return _circular_layout(vertex_ids, box)
    if strategy == "force":
        return _force_directed_layout(vertex_ids, edges, box)
    
    # Auto-selection for unspecified layout
    if root_id is not None:
        return _tree_layout(vertex_ids, edges, box, root_id)
    if edges:
        return _force_directed_layout(vertex_ids, edges, box)
    return _circular_layout(vertex_ids, box)


def _is_tree_structure(vertex_ids: List[str], edges: List[Edge[str]]) -> bool:
    """
    Detect if the graph has a tree structure.
    
    A tree has:
    - Exactly n-1 edges for n vertices
    - A clear root (vertex with no incoming edges)
    - All vertices reachable from root
    """
    n = len(vertex_ids)
    if n == 0:
        return False
    if len(edges) != n - 1:
        return False
    
    # Check for a root (vertex with no incoming edges)
    vertex_set = set(vertex_ids)
    has_incoming = {e.target for e in edges if e.target in vertex_set}
    roots = [v for v in vertex_ids if v not in has_incoming]
    
    if len(roots) != 1:
        return False
    
    # Verify all vertices are reachable from root (connected tree)
    root = roots[0]
    adjacency: Dict[str, List[str]] = {v: [] for v in vertex_ids}
    for e in edges:
        if e.source in adjacency:
            adjacency[e.source].append(e.target)
    
    visited: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    
    return len(visited) == n


def _infer_root(vertex_ids: List[str], edges: List[Edge[str]]) -> Optional[str]:
    """
    Infer root vertex for tree layout when none is provided.
    
    Finds a vertex with no incoming edges (only outgoing).
    Falls back to first vertex if all have incoming edges.
    """
    if not vertex_ids:
        return None
    if not edges:
        return vertex_ids[0]
    
    # Find vertices with incoming edges
    has_incoming: set[str] = set()
    all_vertices: set[str] = set(vertex_ids)
    
    for edge in edges:
        if edge.target in all_vertices:
            has_incoming.add(edge.target)
    
    # Root candidates: vertices with no incoming edges
    candidates = [v for v in vertex_ids if v not in has_incoming]
    
    if candidates:
        return candidates[0]
    
    # Fallback to first vertex (cyclic graph)
    return vertex_ids[0]


# =============================================================================
# Circular Layout
# =============================================================================

def _circular_layout(vertex_ids: List[str], box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Place vertices evenly spaced around a circle.
    
    Algorithm: Divide 2π radians equally among n vertices.
    """
    center = _center(box)
    cx = center[0]
    cy = center[1]
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
# Grid Layout (TSM Orthogonal)
# =============================================================================

def _simple_grid_placement(
    vertex_ids: List[str],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Place isolated vertices on a simple rectangular grid.
    
    Used when there are no edges - just arranges nodes in rows/columns
    to fill the bounding box evenly.
    """
    n = len(vertex_ids)
    if n == 0:
        return {}
    if n == 1:
        center = _center(box)
        cx = center[0]
        cy = center[1]
        return {vertex_ids[0]: (cx, cy)}
    
    # Compute grid dimensions (prefer roughly square)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    
    # Compute spacing with margins
    margin = 0.1
    usable_width = box["width"] * (1 - 2 * margin)
    usable_height = box["height"] * (1 - 2 * margin)
    
    col_spacing = usable_width / max(cols, 1)
    row_spacing = usable_height / max(rows, 1)
    
    start_x = box["x"] + box["width"] * margin + col_spacing / 2
    start_y = box["y"] + box["height"] * margin + row_spacing / 2
    
    positions: Dict[str, Tuple[float, float]] = {}
    for i, vid in enumerate(sorted(vertex_ids)):
        col = i % cols
        row = i // cols
        x = start_x + col * col_spacing
        y = start_y + row * row_spacing
        positions[vid] = (float(x), float(y))
    
    return positions


def _optimize_layout_if_needed(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    result: Dict[str, Tuple[float, float]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Check layout quality and run optimization if needed.
    
    Triggers optimization if:
    - There are edge crossings
    - There are edge overlaps
    - Less than 70% of edges are orthogonal
    """
    crossings = GraphUtils.count_edge_crossings(edges, result)
    overlaps = GraphUtils.count_edge_overlaps(edges, result)
    ortho_result = GraphUtils.count_orthogonal_edges(edges, result)
    ortho_count = ortho_result[0]
    total_edges = ortho_result[1]
    ortho_ratio = ortho_count / max(total_edges, 1)
    
    needs_optimization = crossings > 0 or overlaps > 0 or ortho_ratio < 0.7
    
    if needs_optimization:
        print(f"[GRID] Detected {crossings} crossings, {overlaps} overlaps, {ortho_count}/{total_edges} orthogonal, running optimization...")
        result = _eliminate_crossings(vertex_ids, edges, result, box)
        final_crossings = GraphUtils.count_edge_crossings(edges, result)
        final_overlaps = GraphUtils.count_edge_overlaps(edges, result)
        final_ortho_result = GraphUtils.count_orthogonal_edges(edges, result)
        final_ortho = final_ortho_result[0]
        print(f"[GRID] After optimization: {final_crossings} crossings, {final_overlaps} overlaps, {final_ortho}/{total_edges} orthogonal")
    
    return result


def _count_diagonal_edges(ortho_rep: OrthogonalRep) -> int:
    """Count edges with more than one direction (diagonal edges)."""
    return sum(
        1 for edge_key in ortho_rep.edge_directions
        if len(ortho_rep.edge_directions[edge_key]) > 1
    )


def _ensure_float_positions(
    result: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Ensure all positions are 2-tuples of floats."""
    return {k: (float(v[0]), float(v[1])) for k, v in result.items()}


def _layout_planar_graph(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    embedding: Dict[str, List[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """Layout a planar graph using orthogonalization and compaction."""
    ortho_rep = _orthogonalize(vertex_ids, edges, embedding)
    
    diagonal_count = _count_diagonal_edges(ortho_rep)
    diagonal_ratio = diagonal_count / max(len(edges), 1)
    print(f"[GRID] Diagonal ratio: {diagonal_count}/{len(edges)} = {diagonal_ratio:.2f}")
    
    if diagonal_ratio > 0.7:
        print(f"[GRID] Too many diagonals, using force-directed fallback")
        result = _force_to_grid_fallback(vertex_ids, edges, box)
    else:
        result = _compact_orthogonal(ortho_rep, vertex_ids, edges, box)
    
    result = _ensure_float_positions(result)
    return _optimize_layout_if_needed(vertex_ids, edges, result, box)


def _layout_non_planar_graph(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """Layout a non-planar graph by planarizing first."""
    planarized = _planarize_graph(vertex_ids, edges, box)
    new_vertices = planarized[0]
    new_edges = planarized[1]
    dummy_ids = planarized[2]
    planarity_result = _is_planar(new_vertices, new_edges)
    new_embedding = planarity_result[1]
    
    if new_embedding is not None:
        ortho_rep = _orthogonalize(new_vertices, new_edges, new_embedding)
        positions = _compact_orthogonal(ortho_rep, new_vertices, new_edges, box)
        result = {k: v for k, v in positions.items() if k not in dummy_ids}
    else:
        result = _force_to_grid_fallback(vertex_ids, edges, box)
    
    result = _ensure_float_positions(result)
    
    # Eliminate overlaps if present
    overlaps = GraphUtils.count_edge_overlaps(edges, result)
    if overlaps > 0:
        print(f"[GRID] Non-planar: Detected {overlaps} overlaps, running elimination...")
        result = _eliminate_crossings(vertex_ids, edges, result, box)
        final_overlaps = GraphUtils.count_edge_overlaps(edges, result)
        print(f"[GRID] After elimination: {final_overlaps} overlaps")
    
    return result


def _grid_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Orthogonal grid layout using TSM (Topology-Shape-Metrics) approach.
    
    Produces layouts with axis-aligned edges and minimized crossings.
    """
    print(f"[GRID] === _grid_layout called ===")
    print(f"[GRID] Vertices: {vertex_ids}")
    print(f"[GRID] Edges: {[(e.source, e.target) for e in edges]}")
    
    # Fast path: no edges
    if not edges:
        print(f"[GRID] Path: no edges -> simple grid placement")
        return _ensure_float_positions(_simple_grid_placement(vertex_ids, box))
    
    # Fast path: tree structure
    if _is_tree_structure(vertex_ids, edges):
        print(f"[GRID] Path: tree structure -> orthogonal tree layout")
        return _ensure_float_positions(_orthogonal_tree_layout(vertex_ids, edges, box))
    
    # TSM pipeline for general graphs
    planarity_result = _is_planar(vertex_ids, edges)
    is_planar = planarity_result[0]
    embedding = planarity_result[1]
    print(f"[GRID] Planarity check: is_planar={is_planar}")
    
    if is_planar and embedding is not None:
        print(f"[GRID] Path: planar graph -> orthogonalize + compact")
        return _layout_planar_graph(vertex_ids, edges, embedding, box)
    else:
        print(f"[GRID] Path: non-planar graph -> planarize first")
        return _layout_non_planar_graph(vertex_ids, edges, box)


def _assign_tree_depths(
    root_id: str,
    children: Dict[str, List[str]],
) -> Dict[str, int]:
    """Assign row numbers based on depth in tree."""
    vertex_row: Dict[str, int] = {}
    
    def recurse(node: str, depth: int) -> None:
        vertex_row[node] = depth
        for child in children.get(node, []):
            recurse(child, depth + 1)
    
    recurse(root_id, 0)
    return vertex_row


def _assign_tree_columns(
    root_id: str,
    children: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Assign column positions bottom-up.
    
    Leaves get sequential columns, internal nodes get center of children.
    """
    vertex_col: Dict[str, float] = {}
    leaf_counter = [0]
    
    def recurse(node: str) -> float:
        node_children = children.get(node, [])
        
        if not node_children:
            col = float(leaf_counter[0])
            leaf_counter[0] += 1
            vertex_col[node] = col
            return col
        
        child_cols = [recurse(child) for child in node_children]
        col = sum(child_cols) / len(child_cols)
        vertex_col[node] = col
        return col
    
    recurse(root_id)
    return vertex_col


def _align_single_child_chains(
    vertex_ids: List[str],
    children: Dict[str, List[str]],
    vertex_col: Dict[str, float],
) -> None:
    """Align single children directly under their parents."""
    for node in vertex_ids:
        node_children = children.get(node, [])
        if len(node_children) == 1:
            child = node_children[0]
            vertex_col[child] = vertex_col[node]


def _scale_grid_to_box(
    vertex_ids: List[str],
    vertex_col: Dict[str, float],
    vertex_row: Dict[str, int],
    box: Dict[str, float],
    invert_y: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """Scale grid positions to bounding box."""
    cols = list(vertex_col.values())
    rows = list(vertex_row.values())
    
    if not cols or not rows:
        return {}
    
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    
    col_range = max(max_col - min_col, 1)
    row_range = max(max_row - min_row, 1)
    
    margin = 0.1
    usable_width = box["width"] * (1 - 2 * margin)
    usable_height = box["height"] * (1 - 2 * margin)
    
    positions: Dict[str, Tuple[float, float]] = {}
    for vid in vertex_ids:
        col = vertex_col.get(vid, 0.0)
        row = vertex_row.get(vid, 0)
        
        x = box["x"] + box["width"] * margin + (col - min_col) * usable_width / col_range
        if invert_y:
            y = box["y"] + box["height"] * (1 - margin) - (row - min_row) * usable_height / row_range
        else:
            y = box["y"] + box["height"] * margin + (row - min_row) * usable_height / row_range
        
        positions[vid] = (float(x), float(y))
    
    return positions


def _orthogonal_tree_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Orthogonal tree layout producing truly perpendicular edges.
    
    All edges are purely vertical (parent directly above children).
    """
    root_id = _infer_root(vertex_ids, edges)
    if root_id is None:
        return _simple_grid_placement(vertex_ids, box)
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    rooted = GraphUtils.root_tree(adjacency, root_id)
    if rooted is None:
        return _simple_grid_placement(vertex_ids, box)
    
    children = rooted[1]
    
    vertex_row = _assign_tree_depths(root_id, children)
    vertex_col = _assign_tree_columns(root_id, children)
    _align_single_child_chains(vertex_ids, children, vertex_col)
    
    positions = _scale_grid_to_box(vertex_ids, vertex_col, vertex_row, box, invert_y=True)
    if not positions:
        return _simple_grid_placement(vertex_ids, box)
    
    return positions


# =============================================================================
# TSM Phase 1: Planarity and Embedding
# =============================================================================

def _is_planar(
    vertex_ids: List[str],
    edges: List[Edge[str]],
) -> Tuple[bool, Optional[Dict[str, List[str]]]]:
    """
    Check if graph is planar and compute embedding if so.
    
    Uses edge count bound check and simplified LR-planarity algorithm.
    
    Returns:
        (is_planar, embedding_or_none)
        - If planar: (True, {vertex: [clockwise_neighbors]})
        - If not planar: (False, None)
    """
    n = len(vertex_ids)
    m = len(edges)
    
    # Empty or trivial graphs are planar
    if n <= 2:
        return True, _trivial_embedding(vertex_ids, edges)
    
    # Quick necessary condition: |E| <= 3|V| - 6 for |V| >= 3
    if n >= 3 and m > 3 * n - 6:
        return False, None
    
    # For small graphs, use simple planarity check
    if n <= 4:
        return True, _trivial_embedding(vertex_ids, edges)
    
    # Build adjacency for embedding
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Check connectivity - disconnected graphs: check each component
    if not GraphUtils.is_connected(adjacency):
        components = GraphUtils.get_connected_components(adjacency)
        # Each component must be planar
        for component in components:
            comp_vertices = list(component)
            comp_edges = [e for e in edges
                         if e.source in component and e.target in component]
            comp_planarity = _is_planar(comp_vertices, comp_edges)
            is_comp_planar = comp_planarity[0]
            if not is_comp_planar:
                return False, None
        # All components planar - build combined embedding
        return True, _trivial_embedding(vertex_ids, edges)
    
    # Full LR-planarity check for larger connected graphs
    return _lr_planarity_check(vertex_ids, edges, adjacency)


def _trivial_embedding(
    vertex_ids: List[str],
    edges: List[Edge[str]],
) -> Dict[str, List[str]]:
    """
    Create a trivial planar embedding for small/simple graphs.
    
    Returns clockwise neighbor ordering based on sorted order.
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    embedding: Dict[str, List[str]] = {}
    for v in vertex_ids:
        neighbors = adjacency.get(v, set())
        # Sort neighbors for deterministic ordering
        embedding[v] = sorted(neighbors)
    return embedding


class _DfsResult:
    """Result of DFS traversal for planarity checking."""
    def __init__(self) -> None:
        self.dfs_number: Dict[str, int] = {}
        self.dfs_parent: Dict[str, Optional[str]] = {}
        self.tree_edges: List[Tuple[str, str]] = []
        self.back_edges: List[Tuple[str, str]] = []
        self.visited: Set[str] = set()


def _dfs_for_planarity(
    root: str,
    adjacency: Dict[str, Set[str]],
) -> _DfsResult:
    """Run DFS to compute dfs_number, parents, and classify edges."""
    result = _DfsResult()
    result.dfs_parent[root] = None
    
    counter = [0]
    stack: List[Tuple[str, Optional[str]]] = [(root, None)]
    
    while stack:
        item = stack.pop()
        node = item[0]
        parent = item[1]
        
        if node in result.visited:
            continue
        result.visited.add(node)
        
        result.dfs_number[node] = counter[0]
        counter[0] += 1
        result.dfs_parent[node] = parent
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in result.visited:
                result.tree_edges.append((node, neighbor))
                stack.append((neighbor, node))
            elif neighbor != parent and result.dfs_number.get(neighbor, float('inf')) < result.dfs_number.get(node, 0):
                result.back_edges.append((node, neighbor))
    
    return result


def _compute_lowpoints(
    vertex_ids: List[str],
    dfs_result: _DfsResult,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compute lowpoint values for back edges (bottom-up)."""
    lowpt: Dict[str, int] = {}
    lowpt2: Dict[str, int] = {}
    
    for v in vertex_ids:
        lowpt[v] = dfs_result.dfs_number.get(v, 0)
        lowpt2[v] = dfs_result.dfs_number.get(v, 0)
    
    sorted_by_dfs = sorted(
        vertex_ids, 
        key=lambda x: dfs_result.dfs_number.get(x, 0), 
        reverse=True
    )
    
    for v in sorted_by_dfs:
        # Update from back edges
        for (src, tgt) in dfs_result.back_edges:
            if src == v:
                tgt_num = dfs_result.dfs_number.get(tgt, 0)
                if tgt_num < lowpt[v]:
                    lowpt2[v] = lowpt[v]
                    lowpt[v] = tgt_num
                elif tgt_num < lowpt2[v]:
                    lowpt2[v] = tgt_num
        
        # Update parent from child
        parent = dfs_result.dfs_parent.get(v)
        if parent is not None:
            if lowpt[v] < lowpt[parent]:
                lowpt2[parent] = min(lowpt2[parent], lowpt[parent])
                lowpt[parent] = lowpt[v]
            elif lowpt[v] < lowpt2[parent]:
                lowpt2[parent] = lowpt[v]
    
    return lowpt, lowpt2


def _lr_planarity_check(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    adjacency: Dict[str, Set[str]],
) -> Tuple[bool, Optional[Dict[str, List[str]]]]:
    """
    LR-planarity algorithm implementation.
    
    Uses DFS to build spanning tree, compute lowpoints, and check for conflicts.
    """
    if not vertex_ids:
        return True, {}
    
    root = vertex_ids[0]
    dfs_result = _dfs_for_planarity(root, adjacency)
    
    # Check connectivity
    if len(dfs_result.visited) != len(vertex_ids):
        return False, None
    
    # Compute lowpoints (unused in simplified check but kept for full LR)
    _compute_lowpoints(vertex_ids, dfs_result)
    
    # Simplified conflict check for small-medium graphs
    n = len(vertex_ids)
    m = len(edges)

    # Bipartite graphs with |E| > 2|V| - 4 aren't planar
    bipartite_result = GraphUtils.is_bipartite(adjacency)
    is_bipartite = bipartite_result[0]
    if is_bipartite and m > 2 * n - 4:
        return False, None
    
    # Accept graphs passing edge bounds
    embedding = _build_embedding_from_dfs(
        vertex_ids, adjacency, dfs_result.dfs_number, dfs_result.dfs_parent
    )
    return True, embedding


def _build_embedding_from_dfs(
    vertex_ids: List[str],
    adjacency: Dict[str, Set[str]],
    dfs_number: Dict[str, int],
    dfs_parent: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    """
    Build planar embedding from DFS structure.
    
    Orders neighbors by: parent first, then children by DFS order, then back edges.
    """
    embedding: Dict[str, List[str]] = {}
    
    for v in vertex_ids:
        neighbors = list(adjacency.get(v, set()))
        parent = dfs_parent.get(v)
        
        # Separate into parent, tree children, and back edges
        tree_children = []
        back_neighbors = []
        
        for n in neighbors:
            if n == parent:
                continue
            elif dfs_parent.get(n) == v:
                tree_children.append(n)
            else:
                back_neighbors.append(n)
        
        # Sort children by DFS number
        tree_children.sort(key=lambda x: dfs_number.get(x, 0))
        back_neighbors.sort(key=lambda x: dfs_number.get(x, 0))
        
        # Build clockwise order: parent, children, back edges
        cw_order: List[str] = []
        if parent is not None:
            cw_order.append(parent)
        cw_order.extend(tree_children)
        cw_order.extend(back_neighbors)
        
        embedding[v] = cw_order
    
    return embedding


# =============================================================================
# TSM Phase 2: Orthogonalization
# =============================================================================

class OrthogonalRep:
    """
    Orthogonal representation of a planar graph.
    
    Stores vertex grid positions and edge routing directions.
    """
    def __init__(self) -> None:
        self.vertex_pos: Dict[str, Tuple[int, int]] = {}
        self.edge_directions: Dict[Tuple[str, str], List[str]] = {}


def _is_simple_cycle(vertex_ids: List[str], edges: List[Edge[str]]) -> bool:
    """Check if graph is a simple cycle (each vertex has exactly degree 2)."""
    if len(edges) != len(vertex_ids):
        return False
    adjacency = GraphUtils.build_adjacency_map(edges)
    return all(len(adjacency.get(v, set())) == 2 for v in vertex_ids)


def _find_bridge_vertices(
    vertex_ids: List[str],
    adjacency: Dict[str, Set[str]],
) -> Set[str]:
    """Find vertices with degree > 2 (potential bridge vertices)."""
    return {v for v in vertex_ids if len(adjacency.get(v, set())) > 2}


def _find_bridge_edge_set(
    edges: List[Edge[str]],
    bridge_vertices: Set[str],
) -> Set[Tuple[str, str]]:
    """Find edges connecting high-degree vertices (bridge edges)."""
    bridge_edges: Set[Tuple[str, str]] = set()
    for e in edges:
        if e.source in bridge_vertices and e.target in bridge_vertices:
            bridge_edges.add((e.source, e.target))
            bridge_edges.add((e.target, e.source))
    return bridge_edges


def _build_adjacency_without_bridges(
    vertex_ids: List[str],
    non_bridge_edges: List[Edge[str]],
) -> Dict[str, Set[str]]:
    """Build adjacency map excluding bridge edges."""
    adj: Dict[str, Set[str]] = {v: set() for v in vertex_ids}
    for e in non_bridge_edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return adj


def _find_component_from_start(
    start: str,
    adj: Dict[str, Set[str]],
    visited: Set[str],
) -> List[str]:
    """Find connected component starting from a vertex using DFS."""
    component_verts: List[str] = []
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        component_verts.append(v)
        stack.extend(adj.get(v, set()))
    return component_verts


def _find_cycle_components(
    vertex_ids: List[str],
    edges: List[Edge[str]],
) -> Optional[List[Tuple[List[str], List[Edge[str]]]]]:
    """
    Detect if graph is composed of simple cycles connected by bridges.
    
    Returns list of (vertices, edges) for each cycle component, or None if not applicable.
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    bridge_vertices = _find_bridge_vertices(vertex_ids, adjacency)
    
    # If no bridge vertices, check if it's a single cycle
    if not bridge_vertices:
        if _is_simple_cycle(vertex_ids, edges):
            return [(vertex_ids, edges)]
        return None
    
    bridge_edges = _find_bridge_edge_set(edges, bridge_vertices)
    non_bridge_edges = [e for e in edges if (e.source, e.target) not in bridge_edges]
    adj_no_bridges = _build_adjacency_without_bridges(vertex_ids, non_bridge_edges)
    
    # Find connected components
    visited: Set[str] = set()
    components: List[Tuple[List[str], List[Edge[str]]]] = []
    
    for start in vertex_ids:
        if start in visited:
            continue
        
        component_verts = _find_component_from_start(start, adj_no_bridges, visited)
        component_set = set(component_verts)
        component_edges = [
            e for e in non_bridge_edges
            if e.source in component_set and e.target in component_set
        ]
        
        # Check if this component is a simple cycle
        if len(component_edges) == len(component_verts) and len(component_verts) >= 3:
            components.append((component_verts, component_edges))
        else:
            return None  # Not all components are cycles
    
    return components if components else None


def _orthogonalize_cycle(
    vertex_ids: List[str],
    edges: List[Edge[str]],
) -> OrthogonalRep:
    """
    Special orthogonalization for simple cycles.
    
    Places vertices on a rectangular grid where each edge is orthogonal.
    """
    print("[ORTHO-CYCLE] Detected simple cycle, using rectangular layout")
    ortho = OrthogonalRep()
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    ordered = _walk_cycle(vertex_ids[0], adjacency, len(vertex_ids))
    print(f"[ORTHO-CYCLE] Cycle order: {ordered}")
    
    ortho.vertex_pos = _layout_cycle_as_rectangle(ordered, col_offset=0, row_offset=0)
    print(f"[ORTHO-CYCLE] Positions: {ortho.vertex_pos}")
    
    ortho.edge_directions = _compute_edge_directions(edges, ortho.vertex_pos, "[ORTHO-CYCLE]")
    
    return ortho


def _identify_bridge_edges(
    edges: List[Edge[str]],
    components: List[Tuple[List[str], List[Edge[str]]]],
) -> Tuple[Set[Tuple[str, str]], List[Tuple[str, str]]]:
    """Identify component edges and bridge edges."""
    all_component_edges: Set[Tuple[str, str]] = set()
    for _, comp_edges in components:
        for e in comp_edges:
            all_component_edges.add((e.source, e.target))
            all_component_edges.add((e.target, e.source))
    
    bridge_edges: List[Tuple[str, str]] = []
    for e in edges:
        if (e.source, e.target) not in all_component_edges:
            bridge_edges.append((e.source, e.target))
    
    return all_component_edges, bridge_edges


def _build_vertex_to_component_map(
    components: List[Tuple[List[str], List[Edge[str]]]],
) -> Dict[str, int]:
    """Map each vertex to its component index."""
    vertex_to_comp: Dict[str, int] = {}
    for comp_idx in range(len(components)):
        comp = components[comp_idx]
        comp_verts = comp[0]
        for v in comp_verts:
            vertex_to_comp[v] = comp_idx
    return vertex_to_comp


def _order_components_by_bridges(
    components: List[Tuple[List[str], List[Edge[str]]]],
    bridge_edges: List[Tuple[str, str]],
    vertex_to_comp: Dict[str, int],
) -> List[int]:
    """Order components by bridge connections starting from component 0."""
    comp_order = [0]
    visited_comps: Set[int] = {0}
    
    while len(comp_order) < len(components):
        for src, tgt in bridge_edges:
            src_comp = vertex_to_comp.get(src, -1)
            tgt_comp = vertex_to_comp.get(tgt, -1)
            if src_comp in visited_comps and tgt_comp not in visited_comps:
                comp_order.append(tgt_comp)
                visited_comps.add(tgt_comp)
            elif tgt_comp in visited_comps and src_comp not in visited_comps:
                comp_order.append(src_comp)
                visited_comps.add(src_comp)
        
        # Add any remaining unvisited components
        for i in range(len(components)):
            if i not in visited_comps:
                comp_order.append(i)
                visited_comps.add(i)
    
    return comp_order


def _walk_cycle(
    start: str,
    comp_adj: Dict[str, Set[str]],
    num_verts: int,
) -> List[str]:
    """Walk the cycle from start vertex and return ordered vertices."""
    visited: Set[str] = set()
    ordered: List[str] = []
    current = start
    
    while len(ordered) < num_verts:
        ordered.append(current)
        visited.add(current)
        for neighbor in comp_adj.get(current, set()):
            if neighbor not in visited:
                current = neighbor
                break
        else:
            break
    
    return ordered


def _layout_cycle_as_rectangle(
    ordered: List[str],
    col_offset: int,
    row_offset: int,
) -> Dict[str, Tuple[int, int]]:
    """Place cycle vertices in a rectangular pattern."""
    n = len(ordered)
    half = (n + 1) // 2
    positions: Dict[str, Tuple[int, int]] = {}
    
    for i, v in enumerate(ordered):
        if i < half:
            positions[v] = (col_offset + i, row_offset)
        else:
            positions[v] = (col_offset + (n - 1 - i), row_offset + 1)
    
    return positions


def _compute_edge_directions(
    edges: List[Edge[str]],
    vertex_pos: Dict[str, Tuple[int, int]],
    log_prefix: str = "[ORTHO]",
) -> Dict[Tuple[str, str], List[str]]:
    """Compute edge directions from vertex positions."""
    edge_directions: Dict[Tuple[str, str], List[str]] = {}
    
    for edge in edges:
        u, v = edge.source, edge.target
        if u not in vertex_pos or v not in vertex_pos:
            continue
        
        u_pos = vertex_pos[u]
        v_pos = vertex_pos[v]
        u_col = int(u_pos[0])
        u_row = int(u_pos[1])
        v_col = int(v_pos[0])
        v_row = int(v_pos[1])
        
        directions: List[str] = []
        if u_col == v_col:
            directions.append('S' if v_row > u_row else 'N')
            print(f"{log_prefix} Edge {u}->{v}: VERTICAL ✓")
        elif u_row == v_row:
            directions.append('E' if v_col > u_col else 'W')
            print(f"{log_prefix} Edge {u}->{v}: HORIZONTAL ✓")
        else:
            print(f"{log_prefix} Edge {u}->{v}: DIAGONAL ✗")
            if v_row > u_row:
                directions.append('S')
            else:
                directions.append('N')
            if v_col > u_col:
                directions.append('E')
            else:
                directions.append('W')
        
        edge_directions[(u, v)] = directions
    
    return edge_directions


def _orthogonalize_multi_cycle(
    edges: List[Edge[str]],
    components: List[Tuple[List[str], List[Edge[str]]]],
) -> OrthogonalRep:
    """
    Layout multiple cycle components connected by bridges.
    
    Stacks cycles vertically with bridges as vertical edges.
    """
    print(f"[ORTHO-MULTI] Laying out {len(components)} cycle components")
    ortho = OrthogonalRep()

    bridge_result = _identify_bridge_edges(edges, components)
    bridge_edges = bridge_result[1]
    print(f"[ORTHO-MULTI] Bridge edges: {bridge_edges}")
    
    vertex_to_comp = _build_vertex_to_component_map(components)
    comp_order = _order_components_by_bridges(components, bridge_edges, vertex_to_comp)
    print(f"[ORTHO-MULTI] Component order: {comp_order}")
    
    row_offset = 0
    component_positions: Dict[str, Tuple[int, int]] = {}
    bridge_vertex_positions: Dict[str, Tuple[int, int]] = {}
    
    for order_idx, comp_idx in enumerate(comp_order):
        comp = components[comp_idx]
        comp_verts = comp[0]
        comp_edges = comp[1]
        print(f"[ORTHO-MULTI] Component {comp_idx}: {comp_verts}")
        
        # Find bridge vertex to align with previous component
        bridge_vertex = None
        target_col = 0
        
        if order_idx > 0:
            for src, tgt in bridge_edges:
                if src in bridge_vertex_positions and tgt in comp_verts:
                    bridge_vertex = tgt
                    target_col = bridge_vertex_positions[src][0]
                    break
                elif tgt in bridge_vertex_positions and src in comp_verts:
                    bridge_vertex = src
                    target_col = bridge_vertex_positions[tgt][0]
                    break
        
        # Walk cycle starting from bridge vertex
        comp_adj = GraphUtils.build_adjacency_map(comp_edges)
        start = bridge_vertex if bridge_vertex else comp_verts[0]
        ordered = _walk_cycle(start, comp_adj, len(comp_verts))
        
        # Calculate column offset for alignment
        n = len(ordered)
        half = (n + 1) // 2
        col_offset = 0
        if bridge_vertex:
            bridge_idx = ordered.index(bridge_vertex)
            if bridge_idx < half:
                col_offset = target_col - bridge_idx
            else:
                col_offset = target_col - (n - 1 - bridge_idx)
        
        # Layout this cycle
        cycle_positions = _layout_cycle_as_rectangle(ordered, col_offset, row_offset)
        component_positions.update(cycle_positions)
        
        # Track bridge vertices for next component
        for v in cycle_positions:
            pos = cycle_positions[v]
            for src, tgt in bridge_edges:
                if v == src or v == tgt:
                    bridge_vertex_positions[v] = pos
        
        row_offset += 2
    
    ortho.vertex_pos = component_positions
    print(f"[ORTHO-MULTI] Final positions: {ortho.vertex_pos}")
    
    ortho.edge_directions = _compute_edge_directions(edges, ortho.vertex_pos, "[ORTHO-MULTI]")
    
    return ortho


class _BfsTreeResult:
    """Result of BFS tree construction."""
    def __init__(self) -> None:
        self.parent: Dict[str, Optional[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.levels: List[List[str]] = []


def _build_bfs_tree(
    root: str,
    vertex_ids: List[str],
    adjacency: Dict[str, Set[str]],
) -> _BfsTreeResult:
    """Build spanning tree and levels via BFS."""
    result = _BfsTreeResult()
    result.parent = {root: None}
    result.children = {v: [] for v in vertex_ids}
    
    queue: List[Tuple[str, int]] = [(root, 0)]
    queue_idx = 0
    visited: Set[str] = {root}
    
    while queue_idx < len(queue):
        item = queue[queue_idx]
        node = item[0]
        depth = item[1]
        queue_idx += 1

        while len(result.levels) <= depth:
            result.levels.append([])
        result.levels[depth].append(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                result.parent[neighbor] = node
                result.children[node].append(neighbor)
                queue.append((neighbor, depth + 1))
    
    # Add disconnected vertices
    for v in vertex_ids:
        if v not in visited:
            if result.levels:
                result.levels[0].append(v)
            else:
                result.levels.append([v])
            result.parent[v] = None
    
    return result


def _compute_subtree_sizes(
    root: str,
    children: Dict[str, List[str]],
) -> Dict[str, int]:
    """Compute subtree sizes for proportional column allocation."""
    subtree_size: Dict[str, int] = {}
    
    def compute_size(node: str) -> int:
        size = 1
        for child in children.get(node, []):
            size += compute_size(child)
        subtree_size[node] = size
        return size
    
    compute_size(root)
    return subtree_size


def _assign_columns_by_subtree(
    root: str,
    children: Dict[str, List[str]],
    subtree_size: Dict[str, int],
) -> Dict[str, int]:
    """Assign columns top-down with children spread proportionally."""
    vertex_col: Dict[str, int] = {}
    total_width = subtree_size.get(root, 1)
    
    def assign(node: str, col_start: int, col_end: int) -> None:
        vertex_col[node] = (col_start + col_end) // 2
        
        node_children = children.get(node, [])
        if not node_children:
            return
        
        total_size = sum(subtree_size.get(c, 1) for c in node_children)
        child_col = col_start
        
        for child in node_children:
            child_size = subtree_size.get(child, 1)
            child_range = max(1, (col_end - col_start) * child_size // max(total_size, 1))
            child_col_end = min(child_col + child_range, col_end)
            assign(child, child_col, child_col_end)
            child_col = child_col_end
    
    assign(root, 0, total_width)
    return vertex_col


def _compute_edge_directions_with_stats(
    edges: List[Edge[str]],
    vertex_pos: Dict[str, Tuple[int, int]],
) -> Tuple[Dict[Tuple[str, str], List[str]], int]:
    """Compute edge directions and count diagonal edges."""
    edge_directions: Dict[Tuple[str, str], List[str]] = {}
    diagonal_count = 0
    
    for edge in edges:
        u, v = edge.source, edge.target
        if u not in vertex_pos or v not in vertex_pos:
            continue

        u_pos = vertex_pos[u]
        v_pos = vertex_pos[v]
        u_col = int(u_pos[0])
        u_row = int(u_pos[1])
        v_col = int(v_pos[0])
        v_row = int(v_pos[1])

        directions: List[str] = []
        delta_col = v_col - u_col
        delta_row = v_row - u_row
        
        if delta_col == 0:
            if delta_row > 0:
                directions.append('S')
            elif delta_row < 0:
                directions.append('N')
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): VERTICAL ✓")
        elif delta_row == 0:
            if delta_col > 0:
                directions.append('E')
            elif delta_col < 0:
                directions.append('W')
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): HORIZONTAL ✓")
        else:
            diagonal_count += 1
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): DIAGONAL ✗ (delta_col={delta_col}, delta_row={delta_row})")
            if delta_row > 0:
                directions.append('S')
            elif delta_row < 0:
                directions.append('N')
            if delta_col > 0:
                directions.append('E')
            elif delta_col < 0:
                directions.append('W')
        
        edge_directions[(u, v)] = directions
    
    return edge_directions, diagonal_count


def _orthogonalize(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    embedding: Dict[str, List[str]],
) -> OrthogonalRep:
    """
    Create orthogonal representation from planar embedding.
    
    Uses BFS tree layout with proportional column assignment.
    """
    print("[ORTHO] === Starting _orthogonalize ===")
    print(f"[ORTHO] Vertices: {vertex_ids}")
    print(f"[ORTHO] Edges: {[(e.source, e.target) for e in edges]}")
    
    # Special case: simple cycle
    if _is_simple_cycle(vertex_ids, edges):
        return _orthogonalize_cycle(vertex_ids, edges)
    
    # Special case: multiple cycles connected by bridges
    cycle_components = _find_cycle_components(vertex_ids, edges)
    if cycle_components and len(cycle_components) > 1:
        return _orthogonalize_multi_cycle(edges, cycle_components)
    
    ortho = OrthogonalRep()
    if not vertex_ids:
        return ortho
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    root = max(vertex_ids, key=lambda v: len(adjacency.get(v, set())))
    print(f"[ORTHO] Selected root: {root} (degree={len(adjacency.get(root, set()))})")
    
    # Build BFS tree
    tree = _build_bfs_tree(root, vertex_ids, adjacency)
    print(f"[ORTHO] BFS tree structure:")
    print(f"[ORTHO]   Levels: {tree.levels}")
    print(f"[ORTHO]   Parent: {tree.parent}")
    print(f"[ORTHO]   Children: {tree.children}")
    
    # Assign rows from BFS levels
    vertex_row: Dict[str, int] = {}
    for row_idx, level in enumerate(tree.levels):
        for v in level:
            vertex_row[v] = row_idx
    
    # Assign columns using subtree sizes
    subtree_size = _compute_subtree_sizes(root, tree.children)
    vertex_col = _assign_columns_by_subtree(root, tree.children, subtree_size)
    
    print(f"[ORTHO] After initial assignment:")
    print(f"[ORTHO]   Subtree sizes: {subtree_size}")
    for v in vertex_ids:
        print(f"[ORTHO]   {v}: row={vertex_row.get(v)}, col={vertex_col.get(v)}")
    
    # Align for orthogonality
    print(f"[ORTHO] Calling _align_for_orthogonality...")
    _align_for_orthogonality(vertex_ids, edges, vertex_row, vertex_col, tree.parent, tree.children)
    
    print(f"[ORTHO] After alignment:")
    for v in vertex_ids:
        print(f"[ORTHO]   {v}: row={vertex_row.get(v)}, col={vertex_col.get(v)}")
    
    # Store final positions
    for v in vertex_ids:
        ortho.vertex_pos[v] = (vertex_col.get(v, 0), vertex_row.get(v, 0))
    
    # Compute edge directions
    print(f"[ORTHO] Edge analysis:")
    ortho.edge_directions, diagonal_count = _compute_edge_directions_with_stats(edges, ortho.vertex_pos)
    
    print(f"[ORTHO] Summary: {diagonal_count}/{len(edges)} edges are diagonal")
    return ortho


def _align_for_orthogonality(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    vertex_row: Dict[str, int],
    vertex_col: Dict[str, int],
    parent: Dict[str, Optional[str]],
    children: Dict[str, List[str]],
) -> None:
    """
    Post-process to improve orthogonality by aligning vertices.
    
    Strategy:
    1. Identify vertices with many non-tree edges (K4-like structures)
    2. Try to position root vertex among its children for better alignment
    3. For each diagonal edge, try moving one endpoint to align
    """
    # Build adjacency for quick lookup
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Identify non-tree edges
    non_tree_edges: List[Tuple[str, str]] = []
    for e in edges:
        u, v = e.source, e.target
        if parent.get(u) != v and parent.get(v) != u:
            non_tree_edges.append((u, v))
    
    # Helper to check if position is occupied
    def is_occupied(row: int, col: int, exclude: str) -> bool:
        for vid in vertex_ids:
            if vid != exclude:
                if vertex_row.get(vid) == row and vertex_col.get(vid) == col:
                    return True
        return False
    
    # Strategy 1: For root with many children, try to position root among children
    for v in vertex_ids:
        v_children = children.get(v, [])
        if len(v_children) >= 3:
            # This vertex has many children - likely K4-like
            # Try to move it to a column that aligns with at least one child
            v_row = vertex_row.get(v, 0)
            child_cols = [vertex_col.get(c, 0) for c in v_children]
            
            # Try each child's column
            for target_col in child_cols:
                if not is_occupied(v_row, target_col, v):
                    vertex_col[v] = target_col
                    break
    
    # Strategy 2: For each diagonal edge, try to make it orthogonal
    for u, v in non_tree_edges:
        u_row = vertex_row.get(u, 0)
        v_row = vertex_row.get(v, 0)
        u_col = vertex_col.get(u, 0)
        v_col = vertex_col.get(v, 0)
        
        if u_row == v_row or u_col == v_col:
            # Already orthogonal
            continue
        
        # Try to align by moving the vertex with fewer neighbors
        u_degree = len(adjacency.get(u, set()))
        v_degree = len(adjacency.get(v, set()))
        
        if u_degree <= v_degree:
            # Try moving u to v's column (vertical) or v's row (horizontal)
            if not is_occupied(u_row, v_col, u):
                vertex_col[u] = v_col
            elif not is_occupied(v_row, u_col, u):
                vertex_row[u] = v_row
        else:
            # Try moving v to u's column (vertical) or u's row (horizontal)
            if not is_occupied(v_row, u_col, v):
                vertex_col[v] = u_col
            elif not is_occupied(u_row, v_col, v):
                vertex_row[v] = u_row
    
    # Strategy 3: Single child alignment (keep existing logic)
    for v in vertex_ids:
        v_children = children.get(v, [])
        if len(v_children) == 1:
            child = v_children[0]
            child_row = vertex_row.get(child, 0)
            parent_col = vertex_col.get(v, 0)
            
            if not is_occupied(child_row, parent_col, child):
                vertex_col[child] = parent_col


# =============================================================================
# TSM Phase 3: Compaction
# =============================================================================

def _compact_orthogonal(
    ortho_rep: OrthogonalRep,
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Assign final coordinates from orthogonal representation.
    
    Uses the grid positions from orthogonalization and scales to bounding box.
    """
    print(f"[COMPACT] === Starting _compact_orthogonal ===")
    print(f"[COMPACT] Grid positions: {ortho_rep.vertex_pos}")
    
    if not ortho_rep.vertex_pos:
        return _simple_grid_placement(vertex_ids, box)
    
    # Get grid bounds
    cols = [pos[0] for pos in ortho_rep.vertex_pos.values()]
    rows = [pos[1] for pos in ortho_rep.vertex_pos.values()]

    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)

    col_range = max(max_col - min_col, 1)
    row_range = max(max_row - min_row, 1)

    # Scale to bounding box with margins, using SQUARE cells
    margin = 0.1
    usable_width = box["width"] * (1 - 2 * margin)
    usable_height = box["height"] * (1 - 2 * margin)

    # Use the same spacing for both dimensions to preserve square shape
    col_spacing_candidate = usable_width / col_range
    row_spacing_candidate = usable_height / row_range
    cell_size = min(col_spacing_candidate, row_spacing_candidate)
    
    col_spacing = cell_size
    row_spacing = cell_size
    
    print(f"[COMPACT] Using square cells: {cell_size:.1f}px")
    
    positions: Dict[str, Tuple[float, float]] = {}
    for v in vertex_ids:
        if v not in ortho_rep.vertex_pos:
            continue
        pos = ortho_rep.vertex_pos[v]
        col = pos[0]
        row = pos[1]
        
        x = box["x"] + box["width"] * margin + (col - min_col) * col_spacing
        # Invert row so row 0 is at top (higher y)
        y = box["y"] + box["height"] * (1 - margin) - (row - min_row) * row_spacing
        
        positions[v] = (float(x), float(y))
    
    print(f"[COMPACT] Final positions:")
    for v in positions:
        print(f"[COMPACT]   {v}: {positions[v]}")
    
    return positions


# =============================================================================
# Planarization (for non-planar graphs)
# =============================================================================

def _find_all_edge_crossings(
    edges: List[Edge[str]],
    positions: Dict[str, Tuple[float, float]],
) -> List[Tuple[Edge[str], Edge[str], float, float]]:
    """Find all pairs of edges that geometrically cross."""
    crossings: List[Tuple[Edge[str], Edge[str], float, float]] = []
    edge_list = list(edges)
    
    for i, e1 in enumerate(edge_list):
        for e2 in edge_list[i + 1:]:
            # Skip edges sharing a vertex
            if e1.source in (e2.source, e2.target) or e1.target in (e2.source, e2.target):
                continue
            
            crossing = _edge_intersection(
                positions[e1.source], positions[e1.target],
                positions[e2.source], positions[e2.target]
            )
            
            if crossing is not None:
                crossings.append((e1, e2, crossing[0], crossing[1]))
    
    return crossings


def _create_dummy_vertices_for_crossings(
    crossings: List[Tuple[Edge[str], Edge[str], float, float]],
) -> Tuple[List[str], Set[str], Dict[Tuple[str, str], List[str]], Set[Tuple[str, str]]]:
    """Create dummy vertices at crossing points and track edge splits."""
    dummy_vertices: List[str] = []
    dummy_ids: Set[str] = set()
    edge_splits: Dict[Tuple[str, str], List[str]] = {}
    edges_to_remove: Set[Tuple[str, str]] = set()
    
    for idx in range(len(crossings)):
        crossing = crossings[idx]
        e1 = crossing[0]
        e2 = crossing[1]
        dummy_id = f"_dummy_{idx}"
        dummy_vertices.append(dummy_id)
        dummy_ids.add(dummy_id)
        
        key1 = (e1.source, e1.target)
        key2 = (e2.source, e2.target)
        
        edge_splits.setdefault(key1, []).append(dummy_id)
        edge_splits.setdefault(key2, []).append(dummy_id)
        
        edges_to_remove.add(key1)
        edges_to_remove.add(key2)
    
    return dummy_vertices, dummy_ids, edge_splits, edges_to_remove


def _create_split_edges(
    edge_splits: Dict[Tuple[str, str], List[str]],
) -> List[Edge[str]]:
    """Create new edges that connect through dummy vertices."""
    new_edges: List[Edge[str]] = []
    
    for edge_key in edge_splits:
        dummies = edge_splits[edge_key]
        src = edge_key[0]
        tgt = edge_key[1]
        if len(dummies) == 1:
            new_edges.append(Edge(src, dummies[0]))
            new_edges.append(Edge(dummies[0], tgt))
        else:
            new_edges.append(Edge(src, dummies[0]))
            for i in range(len(dummies) - 1):
                new_edges.append(Edge(dummies[i], dummies[i + 1]))
            new_edges.append(Edge(dummies[-1], tgt))
    
    return new_edges


def _planarize_graph(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Tuple[List[str], List[Edge[str]], Set[str]]:
    """
    Planarize a non-planar graph by inserting dummy vertices at crossings.
    
    Returns (new_vertices, new_edges, dummy_vertex_ids).
    """
    initial_positions = _force_directed_layout(vertex_ids, edges, box)
    crossings = _find_all_edge_crossings(edges, initial_positions)
    
    if not crossings:
        return vertex_ids, edges, set()
    
    dummy_vertices, dummy_ids, edge_splits, edges_to_remove = \
        _create_dummy_vertices_for_crossings(crossings)
    
    new_vertices = list(vertex_ids) + dummy_vertices
    
    # Add edges that don't need splitting
    new_edges: List[Edge[str]] = [
        e for e in edges if (e.source, e.target) not in edges_to_remove
    ]
    
    # Add split edges
    new_edges.extend(_create_split_edges(edge_splits))
    
    return new_vertices, new_edges, dummy_ids


def _edge_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection point of two line segments.
    
    Returns (x, y) if segments intersect, None otherwise.
    Uses parametric line intersection.
    """
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])
    x3 = float(p3[0])
    y3 = float(p3[1])
    x4 = float(p4[0])
    y4 = float(p4[1])
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Parallel or coincident
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection is within both segments (not at endpoints)
    eps = 0.01
    if eps < t < 1 - eps and eps < u < 1 - eps:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    
    return None


# =============================================================================
# Crossing Elimination
# =============================================================================

def _find_crossing_pairs(
    edges: List[Edge[str]],
    positions: Dict[str, Tuple[float, float]],
) -> List[Tuple[Edge[str], Edge[str]]]:
    """
    Find all pairs of edges that cross or overlap each other.

    Returns list of (edge1, edge2) tuples where the edges:
    - Cross at an interior point (non-adjacent edges)
    - Are collinear and overlap (non-adjacent edges)
    - Share a vertex but one vertex lies on the other edge (adjacent edges)
    """
    crossing_pairs: List[Tuple[Edge[str], Edge[str]]] = []
    edge_list = list(edges)

    for i in range(len(edge_list)):
        e1 = edge_list[i]
        p1 = positions.get(e1.source)
        p2 = positions.get(e1.target)
        if p1 is None or p2 is None:
            continue

        for j in range(i + 1, len(edge_list)):
            e2 = edge_list[j]
            p3 = positions.get(e2.source)
            p4 = positions.get(e2.target)
            if p3 is None or p4 is None:
                continue

            # Check if edges share a vertex
            shares_vertex = (
                e1.source in (e2.source, e2.target) or
                e1.target in (e2.source, e2.target)
            )

            if shares_vertex:
                # Check for adjacent edge overlap
                if _adjacent_edges_overlap(p1, p2, p3, p4):
                    crossing_pairs.append((e1, e2))
            else:
                # Check for crossing OR collinear overlap
                if GraphUtils.segments_cross(p1, p2, p3, p4) or _edges_collinear_overlap(p1, p2, p3, p4):
                    crossing_pairs.append((e1, e2))

    return crossing_pairs


def _to_grid_coords(
    positions: Dict[str, Tuple[float, float]],
    box: Dict[str, float],
    grid_size: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Convert float positions to integer grid coordinates.
    
    Maps positions within the bounding box to a grid of the specified size.
    """
    if not positions:
        return {}
    
    cell_width = box["width"] / grid_size
    cell_height = box["height"] / grid_size
    
    grid_pos: Dict[str, Tuple[int, int]] = {}
    for vid in positions:
        pos = positions[vid]
        x = float(pos[0])
        y = float(pos[1])
        col = int((x - box["x"]) / cell_width)
        row = int((y - box["y"]) / cell_height)
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))
        grid_pos[vid] = (col, row)
    
    return grid_pos


def _from_grid_coords(
    grid_pos: Dict[str, Tuple[int, int]],
    box: Dict[str, float],
    grid_size: int,
) -> Dict[str, Tuple[float, float]]:
    """
    Convert integer grid coordinates back to float positions.
    
    Places vertices at the center of their grid cells.
    """
    if not grid_pos:
        return {}
    
    cell_width = box["width"] / grid_size
    cell_height = box["height"] / grid_size
    
    positions: Dict[str, Tuple[float, float]] = {}
    for vid in grid_pos:
        gpos = grid_pos[vid]
        col = int(gpos[0])
        row = int(gpos[1])
        x = box["x"] + (col + 0.5) * cell_width
        y = box["y"] + (row + 0.5) * cell_height
        positions[vid] = (float(x), float(y))
    
    return positions


def _edges_collinear_overlap(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """
    Check if two edges are collinear and overlap (share more than just an endpoint).
    
    Two edges overlap if they are on the same line and their projections overlap.
    """
    # Check if all 4 points are collinear
    def cross_product(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    eps = 1e-9
    
    # Check collinearity: all 4 points must be on the same line
    if abs(cross_product(p1, p2, p3)) > eps or abs(cross_product(p1, p2, p4)) > eps:
        return False
    
    # They are collinear - check if projections overlap
    # Project onto the axis with larger span
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    if dx > dy:
        # Project onto x-axis
        min1, max1 = min(p1[0], p2[0]), max(p1[0], p2[0])
        min2, max2 = min(p3[0], p4[0]), max(p3[0], p4[0])
    else:
        # Project onto y-axis
        min1, max1 = min(p1[1], p2[1]), max(p1[1], p2[1])
        min2, max2 = min(p3[1], p4[1]), max(p3[1], p4[1])
    
    # Check if intervals overlap (more than just touching at endpoints)
    overlap = min(max1, max2) - max(min1, min2)
    return overlap > eps


def _point_on_segment(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    """
    Check if a point lies strictly inside a segment (not at endpoints).
    """
    px = float(point[0])
    py = float(point[1])
    x1 = float(seg_start[0])
    y1 = float(seg_start[1])
    x2 = float(seg_end[0])
    y2 = float(seg_end[1])
    
    # Check collinearity using cross product
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    
    # Check if point is between endpoints (strictly inside)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx > dy:
        min_v, max_v = min(x1, x2), max(x1, x2)
        return min_v + eps < px < max_v - eps
    else:
        min_v, max_v = min(y1, y2), max(y1, y2)
        return min_v + eps < py < max_v - eps


def _adjacent_edges_overlap(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """
    Check if two edges that share a vertex overlap.
    
    Returns True if the non-shared vertex of one edge lies on the other edge.
    """
    # Check if any non-shared point lies on the other segment
    if _point_on_segment(p1, p3, p4):
        return True
    if _point_on_segment(p2, p3, p4):
        return True
    if _point_on_segment(p3, p1, p2):
        return True
    if _point_on_segment(p4, p1, p2):
        return True
    return False


def _count_crossings_for_vertex(
    vertex: str,
    pos: Tuple[int, int],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    grid_size: int,
) -> int:
    """
    Count how many edge crossings or overlaps would occur if vertex is at the given position.
    
    Counts:
    - Crossing edges (non-adjacent)
    - Collinear overlapping edges (non-adjacent)
    - Adjacent edge overlaps (vertex on another edge)
    """
    # Temporarily set the vertex position
    old_pos = grid_pos.get(vertex)
    grid_pos[vertex] = pos
    
    # Convert to float positions for crossing detection
    float_pos: Dict[str, Tuple[float, float]] = {}
    for vid in grid_pos:
        gpos = grid_pos[vid]
        float_pos[vid] = (float(gpos[0]), float(gpos[1]))
    
    problems = 0
    incident_edges = [e for e in edges if e.source == vertex or e.target == vertex]
    
    for e1 in incident_edges:
        p1 = float_pos.get(e1.source)
        p2 = float_pos.get(e1.target)
        if p1 is None or p2 is None:
            continue
        
        for e2 in edges:
            if e1 == e2:
                continue
            
            p3 = float_pos.get(e2.source)
            p4 = float_pos.get(e2.target)
            if p3 is None or p4 is None:
                continue
            
            # Check if edges share a vertex
            shares_vertex = (
                e1.source in (e2.source, e2.target) or
                e1.target in (e2.source, e2.target)
            )
            
            if shares_vertex:
                # Check for adjacent edge overlap (vertex on segment)
                if _adjacent_edges_overlap(p1, p2, p3, p4):
                    problems += 1
            else:
                # Check for crossing OR collinear overlap
                if GraphUtils.segments_cross(p1, p2, p3, p4) or _edges_collinear_overlap(p1, p2, p3, p4):
                    problems += 1
    
    # Restore old position
    if old_pos is not None:
        grid_pos[vertex] = old_pos
    
    return problems


def _count_orthogonal_for_vertex(
    vertex: str,
    pos: Tuple[int, int],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
) -> int:
    """
    Count how many orthogonal edges the vertex would have at the given position.
    
    An edge is orthogonal if the vertices share the same row or column.
    """
    col = int(pos[0])
    row = int(pos[1])
    orthogonal = 0
    
    # Check edges incident to this vertex
    for e in edges:
        if e.source == vertex:
            other = e.target
        elif e.target == vertex:
            other = e.source
        else:
            continue
        
        other_pos = grid_pos.get(other)
        if other_pos is None:
            continue

        other_col = int(other_pos[0])
        other_row = int(other_pos[1])
        # Orthogonal if same row OR same column
        if col == other_col or row == other_row:
            orthogonal += 1
    
    return orthogonal


def _score_position(
    vertex: str,
    pos: Tuple[int, int],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    grid_size: int,
) -> Tuple[int, int]:
    """
    Score a position for a vertex.
    
    Returns (crossings, -orthogonal) so that sorting gives:
    - First priority: minimize crossings
    - Second priority: maximize orthogonal edges (negative for sorting)
    """
    crossings = _count_crossings_for_vertex(vertex, pos, edges, grid_pos, grid_size)
    
    # Temporarily set position to count orthogonal edges
    old_pos = grid_pos.get(vertex)
    grid_pos[vertex] = pos
    orthogonal = _count_orthogonal_for_vertex(vertex, pos, edges, grid_pos)
    if old_pos is not None:
        grid_pos[vertex] = old_pos
    
    return (crossings, -orthogonal)  # Negative so higher orthogonal is better


def _find_best_position(
    vertex: str,
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    grid_size: int,
    search_radius: int = 3,
    optimize_orthogonality: bool = True,
) -> Tuple[int, int]:
    """
    Find the best grid position for a vertex.
    
    Optimizes for:
    1. Minimum edge crossings (primary)
    2. Maximum orthogonal edges (secondary, if optimize_orthogonality=True)
    
    Searches positions within search_radius of the current position.
    """
    current_pos = grid_pos.get(vertex)
    if current_pos is None:
        return (0, 0)

    current_col = int(current_pos[0])
    current_row = int(current_pos[1])
    current_score = _score_position(vertex, current_pos, edges, grid_pos, grid_size)
    
    best_pos = current_pos
    best_score = current_score
    
    # Get occupied positions (excluding current vertex)
    occupied = {grid_pos[vid] for vid in grid_pos if vid != vertex}
    
    # Collect candidate positions
    candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    
    for dc in range(-search_radius, search_radius + 1):
        for dr in range(-search_radius, search_radius + 1):
            if dc == 0 and dr == 0:
                continue  # Skip current position
            
            new_col = current_col + dc
            new_row = current_row + dr
            
            # Check bounds
            if new_col < 0 or new_col >= grid_size or new_row < 0 or new_row >= grid_size:
                continue
            
            # Check for overlap with other vertices
            if (new_col, new_row) in occupied:
                continue
            
            new_pos = (new_col, new_row)
            score = _score_position(vertex, new_pos, edges, grid_pos, grid_size)
            candidates.append((new_pos, score))
    
    # Find best candidate
    for pos, score in candidates:
        if optimize_orthogonality:
            # Compare full score (crossings, -orthogonal)
            if score < best_score:
                best_score = score
                best_pos = pos
        else:
            # Only compare crossings
            if score[0] < best_score[0]:
                best_score = score
                best_pos = pos
    
    return best_pos


def _compute_optimization_grid_size(n_vertices: int, n_edges: int) -> int:
    """Compute grid size for optimization based on graph density."""
    edge_density = n_edges / max(n_vertices * (n_vertices - 1) / 2, 1)
    multiplier = 4 if edge_density > 0.5 else 3
    return max(int(math.ceil(math.sqrt(n_vertices))) * multiplier, 12)


def _count_total_orthogonal_grid(
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
) -> int:
    """Count total number of orthogonal edges in grid positions."""
    count = 0
    for e in edges:
        p1 = grid_pos.get(e.source)
        p2 = grid_pos.get(e.target)
        if p1 is None or p2 is None:
            continue
        # Orthogonal if same row or same column
        if p1[0] == p2[0] or p1[1] == p2[1]:
            count += 1
    return count


def _find_best_crossing_move(
    crossing_pair: Tuple[Edge[str], Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    box: Dict[str, float],
    best_crossing_count: int,
) -> Optional[Tuple[str, Tuple[int, int], int]]:
    """
    Find the best vertex move to eliminate a crossing/overlap.
    
    Priority order:
    1. Eliminate overlaps (always accept, overlaps are worst)
    2. Preserve orthogonality when eliminating crossings
    
    Returns (vertex, new_position, new_crossing_count) or None if no improvement.
    """
    e1 = crossing_pair[0]
    e2 = crossing_pair[1]
    involved = [e1.source, e1.target, e2.source, e2.target]
    
    # Track current state
    current_float_pos = _from_grid_coords(grid_pos, box, grid_size)
    current_overlaps = GraphUtils.count_edge_overlaps(edges, current_float_pos)
    current_global_ortho = _count_total_orthogonal_grid(edges, grid_pos)
    
    best_move = None
    # Score: (overlaps, crossings, -global_ortho) - prioritize overlap elimination
    best_move_score = (current_overlaps, best_crossing_count, -current_global_ortho)
    
    for v in involved:
        old_pos = grid_pos[v]
        new_pos = _find_best_position(v, edges, grid_pos, grid_size, optimize_orthogonality=True)
        
        if new_pos != old_pos:
            grid_pos[v] = new_pos
            new_float_pos = _from_grid_coords(grid_pos, box, grid_size)
            new_crossing_count = len(_find_crossing_pairs(edges, new_float_pos))
            new_overlaps = GraphUtils.count_edge_overlaps(edges, new_float_pos)
            new_global_ortho = _count_total_orthogonal_grid(edges, grid_pos)
            grid_pos[v] = old_pos
            
            move_score = (new_overlaps, new_crossing_count, -new_global_ortho)
            ortho_loss = current_global_ortho - new_global_ortho
            
            # Decide whether to accept the move
            accept = False
            
            if new_overlaps < current_overlaps:
                # Reduces overlaps - ALWAYS accept (overlaps are worst)
                accept = True
            elif new_crossing_count < best_crossing_count:
                # Reduces crossings - only if orthogonality preserved
                if ortho_loss <= 0:
                    accept = True
                elif new_crossing_count == 0 and ortho_loss <= 1:
                    accept = True
            
            if accept and (best_move is None or move_score < best_move_score):
                best_move = (v, new_pos, new_crossing_count)
                best_move_score = move_score
    
    return best_move


def _try_expanded_search(
    vertex_ids: List[str],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    box: Dict[str, float],
    best_crossing_count: int,
) -> Tuple[bool, int, Dict[str, Tuple[int, int]]]:
    """
    Try to reduce crossings/overlaps using expanded search radius.
    
    Priority: overlaps > crossings > orthogonality
    
    Returns (improved, new_crossing_count, best_grid_pos).
    """
    best_grid_pos = dict(grid_pos)
    current_float_pos = _from_grid_coords(grid_pos, box, grid_size)
    current_overlaps = GraphUtils.count_edge_overlaps(edges, current_float_pos)
    current_global_ortho = _count_total_orthogonal_grid(edges, grid_pos)
    
    for v in vertex_ids:
        old_pos = grid_pos[v]
        new_pos = _find_best_position(v, edges, grid_pos, grid_size, search_radius=grid_size // 2)
        
        if new_pos != old_pos:
            grid_pos[v] = new_pos
            new_float_pos = _from_grid_coords(grid_pos, box, grid_size)
            new_crossing_count = len(_find_crossing_pairs(edges, new_float_pos))
            new_overlaps = GraphUtils.count_edge_overlaps(edges, new_float_pos)
            new_global_ortho = _count_total_orthogonal_grid(edges, grid_pos)
            
            ortho_loss = current_global_ortho - new_global_ortho
            accept = False
            
            if new_overlaps < current_overlaps:
                # Reduces overlaps - ALWAYS accept
                accept = True
            elif new_crossing_count < best_crossing_count:
                # Reduces crossings - only if orthogonality preserved
                if ortho_loss <= 0 or (new_crossing_count == 0 and ortho_loss <= 1):
                    accept = True
            
            if accept:
                return True, new_crossing_count, dict(grid_pos)
            else:
                grid_pos[v] = old_pos
    
    return False, best_crossing_count, best_grid_pos


def _phase1_eliminate_crossings(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    grid_size: int,
    box: Dict[str, float],
    max_iterations: int,
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """
    Phase 1: Eliminate edge crossings by repositioning vertices.
    
    Returns (best_grid_pos, best_crossing_count).
    """
    best_grid_pos = dict(grid_pos)
    best_crossing_count = len(_find_crossing_pairs(edges, _from_grid_coords(grid_pos, box, grid_size)))
    
    print(f"[CROSSING-ELIM] Starting with {best_crossing_count} crossings, grid_size={grid_size}")
    
    for iteration in range(max_iterations):
        float_pos = _from_grid_coords(grid_pos, box, grid_size)
        crossing_pairs = _find_crossing_pairs(edges, float_pos)
        
        if not crossing_pairs:
            print(f"[CROSSING-ELIM] Eliminated all crossings after {iteration} iterations")
            break
        
        improved = False
        
        for crossing_pair in crossing_pairs:
            best_move = _find_best_crossing_move(
                crossing_pair, grid_pos, edges, grid_size, box, best_crossing_count
            )
            
            if best_move is not None:
                v = best_move[0]
                new_pos = best_move[1]
                new_crossing_count = best_move[2]
                grid_pos[v] = new_pos
                best_crossing_count = new_crossing_count
                best_grid_pos = dict(grid_pos)
                improved = True
                print(f"[CROSSING-ELIM] Iteration {iteration}: moved {v}, crossings now {new_crossing_count}")
                break
        
        if improved:
            continue
        
        # Try expanded search
        expanded_result = _try_expanded_search(
            vertex_ids, grid_pos, edges, grid_size, box, best_crossing_count
        )
        improved = expanded_result[0]
        best_crossing_count = expanded_result[1]
        best_grid_pos = expanded_result[2]
        grid_pos = dict(best_grid_pos)
        
        if not improved:
            print(f"[CROSSING-ELIM] No improvement possible, stopping at {best_crossing_count} crossings")
            break
    
    return best_grid_pos, best_crossing_count


def _try_ortho_position(
    v: str,
    new_pos: Tuple[int, int],
    old_pos: Tuple[int, int],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    best_ortho_count: int,
) -> Tuple[Tuple[int, int], int]:
    """
    Try a position for orthogonality optimization.
    
    Returns (best_pos, best_ortho_count).
    """
    crossings = _count_crossings_for_vertex(v, new_pos, edges, grid_pos, grid_size)
    if crossings > 0:
        return old_pos, best_ortho_count
    
    grid_pos[v] = new_pos
    ortho_count = _count_orthogonal_for_vertex(v, new_pos, edges, grid_pos)
    grid_pos[v] = old_pos
    
    if ortho_count > best_ortho_count:
        return new_pos, ortho_count
    return old_pos, best_ortho_count


def _search_nearby_ortho_positions(
    v: str,
    old_pos: Tuple[int, int],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    occupied: Set[Tuple[int, int]],
) -> Tuple[Tuple[int, int], int]:
    """Search nearby positions for orthogonality improvement."""
    old_col = int(old_pos[0])
    old_row = int(old_pos[1])
    best_ortho_pos = old_pos
    best_ortho_count = _count_orthogonal_for_vertex(v, old_pos, edges, grid_pos)
    
    search_radius = grid_size // 2
    for dc in range(-search_radius, search_radius + 1):
        for dr in range(-search_radius, search_radius + 1):
            if dc == 0 and dr == 0:
                continue
            
            new_col, new_row = old_col + dc, old_row + dr
            
            if not (0 <= new_col < grid_size and 0 <= new_row < grid_size):
                continue
            if (new_col, new_row) in occupied:
                continue
            
            new_pos = (new_col, new_row)
            ortho_result = _try_ortho_position(
                v, new_pos, best_ortho_pos, grid_pos, edges, grid_size, best_ortho_count
            )
            best_ortho_pos = ortho_result[0]
            best_ortho_count = ortho_result[1]
    
    return best_ortho_pos, best_ortho_count


def _search_neighbor_aligned_positions(
    v: str,
    adjacency: Dict[str, List[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    occupied: Set[Tuple[int, int]],
    old_pos: Tuple[int, int],
    best_ortho_pos: Tuple[int, int],
    best_ortho_count: int,
) -> Tuple[Tuple[int, int], int]:
    """Search positions aligned with neighbors for orthogonality improvement."""
    for neighbor in adjacency[v]:
        neighbor_pos = grid_pos[neighbor]
        n_col = int(neighbor_pos[0])
        n_row = int(neighbor_pos[1])
        
        # Try same row as neighbor
        for test_col in range(grid_size):
            if (test_col, n_row) in occupied or (test_col, n_row) == old_pos:
                continue
            new_pos = (test_col, n_row)
            ortho_result = _try_ortho_position(
                v, new_pos, best_ortho_pos, grid_pos, edges, grid_size, best_ortho_count
            )
            best_ortho_pos = ortho_result[0]
            best_ortho_count = ortho_result[1]
        
        # Try same column as neighbor
        for test_row in range(grid_size):
            if (n_col, test_row) in occupied or (n_col, test_row) == old_pos:
                continue
            new_pos = (n_col, test_row)
            ortho_result = _try_ortho_position(
                v, new_pos, best_ortho_pos, grid_pos, edges, grid_size, best_ortho_count
            )
            best_ortho_pos = ortho_result[0]
            best_ortho_count = ortho_result[1]
    
    return best_ortho_pos, best_ortho_count


def _phase2_optimize_orthogonality(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
    grid_size: int,
    box: Dict[str, float],
    max_iterations: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Phase 2: Optimize orthogonality while maintaining 0 crossings.
    
    Returns optimized grid positions.
    """
    float_pos = _from_grid_coords(grid_pos, box, grid_size)
    initial_ortho = GraphUtils.count_orthogonal_edges(edges, float_pos)
    print(f"[ORTHO-OPT] Starting orthogonality optimization: {initial_ortho[0]}/{initial_ortho[1]} edges orthogonal")
    
    # Build adjacency
    adjacency: Dict[str, List[str]] = {v: [] for v in vertex_ids}
    for e in edges:
        adjacency[e.source].append(e.target)
        adjacency[e.target].append(e.source)
    
    best_grid_pos = dict(grid_pos)
    
    for ortho_iter in range(max_iterations):
        improved = False
        
        for v in vertex_ids:
            old_pos = grid_pos[v]
            occupied = {grid_pos[vid] for vid in grid_pos if vid != v}
            
            # Strategy 1: Search nearby positions
            search_result = _search_nearby_ortho_positions(
                v, old_pos, grid_pos, edges, grid_size, occupied
            )
            best_ortho_pos = search_result[0]
            best_ortho_count = search_result[1]

            # Strategy 2: Try aligning with each neighbor
            align_result = _search_neighbor_aligned_positions(
                v, adjacency, grid_pos, edges, grid_size, occupied,
                old_pos, best_ortho_pos, best_ortho_count
            )
            best_ortho_pos = align_result[0]
            best_ortho_count = align_result[1]
            
            if best_ortho_pos != old_pos:
                grid_pos[v] = best_ortho_pos
                best_grid_pos = dict(grid_pos)
                improved = True
        
        if not improved:
            break
    
    final_float_pos = _from_grid_coords(best_grid_pos, box, grid_size)
    final_ortho = GraphUtils.count_orthogonal_edges(edges, final_float_pos)
    print(f"[ORTHO-OPT] Final: {final_ortho[0]}/{final_ortho[1]} edges orthogonal")
    
    return best_grid_pos


def _eliminate_crossings(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    positions: Dict[str, Tuple[float, float]],
    box: Dict[str, float],
    max_iterations: int = 50,
) -> Dict[str, Tuple[float, float]]:
    """
    Iteratively optimize graph layout through three phases.
    
    Phase 1: Eliminate edge crossings by repositioning vertices
    Phase 2: Optimize orthogonality while maintaining 0 crossings
    Phase 3: Equalize orthogonal edge lengths
    
    Args:
        vertex_ids: List of vertex identifiers
        edges: List of edges
        positions: Current vertex positions
        box: Bounding box for the layout
        max_iterations: Maximum iterations per phase
    
    Returns:
        Optimized positions
    """
    grid_size = _compute_optimization_grid_size(len(vertex_ids), len(edges))
    grid_pos = _to_grid_coords(positions, box, grid_size)
    
    # Phase 1: Eliminate crossings
    phase1_result = _phase1_eliminate_crossings(
        vertex_ids, edges, grid_pos, grid_size, box, max_iterations
    )
    best_grid_pos = phase1_result[0]
    best_crossing_count = phase1_result[1]

    # Phase 2: Optimize orthogonality (only if crossings eliminated)
    if best_crossing_count == 0:
        best_grid_pos = _phase2_optimize_orthogonality(
            vertex_ids, edges, dict(best_grid_pos), grid_size, box, max_iterations
        )
    
    # Phase 3: Convert to float positions
    positions = _from_grid_coords(best_grid_pos, box, grid_size)
    
    # Phase 4: Normalize orthogonal edge lengths
    positions = _normalize_orthogonal_edge_lengths(positions, edges)
    
    return positions


def _normalize_orthogonal_edge_lengths(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Edge[str]],
) -> Dict[str, Tuple[float, float]]:
    """
    Normalize orthogonal edges to have uniform length.
    
    Groups vertices by shared x or y coordinate and moves them together
    to create uniform spacing.
    """
    if not positions:
        return positions
    
    # Round coordinates to group vertices on same row/column
    ROUND_PRECISION = 0.1
    
    # Group vertices by y-coordinate (rows)
    y_to_vids: Dict[float, List[str]] = {}
    for vid in positions:
        pos = positions[vid]
        y = float(pos[1])
        rounded_y = round(y / ROUND_PRECISION) * ROUND_PRECISION
        y_to_vids.setdefault(rounded_y, []).append(vid)
    
    # Group vertices by x-coordinate (columns)
    x_to_vids: Dict[float, List[str]] = {}
    for vid in positions:
        pos = positions[vid]
        x = float(pos[0])
        rounded_x = round(x / ROUND_PRECISION) * ROUND_PRECISION
        x_to_vids.setdefault(rounded_x, []).append(vid)
    
    # Get sorted unique coordinates
    sorted_ys = sorted(y_to_vids.keys())
    sorted_xs = sorted(x_to_vids.keys())
    
    if len(sorted_ys) < 2 and len(sorted_xs) < 2:
        return positions
    
    # Calculate gaps
    y_gaps = [sorted_ys[i+1] - sorted_ys[i] for i in range(len(sorted_ys)-1)] if len(sorted_ys) > 1 else []
    x_gaps = [sorted_xs[i+1] - sorted_xs[i] for i in range(len(sorted_xs)-1)] if len(sorted_xs) > 1 else []
    
    all_gaps = [g for g in y_gaps + x_gaps if g > 1.0]  # Filter tiny gaps
    if not all_gaps:
        return positions
    
    # Use median as target spacing
    target_spacing = sorted(all_gaps)[len(all_gaps) // 2]
    
    print(f"[EDGE-NORM] Rows: {len(sorted_ys)}, Cols: {len(sorted_xs)}")
    print(f"[EDGE-NORM] Y gaps: {[round(g, 1) for g in y_gaps]}")
    print(f"[EDGE-NORM] X gaps: {[round(g, 1) for g in x_gaps]}")
    print(f"[EDGE-NORM] Target spacing: {target_spacing:.1f}")
    
    # Create mapping from old coordinates to new uniform coordinates
    new_y_coords: Dict[float, float] = {}
    if sorted_ys:
        base_y = sorted_ys[0]
        for i, old_y in enumerate(sorted_ys):
            new_y_coords[old_y] = base_y + i * target_spacing
    
    new_x_coords: Dict[float, float] = {}
    if sorted_xs:
        base_x = sorted_xs[0]
        for i, old_x in enumerate(sorted_xs):
            new_x_coords[old_x] = base_x + i * target_spacing
    
    # Apply new coordinates - vertices with same old coord get same new coord
    new_positions: Dict[str, Tuple[float, float]] = {}
    for vid in positions:
        pos = positions[vid]
        x = float(pos[0])
        y = float(pos[1])
        rounded_x = round(x / ROUND_PRECISION) * ROUND_PRECISION
        rounded_y = round(y / ROUND_PRECISION) * ROUND_PRECISION
        
        new_x = new_x_coords.get(rounded_x, x)
        new_y = new_y_coords.get(rounded_y, y)
        new_positions[vid] = (float(new_x), float(new_y))
    
    print(f"[EDGE-NORM] Applied uniform spacing")
    
    return new_positions


def _get_ortho_edge_lengths(
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
) -> List[int]:
    """Get lengths of orthogonal (horizontal/vertical) edges."""
    lengths = []
    for e in edges:
        p1 = grid_pos.get(e.source)
        p2 = grid_pos.get(e.target)
        if p1 is None or p2 is None:
            continue
        c1 = int(p1[0])
        r1 = int(p1[1])
        c2 = int(p2[0])
        r2 = int(p2[1])
        if c1 == c2:  # Vertical
            lengths.append(abs(r2 - r1))
        elif r1 == r2:  # Horizontal
            lengths.append(abs(c2 - c1))
    return lengths


def _length_variance(lengths: List[int]) -> float:
    """Calculate variance of edge lengths."""
    if len(lengths) < 2:
        return 0.0
    avg = sum(lengths) / len(lengths)
    return sum((length - avg) ** 2 for length in lengths) / len(lengths)


def _try_edge_eq_position(
    vid: str,
    test_pos: Tuple[int, int],
    old_pos: Tuple[int, int],
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    best_new_variance: float,
    require_no_crossings: bool = True,
) -> Tuple[Tuple[int, int], float]:
    """
    Try a position for edge length equalization.

    Returns (best_pos, best_variance) after testing.
    """
    grid_pos[vid] = test_pos
    
    if require_no_crossings:
        crossings = _count_crossings_for_vertex(vid, test_pos, edges, grid_pos, grid_size)
        if crossings > 0:
            grid_pos[vid] = old_pos
            return old_pos, best_new_variance

    lengths = _get_ortho_edge_lengths(grid_pos, edges)
    var = _length_variance(lengths)
    if var < best_new_variance:
        grid_pos[vid] = old_pos
        return test_pos, var

    grid_pos[vid] = old_pos
    return old_pos, best_new_variance


def _count_orthogonal_for_vid(
    vid: str,
    pos: Tuple[int, int],
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
) -> int:
    """Count orthogonal edges for a vertex at a given position."""
    col = int(pos[0])
    row = int(pos[1])
    count = 0
    for e in edges:
        neighbor = _get_edge_neighbor(e, vid)
        if neighbor is None:
            continue
        n_pos = grid_pos.get(neighbor)
        if n_pos is None:
            continue
        n_col = int(n_pos[0])
        n_row = int(n_pos[1])
        if col == n_col or row == n_row:
            count += 1
    return count


def _get_neighbor_positions(
    vid: str,
    edges: List[Edge[str]],
    grid_pos: Dict[str, Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Get positions of all neighbors of a vertex."""
    positions = []
    for e in edges:
        neighbor = _get_edge_neighbor(e, vid)
        if neighbor is not None:
            n_pos = grid_pos.get(neighbor)
            if n_pos is not None:
                positions.append(n_pos)
    return positions


def _equalize_edge_lengths(
    grid_pos: Dict[str, Tuple[int, int]],
    edges: List[Edge[str]],
    grid_size: int,
    max_iterations: int = 30,
) -> Dict[str, Tuple[int, int]]:
    """
    Try to equalize the length of orthogonal edges.

    Uses multiple strategies:
    1. Try positioning at target_length from each neighbor
    2. Try a range of distances around target_length
    3. Try positions that preserve orthogonality with all neighbors
    """
    initial_lengths = _get_ortho_edge_lengths(grid_pos, edges)
    if not initial_lengths:
        print(f"[EDGE-EQ] No orthogonal edges to equalize")
        return grid_pos

    target_length = max(1, round(sum(initial_lengths) / len(initial_lengths)))
    initial_variance = _length_variance(initial_lengths)
    print(f"[EDGE-EQ] Starting: {len(initial_lengths)} ortho edges, target={target_length}, variance={initial_variance:.2f}")
    print(f"[EDGE-EQ] Initial lengths: {initial_lengths}")

    if initial_variance < 0.01:
        print(f"[EDGE-EQ] Variance already low, skipping")
        return grid_pos

    best_pos = dict(grid_pos)
    best_variance = initial_variance
    positions_tried = 0
    improvements_found = 0

    for iteration in range(max_iterations):
        improved = False

        for vid in list(grid_pos.keys()):
            old_pos = grid_pos[vid]
            old_col = int(old_pos[0])
            old_row = int(old_pos[1])
            occupied = {grid_pos[v] for v in grid_pos if v != vid}
            best_new_pos = old_pos
            best_new_variance = best_variance

            # Get neighbor positions
            neighbor_positions = _get_neighbor_positions(vid, edges, grid_pos)
            if not neighbor_positions:
                continue

            # Count current orthogonal edges for this vertex
            current_ortho = _count_orthogonal_for_vid(vid, old_pos, edges, grid_pos)

            # Strategy 1: Try positions aligned with each neighbor at various distances
            for n_col, n_row in neighbor_positions:
                for dist in range(1, min(target_length + 4, grid_size)):
                    for delta in [-dist, dist]:
                        # Horizontal alignment (same row as neighbor)
                        new_col = n_col + delta
                        if 0 <= new_col < grid_size and (new_col, n_row) not in occupied:
                            test_pos = (new_col, n_row)
                            test_ortho = _count_orthogonal_for_vid(vid, test_pos, edges, grid_pos)
                            if test_ortho >= current_ortho:
                                positions_tried += 1
                                eq_result = _try_edge_eq_position(
                                    vid, test_pos, old_pos, grid_pos, edges, grid_size, best_new_variance
                                )
                                result_pos = eq_result[0]
                                result_var = eq_result[1]
                                if result_pos != old_pos and result_var < best_new_variance:
                                    best_new_pos = result_pos
                                    best_new_variance = result_var

                        # Vertical alignment (same column as neighbor)
                        new_row = n_row + delta
                        if 0 <= new_row < grid_size and (n_col, new_row) not in occupied:
                            test_pos = (n_col, new_row)
                            test_ortho = _count_orthogonal_for_vid(vid, test_pos, edges, grid_pos)
                            if test_ortho >= current_ortho:
                                positions_tried += 1
                                eq_result = _try_edge_eq_position(
                                    vid, test_pos, old_pos, grid_pos, edges, grid_size, best_new_variance
                                )
                                result_pos = eq_result[0]
                                result_var = eq_result[1]
                                if result_pos != old_pos and result_var < best_new_variance:
                                    best_new_pos = result_pos
                                    best_new_variance = result_var

            # Strategy 2: Try moving along current row/column
            for delta in range(-target_length - 3, target_length + 4):
                if delta == 0:
                    continue
                # Same row, different column
                new_col = old_col + delta
                if 0 <= new_col < grid_size and (new_col, old_row) not in occupied:
                    test_pos = (new_col, old_row)
                    test_ortho = _count_orthogonal_for_vid(vid, test_pos, edges, grid_pos)
                    if test_ortho >= current_ortho:
                        positions_tried += 1
                        eq_result = _try_edge_eq_position(
                            vid, test_pos, old_pos, grid_pos, edges, grid_size, best_new_variance
                        )
                        result_pos = eq_result[0]
                        result_var = eq_result[1]
                        if result_pos != old_pos and result_var < best_new_variance:
                            best_new_pos = result_pos
                            best_new_variance = result_var

                # Same column, different row
                new_row = old_row + delta
                if 0 <= new_row < grid_size and (old_col, new_row) not in occupied:
                    test_pos = (old_col, new_row)
                    test_ortho = _count_orthogonal_for_vid(vid, test_pos, edges, grid_pos)
                    if test_ortho >= current_ortho:
                        positions_tried += 1
                        eq_result = _try_edge_eq_position(
                            vid, test_pos, old_pos, grid_pos, edges, grid_size, best_new_variance
                        )
                        result_pos = eq_result[0]
                        result_var = eq_result[1]
                        if result_pos != old_pos and result_var < best_new_variance:
                            best_new_pos = result_pos
                            best_new_variance = result_var

            if best_new_pos != old_pos and best_new_variance < best_variance:
                print(f"[EDGE-EQ] Moving {vid} from {old_pos} to {best_new_pos}, variance: {best_variance:.2f} -> {best_new_variance:.2f}")
                grid_pos[vid] = best_new_pos
                best_pos = dict(grid_pos)
                best_variance = best_new_variance
                improved = True
                improvements_found += 1

        if not improved:
            break

    final_lengths = _get_ortho_edge_lengths(best_pos, edges)
    final_variance = _length_variance(final_lengths)
    print(f"[EDGE-EQ] Tried {positions_tried} positions, found {improvements_found} improvements")
    print(f"[EDGE-EQ] Final lengths: {final_lengths}")
    print(f"[EDGE-EQ] Final variance: {final_variance:.2f}")

    return best_pos


def _get_edge_neighbor(edge: Edge[str], vid: str) -> Optional[str]:
    """Get the neighbor vertex in an edge, or None if vid is not in the edge."""
    if edge.source == vid:
        return edge.target
    elif edge.target == vid:
        return edge.source
    return None


def _snap_to_grid(
    positions: Dict[str, Tuple[float, float]],
    box: Dict[str, float],
    grid_size: int,
) -> Dict[str, Tuple[int, int]]:
    """Snap float positions to integer grid coordinates."""
    cell_width = box["width"] / grid_size
    cell_height = box["height"] / grid_size
    
    grid_positions: Dict[str, Tuple[int, int]] = {}
    for vid in positions:
        pos = positions[vid]
        x = float(pos[0])
        y = float(pos[1])
        col = int((x - box["x"]) / cell_width)
        row = int((y - box["y"]) / cell_height)
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))
        grid_positions[vid] = (col, row)
    
    return grid_positions


def _align_edges_for_orthogonality(
    edges: List[Edge[str]],
    grid_positions: Dict[str, Tuple[int, int]],
    adjacency: Dict[str, Set[str]],
) -> None:
    """Align edges to be orthogonal by moving lower-degree vertices."""
    for edge in edges:
        u, v = edge.source, edge.target
        u_pos = grid_positions[u]
        v_pos = grid_positions[v]
        u_col = int(u_pos[0])
        u_row = int(u_pos[1])
        v_col = int(v_pos[0])
        v_row = int(v_pos[1])

        if u_col == v_col or u_row == v_row:
            continue
        
        col_diff = abs(u_col - v_col)
        row_diff = abs(u_row - v_row)
        
        # Move vertex with fewer neighbors
        move_u = len(adjacency.get(u, set())) <= len(adjacency.get(v, set()))
        
        if col_diff <= row_diff:
            # Align columns (make edge vertical)
            if move_u:
                grid_positions[u] = (v_col, u_row)
            else:
                grid_positions[v] = (u_col, v_row)
        else:
            # Align rows (make edge horizontal)
            if move_u:
                grid_positions[u] = (u_col, v_row)
            else:
                grid_positions[v] = (v_col, u_row)


def _find_nearest_unoccupied(
    col: int,
    row: int,
    occupied: Set[Tuple[int, int]],
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Find nearest unoccupied cell to (col, row)."""
    for delta in range(1, grid_size):
        for dc in range(-delta, delta + 1):
            for dr in range(-delta, delta + 1):
                nc, nr = col + dc, row + dr
                if 0 <= nc < grid_size and 0 <= nr < grid_size:
                    if (nc, nr) not in occupied:
                        return (nc, nr)
    return None


def _resolve_collisions(
    vertex_ids: List[str],
    grid_positions: Dict[str, Tuple[int, int]],
    grid_size: int,
) -> Dict[str, Tuple[int, int]]:
    """Resolve collisions by moving vertices to nearby unoccupied cells."""
    occupied: Set[Tuple[int, int]] = set()
    final_positions: Dict[str, Tuple[int, int]] = {}
    
    for vid in vertex_ids:
        gpos = grid_positions[vid]
        col = int(gpos[0])
        row = int(gpos[1])
        
        if (col, row) not in occupied:
            final_positions[vid] = (col, row)
            occupied.add((col, row))
        else:
            new_pos = _find_nearest_unoccupied(col, row, occupied, grid_size)
            if new_pos is not None:
                final_positions[vid] = new_pos
                occupied.add(new_pos)
            else:
                final_positions[vid] = (col, row)
    
    return final_positions


def _grid_to_box_coords(
    grid_positions: Dict[str, Tuple[int, int]],
    box: Dict[str, float],
    grid_size: int,
) -> Dict[str, Tuple[float, float]]:
    """Convert grid positions to box coordinates."""
    margin = 0.1
    usable_width = box["width"] * (1 - 2 * margin)
    usable_height = box["height"] * (1 - 2 * margin)
    
    result: Dict[str, Tuple[float, float]] = {}
    for vid in grid_positions:
        pos = grid_positions[vid]
        col = int(pos[0])
        row = int(pos[1])
        x = box["x"] + box["width"] * margin + (col + 0.5) * usable_width / grid_size
        y = box["y"] + box["height"] * margin + (row + 0.5) * usable_height / grid_size
        result[vid] = (float(x), float(y))
    
    return result


def _force_to_grid_fallback(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Fallback layout: force-directed snapped to grid with orthogonality alignment.
    """
    positions = _force_directed_layout(vertex_ids, edges, box)
    if not positions:
        return {}
    
    n = len(vertex_ids)
    grid_size = max(int(math.ceil(math.sqrt(n))) + 2, 4)
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    grid_positions = _snap_to_grid(positions, box, grid_size)
    _align_edges_for_orthogonality(edges, grid_positions, adjacency)
    final_positions = _resolve_collisions(vertex_ids, grid_positions, grid_size)
    
    return _grid_to_box_coords(final_positions, box, grid_size)


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

    center = _center(box)
    cx = center[0]
    cy = center[1]
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

    parent = rooted[0]
    children = rooted[1]
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
    iterations: int = 100,
) -> Dict[str, Tuple[float, float]]:
    """
    Spring-electrical model for general graph layout.
    
    Algorithm (Fruchterman-Reingold):
    1. Initialize positions (circular, scaled to 80% of box)
    2. Each iteration:
       a. Compute repulsion between all pairs (k²/d)
       b. Compute attraction along edges (d/k)
       c. Apply displacement capped by temperature
       d. Cool temperature (simulated annealing)
    3. Clamp positions to bounding box with margin
    """
    n = len(vertex_ids)
    if n == 0:
        return {}
    if n == 1:
        center = _center(box)
        cx = center[0]
        cy = center[1]
        return {vertex_ids[0]: (cx, cy)}

    # Use a smaller inner box to keep nodes away from edges
    margin = min(box["width"], box["height"]) * 0.1
    inner_box = {
        "x": box["x"] + margin,
        "y": box["y"] + margin,
        "width": box["width"] - 2 * margin,
        "height": box["height"] - 2 * margin,
    }
    
    # Initialize with circular layout in inner box
    positions = _circular_layout(vertex_ids, inner_box)
    
    # Optimal distance between nodes (Fruchterman-Reingold)
    area = inner_box["width"] * inner_box["height"]
    k = math.sqrt(area / n) * 0.75
    
    # Simulated annealing: start with larger movements, cool down
    temp = min(inner_box["width"], inner_box["height"]) / 5.0
    min_temp = temp * 0.01
    cooling_factor = (min_temp / temp) ** (1.0 / iterations)
    
    for _ in range(iterations):
        displacement = _compute_forces(vertex_ids, edges, positions, k)
        positions = _apply_displacement(vertex_ids, positions, displacement, temp, inner_box)
        temp *= cooling_factor
    
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
    Attraction: Connected pairs attract with force distance/k (spring-like)
    """
    displacement: Dict[str, Tuple[float, float]] = {vid: (0.0, 0.0) for vid in vertex_ids}
    
    # Repulsion between all pairs
    for i, v1 in enumerate(vertex_ids):
        pos1 = positions[v1]
        x1 = float(pos1[0])
        y1 = float(pos1[1])
        for v2 in vertex_ids[i + 1:]:
            pos2 = positions[v2]
            x2 = float(pos2[0])
            y2 = float(pos2[1])
            vec = _vector_between(x1, y1, x2, y2)
            dx = vec[0]
            dy = vec[1]
            dist = vec[2]
            
            # Repulsion: inversely proportional to distance
            repulsion = (k * k) / (dist * dist + 0.1)
            fx = (dx / dist) * repulsion
            fy = (dy / dist) * repulsion
            
            d1 = displacement[v1]
            d2 = displacement[v2]
            displacement[v1] = (d1[0] + fx, d1[1] + fy)
            displacement[v2] = (d2[0] - fx, d2[1] - fy)
    
    # Attraction along edges
    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        pos1 = positions[edge.source]
        pos2 = positions[edge.target]
        x1 = float(pos1[0])
        y1 = float(pos1[1])
        x2 = float(pos2[0])
        y2 = float(pos2[1])
        vec = _vector_between(x2, y2, x1, y1)  # Note: reversed for attraction
        dx = vec[0]
        dy = vec[1]
        dist = vec[2]
        
        if dist < 0.01:
            continue
        
        # Attraction: proportional to distance (spring)
        attraction = dist / k
        fx = (dx / dist) * attraction
        fy = (dy / dist) * attraction
        
        d1 = displacement[edge.source]
        d2 = displacement[edge.target]
        displacement[edge.source] = (d1[0] + fx, d1[1] + fy)
        displacement[edge.target] = (d2[0] - fx, d2[1] - fy)
    
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
        disp = displacement[vid]
        dx = float(disp[0])
        dy = float(disp[1])
        disp_len = math.sqrt(dx * dx + dy * dy)
        
        # Cap displacement by temperature
        if disp_len > 0:
            capped = min(disp_len, temp)
            dx = (dx / disp_len) * capped
            dy = (dy / disp_len) * capped
        
        # Apply and clamp to box
        pos = positions[vid]
        x = float(pos[0])
        y = float(pos[1])
        new_x = max(box["x"], min(box["x"] + box["width"], x + dx))
        new_y = max(box["y"], min(box["y"] + box["height"], y + dy))
        new_positions[vid] = (new_x, new_y)
    
    return new_positions
