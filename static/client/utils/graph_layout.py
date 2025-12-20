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
        cx, cy = _center(box)
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


def _grid_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Orthogonal grid layout using TSM (Topology-Shape-Metrics) approach.
    
    Produces layouts where:
    - All edges are axis-aligned (horizontal or vertical)
    - Vertices are placed on grid points
    - Edge crossings are minimized (eliminated for planar graphs)
    
    Algorithm phases:
    1. Handle special cases (no edges, trees)
    2. Check planarity; planarize if needed
    3. Find planar embedding
    4. Compute orthogonal representation
    5. Compact to grid coordinates
    """
    print(f"[GRID] === _grid_layout called ===")
    print(f"[GRID] Vertices: {vertex_ids}")
    print(f"[GRID] Edges: {[(e.source, e.target) for e in edges]}")
    
    # Fast path: no edges
    if not edges:
        print(f"[GRID] Path: no edges -> simple grid placement")
        result = _simple_grid_placement(vertex_ids, box)
        return {k: (float(v[0]), float(v[1])) for k, v in result.items()}
    
    # Fast path: tree structure
    if _is_tree_structure(vertex_ids, edges):
        print(f"[GRID] Path: tree structure -> orthogonal tree layout")
        result = _orthogonal_tree_layout(vertex_ids, edges, box)
        return {k: (float(v[0]), float(v[1])) for k, v in result.items()}
    
    # TSM pipeline for general graphs
    is_planar, embedding = _is_planar(vertex_ids, edges)
    print(f"[GRID] Planarity check: is_planar={is_planar}")
    
    if is_planar and embedding is not None:
        print(f"[GRID] Path: planar graph -> orthogonalize + compact")
        # Planar graph: use full orthogonal layout
        ortho_rep = _orthogonalize(vertex_ids, edges, embedding)
        
        # Check quality: if too many edges are diagonal, fall back
        diagonal_count = sum(
            1 for edge_key in ortho_rep.edge_directions
            if len(ortho_rep.edge_directions[edge_key]) > 1  # More than one direction = diagonal
        )
        diagonal_ratio = diagonal_count / max(len(edges), 1)
        print(f"[GRID] Diagonal ratio: {diagonal_count}/{len(edges)} = {diagonal_ratio:.2f}")
        
        if diagonal_ratio > 0.7:  # More than 70% diagonal -> fallback
            # Only fall back for very poor layouts; rely on elimination to fix moderate cases
            print(f"[GRID] Too many diagonals, using force-directed fallback")
            result = _force_to_grid_fallback(vertex_ids, edges, box)
        else:
            result = _compact_orthogonal(ortho_rep, vertex_ids, edges, box)
        
        # Check for crossings/overlaps and eliminate if needed
        result = {k: (float(v[0]), float(v[1])) for k, v in result.items()}
        crossings = GraphUtils.count_edge_crossings(edges, result)
        overlaps = GraphUtils.count_edge_overlaps(edges, result)
        
        if crossings > 0 or overlaps > 0:
            print(f"[GRID] Detected {crossings} crossings, {overlaps} overlaps, running elimination...")
            result = _eliminate_crossings(vertex_ids, edges, result, box)
            final_crossings = GraphUtils.count_edge_crossings(edges, result)
            final_overlaps = GraphUtils.count_edge_overlaps(edges, result)
            print(f"[GRID] After elimination: {final_crossings} crossings, {final_overlaps} overlaps")
        
        return result
    else:
        # Non-planar graph: planarize first
        print(f"[GRID] Path: non-planar graph -> planarize first")
        new_vertices, new_edges, dummy_ids = _planarize_graph(vertex_ids, edges, box)
        _, new_embedding = _is_planar(new_vertices, new_edges)
        
        if new_embedding is not None:
            ortho_rep = _orthogonalize(new_vertices, new_edges, new_embedding)
            positions = _compact_orthogonal(ortho_rep, new_vertices, new_edges, box)
            # Remove dummy vertices from output
            result = {k: v for k, v in positions.items() if k not in dummy_ids}
        else:
            # Fallback: snap force-directed to grid
            result = _force_to_grid_fallback(vertex_ids, edges, box)
        
        # Defensive: ensure all positions are 2-tuples
        result = {k: (float(v[0]), float(v[1])) for k, v in result.items()}
        
        # Check for overlaps and eliminate if needed (non-planar graphs can still have overlaps)
        overlaps = GraphUtils.count_edge_overlaps(edges, result)
        if overlaps > 0:
            print(f"[GRID] Non-planar: Detected {overlaps} overlaps, running elimination...")
            result = _eliminate_crossings(vertex_ids, edges, result, box)
            final_overlaps = GraphUtils.count_edge_overlaps(edges, result)
            print(f"[GRID] After elimination: {final_overlaps} overlaps")
        
        return result


def _orthogonal_tree_layout(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Orthogonal tree layout producing truly perpendicular edges.
    
    For trees, we can guarantee perfect orthogonality:
    - Each parent-child edge is VERTICAL (parent and child share same column)
    - Children of the same parent are connected via the parent (L-shaped routing)
    
    Algorithm:
    1. Build tree structure from edges
    2. Assign rows = depth in tree
    3. Assign columns using leaf-counting: each leaf gets unique column,
       internal nodes get column = center of their children's columns
    4. Result: all edges are purely vertical
    """
    # Infer root
    root_id = _infer_root(vertex_ids, edges)
    if root_id is None:
        return _simple_grid_placement(vertex_ids, box)
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    rooted = GraphUtils.root_tree(adjacency, root_id)
    if rooted is None:
        return _simple_grid_placement(vertex_ids, box)
    
    parent, children = rooted
    
    # Assign rows = depth in tree
    vertex_row: Dict[str, int] = {}
    
    def assign_depths(node: str, depth: int) -> None:
        vertex_row[node] = depth
        for child in children.get(node, []):
            assign_depths(child, depth + 1)
    
    assign_depths(root_id, 0)
    
    # Assign columns bottom-up:
    # Leaves get sequential columns (left to right)
    # Internal nodes get column = average of children's columns
    vertex_col: Dict[str, float] = {}
    leaf_counter = [0]
    
    def assign_columns(node: str) -> float:
        """Returns the column position for this node."""
        node_children = children.get(node, [])
        
        if not node_children:
            # Leaf: assign next available column
            col = float(leaf_counter[0])
            leaf_counter[0] += 1
            vertex_col[node] = col
            return col
        
        # Internal node: place at center of children
        child_cols = [assign_columns(child) for child in node_children]
        col = sum(child_cols) / len(child_cols)
        vertex_col[node] = col
        return col
    
    assign_columns(root_id)
    
    # For single-child chains, align child directly under parent
    for node in vertex_ids:
        node_children = children.get(node, [])
        if len(node_children) == 1:
            child = node_children[0]
            # Move child to parent's exact column
            vertex_col[child] = vertex_col[node]
    
    # Scale to bounding box
    cols = list(vertex_col.values())
    rows = list(vertex_row.values())
    
    if not cols or not rows:
        return _simple_grid_placement(vertex_ids, box)
    
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
        # Invert row so root is at top (higher y in math coords)
        y = box["y"] + box["height"] * (1 - margin) - (row - min_row) * usable_height / row_range
        
        positions[vid] = (float(x), float(y))
    
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
            is_comp_planar, _ = _is_planar(comp_vertices, comp_edges)
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


def _lr_planarity_check(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    adjacency: Dict[str, Set[str]],
) -> Tuple[bool, Optional[Dict[str, List[str]]]]:
    """
    LR-planarity algorithm implementation.
    
    Based on the left-right planarity test:
    1. DFS to build spanning tree
    2. Compute lowpoint values for back edges
    3. Order edges and check for conflicts
    4. If no conflicts, extract embedding
    """
    if not vertex_ids:
        return True, {}
    
    root = vertex_ids[0]
    
    # DFS data structures
    dfs_number: Dict[str, int] = {}
    dfs_parent: Dict[str, Optional[str]] = {root: None}
    lowpt: Dict[str, int] = {}
    lowpt2: Dict[str, int] = {}
    
    # Edge classification
    tree_edges: List[Tuple[str, str]] = []
    back_edges: List[Tuple[str, str]] = []
    
    # DFS to compute dfs_number and classify edges
    counter = [0]
    stack: List[Tuple[str, Optional[str]]] = [(root, None)]
    visited: Set[str] = set()
    
    while stack:
        node, parent = stack.pop()
        
        if node in visited:
            continue
        visited.add(node)
        
        dfs_number[node] = counter[0]
        counter[0] += 1
        dfs_parent[node] = parent
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                tree_edges.append((node, neighbor))
                stack.append((neighbor, node))
            elif neighbor != parent and dfs_number.get(neighbor, float('inf')) < dfs_number.get(node, 0):
                # Back edge to ancestor
                back_edges.append((node, neighbor))
    
    # Check if all vertices were visited (connected)
    if len(visited) != len(vertex_ids):
        # Not connected - shouldn't happen if we checked connectivity
        return False, None
    
    # Compute lowpoint values (bottom-up)
    # lowpt[v] = min DFS number reachable from v via tree edges and one back edge
    for v in vertex_ids:
        lowpt[v] = dfs_number.get(v, 0)
        lowpt2[v] = dfs_number.get(v, 0)
    
    # Process in reverse DFS order
    sorted_by_dfs = sorted(vertex_ids, key=lambda x: dfs_number.get(x, 0), reverse=True)
    
    for v in sorted_by_dfs:
        # Update from back edges
        for (src, tgt) in back_edges:
            if src == v:
                tgt_num = dfs_number.get(tgt, 0)
                if tgt_num < lowpt[v]:
                    lowpt2[v] = lowpt[v]
                    lowpt[v] = tgt_num
                elif tgt_num < lowpt2[v]:
                    lowpt2[v] = tgt_num
        
        # Update parent from child
        parent = dfs_parent.get(v)
        if parent is not None:
            if lowpt[v] < lowpt[parent]:
                lowpt2[parent] = min(lowpt2[parent], lowpt[parent])
                lowpt[parent] = lowpt[v]
            elif lowpt[v] < lowpt2[parent]:
                lowpt2[parent] = lowpt[v]
    
    # Simplified conflict check using constraint propagation
    # For each vertex, check if back edges can be consistently oriented
    # This is a simplified version - full LR requires more complex bookkeeping
    
    # For practical purposes with small-medium graphs:
    # Use the edge count bound plus check for K5/K3,3 minors
    n = len(vertex_ids)
    m = len(edges)
    
    # Additional check: bipartite graphs with |E| > 2|V| - 4 aren't planar
    is_bipartite, _ = GraphUtils.is_bipartite(adjacency)
    if is_bipartite and m > 2 * n - 4:
        return False, None
    
    # For this implementation, accept graphs passing edge bounds
    # Full K5/K3,3 detection would require more complex minor testing
    embedding = _build_embedding_from_dfs(vertex_ids, adjacency, dfs_number, dfs_parent)
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


def _find_cycle_components(
    vertex_ids: List[str],
    edges: List[Edge[str]],
) -> Optional[List[Tuple[List[str], List[Edge[str]]]]]:
    """
    Detect if graph is composed of simple cycles connected by bridges.
    
    Returns list of (vertices, edges) for each cycle component, or None if not applicable.
    """
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Find bridge edges (vertices with degree > 2)
    bridge_vertices = {v for v in vertex_ids if len(adjacency.get(v, set())) > 2}
    
    # If no bridge vertices, check if it's a single cycle
    if not bridge_vertices:
        if _is_simple_cycle(vertex_ids, edges):
            return [(vertex_ids, edges)]
        return None
    
    # Find bridge edges (edges connecting high-degree vertices)
    bridge_edges: Set[Tuple[str, str]] = set()
    for e in edges:
        if e.source in bridge_vertices and e.target in bridge_vertices:
            bridge_edges.add((e.source, e.target))
            bridge_edges.add((e.target, e.source))
    
    # Remove bridge edges and find connected components
    non_bridge_edges = [
        e for e in edges 
        if (e.source, e.target) not in bridge_edges
    ]
    
    # Build adjacency without bridges
    adj_no_bridges: Dict[str, Set[str]] = {v: set() for v in vertex_ids}
    for e in non_bridge_edges:
        adj_no_bridges[e.source].add(e.target)
        adj_no_bridges[e.target].add(e.source)
    
    # Find connected components
    visited: Set[str] = set()
    components: List[Tuple[List[str], List[Edge[str]]]] = []
    
    for start in vertex_ids:
        if start in visited:
            continue
        
        # BFS to find component
        component_verts: List[str] = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            component_verts.append(v)
            stack.extend(adj_no_bridges.get(v, set()))
        
        # Get edges for this component
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
    For n vertices, uses ceil(n/2) columns and 2 rows.
    """
    print("[ORTHO-CYCLE] Detected simple cycle, using rectangular layout")
    ortho = OrthogonalRep()
    n = len(vertex_ids)
    
    # Build adjacency and find cycle order
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Walk the cycle to get ordered vertices
    visited: Set[str] = set()
    ordered: List[str] = []
    current = vertex_ids[0]
    
    while len(ordered) < n:
        ordered.append(current)
        visited.add(current)
        neighbors = adjacency.get(current, set())
        for neighbor in neighbors:
            if neighbor not in visited:
                current = neighbor
                break
        else:
            break
    
    print(f"[ORTHO-CYCLE] Cycle order: {ordered}")
    
    # Place vertices in rectangular pattern
    # First half on row 0, second half on row 1 (reversed)
    half = (n + 1) // 2
    
    for i, v in enumerate(ordered):
        if i < half:
            # Top row: left to right
            ortho.vertex_pos[v] = (i, 0)
        else:
            # Bottom row: right to left (to close the cycle)
            ortho.vertex_pos[v] = (n - 1 - i, 1)
    
    print(f"[ORTHO-CYCLE] Positions: {ortho.vertex_pos}")
    
    # Compute edge directions
    for edge in edges:
        u, v = edge.source, edge.target
        u_col, u_row = ortho.vertex_pos[u]
        v_col, v_row = ortho.vertex_pos[v]
        
        directions: List[str] = []
        if u_col == v_col:
            directions.append('S' if v_row > u_row else 'N')
        elif u_row == v_row:
            directions.append('E' if v_col > u_col else 'W')
        else:
            # Diagonal - shouldn't happen for proper cycle layout
            if v_row > u_row:
                directions.append('S')
            else:
                directions.append('N')
            if v_col > u_col:
                directions.append('E')
            else:
                directions.append('W')
        
        ortho.edge_directions[(u, v)] = directions
    
    return ortho


def _orthogonalize_multi_cycle(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    components: List[Tuple[List[str], List[Edge[str]]]],
) -> OrthogonalRep:
    """
    Layout multiple cycle components connected by bridges.
    
    Stacks cycles vertically with bridges as vertical edges.
    """
    print(f"[ORTHO-MULTI] Laying out {len(components)} cycle components")
    ortho = OrthogonalRep()
    
    # Find all edges in components vs bridge edges
    all_component_edges: Set[Tuple[str, str]] = set()
    for _, comp_edges in components:
        for e in comp_edges:
            all_component_edges.add((e.source, e.target))
            all_component_edges.add((e.target, e.source))
    
    # Bridge edges connect different components
    bridge_edges: List[Tuple[str, str]] = []
    for e in edges:
        if (e.source, e.target) not in all_component_edges:
            bridge_edges.append((e.source, e.target))
    
    print(f"[ORTHO-MULTI] Bridge edges: {bridge_edges}")
    
    # Find which component each vertex belongs to
    vertex_to_comp: Dict[str, int] = {}
    for comp_idx in range(len(components)):
        comp_verts = components[comp_idx][0]
        for v in comp_verts:
            vertex_to_comp[v] = comp_idx
    
    # Order components by bridge connections (first to last)
    # Start with component 0, then follow bridges
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
        # Safety: add any remaining
        for i in range(len(components)):
            if i not in visited_comps:
                comp_order.append(i)
                visited_comps.add(i)
    
    print(f"[ORTHO-MULTI] Component order: {comp_order}")
    
    # Layout each cycle, stacking vertically
    row_offset = 0
    component_positions: Dict[str, Tuple[int, int]] = {}
    bridge_vertex_positions: Dict[str, Tuple[int, int]] = {}
    
    for order_idx, comp_idx in enumerate(comp_order):
        comp_verts, comp_edges = components[comp_idx]
        print(f"[ORTHO-MULTI] Component {comp_idx}: {comp_verts}")
        
        # Find bridge vertex in this component (if any) to align with previous
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
        
        # Get ordered cycle vertices, starting from bridge vertex if present
        comp_adj = GraphUtils.build_adjacency_map(comp_edges)
        visited: Set[str] = set()
        ordered: List[str] = []
        current = bridge_vertex if bridge_vertex else comp_verts[0]
        
        while len(ordered) < len(comp_verts):
            ordered.append(current)
            visited.add(current)
            for neighbor in comp_adj.get(current, set()):
                if neighbor not in visited:
                    current = neighbor
                    break
            else:
                break
        
        # Place as rectangle: top row, then bottom row
        n = len(ordered)
        half = (n + 1) // 2
        
        # Calculate column offset so bridge vertex aligns
        if bridge_vertex:
            bridge_idx = ordered.index(bridge_vertex)
            if bridge_idx < half:
                col_offset = target_col - bridge_idx
            else:
                col_offset = target_col - (n - 1 - bridge_idx)
        else:
            col_offset = 0
        
        for i, v in enumerate(ordered):
            if i < half:
                pos = (col_offset + i, row_offset)
            else:
                pos = (col_offset + (n - 1 - i), row_offset + 1)
            component_positions[v] = pos
            
            # Track bridge vertices for next component
            for src, tgt in bridge_edges:
                if v == src or v == tgt:
                    bridge_vertex_positions[v] = pos
        
        row_offset += 2  # Each cycle takes 2 rows, plus gap
    
    # Store positions
    for v, pos in component_positions.items():
        ortho.vertex_pos[v] = pos
    
    print(f"[ORTHO-MULTI] Final positions: {ortho.vertex_pos}")
    
    # Compute edge directions for all edges (including bridges)
    for edge in edges:
        u, v = edge.source, edge.target
        if u not in ortho.vertex_pos or v not in ortho.vertex_pos:
            continue
        
        u_col, u_row = ortho.vertex_pos[u]
        v_col, v_row = ortho.vertex_pos[v]
        
        directions: List[str] = []
        if u_col == v_col:
            directions.append('S' if v_row > u_row else 'N')
            print(f"[ORTHO-MULTI] Edge {u}->{v}: VERTICAL ✓")
        elif u_row == v_row:
            directions.append('E' if v_col > u_col else 'W')
            print(f"[ORTHO-MULTI] Edge {u}->{v}: HORIZONTAL ✓")
        else:
            print(f"[ORTHO-MULTI] Edge {u}->{v}: DIAGONAL ✗")
            if v_row > u_row:
                directions.append('S')
            else:
                directions.append('N')
            if v_col > u_col:
                directions.append('E')
            else:
                directions.append('W')
        
        ortho.edge_directions[(u, v)] = directions
    
    return ortho


def _orthogonalize(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    embedding: Dict[str, List[str]],
) -> OrthogonalRep:
    """
    Create orthogonal representation from planar embedding.
    
    For truly orthogonal edges, connected vertices must share either:
    - Same row (horizontal edge), OR
    - Same column (vertical edge)
    
    Algorithm:
    1. Check for simple cycle -> use rectangular layout
    2. Check for multiple cycles connected by bridges -> layout each cycle
    3. Build spanning tree via BFS
    4. Assign rows = BFS level (tree edges will be vertical)
    5. Children inherit parent's column, spread horizontally if multiple
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
        return _orthogonalize_multi_cycle(vertex_ids, edges, cycle_components)
    
    ortho = OrthogonalRep()
    
    if not vertex_ids:
        return ortho
    
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Find root (prefer vertex with high degree for better spread)
    root = max(vertex_ids, key=lambda v: len(adjacency.get(v, set())))
    print(f"[ORTHO] Selected root: {root} (degree={len(adjacency.get(root, set()))})")
    
    # Build spanning tree and levels via BFS
    parent: Dict[str, Optional[str]] = {root: None}
    children: Dict[str, List[str]] = {v: [] for v in vertex_ids}
    levels: List[List[str]] = []
    
    # Use list as queue (BFS)
    queue: List[Tuple[str, int]] = [(root, 0)]
    queue_idx = 0
    visited: Set[str] = {root}
    
    while queue_idx < len(queue):
        node, depth = queue[queue_idx]
        queue_idx += 1
        
        # Extend levels list if needed
        while len(levels) <= depth:
            levels.append([])
        levels[depth].append(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append((neighbor, depth + 1))
    
    # Add any disconnected vertices
    for v in vertex_ids:
        if v not in visited:
            levels[0].append(v)
            parent[v] = None
    
    print(f"[ORTHO] BFS tree structure:")
    print(f"[ORTHO]   Levels: {levels}")
    print(f"[ORTHO]   Parent: {parent}")
    print(f"[ORTHO]   Children: {children}")
    
    # Assign rows based on BFS level
    vertex_row: Dict[str, int] = {}
    for row_idx, level in enumerate(levels):
        for v in level:
            vertex_row[v] = row_idx
    
    # Assign columns using tree structure
    # Key insight: children should be spread around parent's column
    # This makes tree edges vertical or nearly vertical
    vertex_col: Dict[str, int] = {}
    
    # Compute subtree sizes for proportional column allocation
    subtree_size: Dict[str, int] = {}
    
    def compute_size(node: str) -> int:
        size = 1
        for child in children.get(node, []):
            size += compute_size(child)
        subtree_size[node] = size
        return size
    
    compute_size(root)
    
    # Assign columns top-down
    # Root gets column 0, children spread proportionally
    def assign_columns(node: str, col_start: int, col_end: int) -> None:
        # Place node at center of its allocated column range
        vertex_col[node] = (col_start + col_end) // 2
        
        node_children = children.get(node, [])
        if not node_children:
            return
        
        # Allocate column ranges to children proportionally
        total_size = sum(subtree_size.get(c, 1) for c in node_children)
        child_col = col_start
        
        for child in node_children:
            child_size = subtree_size.get(child, 1)
            # Proportional allocation
            child_range = max(1, (col_end - col_start) * child_size // max(total_size, 1))
            child_col_end = min(child_col + child_range, col_end)
            
            assign_columns(child, child_col, child_col_end)
            child_col = child_col_end
    
    total_width = subtree_size.get(root, len(vertex_ids))
    assign_columns(root, 0, total_width)
    
    print(f"[ORTHO] After initial assignment:")
    print(f"[ORTHO]   Subtree sizes: {subtree_size}")
    for v in vertex_ids:
        print(f"[ORTHO]   {v}: row={vertex_row.get(v)}, col={vertex_col.get(v)}")
    
    # Post-process: align vertices to improve orthogonality
    # For each edge, try to make it purely horizontal or vertical
    print(f"[ORTHO] Calling _align_for_orthogonality...")
    _align_for_orthogonality(vertex_ids, edges, vertex_row, vertex_col, parent, children)
    
    print(f"[ORTHO] After alignment:")
    for v in vertex_ids:
        print(f"[ORTHO]   {v}: row={vertex_row.get(v)}, col={vertex_col.get(v)}")
    
    # Store final positions
    for v in vertex_ids:
        ortho.vertex_pos[v] = (vertex_col.get(v, 0), vertex_row.get(v, 0))
    
    # Compute edge directions
    print(f"[ORTHO] Edge analysis:")
    diagonal_count = 0
    for edge in edges:
        u, v = edge.source, edge.target
        if u not in ortho.vertex_pos or v not in ortho.vertex_pos:
            continue
        
        u_col, u_row = ortho.vertex_pos[u]
        v_col, v_row = ortho.vertex_pos[v]
        
        directions: List[str] = []
        delta_col = v_col - u_col
        delta_row = v_row - u_row
        
        # For orthogonal edges, only one direction should be non-zero
        if delta_col == 0:
            # Purely vertical
            if delta_row > 0:
                directions.append('S')
            elif delta_row < 0:
                directions.append('N')
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): VERTICAL ✓")
        elif delta_row == 0:
            # Purely horizontal
            if delta_col > 0:
                directions.append('E')
            elif delta_col < 0:
                directions.append('W')
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): HORIZONTAL ✓")
        else:
            # Diagonal - needs bend (route as L-shape)
            diagonal_count += 1
            print(f"[ORTHO]   {u}({u_col},{u_row}) -> {v}({v_col},{v_row}): DIAGONAL ✗ (delta_col={delta_col}, delta_row={delta_row})")
            # Go vertical first, then horizontal
            if delta_row > 0:
                directions.append('S')
            elif delta_row < 0:
                directions.append('N')
            if delta_col > 0:
                directions.append('E')
            elif delta_col < 0:
                directions.append('W')
        
        ortho.edge_directions[(u, v)] = directions
    
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
    1. For tree edges (parent-child): align child to parent's column (vertical edge)
    2. For same-level edges: they're already on same row (horizontal edge)
    3. For cross edges: try to align if possible without disrupting tree structure
    """
    # Build edge set for quick lookup
    edge_set: Set[Tuple[str, str]] = set()
    for e in edges:
        edge_set.add((e.source, e.target))
        edge_set.add((e.target, e.source))
    
    # Identify non-tree edges (edges not in parent-child relationship)
    non_tree_edges: List[Tuple[str, str]] = []
    for e in edges:
        u, v = e.source, e.target
        if parent.get(u) != v and parent.get(v) != u:
            non_tree_edges.append((u, v))
    
    # For vertices with single child, align child directly under parent
    for v in vertex_ids:
        v_children = children.get(v, [])
        if len(v_children) == 1:
            child = v_children[0]
            # Check if moving child to parent's column is safe
            # (doesn't collide with siblings in same row)
            child_row = vertex_row.get(child, 0)
            parent_col = vertex_col.get(v, 0)
            
            # Check for collisions in child's row
            collision = False
            for other in vertex_ids:
                if other != child and vertex_row.get(other) == child_row:
                    if vertex_col.get(other) == parent_col:
                        collision = True
                        break
            
            if not collision:
                vertex_col[child] = parent_col
    
    # For non-tree edges between vertices on same row, they're already horizontal
    # For non-tree edges between vertices on adjacent rows, try to align columns
    for u, v in non_tree_edges:
        u_row = vertex_row.get(u, 0)
        v_row = vertex_row.get(v, 0)
        u_col = vertex_col.get(u, 0)
        v_col = vertex_col.get(v, 0)
        
        if u_row == v_row:
            # Same row - already horizontal, good
            continue
        
        if abs(u_row - v_row) == 1:
            # Adjacent rows - could make vertical
            # Try to align the vertex in the lower row to the upper's column
            if u_row < v_row:
                target_col = u_col
                moving_vertex = v
            else:
                target_col = v_col
                moving_vertex = u
            
            # Check for collision
            moving_row = vertex_row.get(moving_vertex, 0)
            collision = False
            for other in vertex_ids:
                if other != moving_vertex and vertex_row.get(other) == moving_row:
                    if vertex_col.get(other) == target_col:
                        collision = True
                        break
            
            if not collision:
                vertex_col[moving_vertex] = target_col


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

def _planarize_graph(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Tuple[List[str], List[Edge[str]], Set[str]]:
    """
    Planarize a non-planar graph by inserting dummy vertices at crossings.
    
    Algorithm:
    1. Get initial positions using force-directed layout
    2. Detect geometric edge crossings
    3. Insert dummy vertices at crossing points
    4. Replace crossing edges with segments through dummy vertices
    
    Returns:
        (new_vertices, new_edges, dummy_vertex_ids)
    """
    # Get initial layout to detect crossings
    initial_positions = _force_directed_layout(vertex_ids, edges, box)
    
    # Find all edge crossings
    crossings: List[Tuple[Edge[str], Edge[str], float, float]] = []
    edge_list = list(edges)
    
    for i, e1 in enumerate(edge_list):
        for e2 in edge_list[i + 1:]:
            # Skip edges sharing a vertex
            if e1.source in (e2.source, e2.target) or e1.target in (e2.source, e2.target):
                continue
            
            # Check for geometric intersection
            crossing = _edge_intersection(
                initial_positions[e1.source], initial_positions[e1.target],
                initial_positions[e2.source], initial_positions[e2.target]
            )
            
            if crossing is not None:
                crossings.append((e1, e2, crossing[0], crossing[1]))
    
    if not crossings:
        # No crossings found - graph might actually be planar with this embedding
        return vertex_ids, edges, set()
    
    # Create new vertices and edges
    new_vertices = list(vertex_ids)
    new_edges: List[Edge[str]] = []
    dummy_ids: Set[str] = set()
    edges_to_remove: Set[Tuple[str, str]] = set()
    
    # Track which edges get split by which dummies
    edge_splits: Dict[Tuple[str, str], List[str]] = {}
    
    for idx in range(len(crossings)):
        crossing = crossings[idx]
        e1 = crossing[0]
        e2 = crossing[1]
        # cx, cy = crossing[2], crossing[3]  # Not needed currently
        
        dummy_id = f"_dummy_{idx}"
        new_vertices.append(dummy_id)
        dummy_ids.add(dummy_id)
        
        # Mark edges for splitting
        key1 = (e1.source, e1.target)
        key2 = (e2.source, e2.target)
        
        edge_splits.setdefault(key1, []).append(dummy_id)
        edge_splits.setdefault(key2, []).append(dummy_id)
        
        edges_to_remove.add(key1)
        edges_to_remove.add(key2)
    
    # Add edges that don't need splitting
    for e in edges:
        key = (e.source, e.target)
        if key not in edges_to_remove:
            new_edges.append(e)
    
    # Add split edges (simplified: just connect endpoints to all dummies on that edge)
    for edge_key in edge_splits:
        src = edge_key[0]
        tgt = edge_key[1]
        dummies = edge_splits[edge_key]
        # Connect source to first dummy, dummies in chain, last dummy to target
        if len(dummies) == 1:
            new_edges.append(Edge(src, dummies[0]))
            new_edges.append(Edge(dummies[0], tgt))
        else:
            new_edges.append(Edge(src, dummies[0]))
            for i in range(len(dummies) - 1):
                new_edges.append(Edge(dummies[i], dummies[i + 1]))
            new_edges.append(Edge(dummies[-1], tgt))
    
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
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
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
    for vid, pos in positions.items():
        x, y = pos
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
    for vid, gpos in grid_pos.items():
        col, row = gpos
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
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
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
    for vid, gpos in grid_pos.items():
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
    col, row = pos
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
        
        other_col, other_row = other_pos
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
    
    current_col, current_row = current_pos
    current_score = _score_position(vertex, current_pos, edges, grid_pos, grid_size)
    
    best_pos = current_pos
    best_score = current_score
    
    # Get occupied positions (excluding current vertex)
    occupied = {gpos for vid, gpos in grid_pos.items() if vid != vertex}
    
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


def _eliminate_crossings(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    positions: Dict[str, Tuple[float, float]],
    box: Dict[str, float],
    max_iterations: int = 50,
) -> Dict[str, Tuple[float, float]]:
    """
    Iteratively eliminate edge crossings by repositioning vertices.
    
    Algorithm:
    1. Convert positions to grid coordinates
    2. Find all crossing edge pairs
    3. For each crossing, try moving involved vertices to eliminate it
    4. Repeat until no crossings or max iterations reached
    
    Args:
        vertex_ids: List of vertex identifiers
        edges: List of edges
        positions: Current vertex positions
        box: Bounding box for the layout
        max_iterations: Maximum number of improvement iterations
    
    Returns:
        Improved positions with fewer/no crossings
    """
    # Determine grid size based on vertex count
    # Use larger grid to allow more flexibility for orthogonal placement
    n = len(vertex_ids)
    grid_size = max(int(math.ceil(math.sqrt(n))) * 3, 10)
    
    # Convert to grid coordinates
    grid_pos = _to_grid_coords(positions, box, grid_size)
    
    # Track best result
    best_grid_pos = dict(grid_pos)
    best_crossing_count = len(_find_crossing_pairs(edges, _from_grid_coords(grid_pos, box, grid_size)))
    
    print(f"[CROSSING-ELIM] Starting with {best_crossing_count} crossings, grid_size={grid_size}")
    
    for iteration in range(max_iterations):
        # Get current positions as floats for crossing detection
        float_pos = _from_grid_coords(grid_pos, box, grid_size)
        crossing_pairs = _find_crossing_pairs(edges, float_pos)
        
        if not crossing_pairs:
            print(f"[CROSSING-ELIM] Eliminated all crossings after {iteration} iterations")
            break
        
        # Try to fix crossings
        improved = False
        
        for e1, e2 in crossing_pairs:
            # Get the 4 vertices involved in this crossing
            involved = [e1.source, e1.target, e2.source, e2.target]
            
            # Try moving each vertex to find a better position
            for v in involved:
                old_pos = grid_pos[v]
                new_pos = _find_best_position(v, edges, grid_pos, grid_size)
                
                if new_pos != old_pos:
                    grid_pos[v] = new_pos
                    
                    # Check if this improved the total crossing count
                    new_float_pos = _from_grid_coords(grid_pos, box, grid_size)
                    new_crossing_count = len(_find_crossing_pairs(edges, new_float_pos))
                    
                    if new_crossing_count < best_crossing_count:
                        best_crossing_count = new_crossing_count
                        best_grid_pos = dict(grid_pos)
                        improved = True
                        print(f"[CROSSING-ELIM] Iteration {iteration}: moved {v}, crossings now {new_crossing_count}")
                        break
                    else:
                        # Revert if not an improvement
                        grid_pos[v] = old_pos
            
            if improved:
                break  # Restart crossing detection with updated positions
        
        if not improved:
            # Try expanding search radius
            for v in vertex_ids:
                old_pos = grid_pos[v]
                new_pos = _find_best_position(v, edges, grid_pos, grid_size, search_radius=5)
                
                if new_pos != old_pos:
                    grid_pos[v] = new_pos
                    new_float_pos = _from_grid_coords(grid_pos, box, grid_size)
                    new_crossing_count = len(_find_crossing_pairs(edges, new_float_pos))
                    
                    if new_crossing_count < best_crossing_count:
                        best_crossing_count = new_crossing_count
                        best_grid_pos = dict(grid_pos)
                        improved = True
                        break
                    else:
                        grid_pos[v] = old_pos
            
            if not improved:
                print(f"[CROSSING-ELIM] No improvement possible, stopping at {best_crossing_count} crossings")
                break
    
    # Phase 2: Optimize orthogonality while maintaining 0 crossings
    if best_crossing_count == 0:
        grid_pos = dict(best_grid_pos)
        
        # Count initial orthogonal edges
        float_pos = _from_grid_coords(grid_pos, box, grid_size)
        initial_ortho = GraphUtils.count_orthogonal_edges(edges, float_pos)
        print(f"[ORTHO-OPT] Starting orthogonality optimization: {initial_ortho[0]}/{initial_ortho[1]} edges orthogonal")
        
        # Build adjacency for neighbor-aware optimization
        adjacency: Dict[str, List[str]] = {v: [] for v in vertex_ids}
        for e in edges:
            adjacency[e.source].append(e.target)
            adjacency[e.target].append(e.source)
        
        for ortho_iter in range(max_iterations):
            improved = False
            
            for v in vertex_ids:
                old_pos = grid_pos[v]
                old_col, old_row = old_pos
                
                # Find position that maximizes orthogonality without creating crossings
                best_ortho_pos = old_pos
                best_ortho_count = _count_orthogonal_for_vertex(v, old_pos, edges, grid_pos)
                
                # Get occupied positions
                occupied = {gpos for vid, gpos in grid_pos.items() if vid != v}
                
                # Strategy 1: Search nearby positions (wider radius)
                search_radius = grid_size // 2
                for dc in range(-search_radius, search_radius + 1):
                    for dr in range(-search_radius, search_radius + 1):
                        if dc == 0 and dr == 0:
                            continue
                        
                        new_col = old_col + dc
                        new_row = old_row + dr
                        
                        if new_col < 0 or new_col >= grid_size or new_row < 0 or new_row >= grid_size:
                            continue
                        
                        if (new_col, new_row) in occupied:
                            continue
                        
                        new_pos = (new_col, new_row)
                        
                        # Check that this doesn't create crossings
                        crossings = _count_crossings_for_vertex(v, new_pos, edges, grid_pos, grid_size)
                        if crossings > 0:
                            continue
                        
                        # Count orthogonal edges at new position
                        grid_pos[v] = new_pos
                        ortho_count = _count_orthogonal_for_vertex(v, new_pos, edges, grid_pos)
                        grid_pos[v] = old_pos
                        
                        if ortho_count > best_ortho_count:
                            best_ortho_count = ortho_count
                            best_ortho_pos = new_pos
                
                # Strategy 2: Try aligning with each neighbor
                for neighbor in adjacency[v]:
                    neighbor_pos = grid_pos[neighbor]
                    n_col, n_row = neighbor_pos
                    
                    # Try same row as neighbor
                    for test_col in range(grid_size):
                        if (test_col, n_row) in occupied or (test_col, n_row) == old_pos:
                            continue
                        new_pos = (test_col, n_row)
                        crossings = _count_crossings_for_vertex(v, new_pos, edges, grid_pos, grid_size)
                        if crossings > 0:
                            continue
                        grid_pos[v] = new_pos
                        ortho_count = _count_orthogonal_for_vertex(v, new_pos, edges, grid_pos)
                        grid_pos[v] = old_pos
                        if ortho_count > best_ortho_count:
                            best_ortho_count = ortho_count
                            best_ortho_pos = new_pos
                    
                    # Try same column as neighbor
                    for test_row in range(grid_size):
                        if (n_col, test_row) in occupied or (n_col, test_row) == old_pos:
                            continue
                        new_pos = (n_col, test_row)
                        crossings = _count_crossings_for_vertex(v, new_pos, edges, grid_pos, grid_size)
                        if crossings > 0:
                            continue
                        grid_pos[v] = new_pos
                        ortho_count = _count_orthogonal_for_vertex(v, new_pos, edges, grid_pos)
                        grid_pos[v] = old_pos
                        if ortho_count > best_ortho_count:
                            best_ortho_count = ortho_count
                            best_ortho_pos = new_pos
                
                if best_ortho_pos != old_pos:
                    grid_pos[v] = best_ortho_pos
                    best_grid_pos = dict(grid_pos)
                    improved = True
            
            if not improved:
                break
        
        final_float_pos = _from_grid_coords(best_grid_pos, box, grid_size)
        final_ortho = GraphUtils.count_orthogonal_edges(edges, final_float_pos)
        print(f"[ORTHO-OPT] Final: {final_ortho[0]}/{final_ortho[1]} edges orthogonal")
    
    # Return best result found
    return _from_grid_coords(best_grid_pos, box, grid_size)


def _force_to_grid_fallback(
    vertex_ids: List[str],
    edges: List[Edge[str]],
    box: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Fallback for non-planar graphs: force-directed layout snapped to grid.
    
    Enhanced to improve orthogonality:
    1. Compute force-directed layout
    2. Snap positions to grid
    3. Try to align connected vertices on same row OR column
    4. Resolve collisions
    """
    # Get force-directed positions
    positions = _force_directed_layout(vertex_ids, edges, box)
    
    if not positions:
        return {}
    
    # Determine grid granularity based on vertex count
    n = len(vertex_ids)
    grid_size = max(int(math.ceil(math.sqrt(n))) + 2, 4)
    
    cell_width = box["width"] / grid_size
    cell_height = box["height"] / grid_size
    
    # Build adjacency for alignment
    adjacency = GraphUtils.build_adjacency_map(edges)
    
    # Initial snap to grid
    grid_positions: Dict[str, Tuple[int, int]] = {}
    
    for vid in vertex_ids:
        x, y = positions[vid]
        col = int((x - box["x"]) / cell_width)
        row = int((y - box["y"]) / cell_height)
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))
        grid_positions[vid] = (col, row)
    
    # Try to align connected vertices (improve orthogonality)
    # Process edges and try to make them orthogonal
    for edge in edges:
        u, v = edge.source, edge.target
        u_col, u_row = grid_positions[u]
        v_col, v_row = grid_positions[v]
        
        # Skip if already orthogonal
        if u_col == v_col or u_row == v_row:
            continue
        
        # Decide whether to align horizontally or vertically
        # based on which requires less movement
        col_diff = abs(u_col - v_col)
        row_diff = abs(u_row - v_row)
        
        if col_diff <= row_diff:
            # Try to align columns (make edge vertical)
            target_col = (u_col + v_col) // 2
            # Check if we can move both to target column
            # For simplicity, move the vertex with fewer neighbors
            if len(adjacency.get(u, set())) <= len(adjacency.get(v, set())):
                grid_positions[u] = (v_col, u_row)
            else:
                grid_positions[v] = (u_col, v_row)
        else:
            # Try to align rows (make edge horizontal)
            if len(adjacency.get(u, set())) <= len(adjacency.get(v, set())):
                grid_positions[u] = (u_col, v_row)
            else:
                grid_positions[v] = (v_col, u_row)
    
    # Resolve collisions
    occupied: Set[Tuple[int, int]] = set()
    final_positions: Dict[str, Tuple[int, int]] = {}
    
    for vid in vertex_ids:
        col, row = grid_positions[vid]
        
        if (col, row) not in occupied:
            final_positions[vid] = (col, row)
            occupied.add((col, row))
        else:
            # Find nearest unoccupied cell
            found = False
            for delta in range(1, grid_size):
                for dc in range(-delta, delta + 1):
                    for dr in range(-delta, delta + 1):
                        nc, nr = col + dc, row + dr
                        if 0 <= nc < grid_size and 0 <= nr < grid_size:
                            if (nc, nr) not in occupied:
                                final_positions[vid] = (nc, nr)
                                occupied.add((nc, nr))
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                final_positions[vid] = (col, row)
    
    # Convert grid positions to box coordinates
    margin = 0.1
    usable_width = box["width"] * (1 - 2 * margin)
    usable_height = box["height"] * (1 - 2 * margin)
    
    result: Dict[str, Tuple[float, float]] = {}
    for vid in final_positions:
        pos = final_positions[vid]
        col = pos[0]
        row = pos[1]
        x = box["x"] + box["width"] * margin + (col + 0.5) * usable_width / grid_size
        y = box["y"] + box["height"] * margin + (row + 0.5) * usable_height / grid_size
        result[vid] = (float(x), float(y))
    
    return result


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
        cx, cy = _center(box)
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
        x1, y1 = positions[v1]
        for v2 in vertex_ids[i + 1:]:
            x2, y2 = positions[v2]
            dx, dy, dist = _vector_between(x1, y1, x2, y2)
            
            # Repulsion: inversely proportional to distance
            repulsion = (k * k) / (dist * dist + 0.1)
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
        
        # Attraction: proportional to distance (spring)
        attraction = dist / k
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
