"""Graph analysis operations dispatcher.

This module provides the GraphAnalyzer class which dispatches graph analysis
operations to the appropriate GraphUtils algorithms based on operation name.

Key Features:
    - Shortest path computation (BFS unweighted, Dijkstra weighted)
    - Minimum spanning tree extraction
    - Topological sorting for DAGs
    - Bridge and articulation point detection
    - Euler path/circuit status
    - Bipartite graph detection and coloring
    - Tree operations (levels, diameter, LCA, reroot)
    - Convex hull computation from vertex positions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry.graph_state import GraphEdgeDescriptor, GraphState, TreeState
from utils.geometry_utils import GeometryUtils
from utils.graph_utils import Edge, GraphUtils


class GraphAnalyzer:
    @staticmethod
    def _build_edges(state: GraphState) -> List[Edge[str]]:
        return [Edge(edge.source, edge.target) for edge in state.edges]

    @staticmethod
    def _build_adjacency(state: GraphState, directed: bool) -> Dict[str, set[str]]:
        edges = GraphAnalyzer._build_edges(state)
        if directed:
            raw = GraphUtils.build_directed_adjacency_map(edges)
            return {k: set(v) for k, v in raw.items()}
        raw = GraphUtils.build_adjacency_map(edges)
        return {k: set(v) for k, v in raw.items()}

    @staticmethod
    def _weight_lookup(state: GraphState) -> Dict[Tuple[str, str], float]:
        weights: Dict[Tuple[str, str], float] = {}
        for edge in state.edges:
            if edge.weight is not None:
                weights[(edge.source, edge.target)] = float(edge.weight)
        return weights

    @staticmethod
    def _edge_name_for_endpoints(
        state: GraphState,
        u: str,
        v: str,
        directed: bool,
    ) -> Optional[str]:
        for edge in state.edges:
            if directed:
                if edge.source == u and edge.target == v:
                    return edge.name or edge.id
            else:
                if (edge.source == u and edge.target == v) or (edge.source == v and edge.target == u):
                    return edge.name or edge.id
        return None

    @staticmethod
    def _resolve_root(
        state: GraphState, adjacency: Dict[str, set[str]], params: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Resolve root from params or state, handling old internal ID format."""
        params = params or {}
        root = params.get("root") or getattr(state, "root", None)
        if root is None:
            # No root specified, use first vertex from state if available
            if state.vertices:
                return state.vertices[0].id
            if adjacency:
                return next(iter(adjacency.keys()))
            return None
        # If root is already in adjacency, use it directly
        if root in adjacency:
            return root
        # Handle old format: internal IDs like "v0", "v1"
        if isinstance(root, str) and root.startswith("v") and root[1:].isdigit():
            idx = int(root[1:])
            # First try state.vertices (more reliable ordering)
            if state.vertices and 0 <= idx < len(state.vertices):
                candidate = state.vertices[idx].id
                if candidate in adjacency:
                    return candidate
            # Fall back to sorted adjacency keys
            vertex_ids = sorted(adjacency.keys())
            if 0 <= idx < len(vertex_ids):
                return vertex_ids[idx]
        # Try matching root against vertex names in state
        for v in state.vertices:
            if v.name == root and v.id in adjacency:
                return v.id
        # Fallback: return first vertex from adjacency
        if adjacency:
            return next(iter(adjacency.keys()))
        return None

    @staticmethod
    def analyze(state: GraphState, operation: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        directed = bool(state.directed)
        adjacency = GraphAnalyzer._build_adjacency(state, directed)
        edges = GraphAnalyzer._build_edges(state)
        weights = GraphAnalyzer._weight_lookup(state)
        result: Dict[str, Any] = {"operation": operation}

        if operation == "shortest_path":
            start = params.get("start")
            goal = params.get("goal")
            if start is None or goal is None:
                return {"error": "start and goal are required for shortest_path"}
            if weights:
                path_data = GraphUtils.shortest_path_dijkstra(
                    edges,
                    start,
                    goal,
                    weight_lookup=weights,
                    directed=directed,
                )
                if path_data is None:
                    return {"path": None}
                path, cost = path_data
                result["cost"] = cost
            else:
                path = GraphUtils.shortest_path_unweighted(edges, start, goal, directed=directed)
                if path is None:
                    return {"path": None}
            result["path"] = path
            result["highlight_vectors"] = GraphAnalyzer._edge_names_from_path(state, path, directed)
            return result

        if operation == "mst":
            mst_edges = GraphUtils.minimum_spanning_tree(edges, weight_lookup=weights)
            edge_names = GraphAnalyzer._edge_names_from_edges(state, mst_edges, directed=False)
            result["edges"] = [e.as_tuple() for e in mst_edges]
            result["highlight_vectors"] = edge_names
            return result

        if operation == "topological_sort":
            order = GraphUtils.topological_sort(adjacency)
            return {"order": order}

        if operation == "bridges":
            bridges = GraphUtils.find_bridges(adjacency)
            names = []
            for u, v in bridges:
                name = GraphAnalyzer._edge_name_for_endpoints(state, u, v, directed=False)
                if name:
                    names.append(name)
            return {"bridges": bridges, "highlight_vectors": names}

        if operation == "articulation_points":
            points = list(GraphUtils.find_articulation_points(adjacency))
            return {"articulation_points": points}

        if operation == "euler_status":
            status = GraphUtils.euler_status(adjacency)
            return {"status": status}

        if operation == "bipartite":
            is_bipartite, color_map = GraphUtils.is_bipartite(adjacency)
            return {"is_bipartite": is_bipartite, "coloring": color_map}

        if operation == "bfs":
            start = params.get("start")
            order = GraphUtils.bfs_order(start, adjacency) if start else None
            return {"order": order}

        if operation == "dfs":
            start = params.get("start")
            order = GraphUtils.dfs_preorder(start, adjacency) if start else None
            return {"order": order}

        if operation == "levels":
            root = GraphAnalyzer._resolve_root(state, adjacency, params)
            levels = GraphUtils.tree_levels(root, adjacency) if root else None
            return {"levels": levels}

        if operation == "diameter":
            path = GraphUtils.tree_diameter(adjacency)
            result["path"] = path
            if path:
                result["highlight_vectors"] = GraphAnalyzer._edge_names_from_path(state, path, directed=False)
            return result

        if operation == "lca":
            root = GraphAnalyzer._resolve_root(state, adjacency, params)
            a = params.get("a")
            b = params.get("b")
            if root is None or a is None or b is None:
                return {"error": f"root, a, and b are required for lca (root={root!r}, a={a!r}, b={b!r})"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                adj_keys = list(adjacency.keys())[:5]
                return {
                    "error": f"invalid tree structure: root={root!r}, adjacency_sample={adj_keys}, edges={len(state.edges)}"
                }
            parent, children = rooted
            depths = GraphUtils.node_depths(root, adjacency) or {}
            lca_node = GraphUtils.lowest_common_ancestor(parent, depths, a, b)
            return {"lca": lca_node}

        if operation == "balance_children":
            root = GraphAnalyzer._resolve_root(state, adjacency, params)
            if root is None:
                return {"error": f"root is required for balance_children (adjacency has {len(adjacency)} vertices)"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                adj_keys = list(adjacency.keys())[:5]
                return {
                    "error": f"invalid tree structure: root={root!r}, adjacency_sample={adj_keys}, edges={len(state.edges)}"
                }
            _, children = rooted
            balanced = GraphUtils.balance_children(root, children)
            return {"children": balanced}

        if operation == "invert_children":
            root = GraphAnalyzer._resolve_root(state, adjacency, params)
            if root is None:
                return {"error": f"root is required for invert_children (adjacency has {len(adjacency)} vertices)"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                adj_keys = list(adjacency.keys())[:5]
                return {
                    "error": f"invalid tree structure: root={root!r}, adjacency_sample={adj_keys}, edges={len(state.edges)}"
                }
            _, children = rooted
            inverted = GraphUtils.invert_children(children)
            return {"children": inverted}

        if operation == "reroot":
            root = GraphAnalyzer._resolve_root(state, adjacency, params)
            new_root = params.get("new_root")
            if root is None or new_root is None:
                return {"error": f"root and new_root are required for reroot (root={root!r}, new_root={new_root!r})"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                adj_keys = list(adjacency.keys())[:5]
                return {
                    "error": f"invalid tree structure: root={root!r}, adjacency_sample={adj_keys}, edges={len(state.edges)}"
                }
            parent, children = rooted
            rerooted = GraphUtils.reroot_tree(parent, children, new_root)
            if rerooted is None:
                return {"error": "reroot failed"}
            return {"parent": rerooted[0], "children": rerooted[1]}

        if operation == "convex_hull":
            # Extract vertex positions from state
            positions: List[Tuple[float, float]] = []
            vertex_at_pos: Dict[Tuple[float, float], str] = {}
            for v in state.vertices:
                if v.x is not None and v.y is not None:
                    pos = (float(v.x), float(v.y))
                    positions.append(pos)
                    vertex_at_pos[pos] = v.id
            if len(positions) < 3:
                # Convert tuples to lists for JSON serialization
                hull_as_lists = [list(p) for p in positions]
                return {"hull": hull_as_lists, "hull_vertices": [vertex_at_pos.get(p, "") for p in positions]}
            hull = GeometryUtils.convex_hull(positions)
            hull_vertices = [vertex_at_pos.get(p, "") for p in hull]
            # Convert tuples to lists for JSON serialization
            hull_as_lists = [list(p) for p in hull]
            return {"hull": hull_as_lists, "hull_vertices": hull_vertices}

        if operation == "point_in_hull":
            x = params.get("x")
            y = params.get("y")
            if x is None or y is None:
                return {"error": "x and y coordinates are required for point_in_hull"}
            positions = [(float(v.x), float(v.y)) for v in state.vertices if v.x is not None and v.y is not None]
            hull = GeometryUtils.convex_hull(positions)
            inside = GeometryUtils.point_in_convex_hull((float(x), float(y)), hull)
            # Convert tuples to lists for JSON serialization
            hull_as_lists = [list(p) for p in hull]
            return {"inside": inside, "hull": hull_as_lists}

        return {"error": f"Unsupported operation '{operation}'"}

    @staticmethod
    def _edge_names_from_edges(
        state: GraphState,
        edges: Sequence[Edge[str]],
        directed: bool,
    ) -> List[str]:
        names: List[str] = []
        for edge in edges:
            name = GraphAnalyzer._edge_name_for_endpoints(state, edge.source, edge.target, directed)
            if name:
                names.append(name)
        return names

    @staticmethod
    def _edge_names_from_path(state: GraphState, path: List[str], directed: bool) -> List[str]:
        if not path:
            return []
        names: List[str] = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            name = GraphAnalyzer._edge_name_for_endpoints(state, u, v, directed)
            if name:
                names.append(name)
        return names
