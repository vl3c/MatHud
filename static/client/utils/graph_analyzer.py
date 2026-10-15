"""Graph analysis operations dispatcher.

This module provides the GraphAnalyzer class which dispatches graph analysis
operations to the appropriate GraphUtils algorithms based on operation name.

Key Features:
    - Shortest path computation (BFS unweighted, Dijkstra / Bellman-Ford weighted)
    - Minimum spanning tree extraction
    - Topological sorting for DAGs
    - Bridge and articulation point detection
    - Euler path/circuit status
    - Bipartite graph detection and coloring
    - Diameter (exact, by BFS from every vertex)
    - Tree operations (levels, LCA, child ordering, reroot) on graphs that are trees
    - Convex hull computation from vertex positions

Vertex names given in params are checked once up front: an unknown name returns an
error listing the graph's vertices instead of an empty result.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from geometry.graph_state import GraphState
from utils.geometry_utils import GeometryUtils
from utils.graph_utils import Edge, GraphUtils

Result = Dict[str, Any]
# A rooted tree: (root, parent map, children map, depths), or an error result.
RootedTree = Tuple[str, Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, int]]

# Vertex params each operation reads; they must name vertices of the graph.
_VERTEX_PARAMS: Dict[str, Tuple[str, ...]] = {
    "shortest_path": ("start", "goal"),
    "bfs": ("start",),
    "dfs": ("start",),
    "lca": ("a", "b"),
    "reroot": ("new_root",),
}
_MAX_LISTED_VERTICES = 40


class GraphAnalyzer:
    @staticmethod
    def analyze(state: GraphState, operation: str, params: Optional[Dict[str, Any]] = None) -> Result:
        params = params or {}
        handler = _HANDLERS.get(operation)
        if handler is None:
            return {"error": f"Unsupported operation '{operation}'"}
        missing = GraphAnalyzer._missing_vertex_params(state, operation, params)
        if missing:
            return {"error": missing}
        return handler(state, params)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _string_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _vertex_ids(state: GraphState) -> List[str]:
        """Vertex ids in state order, then any edge endpoint the vertex list misses."""
        ids: List[str] = []
        seen = set()
        candidates = [v.id for v in state.vertices]
        for edge in state.edges:
            candidates.extend([edge.source, edge.target])
        for vertex_id in candidates:
            if vertex_id not in seen:
                seen.add(vertex_id)
                ids.append(vertex_id)
        return ids

    @staticmethod
    def _vertex_list_text(state: GraphState) -> str:
        ids = sorted(GraphAnalyzer._vertex_ids(state))
        shown = ", ".join(ids[:_MAX_LISTED_VERTICES])
        if len(ids) > _MAX_LISTED_VERTICES:
            shown += f", ... ({len(ids)} in all)"
        return f"vertices of {state.name}: {shown}" if ids else f"{state.name} has no vertices"

    @staticmethod
    def _missing_vertex_params(state: GraphState, operation: str, params: Dict[str, Any]) -> Optional[str]:
        """Error text when a vertex param the operation needs is absent or names no vertex."""
        names = _VERTEX_PARAMS.get(operation, ())
        absent = [key for key in names if params.get(key) is None]
        if absent:
            return f"{operation} requires {' and '.join(absent)} (vertex names)"
        known = set(GraphAnalyzer._vertex_ids(state))
        unknown = [
            (key, GraphAnalyzer._string_or_none(params.get(key)))
            for key in names
            if GraphAnalyzer._string_or_none(params.get(key)) not in known
        ]
        if not unknown:
            return None
        listed = "; ".join(f"{key}={value!r}" for key, value in unknown)
        return f"Unknown vertex for {operation} ({listed}); {GraphAnalyzer._vertex_list_text(state)}"

    @staticmethod
    def _build_edges(state: GraphState) -> List[Edge[str]]:
        return [Edge(edge.source, edge.target) for edge in state.edges]

    @staticmethod
    def _build_adjacency(state: GraphState, directed: bool) -> Dict[str, set[str]]:
        """Adjacency over all state vertices, so isolated vertices are included."""
        edges = GraphAnalyzer._build_edges(state)
        if directed:
            raw = GraphUtils.build_directed_adjacency_map(edges)
        else:
            raw = GraphUtils.build_adjacency_map(edges)
        adjacency = {k: set(v) for k, v in raw.items()}
        for vertex in state.vertices:
            adjacency.setdefault(vertex.id, set())
        return adjacency

    @staticmethod
    def _has_weights(state: GraphState) -> bool:
        return any(edge.weight is not None for edge in state.edges)

    @staticmethod
    def _weight_lookup(state: GraphState, directed: bool) -> Dict[Tuple[str, str], float]:
        """Weight of the cheapest edge between each pair of vertices (unweighted edges count 1).

        Undirected graphs store both orientations so either edge direction resolves.
        """
        weights: Dict[Tuple[str, str], float] = {}
        cheapest = GraphUtils.cheapest_parallel_edges(state.edges, GraphUtils.descriptor_weight, directed)
        for edge, weight in cheapest.values():
            weights[(edge.source, edge.target)] = weight
            if not directed:
                weights[(edge.target, edge.source)] = weight
        return weights

    @staticmethod
    def _edge_namer(state: GraphState, directed: bool) -> Callable[[str, str], Optional[str]]:
        """Name of the edge from u to v; among parallel edges, the cheapest (the one a path or MST uses)."""
        cheapest = GraphUtils.cheapest_parallel_edges(state.edges, GraphUtils.descriptor_weight, directed)

        def name_for(u: str, v: str) -> Optional[str]:
            entry = cheapest.get(GraphUtils.pair_key(u, v, directed))
            if entry is None:
                return None
            edge = entry[0]
            return GraphAnalyzer._string_or_none(edge.name) or GraphAnalyzer._string_or_none(edge.id)

        return name_for

    @staticmethod
    def _edge_names(state: GraphState, pairs: Sequence[Tuple[str, str]], directed: bool) -> List[str]:
        name_for = GraphAnalyzer._edge_namer(state, directed)
        names: List[str] = []
        for u, v in pairs:
            name = name_for(u, v)
            if name:
                names.append(name)
        return names

    @staticmethod
    def _path_pairs(path: Sequence[str]) -> List[Tuple[str, str]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # ------------------------------------------------------------------
    # Paths and orderings
    # ------------------------------------------------------------------
    @staticmethod
    def _shortest_path(state: GraphState, params: Dict[str, Any]) -> Result:
        directed = bool(state.directed)
        start = str(params["start"])
        goal = str(params["goal"])
        edges = GraphAnalyzer._build_edges(state)
        endpoints = {vertex for edge in edges for vertex in (edge.source, edge.target)}
        if start == goal and start not in endpoints:
            return {"operation": "shortest_path", "path": [start], "cost": 0.0, "highlight_vectors": []}

        cost: float
        if GraphAnalyzer._has_weights(state):
            try:
                path_data = GraphUtils.shortest_path_dijkstra(
                    edges,
                    start,
                    goal,
                    weight_lookup=GraphAnalyzer._weight_lookup(state, directed),
                    directed=directed,
                )
            except ValueError as exc:
                return {"error": str(exc)}
            if path_data is None:
                return {"path": None, "note": f"No path from {start} to {goal}."}
            path, cost = path_data
        else:
            found = GraphUtils.shortest_path_unweighted(edges, start, goal, directed=directed)
            if found is None:
                return {"path": None, "note": f"No path from {start} to {goal}."}
            path, cost = found, float(len(found) - 1)
        return {
            "operation": "shortest_path",
            "cost": cost,
            "path": path,
            "highlight_vectors": GraphAnalyzer._edge_names(state, GraphAnalyzer._path_pairs(path), directed),
        }

    @staticmethod
    def _mst(state: GraphState, params: Dict[str, Any]) -> Result:
        directed = bool(state.directed)
        adjacency = GraphAnalyzer._build_adjacency(state, directed)
        mst_edges, components = GraphUtils.minimum_spanning_forest(
            GraphAnalyzer._build_edges(state),
            weight_lookup=GraphAnalyzer._weight_lookup(state, directed),
            vertices=list(adjacency.keys()),
        )
        pairs = [(e.source, e.target) for e in mst_edges]
        result: Result = {
            "operation": "mst",
            # [source, target] lists: the result is sent back to the model as JSON
            "edges": [[u, v] for u, v in pairs],
            "highlight_vectors": GraphAnalyzer._edge_names(state, pairs, directed=False),
            "connected": components <= 1,
        }
        if components > 1:
            result["note"] = (
                f"Graph is disconnected ({components} components); "
                "returned a minimum spanning forest (one tree per component)."
            )
        return result

    @staticmethod
    def _topological_sort(state: GraphState, params: Dict[str, Any]) -> Result:
        if not state.directed:
            return {"error": f"topological_sort needs a directed graph; {state.name} is undirected."}
        adjacency = GraphAnalyzer._build_adjacency(state, True)
        order = GraphUtils.topological_sort(adjacency)
        if order is None:
            cycle = GraphUtils.find_directed_cycle(adjacency) or []
            shown = " -> ".join(cycle)
            return {"error": f"{state.name} has a cycle ({shown}), so it has no topological order."}
        return {"order": order}

    @staticmethod
    def _bfs(state: GraphState, params: Dict[str, Any]) -> Result:
        adjacency = GraphAnalyzer._build_adjacency(state, bool(state.directed))
        return {"order": GraphUtils.bfs_order(str(params["start"]), adjacency)}

    @staticmethod
    def _dfs(state: GraphState, params: Dict[str, Any]) -> Result:
        adjacency = GraphAnalyzer._build_adjacency(state, bool(state.directed))
        return {"order": GraphUtils.dfs_preorder(str(params["start"]), adjacency)}

    @staticmethod
    def _diameter(state: GraphState, params: Dict[str, Any]) -> Result:
        """Longest shortest path counted in edges, over the underlying undirected graph."""
        adjacency = GraphAnalyzer._build_adjacency(state, False)
        path, components = GraphUtils.graph_diameter(adjacency)
        if path is None:
            return {"error": f"{state.name} has no vertices."}
        result: Result = {
            "operation": "diameter",
            "path": path,
            "length": len(path) - 1,
            "connected": components <= 1,
            "highlight_vectors": GraphAnalyzer._edge_names(state, GraphAnalyzer._path_pairs(path), directed=False),
        }
        if components > 1:
            result["note"] = (
                f"Graph is disconnected ({components} components), so its diameter is infinite; "
                "path and length give the largest diameter of a single component."
            )
        return result

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    @staticmethod
    def _bridges(state: GraphState, params: Dict[str, Any]) -> Result:
        bridges = GraphUtils.find_multigraph_bridges(GraphAnalyzer._build_edges(state))
        return {
            "bridges": [[u, v] for u, v in bridges],
            "highlight_vectors": GraphAnalyzer._edge_names(state, bridges, directed=False),
        }

    @staticmethod
    def _articulation_points(state: GraphState, params: Dict[str, Any]) -> Result:
        points = GraphUtils.find_articulation_points(GraphAnalyzer._build_adjacency(state, False))
        return {"articulation_points": sorted(points)}

    @staticmethod
    def _euler_status(state: GraphState, params: Dict[str, Any]) -> Result:
        edges = GraphAnalyzer._build_edges(state)
        if state.directed:
            return {"status": GraphUtils.directed_euler_status(edges)}
        return {"status": GraphUtils.undirected_euler_status(edges)}

    @staticmethod
    def _bipartite(state: GraphState, params: Dict[str, Any]) -> Result:
        is_bipartite, color_map = GraphUtils.is_bipartite(GraphAnalyzer._build_adjacency(state, False))
        return {"is_bipartite": is_bipartite, "coloring": color_map}

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_root(state: GraphState, params: Dict[str, Any]) -> Union[str, Result]:
        """Root from params, else the stored tree root, else a default; an error result if unknown.

        The default is the only vertex without incoming edges on a directed graph, and the
        first vertex otherwise. Stored roots from older saves may be internal ids ("v0").
        """
        vertex_ids = GraphAnalyzer._vertex_ids(state)
        requested = GraphAnalyzer._string_or_none(params.get("root") or getattr(state, "root", None))
        if requested is None:
            return GraphAnalyzer._default_root(state, vertex_ids)
        if requested in vertex_ids:
            return requested
        for vertex in state.vertices:
            if vertex.name == requested:
                return vertex.id
        if requested.startswith("v") and requested[1:].isdigit():
            index = int(requested[1:])
            if 0 <= index < len(state.vertices):
                return state.vertices[index].id
        return {"error": f"Unknown root {requested!r}; {GraphAnalyzer._vertex_list_text(state)}"}

    @staticmethod
    def _default_root(state: GraphState, vertex_ids: List[str]) -> Union[str, Result]:
        if not vertex_ids:
            return {"error": f"{state.name} has no vertices."}
        if not state.directed:
            return vertex_ids[0]
        targets = {edge.target for edge in state.edges}
        sources = [vertex for vertex in vertex_ids if vertex not in targets]
        if len(sources) == 1:
            return sources[0]
        found = ", ".join(sorted(sources)) if sources else "none"
        return {
            "error": f"Pass a root: {state.name} needs exactly one vertex without incoming edges "
            f"to pick a default root (found: {found})."
        }

    @staticmethod
    def _not_a_tree_reason(state: GraphState) -> Optional[str]:
        vertex_ids = GraphAnalyzer._vertex_ids(state)
        edges = GraphAnalyzer._build_edges(state)
        if GraphUtils.is_tree_edges(vertex_ids, edges):
            return None
        components = len(GraphUtils.get_connected_components(GraphAnalyzer._build_adjacency(state, False)))
        if components > 1:
            return f"it is disconnected ({components} components)"
        if not vertex_ids:
            return "it has no vertices"
        return f"it has a cycle ({len(edges)} edges for {len(vertex_ids)} vertices; a tree has one fewer edge than vertices)"

    @staticmethod
    def _rooted_tree(state: GraphState, params: Dict[str, Any], operation: str) -> Union[RootedTree, Result]:
        """Root the graph's underlying undirected tree; an error result if it is not a tree."""
        reason = GraphAnalyzer._not_a_tree_reason(state)
        if reason is not None:
            return {"error": f"{operation} needs a tree, but {state.name} is not one: {reason}."}
        root = GraphAnalyzer._resolve_root(state, params)
        if isinstance(root, dict):
            return root
        rooted = GraphUtils.root_tree(GraphAnalyzer._build_adjacency(state, False), root)
        if rooted is None:
            return {"error": f"{operation}: could not root {state.name} at {root}."}
        parent, children = rooted
        return root, parent, children, GraphUtils.depths_from_parents(parent)

    @staticmethod
    def _levels(state: GraphState, params: Dict[str, Any]) -> Result:
        if GraphAnalyzer._not_a_tree_reason(state) is not None:
            return GraphAnalyzer._bfs_levels(state, params)
        rooted = GraphAnalyzer._rooted_tree(state, params, "levels")
        if isinstance(rooted, dict):
            return rooted
        root, _, children, _ = rooted
        levels: List[List[str]] = []
        current = [root]
        while current:
            levels.append(current)
            current = [child for node in current for child in children.get(node, [])]
        return {"root": root, "levels": levels}

    @staticmethod
    def _bfs_levels(state: GraphState, params: Dict[str, Any]) -> Result:
        """Levels of a graph that is not a tree: vertices grouped by BFS distance from the root.

        Edges are followed in their direction on a directed graph (a DAG's layers from its
        source); vertices the root cannot reach are not listed.
        """
        root = GraphAnalyzer._resolve_root(state, params)
        if isinstance(root, dict):
            return root
        adjacency = GraphAnalyzer._build_adjacency(state, bool(state.directed))
        levels = GraphUtils.tree_levels(root, adjacency) or [[root]]
        return {
            "root": root,
            "levels": levels,
            "note": f"{state.name} is not a tree, so levels are BFS distances (in edges) from {root}.",
        }

    @staticmethod
    def _lca(state: GraphState, params: Dict[str, Any]) -> Result:
        rooted = GraphAnalyzer._rooted_tree(state, params, "lca")
        if isinstance(rooted, dict):
            return rooted
        root, parent, _, depths = rooted
        lca_node = GraphUtils.lowest_common_ancestor(parent, depths, str(params["a"]), str(params["b"]))
        return {"root": root, "lca": lca_node}

    @staticmethod
    def _balance_children(state: GraphState, params: Dict[str, Any]) -> Result:
        rooted = GraphAnalyzer._rooted_tree(state, params, "balance_children")
        if isinstance(rooted, dict):
            return rooted
        root, _, children, _ = rooted
        return {"root": root, "children": GraphUtils.balance_children(root, children)}

    @staticmethod
    def _invert_children(state: GraphState, params: Dict[str, Any]) -> Result:
        rooted = GraphAnalyzer._rooted_tree(state, params, "invert_children")
        if isinstance(rooted, dict):
            return rooted
        root, _, children, _ = rooted
        return {"root": root, "children": GraphUtils.invert_children(children)}

    @staticmethod
    def _reroot(state: GraphState, params: Dict[str, Any]) -> Result:
        rooted = GraphAnalyzer._rooted_tree(state, params, "reroot")
        if isinstance(rooted, dict):
            return rooted
        _, parent, children, _ = rooted
        rerooted = GraphUtils.reroot_tree(parent, children, str(params["new_root"]))
        if rerooted is None:
            return {"error": "reroot failed"}
        return {"parent": rerooted[0], "children": rerooted[1]}

    # ------------------------------------------------------------------
    # Geometry on vertex positions
    # ------------------------------------------------------------------
    @staticmethod
    def _convex_hull(state: GraphState, params: Dict[str, Any]) -> Result:
        positions: List[Tuple[float, float]] = []
        vertex_at_pos: Dict[Tuple[float, float], str] = {}
        for v in state.vertices:
            if v.x is not None and v.y is not None:
                pos = (float(v.x), float(v.y))
                positions.append(pos)
                vertex_at_pos[pos] = v.id
        hull = positions if len(positions) < 3 else GeometryUtils.convex_hull(positions)
        # Lists, not tuples, for JSON serialization
        return {"hull": [list(p) for p in hull], "hull_vertices": [vertex_at_pos.get(p, "") for p in hull]}

    @staticmethod
    def _point_in_hull(state: GraphState, params: Dict[str, Any]) -> Result:
        x = params.get("x")
        y = params.get("y")
        if x is None or y is None:
            return {"error": "x and y coordinates are required for point_in_hull"}
        positions = [(float(v.x), float(v.y)) for v in state.vertices if v.x is not None and v.y is not None]
        hull = GeometryUtils.convex_hull(positions)
        inside = GeometryUtils.point_in_convex_hull((float(x), float(y)), hull)
        return {"inside": inside, "hull": [list(p) for p in hull]}


_HANDLERS: Dict[str, Callable[[GraphState, Dict[str, Any]], Result]] = {
    "shortest_path": GraphAnalyzer._shortest_path,
    "mst": GraphAnalyzer._mst,
    "topological_sort": GraphAnalyzer._topological_sort,
    "bridges": GraphAnalyzer._bridges,
    "articulation_points": GraphAnalyzer._articulation_points,
    "euler_status": GraphAnalyzer._euler_status,
    "bipartite": GraphAnalyzer._bipartite,
    "bfs": GraphAnalyzer._bfs,
    "dfs": GraphAnalyzer._dfs,
    "levels": GraphAnalyzer._levels,
    "diameter": GraphAnalyzer._diameter,
    "lca": GraphAnalyzer._lca,
    "balance_children": GraphAnalyzer._balance_children,
    "invert_children": GraphAnalyzer._invert_children,
    "reroot": GraphAnalyzer._reroot,
    "convex_hull": GraphAnalyzer._convex_hull,
    "point_in_hull": GraphAnalyzer._point_in_hull,
}
