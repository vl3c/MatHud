from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry.graph_state import GraphEdgeDescriptor, GraphState, TreeState
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
    def analyze(state: GraphState, operation: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        directed = bool(state.directed or state.graph_type == "dag")
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
            root = params.get("root") or getattr(state, "root", None)
            levels = GraphUtils.tree_levels(root, adjacency) if root else None
            return {"levels": levels}

        if operation == "diameter":
            path = GraphUtils.tree_diameter(adjacency)
            result["path"] = path
            if path:
                result["highlight_vectors"] = GraphAnalyzer._edge_names_from_path(state, path, directed=False)
            return result

        if operation == "lca":
            root = params.get("root") or getattr(state, "root", None)
            a = params.get("a")
            b = params.get("b")
            if root is None or a is None or b is None:
                return {"error": "root, a, and b are required for lca"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                return {"error": "invalid tree structure"}
            parent, children = rooted
            depths = GraphUtils.node_depths(root, adjacency) or {}
            lca_node = GraphUtils.lowest_common_ancestor(parent, depths, a, b)
            return {"lca": lca_node}

        if operation == "balance_children":
            root = params.get("root") or getattr(state, "root", None)
            if root is None:
                return {"error": "root is required for balance_children"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                return {"error": "invalid tree structure"}
            _, children = rooted
            balanced = GraphUtils.balance_children(root, children)
            return {"children": balanced}

        if operation == "invert_children":
            root = params.get("root") or getattr(state, "root", None)
            if root is None:
                return {"error": "root is required for invert_children"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                return {"error": "invalid tree structure"}
            _, children = rooted
            inverted = GraphUtils.invert_children(children)
            return {"children": inverted}

        if operation == "reroot":
            root = params.get("root") or getattr(state, "root", None)
            new_root = params.get("new_root")
            if root is None or new_root is None:
                return {"error": "root and new_root are required for reroot"}
            rooted = GraphUtils.root_tree(adjacency, root)
            if rooted is None:
                return {"error": "invalid tree structure"}
            parent, children = rooted
            rerooted = GraphUtils.reroot_tree(parent, children, new_root)
            if rerooted is None:
                return {"error": "reroot failed"}
            return {"parent": rerooted[0], "children": rerooted[1]}

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



