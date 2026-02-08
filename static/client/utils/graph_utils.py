"""
MatHud Graph Theory Utilities Module

Abstract graph theory primitives for connectivity and structure validation.
Provides functions for analyzing graph topology independent of geometric interpretation.

Key Concepts:
    - Edge: Ordered pair of vertex identifiers (source, target)
    - Directed graph: Edge(A, B) and Edge(B, A) are distinct
    - Undirected graph: Edge(A, B) and Edge(B, A) are treated as equivalent

Graph Operations:
    - Adjacency map construction (directed and undirected)
    - Vertex degree analysis (degree, in-degree, out-degree)
    - Connectivity analysis (connected components)
    - Path and cycle detection

Use Cases:
    - Validate segment chains form simple paths
    - Validate segment loops form simple cycles
    - Detect branching (T-structures) or disconnected components
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, FrozenSet, Generic, List, Optional, Sequence, Set, Tuple, TypeVar

from geometry.graph_state import GraphEdgeDescriptor, GraphVertexDescriptor

if TYPE_CHECKING:
    from drawables.label import Label
    from drawables.point import Point
    from drawables.segment import Segment
    from drawables.vector import Vector

V = TypeVar("V")


class Edge(Generic[V]):
    """Ordered pair of vertex identifiers representing a directed edge.

    For directed graphs, Edge(A, B) and Edge(B, A) are distinct edges.
    For undirected operations, they are treated as equivalent.

    Attributes:
        source: The starting vertex of the edge
        target: The ending vertex of the edge
    """
    __slots__ = ("_source", "_target")

    def __init__(self, source: V, target: V) -> None:
        self._source: V = source
        self._target: V = target

    @property
    def source(self) -> V:
        return self._source

    @property
    def target(self) -> V:
        return self._target

    def reversed(self) -> Edge[V]:
        """Return a new edge with source and target swapped."""
        return Edge(self._target, self._source)

    def as_tuple(self) -> Tuple[V, V]:
        """Return the edge as an ordered tuple (source, target)."""
        return (self._source, self._target)

    def as_frozenset(self) -> FrozenSet[V]:
        """Return the edge as an unordered frozenset for undirected comparison."""
        return frozenset((self._source, self._target))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self._source == other._source and self._target == other._target

    def __hash__(self) -> int:
        return hash((self._source, self._target))

    def __repr__(self) -> str:
        return f"Edge({self._source!r}, {self._target!r})"


class GraphUtils:
    """Abstract graph theory utilities for topology analysis.

    All methods work with generic vertex identifiers and Edge objects,
    independent of geometric interpretation.
    """

    # ------------------------------------------------------------------
    # Drawable extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_weight_from_label(label: Optional["Label"]) -> Optional[float]:
        """Parse a numeric weight from a label if present."""
        if label is None:
            return None
        raw = getattr(label, "text", "")
        try:
            normalized = str(raw).strip()
            if normalized == "":
                return None
            return float(normalized)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _add_vertex_descriptor(
        vertices: Dict[str, GraphVertexDescriptor],
        point: "Point",
    ) -> None:
        """Ensure a point is represented as a vertex descriptor keyed by its name."""
        if point.name in vertices:
            return
        vertices[point.name] = GraphVertexDescriptor(
            point.name,
            name=point.name,
            x=point.x,
            y=point.y,
            color=getattr(point, "color", None),
        )

    @staticmethod
    def drawables_to_descriptors(
        segments: Sequence["Segment"],
        vectors: Sequence["Vector"],
        isolated_points: Optional[Sequence["Point"]] = None,
    ) -> Tuple[List[GraphVertexDescriptor], List[GraphEdgeDescriptor]]:
        """Convert drawables into graph descriptors with weights.

        Args:
            segments: Undirected edge drawables; weights come from segment labels.
            vectors: Directed edge drawables; weights come from the underlying segment labels.
            isolated_points: Points not referenced by an edge.

        Returns:
            Tuple of (vertices, edges) where vertices are GraphVertexDescriptor objects
            keyed by point names and edges are GraphEdgeDescriptor objects with optional
            weights, colors, and label references.
        """
        vertices: Dict[str, GraphVertexDescriptor] = {}
        edges: List[GraphEdgeDescriptor] = []

        for idx, segment in enumerate(segments):
            GraphUtils._add_vertex_descriptor(vertices, segment.point1)
            GraphUtils._add_vertex_descriptor(vertices, segment.point2)
            label = getattr(segment, "label", None)
            edge_id = segment.name or f"segment_{idx}"
            edges.append(
                GraphEdgeDescriptor(
                    edge_id,
                    segment.point1.name,
                    segment.point2.name,
                    weight=GraphUtils._extract_weight_from_label(label),
                    name=segment.name or None,
                    color=getattr(segment, "color", None),
                    directed=False,
                    segment_name=segment.name or None,
                    label_name=getattr(label, "name", None) or None,
                )
            )

        offset = len(edges)
        for j, vector in enumerate(vectors):
            origin = vector.origin
            tip = vector.tip
            GraphUtils._add_vertex_descriptor(vertices, origin)
            GraphUtils._add_vertex_descriptor(vertices, tip)
            label = getattr(getattr(vector, "segment", None), "label", None)
            edge_id = vector.name or f"vector_{offset + j}"
            edges.append(
                GraphEdgeDescriptor(
                    edge_id,
                    origin.name,
                    tip.name,
                    weight=GraphUtils._extract_weight_from_label(label),
                    name=vector.name or None,
                    color=getattr(vector, "color", None) or getattr(getattr(vector, "segment", None), "color", None),
                    directed=True,
                    vector_name=vector.name or None,
                    label_name=getattr(label, "name", None) or None,
                )
            )

        if isolated_points:
            for point in isolated_points:
                GraphUtils._add_vertex_descriptor(vertices, point)

        return list(vertices.values()), edges

    @staticmethod
    def adjacency_matrix_from_descriptors(
        vertices: Sequence[GraphVertexDescriptor],
        edges: Sequence[GraphEdgeDescriptor],
        *,
        directed: bool = False,
    ) -> List[List[float]]:
        """Build an adjacency matrix from descriptors (weight default = 1.0)."""
        index: Dict[str, int] = {v.id: idx for idx, v in enumerate(vertices)}
        size = len(vertices)
        matrix: List[List[float]] = [[0.0 for _ in range(size)] for _ in range(size)]

        for edge in edges:
            if edge.source not in index or edge.target not in index:
                continue
            w = float(edge.weight) if edge.weight is not None else 1.0
            i = index[edge.source]
            j = index[edge.target]
            matrix[i][j] = w
            if not directed:
                matrix[j][i] = w

        return matrix

    @staticmethod
    def build_adjacency_map(edges: Sequence[Edge[V]]) -> Dict[V, Set[V]]:
        """Build an undirected adjacency map from edges.

        Treats Edge(A, B) and Edge(B, A) as the same connection,
        adding both directions to the adjacency map.

        Args:
            edges: Sequence of Edge objects

        Returns:
            Dictionary mapping each vertex to its set of neighbors
        """
        adjacency: Dict[V, Set[V]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set()).add(edge.source)
        return adjacency

    @staticmethod
    def build_directed_adjacency_map(edges: Sequence[Edge[V]]) -> Dict[V, Set[V]]:
        """Build a directed adjacency map from edges.

        Preserves edge direction: only adds source -> target mapping.

        Args:
            edges: Sequence of Edge objects

        Returns:
            Dictionary mapping each source vertex to its set of targets
        """
        adjacency: Dict[V, Set[V]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            if edge.target not in adjacency:
                adjacency[edge.target] = set()
        return adjacency

    @staticmethod
    def get_all_vertices(edges: Sequence[Edge[V]]) -> Set[V]:
        """Extract all unique vertices from a sequence of edges.

        Args:
            edges: Sequence of Edge objects

        Returns:
            Set of all vertices appearing in any edge
        """
        vertices: Set[V] = set()
        for edge in edges:
            vertices.add(edge.source)
            vertices.add(edge.target)
        return vertices

    @staticmethod
    def get_vertex_degrees(adjacency: Dict[V, Set[V]]) -> Dict[V, int]:
        """Calculate degree of each vertex in an undirected graph.

        Degree is the number of edges incident to the vertex.

        Args:
            adjacency: Undirected adjacency map

        Returns:
            Dictionary mapping each vertex to its degree
        """
        return {vertex: len(neighbors) for vertex, neighbors in adjacency.items()}

    @staticmethod
    def get_out_degrees(adjacency: Dict[V, Set[V]]) -> Dict[V, int]:
        """Calculate out-degree of each vertex in a directed graph.

        Out-degree is the number of edges leaving the vertex.

        Args:
            adjacency: Directed adjacency map (source -> targets)

        Returns:
            Dictionary mapping each vertex to its out-degree
        """
        return {vertex: len(targets) for vertex, targets in adjacency.items()}

    @staticmethod
    def get_in_degrees(adjacency: Dict[V, Set[V]]) -> Dict[V, int]:
        """Calculate in-degree of each vertex in a directed graph.

        In-degree is the number of edges entering the vertex.

        Args:
            adjacency: Directed adjacency map (source -> targets)

        Returns:
            Dictionary mapping each vertex to its in-degree
        """
        in_degrees: Dict[V, int] = {vertex: 0 for vertex in adjacency}
        for targets in adjacency.values():
            for target in targets:
                in_degrees[target] = in_degrees.get(target, 0) + 1
        return in_degrees

    @staticmethod
    def get_endpoints(adjacency: Dict[V, Set[V]]) -> Set[V]:
        """Find vertices with degree 1 (endpoints of a path).

        Args:
            adjacency: Undirected adjacency map

        Returns:
            Set of vertices with exactly one neighbor
        """
        return {vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1}

    @staticmethod
    def is_connected(adjacency: Dict[V, Set[V]]) -> bool:
        """Check if an undirected graph is connected (single component).

        An empty graph is considered connected.

        Args:
            adjacency: Undirected adjacency map

        Returns:
            True if all vertices are reachable from any starting vertex
        """
        if not adjacency:
            return True

        start = next(iter(adjacency))
        visited: Set[V] = set()
        stack: List[V] = [start]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)

        return len(visited) == len(adjacency)

    @staticmethod
    def get_connected_components(adjacency: Dict[V, Set[V]]) -> List[Set[V]]:
        """Find all connected components in an undirected graph.

        Args:
            adjacency: Undirected adjacency map

        Returns:
            List of sets, each containing vertices of one component
        """
        if not adjacency:
            return []

        remaining = set(adjacency.keys())
        components: List[Set[V]] = []

        while remaining:
            start = next(iter(remaining))
            component: Set[V] = set()
            stack: List[V] = [start]

            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)

            components.append(component)
            remaining -= component

        return components

    @staticmethod
    def is_simple_path(edges: Sequence[Edge[V]]) -> bool:
        """Check if edges form a simple path (chain).

        A simple path requires:
            - At least one edge
            - Exactly 2 vertices with degree 1 (the endpoints)
            - All other vertices have degree 2
            - The graph is connected

        Args:
            edges: Sequence of Edge objects

        Returns:
            True if edges form a valid simple path
        """
        if len(edges) == 0:
            return False

        adjacency = GraphUtils.build_adjacency_map(edges)

        if not GraphUtils.is_connected(adjacency):
            return False

        degrees = GraphUtils.get_vertex_degrees(adjacency)
        degree_1_count = sum(1 for d in degrees.values() if d == 1)
        degree_2_count = sum(1 for d in degrees.values() if d == 2)

        if degree_1_count != 2:
            return False

        if degree_1_count + degree_2_count != len(degrees):
            return False

        return True

    @staticmethod
    def is_simple_cycle(edges: Sequence[Edge[V]]) -> bool:
        """Check if edges form a simple cycle (closed loop).

        A simple cycle requires:
            - At least 3 edges
            - All vertices have degree 2
            - The graph is connected

        Args:
            edges: Sequence of Edge objects

        Returns:
            True if edges form a valid simple cycle
        """
        if len(edges) < 3:
            return False

        adjacency = GraphUtils.build_adjacency_map(edges)

        if not GraphUtils.is_connected(adjacency):
            return False

        degrees = GraphUtils.get_vertex_degrees(adjacency)
        return all(d == 2 for d in degrees.values())

    @staticmethod
    def order_path_vertices(edges: Sequence[Edge[V]]) -> Optional[List[V]]:
        """Order vertices along a simple path from one endpoint to the other.

        Args:
            edges: Sequence of Edge objects forming a simple path

        Returns:
            Ordered list of vertices from first endpoint to last endpoint,
            or None if edges do not form a valid simple path
        """
        if not GraphUtils.is_simple_path(edges):
            return None

        adjacency = GraphUtils.build_adjacency_map(edges)
        endpoints = GraphUtils.get_endpoints(adjacency)

        start = min(endpoints)

        ordered: List[V] = [start]
        visited: Set[V] = {start}
        current = start

        while True:
            neighbors = adjacency[current] - visited
            if not neighbors:
                break
            next_vertex = next(iter(neighbors))
            ordered.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex

        return ordered

    @staticmethod
    def order_cycle_vertices(edges: Sequence[Edge[V]]) -> Optional[List[V]]:
        """Order vertices around a simple cycle.

        Starts from the lexicographically smallest vertex and traverses
        the cycle in a deterministic direction.

        Args:
            edges: Sequence of Edge objects forming a simple cycle

        Returns:
            Ordered list of vertices around the cycle (without repeating first),
            or None if edges do not form a valid simple cycle
        """
        if not GraphUtils.is_simple_cycle(edges):
            return None

        adjacency = GraphUtils.build_adjacency_map(edges)

        start = min(adjacency.keys())

        ordered: List[V] = [start]
        visited: Set[V] = {start}
        current = start

        while len(ordered) < len(adjacency):
            neighbors = adjacency[current] - visited
            if not neighbors:
                break
            next_vertex = min(neighbors)
            ordered.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex

        return ordered

    # ------------------------------------------------------------------
    # Weighted helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_weight(
        edge: Edge[V],
        weight_lookup: Optional[Dict[Tuple[V, V], float]] = None,
        weight_fn: Optional[Callable[[Edge[V]], float]] = None,
        default_weight: float = 1.0,
        undirected: bool = False,
    ) -> float:
        if weight_fn is not None:
            return float(weight_fn(edge))
        if weight_lookup is None:
            return float(default_weight)
        key = (edge.source, edge.target)
        if key in weight_lookup:
            return float(weight_lookup[key])
        if undirected:
            rev = (edge.target, edge.source)
            if rev in weight_lookup:
                return float(weight_lookup[rev])
        return float(default_weight)

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------
    @staticmethod
    def shortest_path_unweighted(
        edges: Sequence[Edge[V]],
        start: V,
        goal: V,
        directed: bool = False,
    ) -> Optional[List[V]]:
        adjacency = (
            GraphUtils.build_directed_adjacency_map(edges)
            if directed
            else GraphUtils.build_adjacency_map(edges)
        )
        if start not in adjacency or goal not in adjacency:
            return None

        from collections import deque

        queue: deque[V] = deque([start])
        visited: Set[V] = {start}
        parent: Dict[V, V] = {}

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

        if goal not in visited:
            return None

        path: List[V] = []
        node = goal
        while True:
            path.append(node)
            if node == start:
                break
            node = parent[node]
        path.reverse()
        return path

    @staticmethod
    def shortest_path_dijkstra(
        edges: Sequence[Edge[V]],
        start: V,
        goal: V,
        weight_lookup: Optional[Dict[Tuple[V, V], float]] = None,
        weight_fn: Optional[Callable[[Edge[V]], float]] = None,
        default_weight: float = 1.0,
        directed: bool = False,
    ) -> Optional[Tuple[List[V], float]]:
        adjacency: Dict[V, List[Tuple[V, float]]] = {}
        for edge in edges:
            w = GraphUtils._resolve_weight(edge, weight_lookup, weight_fn, default_weight, not directed)
            adjacency.setdefault(edge.source, []).append((edge.target, w))
            if not directed:
                adjacency.setdefault(edge.target, []).append((edge.source, w))
            else:
                adjacency.setdefault(edge.target, [])

        if start not in adjacency or goal not in adjacency:
            return None

        import heapq

        dist: Dict[V, float] = {vertex: float("inf") for vertex in adjacency}
        prev: Dict[V, V] = {}
        dist[start] = 0.0
        heap: List[Tuple[float, V]] = [(0.0, start)]

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current == goal:
                break
            if current_dist > dist[current]:
                continue
            for neighbor, weight in adjacency[current]:
                candidate = current_dist + weight
                if candidate < dist[neighbor]:
                    dist[neighbor] = candidate
                    prev[neighbor] = current
                    heapq.heappush(heap, (candidate, neighbor))

        if dist[goal] == float("inf"):
            return None

        path: List[V] = []
        node = goal
        while True:
            path.append(node)
            if node == start:
                break
            node = prev[node]
        path.reverse()
        return path, dist[goal]

    # ------------------------------------------------------------------
    # Spanning tree and ordering
    # ------------------------------------------------------------------
    @staticmethod
    def minimum_spanning_tree(
        edges: Sequence[Edge[V]],
        weight_lookup: Optional[Dict[Tuple[V, V], float]] = None,
        weight_fn: Optional[Callable[[Edge[V]], float]] = None,
        default_weight: float = 1.0,
    ) -> List[Edge[V]]:
        adjacency: Dict[V, List[Tuple[V, float]]] = {}
        undirected_edges: List[Tuple[V, V, float]] = []
        seen: Set[FrozenSet[V]] = set()
        for edge in edges:
            key = edge.as_frozenset()
            if key in seen:
                continue
            seen.add(key)
            weight = GraphUtils._resolve_weight(edge, weight_lookup, weight_fn, default_weight, True)
            undirected_edges.append((edge.source, edge.target, weight))
            adjacency.setdefault(edge.source, []).append((edge.target, weight))
            adjacency.setdefault(edge.target, []).append((edge.source, weight))

        if not adjacency:
            return []

        import heapq

        start = next(iter(adjacency))
        visited: Set[V] = {start}
        heap: List[Tuple[float, V, V]] = []
        for neighbor, weight in adjacency[start]:
            heapq.heappush(heap, (weight, start, neighbor))

        result: List[Edge[V]] = []
        while heap and len(visited) < len(adjacency):
            weight, u, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            result.append(Edge(u, v))
            for nxt, w in adjacency[v]:
                if nxt not in visited:
                    heapq.heappush(heap, (w, v, nxt))

        if len(visited) != len(adjacency):
            return []
        return result

    @staticmethod
    def topological_sort(adjacency: Dict[V, Set[V]]) -> Optional[List[V]]:
        adj = {k: set(v) for k, v in adjacency.items()}
        in_deg: Dict[V, int] = {v: 0 for v in adj}
        for targets in adj.values():
            for t in targets:
                in_deg[t] = in_deg.get(t, 0) + 1
                if t not in adj:
                    adj[t] = set()

        queue: List[V] = [v for v, d in in_deg.items() if d == 0]
        order: List[V] = []

        while queue:
            current = queue.pop()
            order.append(current)
            for neighbor in adj[current]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(adj):
            return None
        return order

    # ------------------------------------------------------------------
    # Connectivity analysis
    # ------------------------------------------------------------------
    @staticmethod
    def find_bridges(adjacency: Dict[V, Set[V]]) -> List[Tuple[V, V]]:
        time = 0
        visited: Set[V] = set()
        disc: Dict[V, int] = {}
        low: Dict[V, int] = {}
        bridges: List[Tuple[V, V]] = []

        def dfs(u: V, parent: Optional[V]) -> None:
            nonlocal time
            visited.add(u)
            time += 1
            disc[u] = low[u] = time
            for v in adjacency[u]:
                if v == parent:
                    continue
                if v not in visited:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                else:
                    low[u] = min(low[u], disc[v])

        for vertex in adjacency:
            if vertex not in visited:
                dfs(vertex, None)
        return bridges

    @staticmethod
    def find_articulation_points(adjacency: Dict[V, Set[V]]) -> Set[V]:
        time = 0
        visited: Set[V] = set()
        disc: Dict[V, int] = {}
        low: Dict[V, int] = {}
        parent: Dict[V, Optional[V]] = {}
        ap: Set[V] = set()

        def dfs(u: V) -> None:
            nonlocal time
            visited.add(u)
            children = 0
            time += 1
            disc[u] = low[u] = time
            for v in adjacency[u]:
                if v not in visited:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for vertex in adjacency:
            if vertex not in visited:
                parent[vertex] = None
                dfs(vertex)
        return ap

    @staticmethod
    def euler_status(adjacency: Dict[V, Set[V]]) -> Optional[str]:
        if not GraphUtils.is_connected(adjacency):
            return None
        degrees = GraphUtils.get_vertex_degrees(adjacency)
        odd = sum(1 for d in degrees.values() if d % 2 == 1)
        if odd == 0:
            return "cycle"
        if odd == 2:
            return "path"
        return None

    @staticmethod
    def is_bipartite(adjacency: Dict[V, Set[V]]) -> Tuple[bool, Dict[V, int]]:
        color: Dict[V, int] = {}
        from collections import deque

        for start in adjacency:
            if start in color:
                continue
            queue: deque[V] = deque([start])
            color[start] = 0
            while queue:
                u = queue.popleft()
                for v in adjacency[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, color
        return True, color

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_tree(adjacency: Dict[V, Set[V]]) -> bool:
        if not adjacency:
            return False
        if not GraphUtils.is_connected(adjacency):
            return False
        edge_count = sum(len(neigh) for neigh in adjacency.values()) // 2
        return edge_count == len(adjacency) - 1

    @staticmethod
    def root_tree(adjacency: Dict[V, Set[V]], root: V) -> Optional[Tuple[Dict[V, Optional[V]], Dict[V, List[V]]]]:
        if root not in adjacency:
            return None
        parent: Dict[V, Optional[V]] = {root: None}
        children: Dict[V, List[V]] = {root: []}
        stack: List[V] = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                children.setdefault(current, []).append(neighbor)
                children.setdefault(neighbor, [])
                stack.append(neighbor)
        if len(parent) != len(adjacency):
            return None
        return parent, children

    @staticmethod
    def bfs_order(root: V, adjacency: Dict[V, Set[V]]) -> Optional[List[V]]:
        if root not in adjacency:
            return None
        from collections import deque

        order: List[V] = []
        queue: deque[V] = deque([root])
        visited: Set[V] = {root}
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return order

    @staticmethod
    def dfs_preorder(root: V, adjacency: Dict[V, Set[V]]) -> Optional[List[V]]:
        if root not in adjacency:
            return None
        order: List[V] = []
        stack: List[V] = [root]
        visited: Set[V] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
        return order

    @staticmethod
    def node_depths(root: V, adjacency: Dict[V, Set[V]]) -> Optional[Dict[V, int]]:
        if root not in adjacency:
            return None
        depths: Dict[V, int] = {root: 0}
        from collections import deque

        queue: deque[V] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in depths:
                    continue
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
        return depths

    @staticmethod
    def tree_levels(root: V, adjacency: Dict[V, Set[V]]) -> Optional[List[List[V]]]:
        if root not in adjacency:
            return None
        from collections import deque

        levels: List[List[V]] = []
        queue: deque[Tuple[V, int]] = deque([(root, 0)])
        seen: Set[V] = {root}
        while queue:
            node, depth = queue.popleft()
            if depth == len(levels):
                levels.append([])
            levels[depth].append(node)
            for neighbor in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
        return levels

    @staticmethod
    def tree_diameter(adjacency: Dict[V, Set[V]]) -> Optional[List[V]]:
        if not adjacency:
            return None

        def farthest(start: V) -> Tuple[V, Dict[V, V]]:
            from collections import deque

            queue: deque[V] = deque([start])
            visited: Set[V] = {start}
            parent: Dict[V, V] = {start: start}
            last = start
            while queue:
                node = queue.popleft()
                last = node
                for neighbor in adjacency[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
            return last, parent

        start = next(iter(adjacency))
        far_node, _ = farthest(start)
        end_node, parent = farthest(far_node)

        path: List[V] = []
        node = end_node
        while True:
            path.append(node)
            if node == far_node:
                break
            node = parent[node]
        path.reverse()
        return path

    @staticmethod
    def lowest_common_ancestor(
        parent: Dict[V, Optional[V]],
        depths: Dict[V, int],
        a: V,
        b: V,
    ) -> Optional[V]:
        if a not in depths or b not in depths:
            return None

        def climb(node: V, steps: int) -> V:
            current = node
            for _ in range(steps):
                parent_node = parent.get(current)
                if parent_node is None:
                    return current
                current = parent_node
            return current

        da = depths[a]
        db = depths[b]
        if da > db:
            a = climb(a, da - db)
        elif db > da:
            b = climb(b, db - da)

        while a != b:
            pa = parent.get(a)
            pb = parent.get(b)
            if pa is None or pb is None:
                return None
            a = pa
            b = pb
        return a

    @staticmethod
    def subtree_sizes(root: V, children: Dict[V, List[V]]) -> Dict[V, int]:
        sizes: Dict[V, int] = {}

        def dfs(node: V) -> int:
            total = 1
            for child in children.get(node, []):
                total += dfs(child)
            sizes[node] = total
            return total

        dfs(root)
        return sizes

    @staticmethod
    def balance_children(root: V, children: Dict[V, List[V]]) -> Dict[V, List[V]]:
        sizes = GraphUtils.subtree_sizes(root, children)
        balanced: Dict[V, List[V]] = {}
        for node, kids in children.items():
            balanced[node] = sorted(kids, key=lambda c: sizes.get(c, 0), reverse=True)
        return balanced

    @staticmethod
    def invert_children(children: Dict[V, List[V]]) -> Dict[V, List[V]]:
        return {node: list(reversed(kids)) for node, kids in children.items()}

    @staticmethod
    def reroot_tree(
        parent: Dict[V, Optional[V]],
        children: Dict[V, List[V]],
        new_root: V,
    ) -> Optional[Tuple[Dict[V, Optional[V]], Dict[V, List[V]]]]:
        if new_root not in parent:
            return None

        new_parent: Dict[V, Optional[V]] = {new_root: None}
        new_children: Dict[V, List[V]] = {new_root: []}

        stack: List[V] = [new_root]
        while stack:
            node = stack.pop()
            neighbors: List[V] = []
            if parent.get(node) is not None:
                neighbors.append(parent[node])  # type: ignore[arg-type]
            neighbors.extend(children.get(node, []))

            for nxt in neighbors:
                if nxt in new_parent:
                    continue
                new_parent[nxt] = node
                new_children.setdefault(node, []).append(nxt)
                new_children.setdefault(nxt, [])
                stack.append(nxt)

        return new_parent, new_children

    # =========================================================================
    # Geometric Edge Crossing Detection
    # =========================================================================

    @staticmethod
    def segments_cross(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        eps: float = 0.01,
    ) -> bool:
        """
        Check if line segment p1-p2 crosses segment p3-p4.

        Uses parametric line intersection to detect if segments cross at an
        interior point (not at endpoints).

        Args:
            p1: First endpoint of segment 1
            p2: Second endpoint of segment 1
            p3: First endpoint of segment 2
            p4: Second endpoint of segment 2
            eps: Epsilon for endpoint exclusion (default 0.01)

        Returns:
            True if segments cross at an interior point, False otherwise.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False  # Parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection is strictly inside both segments (not at endpoints)
        return eps < t < 1 - eps and eps < u < 1 - eps

    @staticmethod
    def count_edge_crossings(
        edges: List["Edge[V]"],
        positions: Dict[V, Tuple[float, float]],
    ) -> int:
        """
        Count the number of edge pairs that cross in a graph layout.

        Two edges cross if they intersect at a point that is not a shared
        endpoint. Edges that share a vertex are not counted as crossing.

        Args:
            edges: List of edges in the graph
            positions: Mapping from vertex to (x, y) coordinates

        Returns:
            Number of crossing edge pairs.
        """
        crossings = 0
        edge_list = list(edges)

        for i in range(len(edge_list)):
            e1 = edge_list[i]
            p1 = positions.get(e1.source)
            p2 = positions.get(e1.target)
            if p1 is None or p2 is None:
                continue

            for j in range(i + 1, len(edge_list)):
                e2 = edge_list[j]
                # Skip edges sharing a vertex
                if e1.source in (e2.source, e2.target) or e1.target in (e2.source, e2.target):
                    continue

                p3 = positions.get(e2.source)
                p4 = positions.get(e2.target)
                if p3 is None or p4 is None:
                    continue

                if GraphUtils.segments_cross(p1, p2, p3, p4):
                    crossings += 1

        return crossings

    @staticmethod
    def is_edge_orthogonal(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        tolerance: float = 0.01,
    ) -> bool:
        """
        Check if an edge is orthogonal (horizontal or vertical).

        An edge is orthogonal if the x-coordinates are equal (vertical)
        or the y-coordinates are equal (horizontal), within tolerance.

        Args:
            p1: First endpoint (x, y)
            p2: Second endpoint (x, y)
            tolerance: Relative tolerance for coordinate comparison

        Returns:
            True if the edge is horizontal or vertical.
        """
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Use relative tolerance based on edge length
        edge_length = max(dx, dy, 1.0)
        tol = tolerance * edge_length

        # Horizontal: same y (within tolerance)
        if dy < tol:
            return True
        # Vertical: same x (within tolerance)
        if dx < tol:
            return True

        return False

    @staticmethod
    def count_orthogonal_edges(
        edges: List["Edge[V]"],
        positions: Dict[V, Tuple[float, float]],
        tolerance: float = 0.01,
    ) -> Tuple[int, int]:
        """
        Count orthogonal vs total edges in a graph layout.

        Args:
            edges: List of edges in the graph
            positions: Mapping from vertex to (x, y) coordinates
            tolerance: Relative tolerance for orthogonality check

        Returns:
            Tuple of (orthogonal_count, total_count)
        """
        orthogonal = 0
        total = 0

        for edge in edges:
            p1 = positions.get(edge.source)
            p2 = positions.get(edge.target)
            if p1 is None or p2 is None:
                continue

            total += 1
            if GraphUtils.is_edge_orthogonal(p1, p2, tolerance):
                orthogonal += 1

        return (orthogonal, total)

    @staticmethod
    def edges_overlap(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        eps: float = 1e-9,
    ) -> bool:
        """
        Check if two edges are collinear and overlap (share more than just an endpoint).

        Two edges overlap if they lie on the same line and their projections
        onto that line overlap by more than just a point.

        Args:
            p1, p2: Endpoints of first edge
            p3, p4: Endpoints of second edge
            eps: Tolerance for floating-point comparisons

        Returns:
            True if the edges are collinear and overlap.
        """
        # Cross product to check collinearity
        def cross_product(
            o: Tuple[float, float],
            a: Tuple[float, float],
            b: Tuple[float, float],
        ) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Check if all 4 points are collinear
        if abs(cross_product(p1, p2, p3)) > eps or abs(cross_product(p1, p2, p4)) > eps:
            return False

        # They are collinear - check if projections overlap
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

    @staticmethod
    def point_on_segment(
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
        eps: float = 1e-9,
    ) -> bool:
        """
        Check if a point lies strictly inside a segment (not at endpoints).

        Args:
            point: The point to check
            seg_start, seg_end: Segment endpoints
            eps: Tolerance for floating-point comparisons

        Returns:
            True if point is on the segment interior (not at endpoints).
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
            # Use x-coordinate
            min_x, max_x = min(x1, x2), max(x1, x2)
            return min_x + eps < px < max_x - eps
        else:
            # Use y-coordinate
            min_y, max_y = min(y1, y2), max(y1, y2)
            return min_y + eps < py < max_y - eps

    @staticmethod
    def adjacent_edges_overlap(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        shared_is_p2_and_p3: bool,
        eps: float = 1e-9,
    ) -> bool:
        """
        Check if two edges that share a vertex overlap.

        Two adjacent edges overlap if they are collinear and the non-shared
        vertex of one edge lies on the other edge.

        Args:
            p1, p2: Endpoints of first edge
            p3, p4: Endpoints of second edge
            shared_is_p2_and_p3: True if p2==p3 is the shared vertex,
                                 False if some other combination
            eps: Tolerance for floating-point comparisons

        Returns:
            True if the adjacent edges overlap.
        """
        # Determine shared and non-shared vertices based on which are equal
        # We need to check if the non-shared vertex of one edge lies on the other

        # Check if p1 (non-shared from edge1) lies on edge2 (p3-p4)
        if GraphUtils.point_on_segment(p1, p3, p4, eps):
            return True

        # Check if p4 (non-shared from edge2) lies on edge1 (p1-p2)
        if GraphUtils.point_on_segment(p4, p1, p2, eps):
            return True

        # Also check p2 on edge2 and p3 on edge1 for other shared vertex cases
        if GraphUtils.point_on_segment(p2, p3, p4, eps):
            return True
        if GraphUtils.point_on_segment(p3, p1, p2, eps):
            return True

        return False

    @staticmethod
    def count_edge_overlaps(
        edges: List["Edge[V]"],
        positions: Dict[V, Tuple[float, float]],
    ) -> int:
        """
        Count the number of edge pairs that overlap in a graph layout.

        Two edges overlap if:
        1. They don't share a vertex and are collinear with overlapping spans, OR
        2. They share a vertex and the non-shared vertex of one lies on the other

        Args:
            edges: List of edges in the graph
            positions: Mapping from vertex to (x, y) coordinates

        Returns:
            Number of overlapping edge pairs.
        """
        overlaps = 0
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
                    if GraphUtils.adjacent_edges_overlap(p1, p2, p3, p4, True):
                        overlaps += 1
                else:
                    # Check for non-adjacent edge overlap
                    if GraphUtils.edges_overlap(p1, p2, p3, p4):
                        overlaps += 1

        return overlaps

    @staticmethod
    def compute_edge_length(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Compute Euclidean length of an edge."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def get_edge_lengths(
        edges: List[Edge[str]],
        positions: Dict[str, Tuple[float, float]],
    ) -> List[float]:
        """Get list of edge lengths for all edges."""
        lengths: List[float] = []
        for e in edges:
            p1 = positions.get(e.source)
            p2 = positions.get(e.target)
            if p1 is not None and p2 is not None:
                lengths.append(GraphUtils.compute_edge_length(p1, p2))
        return lengths

    @staticmethod
    def count_edges_with_same_length(
        edges: List[Edge[str]],
        positions: Dict[str, Tuple[float, float]],
        tolerance: float = 0.1,
    ) -> Tuple[int, int, float]:
        """
        Count edges that have the same length (within tolerance).

        Args:
            edges: List of edges
            positions: Vertex positions
            tolerance: Relative tolerance for length comparison (0.1 = 10%)

        Returns:
            (count_of_most_common_length, total_edges, most_common_length)
        """
        lengths = GraphUtils.get_edge_lengths(edges, positions)
        if not lengths:
            return 0, 0, 0.0

        # Group lengths by similarity
        # Add small epsilon for floating point comparison robustness
        eps = 1e-9
        groups: List[List[float]] = []
        for length in lengths:
            found_group = False
            for group in groups:
                # Check if this length is similar to the group's first element
                # Use relative tolerance plus small epsilon for floating point safety
                if abs(length - group[0]) <= tolerance * group[0] + eps:
                    group.append(length)
                    found_group = True
                    break
            if not found_group:
                groups.append([length])

        # Find largest group
        largest_group = max(groups, key=len) if groups else []
        most_common_length = sum(largest_group) / len(largest_group) if largest_group else 0.0

        return len(largest_group), len(lengths), most_common_length

    @staticmethod
    def edge_length_variance(
        edges: List[Edge[str]],
        positions: Dict[str, Tuple[float, float]],
    ) -> float:
        """
        Compute variance of edge lengths.

        Lower variance means more uniform edge lengths.
        Returns 0.0 if fewer than 2 edges.
        """
        lengths = GraphUtils.get_edge_lengths(edges, positions)
        if len(lengths) < 2:
            return 0.0

        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return variance

    @staticmethod
    def edge_length_uniformity_ratio(
        edges: List[Edge[str]],
        positions: Dict[str, Tuple[float, float]],
        tolerance: float = 0.1,
    ) -> float:
        """
        Compute what fraction of edges have the most common length.

        Returns a value between 0 and 1, where 1 means all edges have the same length.
        """
        same_count, total, _ = GraphUtils.count_edges_with_same_length(edges, positions, tolerance)
        if total == 0:
            return 1.0
        return same_count / total

