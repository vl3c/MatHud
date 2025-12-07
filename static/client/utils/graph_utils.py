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

from typing import Callable, Dict, FrozenSet, Generic, List, Optional, Sequence, Set, Tuple, TypeVar

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
        in_deg: Dict[V, int] = {v: 0 for v in adjacency}
        for targets in adjacency.values():
            for t in targets:
                in_deg[t] = in_deg.get(t, 0) + 1
                if t not in adjacency:
                    adjacency[t] = set()

        queue: List[V] = [v for v, d in in_deg.items() if d == 0]
        order: List[V] = []

        while queue:
            current = queue.pop()
            order.append(current)
            for neighbor in adjacency[current]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(adjacency):
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

