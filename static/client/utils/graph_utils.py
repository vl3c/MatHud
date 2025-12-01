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

from typing import Dict, FrozenSet, Generic, List, Optional, Sequence, Set, Tuple, TypeVar

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

