"""Directed graph drawable using vectors as edges.

This module provides the DirectedGraph class for representing directed
graphs where Vector drawables define the edges between vertices.

Key Features:
    - Vector-based edge representation with direction
    - Lazy computation of vertex/edge descriptors
    - Cached adjacency matrix generation
    - Integration with graph algorithms via GraphUtils
    - Support for isolated vertices without edges
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.graph import Graph
from geometry.graph_state import GraphEdgeDescriptor, GraphVertexDescriptor
from utils.graph_utils import GraphUtils

if TYPE_CHECKING:
    from drawables.point import Point
    from drawables.vector import Vector


class DirectedGraph(Graph):
    """Directed graph where edges are represented by Vector drawables.

    The graph vertices are inferred from the start/end points of vectors.
    Descriptor computation is cached and invalidated on modifications.

    Attributes:
        _vectors: List of Vector drawables representing directed edges.
        _cached_descriptors: Cached (vertices, edges, adjacency_matrix) tuple.
    """

    def __init__(
        self,
        name: str,
        *,
        isolated_points: Optional[List["Point"]] = None,
        vectors: Optional[List["Vector"]] = None,
    ) -> None:
        super().__init__(
            name=name,
            isolated_points=isolated_points,
            is_renderable=False,
        )
        self._vectors: List["Vector"] = list(vectors or [])
        self._cached_descriptors: Optional[
            tuple[List[GraphVertexDescriptor], List[GraphEdgeDescriptor], List[List[float]]]
        ] = None

    @property
    def directed(self) -> bool:
        return True

    @property
    def vectors(self) -> List["Vector"]:
        """Return list of vectors."""
        return list(self._vectors)

    def get_class_name(self) -> str:
        return "DirectedGraph"

    # ------------------------------------------------------------------
    # Computed graph data from vectors
    # ------------------------------------------------------------------
    def _invalidate_cache(self) -> None:
        self._cached_descriptors = None

    def remove_point(self, point: "Point") -> bool:
        removed = super().remove_point(point)
        if removed:
            self._invalidate_cache()
        return removed

    def _compute_descriptors(self) -> tuple[List[GraphVertexDescriptor], List[GraphEdgeDescriptor], List[List[float]]]:
        if self._cached_descriptors is not None:
            return self._cached_descriptors
        vertices, edges = GraphUtils.drawables_to_descriptors([], self._vectors, isolated_points=self._isolated_points)
        vertices = sorted(vertices, key=lambda v: v.id)
        adjacency_matrix = GraphUtils.adjacency_matrix_from_descriptors(vertices, edges, directed=True)
        self._cached_descriptors = (vertices, edges, adjacency_matrix)
        return self._cached_descriptors

    def remove_vector(self, vector: "Vector") -> bool:
        """Remove a vector reference from this graph."""
        if vector in self._vectors:
            self._vectors.remove(vector)
            self._invalidate_cache()
            return True
        return False

    @property
    def vertices(self) -> Dict[str, str]:
        vertices, _, _ = self._compute_descriptors()
        return {v.id: v.name or v.id for v in vertices}

    @property
    def edges(self) -> List[Dict[str, Any]]:
        _, edges, _ = self._compute_descriptors()
        return [e.to_dict() for e in edges]

    @property
    def adjacency_matrix(self) -> List[List[float]]:
        _, _, matrix = self._compute_descriptors()
        return matrix

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"]["vectors"] = [getattr(v, "name", "") for v in self._vectors]
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DirectedGraph":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = DirectedGraph(
            name=self.name,
            vectors=deepcopy(self._vectors, memo),
            isolated_points=deepcopy(self._isolated_points, memo),
        )
        memo[id(self)] = copied
        return copied
