"""Undirected graph drawable using segments as edges.

This module provides the UndirectedGraph class for representing undirected
graphs where Segment drawables define the edges between vertices.

Key Features:
    - Segment-based edge representation (bidirectional)
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
    from drawables.segment import Segment


class UndirectedGraph(Graph):
    """Undirected graph where edges are represented by Segment drawables.

    The graph vertices are inferred from the endpoints of segments.
    Descriptor computation is cached and invalidated on modifications.

    Attributes:
        _segments: List of Segment drawables representing undirected edges.
        _cached_descriptors: Cached (vertices, edges, adjacency_matrix) tuple.
    """

    def __init__(
        self,
        name: str,
        *,
        isolated_points: Optional[List["Point"]] = None,
        segments: Optional[List["Segment"]] = None,
    ) -> None:
        super().__init__(
            name=name,
            isolated_points=isolated_points,
            is_renderable=False,
        )
        self._segments: List["Segment"] = list(segments or [])
        self._cached_descriptors: Optional[
            tuple[List[GraphVertexDescriptor], List[GraphEdgeDescriptor], List[List[float]]]
        ] = None

    @property
    def segments(self) -> List["Segment"]:
        """Return list of segments."""
        return list(self._segments)

    def get_class_name(self) -> str:
        return "UndirectedGraph"

    # ------------------------------------------------------------------
    # Computed graph data from segments
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
        vertices, edges = GraphUtils.drawables_to_descriptors(self._segments, [], isolated_points=self._isolated_points)
        vertices = sorted(vertices, key=lambda v: v.id)
        adjacency_matrix = GraphUtils.adjacency_matrix_from_descriptors(vertices, edges, directed=False)
        self._cached_descriptors = (vertices, edges, adjacency_matrix)
        return self._cached_descriptors

    def remove_segment(self, segment: "Segment") -> bool:
        """Remove a segment reference from this graph."""
        if segment in self._segments:
            self._segments.remove(segment)
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
        state["args"]["segments"] = [getattr(s, "name", "") for s in self._segments]
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UndirectedGraph":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = UndirectedGraph(
            name=self.name,
            segments=deepcopy(self._segments, memo),
            isolated_points=deepcopy(self._isolated_points, memo),
        )
        memo[id(self)] = copied
        return copied
