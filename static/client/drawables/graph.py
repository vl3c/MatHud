"""Abstract graph drawable for representing graph data structures.

This module provides the Graph abstract base class that defines the interface
for graph drawables. Concrete implementations include DirectedGraph and
UndirectedGraph.

Key Features:
    - Abstract interface for vertex and edge access
    - Adjacency matrix generation
    - Isolated point management for vertices without edges
    - Tracking of pre-existing points and edges the graph reuses but does not own
    - Serialization support for workspace persistence
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from drawables.drawable import Drawable

if TYPE_CHECKING:
    from drawables.point import Point


class Graph(Drawable):
    """Abstract graph interface for directed and undirected graphs.

    Subclasses implement edge storage and provide vertex/edge descriptors.

    Attributes:
        _isolated_points: List of vertex points not connected by edges.
        _preexisting_points: Vertex points that existed before the graph was created.
        _preexisting_edges: Edge segments or vectors that existed before the graph was created.
        _original_edge_labels: (edge, text, visible) label of each pre-existing edge the graph
            relabelled with its weight, restored when the graph is deleted.
        directed: Whether the graph is directed (override in subclass).
    """

    def __init__(
        self,
        name: str,
        *,
        isolated_points: Optional[List["Point"]] = None,
        is_renderable: bool = False,
    ) -> None:
        super().__init__(name=name, is_renderable=is_renderable)
        self._isolated_points: List["Point"] = list(isolated_points or [])
        # Deleting the graph leaves these on the canvas; it owns every other vertex and edge.
        self._preexisting_points: List["Point"] = []
        self._preexisting_edges: List[Drawable] = []
        self._original_edge_labels: List[Tuple[Drawable, str, bool]] = []

    @property
    def directed(self) -> bool:
        """Whether the graph is directed. Subclasses override this."""
        return False

    def get_class_name(self) -> str:
        return "Graph"

    # ------------------------------------------------------------------
    # Abstract interface - subclasses must implement
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def vertices(self) -> Dict[str, str]:
        """Return mapping of vertex id to vertex name."""
        ...

    @property
    @abstractmethod
    def edges(self) -> List[Dict[str, Any]]:
        """Return list of edge descriptor dicts."""
        ...

    @property
    @abstractmethod
    def adjacency_matrix(self) -> List[List[float]]:
        """Return adjacency matrix."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return minimal state for serialization. Subclasses add edge references."""
        args: Dict[str, Any] = {
            # Vertex points tracked outside edges; needed to restore edge-less vertices.
            "isolated_points": [getattr(p, "name", "") for p in self._isolated_points],
        }
        # Written only when the graph reuses existing drawables, so other saves are unchanged.
        if self._preexisting_points:
            args["preexisting_points"] = [getattr(p, "name", "") for p in self._preexisting_points]
        if self._preexisting_edges:
            args["preexisting_edges"] = [getattr(e, "name", "") for e in self._preexisting_edges]
        if self._original_edge_labels:
            args["preexisting_edge_labels"] = [
                {"edge": getattr(edge, "name", ""), "text": text, "visible": visible}
                for edge, text, visible in self._original_edge_labels
            ]
        return {"name": self.name, "args": args}

    # ------------------------------------------------------------------
    # Ownership of reused drawables
    # ------------------------------------------------------------------
    def set_preexisting(
        self,
        points: Iterable["Point"],
        edges: Iterable[Drawable],
        original_edge_labels: Iterable[Tuple[Drawable, str, bool]] = (),
    ) -> None:
        """Record the vertices and edges that existed before this graph was created.

        ``original_edge_labels`` holds the label each reused edge had before the graph
        wrote its weight there.
        """
        self._preexisting_points = list(points)
        self._preexisting_edges = list(edges)
        self._original_edge_labels = list(original_edge_labels)

    @property
    def original_edge_labels(self) -> List[Tuple[Drawable, str, bool]]:
        """Labels to restore on pre-existing edges when the graph is deleted."""
        return list(self._original_edge_labels)

    def adopt(self, drawable: Drawable) -> None:
        """Take ownership of a pre-existing vertex or edge whose creator was deleted."""
        self._forget_preexisting(drawable)

    def is_preexisting(self, drawable: Drawable) -> bool:
        """Return True if the vertex or edge existed before the graph and is not owned by it."""
        tracked = self._preexisting_points + self._preexisting_edges
        return any(item is drawable for item in tracked)

    def _forget_preexisting(self, drawable: Drawable) -> None:
        self._preexisting_points = [p for p in self._preexisting_points if p is not drawable]
        self._preexisting_edges = [e for e in self._preexisting_edges if e is not drawable]
        self._original_edge_labels = [r for r in self._original_edge_labels if r[0] is not drawable]

    def _copy_preexisting_to(self, copied: "Graph", memo: Dict[int, Any]) -> None:
        """Carry the ownership records into a deep copy (used by undo snapshots)."""
        copied._preexisting_points = deepcopy(self._preexisting_points, memo)
        copied._preexisting_edges = deepcopy(self._preexisting_edges, memo)
        copied._original_edge_labels = deepcopy(self._original_edge_labels, memo)

    def remove_point(self, point: "Point") -> bool:
        """Remove an isolated point reference from this graph."""
        self._forget_preexisting(point)
        if point in self._isolated_points:
            self._isolated_points.remove(point)
            return True
        return False

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Graph":
        raise NotImplementedError("Graph is abstract; use a concrete subclass")

    def rotate(self, angle: float) -> Any:
        return False, "Graph rotation is not supported"
