"""Abstract graph drawable for representing graph data structures.

This module provides the Graph abstract base class that defines the interface
for graph drawables. Concrete implementations include DirectedGraph and
UndirectedGraph.

Key Features:
    - Abstract interface for vertex and edge access
    - Adjacency matrix generation
    - Isolated point management for vertices without edges
    - Serialization support for workspace persistence
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.drawable import Drawable

if TYPE_CHECKING:
    from drawables.point import Point


class Graph(Drawable):
    """Abstract graph interface for directed and undirected graphs.

    Subclasses implement edge storage and provide vertex/edge descriptors.

    Attributes:
        _isolated_points: List of vertex points not connected by edges.
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
        return {
            "name": self.name,
            "args": {},
        }

    def remove_point(self, point: "Point") -> bool:
        """Remove an isolated point reference from this graph."""
        if point in self._isolated_points:
            self._isolated_points.remove(point)
            return True
        return False

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Graph":
        raise NotImplementedError("Graph is abstract; use a concrete subclass")

    def rotate(self, angle: float) -> Any:
        return False, "Graph rotation is not supported"
