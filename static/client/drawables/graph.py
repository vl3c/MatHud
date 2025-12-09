from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.drawable import Drawable

if TYPE_CHECKING:
    from drawables.point import Point


class Graph(Drawable):
    """Abstract graph interface; subclasses implement edge storage and descriptor computation."""

    def __init__(
        self,
        name: str,
        *,
        directed: bool,
        graph_type: str = "graph",
        isolated_points: Optional[List["Point"]] = None,
        is_renderable: bool = False,
    ) -> None:
        super().__init__(name=name, is_renderable=is_renderable)
        self.directed: bool = directed
        self.graph_type: str = graph_type
        self._isolated_points: List["Point"] = list(isolated_points or [])

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
        return {
            "name": self.name,
            "args": {
                "directed": self.directed,
                "graph_type": self.graph_type,
                "vertices": self.vertices,
                "edges": self.edges,
                "adjacency_matrix": self.adjacency_matrix,
            },
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
