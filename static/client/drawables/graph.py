from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.drawable import Drawable


class Graph(Drawable):
    """Base graph container holding vertex and edge references."""

    def __init__(
        self,
        name: str,
        *,
        directed: bool,
        graph_type: str = "graph",
        vertices: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        points: Optional[List[str]] = None,
        is_renderable: bool = False,
    ) -> None:
        super().__init__(name=name, is_renderable=is_renderable)
        self.directed: bool = directed
        self.graph_type: str = graph_type
        self.vertices: Dict[str, str] = vertices or {}
        self.edges: List[Dict[str, Any]] = edges or []
        self.points: List[str] = list(points or [])

    def get_class_name(self) -> str:
        return "Graph"

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": {
                "directed": self.directed,
                "graph_type": self.graph_type,
                "vertices": self.vertices,
                "edges": self.edges,
                "points": self.points,
            },
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Graph":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = Graph(
            name=self.name,
            directed=self.directed,
            graph_type=self.graph_type,
            vertices=deepcopy(self.vertices, memo),
            edges=deepcopy(self.edges, memo),
            points=deepcopy(self.points, memo),
            is_renderable=self.is_renderable,
        )
        memo[id(self)] = copied
        return copied

    def rotate(self, angle: float) -> Any:
        return False, "Graph rotation is not supported"
