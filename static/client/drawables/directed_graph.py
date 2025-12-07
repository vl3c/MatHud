from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.graph import Graph


class DirectedGraph(Graph):
    """Graph where edges are vectors (renderable container)."""

    def __init__(
        self,
        name: str,
        *,
        vertices: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        points: Optional[List[str]] = None,
        vectors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name=name,
            directed=True,
            graph_type="graph",
            vertices=vertices,
            edges=edges,
            points=points,
            is_renderable=True,
        )
        self.vectors: List[str] = list(vectors or [])

    def get_class_name(self) -> str:
        return "DirectedGraph"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"]["vectors"] = list(self.vectors)
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DirectedGraph":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = DirectedGraph(
            name=self.name,
            vertices=deepcopy(self.vertices, memo),
            edges=deepcopy(self.edges, memo),
            points=deepcopy(self.points, memo),
            vectors=deepcopy(self.vectors, memo),
        )
        memo[id(self)] = copied
        return copied

