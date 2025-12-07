from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.graph import Graph


class UndirectedGraph(Graph):
    """Graph where edges are segments (renderable container)."""

    def __init__(
        self,
        name: str,
        *,
        vertices: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        points: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name=name,
            directed=False,
            graph_type="graph",
            vertices=vertices,
            edges=edges,
            points=points,
            is_renderable=True,
        )
        self.segments: List[str] = list(segments or [])

    def get_class_name(self) -> str:
        return "UndirectedGraph"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"]["segments"] = list(self.segments)
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UndirectedGraph":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = UndirectedGraph(
            name=self.name,
            vertices=deepcopy(self.vertices, memo),
            edges=deepcopy(self.edges, memo),
            points=deepcopy(self.points, memo),
            segments=deepcopy(self.segments, memo),
        )
        memo[id(self)] = copied
        return copied

