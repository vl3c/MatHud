from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.undirected_graph import UndirectedGraph


class Tree(UndirectedGraph):
    def __init__(
        self,
        name: str,
        *,
        root: Optional[str],
        vertices: Optional[Dict[str, str]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        points: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name, vertices=vertices, edges=edges, points=points, segments=segments)
        self.graph_type = "tree"
        self.root: Optional[str] = root

    def get_class_name(self) -> str:
        return "Tree"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"]["root"] = self.root
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Tree":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = Tree(
            name=self.name,
            root=self.root,
            vertices=deepcopy(self.vertices, memo),
            edges=deepcopy(self.edges, memo),
            points=deepcopy(self.points, memo),
            segments=deepcopy(self.segments, memo),
        )
        memo[id(self)] = copied
        return copied

