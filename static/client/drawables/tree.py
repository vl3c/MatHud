"""Tree drawable as an undirected graph with a designated root.

This module provides the Tree class for representing tree data structures
as an undirected graph with a root vertex for hierarchical operations.

Key Features:
    - Extends UndirectedGraph with root designation
    - Root vertex identifier for tree traversal algorithms
    - Segment-based edge representation inherited from UndirectedGraph
    - Serialization includes root identifier
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from drawables.undirected_graph import UndirectedGraph

if TYPE_CHECKING:
    from drawables.point import Point
    from drawables.segment import Segment


class Tree(UndirectedGraph):
    """Tree is an undirected graph with a designated root vertex.

    Extends UndirectedGraph to add a root vertex identifier used for
    hierarchical tree traversals and visualizations.

    Attributes:
        root: Optional vertex identifier string designating the tree root.
    """

    def __init__(
        self,
        name: str,
        *,
        root: Optional[str],
        isolated_points: Optional[List["Point"]] = None,
        segments: Optional[List["Segment"]] = None,
    ) -> None:
        super().__init__(name=name, isolated_points=isolated_points, segments=segments)
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
            segments=deepcopy(self._segments, memo),
            isolated_points=deepcopy(self._isolated_points, memo),
        )
        memo[id(self)] = copied
        return copied
