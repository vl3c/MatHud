from __future__ import annotations

from typing import Any, Dict, List, Optional


class GraphVertexDescriptor:
    def __init__(
        self,
        vertex_id: str,
        name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        self.id = vertex_id
        self.name = name
        self.x = x
        self.y = y
        self.color = color
        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "label": self.label,
        }


class GraphEdgeDescriptor:
    def __init__(
        self,
        edge_id: str,
        source: str,
        target: str,
        *,
        weight: Optional[float] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        directed: Optional[bool] = None,
        vector_name: Optional[str] = None,
        segment_name: Optional[str] = None,
        label_name: Optional[str] = None,
    ) -> None:
        self.id = edge_id
        self.source = source
        self.target = target
        self.weight = weight
        self.name = name
        self.color = color
        self.label = label
        self.directed = directed
        self.vector_name = vector_name
        self.segment_name = segment_name
        self.label_name = label_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "name": self.name,
            "color": self.color,
            "label": self.label,
            "directed": self.directed,
            "vector_name": self.vector_name,
            "segment_name": self.segment_name,
            "label_name": self.label_name,
        }


class GraphState:
    def __init__(
        self,
        name: str,
        vertices: List[GraphVertexDescriptor],
        edges: List[GraphEdgeDescriptor],
        *,
        directed: bool = False,
        graph_type: str = "graph",
        layout: Optional[str] = None,
        placement_box: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.vertices = vertices
        self.edges = edges
        self.directed = directed
        self.graph_type = graph_type
        self.layout = layout
        self.placement_box = placement_box or {}
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vertices": [v.to_dict() for v in self.vertices],
            "edges": [e.to_dict() for e in self.edges],
            "directed": self.directed,
            "graph_type": self.graph_type,
            "layout": self.layout,
            "placement_box": self.placement_box,
            "metadata": self.metadata,
        }


class TreeState(GraphState):
    def __init__(
        self,
        name: str,
        vertices: List[GraphVertexDescriptor],
        edges: List[GraphEdgeDescriptor],
        *,
        root: Optional[str],
        layout: Optional[str] = None,
        placement_box: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            vertices=vertices,
            edges=edges,
            directed=False,
            graph_type="tree",
            layout=layout,
            placement_box=placement_box,
            metadata=metadata,
        )
        self.root = root

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["root"] = self.root
        return data



