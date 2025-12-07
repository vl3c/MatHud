from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from drawables.graph import Graph
from drawables.directed_graph import DirectedGraph
from drawables.undirected_graph import UndirectedGraph
from drawables.tree import Tree
from drawables.tree import Tree
from geometry.graph_state import GraphEdgeDescriptor, GraphState, GraphVertexDescriptor, TreeState
from utils.graph_layout import layout_vertices
from utils.graph_utils import Edge

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.point import Point
    from drawables.vector import Vector
    from drawables.segment import Segment
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.label_manager import LabelManager
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from managers.vector_manager import VectorManager
    from name_generator.drawable import DrawableNameGenerator


class GraphManager:
    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        segment_manager: "SegmentManager",
        vector_manager: "VectorManager",
        label_manager: "LabelManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.segment_manager = segment_manager
        self.vector_manager = vector_manager
        self.label_manager = label_manager
        self.drawable_manager = drawable_manager_proxy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_graph(self, state: GraphState) -> Graph:
        self.canvas.undo_redo_manager.archive()

        vertex_positions = self._resolve_positions(state)
        vertex_name_map: Dict[str, str] = {}
        id_to_point: Dict[str, Point] = {}

        for vertex in state.vertices:
            name = self.name_generator.generate_point_name(vertex.name or "")
            coords = vertex_positions.get(vertex.id, (0.0, 0.0))
            point = self.point_manager.create_point(
                coords[0],
                coords[1],
                name=name,
                color=vertex.color,
                extra_graphics=False,
            )
            vertex_name_map[vertex.id] = point.name
            id_to_point[vertex.id] = point

        edge_records: List[Dict[str, Any]] = []
        for edge in state.edges:
            edge_record = self._create_edge(edge, state.directed, id_to_point)
            edge_records.append(edge_record)

        if isinstance(state, TreeState):
            graph = Tree(
                state.name,
                root=getattr(state, "root", None),
                vertices=vertex_name_map,
                edges=edge_records,
                segments=[record.get("segment_name", "") for record in edge_records if record.get("segment_name")],
                points=list(vertex_name_map.values()),
            )
        else:
            if state.directed:
                graph = DirectedGraph(
                    state.name,
                    vertices=vertex_name_map,
                    edges=edge_records,
                    vectors=[record.get("vector_name", "") for record in edge_records if record.get("vector_name")],
                    points=list(vertex_name_map.values()),
                )
            else:
                graph = UndirectedGraph(
                    state.name,
                    vertices=vertex_name_map,
                    edges=edge_records,
                    segments=[record.get("segment_name", "") for record in edge_records if record.get("segment_name")],
                    points=list(vertex_name_map.values()),
                )

        self.drawables.add(graph)
        return graph

    def build_graph_state(
        self,
        *,
        name: str,
        graph_type: str,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        adjacency_matrix: Optional[List[List[float]]],
        directed: Optional[bool],
        root: Optional[str],
        layout: Optional[str],
        placement_box: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]],
    ) -> GraphState:
        vertex_descriptors: List[GraphVertexDescriptor] = []
        for idx, vertex in enumerate(vertices):
            vertex_id = f"v{idx}"
            vertex_descriptors.append(
                GraphVertexDescriptor(
                    vertex_id,
                    name=vertex.get("name"),
                    x=vertex.get("x"),
                    y=vertex.get("y"),
                    color=vertex.get("color"),
                    label=vertex.get("label"),
                )
            )

        if adjacency_matrix and not vertex_descriptors:
            size = len(adjacency_matrix)
            for i in range(size):
                vertex_descriptors.append(GraphVertexDescriptor(f"v{i}", name=None))

        id_list = [vd.id for vd in vertex_descriptors]

        edge_descriptors: List[GraphEdgeDescriptor] = []
        for idx, edge in enumerate(edges):
            src_idx = int(edge.get("source", 0))
            tgt_idx = int(edge.get("target", 0))
            src_id = id_list[src_idx] if src_idx < len(id_list) else f"v{src_idx}"
            tgt_id = id_list[tgt_idx] if tgt_idx < len(id_list) else f"v{tgt_idx}"
            edge_descriptors.append(
                GraphEdgeDescriptor(
                    f"e{idx}",
                    src_id,
                    tgt_id,
                    weight=edge.get("weight"),
                    name=edge.get("name"),
                    color=edge.get("color"),
                    label=edge.get("label"),
                    directed=edge.get("directed"),
                )
            )

        if adjacency_matrix and not edges:
            n = len(adjacency_matrix)
            for i in range(n):
                for j in range(n):
                    weight = adjacency_matrix[i][j]
                    if weight:
                        src = id_list[i] if i < len(id_list) else f"v{i}"
                        tgt = id_list[j] if j < len(id_list) else f"v{j}"
                        edge_descriptors.append(
                            GraphEdgeDescriptor(
                                f"m{i}_{j}",
                                src,
                                tgt,
                                weight=float(weight),
                                directed=True,
                            )
                        )

        resolved_directed = directed if directed is not None else graph_type in ("dag", "directed")

        if graph_type == "tree":
            return TreeState(
                name=name,
                vertices=vertex_descriptors,
                edges=edge_descriptors,
                root=root,
                layout=layout,
                placement_box=placement_box,
                metadata=metadata,
            )

        return GraphState(
            name=name,
            vertices=vertex_descriptors,
            edges=edge_descriptors,
            directed=resolved_directed,
            graph_type=graph_type,
            layout=layout,
            placement_box=placement_box,
            metadata=metadata,
        )

    def delete_graph(self, name: str) -> bool:
        existing = self.get_graph(name)
        if existing is None:
            return False

        for edge in existing.edges:
            label_name = edge.get("label_name")
            if label_name:
                self.label_manager.delete_label(label_name)

            vector_name = edge.get("vector_name")
            if vector_name:
                vector = self.vector_manager.get_vector_by_name(vector_name)
                if vector:
                    self.vector_manager.delete_vector(
                        vector.origin.x, vector.origin.y, vector.tip.x, vector.tip.y
                    )

            segment_name = edge.get("segment_name")
            if segment_name:
                self.segment_manager.delete_segment_by_name(segment_name)

        vertex_names = list(existing.vertices.values())
        for v_name in vertex_names:
            self.point_manager.delete_point_by_name(v_name)

        removed = self.drawables.remove(existing)
        if removed and self.canvas.draw_enabled:
            self.canvas.draw()
        return bool(removed)

    def get_graph(self, name: str) -> Optional[Graph]:
        for graph in self.drawables.get_by_class_name("Graph"):
            if graph.name == name:
                return graph  # type: ignore[return-value]
        for graph in self.drawables.get_by_class_name("DirectedGraph"):
            if graph.name == name:
                return graph  # type: ignore[return-value]
        for graph in self.drawables.get_by_class_name("UndirectedGraph"):
            if graph.name == name:
                return graph  # type: ignore[return-value]
        for tree in self.drawables.get_by_class_name("Tree"):
            if tree.name == name:
                return tree  # type: ignore[return-value]
        return None

    def capture_state(self, name: str) -> Optional[GraphState]:
        graph = self.get_graph(name)
        if graph is None:
            return None

        inverse_vertices = {point_name: vertex_id for vertex_id, point_name in graph.vertices.items()}
        vertex_descriptors: List[GraphVertexDescriptor] = []
        for vertex_id, point_name in graph.vertices.items():
            point = self.point_manager.get_point_by_name(point_name)
            x = point.x if point else 0.0
            y = point.y if point else 0.0
            vertex_descriptors.append(
                GraphVertexDescriptor(vertex_id, name=point_name, x=x, y=y)
            )

        edge_descriptors: List[GraphEdgeDescriptor] = []
        for edge_record in graph.edges:
            source_name = edge_record.get("source")
            target_name = edge_record.get("target")
            source_id = inverse_vertices.get(source_name, source_name)
            target_id = inverse_vertices.get(target_name, target_name)
            edge_descriptors.append(
                GraphEdgeDescriptor(
                    edge_record.get("id", ""),
                    source_id,
                    target_id,
                    weight=edge_record.get("weight"),
                    name=edge_record.get("vector_name") or edge_record.get("segment_name"),
                    color=None,
                    label=edge_record.get("label_name"),
                    directed=graph.directed,
                    vector_name=edge_record.get("vector_name"),
                    segment_name=edge_record.get("segment_name"),
                    label_name=edge_record.get("label_name"),
                )
            )

        if isinstance(graph, Tree):
            return TreeState(
                graph.name,
                vertex_descriptors,
                edge_descriptors,
                root=getattr(graph, "root", None),
                metadata=getattr(graph, "metadata", {}),
            )

        return GraphState(
            graph.name,
            vertex_descriptors,
            edge_descriptors,
            directed=graph.directed,
            graph_type=graph.graph_type,
            metadata=getattr(graph, "metadata", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_positions(self, state: GraphState) -> Dict[str, Tuple[float, float]]:
        provided: Dict[str, Tuple[float, float]] = {}
        missing: List[str] = []
        for vertex in state.vertices:
            if vertex.x is not None and vertex.y is not None:
                provided[vertex.id] = (float(vertex.x), float(vertex.y))
            else:
                missing.append(vertex.id)

        if not missing:
            return provided

        edges_for_layout = [Edge(e.source, e.target) for e in state.edges]
        layout_positions = layout_vertices(
            [v.id for v in state.vertices],
            edges_for_layout,
            layout=state.layout,
            placement_box=state.placement_box,
            canvas_width=float(getattr(self.canvas, "width", 1000.0)),
            canvas_height=float(getattr(self.canvas, "height", 800.0)),
            root_id=getattr(state, "root", None),
        )

        for vid in missing:
            if vid in layout_positions:
                provided[vid] = layout_positions[vid]
        for vid, coords in provided.items():
            if vid not in layout_positions:
                layout_positions[vid] = coords
        return layout_positions

    def _create_edge(
        self,
        edge: GraphEdgeDescriptor,
        default_directed: bool,
        id_to_point: Dict[str, "Point"],
    ) -> Dict[str, Any]:
        source_point = id_to_point[edge.source]
        target_point = id_to_point[edge.target]
        directed = edge.directed if edge.directed is not None else default_directed
        color_value = edge.color

        if directed:
            vector_name = edge.name or ""
            vector = self.vector_manager.create_vector(
                source_point.x,
                source_point.y,
                target_point.x,
                target_point.y,
                name=vector_name,
                color=color_value,
                extra_graphics=False,
            )
            drawable_name = vector.name
            record: Dict[str, Any] = {
                "id": edge.id,
                "source": source_point.name,
                "target": target_point.name,
                "vector_name": drawable_name,
            }
        else:
            segment_name = edge.name or ""
            segment = self.segment_manager.create_segment(
                source_point.x,
                source_point.y,
                target_point.x,
                target_point.y,
                name=segment_name,
                color=color_value,
                extra_graphics=False,
            )
            drawable_name = segment.name
            record = {
                "id": edge.id,
                "source": source_point.name,
                "target": target_point.name,
                "segment_name": drawable_name,
            }

        if edge.weight is not None:
            mid_x = (source_point.x + target_point.x) / 2
            mid_y = (source_point.y + target_point.y) / 2
            label_text = edge.label if edge.label is not None else str(edge.weight)
            label = self.label_manager.create_label(
                mid_x,
                mid_y,
                str(label_text),
                color=color_value,
            )
            record["weight"] = edge.weight
            record["label_name"] = label.name

        return record

