"""Graph manager for creating and managing graph drawables.

This module provides the GraphManager class which handles creation,
deletion, and state management for directed and undirected graphs.

Key Features:
    - Graph creation from vertex/edge descriptors with automatic layout
    - Support for DirectedGraph, UndirectedGraph, and Tree drawables
    - Automatic point and edge (segment/vector) creation
    - Graph state capture for workspace persistence
    - Layout position resolution with visible bounds fallback
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from drawables.graph import Graph
from drawables.directed_graph import DirectedGraph
from drawables.undirected_graph import UndirectedGraph
from drawables.tree import Tree
from geometry.graph_state import GraphEdgeDescriptor, GraphState, GraphVertexDescriptor, TreeState
from managers.dependency_removal import remove_drawable_with_dependencies
from utils.graph_layout import layout_vertices
from utils.graph_utils import Edge, GraphUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.point import Point
    from drawables.vector import Vector
    from drawables.segment import Segment
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
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
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.segment_manager = segment_manager
        self.vector_manager = vector_manager
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

        segments_created: List["Segment"] = []
        vectors_created: List["Vector"] = []
        is_tree = isinstance(state, TreeState)
        for edge in state.edges:
            edge_directed = state.directed if is_tree else None
            segment_obj, vector_obj = self._create_edge(edge, state.directed, id_to_point, force_directed=edge_directed)
            if segment_obj:
                segments_created.append(segment_obj)
            if vector_obj:
                vectors_created.append(vector_obj)

        if is_tree:
            # Translate internal vertex ID to actual point name for root
            internal_root = getattr(state, "root", None)
            root_point_name = vertex_name_map.get(internal_root) if internal_root else None
            graph = Tree(
                state.name,
                root=root_point_name,
                segments=segments_created,
                isolated_points=list(id_to_point.values()),
            )
        else:
            if state.directed:
                graph = DirectedGraph(
                    state.name,
                    vectors=vectors_created,
                    isolated_points=list(id_to_point.values()),
                )
            else:
                graph = UndirectedGraph(
                    state.name,
                    segments=segments_created,
                    isolated_points=list(id_to_point.values()),
                )

        self.drawables.add(graph)
        self.dependency_manager.analyze_drawable_for_dependencies(graph)
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
            if src_idx < 0 or src_idx >= len(id_list) or tgt_idx < 0 or tgt_idx >= len(id_list):
                continue
            src_id = id_list[src_idx]
            tgt_id = id_list[tgt_idx]
            edge_descriptors.append(
                GraphEdgeDescriptor(
                    f"e{idx}",
                    src_id,
                    tgt_id,
                    weight=edge.get("weight"),
                    name=edge.get("name"),
                    color=edge.get("color"),
                    directed=edge.get("directed"),
                )
            )

        if adjacency_matrix and not edges:
            n = len(adjacency_matrix)
            for i in range(n):
                for j in range(n):
                    weight = adjacency_matrix[i][j]
                    if weight and i < len(id_list) and j < len(id_list):
                        edge_descriptors.append(
                            GraphEdgeDescriptor(
                                f"m{i}_{j}",
                                id_list[i],
                                id_list[j],
                                weight=float(weight),
                                directed=True,
                            )
                        )

        resolved_directed = directed if directed is not None else graph_type in ("dag", "directed")

        resolved_root: Optional[str] = None
        if root is not None:
            try:
                root_idx = int(root)
                if 0 <= root_idx < len(id_list):
                    resolved_root = id_list[root_idx]
            except (ValueError, TypeError):
                pass
            if resolved_root is None:
                for vd in vertex_descriptors:
                    if vd.name == root or vd.id == root:
                        resolved_root = vd.id
                        break
            if resolved_root is None and id_list:
                resolved_root = id_list[0]

        if graph_type == "tree":
            # Default to hierarchical tree layout for trees unless explicitly overridden
            # "radial" is only used if explicitly requested AND user clearly wants radial
            tree_layout = layout if layout in ("radial", "hierarchical", "tree") else "tree"
            return TreeState(
                name=name,
                vertices=vertex_descriptors,
                edges=edge_descriptors,
                root=resolved_root,
                layout=tree_layout,
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

        point_names: set[str] = set()

        if isinstance(existing, DirectedGraph):
            vectors: List["Vector"] = list(existing.vectors)
            for vector in vectors:
                self.vector_manager.delete_vector(vector.origin.x, vector.origin.y, vector.tip.x, vector.tip.y)
                point_names.add(vector.origin.name)
                point_names.add(vector.tip.name)
        else:
            segments: List["Segment"] = list(getattr(existing, "segments", []))
            for segment in segments:
                self.segment_manager.delete_segment(
                    segment.point1.x,
                    segment.point1.y,
                    segment.point2.x,
                    segment.point2.y,
                    delete_children=True,
                    delete_parents=False,
                )
                point_names.add(segment.point1.name)
                point_names.add(segment.point2.name)

        # Also remove isolated points tracked on the graph
        isolated_pts = getattr(existing, "_isolated_points", [])
        for p in isolated_pts:
            point_names.add(getattr(p, "name", ""))

        for v_name in point_names:
            self.point_manager.delete_point_by_name(v_name)

        removed = remove_drawable_with_dependencies(self.drawables, self.dependency_manager, existing)
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

        segments: List["Segment"] = list(getattr(graph, "segments", [])) if not isinstance(graph, DirectedGraph) else []
        vectors: List["Vector"] = list(getattr(graph, "vectors", [])) if isinstance(graph, DirectedGraph) else []
        isolated_points: List["Point"] = list(getattr(graph, "_isolated_points", []))

        vertex_descriptors, edge_descriptors = GraphUtils.drawables_to_descriptors(
            segments, vectors, isolated_points=isolated_points
        )
        # Deterministic ordering for downstream consumers and adjacency matrix
        vertex_descriptors = sorted(vertex_descriptors, key=lambda v: v.id)
        adjacency_matrix = GraphUtils.adjacency_matrix_from_descriptors(
            vertex_descriptors, edge_descriptors, directed=graph.directed
        )

        if isinstance(graph, Tree):
            # Translate root to vertex descriptor ID if needed
            stored_root = getattr(graph, "root", None)
            resolved_root = stored_root
            if stored_root is not None:
                # Check if stored root matches a vertex descriptor ID (point name)
                vertex_ids = {vd.id for vd in vertex_descriptors}
                if stored_root not in vertex_ids:
                    # Old format: root is internal ID like "v0"
                    # Try to extract index and map to isolated points
                    if stored_root.startswith("v") and stored_root[1:].isdigit():
                        idx = int(stored_root[1:])
                        isolated_pts = getattr(graph, "_isolated_points", [])
                        if 0 <= idx < len(isolated_pts):
                            resolved_root = isolated_pts[idx].name
            return TreeState(
                graph.name,
                vertex_descriptors,
                edge_descriptors,
                root=resolved_root,
                metadata=getattr(graph, "metadata", {}),
                adjacency_matrix=adjacency_matrix,
            )

        return GraphState(
            graph.name,
            vertex_descriptors,
            edge_descriptors,
            directed=graph.directed,
            graph_type=graph.get_class_name().lower(),
            metadata=getattr(graph, "metadata", {}),
            adjacency_matrix=adjacency_matrix,
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

        # Use current visible bounds as placement box if not provided
        # This ensures the graph is placed in the visible viewport
        placement_box = state.placement_box
        if not placement_box:  # Catches None, {}, and empty values
            try:
                visible_bounds = self.canvas.coordinate_mapper.get_visible_bounds()
                placement_box = {
                    "x": visible_bounds.get("left", 0.0),
                    "y": visible_bounds.get("bottom", 0.0),
                    "width": visible_bounds.get("right", 1000.0) - visible_bounds.get("left", 0.0),
                    "height": visible_bounds.get("top", 800.0) - visible_bounds.get("bottom", 0.0),
                }
            except (AttributeError, KeyError):
                placement_box = None

        edges_for_layout = [Edge(e.source, e.target) for e in state.edges]
        layout_positions = layout_vertices(
            [v.id for v in state.vertices],
            edges_for_layout,
            layout=state.layout,
            placement_box=placement_box,
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
        force_directed: Optional[bool] = None,
    ) -> Tuple[Optional["Segment"], Optional["Vector"]]:
        source_point = id_to_point[edge.source]
        target_point = id_to_point[edge.target]
        if force_directed is not None:
            directed = force_directed
        else:
            directed = edge.directed if edge.directed is not None else default_directed
        color_value = edge.color

        label_text: Optional[str] = None
        if edge.weight is not None:
            label_text = str(edge.weight)

        if directed:
            vector_name = edge.name or ""
            vector = self.vector_manager.create_vector_from_points(
                source_point,
                target_point,
                name=vector_name,
                color=color_value,
            )
            if label_text:
                try:
                    vector.segment.update_label_text(label_text)
                    vector.segment.set_label_visibility(True)
                except Exception:
                    pass
            segment: Optional["Segment"] = None
        else:
            segment = self.segment_manager.create_segment_from_points(
                source_point,
                target_point,
                name=edge.name or "",
                color=color_value,
                label_text=label_text if label_text is not None else "",
                label_visible=label_text is not None,
            )
            vector = None

        return segment, vector
