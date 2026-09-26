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
from managers.dependency_removal import (
    find_point_users,
    hand_over_to_graphs,
    is_on_canvas,
    release_segment,
    remove_drawable_with_dependencies,
)
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

        # Vertices and edges may reuse drawables already on the canvas; the graph must not own those.
        existing_drawables: List[Any] = (
            list(self.drawables.Points) + list(self.drawables.Segments) + list(self.drawables.Vectors)
        )
        existing_ids = {id(drawable) for drawable in existing_drawables}

        vertex_positions = self._resolve_positions(state)
        vertex_name_map: Dict[str, str] = {}
        id_to_point: Dict[str, Point] = {}

        for vertex in state.vertices:
            coords = vertex_positions.get(vertex.id, (0.0, 0.0))
            point = self._create_vertex_point(vertex, coords)
            vertex_name_map[vertex.id] = point.name
            id_to_point[vertex.id] = point

        segments_created: List["Segment"] = []
        vectors_created: List["Vector"] = []
        original_labels: List[Tuple[Any, str, bool]] = []
        is_tree = isinstance(state, TreeState)
        for edge in state.edges:
            edge_directed = state.directed if is_tree else None
            segment_obj, vector_obj = self._create_edge(edge, state.directed, id_to_point, force_directed=edge_directed)
            if segment_obj:
                segments_created.append(segment_obj)
            if vector_obj:
                vectors_created.append(vector_obj)
            edge_obj = segment_obj or vector_obj
            if edge_obj is not None and edge.weight is not None:
                self._apply_weight_label(edge_obj, str(edge.weight), id(edge_obj) in existing_ids, original_labels)

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

        graph.set_preexisting(
            self._unique_preexisting(list(id_to_point.values()), existing_ids),
            self._unique_preexisting(segments_created + vectors_created, existing_ids),
            original_labels,
        )

        self.drawables.add(graph)
        self.dependency_manager.analyze_drawable_for_dependencies(graph)
        return graph

    @staticmethod
    def _unique_preexisting(drawables: List[Any], existing_ids: set[int]) -> List[Any]:
        """Return the drawables whose ids were on the canvas before, without duplicates."""
        result: List[Any] = []
        for drawable in drawables:
            if id(drawable) in existing_ids and not any(item is drawable for item in result):
                result.append(drawable)
        return result

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

        resolved_directed = directed if directed is not None else graph_type in ("dag", "directed")

        if adjacency_matrix and not edges:
            # Trees are always undirected, so their matrix is read like an undirected graph's.
            matrix_directed = resolved_directed and graph_type != "tree"
            edge_descriptors.extend(self._edges_from_adjacency_matrix(adjacency_matrix, id_list, matrix_directed))

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

        undo_manager = self.canvas.undo_redo_manager
        undo_manager.archive()
        # One archive covers the whole delete; the edge and point deletes below would otherwise archive again.
        undo_manager.suspend_archiving()
        try:
            removed = self._delete_graph_and_owned_drawables(existing)
        finally:
            undo_manager.resume_archiving()

        if removed and self.canvas.draw_enabled:
            self.canvas.draw()
        return bool(removed)

    def _delete_graph_and_owned_drawables(self, graph: Graph) -> bool:
        """Delete the graph with the vertices and edges it created that nothing else uses.

        Points and edges that existed before the graph (reused as vertices or edges)
        stay, with their original labels back. So do graph-created ones that another
        object (a triangle built on a vertex, another graph, ...) now uses; when only
        other graphs use them, those graphs take ownership.
        """
        is_directed = isinstance(graph, DirectedGraph)
        edges: List[Any] = list(graph.vectors) if is_directed else list(getattr(graph, "segments", []))
        vertex_points: List["Point"] = []
        for point in self._edge_endpoints(edges) + list(getattr(graph, "_isolated_points", [])):
            if not any(item is point for item in vertex_points):
                vertex_points.append(point)

        self._restore_original_labels(graph)
        if not remove_drawable_with_dependencies(self.drawables, self.dependency_manager, graph):
            return False

        for edge in edges:
            if graph.is_preexisting(edge) or not is_on_canvas(self.drawables, edge):
                continue
            if is_directed:
                self._release_vector(edge)
            else:
                release_segment(edge, self.drawables, self.dependency_manager, self.segment_manager)

        for point in vertex_points:
            if graph.is_preexisting(point) or not is_on_canvas(self.drawables, point):
                continue
            users = find_point_users(point, self.drawables, self.dependency_manager)
            if users:
                hand_over_to_graphs(point, users)
            elif getattr(point, "name", ""):
                self.point_manager.delete_point_by_name(point.name)
        return True

    @staticmethod
    def _edge_endpoints(edges: List[Any]) -> List["Point"]:
        points: List["Point"] = []
        for edge in edges:
            if hasattr(edge, "origin"):
                points.extend([edge.origin, edge.tip])
            else:
                points.extend([edge.point1, edge.point2])
        return points

    def _release_vector(self, vector: "Vector") -> None:
        """Delete a graph-created vector unless something else still uses it."""
        users = [d for d in self.dependency_manager.get_all_children(vector) if is_on_canvas(self.drawables, d)]
        if users:
            hand_over_to_graphs(vector, users)
            return
        self.vector_manager.delete_vector(vector.origin.x, vector.origin.y, vector.tip.x, vector.tip.y)

    @staticmethod
    def _label_holder(edge: Any) -> Any:
        """Return the segment carrying the edge's label (a vector's internal segment)."""
        if edge.get_class_name() == "Vector":
            return edge.segment
        return edge

    def _apply_weight_label(
        self,
        edge: Any,
        label_text: str,
        preexisting: bool,
        original_labels: List[Tuple[Any, str, bool]],
    ) -> None:
        """Show the edge weight as a label, remembering a reused edge's own label first.

        New segments already got their label on creation; new vectors get it here.
        """
        is_vector = edge.get_class_name() == "Vector"
        if not preexisting and not is_vector:
            return
        holder = self._label_holder(edge)
        if preexisting and not any(record[0] is edge for record in original_labels):
            label = getattr(holder, "label", None)
            original_labels.append(
                (edge, str(getattr(label, "text", "") or ""), bool(getattr(label, "visible", False)))
            )
        try:
            holder.update_label_text(label_text)
            holder.set_label_visibility(True)
        except Exception:
            pass

    def _restore_original_labels(self, graph: Graph) -> None:
        """Put back the labels that reused edges had before the graph showed its weights."""
        for edge, text, visible in graph.original_edge_labels:
            if not is_on_canvas(self.drawables, edge):
                continue
            holder = self._label_holder(edge)
            try:
                holder.update_label_text(text)
                holder.set_label_visibility(visible)
            except Exception:
                pass

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
    @staticmethod
    def _edges_from_adjacency_matrix(
        adjacency_matrix: List[List[float]], id_list: List[str], directed: bool
    ) -> List[GraphEdgeDescriptor]:
        """Build edge descriptors from a weighted adjacency matrix.

        Directed graphs get one edge per nonzero entry. Undirected graphs get one edge per
        vertex pair, read from the upper triangle (or the lower one where the upper is zero).
        Edges take their direction from the graph.
        """
        descriptors: List[GraphEdgeDescriptor] = []
        n = min(len(adjacency_matrix), len(id_list))
        for i in range(n):
            for j in range(n):
                if directed:
                    weight = GraphManager._matrix_entry(adjacency_matrix, i, j)
                elif j > i:
                    weight = GraphManager._matrix_entry(adjacency_matrix, i, j) or GraphManager._matrix_entry(
                        adjacency_matrix, j, i
                    )
                else:
                    continue
                if weight:
                    descriptors.append(
                        GraphEdgeDescriptor(f"m{i}_{j}", id_list[i], id_list[j], weight=float(weight), directed=None)
                    )
        return descriptors

    @staticmethod
    def _matrix_entry(matrix: List[List[float]], row: int, col: int) -> float:
        if row < len(matrix) and col < len(matrix[row]):
            return matrix[row][col]
        return 0

    def _create_vertex_point(self, vertex: GraphVertexDescriptor, coords: Tuple[float, float]) -> "Point":
        """Create the point for a vertex, honouring its requested name when valid and unused."""
        requested_name = self._resolve_requested_vertex_name(vertex.name)
        existing_point = self.point_manager.get_point(coords[0], coords[1])
        if requested_name:
            name = self.name_generator.generate_point_name("")
        else:
            name = self.name_generator.generate_point_name(vertex.name or "")
        point = self.point_manager.create_point(
            coords[0],
            coords[1],
            name=name,
            color=vertex.color,
            extra_graphics=False,
        )
        if requested_name and point is not existing_point and point.name != requested_name:
            point.update_name(requested_name)
        return point

    def _resolve_requested_vertex_name(self, requested: Optional[str]) -> Optional[str]:
        """Return the filtered requested vertex name if it is non-empty and not used by another point."""
        if not requested:
            return None
        filtered = str(self.name_generator.filter_string(requested)).strip()
        if not filtered or self.point_manager.get_point_by_name(filtered) is not None:
            return None
        return filtered

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
            # create_graph writes the weight label, after saving a reused vector's own label.
            vector_name = edge.name or ""
            vector = self.vector_manager.create_vector_from_points(
                source_point,
                target_point,
                name=vector_name,
                color=color_value,
            )
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
