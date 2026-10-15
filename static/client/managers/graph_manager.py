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
    GRAPH_CLASS_NAMES,
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


# See drawables.graph.EdgeLabelRecord.
LabelRecord = Tuple[Any, str, bool, Optional[str], Optional[int], bool]

# Two vertex positions closer than this on both axes are the same point.
_SAME_POSITION_TOLERANCE = 1e-9


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
        original_labels: List[LabelRecord] = []
        label_seq = self._next_label_seq()
        is_tree = isinstance(state, TreeState)
        # Parallel edges share one segment or vector, which carries their cheapest weight.
        weights_written: Dict[int, Any] = {}
        for edge in state.edges:
            # Every edge takes the graph's direction, so the graph keeps all of them.
            segment_obj, vector_obj = self._create_edge(edge, bool(state.directed), id_to_point)
            if segment_obj:
                segments_created.append(segment_obj)
            if vector_obj:
                vectors_created.append(vector_obj)
            edge_obj = segment_obj or vector_obj
            if edge_obj is None:
                continue
            seen = id(edge_obj) in weights_written
            weight = self._cheaper_weight(weights_written[id(edge_obj)], edge.weight) if seen else edge.weight
            if weight is None or (seen and weight == weights_written[id(edge_obj)]):
                weights_written.setdefault(id(edge_obj), weight)
                continue
            weights_written[id(edge_obj)] = weight
            preexisting = id(edge_obj) in existing_ids
            self._apply_weight_label(edge_obj, str(weight), preexisting, original_labels, label_seq, rewrite=seen)

        graph_name = self.drawables.unique_name(state.name)
        if is_tree:
            # Translate internal vertex ID to actual point name for root
            internal_root = getattr(state, "root", None)
            root_point_name = vertex_name_map.get(internal_root) if internal_root else None
            graph = Tree(
                graph_name,
                root=root_point_name,
                segments=segments_created,
                isolated_points=list(id_to_point.values()),
            )
        else:
            if state.directed:
                graph = DirectedGraph(
                    graph_name,
                    vectors=vectors_created,
                    isolated_points=list(id_to_point.values()),
                )
            else:
                graph = UndirectedGraph(
                    graph_name,
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
    def _cheaper_weight(current: Any, new: Any) -> Any:
        """Weight a shared parallel edge shows: the cheaper one, an unweighted edge counting as 1."""
        if current is None and new is None:
            return None
        current_value = 1 if current is None else current
        new_value = 1 if new is None else new
        return new_value if float(new_value) < float(current_value) else current_value

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
        """Validate a generate_graph request and turn it into a graph state.

        Raises:
            ValueError: for edge indices that name no vertex, an unknown root, two vertices
                given the same coordinates, a tree whose edges do not form a tree, or a dag
                with a cycle.
        """
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

        self._check_distinct_coordinates(vertex_descriptors)
        id_list = [vd.id for vd in vertex_descriptors]

        edge_descriptors: List[GraphEdgeDescriptor] = []
        for idx, edge in enumerate(edges):
            src_idx = self._edge_vertex_index(edge.get("source"), "source", idx, len(id_list))
            tgt_idx = self._edge_vertex_index(edge.get("target"), "target", idx, len(id_list))
            edge_descriptors.append(
                GraphEdgeDescriptor(
                    f"e{idx}",
                    id_list[src_idx],
                    id_list[tgt_idx],
                    weight=edge.get("weight"),
                    name=edge.get("name"),
                    color=edge.get("color"),
                    directed=edge.get("directed"),
                )
            )

        # A dag is always directed and a tree never is; edges take the graph's direction.
        if graph_type == "dag":
            resolved_directed = True
        elif graph_type == "tree":
            resolved_directed = False
        else:
            resolved_directed = bool(directed) if directed is not None else graph_type == "directed"

        if adjacency_matrix and not edges:
            edge_descriptors.extend(self._edges_from_adjacency_matrix(adjacency_matrix, id_list, resolved_directed))

        if graph_type == "tree":
            resolved_root = self._resolve_requested_root(root, vertex_descriptors)
            self._check_tree(name, id_list, edge_descriptors)
            if resolved_root is None:
                resolved_root = self._inferred_tree_root(id_list, edge_descriptors)
            return TreeState(
                name=name,
                vertices=vertex_descriptors,
                edges=edge_descriptors,
                root=resolved_root,
                layout=layout or "tree",
                placement_box=placement_box,
                metadata=metadata,
            )

        if graph_type == "dag":
            self._check_acyclic(name, id_list, edge_descriptors, vertex_descriptors)

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

    @staticmethod
    def _vertex_label(vertex: GraphVertexDescriptor, index: int) -> str:
        return f"{vertex.name!r} (index {index})" if vertex.name else f"index {index}"

    @staticmethod
    def _check_distinct_coordinates(vertices: List[GraphVertexDescriptor]) -> None:
        """Reject two vertices given the same explicit position: they would become one point."""
        placed: List[Tuple[int, float, float]] = []
        for index, vertex in enumerate(vertices):
            if vertex.x is None or vertex.y is None:
                continue
            x, y = float(vertex.x), float(vertex.y)
            for other_index, ox, oy in placed:
                if abs(ox - x) <= _SAME_POSITION_TOLERANCE and abs(oy - y) <= _SAME_POSITION_TOLERANCE:
                    first = GraphManager._vertex_label(vertices[other_index], other_index)
                    second = GraphManager._vertex_label(vertex, index)
                    raise ValueError(
                        f"Vertices {first} and {second} have the same coordinates ({x:g}, {y:g}); "
                        "give each vertex its own position."
                    )
            placed.append((index, x, y))

    @staticmethod
    def _edge_vertex_index(value: Any, field: str, edge_index: int, vertex_count: int) -> int:
        """Return an edge endpoint as a vertex index, or raise if it is not one."""
        index: Optional[int] = None
        if isinstance(value, bool) or value is None:
            index = None
        elif isinstance(value, int):
            index = value
        elif isinstance(value, float) and value.is_integer():
            index = int(value)
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            index = int(value.strip())
        if index is None:
            raise ValueError(f"Edge {edge_index}: {field} {value!r} is not a vertex index (an integer).")
        if not 0 <= index < vertex_count:
            valid = f"0-{vertex_count - 1}" if vertex_count else "none: the graph has no vertices"
            raise ValueError(f"Edge {edge_index}: {field} {index} is out of range (vertex indices: {valid}).")
        return index

    @staticmethod
    def _resolve_requested_root(root: Optional[str], vertices: List[GraphVertexDescriptor]) -> Optional[str]:
        """Vertex id of the requested root, matched by vertex name first, then by index."""
        if root is None:
            return None
        text = str(root).strip()
        for vd in vertices:
            if vd.name == text or vd.id == text:
                return vd.id
        if text.isdigit() and int(text) < len(vertices):
            return vertices[int(text)].id
        names = [vd.name or vd.id for vd in vertices]
        raise ValueError(
            f"Unknown root {root!r}: give a vertex name or index (vertices: {', '.join(names) or 'none'})."
        )

    @staticmethod
    def _inferred_tree_root(id_list: List[str], edges: List[GraphEdgeDescriptor]) -> Optional[str]:
        """Root of a tree given without one: the first vertex no edge points to (as the layout picks it)."""
        targets = {edge.target for edge in edges}
        for vertex_id in id_list:
            if vertex_id not in targets:
                return vertex_id
        return id_list[0] if id_list else None

    @staticmethod
    def _check_tree(name: str, id_list: List[str], edges: List[GraphEdgeDescriptor]) -> None:
        if not id_list:
            raise ValueError(f"Tree '{name}' needs at least one vertex.")
        graph_edges = [Edge(edge.source, edge.target) for edge in edges]
        if GraphUtils.is_tree_edges(id_list, graph_edges):
            return
        expected = len(id_list) - 1
        raise ValueError(
            f"Tree '{name}' is not a tree: {len(id_list)} vertices need exactly {expected} edges forming one "
            f"connected piece with no cycle (got {len(edges)} edges). Use graph_type 'graph' for other graphs."
        )

    @staticmethod
    def _check_acyclic(
        name: str,
        id_list: List[str],
        edges: List[GraphEdgeDescriptor],
        vertices: List[GraphVertexDescriptor],
    ) -> None:
        adjacency: Dict[str, set[str]] = {vertex_id: set() for vertex_id in id_list}
        for edge in edges:
            adjacency[edge.source].add(edge.target)
        cycle = GraphUtils.find_directed_cycle(adjacency)
        if cycle is None:
            return
        labels = {vd.id: vd.name or vd.id for vd in vertices}
        shown = " -> ".join(labels.get(vertex_id, vertex_id) for vertex_id in cycle)
        raise ValueError(f"DAG '{name}' has a cycle ({shown}); a dag must be acyclic. Use graph_type 'graph'.")

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

        self._release_weight_labels(graph)
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
        original_labels: List[LabelRecord],
        label_seq: int,
        rewrite: bool = False,
    ) -> None:
        """Show the edge weight as a label and record what it replaced.

        New segments already got their label on creation; new vectors, reused edges and
        edges a cheaper parallel edge relabels (``rewrite``) get it here. A created edge
        records an empty original and is never restored.
        """
        holder = self._label_holder(edge)
        label = getattr(holder, "label", None)
        original = (str(getattr(label, "text", "") or ""), bool(getattr(label, "visible", False)))
        if preexisting or rewrite or edge.get_class_name() == "Vector":
            try:
                holder.update_label_text(label_text)
                holder.set_label_visibility(True)
            except Exception:
                return
        if not preexisting:
            original = ("", False)
        # Record the text as the label stores it, so a later comparison matches exactly.
        written = str(getattr(getattr(holder, "label", None), "text", "") or "")
        for index, record in enumerate(original_labels):
            if record[0] is edge:
                original_labels[index] = (edge, record[1], record[2], written, record[4], record[5])
                return
        original_labels.append((edge, original[0], original[1], written, label_seq, not preexisting))

    def _next_label_seq(self) -> int:
        """Return a creation sequence number above every graph's on the canvas."""
        seqs = [g.label_seq for g in self._all_graphs() if g.label_seq is not None]
        return (max(seqs) if seqs else 0) + 1

    def _release_weight_labels(self, graph: Graph) -> None:
        """Undo this graph's weight labels without touching labels written after them.

        Graphs that write a weight on the same edge form a chain ordered by creation;
        each records the label it replaced. When a graph leaves, the next graph created
        after it whose recorded original is this graph's weight inherits this graph's
        original. Otherwise, on an edge the graph reused, the original comes back if the
        edge still shows this graph's weight; a label changed since stays. On an edge
        the graph created, the weight stays for whatever keeps the edge.
        """
        others = [g for g in self._all_graphs() if g is not graph]
        for edge, text, visible, written, seq, created in graph.original_edge_labels:
            if written is None or not is_on_canvas(self.drawables, edge):
                continue
            successor = self._next_in_label_chain(edge, others, written, seq)
            if successor is not None:
                successor.replace_label_original(edge, text, visible)
            elif not created:
                self._restore_label_if_unchanged(edge, text, visible, written)

    @staticmethod
    def _next_in_label_chain(edge: Any, graphs: List[Graph], written: str, seq: Optional[int]) -> Optional[Graph]:
        """Return the graph that relabelled the edge right after the one that wrote ``written``.

        Only graphs created later count. Records from older saves have no sequence;
        they match on the label text alone.
        """
        best: Optional[Graph] = None
        best_seq: Optional[int] = None
        for other in graphs:
            record = other.label_record(edge)
            if record is None or record[1] != written:
                continue
            other_seq = record[4]
            if seq is not None and other_seq is not None:
                if other_seq <= seq or (best_seq is not None and other_seq >= best_seq):
                    continue
                best, best_seq = other, other_seq
            elif best is None:
                best = other
        return best

    def _restore_label_if_unchanged(self, edge: Any, text: str, visible: bool, written: str) -> None:
        holder = self._label_holder(edge)
        current = str(getattr(getattr(holder, "label", None), "text", "") or "")
        if current != written:
            return
        try:
            holder.update_label_text(text)
            holder.set_label_visibility(visible)
        except Exception:
            pass

    def _all_graphs(self) -> List[Graph]:
        graphs: List[Graph] = []
        for class_name in GRAPH_CLASS_NAMES:
            graphs.extend(self.drawables.get_by_class_name(class_name))  # type: ignore[arg-type]
        return graphs

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
        A nonzero diagonal entry is a self-loop on a directed graph and is ignored on an
        undirected one (a segment from a point to itself has no length). 0 means no edge,
        so a weight-0 edge cannot be given in a matrix. Edges take their direction from the graph.
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
        """Return the filtered requested vertex name if it is non-empty and no point or other drawable has it."""
        if not requested:
            return None
        filtered = str(self.name_generator.filter_string(requested)).strip()
        if not filtered or self.point_manager.get_point_by_name(filtered) is not None:
            return None
        if filtered in self.drawables.names_in_use():
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

        generated = {vid: layout_positions[vid] for vid in missing if vid in layout_positions}
        occupied = list(provided.values()) + [(float(p.x), float(p.y)) for p in self.drawables.Points]
        provided.update(self._clear_of_occupied(generated, occupied, placement_box))
        return provided

    @staticmethod
    def _clear_of_occupied(
        generated: Dict[str, Tuple[float, float]],
        occupied: List[Tuple[float, float]],
        placement_box: Optional[Dict[str, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Move layout positions off points that already exist, so they get their own points.

        A point at a vertex position would otherwise be reused as the vertex, merging a second
        graph laid out in the same place into the first. Only positions given explicitly reuse
        existing points. The whole layout shifts together (keeping its shape) by the smallest
        step that clears every occupied position; vertices are nudged one by one if none does.
        """

        def clashes(position: Tuple[float, float], others: List[Tuple[float, float]]) -> bool:
            return any(
                abs(position[0] - ox) <= _SAME_POSITION_TOLERANCE and abs(position[1] - oy) <= _SAME_POSITION_TOLERANCE
                for ox, oy in others
            )

        if not any(clashes(pos, occupied) for pos in generated.values()):
            return generated

        step = GraphManager._clearance_step(generated, placement_box)
        directions = [(1, 0), (0, -1), (-1, 0), (0, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]
        for ring in range(1, 9):
            for dx, dy in directions:
                shifted = {
                    vid: (pos[0] + dx * ring * step, pos[1] + dy * ring * step) for vid, pos in generated.items()
                }
                if not any(clashes(pos, occupied) for pos in shifted.values()):
                    return shifted

        placed: Dict[str, Tuple[float, float]] = {}
        taken = list(occupied)
        for vid in sorted(generated):
            pos = generated[vid]
            ring = 1
            while clashes(pos, taken):
                dx, dy = directions[(ring - 1) % len(directions)]
                scale = (ring + len(directions) - 1) // len(directions)
                pos = (generated[vid][0] + dx * scale * step, generated[vid][1] + dy * scale * step)
                ring += 1
            placed[vid] = pos
            taken.append(pos)
        return placed

    @staticmethod
    def _clearance_step(generated: Dict[str, Tuple[float, float]], placement_box: Optional[Dict[str, float]]) -> float:
        """Shift step for clearing occupied positions: 5% of the placement box (or of the layout's spread)."""
        size = 0.0
        if placement_box:
            size = min(float(placement_box.get("width", 0.0)), float(placement_box.get("height", 0.0)))
        if size <= 0.0 and generated:
            xs = [pos[0] for pos in generated.values()]
            ys = [pos[1] for pos in generated.values()]
            size = max(max(xs) - min(xs), max(ys) - min(ys))
        return 0.05 * size if size > 0.0 else 1.0

    def _create_edge(
        self,
        edge: GraphEdgeDescriptor,
        directed: bool,
        id_to_point: Dict[str, "Point"],
    ) -> Tuple[Optional["Segment"], Optional["Vector"]]:
        source_point = id_to_point[edge.source]
        target_point = id_to_point[edge.target]
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
