"""Shared helpers for dependency-aware drawable removal.

This module centralizes common patterns used across managers:
remove a drawable from the drawables container and, if successful, also
remove its dependency-graph entries. Keeping this logic in one place helps
prevent stale dependency edges and preserves graph invariants.

It also holds the shared deletion rule for helper objects: a delete removes
the object itself plus the helper segments and points it created, but only
those that nothing else (a polygon, angle, graph or other dependent drawable)
still uses. When the only remaining users are graphs, those graphs take
ownership, so deleting the last of them removes the helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List

from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.segment_manager import SegmentManager

GRAPH_CLASS_NAMES = ("Graph", "DirectedGraph", "UndirectedGraph", "Tree")


def remove_drawable_with_dependencies(
    drawables: "DrawablesContainer",
    dependency_manager: "DrawableDependencyManager",
    drawable: "Drawable",
) -> bool:
    """Remove drawable from container and dependency graph in one place."""
    removed = drawables.remove(drawable)
    if removed and hasattr(dependency_manager, "remove_drawable"):
        dependency_manager.remove_drawable(drawable)
    return bool(removed)


def get_polygon_segments(polygon: Any) -> List[Any]:
    """Extract segments from any polygon type.

    Prefers ``get_segments()`` (used by Pentagon through GenericPolygon),
    falling back to named ``segment1``..``segment4`` attributes (used by
    Triangle, Rectangle, and Quadrilateral).
    """
    if hasattr(polygon, "get_segments") and callable(getattr(polygon, "get_segments")):
        return polygon.get_segments()
    segments: List[Any] = []
    for attr in ("segment1", "segment2", "segment3", "segment4"):
        seg = getattr(polygon, attr, None)
        if seg is not None:
            segments.append(seg)
    return segments


def _class_name(drawable: Any) -> str:
    getter = getattr(drawable, "get_class_name", None)
    return str(getter()) if callable(getter) else ""


def _contains(items: Iterable[Any], drawable: Any) -> bool:
    return any(item is drawable for item in items)


def is_on_canvas(drawables: "DrawablesContainer", drawable: Any) -> bool:
    """Return True if the drawable is still in the container (not a stale dependency entry)."""
    return _contains(drawables.get_by_class_name(_class_name(drawable)), drawable)


def _built_only_from(area: Any, released: Iterable[Any], dependency_manager: "DrawableDependencyManager") -> bool:
    """Return True if every parent of the area is one of the released segments."""
    released_list = list(released)
    parents = dependency_manager.get_parents(area)
    return bool(parents) and all(_contains(released_list, parent) for parent in parents)


def find_segment_users(
    segment: Any,
    drawables: "DrawablesContainer",
    dependency_manager: "DrawableDependencyManager",
    *,
    released: Iterable[Any] = (),
) -> List[Any]:
    """Return the drawables on the canvas that still use the segment.

    Split child segments do not count; they go with their parent segment. A region
    area bounded only by ``released`` segments (the edges of the polygon being deleted)
    was built from that polygon and goes with it, so it does not count either. The
    polygon scan mirrors how SegmentManager finds the polygons it would delete, in
    case the dependency graph missed one.
    """
    released_list = list(released)
    users: List[Any] = []
    for dependent in dependency_manager.get_all_children(segment):
        class_name = _class_name(dependent)
        if class_name == "Segment" or not is_on_canvas(drawables, dependent):
            continue
        if class_name == "ClosedShapeColoredArea" and _built_only_from(dependent, released_list, dependency_manager):
            continue
        if not _contains(users, dependent):
            users.append(dependent)

    x1, y1 = segment.point1.x, segment.point1.y
    x2, y2 = segment.point2.x, segment.point2.y
    for polygon in drawables.iter_polygons():
        if _contains(users, polygon):
            continue
        for edge in get_polygon_segments(polygon):
            if edge is not None and MathUtils.segment_matches_coordinates(edge, x1, y1, x2, y2):
                users.append(polygon)
                break
    return users


def find_point_users(
    point: Any,
    drawables: "DrawablesContainer",
    dependency_manager: "DrawableDependencyManager",
) -> List[Any]:
    """Return the drawables on the canvas that still use the point.

    Vectors are found by scanning, since they depend on their internal segment
    rather than on their points.
    """
    users = [d for d in dependency_manager.get_children(point) if is_on_canvas(drawables, d)]
    for vector in drawables.get_by_class_name("Vector"):
        if (vector.origin is point or vector.tip is point) and not _contains(users, vector):
            users.append(vector)
    return users


def hand_over_to_graphs(drawable: Any, users: List[Any]) -> None:
    """If only graphs (and their own edges) still use the drawable, make those graphs its owners.

    A graph that reused a pre-existing helper leaves it on the canvas when deleted.
    Once the object that created the helper is gone, the graph owns it instead. A
    vertex point counts as graph-held when its remaining users are graphs and edges
    of those graphs.
    """
    graphs = [user for user in users if _class_name(user) in GRAPH_CLASS_NAMES]
    if not graphs:
        return
    graph_edges: List[Any] = []
    for graph in graphs:
        graph_edges.extend(getattr(graph, "segments", []) or [])
        graph_edges.extend(getattr(graph, "vectors", []) or [])
    if all(_contains(graphs, user) or _contains(graph_edges, user) for user in users):
        for graph in graphs:
            adopt = getattr(graph, "adopt", None)
            if callable(adopt):
                adopt(drawable)


def release_segment(
    segment: Any,
    drawables: "DrawablesContainer",
    dependency_manager: "DrawableDependencyManager",
    segment_manager: "SegmentManager",
    *,
    released: Iterable[Any] = (),
) -> bool:
    """Delete a helper segment unless something still uses it; return True if deleted."""
    users = find_segment_users(segment, drawables, dependency_manager, released=released)
    if users:
        hand_over_to_graphs(segment, users)
        return False
    segment_manager.delete_segment(segment.point1.x, segment.point1.y, segment.point2.x, segment.point2.y)
    return True
