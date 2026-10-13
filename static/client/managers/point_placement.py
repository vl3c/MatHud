"""Place labelled points at computed locations as one undo step.

Shared by the tools that report positions and can mark them on the canvas
(find_function_features, find_intersections).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Sequence

if TYPE_CHECKING:
    from managers.point_manager import PointManager
    from managers.undo_redo_manager import UndoRedoManager


def place_points_at(
    point_manager: "PointManager",
    undo_manager: "UndoRedoManager",
    locations: Sequence[MutableMapping[str, Any]],
    what: str,
) -> Dict[str, Any]:
    """Create (or reuse) a point at each location's x and y, all as one undo step.

    Sets ``point_name`` on every location. Locations at the same spot share one point. A
    point that already existed at a location is reused and left unchanged; it is reported
    apart from the created ones, so the caller never deletes a point of the user's drawing
    thinking this call made it.

    Args:
        what: What the locations are ("feature", "intersection"), for the note.

    Returns:
        point_names (distinct, in location order), created_point_names, reused_point_names,
        and a note when points were reused.
    """
    names: List[str] = []
    created: List[str] = []
    reused: List[str] = []
    undo_manager.begin_batch()
    try:
        for location in locations:
            x, y = location["x"], location["y"]
            existed_before = point_manager.get_point(x, y) is not None
            point = point_manager.create_point(x, y, name="", extra_graphics=False)
            name = str(point.name)
            location["point_name"] = name
            if name in names:
                continue
            names.append(name)
            (reused if existed_before else created).append(name)
    finally:
        undo_manager.end_batch()
    placed: Dict[str, Any] = {
        "point_names": names,
        "created_point_names": created,
        "reused_point_names": reused,
    }
    if reused:
        reused_text = ", ".join(reused)
        placed["note"] = (
            f"Points {reused_text} already existed at {what} locations and were reused, not created; "
            f"to remove the {what} points, delete only created_point_names."
        )
    return placed
