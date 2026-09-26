"""Builders for canvas-state dicts in the shape ``Canvas.get_canvas_state()`` returns.

Used by the scenario check-engine tests; several states copy observations from
section 6 of documentation/development/agentic_scenario_testing.md.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

VIEW = {"left_bound": -628, "right_bound": 628, "top_bound": 481.5, "bottom_bound": -481.5}


def state(
    *items: tuple[str, dict[str, Any]], view: Optional[dict[str, Any]] = None, mode: str = "cartesian"
) -> dict[str, Any]:
    """A state from ``(bucket, entry)`` pairs."""
    result: dict[str, Any] = {
        "Cartesian_System_Visibility": dict(view or VIEW),
        "coordinate_system": {"mode": mode},
    }
    for bucket, entry in items:
        result.setdefault(bucket, []).append(entry)
    return result


def point(name: str, x: float, y: float) -> tuple[str, dict[str, Any]]:
    return "Points", {"name": name, "args": {"position": {"x": x, "y": y}}}


def points(**coords: tuple[float, float]) -> list[tuple[str, dict[str, Any]]]:
    return [point(name, x, y) for name, (x, y) in coords.items()]


def segment(
    p1: str, p2: str, c1: Iterable[float], c2: Iterable[float], label: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    entry: dict[str, Any] = {
        "name": p1 + p2,
        "args": {"p1": p1, "p2": p2},
        "_p1_coords": list(c1),
        "_p2_coords": list(c2),
    }
    if label is not None:
        entry["args"]["label"] = {"text": label, "visible": True}
    return "Segments", entry


def vector(origin: str, tip: str, c1: Iterable[float], c2: Iterable[float]) -> tuple[str, dict[str, Any]]:
    return "Vectors", {
        "name": origin + tip,
        "args": {"origin": origin, "tip": tip},
        "_origin_coords": list(c1),
        "_tip_coords": list(c2),
    }


def triangle(a: str, b: str, c: str, types: list[str]) -> tuple[str, dict[str, Any]]:
    return "Triangles", {"name": a + b + c, "args": {"p1": a, "p2": b, "p3": c}, "types": types}


def circle(center: str, radius: float, cx: float, cy: float) -> tuple[str, dict[str, Any]]:
    formula = f"(x - {float(cx)})**2 + (y - {float(cy)})**2 = {radius}**2"
    return "Circles", {
        "name": f"{center}({radius})",
        "args": {"center": center, "radius": radius, "circle_formula": formula},
    }


def function(name: str, expression: str, **args: Any) -> tuple[str, dict[str, Any]]:
    entry_args: dict[str, Any] = {"function_string": expression, "left_bound": None, "right_bound": None}
    entry_args.update(args)
    return "Functions", {"name": name, "args": entry_args}


def angle(name: str, segment1: str, segment2: str, is_reflex: bool = False) -> tuple[str, dict[str, Any]]:
    return "Angles", {
        "name": name,
        "type": "angle",
        "args": {"segment1_name": segment1, "segment2_name": segment2, "color": "blue", "is_reflex": is_reflex},
    }


def right_triangle(types: Optional[list[str]] = None) -> list[tuple[str, dict[str, Any]]]:
    """Triangle ABC with A(0,0), B(4,0), C(0,3), its sides and its type flags."""
    return points(A=(0, 0), B=(4, 0), C=(0, 3)) + [
        segment("A", "B", (0, 0), (4, 0)),
        segment("B", "C", (4, 0), (0, 3)),
        segment("C", "A", (0, 3), (0, 0)),
        triangle("A", "B", "C", types if types is not None else ["triangle", "scalene", "right"]),
    ]


def inspection(*drawables: dict[str, Any], **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "drawables": list(drawables),
        "undo_depth": 0,
        "redo_depth": 0,
        "coordinate_mode": "cartesian",
        "grid_visible": {"cartesian": True, "polar": True, "active": True},
        "name_hints": {},
    }
    data.update(extra)
    return data
