"""Normalized geometry view of a canvas state, plus tolerances.

A ``CanvasView`` wraps one canvas state (``Canvas.get_canvas_state()``, as the
``getMatHudCanvasState`` hook returns it) and, optionally, the hook's inspection
view. It resolves point references to coordinates so checks can talk about
segments, polygons, circles and angles by geometry instead of by name.

Everything here is pure Python: no browser, so the check engine can run on
stored states (``--regrade``) and in pytest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

Coord = tuple[float, float]

# Top-level state keys that are not drawable buckets.
NON_DRAWABLE_KEYS = frozenset({"Cartesian_System_Visibility", "coordinate_system", "computations"})

POLYGON_TYPES = frozenset(
    {
        "Triangle",
        "Rectangle",
        "Quadrilateral",
        "Pentagon",
        "Hexagon",
        "Heptagon",
        "Octagon",
        "Nonagon",
        "Decagon",
        "GenericPolygon",
    }
)
FUNCTION_TYPES = frozenset({"Function", "PiecewiseFunction"})
COLORED_AREA_TYPES = frozenset(
    {
        "FunctionsBoundedColoredArea",
        "SegmentsBoundedColoredArea",
        "FunctionSegmentBoundedColoredArea",
        "ClosedShapeColoredArea",
    }
)
GRAPH_TYPES = frozenset({"UndirectedGraph", "DirectedGraph", "Tree"})
PLOT_TYPES = frozenset({"ContinuousPlot", "DiscretePlot", "BarsPlot"})
# Selector type names that match a family of types.
TYPE_FAMILIES: dict[str, frozenset[str]] = {
    "Polygon": POLYGON_TYPES,
    "AnyFunction": FUNCTION_TYPES,
    "ColoredArea": COLORED_AREA_TYPES,
    "Graph": GRAPH_TYPES,
    "Plot": PLOT_TYPES,
}


@dataclass(frozen=True)
class Tolerance:
    """Absolute plus relative tolerance: ``|a - b| <= abs + rel * max(|a|, |b|)``."""

    abs: float = 1e-9
    rel: float = 1e-9

    def close(self, a: float, b: float) -> bool:
        """True when ``a`` and ``b`` agree within the tolerance (NaN never agrees)."""
        if math.isnan(a) or math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= self.abs + self.rel * max(abs(a), abs(b))

    def close_coords(self, a: Sequence[float], b: Sequence[float]) -> bool:
        """Component-wise ``close`` for two coordinate sequences of equal length."""
        return len(a) == len(b) and all(self.close(float(x), float(y)) for x, y in zip(a, b))

    def merged(self, override: Any) -> "Tolerance":
        """A tolerance with ``override`` applied: a number sets both parts, a dict sets either."""
        if override is None:
            return self
        if isinstance(override, (int, float)) and not isinstance(override, bool):
            return Tolerance(float(override), float(override))
        if isinstance(override, dict):
            return Tolerance(float(override.get("abs", self.abs)), float(override.get("rel", self.rel)))
        raise ValueError(f"Invalid tolerance: {override!r}")


def bucket_type(bucket: str) -> str:
    """``"Points"`` -> ``"Point"``: every drawable bucket is its class name plus ``s``."""
    return bucket[:-1] if bucket.endswith("s") else bucket


def type_matches(requested: str, actual: str) -> bool:
    """Whether a selector's ``type`` covers an object's type (families such as ``Polygon`` included)."""
    family = TYPE_FAMILIES.get(requested)
    return actual in family if family is not None else requested == actual


@dataclass
class CanvasObject:
    """One drawable in a state: its bucket, type, name, raw state entry and inspection record."""

    bucket: str
    type: str
    name: str
    data: dict[str, Any]
    inspect: Optional[dict[str, Any]] = None

    @property
    def args(self) -> dict[str, Any]:
        args = self.data.get("args")
        return args if isinstance(args, dict) else {}

    @property
    def key(self) -> tuple[str, str]:
        return (self.bucket, self.name)


@dataclass
class SampleRequests:
    """Function samples a check needed but the view did not have (``name -> x values``)."""

    x: dict[str, set[float]] = field(default_factory=dict)
    t: dict[str, set[float]] = field(default_factory=dict)

    def add_x(self, name: str, value: float) -> None:
        self.x.setdefault(name, set()).add(float(value))

    def add_t(self, name: str, value: float) -> None:
        self.t.setdefault(name, set()).add(float(value))

    def merge(self, other: "SampleRequests") -> None:
        for name, values in other.x.items():
            self.x.setdefault(name, set()).update(values)
        for name, values in other.t.items():
            self.t.setdefault(name, set()).update(values)

    def __bool__(self) -> bool:
        return bool(self.x or self.t)

    def as_options(self) -> dict[str, Any]:
        """Options for ``getMatHudCanvasState`` that fetch these samples."""
        options: dict[str, Any] = {}
        if self.x:
            options["samples"] = {name: sorted(values) for name, values in self.x.items()}
        if self.t:
            options["t_samples"] = {name: sorted(values) for name, values in self.t.items()}
        return options


class MissingSample(Exception):
    """A function value is needed that the view has not sampled."""


class CanvasView:
    """A canvas state with point references resolved to coordinates."""

    def __init__(self, state: Optional[dict[str, Any]], inspection: Optional[dict[str, Any]] = None) -> None:
        self.state: dict[str, Any] = state if isinstance(state, dict) else {}
        self.inspection: dict[str, Any] = inspection if isinstance(inspection, dict) else {}
        self.requests = SampleRequests()
        inspect_index: dict[tuple[str, str], dict[str, Any]] = {}
        for record in self.inspection.get("drawables", []) or []:
            if isinstance(record, dict):
                inspect_index[(str(record.get("class")), str(record.get("name")))] = record
        self.objects: list[CanvasObject] = []
        for bucket, items in self.state.items():
            if bucket in NON_DRAWABLE_KEYS or not isinstance(items, list):
                continue
            obj_type = bucket_type(bucket)
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    name = str(item["name"])
                    self.objects.append(CanvasObject(bucket, obj_type, name, item, inspect_index.get((obj_type, name))))
        self._points: dict[str, Coord] = {}
        for obj in self.objects:
            if obj.type == "Point":
                position = obj.args.get("position") or {}
                try:
                    self._points[obj.name] = (float(position["x"]), float(position["y"]))
                except (KeyError, TypeError, ValueError):
                    continue

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def of_type(self, requested: str) -> list[CanvasObject]:
        return [obj for obj in self.objects if type_matches(requested, obj.type)]

    def find(self, bucket: str, name: str) -> Optional[CanvasObject]:
        for obj in self.objects:
            if obj.bucket == bucket and obj.name == name:
                return obj
        return None

    def find_by_type_and_name(self, obj_type: str, name: str) -> Optional[CanvasObject]:
        for obj in self.objects:
            if type_matches(obj_type, obj.type) and obj.name == name:
                return obj
        return None

    def names(self, obj_type: str) -> list[str]:
        return [obj.name for obj in self.of_type(obj_type)]

    def point(self, name: Any) -> Optional[Coord]:
        return self._points.get(str(name)) if name is not None else None

    @property
    def view_bounds(self) -> dict[str, float]:
        bounds = self.state.get("Cartesian_System_Visibility")
        return bounds if isinstance(bounds, dict) else {}

    @property
    def coordinate_mode(self) -> Optional[str]:
        system = self.state.get("coordinate_system")
        if isinstance(system, dict) and system.get("mode"):
            return str(system["mode"])
        mode = self.inspection.get("coordinate_mode")
        return str(mode) if mode else None

    # ------------------------------------------------------------------
    # Geometry of objects
    # ------------------------------------------------------------------

    def segment_ends(self, obj: CanvasObject) -> Optional[tuple[Coord, Coord]]:
        """Endpoints of a segment or vector (origin first for vectors)."""
        args = obj.args
        if obj.type == "Vector":
            a, b = self.point(args.get("origin")), self.point(args.get("tip"))
            fallback: tuple[Any, Any] = (obj.data.get("_origin_coords"), obj.data.get("_tip_coords"))
        else:
            a, b = self.point(args.get("p1")), self.point(args.get("p2"))
            fallback = (obj.data.get("_p1_coords"), obj.data.get("_p2_coords"))
        if a is None and _is_pair(fallback[0]):
            a = (float(fallback[0][0]), float(fallback[0][1]))
        if b is None and _is_pair(fallback[1]):
            b = (float(fallback[1][0]), float(fallback[1][1]))
        if a is None or b is None:
            return None
        return a, b

    def polygon_vertex_names(self, obj: CanvasObject) -> list[str]:
        args = obj.args
        keys = sorted((k for k in args if k.startswith("p") and k[1:].isdigit()), key=lambda k: int(k[1:]))
        return [str(args[k]) for k in keys]

    def polygon_vertices(self, obj: CanvasObject) -> Optional[list[Coord]]:
        """Vertex coordinates, in boundary order when the inspection view has it."""
        if obj.inspect and isinstance(obj.inspect.get("vertices"), list):
            vertices = [(float(v[0]), float(v[1])) for v in obj.inspect["vertices"] if _is_pair(v)]
            if vertices:
                return order_around_centroid(vertices)
        coords = [self.point(name) for name in self.polygon_vertex_names(obj)]
        if not coords or any(c is None for c in coords):
            return None
        return order_around_centroid([c for c in coords if c is not None])

    def circle(self, obj: CanvasObject) -> Optional[tuple[Coord, float]]:
        center = self.point(obj.args.get("center"))
        radius = obj.args.get("radius")
        if center is None or not isinstance(radius, (int, float)):
            return None
        return center, float(radius)

    def ellipse(self, obj: CanvasObject) -> Optional[tuple[Coord, float, float, float]]:
        center = self.point(obj.args.get("center"))
        args = obj.args
        if center is None:
            return None
        try:
            return center, float(args["radius_x"]), float(args["radius_y"]), float(args.get("rotation_angle") or 0.0)
        except (KeyError, TypeError, ValueError):
            return None

    def arc(self, obj: CanvasObject) -> Optional[dict[str, Any]]:
        args = obj.args
        try:
            return {
                "p1": self.point(args.get("point1_name")),
                "p2": self.point(args.get("point2_name")),
                "center": (float(args["center_x"]), float(args["center_y"])),
                "radius": float(args["radius"]),
                "major": bool(args.get("use_major_arc")),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def angle_points(self, obj: CanvasObject) -> Optional[tuple[Coord, Coord, Coord]]:
        """``(vertex, arm1, arm2)`` of an angle, from its two segments' shared endpoint."""
        seg1 = self.find("Segments", str(obj.args.get("segment1_name")))
        seg2 = self.find("Segments", str(obj.args.get("segment2_name")))
        if seg1 is None or seg2 is None:
            return None
        names1 = [seg1.args.get("p1"), seg1.args.get("p2")]
        names2 = [seg2.args.get("p1"), seg2.args.get("p2")]
        shared = [n for n in names1 if n in names2]
        if len(shared) != 1:
            return None
        vertex = shared[0]
        arm1 = next(n for n in names1 if n != vertex)
        arm2 = next(n for n in names2 if n != vertex)
        v, a1, a2 = self.point(vertex), self.point(arm1), self.point(arm2)
        if v is None or a1 is None or a2 is None:
            return None
        return v, a1, a2

    def angle_degrees(self, obj: CanvasObject) -> Optional[float]:
        """Geometric size of an angle marker: counter-clockwise from arm1 to arm2, or its reflex."""
        points = self.angle_points(obj)
        if points is None:
            return None
        raw = ccw_angle_degrees(*points)
        small = raw if raw <= 180 else 360 - raw
        return 360 - small if obj.args.get("is_reflex") else small

    def label_position(self, obj: CanvasObject) -> Optional[Coord]:
        position = obj.args.get("position") or {}
        try:
            return float(position["x"]), float(position["y"])
        except (KeyError, TypeError, ValueError):
            return None

    def anchor(self, obj: CanvasObject) -> Optional[Coord]:
        """A single representative location: a point, a label, or a circle or ellipse centre."""
        if obj.type == "Point":
            return self.point(obj.name)
        if obj.type == "Label":
            return self.label_position(obj)
        if obj.type in ("Circle", "Ellipse"):
            return self.point(obj.args.get("center"))
        return None

    # ------------------------------------------------------------------
    # Function samples
    # ------------------------------------------------------------------

    def sample(self, obj: CanvasObject, x: float) -> Optional[float]:
        """Value of a function at ``x`` from the inspection samples; raises MissingSample if absent."""
        for pair in (obj.inspect or {}).get("samples", []) or []:
            if _is_pair(pair) and math.isclose(float(pair[0]), x, rel_tol=1e-12, abs_tol=1e-12):
                return None if pair[1] is None else float(pair[1])
        self.requests.add_x(obj.name, x)
        raise MissingSample(f"{obj.name}({x})")

    def sample_t(self, obj: CanvasObject, t: float) -> Optional[Coord]:
        """Point of a parametric curve at ``t``; raises MissingSample if absent."""
        for triple in (obj.inspect or {}).get("t_samples", []) or []:
            if isinstance(triple, list) and len(triple) == 3 and math.isclose(float(triple[0]), t, abs_tol=1e-12):
                if triple[1] is None or triple[2] is None:
                    return None
                return float(triple[1]), float(triple[2])
        self.requests.add_t(obj.name, t)
        raise MissingSample(f"{obj.name}(t={t})")

    # ------------------------------------------------------------------
    # Canonical geometry (for name-independent comparisons)
    # ------------------------------------------------------------------

    def signature(self, obj: CanvasObject, digits: int = 6) -> Any:
        """A name-independent description of an object's geometry, rounded to ``digits``."""

        def r(value: float) -> float:
            return round(value, digits) + 0.0

        def rc(coord: Coord) -> tuple[float, float]:
            return (r(coord[0]), r(coord[1]))

        if obj.type == "Point":
            p = self.point(obj.name)
            return ("Point", rc(p)) if p else ("Point", None)
        if obj.type in ("Segment", "Vector"):
            ends = self.segment_ends(obj)
            if ends is None:
                return (obj.type, None)
            pair = (rc(ends[0]), rc(ends[1]))
            return (obj.type, pair if obj.type == "Vector" else tuple(sorted(pair)))
        if obj.type in POLYGON_TYPES:
            vertices = self.polygon_vertices(obj)
            return (obj.type, tuple(sorted(rc(v) for v in vertices)) if vertices else None)
        if obj.type == "Circle":
            circle = self.circle(obj)
            return ("Circle", (rc(circle[0]), r(circle[1])) if circle else None)
        return (obj.type, _freeze(_strip_names(obj.args)))


# ----------------------------------------------------------------------
# Plain geometry helpers
# ----------------------------------------------------------------------


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2])


def _strip_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_names(v) for k, v in value.items() if not str(k).endswith("name")}
    if isinstance(value, list):
        return [_strip_names(v) for v in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def distance(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def order_around_centroid(points: Sequence[Coord]) -> list[Coord]:
    """Sort vertices counter-clockwise around their centroid (boundary order for convex polygons)."""
    if len(points) < 3:
        return list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def polygon_area(points: Sequence[Coord]) -> float:
    """Shoelace area of a polygon given in boundary order."""
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(point: Coord, polygon: Sequence[Coord]) -> bool:
    """Ray casting; the polygon is in boundary order."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross:
                inside = not inside
    return inside


def line_distance(point: Coord, a: Coord, b: Coord) -> float:
    """Distance from ``point`` to the infinite line through ``a`` and ``b``."""
    length = distance(a, b)
    if length == 0:
        return distance(point, a)
    return abs((b[0] - a[0]) * (a[1] - point[1]) - (a[0] - point[0]) * (b[1] - a[1])) / length


def segment_distance(point: Coord, a: Coord, b: Coord) -> float:
    """Distance from ``point`` to the segment ``ab``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return distance(point, a)
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq))
    return distance(point, (a[0] + t * dx, a[1] + t * dy))


def direction(a: Coord, b: Coord) -> Coord:
    return (b[0] - a[0], b[1] - a[1])


def cross(u: Coord, v: Coord) -> float:
    return u[0] * v[1] - u[1] * v[0]


def dot(u: Coord, v: Coord) -> float:
    return u[0] * v[0] + u[1] * v[1]


def unit_angle_between(u: Coord, v: Coord) -> float:
    """Unsigned angle between two directions, in degrees (0 to 180)."""
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return float("nan")
    cosine = max(-1.0, min(1.0, dot(u, v) / (nu * nv)))
    return math.degrees(math.acos(cosine))


def ccw_angle_degrees(vertex: Coord, arm1: Coord, arm2: Coord) -> float:
    """Counter-clockwise angle from arm1 to arm2 around ``vertex``, in [0, 360)."""
    a1 = math.atan2(arm1[1] - vertex[1], arm1[0] - vertex[0])
    a2 = math.atan2(arm2[1] - vertex[1], arm2[0] - vertex[0])
    return math.degrees(a2 - a1) % 360.0


def triangle_flags(vertices: Sequence[Coord], tol: float = 1e-6) -> set[str]:
    """Classification flags of a triangle: equilateral, isosceles, scalene, right, obtuse, acute."""
    a, b, c = vertices
    sides = sorted([distance(a, b), distance(b, c), distance(c, a)])
    flags: set[str] = set()

    def same(x: float, y: float) -> bool:
        return abs(x - y) <= tol * max(1.0, x, y)

    equal_pairs = sum(1 for i, j in ((0, 1), (1, 2), (0, 2)) if same(sides[i], sides[j]))
    if equal_pairs == 3:
        flags.update({"equilateral", "isosceles"})
    elif equal_pairs >= 1:
        flags.add("isosceles")
    else:
        flags.add("scalene")
    lhs, rhs = sides[0] ** 2 + sides[1] ** 2, sides[2] ** 2
    if same(lhs, rhs):
        flags.add("right")
    elif lhs < rhs:
        flags.add("obtuse")
    else:
        flags.add("acute")
    return flags


def iter_numbers(value: Any, path: str = "") -> Iterable[tuple[str, float]]:
    """Every number inside a JSON value, with its dotted path."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_numbers(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_numbers(item, f"{path}[{index}]")
