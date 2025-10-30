from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, cast


class PointDict(TypedDict, total=False):
    x: float
    y: float
    name: str


class SegmentDict(TypedDict, total=False):
    point1: Dict[str, float]
    point2: Dict[str, float]
    name: str


class CircleDict(TypedDict, total=False):
    center: Dict[str, float]
    radius: float
    name: str


class RectangleDict(TypedDict, total=False):
    point1: Dict[str, float]
    point3: Dict[str, float]
    name: str


class TriangleDict(TypedDict, total=False):
    point1: Dict[str, float]
    point2: Dict[str, float]
    point3: Dict[str, float]
    name: str


class EllipseDict(TypedDict, total=False):
    center: Dict[str, float]
    radius_x: float
    radius_y: float
    rotation_angle: float
    name: str


class FunctionDict(TypedDict, total=False):
    function_string: str
    name: str
    left_bound: Optional[float]
    right_bound: Optional[float]


class VectorDict(TypedDict, total=False):
    origin: Dict[str, float]
    tip: Dict[str, float]
    name: str


class ComputationDict(TypedDict):
    expression: str
    result: Any


class CanvasStateDict(TypedDict, total=False):
    Points: List[PointDict]
    Segments: List[SegmentDict]
    Circles: List[CircleDict]
    Rectangles: List[RectangleDict]
    Triangles: List[TriangleDict]
    Ellipses: List[EllipseDict]
    Functions: List[FunctionDict]
    Vectors: List[VectorDict]
    computations: List[ComputationDict]


class MockCanvas:
    def __init__(self, width: float, height: float, draw_enabled: bool = True) -> None:
        self.width: float = width
        self.height: float = height
        self.draw_enabled: bool = draw_enabled
        self.points: List[PointDict] = []
        self.segments: List[SegmentDict] = []
        self.circles: List[CircleDict] = []
        self.rectangles: List[RectangleDict] = []
        self.triangles: List[TriangleDict] = []
        self.ellipses: List[EllipseDict] = []
        self.functions: List[FunctionDict] = []
        self.vectors: List[VectorDict] = []
        self.computations: List[ComputationDict] = []

    def get_drawables(self) -> List[Dict[str, Any]]:
        """Get all drawable objects on the canvas."""
        return cast(List[Dict[str, Any]], (
            self.points +
            self.segments +
            self.circles +
            self.rectangles +
            self.triangles +
            self.ellipses +
            self.functions +
            self.vectors
        ))

    def get_drawables_by_class_name(self, class_name: str) -> List[Dict[str, Any]]:
        """Get drawables of a specific class."""
        class_map = {
            "Point": self.points,
            "Segment": self.segments,
            "Circle": self.circles,
            "Rectangle": self.rectangles,
            "Triangle": self.triangles,
            "Ellipse": self.ellipses,
            "Function": self.functions,
            "Vector": self.vectors
        }
        return cast(List[Dict[str, Any]], class_map.get(class_name, []))

    def get_canvas_state(self) -> CanvasStateDict:
        """Return the current state of the canvas."""
        return {
            "Points": self.points,
            "Segments": self.segments,
            "Circles": self.circles,
            "Rectangles": self.rectangles,
            "Triangles": self.triangles,
            "Ellipses": self.ellipses,
            "Functions": self.functions,
            "Vectors": self.vectors,
            "computations": self.computations
        }

    def clear(self) -> None:
        """Clear all objects from the canvas."""
        self.points = []
        self.segments = []
        self.circles = []
        self.rectangles = []
        self.triangles = []
        self.ellipses = []
        self.functions = []
        self.vectors = []
        self.computations = []

    def create_point(self, x: float, y: float, name: str = "") -> PointDict:
        """Create a point on the canvas."""
        point: PointDict = cast(PointDict, {"x": x, "y": y, "name": name})
        self.points.append(point)
        return point

    def create_segment(self, x1: float, y1: float, x2: float, y2: float, name: str = "") -> SegmentDict:
        """Create a segment on the canvas."""
        segment: SegmentDict = cast(SegmentDict, {
            "point1": {"x": x1, "y": y1},
            "point2": {"x": x2, "y": y2},
            "name": name
        })
        self.segments.append(segment)
        return segment

    def create_circle(self, x: float, y: float, radius: float, name: str = "") -> CircleDict:
        """Create a circle on the canvas."""
        circle: CircleDict = cast(CircleDict, {
            "center": {"x": x, "y": y},
            "radius": radius,
            "name": name
        })
        self.circles.append(circle)
        return circle

    def create_rectangle(self, x1: float, y1: float, x2: float, y2: float, name: str = "") -> RectangleDict:
        """Create a rectangle on the canvas."""
        rectangle: RectangleDict = cast(RectangleDict, {
            "point1": {"x": x1, "y": y1},
            "point3": {"x": x2, "y": y2},
            "name": name
        })
        self.rectangles.append(rectangle)
        return rectangle

    def create_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, name: str = "") -> TriangleDict:
        """Create a triangle on the canvas."""
        triangle: TriangleDict = cast(TriangleDict, {
            "point1": {"x": x1, "y": y1},
            "point2": {"x": x2, "y": y2},
            "point3": {"x": x3, "y": y3},
            "name": name
        })
        self.triangles.append(triangle)
        return triangle

    def create_ellipse(self, x: float, y: float, radius_x: float, radius_y: float, rotation_angle: float = 0, name: str = "") -> EllipseDict:
        """Create an ellipse on the canvas."""
        ellipse: EllipseDict = cast(EllipseDict, {
            "center": {"x": x, "y": y},
            "radius_x": radius_x,
            "radius_y": radius_y,
            "rotation_angle": rotation_angle,
            "name": name
        })
        self.ellipses.append(ellipse)
        return ellipse

    def draw_function(self, function_string: str, name: str = "", left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> FunctionDict:
        """Add a function to the canvas."""
        function: FunctionDict = cast(FunctionDict, {
            "function_string": function_string,
            "name": name,
            "left_bound": left_bound,
            "right_bound": right_bound
        })
        self.functions.append(function)
        return function

    def create_vector(self, x1: float, y1: float, x2: float, y2: float, name: str = "") -> VectorDict:
        """Create a vector on the canvas."""
        vector: VectorDict = cast(VectorDict, {
            "origin": {"x": x1, "y": y1},
            "tip": {"x": x2, "y": y2},
            "name": name
        })
        self.vectors.append(vector)
        return vector

    def add_computation(self, expression: str, result: Any) -> ComputationDict:
        """Add a computation to the canvas."""
        computation: ComputationDict = cast(ComputationDict, {
            "expression": expression,
            "result": result
        })
        self.computations.append(computation)
        return computation 