"""
MatHud Drawables Aggregator

Convenience import aggregator for all drawable geometric objects.
Provides a single import point for all mathematical visualization classes.

Exported Classes:
    - Drawable: Base class for all canvas objects
    - Point, Position: Point and position representations
    - Segment, Vector: Line-based geometric objects
    - Triangle, Rectangle: Polygon shapes
    - Circle, Ellipse: Curved geometric objects
    - Function: Mathematical function plotting

Dependencies:
    - drawables.*: Individual geometric class implementations
"""

from __future__ import annotations

# Import all drawable classes from their individual files
from drawables.circle import Circle
from drawables.drawable import Drawable
from drawables.ellipse import Ellipse
from drawables.function import Function
from drawables.rectangle import Rectangle
from drawables.quadrilateral import Quadrilateral
from drawables.pentagon import Pentagon
from drawables.hexagon import Hexagon
from drawables.segment import Segment
from drawables.triangle import Triangle
from drawables.vector import Vector
from drawables.point import Point
from drawables.position import Position
from drawables.graph import Graph
from drawables.directed_graph import DirectedGraph
from drawables.undirected_graph import UndirectedGraph
from drawables.tree import Tree

# Re-export all classes for convenient importing
__all__: list[str] = [
    "Drawable",
    "Point",
    "Position",
    "Segment",
    "Vector",
    "Triangle",
    "Quadrilateral",
    "Pentagon",
    "Hexagon",
    "Rectangle",
    "Circle",
    "Ellipse",
    "Function",
    "Graph",
    "DirectedGraph",
    "UndirectedGraph",
    "Tree",
]
