"""Tests for geometric transform methods on drawables and TransformationsManager."""

from __future__ import annotations

import math
import unittest
from typing import Dict, Iterable, List, Set, Tuple

from drawables_aggregator import Point, Segment, Triangle, Rectangle, Circle, Ellipse
from drawables.vector import Vector
from managers.transformations_manager import TransformationsManager
from client_tests.simple_mock import SimpleMock


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _IdentityDependencyManager:
    """Minimal dependency manager for manager-level tests."""

    def __init__(self) -> None:
        self._edges: Dict[int, Set[int]] = {}
        self._lookup: Dict[int, object] = {}

    def register(self, parent: object, child: object) -> None:
        self._edges.setdefault(id(parent), set()).add(id(child))
        self._lookup[id(parent)] = parent
        self._lookup[id(child)] = child

    def get_children(self, drawable: object) -> set:
        child_ids = self._edges.get(id(drawable), set())
        return {self._lookup[cid] for cid in child_ids if cid in self._lookup}


def _build_canvas(
    primary_drawable: object,
    segments: List[Segment],
    extra_drawables: List[object] | None = None,
) -> Tuple[SimpleMock, _IdentityDependencyManager, SimpleMock]:
    renderer = SimpleMock()
    renderer.invalidate_drawable_cache = SimpleMock()

    dependency_manager = _IdentityDependencyManager()
    for seg in segments:
        dependency_manager.register(seg, primary_drawable)

    all_drawables: list = [primary_drawable]
    if extra_drawables:
        all_drawables.extend(extra_drawables)

    drawables_container = SimpleMock(Segments=list(segments))
    drawable_manager = SimpleMock(
        get_drawables=SimpleMock(return_value=all_drawables),
        drawables=drawables_container,
    )
    canvas = SimpleMock(
        renderer=renderer,
        dependency_manager=dependency_manager,
        drawable_manager=drawable_manager,
        draw_enabled=False,
        draw=SimpleMock(),
        undo_redo_manager=SimpleMock(archive=SimpleMock()),
    )
    return canvas, dependency_manager, renderer


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ===================================================================
# Drawable-level tests
# ===================================================================


class TestTransforms(unittest.TestCase):
    """Comprehensive tests for geometric transform methods."""

    # ---------------------------------------------------------------
    # Point: reflect
    # ---------------------------------------------------------------
    def test_point_reflect_x_axis(self) -> None:
        p = Point(3, 4, name="P")
        p.reflect("x_axis")
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, -4))

    def test_point_reflect_y_axis(self) -> None:
        p = Point(3, 4, name="P")
        p.reflect("y_axis")
        self.assertTrue(_approx(p.x, -3))
        self.assertTrue(_approx(p.y, 4))

    def test_point_reflect_line_y_equals_x(self) -> None:
        """Reflect across y = x (a=1, b=-1, c=0)."""
        p = Point(3, 4, name="P")
        p.reflect("line", a=1, b=-1, c=0)
        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 3))

    def test_point_reflect_degenerate_line(self) -> None:
        """Degenerate line (a=b=c=0) should leave point unchanged."""
        p = Point(3, 4, name="P")
        p.reflect("line", a=0, b=0, c=0)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 4))

    # ---------------------------------------------------------------
    # Point: scale
    # ---------------------------------------------------------------
    def test_point_scale_uniform_from_origin(self) -> None:
        p = Point(2, 3, name="P")
        p.scale(2, 2, 0, 0)
        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 6))

    def test_point_scale_nonuniform_from_center(self) -> None:
        p = Point(4, 6, name="P")
        p.scale(0.5, 2, 2, 2)
        # x: 2 + 0.5*(4-2) = 3, y: 2 + 2*(6-2) = 10
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 10))

    # ---------------------------------------------------------------
    # Point: shear
    # ---------------------------------------------------------------
    def test_point_shear_horizontal(self) -> None:
        p = Point(1, 2, name="P")
        p.shear("horizontal", 0.5, 0, 0)
        # x = 0 + (1 - 0) + 0.5 * (2 - 0) = 2, y = 2
        self.assertTrue(_approx(p.x, 2))
        self.assertTrue(_approx(p.y, 2))

    def test_point_shear_vertical(self) -> None:
        p = Point(3, 1, name="P")
        p.shear("vertical", 2, 0, 0)
        # x = 3, y = 0 + (1 - 0) + 2 * (3 - 0) = 7
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 7))

    # ---------------------------------------------------------------
    # Point: rotate_around
    # ---------------------------------------------------------------
    def test_point_rotate_around_origin_90(self) -> None:
        p = Point(1, 0, name="P")
        p.rotate_around(90, 0, 0)
        self.assertTrue(_approx(p.x, 0))
        self.assertTrue(_approx(p.y, 1))

    def test_point_rotate_around_arbitrary_center(self) -> None:
        p = Point(3, 0, name="P")
        p.rotate_around(180, 2, 0)
        self.assertTrue(_approx(p.x, 1))
        self.assertTrue(_approx(p.y, 0, tol=1e-9))

    # ---------------------------------------------------------------
    # Segment transforms
    # ---------------------------------------------------------------
    def test_segment_reflect_x_axis(self) -> None:
        p1 = Point(0, 1, name="A")
        p2 = Point(4, 3, name="B")
        s = Segment(p1, p2)
        s.reflect("x_axis")
        self.assertTrue(_approx(p1.y, -1))
        self.assertTrue(_approx(p2.y, -3))
        # Line formula should be recalculated
        self.assertIsNotNone(s.line_formula)

    def test_segment_scale(self) -> None:
        p1 = Point(1, 1, name="A")
        p2 = Point(3, 1, name="B")
        s = Segment(p1, p2)
        s.scale(2, 2, 0, 0)
        self.assertTrue(_approx(p1.x, 2))
        self.assertTrue(_approx(p2.x, 6))

    def test_segment_shear(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(2, 0, name="B")
        s = Segment(p1, p2)
        s.shear("horizontal", 1, 0, 0)
        # p1 unchanged (dy=0), p2 unchanged (dy=0)
        self.assertTrue(_approx(p1.x, 0))
        self.assertTrue(_approx(p2.x, 2))

    def test_segment_rotate_around(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(2, 0, name="B")
        s = Segment(p1, p2)
        s.rotate_around(90, 0, 0)
        self.assertTrue(_approx(p1.x, 0))
        self.assertTrue(_approx(p1.y, 0))
        self.assertTrue(_approx(p2.x, 0))
        self.assertTrue(_approx(p2.y, 2))

    # ---------------------------------------------------------------
    # Vector transforms
    # ---------------------------------------------------------------
    def test_vector_reflect(self) -> None:
        v = Vector(Point(0, 0, name="O"), Point(1, 1, name="T"))
        v.reflect("y_axis")
        self.assertTrue(_approx(v.origin.x, 0))
        self.assertTrue(_approx(v.tip.x, -1))

    def test_vector_scale(self) -> None:
        v = Vector(Point(1, 0, name="O"), Point(3, 0, name="T"))
        v.scale(2, 2, 0, 0)
        self.assertTrue(_approx(v.origin.x, 2))
        self.assertTrue(_approx(v.tip.x, 6))

    # ---------------------------------------------------------------
    # Polygon (Triangle) transforms
    # ---------------------------------------------------------------
    def test_triangle_reflect_x_axis(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        tri.reflect("x_axis")
        self.assertTrue(_approx(p3.y, -3))

    def test_triangle_scale_from_external_center(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        tri.scale(2, 2, 0, 0)
        self.assertTrue(_approx(p2.x, 8))
        self.assertTrue(_approx(p3.y, 6))

    def test_triangle_shear(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        tri.shear("horizontal", 1, 0, 0)
        # p3: x = 2 + 1*3 = 5
        self.assertTrue(_approx(p3.x, 5))

    def test_triangle_rotate_around(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        tri.rotate_around(180, 2, 0)
        # p1: (0,0) → rotate 180 around (2,0): (4, 0)
        self.assertTrue(_approx(p1.x, 4))
        self.assertTrue(_approx(p1.y, 0, tol=1e-9))
        # p2: (4,0) → (0, 0)
        self.assertTrue(_approx(p2.x, 0, tol=1e-9))

    # ---------------------------------------------------------------
    # Circle transforms
    # ---------------------------------------------------------------
    def test_circle_reflect_center_moves(self) -> None:
        c = Circle(Point(3, 4, name="C"), 5)
        c.reflect("x_axis")
        self.assertTrue(_approx(c.center.y, -4))
        self.assertTrue(_approx(c.radius, 5))

    def test_circle_scale_uniform(self) -> None:
        c = Circle(Point(1, 1, name="C"), 3)
        c.scale(2, 2, 0, 0)
        self.assertTrue(_approx(c.center.x, 2))
        self.assertTrue(_approx(c.center.y, 2))
        self.assertTrue(_approx(c.radius, 6))

    def test_circle_scale_negative_uniform(self) -> None:
        """Negative uniform factor should use abs for radius."""
        c = Circle(Point(1, 1, name="C"), 3)
        c.scale(-2, -2, 0, 0)
        self.assertTrue(_approx(c.radius, 6))

    def test_circle_scale_nonuniform_raises(self) -> None:
        c = Circle(Point(0, 0, name="C"), 5)
        with self.assertRaises(ValueError):
            c.scale(2, 3, 0, 0)

    def test_circle_shear_raises(self) -> None:
        c = Circle(Point(0, 0, name="C"), 5)
        with self.assertRaises(ValueError):
            c.shear("horizontal", 1, 0, 0)

    def test_circle_scale_zero_raises(self) -> None:
        c = Circle(Point(0, 0, name="C"), 5)
        with self.assertRaises(ValueError):
            c.scale(0, 0, 0, 0)

    def test_circle_rotate_around(self) -> None:
        c = Circle(Point(2, 0, name="C"), 3)
        c.rotate_around(90, 0, 0)
        self.assertTrue(_approx(c.center.x, 0))
        self.assertTrue(_approx(c.center.y, 2))
        self.assertTrue(_approx(c.radius, 3))

    # ---------------------------------------------------------------
    # Ellipse transforms
    # ---------------------------------------------------------------
    def test_ellipse_reflect_x_axis(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 5, 3, rotation_angle=30)
        e.reflect("x_axis")
        self.assertTrue(_approx(e.rotation_angle, 330))

    def test_ellipse_reflect_y_axis(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 5, 3, rotation_angle=30)
        e.reflect("y_axis")
        self.assertTrue(_approx(e.rotation_angle, 150))

    def test_ellipse_scale_uniform(self) -> None:
        e = Ellipse(Point(1, 1, name="E"), 4, 2)
        e.scale(3, 3, 0, 0)
        self.assertTrue(_approx(e.radius_x, 12))
        self.assertTrue(_approx(e.radius_y, 6))

    def test_ellipse_scale_nonuniform_axis_aligned(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 4, 2, rotation_angle=0)
        e.scale(2, 3, 0, 0)
        self.assertTrue(_approx(e.radius_x, 8))
        self.assertTrue(_approx(e.radius_y, 6))

    def test_ellipse_scale_nonuniform_rotated_raises(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 4, 2, rotation_angle=45)
        with self.assertRaises(ValueError):
            e.scale(2, 3, 0, 0)

    def test_ellipse_shear_raises(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 4, 2)
        with self.assertRaises(ValueError):
            e.shear("horizontal", 1, 0, 0)

    def test_ellipse_rotate_around(self) -> None:
        e = Ellipse(Point(2, 0, name="E"), 4, 2, rotation_angle=0)
        e.rotate_around(90, 0, 0)
        self.assertTrue(_approx(e.center.x, 0))
        self.assertTrue(_approx(e.center.y, 2))
        self.assertTrue(_approx(e.rotation_angle, 90))

    # ---------------------------------------------------------------
    # Manager-level: reflect_object
    # ---------------------------------------------------------------
    def test_manager_reflect_triangle(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)

        canvas, _, renderer = _build_canvas(tri, [s1, s2, s3])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object(tri.name, "x_axis")

        self.assertTrue(_approx(p3.y, -3))
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    def test_manager_reflect_via_segment(self) -> None:
        """Reflect a point across a segment acting as the axis."""
        p = Point(3, 4, name="P")
        # Segment along x-axis
        sp1 = Point(0, 0, name="S1")
        sp2 = Point(1, 0, name="S2")
        seg = Segment(sp1, sp2)

        canvas, _, _ = _build_canvas(p, [], extra_drawables=[seg])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "segment", segment_name=seg.name)

        # Reflection across y=0 line: y negated
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, -4))

    def test_manager_reflect_zero_length_segment_raises(self) -> None:
        p = Point(1, 1, name="P")
        sp = Point(0, 0, name="S")
        seg = Segment(sp, Point(0, 0, name="S2"))

        canvas, _, _ = _build_canvas(p, [], extra_drawables=[seg])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.reflect_object("P", "segment", segment_name=seg.name)

    # ---------------------------------------------------------------
    # Manager-level: scale_object
    # ---------------------------------------------------------------
    def test_manager_scale_circle_uniform(self) -> None:
        center = Point(0, 0, name="C")
        circle = Circle(center, 5)

        canvas, _, renderer = _build_canvas(circle, [])
        mgr = TransformationsManager(canvas)
        mgr.scale_object(circle.name, 2, 2, 0, 0)

        self.assertTrue(_approx(circle.radius, 10))
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    def test_manager_scale_circle_nonuniform_raises(self) -> None:
        center = Point(0, 0, name="C")
        circle = Circle(center, 5)

        canvas, _, _ = _build_canvas(circle, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.scale_object(circle.name, 2, 3, 0, 0)

    # ---------------------------------------------------------------
    # Manager-level: shear_object
    # ---------------------------------------------------------------
    def test_manager_shear_rectangle(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(4, 3, name="C")
        p4 = Point(0, 3, name="D")
        s1 = Segment(p1, p2)
        s2 = Segment(p2, p3)
        s3 = Segment(p3, p4)
        s4 = Segment(p4, p1)
        rect = Rectangle(s1, s2, s3, s4)

        canvas, _, renderer = _build_canvas(rect, [s1, s2, s3, s4])
        mgr = TransformationsManager(canvas)
        mgr.shear_object(rect.name, "horizontal", 0.5, 0, 0)

        # p3: x = 4 + 0.5 * 3 = 5.5
        self.assertTrue(_approx(p3.x, 5.5))
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    # ---------------------------------------------------------------
    # Manager-level: rotate_object with arbitrary center
    # ---------------------------------------------------------------
    def test_manager_rotate_point_around_center(self) -> None:
        p = Point(1, 0, name="P")

        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.rotate_object("P", 90, center_x=0, center_y=0)

        self.assertTrue(_approx(p.x, 0))
        self.assertTrue(_approx(p.y, 1))

    def test_manager_rotate_one_center_coord_raises(self) -> None:
        p = Point(1, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.rotate_object("P", 90, center_x=0, center_y=None)

    # ---------------------------------------------------------------
    # Manager-level: missing drawable raises
    # ---------------------------------------------------------------
    def test_manager_missing_drawable_raises(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.reflect_object("NONEXISTENT", "x_axis")

    # ---------------------------------------------------------------
    # Manager-level: undo archiving called
    # ---------------------------------------------------------------
    def test_manager_archive_called_before_reflect(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "x_axis")

        self.assertTrue(canvas.undo_redo_manager.archive.calls)

    def test_manager_archive_called_before_scale(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.scale_object("P", 2, 2, 0, 0)

        self.assertTrue(canvas.undo_redo_manager.archive.calls)

    def test_manager_archive_called_before_shear(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.shear_object("P", "horizontal", 1, 0, 0)

        self.assertTrue(canvas.undo_redo_manager.archive.calls)
