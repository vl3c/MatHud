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

    # ---------------------------------------------------------------
    # Manager-level: no archive on unsupported transforms
    # ---------------------------------------------------------------
    def test_manager_no_archive_on_circle_shear(self) -> None:
        """Shearing a circle should fail before archiving undo state."""
        c = Circle(Point(0, 0, name="C"), 5)
        canvas, _, _ = _build_canvas(c, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.shear_object(c.name, "horizontal", 1, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_no_archive_on_ellipse_shear(self) -> None:
        """Shearing an ellipse should fail before archiving undo state."""
        e = Ellipse(Point(0, 0, name="E"), 4, 2)
        canvas, _, _ = _build_canvas(e, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.shear_object(e.name, "horizontal", 1, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_no_archive_on_circle_nonuniform_scale(self) -> None:
        """Non-uniform scaling a circle should fail before archiving undo state."""
        c = Circle(Point(0, 0, name="C"), 5)
        canvas, _, _ = _build_canvas(c, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.scale_object(c.name, 2, 3, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_no_archive_on_rotated_ellipse_nonuniform_scale(self) -> None:
        """Non-uniform scaling a rotated ellipse should fail before archiving."""
        e = Ellipse(Point(0, 0, name="E"), 4, 2, rotation_angle=45)
        canvas, _, _ = _build_canvas(e, [])
        mgr = TransformationsManager(canvas)

        with self.assertRaises(ValueError):
            mgr.scale_object(e.name, 2, 3, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    # ===================================================================
    # Additional edge-case tests
    # ===================================================================

    # ---------------------------------------------------------------
    # Point: reflect across general line with non-zero c
    # ---------------------------------------------------------------
    def test_point_reflect_line_with_offset(self) -> None:
        """Reflect across 0x + 1y - 2 = 0  (the line y = 2)."""
        p = Point(5, 0, name="P")
        p.reflect("line", a=0, b=1, c=-2)
        self.assertTrue(_approx(p.x, 5))
        self.assertTrue(_approx(p.y, 4))

    def test_point_reflect_general_line(self) -> None:
        """Reflect (1, 1) across 1x + 1y + 0 = 0  (the line x + y = 0)."""
        p = Point(1, 1, name="P")
        p.reflect("line", a=1, b=1, c=0)
        self.assertTrue(_approx(p.x, -1))
        self.assertTrue(_approx(p.y, -1))

    # ---------------------------------------------------------------
    # Point: identity transforms
    # ---------------------------------------------------------------
    def test_point_scale_identity(self) -> None:
        """Scale by 1 from any center is a no-op."""
        p = Point(7, -3, name="P")
        p.scale(1, 1, 99, 99)
        self.assertTrue(_approx(p.x, 7))
        self.assertTrue(_approx(p.y, -3))

    def test_point_shear_zero_factor(self) -> None:
        """Shear with factor 0 is a no-op."""
        p = Point(4, 5, name="P")
        p.shear("horizontal", 0, 0, 0)
        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 5))

    def test_point_rotate_360_identity(self) -> None:
        """Full rotation returns to the original position."""
        p = Point(3, 7, name="P")
        p.rotate_around(360, 1, 1)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 7))

    def test_point_rotate_negative_angle(self) -> None:
        """Negative angle rotates clockwise."""
        p = Point(0, 1, name="P")
        p.rotate_around(-90, 0, 0)
        self.assertTrue(_approx(p.x, 1))
        self.assertTrue(_approx(p.y, 0))

    def test_point_scale_negative_mirror(self) -> None:
        """Negative scale factor mirrors through center."""
        p = Point(3, 4, name="P")
        p.scale(-1, -1, 0, 0)
        self.assertTrue(_approx(p.x, -3))
        self.assertTrue(_approx(p.y, -4))

    # ---------------------------------------------------------------
    # Point: shear from non-origin center
    # ---------------------------------------------------------------
    def test_point_shear_horizontal_nonorigin_center(self) -> None:
        """Shear horizontally from a center that is not the origin."""
        p = Point(3, 5, name="P")
        p.shear("horizontal", 1.0, 1, 2)
        # dx = 3-1 = 2, dy = 5-2 = 3 => new x = 1 + 2 + 1*3 = 6, y = 2 + 3 = 5
        self.assertTrue(_approx(p.x, 6))
        self.assertTrue(_approx(p.y, 5))

    def test_point_shear_vertical_nonorigin_center(self) -> None:
        """Shear vertically from a center that is not the origin."""
        p = Point(5, 1, name="P")
        p.shear("vertical", 2.0, 2, 0)
        # dx = 5-2 = 3, dy = 1-0 = 1 => x = 2+3 = 5, y = 0 + 1 + 2*3 = 7
        self.assertTrue(_approx(p.x, 5))
        self.assertTrue(_approx(p.y, 7))

    # ---------------------------------------------------------------
    # Segment: reflect across y_axis and arbitrary line
    # ---------------------------------------------------------------
    def test_segment_reflect_y_axis(self) -> None:
        p1 = Point(1, 2, name="A")
        p2 = Point(3, 4, name="B")
        s = Segment(p1, p2)
        s.reflect("y_axis")
        self.assertTrue(_approx(p1.x, -1))
        self.assertTrue(_approx(p2.x, -3))
        self.assertTrue(_approx(p1.y, 2))
        self.assertTrue(_approx(p2.y, 4))

    def test_segment_reflect_arbitrary_line(self) -> None:
        """Reflect a segment across y = x."""
        p1 = Point(0, 2, name="A")
        p2 = Point(4, 0, name="B")
        s = Segment(p1, p2)
        s.reflect("line", a=1, b=-1, c=0)
        self.assertTrue(_approx(p1.x, 2))
        self.assertTrue(_approx(p1.y, 0))
        self.assertTrue(_approx(p2.x, 0))
        self.assertTrue(_approx(p2.y, 4))

    def test_segment_shear_with_nonzero_dy(self) -> None:
        """Shear where dy != 0 so shear has a visible effect."""
        p1 = Point(0, 0, name="A")
        p2 = Point(2, 3, name="B")
        s = Segment(p1, p2)
        s.shear("horizontal", 1, 0, 0)
        # p1 unchanged (dy=0), p2: x = 2 + 1*3 = 5, y = 3
        self.assertTrue(_approx(p1.x, 0))
        self.assertTrue(_approx(p2.x, 5))
        self.assertTrue(_approx(p2.y, 3))

    def test_segment_scale_changes_line_formula(self) -> None:
        """Scaling a non-origin segment should change its line_formula."""
        p1 = Point(1, 0, name="A")
        p2 = Point(1, 2, name="B")
        s = Segment(p1, p2)
        formula_before = s.line_formula
        s.scale(2, 1, 0, 0)
        # After scale: A=(2,0), B=(2,2) — still vertical x=2 but different intercept
        self.assertIsNotNone(s.line_formula)
        self.assertNotEqual(s.line_formula, formula_before)

    # ---------------------------------------------------------------
    # Vector: shear and rotate_around
    # ---------------------------------------------------------------
    def test_vector_shear(self) -> None:
        v = Vector(Point(0, 0, name="O"), Point(2, 3, name="T"))
        v.shear("horizontal", 1, 0, 0)
        self.assertTrue(_approx(v.tip.x, 5))  # 2 + 1*3
        self.assertTrue(_approx(v.tip.y, 3))

    def test_vector_rotate_around(self) -> None:
        v = Vector(Point(1, 0, name="O"), Point(3, 0, name="T"))
        v.rotate_around(90, 0, 0)
        self.assertTrue(_approx(v.origin.x, 0))
        self.assertTrue(_approx(v.origin.y, 1))
        self.assertTrue(_approx(v.tip.x, 0))
        self.assertTrue(_approx(v.tip.y, 3))

    # ---------------------------------------------------------------
    # Rectangle: drawable-level transforms
    # ---------------------------------------------------------------
    def test_rectangle_reflect_y_axis(self) -> None:
        p1 = Point(1, 0, name="A")
        p2 = Point(5, 0, name="B")
        p3 = Point(5, 3, name="C")
        p4 = Point(1, 3, name="D")
        s1, s2, s3, s4 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p4), Segment(p4, p1)
        rect = Rectangle(s1, s2, s3, s4)
        rect.reflect("y_axis")
        self.assertTrue(_approx(p1.x, -1))
        self.assertTrue(_approx(p2.x, -5))
        self.assertTrue(_approx(p3.x, -5))
        self.assertTrue(_approx(p4.x, -1))

    def test_rectangle_scale_nonuniform(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(4, 2, name="C")
        p4 = Point(0, 2, name="D")
        s1, s2, s3, s4 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p4), Segment(p4, p1)
        rect = Rectangle(s1, s2, s3, s4)
        rect.scale(2, 3, 0, 0)
        self.assertTrue(_approx(p2.x, 8))
        self.assertTrue(_approx(p3.y, 6))

    def test_rectangle_rotate_around(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(2, 0, name="B")
        p3 = Point(2, 1, name="C")
        p4 = Point(0, 1, name="D")
        s1, s2, s3, s4 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p4), Segment(p4, p1)
        rect = Rectangle(s1, s2, s3, s4)
        rect.rotate_around(180, 1, 0.5)
        # Each vertex should be mirrored through center (1, 0.5)
        self.assertTrue(_approx(p1.x, 2, tol=1e-9))
        self.assertTrue(_approx(p1.y, 1, tol=1e-9))
        self.assertTrue(_approx(p2.x, 0, tol=1e-9))
        self.assertTrue(_approx(p2.y, 1, tol=1e-9))

    # ---------------------------------------------------------------
    # Circle: additional reflect axes
    # ---------------------------------------------------------------
    def test_circle_reflect_y_axis(self) -> None:
        c = Circle(Point(3, 4, name="C"), 5)
        c.reflect("y_axis")
        self.assertTrue(_approx(c.center.x, -3))
        self.assertTrue(_approx(c.center.y, 4))
        self.assertTrue(_approx(c.radius, 5))

    def test_circle_reflect_line(self) -> None:
        """Reflect circle across the line y = x."""
        c = Circle(Point(3, 1, name="C"), 2)
        c.reflect("line", a=1, b=-1, c=0)
        self.assertTrue(_approx(c.center.x, 1))
        self.assertTrue(_approx(c.center.y, 3))
        self.assertTrue(_approx(c.radius, 2))

    def test_circle_scale_from_nonorigin(self) -> None:
        """Scale from a center that is not the origin."""
        c = Circle(Point(4, 0, name="C"), 3)
        c.scale(2, 2, 2, 0)
        # center: 2 + 2*(4-2) = 6; radius: 3*2 = 6
        self.assertTrue(_approx(c.center.x, 6))
        self.assertTrue(_approx(c.radius, 6))

    # ---------------------------------------------------------------
    # Ellipse: additional edges
    # ---------------------------------------------------------------
    def test_ellipse_reflect_line(self) -> None:
        """Reflect across y = x (a=1, b=-1, c=0)."""
        e = Ellipse(Point(2, 0, name="E"), 5, 3, rotation_angle=0)
        e.reflect("line", a=1, b=-1, c=0)
        # Center: (2,0) -> (0,2)
        self.assertTrue(_approx(e.center.x, 0))
        self.assertTrue(_approx(e.center.y, 2))
        # line angle = atan2(-1, -1) = -135 deg -> mapped to 225 or we just verify it changed
        # For rotation_angle=0, new = (2*line_angle - 0) % 360
        line_angle_deg = math.degrees(math.atan2(-1, -1))
        expected_rot = (2 * line_angle_deg - 0) % 360
        self.assertTrue(_approx(e.rotation_angle, expected_rot))

    def test_ellipse_scale_zero_raises(self) -> None:
        e = Ellipse(Point(0, 0, name="E"), 4, 2)
        with self.assertRaises(ValueError):
            e.scale(0, 0, 0, 0)

    def test_ellipse_scale_negative_uniform(self) -> None:
        """Negative uniform factor should use abs for radii."""
        e = Ellipse(Point(1, 1, name="E"), 4, 2)
        e.scale(-2, -2, 0, 0)
        self.assertTrue(_approx(e.radius_x, 8))
        self.assertTrue(_approx(e.radius_y, 4))

    def test_ellipse_scale_nonuniform_on_180_aligned(self) -> None:
        """rotation_angle=180 is axis-aligned (180 % 180 == 0), should succeed."""
        e = Ellipse(Point(0, 0, name="E"), 4, 2, rotation_angle=180)
        e.scale(2, 3, 0, 0)
        self.assertTrue(_approx(e.radius_x, 8))
        self.assertTrue(_approx(e.radius_y, 6))

    def test_ellipse_rotate_around_accumulates(self) -> None:
        """Two successive rotations should accumulate correctly."""
        e = Ellipse(Point(2, 0, name="E"), 4, 2, rotation_angle=10)
        e.rotate_around(45, 0, 0)
        e.rotate_around(45, 0, 0)
        self.assertTrue(_approx(e.rotation_angle, 100))

    def test_ellipse_reflect_degenerate_line(self) -> None:
        """Degenerate line coefficients should leave rotation_angle unchanged."""
        e = Ellipse(Point(0, 0, name="E"), 4, 2, rotation_angle=30)
        e.reflect("line", a=0, b=0, c=0)
        # center unchanged (degenerate), rotation_angle unchanged
        self.assertTrue(_approx(e.rotation_angle, 30))

    # ---------------------------------------------------------------
    # Manager: reflect_object with axis='line'
    # ---------------------------------------------------------------
    def test_manager_reflect_line_axis(self) -> None:
        """Reflect a point across y = 0 via line coefficients (0x + 1y + 0 = 0)."""
        p = Point(3, 4, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "line", line_a=0, line_b=1, line_c=0)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, -4))

    def test_manager_reflect_invalid_axis_raises(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("P", "diagonal")

    def test_manager_reflect_line_zero_coefficients_raises(self) -> None:
        """axis='line' with a=b=0 is degenerate and should raise."""
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("P", "line", line_a=0, line_b=0, line_c=0)

    def test_manager_reflect_segment_not_found_raises(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("P", "segment", segment_name="NONEXISTENT")

    def test_manager_reflect_empty_segment_name_raises(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("P", "segment", segment_name="")

    # ---------------------------------------------------------------
    # Manager: scale_object edge cases
    # ---------------------------------------------------------------
    def test_manager_scale_zero_factor_raises(self) -> None:
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.scale_object("P", 0, 0, 0, 0)

    def test_manager_scale_one_zero_factor_raises(self) -> None:
        """Even if only one factor is zero, it should raise."""
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.scale_object("P", 2, 0, 0, 0)

    # ---------------------------------------------------------------
    # Manager: shear_object edge cases
    # ---------------------------------------------------------------
    def test_manager_shear_invalid_axis_raises(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.shear_object("P", "diagonal", 1, 0, 0)

    def test_manager_shear_vertical(self) -> None:
        """Shear a point vertically via manager."""
        p = Point(3, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        mgr.shear_object("P", "vertical", 2, 0, 0)
        # y = 0 + (1-0) + 2*(3-0) = 7
        self.assertTrue(_approx(p.y, 7))

    # ---------------------------------------------------------------
    # Manager: rotate_object default (no center)
    # ---------------------------------------------------------------
    def test_manager_rotate_segment_no_center(self) -> None:
        """Rotate a segment around its own center (original behaviour)."""
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        s = Segment(p1, p2)
        canvas, _, _ = _build_canvas(s, [s])
        mgr = TransformationsManager(canvas)
        mgr.rotate_object(s.name, 90)
        # Segment rotates around midpoint (2, 0); endpoints move
        self.assertTrue(_approx(p1.x, 2, tol=1e-6))
        self.assertTrue(_approx(p1.y, -2, tol=1e-6))

    def test_manager_rotate_circle_around_center(self) -> None:
        """Circle can rotate around arbitrary center at manager level."""
        c = Circle(Point(3, 0, name="C"), 2)
        canvas, _, renderer = _build_canvas(c, [])
        mgr = TransformationsManager(canvas)
        mgr.rotate_object(c.name, 90, center_x=0, center_y=0)
        self.assertTrue(_approx(c.center.x, 0))
        self.assertTrue(_approx(c.center.y, 3))
        self.assertTrue(_approx(c.radius, 2))

    # ---------------------------------------------------------------
    # Manager: reflect on circle and ellipse
    # ---------------------------------------------------------------
    def test_manager_reflect_circle(self) -> None:
        c = Circle(Point(3, 4, name="C"), 5)
        canvas, _, renderer = _build_canvas(c, [])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object(c.name, "y_axis")
        self.assertTrue(_approx(c.center.x, -3))
        self.assertTrue(_approx(c.radius, 5))
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    def test_manager_reflect_ellipse(self) -> None:
        e = Ellipse(Point(2, 0, name="E"), 5, 3, rotation_angle=30)
        canvas, _, renderer = _build_canvas(e, [])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object(e.name, "x_axis")
        self.assertTrue(_approx(e.rotation_angle, 330))
        self.assertTrue(renderer.invalidate_drawable_cache.calls)

    def test_manager_scale_ellipse_uniform(self) -> None:
        e = Ellipse(Point(1, 1, name="E"), 4, 2)
        canvas, _, renderer = _build_canvas(e, [])
        mgr = TransformationsManager(canvas)
        mgr.scale_object(e.name, 3, 3, 0, 0)
        self.assertTrue(_approx(e.radius_x, 12))
        self.assertTrue(_approx(e.radius_y, 6))

    # ---------------------------------------------------------------
    # Manager: excluded types properly rejected
    # ---------------------------------------------------------------
    def test_manager_excluded_type_raises(self) -> None:
        """Transform on an excluded type (e.g. Function) should raise ValueError."""
        func_mock = SimpleMock()
        func_mock.name = "f1"
        func_mock.get_class_name = SimpleMock(return_value="Function")
        canvas, _, _ = _build_canvas(func_mock, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("f1", "x_axis")

    def test_manager_excluded_type_scale_raises(self) -> None:
        graph_mock = SimpleMock()
        graph_mock.name = "G1"
        graph_mock.get_class_name = SimpleMock(return_value="Graph")
        canvas, _, _ = _build_canvas(graph_mock, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.scale_object("G1", 2, 2, 0, 0)

    def test_manager_excluded_type_shear_raises(self) -> None:
        angle_mock = SimpleMock()
        angle_mock.name = "ang1"
        angle_mock.get_class_name = SimpleMock(return_value="Angle")
        canvas, _, _ = _build_canvas(angle_mock, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.shear_object("ang1", "horizontal", 1, 0, 0)

    # ---------------------------------------------------------------
    # Manager: no archive on missing drawable
    # ---------------------------------------------------------------
    def test_manager_no_archive_on_missing_reflect(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.reflect_object("GONE", "x_axis")
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_no_archive_on_missing_scale(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.scale_object("GONE", 2, 2, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_no_archive_on_missing_shear(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.shear_object("GONE", "horizontal", 1, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    # ---------------------------------------------------------------
    # Mathematical invariants: double-reflect, scale+inverse, 360
    # ---------------------------------------------------------------
    def test_invariant_double_reflect_x_identity(self) -> None:
        """Reflecting across x-axis twice returns to original."""
        p = Point(3, 7, name="P")
        p.reflect("x_axis")
        p.reflect("x_axis")
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 7))

    def test_invariant_double_reflect_y_identity(self) -> None:
        p = Point(-2, 5, name="P")
        p.reflect("y_axis")
        p.reflect("y_axis")
        self.assertTrue(_approx(p.x, -2))
        self.assertTrue(_approx(p.y, 5))

    def test_invariant_double_reflect_line_identity(self) -> None:
        """Reflecting across an arbitrary line twice returns to original."""
        p = Point(3, 7, name="P")
        p.reflect("line", a=2, b=-3, c=1)
        p.reflect("line", a=2, b=-3, c=1)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 7))

    def test_invariant_scale_inverse_identity(self) -> None:
        """Scale by k then by 1/k returns to original."""
        p = Point(5, -3, name="P")
        p.scale(3, 2, 1, 1)
        p.scale(1 / 3, 1 / 2, 1, 1)
        self.assertTrue(_approx(p.x, 5))
        self.assertTrue(_approx(p.y, -3))

    def test_invariant_rotate_360_identity(self) -> None:
        """Rotating 360 degrees returns to original."""
        p = Point(4, 5, name="P")
        p.rotate_around(360, 2, 3)
        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 5))

    def test_invariant_rotate_four_90s_identity(self) -> None:
        """Four 90-degree rotations return to original."""
        p = Point(4, 5, name="P")
        for _ in range(4):
            p.rotate_around(90, 2, 3)
        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 5))

    def test_invariant_shear_inverse_identity(self) -> None:
        """Shear by k then by -k returns to original."""
        p = Point(3, 7, name="P")
        p.shear("horizontal", 2.5, 1, 1)
        p.shear("horizontal", -2.5, 1, 1)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 7))

    # ---------------------------------------------------------------
    # Composition: sequential transforms on a triangle
    # ---------------------------------------------------------------
    def test_composition_reflect_then_scale(self) -> None:
        """Reflect across x-axis then scale by 2 from origin."""
        p1 = Point(0, 0, name="A")
        p2 = Point(3, 0, name="B")
        p3 = Point(0, 4, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        tri.reflect("x_axis")
        tri.scale(2, 2, 0, 0)
        # C: (0,4) -> (0,-4) -> (0,-8)
        self.assertTrue(_approx(p3.x, 0))
        self.assertTrue(_approx(p3.y, -8))
        # B: (3,0) -> (3,0) -> (6,0)
        self.assertTrue(_approx(p2.x, 6))

    def test_composition_shear_then_rotate(self) -> None:
        """Shear a point (with non-zero dy) then rotate it."""
        p = Point(1, 2, name="P")
        p.shear("horizontal", 1, 0, 0)  # x = 1 + 1*2 = 3, y = 2
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 2))
        p.rotate_around(90, 0, 0)  # (3, 2) -> (-2, 3)
        self.assertTrue(_approx(p.x, -2))
        self.assertTrue(_approx(p.y, 3))

    def test_composition_scale_circle_then_reflect(self) -> None:
        """Scale a circle then reflect it."""
        c = Circle(Point(2, 3, name="C"), 5)
        c.scale(2, 2, 0, 0)
        c.reflect("x_axis")
        self.assertTrue(_approx(c.center.x, 4))
        self.assertTrue(_approx(c.center.y, -6))
        self.assertTrue(_approx(c.radius, 10))

    def test_composition_ellipse_rotate_then_reflect(self) -> None:
        """Rotate an ellipse around external center, then reflect across y-axis."""
        e = Ellipse(Point(3, 0, name="E"), 5, 2, rotation_angle=0)
        e.rotate_around(90, 0, 0)
        # center: (0, 3), rotation_angle: 90
        self.assertTrue(_approx(e.center.x, 0))
        self.assertTrue(_approx(e.center.y, 3))
        self.assertTrue(_approx(e.rotation_angle, 90))
        e.reflect("y_axis")
        # center: (0, 3), rotation_angle: (180 - 90) % 360 = 90
        self.assertTrue(_approx(e.center.x, 0))
        self.assertTrue(_approx(e.rotation_angle, 90))

    # ---------------------------------------------------------------
    # Manager: redraw called on success
    # ---------------------------------------------------------------
    def test_manager_redraw_called_when_enabled(self) -> None:
        """When draw_enabled is True, draw() should be called after transform."""
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        canvas.draw_enabled = True
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "x_axis")
        self.assertTrue(canvas.draw.calls)

    def test_manager_no_redraw_when_disabled(self) -> None:
        """When draw_enabled is False, draw() should not be called."""
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        canvas.draw_enabled = False
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "x_axis")
        self.assertFalse(canvas.draw.calls)

    # ---------------------------------------------------------------
    # Manager: segment formula refresh after polygon transform
    # ---------------------------------------------------------------
    def test_manager_segment_formulas_refreshed_after_reflect(self) -> None:
        """Reflecting a triangle should recalculate its segment line formulas."""
        p1 = Point(0, 0, name="A")
        p2 = Point(4, 0, name="B")
        p3 = Point(2, 3, name="C")
        s1, s2, s3 = Segment(p1, p2), Segment(p2, p3), Segment(p3, p1)
        tri = Triangle(s1, s2, s3)
        formula_before = s2.line_formula

        canvas, dep_mgr, _ = _build_canvas(tri, [s1, s2, s3])
        # Register segments as children of triangle for dependency refresh
        for seg in [s1, s2, s3]:
            dep_mgr.register(tri, seg)

        mgr = TransformationsManager(canvas)
        mgr.reflect_object(tri.name, "x_axis")

        # p3 moved to (2, -3), so s2 (B-C) formula must have changed
        self.assertNotEqual(s2.line_formula, formula_before)

    # ===================================================================
    # Codex-review edge cases
    # ===================================================================

    # ---------------------------------------------------------------
    # Circle: negative scale also moves center
    # ---------------------------------------------------------------
    def test_circle_scale_negative_moves_center(self) -> None:
        """Negative uniform scale mirrors center through scaling center."""
        c = Circle(Point(4, 1, name="C"), 3)
        c.scale(-2, -2, 1, 1)
        # center: 1 + (-2)*(4-1) = -5, 1 + (-2)*(1-1) = 1
        self.assertTrue(_approx(c.center.x, -5))
        self.assertTrue(_approx(c.center.y, 1))
        self.assertTrue(_approx(c.radius, 6))

    # ---------------------------------------------------------------
    # Ellipse: rotation_angle modulo wrap at 360 boundary
    # ---------------------------------------------------------------
    def test_ellipse_rotate_around_wraps_modulo(self) -> None:
        """350 + 20 = 370 should wrap to 10."""
        e = Ellipse(Point(2, 3, name="E"), 4, 2, rotation_angle=350)
        e.rotate_around(20, 2, 3)
        # Center unchanged (rotating around self), angle wraps
        self.assertTrue(_approx(e.center.x, 2))
        self.assertTrue(_approx(e.center.y, 3))
        self.assertTrue(_approx(e.rotation_angle, 10))

    # ---------------------------------------------------------------
    # Ellipse: reflection across offset line (non-zero c)
    # ---------------------------------------------------------------
    def test_ellipse_reflect_offset_line(self) -> None:
        """Reflect ellipse across x - y + 1 = 0 (a=1, b=-1, c=1)."""
        e = Ellipse(Point(2, 0, name="E"), 5, 3, rotation_angle=30)
        e.reflect("line", a=1, b=-1, c=1)
        # Center reflection: dot = 1*2 + (-1)*0 + 1 = 3; denom = 2
        # new_x = 2 - 2*1*3/2 = -1, new_y = 0 - 2*(-1)*3/2 = 3
        self.assertTrue(_approx(e.center.x, -1))
        self.assertTrue(_approx(e.center.y, 3))
        # line_angle_deg = atan2(-1, -1) = -135 degrees
        # new rotation = (2*(-135) - 30) % 360 = (-300) % 360 = 60
        self.assertTrue(_approx(e.rotation_angle, 60))

    # ---------------------------------------------------------------
    # Point on reflection axis is a fixed point
    # ---------------------------------------------------------------
    def test_point_on_segment_axis_is_fixed(self) -> None:
        """A point lying on the reflection segment should not move."""
        # Segment from (0,0) to (2,1) defines line y = x/2 => 1x - 2y + 0 = 0
        sp1 = Point(0, 0, name="S1")
        sp2 = Point(2, 1, name="S2")
        seg = Segment(sp1, sp2)
        # Point on the same line
        p = Point(4, 2, name="P")

        canvas, _, _ = _build_canvas(p, [], extra_drawables=[seg])
        mgr = TransformationsManager(canvas)
        mgr.reflect_object("P", "segment", segment_name=seg.name)

        self.assertTrue(_approx(p.x, 4))
        self.assertTrue(_approx(p.y, 2))

    # ---------------------------------------------------------------
    # Scale factor threshold boundary (1e-18)
    # ---------------------------------------------------------------
    def test_manager_scale_just_below_threshold_raises(self) -> None:
        """Scale factor abs < 1e-18 should raise."""
        p = Point(2, 3, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.scale_object("P", 1e-19, 1, 0, 0)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    def test_manager_scale_at_threshold_succeeds(self) -> None:
        """Scale factor abs == 1e-18 should not raise (boundary is strict <)."""
        p = Point(2, 3, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        # 1e-18 is exactly at the boundary — abs(1e-18) < 1e-18 is False
        mgr.scale_object("P", 1e-18, 1e-18, 0, 0)
        # Should succeed (extreme but valid)
        self.assertTrue(canvas.undo_redo_manager.archive.calls)

    # ---------------------------------------------------------------
    # Point: unknown axis silently no-ops
    # ---------------------------------------------------------------
    def test_point_reflect_unknown_axis_no_op(self) -> None:
        """An unrecognised axis string should not move the point."""
        p = Point(3, 4, name="P")
        p.reflect("unknown_axis")
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 4))

    def test_point_shear_unknown_axis_no_op(self) -> None:
        """An unrecognised shear axis should not move the point."""
        p = Point(3, 4, name="P")
        p.shear("diagonal", 1, 0, 0)
        self.assertTrue(_approx(p.x, 3))
        self.assertTrue(_approx(p.y, 4))

    # ---------------------------------------------------------------
    # Manager: rotate_object excludes Point/Circle with no center
    # ---------------------------------------------------------------
    def test_manager_rotate_point_no_center_raises(self) -> None:
        """Rotating a point without center is a no-op exclusion."""
        p = Point(1, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.rotate_object("P", 90)

    def test_manager_rotate_circle_no_center_raises(self) -> None:
        """Rotating a circle without center is a no-op exclusion."""
        c = Circle(Point(3, 0, name="C"), 2)
        canvas, _, _ = _build_canvas(c, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.rotate_object(c.name, 45)

    def test_manager_rotate_no_center_no_archive(self) -> None:
        """Excluded rotate should not archive (ValueError before archive)."""
        p = Point(1, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        with self.assertRaises(ValueError):
            mgr.rotate_object("P", 90)
        self.assertFalse(canvas.undo_redo_manager.archive.calls)

    # ---------------------------------------------------------------
    # Manager: return value is True on success
    # ---------------------------------------------------------------
    def test_manager_reflect_returns_true(self) -> None:
        p = Point(0, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        result = mgr.reflect_object("P", "x_axis")
        self.assertTrue(result)

    def test_manager_scale_returns_true(self) -> None:
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        result = mgr.scale_object("P", 2, 2, 0, 0)
        self.assertTrue(result)

    def test_manager_shear_returns_true(self) -> None:
        p = Point(1, 1, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        result = mgr.shear_object("P", "horizontal", 1, 0, 0)
        self.assertTrue(result)

    def test_manager_rotate_returns_true(self) -> None:
        p = Point(1, 0, name="P")
        canvas, _, _ = _build_canvas(p, [])
        mgr = TransformationsManager(canvas)
        result = mgr.rotate_object("P", 90, center_x=0, center_y=0)
        self.assertTrue(result)
