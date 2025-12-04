from __future__ import annotations

import unittest

from typing import Any, Optional

from drawables.circle import Circle
from drawables.ellipse import Ellipse
from drawables.point import Point
from drawables.rectangle import Rectangle
from drawables.triangle import Triangle
from drawables.segment import Segment
from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from drawables.function import Function
from managers.colored_area_manager import ColoredAreaManager
from managers.polygon_type import PolygonType
from managers.drawables_container import DrawablesContainer
from .simple_mock import SimpleMock
from constants import default_area_fill_color, default_area_opacity, default_closed_shape_resolution


class TestColoredAreaManager(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = SimpleMock(
            name="CanvasMock",
            draw_enabled=True,
            draw=SimpleMock(),
            undo_redo_manager=SimpleMock(
                name="UndoRedoMock",
                archive=SimpleMock(),
            ),
        )
        self.canvas._validate_color_and_opacity = SimpleMock()

        self.drawables = DrawablesContainer()
        self.name_generator = SimpleMock(name="NameGeneratorMock")
        self.dependency_manager = SimpleMock(name="DependencyManagerMock")
        self.dependency_manager.analyze_drawable_for_dependencies = SimpleMock(return_value=[])
        self.drawable_manager_proxy = SimpleMock(name="DrawableManagerProxyMock")

        def get_segment_by_name(name: str) -> Optional[Segment]:
            for segment in self.drawables.Segments:
                if getattr(segment, "name", "") == name:
                    return segment
            return None

        def get_circle_by_name(name: str) -> Optional[Circle]:
            for circle in self.drawables.Circles:
                if getattr(circle, "name", "") == name:
                    return circle
            return None

        def get_ellipse_by_name(name: str) -> Optional[Ellipse]:
            for ellipse in self.drawables.Ellipses:
                if getattr(ellipse, "name", "") == name:
                    return ellipse
            return None

        def get_polygon_by_name(name: str, polygon_type: Optional[Any] = None):
            if polygon_type == PolygonType.TRIANGLE:
                for triangle in self.drawables.Triangles:
                    if getattr(triangle, "name", "") == name:
                        return triangle
                return None
            if polygon_type == PolygonType.RECTANGLE:
                for rectangle in self.drawables.Rectangles:
                    if getattr(rectangle, "name", "") == name:
                        return rectangle
                return None
            return None

        def get_function(name: str) -> Optional[Function]:
            for function in self.drawables.Functions:
                if getattr(function, "name", "") == name:
                    return function
            return None

        def create_point(x: float, y: float, name: str = "", **_: Any) -> Point:
            point = Point(x, y, name=name)
            self.drawables.add(point)
            return point

        self.drawable_manager_proxy.get_segment_by_name = get_segment_by_name
        self.drawable_manager_proxy.get_circle_by_name = get_circle_by_name
        self.drawable_manager_proxy.get_ellipse_by_name = get_ellipse_by_name
        self.drawable_manager_proxy.get_polygon_by_name = get_polygon_by_name
        self.drawable_manager_proxy.get_function = get_function
        self.drawable_manager_proxy.create_point = create_point

        self.manager = ColoredAreaManager(
            canvas=self.canvas,
            drawables_container=self.drawables,
            name_generator=self.name_generator,
            dependency_manager=self.dependency_manager,
            drawable_manager_proxy=self.drawable_manager_proxy,
        )

    def _add_functions_area(self) -> FunctionsBoundedColoredArea:
        func = SimpleMock(name="f1", function=lambda x: x)
        area = FunctionsBoundedColoredArea(func, None, left_bound=0.0, right_bound=5.0, color="#abcdef", opacity=0.3)
        self.drawables.add(area)
        return area

    def _add_segments_area(self) -> SegmentsBoundedColoredArea:
        s1 = Segment(Point(0.0, 0.0, "A"), Point(1.0, 1.0, "B"))
        area = SegmentsBoundedColoredArea(s1, None, color="#123456", opacity=0.4)
        self.drawables.add(area)
        return area

    def _add_function_segment_area(self) -> FunctionSegmentBoundedColoredArea:
        func = SimpleMock(name="f2", function=lambda x: 2 * x)
        seg = Segment(Point(0.0, 0.0, "S1"), Point(1.0, 0.0, "S2"))
        area = FunctionSegmentBoundedColoredArea(func, seg, color="#654321", opacity=0.5)
        self.drawables.add(area)
        return area

    def test_update_colored_area_updates_style_and_bounds(self) -> None:
        area = self._add_functions_area()

        result = self.manager.update_colored_area(
            area.name,
            new_color="#ff00ff",
            new_opacity=0.6,
            new_left_bound=1.0,
            new_right_bound=6.0,
        )

        self.assertTrue(result)
        self.assertEqual(area.color, "#ff00ff")
        self.assertEqual(area.opacity, 0.6)
        self.assertEqual(area.left_bound, 1.0)
        self.assertEqual(area.right_bound, 6.0)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()

    def test_update_colored_area_blocks_bounds_for_segment_area(self) -> None:
        area = self._add_segments_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_left_bound=0.0)

    def test_update_colored_area_validates_color(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_color="not-a-color")

    def test_update_colored_area_validates_opacity(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_opacity=2.0)

    def test_update_colored_area_validates_bounds_relation(self) -> None:
        area = self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area(area.name, new_left_bound=10.0)

    def test_update_colored_area_style_only_segments_area(self) -> None:
        area = self._add_segments_area()

        result = self.manager.update_colored_area(area.name, new_color="#999999", new_opacity=0.2)

        self.assertTrue(result)
        self.assertEqual(area.color, "#999999")
        self.assertEqual(area.opacity, 0.2)

    def test_update_colored_area_style_only_function_segment_area(self) -> None:
        area = self._add_function_segment_area()

        result = self.manager.update_colored_area(area.name, new_color="#00ffaa", new_opacity=0.7)

        self.assertTrue(result)
        self.assertEqual(area.color, "#00ffaa")
        self.assertEqual(area.opacity, 0.7)

    def test_update_colored_area_missing_name(self) -> None:
        self._add_functions_area()

        with self.assertRaises(ValueError):
            self.manager.update_colored_area("missing", new_color="#000000")

    def test_create_closed_shape_polygon_from_segments(self) -> None:
        a = Point(0.0, 0.0, "A")
        b = Point(1.0, 0.0, "B")
        c = Point(0.0, 1.0, "C")
        segments = {
            "AB": Segment(a, b),
            "BC": Segment(b, c),
            "CA": Segment(c, a),
        }

        def fake_get_segment(name: str):
            return segments.get(name)

        self.drawable_manager_proxy.get_segment_by_name = fake_get_segment

        area = self.manager.create_region_colored_area(
            polygon_segment_names=["AB", "BC", "CA"],
            color="salmon",
            opacity=0.4,
        )

        self.assertEqual(area.shape_type, "polygon")
        self.assertEqual(len(area.segments), 3)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_polygon_requires_loop(self) -> None:
        a = Point(0.0, 0.0, "A")
        b = Point(1.0, 0.0, "B")
        c = Point(2.0, 1.0, "C")
        d = Point(3.0, 1.5, "D")
        segments = {
            "AB": Segment(a, b),
            "BC": Segment(b, c),
            "CD": Segment(c, d),
        }

        def fake_get_segment(name: str):
            return segments.get(name)

        self.drawable_manager_proxy.get_segment_by_name = fake_get_segment

        with self.assertRaises(ValueError):
            self.manager.create_region_colored_area(
                polygon_segment_names=["AB", "BC", "CD"],
                color="gray",
                opacity=0.3,
            )

    def test_create_closed_shape_circle_segment(self) -> None:
        circle_center = Point(0.0, 0.0, "O")
        circle = Circle(circle_center, 5.0)
        chord = Segment(Point(5.0, 0.0, "P"), Point(-5.0, 0.0, "Q"))

        self.drawable_manager_proxy.get_circle_by_name = lambda name: circle if name == circle.name else None

        def fake_get_segment(name: str):
            return chord if name == chord.name else None

        self.drawable_manager_proxy.get_segment_by_name = fake_get_segment

        area = self.manager.create_region_colored_area(
            circle_name=circle.name,
            chord_segment_name=chord.name,
            color="gold",
            opacity=0.5,
        )

        self.assertEqual(area.shape_type, "circle_segment")
        self.assertIs(area.circle, circle)
        self.assertEqual(area.chord_segment, chord)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_full_circle(self) -> None:
        circle_center = Point(0.0, 0.0, "O")
        circle = Circle(circle_center, 4.0)
        self.drawable_manager_proxy.get_circle_by_name = lambda name: circle if name == circle.name else None

        area = self.manager.create_region_colored_area(
            circle_name=circle.name,
            color="skyblue",
            opacity=0.4,
        )

        self.assertEqual(area.shape_type, "circle")
        self.assertIs(area.circle, circle)
        self.assertIsNone(area.chord_segment)
        self.assertEqual(area.color, "skyblue")
        self.assertEqual(area.opacity, 0.4)
        self.assertEqual(area.resolution, default_closed_shape_resolution)

    def test_create_closed_shape_with_triangle_name(self) -> None:
        point_a = Point(0.0, 0.0, "A")
        point_b = Point(1.0, 0.0, "B")
        point_c = Point(0.0, 1.5, "C")
        triangle = Triangle(
            Segment(point_a, point_b),
            Segment(point_b, point_c),
            Segment(point_c, point_a),
        )
        self.drawables.add(triangle)

        area = self.manager.create_region_colored_area(
            triangle_name=triangle.name,
            color="orchid",
            opacity=0.5,
        )

        self.assertEqual(area.shape_type, "polygon")
        self.assertEqual(len(area.segments), 3)
        self.assertEqual({seg.name for seg in area.segments}, {"AB", "BC", "CA"})

    def test_create_closed_shape_triangle_not_found(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.create_region_colored_area(triangle_name="Missing")

    def test_create_closed_shape_invalid_polygon_loop(self) -> None:
        seg1 = Segment(Point(0.0, 0.0, "A"), Point(1.0, 0.0, "B"))
        seg2 = Segment(Point(2.0, 0.0, "C"), Point(3.0, 0.0, "D"))
        self.drawable_manager_proxy.get_segment_by_name = lambda name: {"AB": seg1, "CD": seg2}.get(name)

        with self.assertRaises(ValueError):
            self.manager.create_region_colored_area(polygon_segment_names=["AB", "CD"])

    def test_create_colored_area_segment_function_swap(self) -> None:
        func = Function("sin(x)", name="f1")
        seg = Segment(Point(0.0, 0.0, "A"), Point(1.0, 1.0, "B"))
        self.drawable_manager_proxy.get_function = lambda name: func if name == func.name else None
        self.drawable_manager_proxy.get_segment_by_name = lambda name: seg if name == seg.name else None

        area = self.manager.create_colored_area(
            drawable1_name=seg.name,
            drawable2_name=func.name,
            color="tomato",
            opacity=0.2,
        )

        self.assertIsInstance(area, FunctionSegmentBoundedColoredArea)
        self.assertEqual(len(self.drawables.FunctionSegmentBoundedColoredAreas), 1)
        self.canvas.undo_redo_manager.archive.assert_called()
        self.canvas.draw.assert_called()

    def test_create_colored_area_function_function(self) -> None:
        func1 = Function("x", name="f1")
        func2 = Function("x^2", name="f2")
        self.drawable_manager_proxy.get_function = lambda name: {"f1": func1, "f2": func2}.get(name)

        area = self.manager.create_colored_area(drawable1_name="f1", drawable2_name="f2")

        self.assertIsInstance(area, FunctionsBoundedColoredArea)
        self.assertEqual(area.func1, func1)
        self.assertEqual(area.func2, func2)
        self.assertEqual(len(self.drawables.FunctionsBoundedColoredAreas), 1)
        self.assertEqual(area.color, default_area_fill_color)
        self.assertEqual(area.opacity, default_area_opacity)

    def test_create_colored_area_segment_segment(self) -> None:
        seg1 = Segment(Point(0.0, 0.0, "A"), Point(1.0, 1.0, "B"))
        seg2 = Segment(Point(0.0, 1.0, "C"), Point(1.0, 2.0, "D"))
        self.drawable_manager_proxy.get_segment_by_name = lambda name: {"AB": seg1, "CD": seg2}.get(name)

        area = self.manager.create_colored_area(drawable1_name="AB", drawable2_name="CD")

        self.assertIsInstance(area, SegmentsBoundedColoredArea)
        self.assertEqual(len(self.drawables.SegmentsBoundedColoredAreas), 1)

    def test_create_colored_area_missing_drawable_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.create_colored_area(drawable1_name="missing")

    def test_create_closed_shape_with_rectangle_name(self) -> None:
        point_a = Point(0.0, 0.0, "A")
        point_b = Point(3.0, 0.0, "B")
        point_c = Point(3.0, 2.0, "C")
        point_d = Point(0.0, 2.0, "D")
        rectangle = Rectangle(
            Segment(point_a, point_b),
            Segment(point_b, point_c),
            Segment(point_c, point_d),
            Segment(point_d, point_a),
        )
        self.drawables.add(rectangle)

        area = self.manager.create_region_colored_area(rectangle_name=rectangle.name)
        self.assertEqual(area.shape_type, "polygon")
        self.assertEqual(len(area.segments), 4)

    def test_create_closed_shape_triangle(self) -> None:
        a = Point(0.0, 0.0, "A")
        b = Point(2.0, 0.0, "B")
        c = Point(0.0, 2.0, "C")
        s1 = Segment(a, b)
        s2 = Segment(b, c)
        s3 = Segment(c, a)
        triangle = Triangle(s1, s2, s3)
        self.drawables.add(triangle)

        area = self.manager.create_region_colored_area(
            triangle_name=triangle.name,
            color="plum",
            opacity=0.6,
        )

        self.assertEqual(area.shape_type, "polygon")
        self.assertEqual(set(area.segments), {s1, s2, s3})
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_rectangle(self) -> None:
        a = Point(0.0, 0.0, "A")
        b = Point(3.0, 0.0, "B")
        c = Point(3.0, 2.0, "C")
        d = Point(0.0, 2.0, "D")
        s1 = Segment(a, b)
        s2 = Segment(b, c)
        s3 = Segment(c, d)
        s4 = Segment(d, a)
        rectangle = Rectangle(s1, s2, s3, s4)
        self.drawables.add(rectangle)

        area = self.manager.create_region_colored_area(
            rectangle_name=rectangle.name,
            color="khaki",
            opacity=0.45,
        )

        self.assertEqual(area.shape_type, "polygon")
        self.assertEqual(set(area.segments), {s1, s2, s3, s4})
        self.assertIsNone(area.circle)
        self.assertIsNone(area.ellipse)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_circle_full_defaults_resolution(self) -> None:
        center = Point(0.0, 0.0, "O")
        circle = Circle(center, 4.5)
        self.drawable_manager_proxy.get_circle_by_name = lambda name: circle if name == circle.name else None

        area = self.manager.create_region_colored_area(
            circle_name=circle.name,
            resolution=None,
            color="lightblue",
            opacity=0.25,
        )

        self.assertEqual(area.shape_type, "circle")
        self.assertEqual(area.circle, circle)
        self.assertIsNone(area.chord_segment)
        self.assertEqual(area.resolution, 96)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_ellipse_full(self) -> None:
        center = Point(1.0, -1.0, "H")
        ellipse = Ellipse(center, 6.0, 3.0, rotation_angle=15.0)
        self.drawable_manager_proxy.get_ellipse_by_name = lambda name: ellipse if name == ellipse.name else None

        area = self.manager.create_region_colored_area(
            ellipse_name=ellipse.name,
            color="lavender",
            opacity=0.5,
        )

        self.assertEqual(area.shape_type, "ellipse")
        self.assertIs(area.ellipse, ellipse)
        self.assertIsNone(area.chord_segment)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_closed_shape_ellipse_segment(self) -> None:
        center = Point(-2.0, 1.5, "K")
        ellipse = Ellipse(center, 5.0, 2.5, rotation_angle=30.0)
        chord = Segment(Point(0.0, 0.0, "P"), Point(1.0, 1.0, "Q"))

        self.drawable_manager_proxy.get_ellipse_by_name = lambda name: ellipse if name == ellipse.name else None

        def fake_get_segment(name: str):
            return chord if name == chord.name else None

        self.drawable_manager_proxy.get_segment_by_name = fake_get_segment

        area = self.manager.create_region_colored_area(
            ellipse_name=ellipse.name,
            chord_segment_name=chord.name,
            arc_clockwise=True,
            color="peachpuff",
            opacity=0.55,
        )

        self.assertEqual(area.shape_type, "ellipse_segment")
        self.assertIs(area.ellipse, ellipse)
        self.assertEqual(area.chord_segment, chord)
        self.assertTrue(area.arc_clockwise)
        self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        self.canvas.undo_redo_manager.archive.assert_called_once()
        self.canvas.draw.assert_called_once()
        self.dependency_manager.analyze_drawable_for_dependencies.assert_called_once_with(area)

    def test_create_region_from_expression_creates_region_type(self) -> None:
        import sys
        from types import ModuleType

        mock_region = SimpleMock(name="MockRegion")
        mock_region._sample_to_points = SimpleMock(return_value=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])

        mock_result = SimpleMock()
        mock_result.error = None
        mock_result.region = mock_region

        original_modules = {}
        mock_evaluator = SimpleMock()
        mock_evaluator.evaluate = SimpleMock(return_value=mock_result)

        mock_module = ModuleType("utils.area_expression_evaluator")
        mock_module.AreaExpressionEvaluator = mock_evaluator
        original_modules["utils.area_expression_evaluator"] = sys.modules.get("utils.area_expression_evaluator")
        sys.modules["utils.area_expression_evaluator"] = mock_module

        try:
            area = self.manager.create_region_colored_area(
                expression="circle_A & triangle_ABC",
                color="#D2B48C",
                opacity=0.5,
            )

            self.assertEqual(area.shape_type, "region")
            self.assertEqual(area.expression, "circle_A & triangle_ABC")
            self.assertEqual(len(area.points), 3)
            self.assertEqual(area.color, "#D2B48C")
            self.assertEqual(area.opacity, 0.5)
            self.assertIn(area, self.drawables.ClosedShapeColoredAreas)
        finally:
            if original_modules.get("utils.area_expression_evaluator") is not None:
                sys.modules["utils.area_expression_evaluator"] = original_modules["utils.area_expression_evaluator"]

    def test_create_region_from_expression_error_raises(self) -> None:
        import sys
        from types import ModuleType

        mock_result = SimpleMock()
        mock_result.error = "Invalid expression"
        mock_result.region = None

        mock_evaluator = SimpleMock()
        mock_evaluator.evaluate = SimpleMock(return_value=mock_result)

        mock_module = ModuleType("utils.area_expression_evaluator")
        mock_module.AreaExpressionEvaluator = mock_evaluator
        original = sys.modules.get("utils.area_expression_evaluator")
        sys.modules["utils.area_expression_evaluator"] = mock_module

        try:
            with self.assertRaises(ValueError) as ctx:
                self.manager.create_region_colored_area(
                    expression="invalid_expr",
                    color="red",
                    opacity=0.3,
                )
            self.assertIn("Invalid expression", str(ctx.exception))
        finally:
            if original is not None:
                sys.modules["utils.area_expression_evaluator"] = original

    def test_create_region_no_params_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.manager.create_region_colored_area()
        self.assertIn("expression", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

