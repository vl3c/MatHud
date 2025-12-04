"""
Tests for AreaExpressionEvaluator

Tests tokenization, parsing, validation, and evaluation of boolean area expressions.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from utils.area_expression_evaluator import (
    AreaExpressionEvaluator,
    AreaExpressionResult,
    _NameNode,
    _BinaryOpNode,
)
from .simple_mock import SimpleMock


# =============================================================================
# Mock Classes
# =============================================================================

class MockPoint:
    """Mock Point for testing."""
    
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class MockCircle:
    """Mock Circle drawable for testing."""
    
    def __init__(self, center_x: float, center_y: float, radius: float, name: str = "circle_A") -> None:
        self.center = MockPoint(center_x, center_y)
        self.radius = radius
        self.name = name
    
    def get_class_name(self) -> str:
        return "Circle"


class MockEllipse:
    """Mock Ellipse drawable for testing."""
    
    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        rotation_angle: float = 0,
        name: str = "ellipse_A",
    ) -> None:
        self.center = MockPoint(center_x, center_y)
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.rotation_angle = rotation_angle
        self.name = name
    
    def get_class_name(self) -> str:
        return "Ellipse"


class MockSegment:
    """Mock Segment for polygon vertices."""
    
    def __init__(self, p1: MockPoint, p2: MockPoint) -> None:
        self.point1 = p1
        self.point2 = p2


class MockSegmentDrawable:
    """Mock Segment drawable for half-plane testing."""
    
    def __init__(self, p1: tuple, p2: tuple, name: str = "AB") -> None:
        self.point1 = MockPoint(*p1)
        self.point2 = MockPoint(*p2)
        self.name = name
    
    def get_class_name(self) -> str:
        return "Segment"


class MockCircleArc:
    """Mock CircleArc drawable for arc region testing."""
    
    def __init__(
        self,
        p1: tuple,
        p2: tuple,
        center_x: float,
        center_y: float,
        radius: float,
        use_major_arc: bool = False,
        name: str = "ArcMin_AB",
    ) -> None:
        self.point1 = MockPoint(*p1)
        self.point2 = MockPoint(*p2)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.use_major_arc = use_major_arc
        self.name = name
    
    def get_class_name(self) -> str:
        return "CircleArc"


class MockTriangle:
    """Mock Triangle drawable for testing (uses _segments like real Triangle)."""
    
    def __init__(self, vertices: List[tuple], name: str = "triangle_ABC") -> None:
        self._vertices = vertices
        self.name = name
        self._segments = self._build_segments()
    
    def _build_segments(self) -> List[MockSegment]:
        segments = []
        for i in range(len(self._vertices)):
            p1 = MockPoint(*self._vertices[i])
            p2 = MockPoint(*self._vertices[(i + 1) % len(self._vertices)])
            segments.append(MockSegment(p1, p2))
        return segments
    
    def get_class_name(self) -> str:
        return "Triangle"


class MockQuadrilateral:
    """Mock Quadrilateral drawable for testing (uses get_segments like real Quadrilateral)."""
    
    def __init__(self, vertices: List[tuple], name: str = "quad_ABCD") -> None:
        self._vertices = vertices
        self.name = name
    
    def get_class_name(self) -> str:
        return "Quadrilateral"
    
    def get_segments(self) -> List[MockSegment]:
        segments = []
        for i in range(len(self._vertices)):
            p1 = MockPoint(*self._vertices[i])
            p2 = MockPoint(*self._vertices[(i + 1) % len(self._vertices)])
            segments.append(MockSegment(p1, p2))
        return segments


def _make_polygon_mock(class_name: str):
    """Factory to create mock polygon classes with get_segments."""
    class MockPolygon:
        def __init__(self, vertices: List[tuple], name: str = "") -> None:
            self._vertices = vertices
            self.name = name or f"{class_name.lower()}_mock"
        
        def get_class_name(self) -> str:
            return class_name
        
        def get_segments(self) -> List[MockSegment]:
            segments = []
            for i in range(len(self._vertices)):
                p1 = MockPoint(*self._vertices[i])
                p2 = MockPoint(*self._vertices[(i + 1) % len(self._vertices)])
                segments.append(MockSegment(p1, p2))
            return segments
    
    return MockPolygon


MockRectangle = _make_polygon_mock("Rectangle")
MockPentagon = _make_polygon_mock("Pentagon")
MockHexagon = _make_polygon_mock("Hexagon")
MockHeptagon = _make_polygon_mock("Heptagon")
MockOctagon = _make_polygon_mock("Octagon")
MockNonagon = _make_polygon_mock("Nonagon")
MockDecagon = _make_polygon_mock("Decagon")
MockGenericPolygon = _make_polygon_mock("GenericPolygon")


class MockDrawableManager:
    """Mock DrawableManager for testing."""
    
    def __init__(self) -> None:
        self._drawables: Dict[str, Any] = {}
    
    def add_drawable(self, name: str, drawable: Any) -> None:
        self._drawables[name] = drawable
    
    def get_region_capable_drawable_by_name(self, name: str) -> Optional[Any]:
        return self._drawables.get(name)


class MockCanvas:
    """Mock Canvas for testing."""
    
    def __init__(self) -> None:
        self.drawable_manager = MockDrawableManager()


# =============================================================================
# Area Calculation Tests
# =============================================================================

class TestAreaCalculation(unittest.TestCase):
    """Test area calculation with various shapes and operations."""
    
    def setUp(self) -> None:
        self.canvas = MockCanvas()
    
    # -------------------------------------------------------------------------
    # Tokenizer Tests
    # -------------------------------------------------------------------------
    
    def test_tokenize_single_name(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("circle_A")
        self.assertEqual(tokens, ["circle_A"])
    
    def test_tokenize_name_with_prime(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A'")
        self.assertEqual(tokens, ["A'"])
    
    def test_tokenize_name_with_double_prime(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("B''")
        self.assertEqual(tokens, ["B''"])
    
    def test_tokenize_compound_name_with_primes(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A''E'")
        self.assertEqual(tokens, ["A''E'"])
    
    def test_tokenize_expression_with_prime_names(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("ArcMajor & A''E'")
        self.assertEqual(tokens, ["ArcMajor", "&", "A''E'"])
    
    def test_tokenize_name_with_parenthesized_suffix(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A(25)")
        self.assertEqual(tokens, ["A(25)"])
    
    def test_tokenize_name_with_multiple_values_in_suffix(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A(25, 15)")
        self.assertEqual(tokens, ["A(25, 15)"])
    
    def test_tokenize_intersection_operator(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("circle_A & triangle_B")
        self.assertEqual(tokens, ["circle_A", "&", "triangle_B"])
    
    def test_tokenize_union_operator(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("circle_A | triangle_B")
        self.assertEqual(tokens, ["circle_A", "|", "triangle_B"])
    
    def test_tokenize_difference_operator(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("circle_A - triangle_B")
        self.assertEqual(tokens, ["circle_A", "-", "triangle_B"])
    
    def test_tokenize_symmetric_difference_operator(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("circle_A ^ triangle_B")
        self.assertEqual(tokens, ["circle_A", "^", "triangle_B"])
    
    def test_tokenize_parentheses_grouping(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("(A & B) | C")
        self.assertEqual(tokens, ["(", "A", "&", "B", ")", "|", "C"])
    
    def test_tokenize_nested_parentheses(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("((A & B) - C) | D")
        self.assertEqual(tokens, ["(", "(", "A", "&", "B", ")", "-", "C", ")", "|", "D"])
    
    def test_tokenize_whitespace_handling(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("  A   &   B  ")
        self.assertEqual(tokens, ["A", "&", "B"])
    
    def test_tokenize_no_whitespace(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A&B|C")
        self.assertEqual(tokens, ["A", "&", "B", "|", "C"])
    
    def test_tokenize_parenthesized_names_with_grouping_parens(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("(A(25) & B(30))")
        self.assertEqual(tokens, ["(", "A(25)", "&", "B(30)", ")"])
    
    def test_tokenize_parenthesized_names_no_whitespace(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("A(25)&B(30)")
        self.assertEqual(tokens, ["A(25)", "&", "B(30)"])
    
    def test_tokenize_nested_grouping_with_parenthesized_names(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("(A(25) & B(30)) - C(10)")
        self.assertEqual(tokens, ["(", "A(25)", "&", "B(30)", ")", "-", "C(10)"])
    
    def test_tokenize_ellipse_name_with_comma(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("E(50, 30) & C(25)")
        self.assertEqual(tokens, ["E(50, 30)", "&", "C(25)"])
    
    def test_tokenize_grouping_paren_after_paren_name(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("(C(50) & D(30)) | E(10)")
        self.assertEqual(tokens, ["(", "C(50)", "&", "D(30)", ")", "|", "E(10)"])
    
    def test_tokenize_double_grouping_with_paren_names(self) -> None:
        tokens = AreaExpressionEvaluator._tokenize("((A(1) | B(2)) & C(3))")
        self.assertEqual(tokens, ["(", "(", "A(1)", "|", "B(2)", ")", "&", "C(3)", ")"])
    
    # -------------------------------------------------------------------------
    # Parser Tests
    # -------------------------------------------------------------------------
    
    def test_parse_single_name(self) -> None:
        tokens = ["circle_A"]
        ast = AreaExpressionEvaluator._parse(tokens)
        self.assertIsInstance(ast, _NameNode)
        self.assertEqual(ast.name, "circle_A")
    
    def test_parse_binary_operation(self) -> None:
        tokens = ["A", "&", "B"]
        ast = AreaExpressionEvaluator._parse(tokens)
        self.assertIsInstance(ast, _BinaryOpNode)
        self.assertEqual(ast.op, "&")
        self.assertIsInstance(ast.left, _NameNode)
        self.assertIsInstance(ast.right, _NameNode)
    
    def test_parse_left_associative(self) -> None:
        tokens = ["A", "&", "B", "&", "C"]
        ast = AreaExpressionEvaluator._parse(tokens)
        self.assertIsInstance(ast, _BinaryOpNode)
        self.assertEqual(ast.op, "&")
        self.assertIsInstance(ast.left, _BinaryOpNode)
        self.assertEqual(ast.left.op, "&")
    
    def test_parse_precedence_intersection_over_union(self) -> None:
        tokens = ["A", "|", "B", "&", "C"]
        ast = AreaExpressionEvaluator._parse(tokens)
        self.assertIsInstance(ast, _BinaryOpNode)
        self.assertEqual(ast.op, "|")
        self.assertIsInstance(ast.right, _BinaryOpNode)
        self.assertEqual(ast.right.op, "&")
    
    def test_parse_parentheses_override_precedence(self) -> None:
        tokens = ["(", "A", "|", "B", ")", "&", "C"]
        ast = AreaExpressionEvaluator._parse(tokens)
        self.assertIsInstance(ast, _BinaryOpNode)
        self.assertEqual(ast.op, "&")
        self.assertIsInstance(ast.left, _BinaryOpNode)
        self.assertEqual(ast.left.op, "|")
    
    def test_parse_empty_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AreaExpressionEvaluator._parse([])
        self.assertIn("Empty", str(ctx.exception))
    
    def test_parse_unexpected_operator_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AreaExpressionEvaluator._parse(["&", "A"])
        self.assertIn("Unexpected operator", str(ctx.exception))
    
    def test_parse_missing_closing_paren_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AreaExpressionEvaluator._parse(["(", "A", "&", "B"])
        self.assertIn("parenthesis", str(ctx.exception))
    
    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------
    
    def test_validate_empty_expression_raises(self) -> None:
        with self.assertRaises(ValueError):
            AreaExpressionEvaluator._validate_expression("")
    
    def test_validate_whitespace_only_raises(self) -> None:
        with self.assertRaises(ValueError):
            AreaExpressionEvaluator._validate_expression("   ")
    
    def test_validate_unbalanced_open_paren_raises(self) -> None:
        with self.assertRaises(ValueError):
            AreaExpressionEvaluator._validate_expression("(A & B")
    
    def test_validate_unbalanced_close_paren_raises(self) -> None:
        with self.assertRaises(ValueError):
            AreaExpressionEvaluator._validate_expression("A & B)")
    
    def test_validate_balanced_parens_ok(self) -> None:
        AreaExpressionEvaluator._validate_expression("((A & B) | C)")
    
    # -------------------------------------------------------------------------
    # Basic Single Shape Area
    # -------------------------------------------------------------------------
    
    def test_single_circle_area(self) -> None:
        circle = MockCircle(0, 0, 1, "circle_A")
        self.canvas.drawable_manager.add_drawable("circle_A", circle)
        result = AreaExpressionEvaluator.evaluate("circle_A", self.canvas)
        expected_area = math.pi * 1.0 * 1.0
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, expected_area, places=1)
    
    def test_single_ellipse_area(self) -> None:
        ellipse = MockEllipse(0, 0, 2, 1, 0, "ellipse_A")
        self.canvas.drawable_manager.add_drawable("ellipse_A", ellipse)
        result = AreaExpressionEvaluator.evaluate("ellipse_A", self.canvas)
        expected_area = math.pi * 2.0 * 1.0
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, expected_area, places=1)
    
    def test_single_triangle_area(self) -> None:
        triangle = MockTriangle([(0, 0), (4, 0), (0, 3)], "triangle_ABC")
        self.canvas.drawable_manager.add_drawable("triangle_ABC", triangle)
        result = AreaExpressionEvaluator.evaluate("triangle_ABC", self.canvas)
        expected_area = 6.0
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, expected_area, places=1)
    
    def test_quadrilateral_area(self) -> None:
        quad = MockQuadrilateral([(0, 0), (4, 0), (4, 3), (0, 3)], "quad_ABCD")
        self.canvas.drawable_manager.add_drawable("quad_ABCD", quad)
        result = AreaExpressionEvaluator.evaluate("quad_ABCD", self.canvas)
        expected_area = 12.0
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, expected_area, places=1)
    
    # -------------------------------------------------------------------------
    # Binary Operations
    # -------------------------------------------------------------------------
    
    def test_intersection_of_two_circles(self) -> None:
        circle_a = MockCircle(0, 0, 2, "circle_A")
        circle_b = MockCircle(1, 0, 2, "circle_B")
        self.canvas.drawable_manager.add_drawable("circle_A", circle_a)
        self.canvas.drawable_manager.add_drawable("circle_B", circle_b)
        result = AreaExpressionEvaluator.evaluate("circle_A & circle_B", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
        single_circle_area = math.pi * 4
        self.assertLess(result.area, single_circle_area)
    
    def test_union_of_two_circles(self) -> None:
        circle_a = MockCircle(0, 0, 1, "circle_A")
        circle_b = MockCircle(0.5, 0, 1, "circle_B")
        self.canvas.drawable_manager.add_drawable("circle_A", circle_a)
        self.canvas.drawable_manager.add_drawable("circle_B", circle_b)
        result = AreaExpressionEvaluator.evaluate("circle_A | circle_B", self.canvas)
        self.assertIsNone(result.error)
        single_circle_area = math.pi * 1
        self.assertGreater(result.area, single_circle_area)
    
    def test_difference_circle_minus_smaller(self) -> None:
        big_circle = MockCircle(0, 0, 2, "big")
        small_circle = MockCircle(0, 0, 1, "small")
        self.canvas.drawable_manager.add_drawable("big", big_circle)
        self.canvas.drawable_manager.add_drawable("small", small_circle)
        result = AreaExpressionEvaluator.evaluate("big - small", self.canvas)
        self.assertIsNone(result.error)
        big_area = math.pi * 4
        small_area = math.pi * 1
        expected = big_area - small_area
        self.assertAlmostEqual(result.area, expected, places=0)
    
    def test_triangle_minus_circle(self) -> None:
        triangle = MockTriangle([(0, 0), (6, 0), (3, 6)], "GHI")
        circle = MockCircle(3, 2, 1, "C(1)")
        self.canvas.drawable_manager.add_drawable("GHI", triangle)
        self.canvas.drawable_manager.add_drawable("C(1)", circle)
        result = AreaExpressionEvaluator.evaluate("GHI - C(1)", self.canvas)
        self.assertIsNone(result.error)
        triangle_area = 18.0
        circle_area = math.pi * 1
        self.assertAlmostEqual(result.area, triangle_area - circle_area, places=0)
    
    def test_polygon_difference(self) -> None:
        large = MockQuadrilateral([(0, 0), (10, 0), (10, 10), (0, 10)], "ABCD")
        small = MockTriangle([(2, 2), (8, 2), (5, 8)], "EFG")
        self.canvas.drawable_manager.add_drawable("ABCD", large)
        self.canvas.drawable_manager.add_drawable("EFG", small)
        result = AreaExpressionEvaluator.evaluate("ABCD - EFG", self.canvas)
        self.assertIsNone(result.error)
        large_area = 100.0
        small_area = 18.0
        self.assertAlmostEqual(result.area, large_area - small_area, places=0)
    
    # -------------------------------------------------------------------------
    # Nested and Complex Expressions
    # -------------------------------------------------------------------------
    
    def test_nested_expression(self) -> None:
        circle_a = MockCircle(0, 0, 2, "A")
        circle_b = MockCircle(1, 0, 2, "B")
        circle_c = MockCircle(0.5, 0, 1, "C")
        self.canvas.drawable_manager.add_drawable("A", circle_a)
        self.canvas.drawable_manager.add_drawable("B", circle_b)
        self.canvas.drawable_manager.add_drawable("C", circle_c)
        result = AreaExpressionEvaluator.evaluate("(A & B) - C", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_three_shape_expression(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("(circle & quad) - triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_four_shape_expression(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("(circle | ellipse) & (triangle | quad)", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_mixed_operations(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("quad - circle | triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    # -------------------------------------------------------------------------
    # Parenthesized Drawable Names
    # -------------------------------------------------------------------------
    
    def test_drawable_name_with_parenthesized_suffix(self) -> None:
        circle = MockCircle(0, 0, 25, "A(25)")
        self.canvas.drawable_manager.add_drawable("A(25)", circle)
        result = AreaExpressionEvaluator.evaluate("A(25)", self.canvas)
        self.assertIsNone(result.error)
        expected_area = math.pi * 25 * 25
        self.assertAlmostEqual(result.area, expected_area, places=0)
    
    def test_parenthesized_names_with_grouping(self) -> None:
        circle_a = MockCircle(0, 0, 2, "C(2)")
        circle_b = MockCircle(1, 0, 2, "D(2)")
        self.canvas.drawable_manager.add_drawable("C(2)", circle_a)
        self.canvas.drawable_manager.add_drawable("D(2)", circle_b)
        result = AreaExpressionEvaluator.evaluate("(C(2) & D(2))", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_complex_expression_with_parenthesized_names(self) -> None:
        circle_a = MockCircle(0, 0, 3, "A(3)")
        circle_b = MockCircle(1, 0, 3, "B(3)")
        circle_c = MockCircle(0.5, 0, 1, "C(1)")
        self.canvas.drawable_manager.add_drawable("A(3)", circle_a)
        self.canvas.drawable_manager.add_drawable("B(3)", circle_b)
        self.canvas.drawable_manager.add_drawable("C(1)", circle_c)
        result = AreaExpressionEvaluator.evaluate("(A(3) & B(3)) - C(1)", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_name_evaluation(self) -> None:
        ellipse = MockEllipse(0, 0, 3, 2, 0, "E(3, 2)")
        self.canvas.drawable_manager.add_drawable("E(3, 2)", ellipse)
        result = AreaExpressionEvaluator.evaluate("E(3, 2)", self.canvas)
        self.assertIsNone(result.error)
        expected_area = math.pi * 3 * 2
        self.assertAlmostEqual(result.area, expected_area, places=1)
    
    # -------------------------------------------------------------------------
    # Shape Operation Combinations (Circle)
    # -------------------------------------------------------------------------
    
    def test_circle_circle_intersection(self) -> None:
        self._setup_base_shapes()
        c2 = MockCircle(7, 5, 3, "circle2")
        self.canvas.drawable_manager.add_drawable("circle2", c2)
        result = AreaExpressionEvaluator.evaluate("circle & circle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_circle_union(self) -> None:
        self._setup_base_shapes()
        c2 = MockCircle(7, 5, 3, "circle2")
        self.canvas.drawable_manager.add_drawable("circle2", c2)
        result = AreaExpressionEvaluator.evaluate("circle | circle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, math.pi * 9)
    
    def test_circle_circle_difference(self) -> None:
        self._setup_base_shapes()
        c2 = MockCircle(7, 5, 3, "circle2")
        self.canvas.drawable_manager.add_drawable("circle2", c2)
        result = AreaExpressionEvaluator.evaluate("circle - circle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_circle_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        c2 = MockCircle(7, 5, 3, "circle2")
        self.canvas.drawable_manager.add_drawable("circle2", c2)
        result = AreaExpressionEvaluator.evaluate("circle ^ circle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_ellipse_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle & ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_ellipse_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle | ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_ellipse_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle - ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_ellipse_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle ^ ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_triangle_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle & triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_triangle_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle | triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_triangle_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle - triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_triangle_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle ^ triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_quad_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle & quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_quad_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle | quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_circle_quad_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle - quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_circle_quad_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("circle ^ quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    # -------------------------------------------------------------------------
    # Shape Operation Combinations (Ellipse)
    # -------------------------------------------------------------------------
    
    def test_ellipse_ellipse_intersection(self) -> None:
        self._setup_base_shapes()
        e2 = MockEllipse(7, 5, 4, 2, 0, "ellipse2")
        self.canvas.drawable_manager.add_drawable("ellipse2", e2)
        result = AreaExpressionEvaluator.evaluate("ellipse & ellipse2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_ellipse_union(self) -> None:
        self._setup_base_shapes()
        e2 = MockEllipse(7, 5, 4, 2, 0, "ellipse2")
        self.canvas.drawable_manager.add_drawable("ellipse2", e2)
        result = AreaExpressionEvaluator.evaluate("ellipse | ellipse2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_ellipse_difference(self) -> None:
        self._setup_base_shapes()
        e2 = MockEllipse(7, 5, 4, 2, 0, "ellipse2")
        self.canvas.drawable_manager.add_drawable("ellipse2", e2)
        result = AreaExpressionEvaluator.evaluate("ellipse - ellipse2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_ellipse_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        e2 = MockEllipse(7, 5, 4, 2, 0, "ellipse2")
        self.canvas.drawable_manager.add_drawable("ellipse2", e2)
        result = AreaExpressionEvaluator.evaluate("ellipse ^ ellipse2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_triangle_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse & triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_triangle_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse | triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_triangle_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse - triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_triangle_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse ^ triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_quad_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse & quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_quad_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse | quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_ellipse_quad_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse - quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_ellipse_quad_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("ellipse ^ quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    # -------------------------------------------------------------------------
    # Shape Operation Combinations (Polygons)
    # -------------------------------------------------------------------------
    
    def test_triangle_triangle_intersection(self) -> None:
        self._setup_base_shapes()
        t2 = MockTriangle([(2, 0), (12, 0), (7, 8)], "triangle2")
        self.canvas.drawable_manager.add_drawable("triangle2", t2)
        result = AreaExpressionEvaluator.evaluate("triangle & triangle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_triangle_triangle_union(self) -> None:
        self._setup_base_shapes()
        t2 = MockTriangle([(2, 0), (12, 0), (7, 8)], "triangle2")
        self.canvas.drawable_manager.add_drawable("triangle2", t2)
        result = AreaExpressionEvaluator.evaluate("triangle | triangle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_triangle_triangle_difference(self) -> None:
        self._setup_base_shapes()
        t2 = MockTriangle([(2, 0), (12, 0), (7, 8)], "triangle2")
        self.canvas.drawable_manager.add_drawable("triangle2", t2)
        result = AreaExpressionEvaluator.evaluate("triangle - triangle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_triangle_triangle_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        t2 = MockTriangle([(2, 0), (12, 0), (7, 8)], "triangle2")
        self.canvas.drawable_manager.add_drawable("triangle2", t2)
        result = AreaExpressionEvaluator.evaluate("triangle ^ triangle2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_triangle_quad_intersection(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("triangle & quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_triangle_quad_union(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("triangle | quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_triangle_quad_difference(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("triangle - quad", self.canvas)
        self.assertIsNone(result.error)
        # Triangle is contained in quad, so difference is empty (area clamped to 0)
        self.assertEqual(result.area, 0)
    
    def test_pentagon_minus_triangle(self) -> None:
        # Real-world test: pentagon with triangle inside
        # Pentagon: (143, 368), (401, 310), (302, 170), (45, 184), (-54, 270) -> area 58583
        # Triangle: (134, 256), (285, 296), (218, 214) -> area 4851
        # Difference: 58583 - 4851 = 53732
        pentagon = MockPentagon(
            [(143.0, 368.0), (401.0, 310.0), (302.0, 170.0), (45.0, 184.0), (-54.0, 270.0)],
            "pentagon",
        )
        triangle = MockTriangle(
            [(134.0, 256.0), (285.0, 296.0), (218.0, 214.0)],
            "inner_triangle",
        )
        self.canvas.drawable_manager.add_drawable("pentagon", pentagon)
        self.canvas.drawable_manager.add_drawable("inner_triangle", triangle)
        
        result = AreaExpressionEvaluator.evaluate("pentagon - inner_triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 53732, delta=100)
    
    def test_pentagon_triangle_partial_intersection(self) -> None:
        # Real-world test: pentagon with triangle partially overlapping
        # Pentagon: (143, 368), (401, 310), (302, 170), (45, 184), (-54, 270)
        # Triangle IJK: (-97, 208), (-7, 250), (-65, 192)
        # Triangle has one vertex inside pentagon (-7, 250), two outside
        # Intersection is a small triangle with area ~46.36
        pentagon = MockPentagon(
            [(143.0, 368.0), (401.0, 310.0), (302.0, 170.0), (45.0, 184.0), (-54.0, 270.0)],
            "ABCDE",
        )
        triangle = MockTriangle(
            [(-97.0, 208.0), (-7.0, 250.0), (-65.0, 192.0)],
            "IJK",
        )
        self.canvas.drawable_manager.add_drawable("ABCDE", pentagon)
        self.canvas.drawable_manager.add_drawable("IJK", triangle)
        
        result = AreaExpressionEvaluator.evaluate("ABCDE & IJK", self.canvas)
        self.assertIsNone(result.error)
        # Intersection is clipped triangle with vertices at approximately:
        # (-18.15, 238.85), (-7, 250), (-22.60, 242.72)
        self.assertAlmostEqual(result.area, 46.36, delta=5)
    
    def test_triangle_circle_partial_overlap(self) -> None:
        # Triangle (0,0)-(10,0)-(5,8) and Circle center (8,4) r=5
        triangle = MockTriangle([(0, 0), (10, 0), (5, 8)], "tri")
        circle = MockCircle(8, 4, 5, "circ")
        self.canvas.drawable_manager.add_drawable("tri", triangle)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("tri & circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 30.35, delta=5)
    
    def test_triangle_ellipse_partial_overlap(self) -> None:
        # Triangle (0,0)-(12,0)-(6,10) and Ellipse center (10,5) rx=6 ry=4
        triangle = MockTriangle([(0, 0), (12, 0), (6, 10)], "tri")
        ellipse = MockEllipse(10, 5, 6, 4, 0, "ell")
        self.canvas.drawable_manager.add_drawable("tri", triangle)
        self.canvas.drawable_manager.add_drawable("ell", ellipse)
        result = AreaExpressionEvaluator.evaluate("tri & ell", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 30.25, delta=5)
    
    def test_two_triangles_partial_overlap(self) -> None:
        # Triangle A (0,0)-(10,0)-(5,8) and Triangle B (3,2)-(13,2)-(8,10)
        tri_a = MockTriangle([(0, 0), (10, 0), (5, 8)], "triA")
        tri_b = MockTriangle([(3, 2), (13, 2), (8, 10)], "triB")
        self.canvas.drawable_manager.add_drawable("triA", tri_a)
        self.canvas.drawable_manager.add_drawable("triB", tri_b)
        result = AreaExpressionEvaluator.evaluate("triA & triB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 13.23, delta=5)
    
    def test_circle_ellipse_partial_overlap(self) -> None:
        # Circle center (0,0) r=5 and Ellipse center (6,0) rx=4 ry=3
        circle = MockCircle(0, 0, 5, "circ")
        ellipse = MockEllipse(6, 0, 4, 3, 0, "ell")
        self.canvas.drawable_manager.add_drawable("circ", circle)
        self.canvas.drawable_manager.add_drawable("ell", ellipse)
        result = AreaExpressionEvaluator.evaluate("circ & ell", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 11.32, delta=5)
    
    def test_arc_circle_partial_overlap(self) -> None:
        # Arc: center (0,0), r=10, sweep 0-90 degrees (minor arc)
        # Circle: center (8,8), r=5
        p1 = (10, 0)  # 0 degrees
        p2 = (0, 10)  # 90 degrees
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        circle = MockCircle(8, 8, 5, "circ")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("arc & circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 20.10, delta=5)
    
    def test_arc_ellipse_partial_overlap(self) -> None:
        # Arc: center (0,0), r=8, sweep 0-120 degrees (major arc has more than 180)
        # Ellipse: center (6,4), rx=5, ry=3
        p1 = (8, 0)  # 0 degrees
        p2 = (-4, 8 * math.sin(math.radians(120)))  # 120 degrees: x = 8*cos(120) = -4
        arc = MockCircleArc(p1, p2, 0, 0, 8, False, "arc")
        ellipse = MockEllipse(6, 4, 5, 3, 0, "ell")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("ell", ellipse)
        result = AreaExpressionEvaluator.evaluate("arc & ell", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 22.41, delta=5)
    
    def test_arc_triangle_partial_overlap(self) -> None:
        # Arc: center (0,0), r=10, sweep 30-150 degrees
        # Triangle: (0,5)-(8,12)-(-5,10)
        p1 = (10 * math.cos(math.radians(30)), 10 * math.sin(math.radians(30)))
        p2 = (10 * math.cos(math.radians(150)), 10 * math.sin(math.radians(150)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        triangle = MockTriangle([(0, 5), (8, 12), (-5, 10)], "tri")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("tri", triangle)
        result = AreaExpressionEvaluator.evaluate("arc & tri", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 22.96, delta=5)
    
    def test_circle_circle_lens_overlap(self) -> None:
        # Circle A: center (0,0), r=5
        # Circle B: center (4,0), r=5 (overlap forms a lens shape)
        circle_a = MockCircle(0, 0, 5, "circA")
        circle_b = MockCircle(4, 0, 5, "circB")
        self.canvas.drawable_manager.add_drawable("circA", circle_a)
        self.canvas.drawable_manager.add_drawable("circB", circle_b)
        result = AreaExpressionEvaluator.evaluate("circA & circB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 39.54, delta=5)
    
    def test_arc_arc_intersection(self) -> None:
        # Arc1: center (0,0), r=8, sweep 0-120 degrees
        # Arc2: center (0,0), r=10, sweep 60-180 degrees
        p1_a = (8, 0)
        p2_a = (8 * math.cos(math.radians(120)), 8 * math.sin(math.radians(120)))
        arc_a = MockCircleArc(p1_a, p2_a, 0, 0, 8, False, "arcA")
        p1_b = (10 * math.cos(math.radians(60)), 10 * math.sin(math.radians(60)))
        p2_b = (-10, 0)
        arc_b = MockCircleArc(p1_b, p2_b, 0, 0, 10, False, "arcB")
        self.canvas.drawable_manager.add_drawable("arcA", arc_a)
        self.canvas.drawable_manager.add_drawable("arcB", arc_b)
        result = AreaExpressionEvaluator.evaluate("arcA & arcB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 10.43, delta=5)
    
    def test_arc_arc_union(self) -> None:
        p1_a = (8, 0)
        p2_a = (8 * math.cos(math.radians(120)), 8 * math.sin(math.radians(120)))
        arc_a = MockCircleArc(p1_a, p2_a, 0, 0, 8, False, "arcA")
        p1_b = (10 * math.cos(math.radians(60)), 10 * math.sin(math.radians(60)))
        p2_b = (-10, 0)
        arc_b = MockCircleArc(p1_b, p2_b, 0, 0, 10, False, "arcB")
        self.canvas.drawable_manager.add_drawable("arcA", arc_a)
        self.canvas.drawable_manager.add_drawable("arcB", arc_b)
        result = AreaExpressionEvaluator.evaluate("arcA | arcB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 90.17, delta=5)
    
    def test_arc_arc_difference(self) -> None:
        p1_a = (8, 0)
        p2_a = (8 * math.cos(math.radians(120)), 8 * math.sin(math.radians(120)))
        arc_a = MockCircleArc(p1_a, p2_a, 0, 0, 8, False, "arcA")
        p1_b = (10 * math.cos(math.radians(60)), 10 * math.sin(math.radians(60)))
        p2_b = (-10, 0)
        arc_b = MockCircleArc(p1_b, p2_b, 0, 0, 10, False, "arcB")
        self.canvas.drawable_manager.add_drawable("arcA", arc_a)
        self.canvas.drawable_manager.add_drawable("arcB", arc_b)
        result = AreaExpressionEvaluator.evaluate("arcA - arcB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 28.83, delta=5)
    
    def test_arc_arc_symmetric_diff(self) -> None:
        p1_a = (8, 0)
        p2_a = (8 * math.cos(math.radians(120)), 8 * math.sin(math.radians(120)))
        arc_a = MockCircleArc(p1_a, p2_a, 0, 0, 8, False, "arcA")
        p1_b = (10 * math.cos(math.radians(60)), 10 * math.sin(math.radians(60)))
        p2_b = (-10, 0)
        arc_b = MockCircleArc(p1_b, p2_b, 0, 0, 10, False, "arcB")
        self.canvas.drawable_manager.add_drawable("arcA", arc_a)
        self.canvas.drawable_manager.add_drawable("arcB", arc_b)
        result = AreaExpressionEvaluator.evaluate("arcA ^ arcB", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 79.74, delta=5)
    
    def test_arc_quad_intersection(self) -> None:
        # Arc: center (0,0), r=10, sweep 0-120 degrees
        # Quad: (2,2)-(12,2)-(12,10)-(2,10)
        p1 = (10, 0)
        p2 = (10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        quad = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("arc & quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 36.83, delta=5)
    
    def test_arc_quad_union(self) -> None:
        p1 = (10, 0)
        p2 = (10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        quad = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("arc | quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 104.52, delta=5)
    
    def test_arc_quad_difference(self) -> None:
        p1 = (10, 0)
        p2 = (10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        quad = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("arc - quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 24.52, delta=5)
    
    def test_arc_quad_symmetric_diff(self) -> None:
        p1 = (10, 0)
        p2 = (10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        quad = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("arc ^ quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 67.69, delta=5)
    
    def test_arc_rectangle_intersection(self) -> None:
        # Arc: center (0,0), r=8, sweep 30-150 degrees
        # Rectangle: (0,0)-(6,0)-(6,8)-(0,8)
        p1 = (8 * math.cos(math.radians(30)), 8 * math.sin(math.radians(30)))
        p2 = (8 * math.cos(math.radians(150)), 8 * math.sin(math.radians(150)))
        arc = MockCircleArc(p1, p2, 0, 0, 8, False, "arc")
        rect = MockRectangle([(0, 0), (6, 0), (6, 8), (0, 8)], "rect")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("rect", rect)
        result = AreaExpressionEvaluator.evaluate("arc & rect", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 18.99, delta=5)
    
    def test_pentagon_circle_intersection(self) -> None:
        # Pentagon: regular, r=8, center (0,0)
        # Circle: center (6,0), r=5
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        circle = MockCircle(6, 0, 5, "circ")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("pent & circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 42.01, delta=5)
    
    def test_pentagon_circle_union(self) -> None:
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        circle = MockCircle(6, 0, 5, "circ")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("pent | circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 188.57, delta=5)
    
    def test_pentagon_circle_difference(self) -> None:
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        circle = MockCircle(6, 0, 5, "circ")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("pent - circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 110.16, delta=5)
    
    def test_pentagon_circle_symmetric_diff(self) -> None:
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        circle = MockCircle(6, 0, 5, "circ")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("pent ^ circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 146.56, delta=5)
    
    def test_pentagon_ellipse_intersection(self) -> None:
        # Pentagon: regular, r=8, center (0,0)
        # Ellipse: center (5,3), rx=6, ry=4
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        ellipse = MockEllipse(5, 3, 6, 4, 0, "ell")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("ell", ellipse)
        result = AreaExpressionEvaluator.evaluate("pent & ell", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 43.25, delta=5)
    
    def test_hexagon_circle_intersection(self) -> None:
        # Hexagon: regular, r=7, center (0,0)
        # Circle: center (5,0), r=4
        vertices = self._regular_polygon_vertices(6, 7, 0, 0)
        hexagon = MockHexagon(vertices, "hex")
        circle = MockCircle(5, 0, 4, "circ")
        self.canvas.drawable_manager.add_drawable("hex", hexagon)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("hex & circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 33.15, delta=5)
    
    def test_rectangle_circle_intersection(self) -> None:
        # Rectangle: (0,0)-(10,0)-(10,6)-(0,6)
        # Circle: center (8,5), r=4
        rect = MockRectangle([(0, 0), (10, 0), (10, 6), (0, 6)], "rect")
        circle = MockCircle(8, 5, 4, "circ")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("rect & circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 26.15, delta=5)
    
    def test_rectangle_circle_union(self) -> None:
        rect = MockRectangle([(0, 0), (10, 0), (10, 6), (0, 6)], "rect")
        circle = MockCircle(8, 5, 4, "circ")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("rect | circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 84.04, delta=5)
    
    def test_rectangle_circle_difference(self) -> None:
        rect = MockRectangle([(0, 0), (10, 0), (10, 6), (0, 6)], "rect")
        circle = MockCircle(8, 5, 4, "circ")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("rect - circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 33.85, delta=5)
    
    def test_rectangle_circle_symmetric_diff(self) -> None:
        rect = MockRectangle([(0, 0), (10, 0), (10, 6), (0, 6)], "rect")
        circle = MockCircle(8, 5, 4, "circ")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        result = AreaExpressionEvaluator.evaluate("rect ^ circ", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 57.89, delta=5)
    
    def test_three_way_intersection(self) -> None:
        # Triangle & Circle & Quad
        triangle = MockTriangle([(0, 0), (10, 0), (5, 8)], "tri")
        circle = MockCircle(5, 4, 6, "circ")
        quad = MockQuadrilateral([(2, 1), (8, 1), (9, 7), (1, 7)], "quad")
        self.canvas.drawable_manager.add_drawable("tri", triangle)
        self.canvas.drawable_manager.add_drawable("circ", circle)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("tri & circ & quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 27.61, delta=5)
    
    def test_triangle_quad_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        result = AreaExpressionEvaluator.evaluate("triangle ^ quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_quad_quad_intersection(self) -> None:
        self._setup_base_shapes()
        q2 = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad2")
        self.canvas.drawable_manager.add_drawable("quad2", q2)
        result = AreaExpressionEvaluator.evaluate("quad & quad2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_quad_quad_union(self) -> None:
        self._setup_base_shapes()
        q2 = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad2")
        self.canvas.drawable_manager.add_drawable("quad2", q2)
        result = AreaExpressionEvaluator.evaluate("quad | quad2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_quad_quad_difference(self) -> None:
        self._setup_base_shapes()
        q2 = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad2")
        self.canvas.drawable_manager.add_drawable("quad2", q2)
        result = AreaExpressionEvaluator.evaluate("quad - quad2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    def test_quad_quad_symmetric_diff(self) -> None:
        self._setup_base_shapes()
        q2 = MockQuadrilateral([(2, 2), (12, 2), (12, 10), (2, 10)], "quad2")
        self.canvas.drawable_manager.add_drawable("quad2", q2)
        result = AreaExpressionEvaluator.evaluate("quad ^ quad2", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.area, 0)
    
    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------
    
    def test_unknown_drawable_error(self) -> None:
        result = AreaExpressionEvaluator.evaluate("nonexistent", self.canvas)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error)
    
    def test_disjoint_shapes_intersection(self) -> None:
        self._setup_base_shapes()
        far_circle = MockCircle(100, 100, 1, "far_circle")
        self.canvas.drawable_manager.add_drawable("far_circle", far_circle)
        result = AreaExpressionEvaluator.evaluate("circle & far_circle", self.canvas)
        self.assertIsNone(result.error)
        self.assertEqual(result.area, 0)
    
    def test_disjoint_shapes_union(self) -> None:
        self._setup_base_shapes()
        far_circle = MockCircle(100, 100, 1, "far_circle")
        self.canvas.drawable_manager.add_drawable("far_circle", far_circle)
        result = AreaExpressionEvaluator.evaluate("circle | far_circle", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_contained_shapes_difference(self) -> None:
        self._setup_base_shapes()
        small = MockCircle(5, 5, 1, "small_circle")
        self.canvas.drawable_manager.add_drawable("small_circle", small)
        result = AreaExpressionEvaluator.evaluate("circle - small_circle", self.canvas)
        self.assertIsNone(result.error)
        outer_area = math.pi * 9
        inner_area = math.pi * 1
        self.assertAlmostEqual(result.area, outer_area - inner_area, places=0)
    
    # -------------------------------------------------------------------------
    # Segment Half-Plane Operations
    # -------------------------------------------------------------------------
    
    def test_circle_cut_by_horizontal_segment(self) -> None:
        """Cut a circle with a horizontal line through center - should get semicircle."""
        circle = MockCircle(0, 0, 5, "circle")
        segment = MockSegmentDrawable((-10, 0), (10, 0), "AB")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("AB", segment)
        result = AreaExpressionEvaluator.evaluate("circle & AB", self.canvas)
        self.assertIsNone(result.error)
        semicircle_area = math.pi * 25 / 2
        self.assertAlmostEqual(result.area, semicircle_area, places=0)
    
    def test_circle_cut_by_vertical_segment(self) -> None:
        """Cut a circle with a vertical line through center."""
        circle = MockCircle(0, 0, 5, "circle")
        segment = MockSegmentDrawable((0, -10), (0, 10), "CD")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("CD", segment)
        result = AreaExpressionEvaluator.evaluate("circle & CD", self.canvas)
        self.assertIsNone(result.error)
        semicircle_area = math.pi * 25 / 2
        self.assertAlmostEqual(result.area, semicircle_area, places=0)
    
    def test_opposite_half_planes(self) -> None:
        """Reversed segment direction should give opposite half."""
        circle = MockCircle(0, 0, 5, "circle")
        seg_left = MockSegmentDrawable((-10, 0), (10, 0), "AB")
        seg_right = MockSegmentDrawable((10, 0), (-10, 0), "BA")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("AB", seg_left)
        self.canvas.drawable_manager.add_drawable("BA", seg_right)
        result_left = AreaExpressionEvaluator.evaluate("circle & AB", self.canvas)
        result_right = AreaExpressionEvaluator.evaluate("circle & BA", self.canvas)
        self.assertIsNone(result_left.error)
        self.assertIsNone(result_right.error)
        full_circle = math.pi * 25
        self.assertAlmostEqual(result_left.area + result_right.area, full_circle, places=0)
    
    def test_quad_cut_by_segment(self) -> None:
        """Cut a rectangle with a diagonal segment."""
        quad = MockQuadrilateral([(0, 0), (10, 0), (10, 10), (0, 10)], "quad")
        segment = MockSegmentDrawable((0, 0), (10, 10), "diag")
        self.canvas.drawable_manager.add_drawable("quad", quad)
        self.canvas.drawable_manager.add_drawable("diag", segment)
        result = AreaExpressionEvaluator.evaluate("quad & diag", self.canvas)
        self.assertIsNone(result.error)
        half_area = 50.0
        self.assertAlmostEqual(result.area, half_area, places=0)
    
    def test_ellipse_cut_by_segment(self) -> None:
        """Cut an ellipse with a segment through center."""
        ellipse = MockEllipse(0, 0, 6, 4, 0, "ellipse")
        segment = MockSegmentDrawable((-10, 0), (10, 0), "AB")
        self.canvas.drawable_manager.add_drawable("ellipse", ellipse)
        self.canvas.drawable_manager.add_drawable("AB", segment)
        result = AreaExpressionEvaluator.evaluate("ellipse & AB", self.canvas)
        self.assertIsNone(result.error)
        semi_ellipse_area = math.pi * 6 * 4 / 2
        self.assertAlmostEqual(result.area, semi_ellipse_area, places=0)
    
    def test_segment_not_intersecting(self) -> None:
        """Segment that doesn't intersect shape gives zero area."""
        circle = MockCircle(0, 0, 5, "circle")
        segment = MockSegmentDrawable((10, 10), (20, 10), "far")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("far", segment)
        result = AreaExpressionEvaluator.evaluate("circle & far", self.canvas)
        self.assertIsNone(result.error)
    
    def test_triangle_cut_by_segment(self) -> None:
        """Cut a triangle with a horizontal segment."""
        triangle = MockTriangle([(0, 0), (10, 0), (5, 10)], "tri")
        segment = MockSegmentDrawable((-5, 5), (15, 5), "cut")
        self.canvas.drawable_manager.add_drawable("tri", triangle)
        self.canvas.drawable_manager.add_drawable("cut", segment)
        result = AreaExpressionEvaluator.evaluate("tri & cut", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
        self.assertLess(result.area, 50)
    
    def test_complex_expression_with_segment(self) -> None:
        """Complex expression combining shapes and segments."""
        circle = MockCircle(0, 0, 5, "circle")
        quad = MockQuadrilateral([(-3, -3), (3, -3), (3, 3), (-3, 3)], "quad")
        segment = MockSegmentDrawable((-10, 0), (10, 0), "AB")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("quad", quad)
        self.canvas.drawable_manager.add_drawable("AB", segment)
        result = AreaExpressionEvaluator.evaluate("(circle & quad) & AB", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_pentagon_cut_by_diagonal_segment(self) -> None:
        # Pentagon: regular, r=8, center (0,0)
        # Segment: diagonal from (-10,2) to (10,6)
        vertices = self._regular_polygon_vertices(5, 8, 0, 0)
        pentagon = MockPentagon(vertices, "pent")
        segment = MockSegmentDrawable((-10, 2), (10, 6), "cut")
        self.canvas.drawable_manager.add_drawable("pent", pentagon)
        self.canvas.drawable_manager.add_drawable("cut", segment)
        result = AreaExpressionEvaluator.evaluate("pent & cut", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 25.63, delta=5)
    
    def test_hexagon_cut_by_horizontal_segment(self) -> None:
        # Hexagon: regular, r=7, center (0,0)
        # Segment: horizontal at y=3
        vertices = self._regular_polygon_vertices(6, 7, 0, 0)
        hexagon = MockHexagon(vertices, "hex")
        segment = MockSegmentDrawable((-8, 3), (8, 3), "cut")
        self.canvas.drawable_manager.add_drawable("hex", hexagon)
        self.canvas.drawable_manager.add_drawable("cut", segment)
        result = AreaExpressionEvaluator.evaluate("hex & cut", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 27.28, delta=5)
    
    def test_rectangle_cut_by_diagonal(self) -> None:
        # Rectangle: (0,0)-(12,0)-(12,8)-(0,8), area=96
        # Segment: diagonal from (0,0) to (12,8) - should cut in half
        rect = MockRectangle([(0, 0), (12, 0), (12, 8), (0, 8)], "rect")
        segment = MockSegmentDrawable((0, 0), (12, 8), "diag")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        self.canvas.drawable_manager.add_drawable("diag", segment)
        result = AreaExpressionEvaluator.evaluate("rect & diag", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 48.0, delta=5)
    
    def test_circle_cut_by_diagonal_segment(self) -> None:
        # Circle: center (0,0), r=6
        # Segment: diagonal through center
        circle = MockCircle(0, 0, 6, "circ")
        segment = MockSegmentDrawable((-6, -3), (6, 3), "diag")
        self.canvas.drawable_manager.add_drawable("circ", circle)
        self.canvas.drawable_manager.add_drawable("diag", segment)
        result = AreaExpressionEvaluator.evaluate("circ & diag", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 56.46, delta=5)
    
    def test_ellipse_cut_by_diagonal_segment(self) -> None:
        # Ellipse: center (0,0), rx=8, ry=5
        # Segment: diagonal from (-8,0) to (8,5)
        ellipse = MockEllipse(0, 0, 8, 5, 0, "ell")
        segment = MockSegmentDrawable((-8, 0), (8, 5), "diag")
        self.canvas.drawable_manager.add_drawable("ell", ellipse)
        self.canvas.drawable_manager.add_drawable("diag", segment)
        result = AreaExpressionEvaluator.evaluate("ell & diag", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 28.21, delta=5)
    
    def test_arc_cut_by_horizontal_segment(self) -> None:
        # Arc: center (0,0), r=10, sweep 0-120 degrees
        # Segment: horizontal at y=5
        p1 = (10, 0)
        p2 = (10 * math.cos(math.radians(120)), 10 * math.sin(math.radians(120)))
        arc = MockCircleArc(p1, p2, 0, 0, 10, False, "arc")
        segment = MockSegmentDrawable((0, 5), (10, 5), "cut")
        self.canvas.drawable_manager.add_drawable("arc", arc)
        self.canvas.drawable_manager.add_drawable("cut", segment)
        result = AreaExpressionEvaluator.evaluate("arc & cut", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 41.88, delta=5)
    
    def test_circle_cut_by_two_segments_wedge(self) -> None:
        # Circle: center (0,0), r=6
        # Two segments at 45 and -45 degrees through origin (90 degree wedge)
        circle = MockCircle(0, 0, 6, "circ")
        seg1 = MockSegmentDrawable((0, 0), (6, 6), "s1")
        seg2 = MockSegmentDrawable((6, -6), (0, 0), "s2")
        self.canvas.drawable_manager.add_drawable("circ", circle)
        self.canvas.drawable_manager.add_drawable("s1", seg1)
        self.canvas.drawable_manager.add_drawable("s2", seg2)
        result = AreaExpressionEvaluator.evaluate("circ & s1 & s2", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 28.23, delta=5)
    
    # -------------------------------------------------------------------------
    # Result Conversion
    # -------------------------------------------------------------------------
    
    def test_success_result_to_dict(self) -> None:
        result = AreaExpressionResult(area=12.5)
        d = result.to_dict()
        self.assertEqual(d["type"], "area")
        self.assertEqual(d["value"], 12.5)
    
    def test_error_result_to_dict(self) -> None:
        result = AreaExpressionResult(area=0.0, error="Something went wrong")
        d = result.to_dict()
        self.assertEqual(d["type"], "error")
        self.assertEqual(d["value"], "Something went wrong")
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _setup_base_shapes(self) -> None:
        """Set up common shapes used across multiple tests."""
        circle = MockCircle(5, 5, 3, "circle")
        ellipse = MockEllipse(5, 5, 4, 2, 0, "ellipse")
        triangle = MockTriangle([(0, 0), (10, 0), (5, 8)], "triangle")
        quad = MockQuadrilateral([(0, 0), (10, 0), (10, 8), (0, 8)], "quad")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        self.canvas.drawable_manager.add_drawable("ellipse", ellipse)
        self.canvas.drawable_manager.add_drawable("triangle", triangle)
        self.canvas.drawable_manager.add_drawable("quad", quad)
    
    def _regular_polygon_vertices(self, n: int, radius: float = 5, cx: float = 0, cy: float = 0) -> List[tuple]:
        """Generate vertices for a regular n-gon."""
        vertices = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append((x, y))
        return vertices


# =============================================================================
# Region Generation Tests
# =============================================================================

class TestRegionGeneration(unittest.TestCase):
    """Test that all drawable types can be converted to regions."""
    
    def setUp(self) -> None:
        self.canvas = MockCanvas()
    
    # -------------------------------------------------------------------------
    # Basic Shapes
    # -------------------------------------------------------------------------
    
    def test_circle_to_region(self) -> None:
        circle = MockCircle(0, 0, 5, "circle")
        self.canvas.drawable_manager.add_drawable("circle", circle)
        result = AreaExpressionEvaluator.evaluate("circle", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, math.pi * 25, places=0)
    
    def test_ellipse_to_region(self) -> None:
        ellipse = MockEllipse(0, 0, 5, 3, 0, "ellipse")
        self.canvas.drawable_manager.add_drawable("ellipse", ellipse)
        result = AreaExpressionEvaluator.evaluate("ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, math.pi * 15, places=0)
    
    def test_rotated_ellipse_to_region(self) -> None:
        ellipse = MockEllipse(0, 0, 5, 3, 45, "rotated_ellipse")
        self.canvas.drawable_manager.add_drawable("rotated_ellipse", ellipse)
        result = AreaExpressionEvaluator.evaluate("rotated_ellipse", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, math.pi * 15, places=0)
    
    # -------------------------------------------------------------------------
    # Arcs
    # -------------------------------------------------------------------------
    
    def test_minor_arc_to_region(self) -> None:
        # Minor arc: quarter circle (90 degrees) with radius 10
        # Circular segment area = sector - triangle = (theta/2 - sin(theta)/2) * r^2
        # For 90 degrees (pi/2): area = (pi/4 - 0.5) * 100 = 28.54
        arc = MockCircleArc(
            p1=(10, 0),
            p2=(0, 10),
            center_x=0,
            center_y=0,
            radius=10,
            use_major_arc=False,
            name="ArcMin_AB",
        )
        self.canvas.drawable_manager.add_drawable("ArcMin_AB", arc)
        result = AreaExpressionEvaluator.evaluate("ArcMin_AB", self.canvas)
        self.assertIsNone(result.error)
        theta = math.pi / 2
        expected_area = 0.5 * 100 * (theta - math.sin(theta))
        self.assertAlmostEqual(result.area, expected_area, places=0)
    
    def test_major_arc_to_region(self) -> None:
        # Major arc: 3/4 circle (270 degrees) with radius 10
        # Circular segment area for 270 degrees (3*pi/2)
        arc = MockCircleArc(
            p1=(10, 0),
            p2=(0, 10),
            center_x=0,
            center_y=0,
            radius=10,
            use_major_arc=True,
            name="ArcMaj_AB",
        )
        self.canvas.drawable_manager.add_drawable("ArcMaj_AB", arc)
        result = AreaExpressionEvaluator.evaluate("ArcMaj_AB", self.canvas)
        self.assertIsNone(result.error)
        theta = 3 * math.pi / 2
        expected_area = 0.5 * 100 * (theta - math.sin(theta))
        self.assertAlmostEqual(result.area, expected_area, places=0)
    
    def test_arc_with_prime_name(self) -> None:
        # Test arc with apostrophes in name like ArcMaj_C'D'
        arc = MockCircleArc(
            p1=(5, 0),
            p2=(0, 5),
            center_x=0,
            center_y=0,
            radius=5,
            use_major_arc=True,
            name="ArcMaj_C'D'",
        )
        self.canvas.drawable_manager.add_drawable("ArcMaj_C'D'", arc)
        result = AreaExpressionEvaluator.evaluate("ArcMaj_C'D'", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_arc_segment_intersection(self) -> None:
        # Real-world test case: purple major arc from test_runner
        # Arc: center (324, -174), radius 109, from (233, -314) to (370, -52)
        # Segment A''E': from (365, -286) to (440, -132)
        # The segment cuts through the arc, keeping ~90% of the arc segment
        arc = MockCircleArc(
            p1=(233.0, -314.0),
            p2=(370.0, -52.0),
            center_x=324.0,
            center_y=-174.0,
            radius=109.0,
            use_major_arc=True,
            name="ArcMaj_C'D'",
        )
        segment = MockSegmentDrawable(
            p1=(365.0, -286.0),
            p2=(440.0, -132.0),
            name="A''E'",
        )
        self.canvas.drawable_manager.add_drawable("ArcMaj_C'D'", arc)
        self.canvas.drawable_manager.add_drawable("A''E'", segment)
        
        result = AreaExpressionEvaluator.evaluate("ArcMaj_C'D' & A''E'", self.canvas)
        self.assertIsNone(result.error)
        # Major arc sweep is 192.36 degrees, full area ~21216
        # Segment cuts through, intersection area ~19094
        self.assertAlmostEqual(result.area, 19094, delta=100)
    
    def test_major_arc_sweep_greater_than_180(self) -> None:
        # Bug fix test: major arc must have sweep > 180 degrees
        # This test verifies the sweep direction calculation is correct
        # Arc from (10, 0) to (-10, 0) around center (0, 0) with radius 10
        # CCW sweep is 180 degrees exactly - edge case
        # For points where CCW < 180, major arc should go CW (longer path)
        arc = MockCircleArc(
            p1=(10, 0),
            p2=(0, -10),  # 90 degrees CW from p1
            center_x=0,
            center_y=0,
            radius=10,
            use_major_arc=True,
            name="ArcMaj_test",
        )
        self.canvas.drawable_manager.add_drawable("ArcMaj_test", arc)
        result = AreaExpressionEvaluator.evaluate("ArcMaj_test", self.canvas)
        self.assertIsNone(result.error)
        # Major arc should be 270 degrees (going CCW the long way)
        # Area = 0.5 * 100 * (3pi/2 - sin(3pi/2)) = 0.5 * 100 * (4.712 + 1) = 285.6
        theta = 3 * math.pi / 2
        expected_area = 0.5 * 100 * (theta - math.sin(theta))
        self.assertAlmostEqual(result.area, expected_area, delta=10)
    
    # -------------------------------------------------------------------------
    # Polygons with _segments (Triangle pattern)
    # -------------------------------------------------------------------------
    
    def test_triangle_to_region(self) -> None:
        triangle = MockTriangle([(0, 0), (6, 0), (3, 4)], "triangle")
        self.canvas.drawable_manager.add_drawable("triangle", triangle)
        result = AreaExpressionEvaluator.evaluate("triangle", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 12.0, places=1)
    
    # -------------------------------------------------------------------------
    # Polygons with get_segments (Quadrilateral pattern)
    # -------------------------------------------------------------------------
    
    def test_quadrilateral_to_region(self) -> None:
        quad = MockQuadrilateral([(0, 0), (4, 0), (4, 3), (0, 3)], "quad")
        self.canvas.drawable_manager.add_drawable("quad", quad)
        result = AreaExpressionEvaluator.evaluate("quad", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 12.0, places=1)
    
    def test_rectangle_to_region(self) -> None:
        rect = MockRectangle([(0, 0), (5, 0), (5, 4), (0, 4)], "rect")
        self.canvas.drawable_manager.add_drawable("rect", rect)
        result = AreaExpressionEvaluator.evaluate("rect", self.canvas)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.area, 20.0, places=1)
    
    def test_pentagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(5)
        pentagon = MockPentagon(vertices, "pentagon")
        self.canvas.drawable_manager.add_drawable("pentagon", pentagon)
        result = AreaExpressionEvaluator.evaluate("pentagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_hexagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(6)
        hexagon = MockHexagon(vertices, "hexagon")
        self.canvas.drawable_manager.add_drawable("hexagon", hexagon)
        result = AreaExpressionEvaluator.evaluate("hexagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_heptagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(7)
        heptagon = MockHeptagon(vertices, "heptagon")
        self.canvas.drawable_manager.add_drawable("heptagon", heptagon)
        result = AreaExpressionEvaluator.evaluate("heptagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_octagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(8)
        octagon = MockOctagon(vertices, "octagon")
        self.canvas.drawable_manager.add_drawable("octagon", octagon)
        result = AreaExpressionEvaluator.evaluate("octagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_nonagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(9)
        nonagon = MockNonagon(vertices, "nonagon")
        self.canvas.drawable_manager.add_drawable("nonagon", nonagon)
        result = AreaExpressionEvaluator.evaluate("nonagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_decagon_to_region(self) -> None:
        vertices = self._regular_polygon_vertices(10)
        decagon = MockDecagon(vertices, "decagon")
        self.canvas.drawable_manager.add_drawable("decagon", decagon)
        result = AreaExpressionEvaluator.evaluate("decagon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    def test_generic_polygon_to_region(self) -> None:
        vertices = [(0, 0), (3, 0), (4, 2), (3, 4), (0, 4), (-1, 2)]
        polygon = MockGenericPolygon(vertices, "polygon")
        self.canvas.drawable_manager.add_drawable("polygon", polygon)
        result = AreaExpressionEvaluator.evaluate("polygon", self.canvas)
        self.assertIsNone(result.error)
        self.assertGreater(result.area, 0)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _regular_polygon_vertices(self, n: int, radius: float = 5, cx: float = 0, cy: float = 0) -> List[tuple]:
        """Generate vertices for a regular n-gon."""
        vertices = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append((x, y))
        return vertices
