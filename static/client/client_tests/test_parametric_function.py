"""Tests for ParametricFunction drawable and renderable."""

import unittest
import copy
import math

from drawables.parametric_function import ParametricFunction
from expression_validator import ExpressionValidator
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper
from rendering.renderables import ParametricFunctionRenderable


class TestParametricFunction(unittest.TestCase):
    """Test suite for ParametricFunction drawable class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.x_expr = "cos(t)"
        self.y_expr = "sin(t)"
        self.name = "Circle"
        self.t_min = 0.0
        self.t_max = 2 * math.pi
        self.func = ParametricFunction(
            x_expression=self.x_expr,
            y_expression=self.y_expr,
            name=self.name,
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        self.assertEqual(self.func.name, self.name)
        self.assertEqual(self.func.t_min, self.t_min)
        self.assertEqual(self.func.t_max, self.t_max)

    def test_init_default_t_max(self) -> None:
        """Test that t_max defaults to 2*pi when not provided."""
        func = ParametricFunction("t", "t")
        self.assertAlmostEqual(func.t_max, 2 * math.pi, places=5)

    def test_init_invalid_expression(self) -> None:
        """Test that invalid expressions raise ValueError."""
        with self.assertRaises(ValueError):
            ParametricFunction("sin(/0)", "t")

    def test_init_invalid_y_expression(self) -> None:
        """Test that invalid y expression raises ValueError."""
        with self.assertRaises(ValueError):
            ParametricFunction("t", "cos(/))")

    def test_get_class_name(self) -> None:
        """Test get_class_name returns correct value."""
        self.assertEqual(self.func.get_class_name(), "ParametricFunction")

    def test_evaluate_circle(self) -> None:
        """Test evaluation of a circle parametric curve."""
        # At t=0: cos(0)=1, sin(0)=0
        x, y = self.func.evaluate(0)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

        # At t=pi/2: cos(pi/2)=0, sin(pi/2)=1
        x, y = self.func.evaluate(math.pi / 2)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)

        # At t=pi: cos(pi)=-1, sin(pi)=0
        x, y = self.func.evaluate(math.pi)
        self.assertAlmostEqual(x, -1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_evaluate_x(self) -> None:
        """Test evaluate_x method."""
        self.assertAlmostEqual(self.func.evaluate_x(0), 1.0, places=5)
        self.assertAlmostEqual(self.func.evaluate_x(math.pi), -1.0, places=5)

    def test_evaluate_y(self) -> None:
        """Test evaluate_y method."""
        self.assertAlmostEqual(self.func.evaluate_y(0), 0.0, places=5)
        self.assertAlmostEqual(self.func.evaluate_y(math.pi / 2), 1.0, places=5)

    def test_evaluate_linear(self) -> None:
        """Test evaluation of linear parametric curve."""
        func = ParametricFunction("t", "2*t", t_min=0, t_max=10)
        x, y = func.evaluate(5)
        self.assertAlmostEqual(x, 5.0, places=5)
        self.assertAlmostEqual(y, 10.0, places=5)

    def test_get_state(self) -> None:
        """Test state serialization."""
        state = self.func.get_state()
        self.assertEqual(state["name"], self.name)
        self.assertEqual(state["args"]["x_expression"], self.x_expr)
        self.assertEqual(state["args"]["y_expression"], self.y_expr)
        self.assertEqual(state["args"]["t_min"], self.t_min)
        self.assertEqual(state["args"]["t_max"], self.t_max)

    def test_deepcopy(self) -> None:
        """Test deep copying."""
        func_copy = copy.deepcopy(self.func)
        self.assertIsNot(func_copy, self.func)
        self.assertEqual(func_copy.x_expression, self.func.x_expression)
        self.assertEqual(func_copy.y_expression, self.func.y_expression)
        self.assertEqual(func_copy.t_min, self.func.t_min)
        self.assertEqual(func_copy.t_max, self.func.t_max)
        self.assertEqual(func_copy.name, self.func.name)

    def test_update_color(self) -> None:
        """Test color update."""
        new_color = "#ff0000"
        self.func.update_color(new_color)
        self.assertEqual(self.func.color, new_color)

    def test_update_t_min(self) -> None:
        """Test t_min update."""
        new_t_min = 1.0
        self.func.update_t_min(new_t_min)
        self.assertEqual(self.func.t_min, new_t_min)

    def test_update_t_max(self) -> None:
        """Test t_max update."""
        new_t_max = 10.0
        self.func.update_t_max(new_t_max)
        self.assertEqual(self.func.t_max, new_t_max)


class TestParametricFunctionRenderable(unittest.TestCase):
    """Test suite for ParametricFunctionRenderable class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.coordinate_mapper = CoordinateMapper(500, 500)
        self.func = ParametricFunction(
            x_expression="cos(t)",
            y_expression="sin(t)",
            name="Circle",
            t_min=0,
            t_max=2 * math.pi,
        )
        self.renderable = ParametricFunctionRenderable(self.func, self.coordinate_mapper)

    def test_build_screen_paths(self) -> None:
        """Test that screen paths are generated."""
        screen_paths = self.renderable.build_screen_paths()
        self.assertTrue(len(screen_paths.paths) > 0)

        # Check that we have points in paths
        points_count = sum(len(path) for path in screen_paths.paths)
        self.assertTrue(points_count > 100)  # Should have many sample points

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation."""
        # Build paths first
        paths1 = self.renderable.build_screen_paths()

        # Invalidate cache
        self.renderable.invalidate_cache()

        # Build paths again
        paths2 = self.renderable.build_screen_paths()

        # Both should have valid paths
        self.assertTrue(len(paths1.paths) > 0)
        self.assertTrue(len(paths2.paths) > 0)

    def test_build_math_paths(self) -> None:
        """Test that math-space paths are generated."""
        math_paths = self.renderable.build_math_paths()
        self.assertTrue(len(math_paths) > 0)

        # For a unit circle, all points should be approximately at radius 1
        for path in math_paths:
            for x, y in path:
                radius = math.sqrt(x * x + y * y)
                self.assertAlmostEqual(radius, 1.0, places=3)

    def test_spiral_path(self) -> None:
        """Test rendering a spiral curve."""
        spiral = ParametricFunction(
            x_expression="t*cos(t)",
            y_expression="t*sin(t)",
            name="Spiral",
            t_min=0,
            t_max=6 * math.pi,
        )
        renderable = ParametricFunctionRenderable(spiral, self.coordinate_mapper)
        screen_paths = renderable.build_screen_paths()
        self.assertTrue(len(screen_paths.paths) > 0)

    def test_lissajous_path(self) -> None:
        """Test rendering a Lissajous curve."""
        lissajous = ParametricFunction(
            x_expression="cos(t)",
            y_expression="sin(2*t)",
            name="Lissajous",
            t_min=0,
            t_max=2 * math.pi,
        )
        renderable = ParametricFunctionRenderable(lissajous, self.coordinate_mapper)
        screen_paths = renderable.build_screen_paths()
        self.assertTrue(len(screen_paths.paths) > 0)


class TestExpressionValidatorParametric(unittest.TestCase):
    """Test suite for parametric expression parsing."""

    def test_parse_parametric_basic(self) -> None:
        """Test basic parametric expression parsing."""
        func = ExpressionValidator.parse_parametric_expression("t")
        self.assertAlmostEqual(func(5), 5.0, places=5)

    def test_parse_parametric_trig(self) -> None:
        """Test trigonometric parametric expression."""
        cos_func = ExpressionValidator.parse_parametric_expression("cos(t)")
        self.assertAlmostEqual(cos_func(0), 1.0, places=5)
        self.assertAlmostEqual(cos_func(math.pi), -1.0, places=5)

    def test_parse_parametric_complex(self) -> None:
        """Test complex parametric expression."""
        func = ExpressionValidator.parse_parametric_expression("t*cos(t) + sin(2*t)")
        # Just verify it can be evaluated without error
        result = func(math.pi)
        self.assertTrue(math.isfinite(result))

    def test_parse_parametric_with_pi(self) -> None:
        """Test parametric expression using pi constant."""
        func = ExpressionValidator.parse_parametric_expression("t/pi")
        self.assertAlmostEqual(func(math.pi), 1.0, places=5)

    def test_parse_parametric_with_e(self) -> None:
        """Test parametric expression using e constant."""
        func = ExpressionValidator.parse_parametric_expression("exp(t)")
        self.assertAlmostEqual(func(0), 1.0, places=5)
        self.assertAlmostEqual(func(1), math.e, places=5)


if __name__ == "__main__":
    unittest.main()
