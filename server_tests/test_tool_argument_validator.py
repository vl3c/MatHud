"""Tests for the centralized tool argument validator.

Covers schema index construction, type validation, required/unknown field
checks, enum enforcement, nested object/array recursion, anyOf unions,
minItems/maxLength constraints, canonicalization rules, and integration
with ToolCallProcessor.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from static.functions_definitions import FUNCTIONS
from static.tool_argument_validator import ToolArgumentValidator, _SCHEMA_INDEX
from static.tool_call_processor import ToolCallProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tool_schema(name: str) -> Dict[str, Any] | None:
    """Return the parameters schema for a tool by name, or None."""
    for entry in FUNCTIONS:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "function":
            continue
        fn = entry.get("function")
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == name:
            return fn.get("parameters", {})
    return None


# ===================================================================
# 6a. Schema Index Tests
# ===================================================================


class TestSchemaIndex(unittest.TestCase):
    """Tests for the schema index built at import time."""

    def test_all_functions_indexed(self) -> None:
        """Every function in FUNCTIONS should appear in the schema index."""
        expected_names: List[str] = []
        for entry in FUNCTIONS:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "function":
                continue
            fn = entry.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    expected_names.append(name)

        self.assertGreater(len(expected_names), 60, "Expected 60+ tool definitions")

        for name in expected_names:
            self.assertIn(
                name,
                _SCHEMA_INDEX,
                f"Tool '{name}' missing from schema index",
            )

    def test_unknown_function_returns_none(self) -> None:
        """get_schema should return None for unregistered function names."""
        self.assertIsNone(ToolArgumentValidator.get_schema("nonexistent_tool"))

    def test_schema_structure_for_create_point(self) -> None:
        """Schema for create_point should have correct structure."""
        schema = ToolArgumentValidator.get_schema("create_point")
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertEqual(schema.get("type"), "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        self.assertEqual(schema.get("additionalProperties"), False)

    def test_schema_matches_functions_definitions(self) -> None:
        """Schema index entries should match what's in FUNCTIONS."""
        for name in ["create_point", "zoom", "reset_canvas", "create_circle"]:
            expected = _find_tool_schema(name)
            actual = ToolArgumentValidator.get_schema(name)
            self.assertEqual(actual, expected, f"Schema mismatch for '{name}'")


# ===================================================================
# 6b. Successful Validation Tests
# ===================================================================


class TestSuccessfulValidation(unittest.TestCase):
    """Tests for valid arguments that should pass validation."""

    def test_create_point_valid(self) -> None:
        """create_point with correct types passes validation."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": 5, "y": 10, "color": None, "name": None}
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["arguments"]["x"], 5)
        self.assertEqual(result["arguments"]["y"], 10)

    def test_create_point_with_string_values(self) -> None:
        """create_point with non-null optional string fields."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": 3.5, "y": -2.0, "color": "red", "name": "A"}
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["arguments"]["color"], "red")
        self.assertEqual(result["arguments"]["name"], "A")

    def test_create_segment_valid(self) -> None:
        """create_segment with all fields populated."""
        result = ToolArgumentValidator.validate(
            "create_segment",
            {
                "x1": 0,
                "y1": 0,
                "x2": 3,
                "y2": 4,
                "color": "red",
                "name": "AB",
                "label_text": None,
                "label_visible": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_create_polygon_valid(self) -> None:
        """create_polygon with 3 vertices passes validation."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": 4, "y": 0},
                    {"x": 2, "y": 3},
                ],
                "polygon_type": "triangle",
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_create_circle_valid(self) -> None:
        """create_circle with correct types passes validation."""
        result = ToolArgumentValidator.validate(
            "create_circle",
            {
                "center_x": 0,
                "center_y": 0,
                "radius": 5,
                "color": None,
                "name": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_zoom_valid(self) -> None:
        """zoom with valid enum value passes validation."""
        result = ToolArgumentValidator.validate(
            "zoom",
            {
                "center_x": 0,
                "center_y": 0,
                "range_val": 5,
                "range_axis": "x",
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_reset_canvas_empty_args(self) -> None:
        """reset_canvas with empty args passes validation."""
        result = ToolArgumentValidator.validate("reset_canvas", {})
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_evaluate_expression_valid(self) -> None:
        """evaluate_expression with expression and null variables."""
        result = ToolArgumentValidator.validate(
            "evaluate_expression",
            {"expression": "2+2", "variables": None},
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_evaluate_expression_with_variables(self) -> None:
        """evaluate_expression with variables dict (freeform object)."""
        result = ToolArgumentValidator.validate(
            "evaluate_expression",
            {"expression": "5*x - 1", "variables": {"x": 2}},
        )
        self.assertTrue(result["valid"])

    def test_evaluate_linear_algebra_scalar(self) -> None:
        """evaluate_linear_algebra_expression with scalar values."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [
                    {"name": "a", "value": 5},
                    {"name": "b", "value": 3},
                ],
                "expression": "a + b",
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_evaluate_linear_algebra_vector(self) -> None:
        """evaluate_linear_algebra_expression with vector values."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [
                    {"name": "v", "value": [1, 2, 3]},
                ],
                "expression": "v",
            },
        )
        self.assertTrue(result["valid"])

    def test_evaluate_linear_algebra_matrix(self) -> None:
        """evaluate_linear_algebra_expression with matrix values."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [
                    {"name": "A", "value": [[1, 2], [3, 4]]},
                    {"name": "B", "value": [[5, 6], [7, 8]]},
                ],
                "expression": "inv(A) * B",
            },
        )
        self.assertTrue(result["valid"])

    def test_numeric_integrate_required_only(self) -> None:
        """numeric_integrate with only required fields."""
        result = ToolArgumentValidator.validate(
            "numeric_integrate",
            {
                "expression": "x^2",
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 1,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_numeric_integrate_all_fields(self) -> None:
        """numeric_integrate with optional method and steps."""
        result = ToolArgumentValidator.validate(
            "numeric_integrate",
            {
                "expression": "sin(x)",
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 3.14159,
                "method": "simpson",
                "steps": 200,
            },
        )
        self.assertTrue(result["valid"])

    def test_plot_distribution_valid(self) -> None:
        """plot_distribution with nested distribution_params and plot_bounds."""
        result = ToolArgumentValidator.validate(
            "plot_distribution",
            {
                "name": None,
                "representation": "continuous",
                "distribution_type": "normal",
                "distribution_params": {"mean": 0, "sigma": 1},
                "plot_bounds": {"left_bound": -4, "right_bound": 4},
                "shade_bounds": {"left_bound": -1, "right_bound": 1},
                "curve_color": None,
                "fill_color": None,
                "fill_opacity": None,
                "bar_count": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_plot_distribution_null_nested_objects(self) -> None:
        """plot_distribution with null nested objects passes validation."""
        result = ToolArgumentValidator.validate(
            "plot_distribution",
            {
                "name": None,
                "representation": "discrete",
                "distribution_type": "normal",
                "distribution_params": None,
                "plot_bounds": None,
                "shade_bounds": None,
                "curve_color": None,
                "fill_color": None,
                "fill_opacity": None,
                "bar_count": 10,
            },
        )
        self.assertTrue(result["valid"])

    def test_generate_graph_valid(self) -> None:
        """generate_graph with vertices, edges, and placement_box."""
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": "G1",
                "graph_type": "graph",
                "directed": None,
                "root": None,
                "layout": None,
                "placement_box": {"x": 0, "y": 0, "width": 10, "height": 10},
                "vertices": [
                    {"name": "A", "x": 0, "y": 0, "color": None, "label": None},
                    {"name": "B", "x": 5, "y": 5, "color": None, "label": None},
                ],
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "weight": 1,
                        "name": None,
                        "color": None,
                        "directed": None,
                    }
                ],
                "adjacency_matrix": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_draw_piecewise_function_valid(self) -> None:
        """draw_piecewise_function with valid pieces."""
        result = ToolArgumentValidator.validate(
            "draw_piecewise_function",
            {
                "pieces": [
                    {
                        "expression": "x^2",
                        "left": None,
                        "right": 0,
                        "left_inclusive": False,
                        "right_inclusive": True,
                        "undefined_at": None,
                    },
                    {
                        "expression": "x",
                        "left": 0,
                        "right": None,
                        "left_inclusive": False,
                        "right_inclusive": False,
                        "undefined_at": [1, 2],
                    },
                ],
                "name": None,
                "color": None,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_set_coordinate_system_valid(self) -> None:
        """set_coordinate_system with valid enum."""
        result = ToolArgumentValidator.validate(
            "set_coordinate_system", {"mode": "polar"}
        )
        self.assertTrue(result["valid"])

    def test_set_grid_visible_valid(self) -> None:
        """set_grid_visible with boolean value."""
        result = ToolArgumentValidator.validate(
            "set_grid_visible", {"visible": True}
        )
        self.assertTrue(result["valid"])

    def test_int_accepted_for_number_field(self) -> None:
        """Python int should be accepted for JSON Schema 'number' type."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": 5, "y": 10, "color": None, "name": None}
        )
        self.assertTrue(result["valid"])
        # int should pass through as-is (not converted to float)
        self.assertIs(type(result["arguments"]["x"]), int)


# ===================================================================
# 6c. Type Rejection Tests
# ===================================================================


class TestTypeRejection(unittest.TestCase):
    """Tests for wrong-type arguments that should be rejected."""

    def test_string_instead_of_number(self) -> None:
        """Non-numeric string for a number field should fail."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "five", "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'x'" in e and "number" in e for e in result["errors"]))

    def test_boolean_instead_of_number(self) -> None:
        """Boolean for a number field should fail (bools are not numbers)."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": True, "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'x'" in e for e in result["errors"]))

    def test_list_instead_of_number(self) -> None:
        """List for a number field should fail."""
        result = ToolArgumentValidator.validate(
            "create_circle",
            {
                "center_x": 0,
                "center_y": 0,
                "radius": [5],
                "color": None,
                "name": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'radius'" in e for e in result["errors"]))

    def test_number_instead_of_string(self) -> None:
        """Number for a string enum field should fail."""
        result = ToolArgumentValidator.validate(
            "zoom",
            {
                "center_x": 0,
                "center_y": 0,
                "range_val": 5,
                "range_axis": 123,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'range_axis'" in e for e in result["errors"]))

    def test_string_instead_of_boolean(self) -> None:
        """String for a boolean field should fail."""
        result = ToolArgumentValidator.validate(
            "set_grid_visible", {"visible": "yes"}
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'visible'" in e for e in result["errors"]))

    def test_string_instead_of_array(self) -> None:
        """String for an array field should fail."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": "not_an_array",
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'vertices'" in e for e in result["errors"]))

    def test_string_instead_of_object(self) -> None:
        """String for an object field should fail."""
        result = ToolArgumentValidator.validate(
            "plot_distribution",
            {
                "name": None,
                "representation": "continuous",
                "distribution_type": "normal",
                "distribution_params": "wrong",
                "plot_bounds": None,
                "shade_bounds": None,
                "curve_color": None,
                "fill_color": None,
                "fill_opacity": None,
                "bar_count": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'distribution_params'" in e for e in result["errors"])
        )

    def test_float_instead_of_integer(self) -> None:
        """Float for an integer field should fail."""
        result = ToolArgumentValidator.validate(
            "numeric_integrate",
            {
                "expression": "x^2",
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 1,
                "steps": 3.5,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'steps'" in e for e in result["errors"]))

    def test_null_where_not_allowed(self) -> None:
        """None for a non-nullable field should fail."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": None, "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'x'" in e for e in result["errors"]))

    def test_multiple_type_errors_reported(self) -> None:
        """All type errors should be reported, not just the first."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "bad", "y": "also_bad", "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(len(result["errors"]), 2)


# ===================================================================
# 6d. Required Field Tests
# ===================================================================


class TestRequiredFields(unittest.TestCase):
    """Tests for missing required fields."""

    def test_missing_required_fields_create_point(self) -> None:
        """Missing y, color, name should produce errors."""
        result = ToolArgumentValidator.validate("create_point", {"x": 5})
        self.assertFalse(result["valid"])
        # Should report missing y, color, and name
        error_text = " ".join(result["errors"])
        self.assertIn("'y'", error_text)
        self.assertIn("'color'", error_text)
        self.assertIn("'name'", error_text)

    def test_missing_required_fields_zoom(self) -> None:
        """Missing center_y, range_val, range_axis should produce errors."""
        result = ToolArgumentValidator.validate("zoom", {"center_x": 0})
        self.assertFalse(result["valid"])
        error_text = " ".join(result["errors"])
        self.assertIn("'center_y'", error_text)
        self.assertIn("'range_val'", error_text)
        self.assertIn("'range_axis'", error_text)

    def test_missing_nested_required_field(self) -> None:
        """Missing required field in nested object should be reported."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [{"x": 0}, {"x": 1, "y": 1}, {"x": 2, "y": 2}],
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertFalse(result["valid"])
        # Should report missing y in vertices[0]
        self.assertTrue(
            any("vertices[0]" in e and "'y'" in e for e in result["errors"])
            or any("vertices[0].y" in e for e in result["errors"])
        )

    def test_empty_args_for_required_fields(self) -> None:
        """Empty args dict for a tool with required fields should fail."""
        result = ToolArgumentValidator.validate("create_point", {})
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(len(result["errors"]), 4)  # x, y, color, name


# ===================================================================
# 6e. Unknown Key Tests
# ===================================================================


class TestUnknownKeys(unittest.TestCase):
    """Tests for extra/unknown keys in arguments."""

    def test_extra_top_level_key(self) -> None:
        """Extra top-level key should be reported."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": 5, "y": 10, "color": None, "name": None, "extra": True},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("'extra'" in e and "unknown" in e for e in result["errors"]))

    def test_extra_key_in_nested_object(self) -> None:
        """Extra key in a nested object (vertex) should be reported."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [
                    {"x": 0, "y": 0, "z": 5},
                    {"x": 1, "y": 1},
                    {"x": 2, "y": 2},
                ],
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'z'" in e or "'vertices[0].z'" in e for e in result["errors"])
        )

    def test_allowed_keys_listed_in_error(self) -> None:
        """Error message for unknown key should list allowed keys."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": 5, "y": 10, "color": None, "name": None, "bogus": 1},
        )
        self.assertFalse(result["valid"])
        error_text = " ".join(result["errors"])
        self.assertIn("allowed:", error_text)


# ===================================================================
# 6f. Enum Validation Tests
# ===================================================================


class TestEnumValidation(unittest.TestCase):
    """Tests for enum value enforcement."""

    def test_invalid_enum_zoom_range_axis(self) -> None:
        """Invalid enum value for range_axis should fail."""
        result = ToolArgumentValidator.validate(
            "zoom",
            {
                "center_x": 0,
                "center_y": 0,
                "range_val": 5,
                "range_axis": "z",
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'range_axis'" in e and "'z'" in e for e in result["errors"])
        )

    def test_invalid_enum_coordinate_system(self) -> None:
        """Invalid enum value for set_coordinate_system should fail."""
        result = ToolArgumentValidator.validate(
            "set_coordinate_system", {"mode": "spherical"}
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'mode'" in e and "'spherical'" in e for e in result["errors"])
        )

    def test_valid_enum_value(self) -> None:
        """Valid enum value should pass."""
        result = ToolArgumentValidator.validate(
            "zoom",
            {
                "center_x": 0,
                "center_y": 0,
                "range_val": 5,
                "range_axis": "y",
            },
        )
        self.assertTrue(result["valid"])

    def test_invalid_polygon_type_enum(self) -> None:
        """Invalid polygon_type enum should fail."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 0, "y": 1},
                ],
                "polygon_type": "circle",
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'polygon_type'" in e and "'circle'" in e for e in result["errors"])
        )

    def test_null_in_nullable_enum(self) -> None:
        """None is a valid enum value for nullable enum fields."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 0, "y": 1},
                ],
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertTrue(result["valid"])

    def test_invalid_numeric_integrate_method(self) -> None:
        """Invalid method enum for numeric_integrate should fail."""
        result = ToolArgumentValidator.validate(
            "numeric_integrate",
            {
                "expression": "x^2",
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 1,
                "method": "euler",
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("'method'" in e and "'euler'" in e for e in result["errors"])
        )


# ===================================================================
# 6g. Canonicalization Tests
# ===================================================================


class TestCanonicalization(unittest.TestCase):
    """Tests for argument canonicalization rules."""

    def test_string_to_number_coercion(self) -> None:
        """Numeric string for a number field should be coerced to float."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "5", "y": "10.5", "color": None, "name": None},
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["arguments"]["x"], 5.0)
        self.assertIsInstance(result["arguments"]["x"], float)
        self.assertAlmostEqual(result["arguments"]["y"], 10.5)

    def test_string_to_integer_coercion(self) -> None:
        """Integer string for an integer field should be coerced to int."""
        result = ToolArgumentValidator.validate(
            "numeric_integrate",
            {
                "expression": "x^2",
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 1,
                "steps": "200",
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["arguments"]["steps"], 200)
        self.assertIsInstance(result["arguments"]["steps"], int)

    def test_empty_string_to_null_for_nullable_string(self) -> None:
        """Empty string for nullable string field should be coerced to None."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": 5, "y": 10, "color": "", "name": ""},
        )
        self.assertTrue(result["valid"])
        self.assertIsNone(result["arguments"]["color"])
        self.assertIsNone(result["arguments"]["name"])

    def test_nan_rejection(self) -> None:
        """NaN for a number field should be rejected."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": float("nan"), "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("finite" in e for e in result["errors"]))

    def test_infinity_rejection(self) -> None:
        """Infinity for a number field should be rejected."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": float("inf"), "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("finite" in e for e in result["errors"]))

    def test_negative_infinity_rejection(self) -> None:
        """Negative infinity for a number field should be rejected."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": float("-inf"), "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])

    def test_null_passes_through_unchanged(self) -> None:
        """None for a nullable field should pass through as None."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": 5, "y": 10, "color": None, "name": None},
        )
        self.assertTrue(result["valid"])
        self.assertIsNone(result["arguments"]["color"])
        self.assertIsNone(result["arguments"]["name"])

    def test_non_numeric_string_not_coerced(self) -> None:
        """Non-numeric string for number field should fail, not be coerced."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "abc", "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])

    def test_string_nan_not_coerced(self) -> None:
        """String 'nan' should not be coerced to float('nan')."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "nan", "y": 10, "color": None, "name": None},
        )
        # "nan" parses to float('nan') which is then rejected as non-finite
        self.assertFalse(result["valid"])

    def test_string_inf_not_coerced(self) -> None:
        """String 'inf' should not be coerced to float('inf')."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": "inf", "y": 10, "color": None, "name": None},
        )
        # "inf" parses to float('inf') which is then rejected as non-finite
        self.assertFalse(result["valid"])

    def test_coercion_does_not_mutate_original(self) -> None:
        """Original arguments dict should not be mutated by canonicalization."""
        original = {"x": "5", "y": "10", "color": None, "name": None}
        original_copy = dict(original)
        result = ToolArgumentValidator.validate("create_point", original)
        self.assertTrue(result["valid"])
        # Original should be unchanged
        self.assertEqual(original, original_copy)
        # Canonicalized result should have the coerced values
        self.assertEqual(result["arguments"]["x"], 5.0)

    def test_coercion_logs_warning(self) -> None:
        """String-to-number coercion should log a warning."""
        with self.assertLogs("static.tool_argument_validator", level="WARNING") as cm:
            ToolArgumentValidator.validate(
                "create_point",
                {"x": "5", "y": 10, "color": None, "name": None},
            )
        self.assertTrue(any("coerced" in msg for msg in cm.output))

    def test_empty_string_canonicalization_logs_warning(self) -> None:
        """Empty-string-to-null canonicalization should log a warning."""
        with self.assertLogs("static.tool_argument_validator", level="WARNING") as cm:
            ToolArgumentValidator.validate(
                "create_point",
                {"x": 5, "y": 10, "color": "", "name": None},
            )
        self.assertTrue(any("canonicalized" in msg for msg in cm.output))

    def test_boolean_not_coerced_to_number(self) -> None:
        """Boolean should NOT be treated as a number (even though bool is int subclass)."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": False, "y": 10, "color": None, "name": None},
        )
        self.assertFalse(result["valid"])


# ===================================================================
# 6h. Nested Structure Tests
# ===================================================================


class TestNestedStructures(unittest.TestCase):
    """Tests for validation of nested objects and arrays."""

    def test_generate_graph_valid_nested(self) -> None:
        """generate_graph with valid nested structures passes."""
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": "G1",
                "graph_type": "tree",
                "directed": True,
                "root": "A",
                "layout": "tree",
                "placement_box": {"x": -5, "y": -5, "width": 20, "height": 20},
                "vertices": [
                    {"name": "A", "x": 0, "y": 0, "color": None, "label": "Root"},
                    {"name": "B", "x": 5, "y": -5, "color": "blue", "label": None},
                ],
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "weight": None,
                        "name": None,
                        "color": None,
                        "directed": True,
                    }
                ],
                "adjacency_matrix": None,
            },
        )
        self.assertTrue(result["valid"])

    def test_generate_graph_invalid_vertex_type(self) -> None:
        """String instead of number in vertex x should fail."""
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": None,
                "graph_type": "graph",
                "directed": None,
                "root": None,
                "layout": None,
                "placement_box": None,
                "vertices": [
                    {"name": "A", "x": "wrong", "y": 0, "color": None, "label": None},
                ],
                "edges": [],
                "adjacency_matrix": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("vertices[0]" in e for e in result["errors"])
        )

    def test_plot_distribution_invalid_nested_type(self) -> None:
        """Invalid type in distribution_params.sigma should fail."""
        result = ToolArgumentValidator.validate(
            "plot_distribution",
            {
                "name": None,
                "representation": "continuous",
                "distribution_type": "normal",
                "distribution_params": {"mean": 0, "sigma": "wide"},
                "plot_bounds": None,
                "shade_bounds": None,
                "curve_color": None,
                "fill_color": None,
                "fill_opacity": None,
                "bar_count": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("sigma" in e for e in result["errors"])
        )

    def test_draw_piecewise_deep_nesting(self) -> None:
        """Invalid type in piecewise function piece should fail."""
        result = ToolArgumentValidator.validate(
            "draw_piecewise_function",
            {
                "pieces": [
                    {
                        "expression": "x^2",
                        "left": "not_a_number",
                        "right": 0,
                        "left_inclusive": True,
                        "right_inclusive": True,
                        "undefined_at": None,
                    }
                ],
                "name": None,
                "color": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("left" in e for e in result["errors"]))

    def test_piecewise_invalid_undefined_at_element(self) -> None:
        """Non-number in undefined_at array should fail."""
        result = ToolArgumentValidator.validate(
            "draw_piecewise_function",
            {
                "pieces": [
                    {
                        "expression": "x",
                        "left": 0,
                        "right": 10,
                        "left_inclusive": True,
                        "right_inclusive": False,
                        "undefined_at": [1, "bad", 3],
                    }
                ],
                "name": None,
                "color": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("undefined_at" in e for e in result["errors"]))

    def test_anyof_number_variant(self) -> None:
        """anyOf value with scalar number should pass."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [{"name": "a", "value": 42}],
                "expression": "a",
            },
        )
        self.assertTrue(result["valid"])

    def test_anyof_vector_variant(self) -> None:
        """anyOf value with 1D array (vector) should pass."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [{"name": "v", "value": [1, 2, 3]}],
                "expression": "v",
            },
        )
        self.assertTrue(result["valid"])

    def test_anyof_matrix_variant(self) -> None:
        """anyOf value with 2D array (matrix) should pass."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [{"name": "M", "value": [[1, 2], [3, 4]]}],
                "expression": "M",
            },
        )
        self.assertTrue(result["valid"])

    def test_anyof_invalid_value(self) -> None:
        """anyOf value with string (not number/array) should fail."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [{"name": "A", "value": "not_valid"}],
                "expression": "A",
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("did not match any" in e for e in result["errors"])
        )

    def test_anyof_invalid_array_element(self) -> None:
        """anyOf value with array containing strings should fail."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {
                "objects": [{"name": "v", "value": [1, "two", 3]}],
                "expression": "v",
            },
        )
        self.assertFalse(result["valid"])

    def test_placement_box_missing_required(self) -> None:
        """Missing required field in nested placement_box should fail."""
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": None,
                "graph_type": "graph",
                "directed": None,
                "root": None,
                "layout": None,
                "placement_box": {"x": 0, "y": 0, "width": 10},
                "vertices": [],
                "edges": [],
                "adjacency_matrix": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("height" in e for e in result["errors"])
        )

    def test_nested_object_unknown_key(self) -> None:
        """Extra key in nested placement_box should be reported."""
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": None,
                "graph_type": "graph",
                "directed": None,
                "root": None,
                "layout": None,
                "placement_box": {
                    "x": 0,
                    "y": 0,
                    "width": 10,
                    "height": 10,
                    "depth": 5,
                },
                "vertices": [],
                "edges": [],
                "adjacency_matrix": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("depth" in e for e in result["errors"]))


# ===================================================================
# 6i. Constraint Tests (minItems, maxLength)
# ===================================================================


class TestConstraints(unittest.TestCase):
    """Tests for minItems and maxLength constraint enforcement."""

    def test_min_items_violation_polygon(self) -> None:
        """create_polygon with only 2 vertices should fail (minItems: 3)."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("at least 3" in e and "got 2" in e for e in result["errors"])
        )

    def test_min_items_exactly_met(self) -> None:
        """create_polygon with exactly 3 vertices should pass."""
        result = ToolArgumentValidator.validate(
            "create_polygon",
            {
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 0, "y": 1},
                ],
                "polygon_type": None,
                "color": None,
                "name": None,
                "subtype": None,
            },
        )
        self.assertTrue(result["valid"])

    def test_min_items_piecewise_empty_array(self) -> None:
        """draw_piecewise_function with empty pieces array should fail (minItems: 1)."""
        result = ToolArgumentValidator.validate(
            "draw_piecewise_function",
            {"pieces": [], "name": None, "color": None},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("at least 1" in e for e in result["errors"])
        )

    def test_min_items_linear_algebra_empty_objects(self) -> None:
        """evaluate_linear_algebra_expression with empty objects should fail."""
        result = ToolArgumentValidator.validate(
            "evaluate_linear_algebra_expression",
            {"objects": [], "expression": "A"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("at least 1" in e for e in result["errors"])
        )

    def test_max_length_violation(self) -> None:
        """create_label with text exceeding maxLength should fail."""
        result = ToolArgumentValidator.validate(
            "create_label",
            {
                "x": 0,
                "y": 0,
                "text": "a" * 200,
                "name": None,
                "color": None,
                "font_size": None,
                "rotation_degrees": None,
            },
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("at most 160" in e and "got 200" in e for e in result["errors"])
        )

    def test_max_length_at_limit(self) -> None:
        """create_label with text exactly at maxLength should pass."""
        result = ToolArgumentValidator.validate(
            "create_label",
            {
                "x": 0,
                "y": 0,
                "text": "a" * 160,
                "name": None,
                "color": None,
                "font_size": None,
                "rotation_degrees": None,
            },
        )
        self.assertTrue(result["valid"])

    def test_max_length_under_limit(self) -> None:
        """create_label with short text should pass."""
        result = ToolArgumentValidator.validate(
            "create_label",
            {
                "x": 0,
                "y": 0,
                "text": "Hello",
                "name": None,
                "color": None,
                "font_size": None,
                "rotation_degrees": None,
            },
        )
        self.assertTrue(result["valid"])


# ===================================================================
# 6j. Edge Cases
# ===================================================================


class TestEdgeCases(unittest.TestCase):
    """Tests for boundary conditions and unusual inputs."""

    def test_none_arguments(self) -> None:
        """None as arguments should be handled gracefully."""
        result = ToolArgumentValidator.validate("reset_canvas", None)  # type: ignore[arg-type]
        self.assertTrue(result["valid"])

    def test_unknown_function_passes_through(self) -> None:
        """Unknown function name should pass with valid=True and log warning."""
        with self.assertLogs("static.tool_argument_validator", level="WARNING") as cm:
            result = ToolArgumentValidator.validate(
                "totally_unknown_function", {"foo": "bar"}
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(any("no schema found" in msg for msg in cm.output))

    def test_unknown_function_returns_original_args(self) -> None:
        """Unknown function should return the original arguments."""
        args = {"foo": "bar", "baz": 42}
        result = ToolArgumentValidator.validate("unknown_fn", args)
        self.assertEqual(result["arguments"], args)

    def test_empty_args_for_no_parameter_tool(self) -> None:
        """Empty args for tool with no parameters should pass."""
        result = ToolArgumentValidator.validate("reset_canvas", {})
        self.assertTrue(result["valid"])
        self.assertEqual(result["arguments"], {})

    def test_large_argument_dict(self) -> None:
        """Large argument dict should not crash."""
        args = {
            "x": 0,
            "y": 0,
            "color": None,
            "name": None,
        }
        # Add many extra keys to test performance
        for i in range(100):
            args[f"extra_{i}"] = i
        result = ToolArgumentValidator.validate("create_point", args)
        # Should fail due to unknown keys, but not crash
        self.assertFalse(result["valid"])

    def test_deeply_nested_valid_graph(self) -> None:
        """generate_graph with many vertices and edges should work."""
        vertices = [
            {"name": f"V{i}", "x": i * 10, "y": i * 5, "color": None, "label": None}
            for i in range(20)
        ]
        edges = [
            {
                "source": i,
                "target": i + 1,
                "weight": 1,
                "name": None,
                "color": None,
                "directed": None,
            }
            for i in range(19)
        ]
        result = ToolArgumentValidator.validate(
            "generate_graph",
            {
                "name": "big_graph",
                "graph_type": "graph",
                "directed": False,
                "root": None,
                "layout": "force",
                "placement_box": None,
                "vertices": vertices,
                "edges": edges,
                "adjacency_matrix": None,
            },
        )
        self.assertTrue(result["valid"])

    def test_error_value_truncation(self) -> None:
        """Long string values in errors should be truncated."""
        long_string = "x" * 200
        result = ToolArgumentValidator.validate(
            "set_coordinate_system", {"mode": long_string}
        )
        self.assertFalse(result["valid"])
        # Error message should contain truncated value, not the full 200 chars
        error_text = result["errors"][0]
        self.assertIn("...", error_text)

    def test_validation_result_structure(self) -> None:
        """ValidationResult should contain valid, arguments, and errors keys."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": 5, "y": 10, "color": None, "name": None}
        )
        self.assertIn("valid", result)
        self.assertIn("arguments", result)
        self.assertIn("errors", result)
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["arguments"], dict)
        self.assertIsInstance(result["errors"], list)

    def test_invalid_returns_original_args(self) -> None:
        """When validation fails, original arguments should be returned."""
        original = {"x": "bad", "y": 10, "color": None, "name": None}
        result = ToolArgumentValidator.validate("create_point", original)
        self.assertFalse(result["valid"])
        # Should return the original, not a deep copy with partial coercion
        self.assertEqual(result["arguments"]["x"], "bad")

    def test_valid_returns_canonical_args(self) -> None:
        """When validation passes, canonical arguments should be returned."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": "5", "y": 10, "color": None, "name": None}
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["arguments"]["x"], 5.0)

    def test_negative_numbers_valid(self) -> None:
        """Negative numbers should be valid for number fields."""
        result = ToolArgumentValidator.validate(
            "create_point",
            {"x": -100.5, "y": -0.001, "color": None, "name": None},
        )
        self.assertTrue(result["valid"])

    def test_zero_valid(self) -> None:
        """Zero should be valid for number fields."""
        result = ToolArgumentValidator.validate(
            "create_point", {"x": 0, "y": 0, "color": None, "name": None}
        )
        self.assertTrue(result["valid"])


# ===================================================================
# 6k. Integration Tests with ToolCallProcessor
# ===================================================================


class TestToolCallProcessorIntegration(unittest.TestCase):
    """Tests that the validator is wired into ToolCallProcessor."""

    @staticmethod
    def _make_tool_call(name: str, arguments_json: str) -> Any:
        """Create a mock OpenAI tool call object."""
        tool_call = MagicMock()
        tool_call.function.name = name
        tool_call.function.arguments = arguments_json
        return tool_call

    def test_valid_call_uses_canonical_args(self) -> None:
        """jsonify_tool_call should use canonicalized args when valid."""
        tool_call = self._make_tool_call(
            "create_point",
            '{"x": "5", "y": 10, "color": null, "name": null}',
        )
        processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        self.assertEqual(processed["function_name"], "create_point")
        # x should be coerced from "5" to 5.0
        self.assertEqual(processed["arguments"]["x"], 5.0)
        self.assertEqual(processed["arguments"]["y"], 10)

    def test_invalid_call_logs_warning_and_proceeds(self) -> None:
        """jsonify_tool_call should log warnings for invalid args but proceed."""
        tool_call = self._make_tool_call(
            "create_point",
            '{"x": "not_a_number", "y": 10, "color": null, "name": null}',
        )
        with self.assertLogs(level="WARNING") as cm:
            processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        # Should still produce a result (log-only mode)
        self.assertEqual(processed["function_name"], "create_point")
        self.assertIn("arguments", processed)
        # Original args should pass through on failure
        self.assertEqual(processed["arguments"]["x"], "not_a_number")
        # Warning should have been logged
        self.assertTrue(any("create_point" in msg for msg in cm.output))

    def test_valid_call_no_canonicalization_needed(self) -> None:
        """jsonify_tool_call with already-correct types should pass through."""
        tool_call = self._make_tool_call(
            "zoom",
            '{"center_x": 0, "center_y": 0, "range_val": 5, "range_axis": "x"}',
        )
        processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        self.assertEqual(processed["function_name"], "zoom")
        self.assertEqual(processed["arguments"]["range_axis"], "x")
        self.assertEqual(processed["arguments"]["center_x"], 0)

    def test_empty_string_canonicalized_in_pipeline(self) -> None:
        """Empty strings in nullable fields should be canonicalized to null."""
        tool_call = self._make_tool_call(
            "create_point",
            '{"x": 5, "y": 10, "color": "", "name": ""}',
        )
        processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        self.assertIsNone(processed["arguments"]["color"])
        self.assertIsNone(processed["arguments"]["name"])

    def test_json_parse_failure_still_works(self) -> None:
        """Malformed JSON should be handled gracefully by ToolCallProcessor."""
        tool_call = self._make_tool_call("create_point", "not valid json")
        processed = ToolCallProcessor.jsonify_tool_call(tool_call)
        self.assertEqual(processed["function_name"], "create_point")
        # Should get empty dict from JSON parse failure
        self.assertIsInstance(processed["arguments"], dict)

    def test_jsonify_tool_calls_batch(self) -> None:
        """jsonify_tool_calls should process multiple calls."""
        calls = [
            self._make_tool_call(
                "create_point",
                '{"x": 1, "y": 2, "color": null, "name": null}',
            ),
            self._make_tool_call("reset_canvas", "{}"),
        ]
        results = ToolCallProcessor.jsonify_tool_calls(calls)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["function_name"], "create_point")
        self.assertEqual(results[1]["function_name"], "reset_canvas")


# ===================================================================
# Helper function tests
# ===================================================================


class TestHelperFunctions(unittest.TestCase):
    """Tests for internal helper functions exposed via the module."""

    def test_get_schema_known(self) -> None:
        """get_schema returns schema dict for known tools."""
        schema = ToolArgumentValidator.get_schema("create_point")
        self.assertIsNotNone(schema)
        self.assertIsInstance(schema, dict)

    def test_get_schema_unknown(self) -> None:
        """get_schema returns None for unknown tools."""
        self.assertIsNone(ToolArgumentValidator.get_schema("nonexistent"))

    def test_validate_returns_typed_dict(self) -> None:
        """validate should return a ValidationResult TypedDict."""
        result = ToolArgumentValidator.validate("reset_canvas", {})
        self.assertIsInstance(result, dict)
        self.assertIn("valid", result)
        self.assertIn("arguments", result)
        self.assertIn("errors", result)
