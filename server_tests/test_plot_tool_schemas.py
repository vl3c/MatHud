from __future__ import annotations

import unittest
from typing import Any, Dict, List

from static.functions_definitions import FUNCTIONS


def _find_tool(name: str) -> Dict[str, Any]:
    for entry in FUNCTIONS:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "function":
            continue
        fn = entry.get("function")
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == name:
            return entry
    raise AssertionError(f"Tool not found: {name}")


def _require_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be dict")
    return value


def _require_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise AssertionError(f"{label} must be list")
    return value


class TestPlotToolSchemas(unittest.TestCase):
    def _get_params(self, tool_name: str) -> Dict[str, Any]:
        tool = _find_tool(tool_name)
        fn = _require_dict(tool.get("function"), "function")
        params = _require_dict(fn.get("parameters"), f"{tool_name}.parameters")
        self.assertEqual(params.get("type"), "object")
        self.assertEqual(params.get("additionalProperties"), False)
        return params

    def test_plot_distribution_schema(self) -> None:
        tool = _find_tool("plot_distribution")
        fn = _require_dict(tool.get("function"), "function")
        self.assertTrue(fn.get("strict"))

        params = self._get_params("plot_distribution")
        props = _require_dict(params.get("properties"), "plot_distribution.properties")
        required = _require_list(params.get("required"), "plot_distribution.required")

        expected_required = [
            "name",
            "representation",
            "distribution_type",
            "distribution_params",
            "plot_bounds",
            "shade_bounds",
            "curve_color",
            "fill_color",
            "fill_opacity",
            "bar_count",
        ]
        self.assertEqual(required, expected_required)

        self.assertIn("name", props)
        self.assertIn("representation", props)
        self.assertIn("distribution_type", props)
        self.assertIn("distribution_params", props)
        self.assertIn("plot_bounds", props)
        self.assertIn("shade_bounds", props)
        self.assertIn("curve_color", props)
        self.assertIn("fill_color", props)
        self.assertIn("fill_opacity", props)
        self.assertIn("bar_count", props)

        self.assertNotIn("left_bound", props)
        self.assertNotIn("right_bound", props)

        representation = _require_dict(props.get("representation"), "representation")
        self.assertEqual(representation.get("type"), "string")
        self.assertEqual(representation.get("enum"), ["continuous", "discrete"])

        dist_type = _require_dict(props.get("distribution_type"), "distribution_type")
        self.assertEqual(dist_type.get("type"), "string")
        self.assertEqual(dist_type.get("enum"), ["normal"])

        dist_params = _require_dict(props.get("distribution_params"), "distribution_params")
        self.assertEqual(dist_params.get("type"), ["object", "null"])
        self.assertEqual(dist_params.get("additionalProperties"), False)
        dp_props = _require_dict(dist_params.get("properties"), "distribution_params.properties")
        self.assertEqual(dist_params.get("required"), ["mean", "sigma"])

        mean = _require_dict(dp_props.get("mean"), "distribution_params.mean")
        sigma = _require_dict(dp_props.get("sigma"), "distribution_params.sigma")
        self.assertEqual(mean.get("type"), ["number", "null"])
        self.assertEqual(sigma.get("type"), ["number", "null"])

        plot_bounds = _require_dict(props.get("plot_bounds"), "plot_bounds")
        self.assertEqual(plot_bounds.get("type"), ["object", "null"])
        self.assertEqual(plot_bounds.get("additionalProperties"), False)
        self.assertEqual(plot_bounds.get("required"), ["left_bound", "right_bound"])
        pb_props = _require_dict(plot_bounds.get("properties"), "plot_bounds.properties")
        self.assertEqual(
            _require_dict(pb_props.get("left_bound"), "plot_bounds.left_bound").get("type"), ["number", "null"]
        )
        self.assertEqual(
            _require_dict(pb_props.get("right_bound"), "plot_bounds.right_bound").get("type"), ["number", "null"]
        )

        shade_bounds = _require_dict(props.get("shade_bounds"), "shade_bounds")
        self.assertEqual(shade_bounds.get("type"), ["object", "null"])
        self.assertEqual(shade_bounds.get("additionalProperties"), False)
        self.assertEqual(shade_bounds.get("required"), ["left_bound", "right_bound"])
        sb_props = _require_dict(shade_bounds.get("properties"), "shade_bounds.properties")
        self.assertEqual(
            _require_dict(sb_props.get("left_bound"), "shade_bounds.left_bound").get("type"), ["number", "null"]
        )
        self.assertEqual(
            _require_dict(sb_props.get("right_bound"), "shade_bounds.right_bound").get("type"), ["number", "null"]
        )

    def test_plot_bars_schema(self) -> None:
        tool = _find_tool("plot_bars")
        fn = _require_dict(tool.get("function"), "function")
        self.assertTrue(fn.get("strict"))

        params = self._get_params("plot_bars")
        props = _require_dict(params.get("properties"), "plot_bars.properties")
        required = _require_list(params.get("required"), "plot_bars.required")

        expected_required = [
            "name",
            "values",
            "labels_below",
            "labels_above",
            "bar_spacing",
            "bar_width",
            "stroke_color",
            "fill_color",
            "fill_opacity",
            "x_start",
            "y_base",
        ]
        self.assertEqual(required, expected_required)

        values = _require_dict(props.get("values"), "values")
        self.assertEqual(values.get("type"), "array")
        values_items = _require_dict(values.get("items"), "values.items")
        self.assertEqual(values_items.get("type"), "number")

        labels_below = _require_dict(props.get("labels_below"), "labels_below")
        self.assertEqual(labels_below.get("type"), "array")
        labels_below_items = _require_dict(labels_below.get("items"), "labels_below.items")
        self.assertEqual(labels_below_items.get("type"), "string")

        labels_above = _require_dict(props.get("labels_above"), "labels_above")
        self.assertEqual(labels_above.get("type"), ["array", "null"])
        labels_above_items = _require_dict(labels_above.get("items"), "labels_above.items")
        self.assertEqual(labels_above_items.get("type"), "string")

        for numeric_optional in ("bar_spacing", "bar_width", "fill_opacity", "x_start", "y_base"):
            item = _require_dict(props.get(numeric_optional), numeric_optional)
            self.assertEqual(item.get("type"), ["number", "null"])

        for string_optional in ("stroke_color", "fill_color"):
            item = _require_dict(props.get(string_optional), string_optional)
            self.assertEqual(item.get("type"), ["string", "null"])

        name = _require_dict(props.get("name"), "name")
        self.assertEqual(name.get("type"), ["string", "null"])

    def test_delete_plot_schema(self) -> None:
        tool = _find_tool("delete_plot")
        fn = _require_dict(tool.get("function"), "function")
        self.assertTrue(fn.get("strict"))

        params = self._get_params("delete_plot")
        props = _require_dict(params.get("properties"), "delete_plot.properties")
        required = _require_list(params.get("required"), "delete_plot.required")

        self.assertEqual(list(props.keys()), ["name"])
        self.assertEqual(required, ["name"])

        name = _require_dict(props.get("name"), "delete_plot.name")
        self.assertEqual(name.get("type"), "string")

    def test_compute_descriptive_statistics_schema(self) -> None:
        tool = _find_tool("compute_descriptive_statistics")
        fn = _require_dict(tool.get("function"), "function")
        self.assertTrue(fn.get("strict"))

        params = self._get_params("compute_descriptive_statistics")
        props = _require_dict(
            params.get("properties"),
            "compute_descriptive_statistics.properties",
        )
        required = _require_list(
            params.get("required"),
            "compute_descriptive_statistics.required",
        )

        self.assertEqual(required, ["data"])

        data = _require_dict(props.get("data"), "data")
        self.assertEqual(data.get("type"), "array")
        data_items = _require_dict(data.get("items"), "data.items")
        self.assertEqual(data_items.get("type"), "number")
        self.assertEqual(data.get("minItems"), 1)
