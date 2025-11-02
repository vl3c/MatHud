"""
Linear Algebra Utility Helpers

Provides vector and matrix validation plus expression evaluation support
using the shared MathJS instance exposed through the browser window.

Responsibilities:
    - Normalize incoming vector and matrix definitions
    - Populate MathJS scope objects for safe expression evaluation
    - Convert MathJS results back into Python-native structures
    - Surface clear error messages for invalid definitions or operations
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Union, TypedDict

from browser import window


class LinearAlgebraObject(TypedDict):
    name: str
    value: Any


class LinearAlgebraResult(TypedDict):
    type: str
    value: Any


class LinearAlgebraUtils:
    """Helper utilities for evaluating linear algebra expressions via MathJS."""

    ALLOWED_FUNCTION_NAMES: Tuple[str, ...] = (
        "transpose",
        "inv",
        "det",
        "dot",
        "cross",
        "norm",
        "trace",
        "diag",
        "identity",
        "zeros",
        "ones",
        "reshape",
        "size",
    )

    ALLOWED_CONSTANT_NAMES: Tuple[str, ...] = ("pi", "e")

    ADDITIONAL_IDENTIFIER_ALLOWLIST: Tuple[str, ...] = (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sqrt",
        "exp",
        "log",
        "abs",
        "pow",
    )

    @staticmethod
    def evaluate_expression(objects: List[LinearAlgebraObject], expression: str) -> LinearAlgebraResult:
        """Evaluate a linear algebra expression with predefined objects."""

        LinearAlgebraUtils._validate_expression(expression)
        scope = LinearAlgebraUtils._build_scope(objects)
        LinearAlgebraUtils._validate_identifiers(expression, scope)

        try:
            result = window.math.evaluate(expression, scope)
        except Exception as exc:  # pragma: no cover - MathJS error message path
            return {"type": "error", "value": f"Error: {exc}"}

        return LinearAlgebraUtils._convert_result(result)

    @staticmethod
    def _validate_expression(expression: str) -> None:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string")

    @staticmethod
    def _build_scope(objects: List[LinearAlgebraObject]) -> Dict[str, Any]:
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")

        scope: Dict[str, Any] = {}
        for entry in objects:
            name, value = LinearAlgebraUtils._extract_object(entry)
            if name in scope:
                raise ValueError(f"Duplicate object name detected: {name}")
            math_value = LinearAlgebraUtils._create_math_value(value)
            scope[name] = math_value

        LinearAlgebraUtils._attach_math_bindings(scope)
        return scope

    @staticmethod
    def _attach_math_bindings(scope: Dict[str, Any]) -> None:
        for func_name in LinearAlgebraUtils.ALLOWED_FUNCTION_NAMES:
            if hasattr(window.math, func_name):
                scope[func_name] = getattr(window.math, func_name)

        for constant_name in LinearAlgebraUtils.ALLOWED_CONSTANT_NAMES:
            if hasattr(window.math, constant_name):
                scope[constant_name] = getattr(window.math, constant_name)

    @staticmethod
    def _validate_identifiers(expression: str, scope: Dict[str, Any]) -> None:
        allowed_tokens = set(scope.keys()) | set(LinearAlgebraUtils.ADDITIONAL_IDENTIFIER_ALLOWLIST)
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))

        unknown_tokens = []
        for token in tokens:
            if token in allowed_tokens:
                continue
            if hasattr(window.math, token):
                continue
            unknown_tokens.append(token)

        unknown_tokens.sort()
        if unknown_tokens:
            raise ValueError(
                "Unknown identifiers in expression: " + ", ".join(unknown_tokens)
            )

    @staticmethod
    def _extract_object(entry: LinearAlgebraObject) -> Tuple[str, Any]:
        if not isinstance(entry, dict):
            raise ValueError("Each object definition must be a dictionary")
        if "name" not in entry or "value" not in entry:
            raise ValueError("Each object definition requires 'name' and 'value'")

        name = entry["name"]
        value = entry["value"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Object name must be a non-empty string")

        return name.strip(), value

    @staticmethod
    def _create_math_value(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, list):
            normalized, is_matrix = LinearAlgebraUtils._normalize_array(value)
            if hasattr(window.math, "matrix"):
                if is_matrix or isinstance(normalized[0], list):
                    return window.math.matrix(normalized)

                return window.math.matrix(normalized)

            return normalized

        raise ValueError("Only numeric scalars, vectors, or matrices are supported")

    @staticmethod
    def _normalize_array(value: List[Any]) -> Tuple[List[Any], bool]:
        if not value:
            raise ValueError("Vectors and matrices cannot be empty")

        if all(isinstance(item, list) for item in value):
            return LinearAlgebraUtils._normalize_matrix(value), True

        if any(isinstance(item, list) for item in value):
            raise ValueError("Mixed dimensional arrays are not supported")

        return LinearAlgebraUtils._normalize_vector(value), False

    @staticmethod
    def _normalize_vector(items: List[Any]) -> List[float]:
        normalized: List[float] = []
        for item in items:
            if not isinstance(item, (int, float)):
                raise ValueError("Vector entries must be numeric")
            normalized.append(float(item))
        return normalized

    @staticmethod
    def _normalize_matrix(rows: List[List[Any]]) -> List[List[float]]:
        row_lengths = {len(row) for row in rows}
        if len(row_lengths) != 1:
            raise ValueError("All matrix rows must have the same length")

        normalized_rows: List[List[float]] = []
        for row in rows:
            normalized_row: List[float] = []
            for item in row:
                if not isinstance(item, (int, float)):
                    raise ValueError("Matrix entries must be numeric")
                normalized_row.append(float(item))
            normalized_rows.append(normalized_row)

        return normalized_rows

    @staticmethod
    def _convert_result(result: Any) -> LinearAlgebraResult:
        try:
            value_type = window.math.typeOf(result)
        except Exception:
            value_type = None

        if value_type == "Matrix":
            content = LinearAlgebraUtils._convert_js_value(result.toArray())
            return LinearAlgebraUtils._wrap_sequence(content)

        if value_type == "Complex":
            formatted = window.math.format(result)
            return {"type": "complex", "value": formatted}

        if value_type in {"BigNumber", "Fraction"}:
            if hasattr(result, "toNumber"):
                return {"type": "scalar", "value": float(result.toNumber())}
            try:
                number_value = float(window.math.number(result))
                return {"type": "scalar", "value": number_value}
            except Exception:
                converted = LinearAlgebraUtils._convert_js_value(result)
                return {"type": "scalar", "value": converted}

        if value_type == "Array":
            content = LinearAlgebraUtils._convert_js_value(result)
            return LinearAlgebraUtils._wrap_sequence(content)

        if isinstance(result, (int, float)):
            return {"type": "scalar", "value": float(result)}

        if hasattr(result, "toArray"):
            content = LinearAlgebraUtils._convert_js_value(result.toArray())
            return LinearAlgebraUtils._wrap_sequence(content)

        if hasattr(result, "toNumber"):
            return {"type": "scalar", "value": float(result.toNumber())}

        if isinstance(result, str):
            return {"type": "string", "value": result}

        return {"type": "unknown", "value": result}

    @staticmethod
    def _convert_js_value(value: Any) -> Any:
        try:
            serialized = window.JSON.stringify(value)
        except Exception:
            return value

        try:
            return json.loads(serialized)
        except Exception:
            return value

    @staticmethod
    def _wrap_sequence(sequence: Any) -> LinearAlgebraResult:
        if isinstance(sequence, list) and sequence and isinstance(sequence[0], list):
            return {"type": "matrix", "value": sequence}
        return {"type": "vector", "value": sequence}
 