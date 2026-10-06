"""
MatHud Tool Argument Validator

Centralized, schema-driven validation and canonicalization layer for AI tool call
arguments. Validates every tool call against the JSON schemas defined in
functions_definitions.py before arguments are sent to the browser client.

Operates in log-only mode: validates and canonicalizes arguments, logs structured
warnings for any issues found, but never blocks tool execution. Canonicalized
arguments are used when validation passes; original arguments pass through on failure.

Supported JSON Schema keywords: type, properties, required, additionalProperties,
enum, items, anyOf, minItems, maxLength.

Canonicalization rules (applied when schema type is matched):
    - String-to-number coercion: "5" -> 5.0 for number fields
    - String-to-integer coercion: "200" -> 200 for integer fields
    - Empty-string-to-null: "" -> None for nullable string fields
    - Null-string-to-null: "null", "None", "undefined" -> None for any field whose
      schema allows null, free-text fields (expressions, labels, text) excepted;
      applied before validation, so also when validation fails. The routes apply
      it to every model reply's tool calls (``normalize_tool_calls``), streaming
      or not, before the client runs them.
    - NaN/Infinity rejection: float('nan'), float('inf') are invalid numbers
    - int pass-through for number fields: 5 stays as 5 (JSON doesn't distinguish)
    - bool is NOT a number: True/False rejected for number/integer fields

Developer guide for adding new tools:
    1. Define the JSON schema in functions_definitions.py with "strict": True,
       "additionalProperties": False, and explicit "required" array.
    2. The validator automatically picks up new schemas at import time.
    3. Add test cases to test_tool_argument_validator.py for the new tool.
    4. Supported schema keywords: type, enum, properties, required,
       additionalProperties, items, anyOf, minItems, maxLength.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# Maximum length for string values included in error messages to prevent log flooding.
_ERROR_VALUE_MAX_LEN = 100

# Strings some models (e.g. Qwen through llama-server) send for an optional argument they
# leave unset. They mean null wherever the schema allows null.
NULL_STRINGS = frozenset({"null", "None", "undefined"})
# Free-text arguments whose value is shown or parsed as written: a label or text "null" is
# kept as text even where the schema allows null.
NULL_STRING_FREE_TEXT = frozenset(
    {"expression", "label", "label_text", "new_label_text", "new_text", "text", "query", "function_string"}
)


class ValidationResult(TypedDict):
    """Result of validating tool call arguments."""

    valid: bool
    arguments: Dict[str, Any]  # Original or canonicalized arguments
    errors: List[str]  # Empty if valid


# ---------------------------------------------------------------------------
# Schema Index — built once at import time
# ---------------------------------------------------------------------------

_SCHEMA_INDEX: Dict[str, Dict[str, Any]] = {}

try:
    from static.functions_definitions import FUNCTIONS

    for _entry in FUNCTIONS:
        if not isinstance(_entry, dict):
            continue
        if _entry.get("type") != "function":
            continue
        _fn = _entry.get("function")
        if not isinstance(_fn, dict):
            continue
        _name = _fn.get("name")
        if isinstance(_name, str) and _name:
            _SCHEMA_INDEX[_name] = _fn.get("parameters", {})
except Exception:
    logger.error(
        "Failed to import FUNCTIONS from functions_definitions; "
        "schema index is empty — all tool calls will pass through unvalidated.",
        exc_info=True,
    )


def _truncate(value: Any) -> str:
    """Truncate a value's repr for safe inclusion in error messages."""
    s = repr(value)
    if len(s) > _ERROR_VALUE_MAX_LEN:
        return s[:_ERROR_VALUE_MAX_LEN] + "..."
    return s


def _type_label(schema_type: Any) -> str:
    """Human-readable label for a schema type declaration."""
    if isinstance(schema_type, list):
        return " | ".join(str(t) for t in schema_type)
    return str(schema_type)


def _python_type_name(value: Any) -> str:
    """Short Python type name for error messages."""
    return type(value).__name__


# ---------------------------------------------------------------------------
# Type checking helpers
# ---------------------------------------------------------------------------


def _matches_single_type(value: Any, type_name: str) -> bool:
    """Check whether *value* matches a single JSON Schema type keyword."""
    if type_name == "number":
        # bool is a subclass of int in Python, but JSON treats them differently.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "null":
        return value is None
    return False


def _matches_type(value: Any, schema_type: Any) -> bool:
    """Check whether *value* matches a (possibly union) JSON Schema type."""
    if isinstance(schema_type, list):
        return any(_matches_single_type(value, t) for t in schema_type)
    if isinstance(schema_type, str):
        return _matches_single_type(value, schema_type)
    # No type declared — anything matches.
    return True


def _is_numeric_type(schema_type: Any) -> bool:
    """Return True if *schema_type* includes 'number' (but not only 'null')."""
    if isinstance(schema_type, list):
        return bool("number" in schema_type)
    return bool(schema_type == "number")


def _is_integer_type(schema_type: Any) -> bool:
    """Return True if *schema_type* includes 'integer'."""
    if isinstance(schema_type, list):
        return bool("integer" in schema_type)
    return bool(schema_type == "integer")


def _is_nullable(schema_type: Any) -> bool:
    """Return True if the schema type allows null."""
    if isinstance(schema_type, list):
        return bool("null" in schema_type)
    return bool(schema_type == "null")


def _schema_allows_null(schema: Dict[str, Any]) -> bool:
    """Return True if the property schema accepts null (its type, or one of its anyOf alternatives)."""
    if _is_nullable(schema.get("type")):
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(isinstance(alt, dict) and _schema_allows_null(alt) for alt in any_of)


def _normalize_null_strings(value: Any, schema: Dict[str, Any], path: str, tool_name: str) -> Any:
    """Return *value* with null spelled as a string replaced by None where *schema* allows null.

    Recurses into object properties and array items, so nested optional fields are
    covered too. Free-text properties (``NULL_STRING_FREE_TEXT``) keep their text.
    """
    if isinstance(value, str):
        if value in NULL_STRINGS and _schema_allows_null(schema):
            logger.info("Tool '%s': argument '%s' canonicalized from string %r to null.", tool_name, path, value)
            return None
        return value
    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        return {
            key: (
                item
                if key in NULL_STRING_FREE_TEXT or not isinstance(properties.get(key), dict)
                else _normalize_null_strings(item, properties[key], f"{path}.{key}" if path else key, tool_name)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        items = schema.get("items")
        if not isinstance(items, dict):
            return value
        return [_normalize_null_strings(item, items, f"{path}[{i}]", tool_name) for i, item in enumerate(value)]
    return value


def _check_nan_inf(value: Any) -> bool:
    """Return True if *value* is NaN or Infinity."""
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


# ---------------------------------------------------------------------------
# Core recursive validation / canonicalization
# ---------------------------------------------------------------------------


def _validate_value(
    value: Any,
    schema: Dict[str, Any],
    path: str,
    tool_name: str,
    errors: List[str],
    canonical: Dict[str, Any] | None,
    canonical_key: str | int | None,
    canonical_container: Any | None,
) -> Any:
    """Validate *value* against *schema* and return the (possibly canonicalized) value.

    Errors are appended to *errors*.  When canonicalization is applied, the
    canonical value is written back to *canonical_container[canonical_key]* if
    those are provided, and also returned.
    """
    schema_type = schema.get("type")

    # --- anyOf handling ---
    any_of = schema.get("anyOf")
    if any_of is not None:
        for alt in any_of:
            test_errors: List[str] = []
            test_value = copy.deepcopy(value)
            # Create a temporary container for canonical writes
            tmp: Dict[str, Any] = {"v": test_value}
            _validate_value(test_value, alt, path, tool_name, test_errors, None, "v", tmp)
            if not test_errors:
                # This alternative matched; use its canonical form.
                result = tmp["v"]
                if canonical_container is not None and canonical_key is not None:
                    canonical_container[canonical_key] = result
                return result
        # None of the alternatives matched.
        alt_descriptions = []
        for alt in any_of:
            alt_type = alt.get("type", "unknown")
            alt_descriptions.append(_type_label(alt_type))
        errors.append(
            f"Tool '{tool_name}': argument '{path}' did not match any allowed type "
            f"({', '.join(alt_descriptions)}), got {_python_type_name(value)} "
            f"({_truncate(value)})."
        )
        return value

    # --- Canonicalization: string-to-number / string-to-integer coercion ---
    if isinstance(value, str) and schema_type is not None:
        if _is_numeric_type(schema_type) and not _matches_type(value, schema_type):
            try:
                coerced = float(value)
                if not (math.isnan(coerced) or math.isinf(coerced)):
                    logger.warning(
                        "Tool '%s': argument '%s' coerced from string %s to number %s.",
                        tool_name,
                        path,
                        _truncate(value),
                        coerced,
                    )
                    value = coerced
                    if canonical_container is not None and canonical_key is not None:
                        canonical_container[canonical_key] = value
            except (ValueError, OverflowError):
                pass
        elif _is_integer_type(schema_type) and not _matches_type(value, schema_type):
            try:
                coerced_int = int(value)
                # Only coerce if the string is exactly an integer representation.
                if str(coerced_int) == value.strip():
                    logger.warning(
                        "Tool '%s': argument '%s' coerced from string %s to integer %d.",
                        tool_name,
                        path,
                        _truncate(value),
                        coerced_int,
                    )
                    value = coerced_int
                    if canonical_container is not None and canonical_key is not None:
                        canonical_container[canonical_key] = value
            except (ValueError, OverflowError):
                pass

    # --- Canonicalization: empty-string-to-null for nullable strings ---
    if isinstance(value, str) and value == "" and _is_nullable(schema_type):
        logger.warning(
            "Tool '%s': argument '%s' canonicalized from empty string to null.",
            tool_name,
            path,
        )
        value = None
        if canonical_container is not None and canonical_key is not None:
            canonical_container[canonical_key] = value

    # --- Type validation ---
    if schema_type is not None and not _matches_type(value, schema_type):
        errors.append(
            f"Tool '{tool_name}': argument '{path}' expected type "
            f"'{_type_label(schema_type)}', got {_python_type_name(value)} "
            f"({_truncate(value)})."
        )
        return value

    # --- NaN / Infinity rejection for numeric values ---
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if _check_nan_inf(value):
            errors.append(f"Tool '{tool_name}': argument '{path}' must be a finite number, got {_truncate(value)}.")
            return value

    # --- Enum validation ---
    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"Tool '{tool_name}': argument '{path}' must be one of {enum_values}, got {_truncate(value)}.")
        return value

    # --- maxLength validation (strings) ---
    max_length = schema.get("maxLength")
    if max_length is not None and isinstance(value, str):
        if len(value) > max_length:
            errors.append(
                f"Tool '{tool_name}': argument '{path}' must be at most {max_length} characters, got {len(value)}."
            )

    # --- Array validation ---
    if isinstance(value, list):
        # minItems check
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(
                f"Tool '{tool_name}': argument '{path}' must have at least {min_items} items, got {len(value)}."
            )

        # Validate each element against items schema
        items_schema = schema.get("items")
        if items_schema is not None:
            for i, item in enumerate(value):
                value[i] = _validate_value(
                    item,
                    items_schema,
                    f"{path}[{i}]",
                    tool_name,
                    errors,
                    canonical,
                    i,
                    value,
                )

    # --- Object validation ---
    if isinstance(value, dict):
        properties = schema.get("properties")
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        if properties is not None:
            # Check for required fields
            for req_key in required:
                if req_key not in value:
                    errors.append(
                        f"Tool '{tool_name}': missing required argument '{path}.{req_key}'."
                        if path
                        else f"Tool '{tool_name}': missing required argument '{req_key}'."
                    )

            # Check for unknown keys
            if additional is False:
                allowed_keys = set(properties.keys())
                for key in value:
                    if key not in allowed_keys:
                        allowed_list = sorted(allowed_keys)
                        errors.append(
                            f"Tool '{tool_name}': unknown argument '{path}.{key}' (allowed: {', '.join(allowed_list)})."
                            if path
                            else f"Tool '{tool_name}': unknown argument '{key}' (allowed: {', '.join(allowed_list)})."
                        )

            # Recurse into each property
            for key, prop_schema in properties.items():
                if key in value:
                    value[key] = _validate_value(
                        value[key],
                        prop_schema,
                        f"{path}.{key}" if path else key,
                        tool_name,
                        errors,
                        canonical,
                        key,
                        value,
                    )

    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ToolArgumentValidator:
    """Validates and canonicalizes tool call arguments against JSON schemas.

    Uses the schemas defined in functions_definitions.py to validate argument
    types, required fields, enum values, and nested structures. Provides
    clear, actionable error messages including tool name, argument path,
    expected type, and received value.

    Usage::

        result = ToolArgumentValidator.validate("create_circle", args)
        if not result["valid"]:
            # Handle errors: result["errors"] contains human-readable messages
            ...
        # Use result["arguments"] which may contain canonicalized values
    """

    @staticmethod
    def validate(function_name: str, arguments: Dict[str, Any]) -> ValidationResult:
        """Validate arguments for a tool call.

        Args:
            function_name: Name of the tool function.
            arguments: Arguments dict parsed from the tool call.

        Returns:
            ValidationResult with valid flag, (possibly canonicalized) arguments,
            and list of error messages if invalid.
        """
        # Graceful handling of None arguments
        if arguments is None:
            arguments = {}

        schema = _SCHEMA_INDEX.get(function_name)

        if schema is None:
            # Unknown function — log a warning but pass through.
            logger.warning(
                "Tool '%s': no schema found in registry; arguments pass through unvalidated.",
                function_name,
            )
            return ValidationResult(
                valid=True,
                arguments=arguments,
                errors=[],
            )

        # Null spelled as a string means null; this applies even when validation fails.
        arguments = ToolArgumentValidator.normalize_null_strings(function_name, arguments)

        # Deep-copy arguments for canonicalization so the original is untouched.
        canonical_args = copy.deepcopy(arguments)
        errors: List[str] = []

        # Top-level schema is always type=object; validate its structure directly.
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        # Required fields
        for req_key in required:
            if req_key not in canonical_args:
                errors.append(f"Tool '{function_name}': missing required argument '{req_key}'.")

        # Unknown keys
        if additional is False and properties is not None:
            allowed_keys = set(properties.keys())
            for key in canonical_args:
                if key not in allowed_keys:
                    allowed_list = sorted(allowed_keys)
                    errors.append(
                        f"Tool '{function_name}': unknown argument '{key}' (allowed: {', '.join(allowed_list)})."
                    )

        # Validate each property
        for key, prop_schema in properties.items():
            if key not in canonical_args:
                continue
            canonical_args[key] = _validate_value(
                canonical_args[key],
                prop_schema,
                key,
                function_name,
                errors,
                canonical_args,
                key,
                canonical_args,
            )

        if errors:
            return ValidationResult(
                valid=False,
                arguments=arguments,  # The original on failure, with null strings as null
                errors=errors,
            )

        return ValidationResult(
            valid=True,
            arguments=canonical_args,
            errors=[],
        )

    @staticmethod
    def normalize_null_strings(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return the arguments with ``"null"``, ``"None"`` and ``"undefined"`` as None where null is allowed.

        Only the exact strings, and only for properties whose schema allows null
        (optional ones; required properties never do). Free-text properties
        (``NULL_STRING_FREE_TEXT``: expressions, labels, text) keep the string.
        Unknown tools and non-dict arguments come back unchanged; the input is
        never modified.
        """
        schema = _SCHEMA_INDEX.get(function_name)
        if schema is None or not isinstance(arguments, dict):
            return arguments
        normalized = _normalize_null_strings(arguments, schema, "", function_name)
        return normalized if isinstance(normalized, dict) else arguments

    @staticmethod
    def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """``normalize_null_strings`` applied to every ``{"function_name", "arguments"}`` call of a model reply.

        The tool-call path of every provider runs this before the calls go to the
        client, which runs them. Calls are copied, never modified.
        """
        normalized: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("function_name") if isinstance(call, dict) else None
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if isinstance(name, str) and isinstance(arguments, dict):
                call = dict(call)
                call["arguments"] = ToolArgumentValidator.normalize_null_strings(name, arguments)
            normalized.append(call)
        return normalized

    @staticmethod
    def get_schema(function_name: str) -> Optional[Dict[str, Any]]:
        """Look up the JSON schema for a function by name.

        Returns None if the function is not found in the registry.
        """
        return _SCHEMA_INDEX.get(function_name)
