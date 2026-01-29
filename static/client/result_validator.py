"""
MatHud Result Validation System

Validates the structure and content of AI function call results.
Ensures data integrity and distinguishes between successful computations and errors.

Validation Features:
    - Structure validation: Checks result dictionary format and types
    - Success detection: Identifies successful vs failed computations
    - Type safety: Enforces allowed data types for results

Dependencies:
    - constants: Success message definitions
"""

from __future__ import annotations

from typing import Any, Dict

from constants import successful_call_message


class ResultValidator:
    """Handles validation of function call results and error detection.
    
    Provides static methods for validating result structure and determining
    whether function calls completed successfully or encountered errors.
    """
    
    @staticmethod
    def validate_results(results: Dict[str, Any]) -> bool:
        """Validates that results have the correct structure and data types.
        
        Checks that the results dictionary contains only allowed types and
        has proper string keys, regardless of whether the results contain errors.
        
        Args:
            results (dict): Dictionary of results to validate with string keys
            
        Returns:
            bool: True if the results have valid structure, False otherwise
        """
        if not isinstance(results, dict):
            return False
        return all(
            k and isinstance(k, str) and ResultValidator._is_allowed_value(v, allow_list=False)
            for k, v in results.items()
        )
    
    @staticmethod
    def is_successful_result(value: Any) -> bool:
        """Checks if a result value represents a successful computation.
        
        Determines whether a result contains actual computed data vs error messages,
        empty values, or success acknowledgment messages.
        
        Args:
            value: The result value to check (any type)
            
        Returns:
            bool: True if the result represents successful computation, False for errors/empty
        """
        if isinstance(value, str):
            if value in [successful_call_message, ""]:
                return False
            if value.startswith("Error:"):
                return False
            return True

        if isinstance(value, dict):
            value_type = value.get("type") if isinstance(value.get("type"), str) else None
            if value_type == "error":
                return False
            return True

        return True

    @staticmethod
    def _is_allowed_value(value: Any, *, allow_list: bool) -> bool:
        """Determine if a result value is permitted."""

        if value is None:
            return bool(allow_list)

        if isinstance(value, (str, int, float, bool)):
            return True

        if isinstance(value, list):
            if not allow_list:
                return False
            return all(ResultValidator._is_allowed_value(item, allow_list=True) for item in value)

        if isinstance(value, dict):
            value_type = value.get("type")
            if isinstance(value_type, str) and "value" in value:
                return ResultValidator._is_allowed_value(value["value"], allow_list=True)
            return all(
                isinstance(k, str) and ResultValidator._is_allowed_value(v, allow_list=allow_list)
                for k, v in value.items()
            )

        return False