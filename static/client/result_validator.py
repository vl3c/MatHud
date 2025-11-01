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
        allowed_types: tuple[type, ...] = (str, int, float, bool)
        if not isinstance(results, dict):
            return False
        print(f"Validating results: {results}")
        return all(
            k and isinstance(k, str) and 
            isinstance(v, allowed_types)
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
        return not (isinstance(value, str) and 
                   (value in [successful_call_message, ""] or 
                    value.startswith("Error:"))) 