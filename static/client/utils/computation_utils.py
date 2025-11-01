"""
MatHud Computation Utilities Module

Computation history management and expression tracking utilities.
Provides functions for managing mathematical computation history and avoiding duplicate calculations.

Key Features:
    - Computation history tracking
    - Duplicate expression detection
    - Expression-result pair management
    - Computation history persistence

History Management:
    - Expression uniqueness validation
    - Result caching and retrieval
    - Computation deduplication
    - History list maintenance

Use Cases:
    - AI computation result tracking
    - Mathematical expression caching
    - Computation history display
    - Performance optimization through caching

Dependencies:
    - None (pure data management utilities)
"""

from __future__ import annotations

from typing import Any, Dict, List


class ComputationUtils:
    """Computation history management utilities for mathematical expression tracking.
    
    Provides static methods for managing computation history, detecting duplicates,
    and maintaining a record of mathematical calculations performed in the canvas.
    """
    @staticmethod
    def has_computation(computations: List[Dict[str, Any]], expression: str) -> bool:
        """
        Check if a computation with the given expression already exists.
        
        Args:
            computations: List of computation dictionaries
            expression: Expression to check
            
        Returns:
            bool: True if the computation exists, False otherwise
        """
        return any(comp["expression"] == expression for comp in computations)

    @staticmethod
    def add_computation(computations: List[Dict[str, Any]], expression: str, result: Any) -> List[Dict[str, Any]]:
        """
        Add a computation to the history if it doesn't already exist.
        
        Args:
            computations: List of computation dictionaries
            expression: Expression to add
            result: Result of the computation
            
        Returns:
            list: Updated list of computations
        """
        if not ComputationUtils.has_computation(computations, expression):
            computations.append({
                "expression": expression,
                "result": result
            })
        return computations 