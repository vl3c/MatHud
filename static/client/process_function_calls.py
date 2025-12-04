"""
MatHud Function Call Processing Facade

Main coordinator for processing AI function calls, expression evaluation, and result validation.
Provides a unified interface to the various specialized processing modules for backward compatibility.

Key Features:
    - Expression evaluation with variable substitution
    - Function call execution and result collection
    - Result validation and error detection
    - Canvas state management during processing

Dependencies:
    - expression_evaluator: Mathematical expression parsing and computation
    - result_processor: Function call execution and result aggregation
    - result_validator: Result structure and success validation
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from expression_evaluator import ExpressionEvaluator
from result_processor import ResultProcessor
from result_validator import ResultValidator
from utils.linear_algebra_utils import LinearAlgebraUtils, LinearAlgebraObject, LinearAlgebraResult
from utils.area_expression_evaluator import AreaExpressionEvaluator, AreaExpressionResult

if TYPE_CHECKING:
    from canvas import Canvas


class ProcessFunctionCalls:
    """Main facade for processing function calls, evaluating expressions, and validating results.
    
    This class serves as a unified interface for the various specialized processing modules,
    maintaining backward compatibility while delegating to appropriate specialized handlers.
    """
    
    @staticmethod
    def evaluate_expression(
        expression: str,
        variables: Optional[Dict[str, Any]] = None,
        canvas: Optional["Canvas"] = None,
    ) -> Any:
        """Evaluates a mathematical expression with optional variable substitution.
        
        Delegates to ExpressionEvaluator for mathematical computation and parsing.
        
        Args:
            expression (str): The mathematical expression to evaluate
            variables (dict, optional): Dictionary of variables for substitution
            canvas (Canvas, optional): Canvas instance for function evaluation context
            
        Returns:
            The computed result of the expression evaluation
        """
        return ExpressionEvaluator.evaluate_expression(expression, variables, canvas)
    
    @staticmethod
    def evaluate_linear_algebra_expression(
        objects: List[LinearAlgebraObject],
        expression: str
    ) -> LinearAlgebraResult:
        """Evaluates a linear algebra expression using predefined objects.
        
        Delegates to LinearAlgebraUtils for matrix and vector validation and
        MathJS-backed evaluation.

        Args:
            objects (list): List of object definitions containing names and values
            expression (str): Linear algebra expression referencing the objects

        Returns:
            The evaluated result converted into Python-native structures.
        """

        return LinearAlgebraUtils.evaluate_expression(objects, expression)
    
    @staticmethod
    def get_results(
        calls: List[Dict[str, Any]],
        available_functions: Dict[str, Any],
        undoable_functions: Tuple[str, ...],
        canvas: "Canvas",
    ) -> Dict[str, Any]:
        """Process function calls and collect their results with state management.
        
        Delegates to ResultProcessor for function execution and result aggregation.
        
        Args:
            calls (list): List of function call dictionaries from AI
            available_functions (dict): Mapping of function names to implementations
            undoable_functions (tuple): Function names that support undo operations
            canvas (Canvas): Canvas instance for state archiving and computation tracking
            
        Returns:
            dict: Mapping of function call strings to their computed results
        """
        return cast(
            Dict[str, Any],
            ResultProcessor.get_results(calls, available_functions, undoable_functions, canvas),
        )
    
    @staticmethod
    def validate_results(results: Dict[str, Any]) -> bool:
        """Validates result structure and data types for integrity.
        
        Delegates to ResultValidator for structure and type validation.
        
        Args:
            results (dict): Dictionary of results to validate
            
        Returns:
            bool: True if results have correct structure, False otherwise
        """
        return bool(ResultValidator.validate_results(results))
    
    @staticmethod
    def is_successful_result(value: Any) -> bool:
        """Checks if a result represents successful computation vs error.
        
        Delegates to ResultValidator for success detection logic.
        
        Args:
            value: The result value to check for success
            
        Returns:
            bool: True if result is successful, False for errors or empty values
        """
        return bool(ResultValidator.is_successful_result(value))
    
    @staticmethod
    def calculate_area(expression: str, canvas: "Canvas") -> Dict[str, Any]:
        """Calculate the area of a region defined by a boolean expression.
        
        Delegates to AreaExpressionEvaluator for parsing and computation.
        
        Args:
            expression: Boolean expression with drawable names
            canvas: Canvas instance to resolve drawable names
            
        Returns:
            Dictionary with 'type' and 'value' keys
        """
        result: AreaExpressionResult = AreaExpressionEvaluator.evaluate(expression, canvas)
        return result.to_dict()