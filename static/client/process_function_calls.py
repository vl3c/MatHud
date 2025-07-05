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

from expression_evaluator import ExpressionEvaluator
from result_processor import ResultProcessor
from result_validator import ResultValidator


class ProcessFunctionCalls:
    """Main facade for processing function calls, evaluating expressions, and validating results.
    
    This class serves as a unified interface for the various specialized processing modules,
    maintaining backward compatibility while delegating to appropriate specialized handlers.
    """
    
    @staticmethod
    def evaluate_expression(expression, variables=None, canvas=None):
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
    def get_results(calls, available_functions, undoable_functions, canvas):
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
        return ResultProcessor.get_results(calls, available_functions, undoable_functions, canvas)
    
    @staticmethod
    def validate_results(results):
        """Validates result structure and data types for integrity.
        
        Delegates to ResultValidator for structure and type validation.
        
        Args:
            results (dict): Dictionary of results to validate
            
        Returns:
            bool: True if results have correct structure, False otherwise
        """
        return ResultValidator.validate_results(results)
    
    @staticmethod
    def is_successful_result(value):
        """Checks if a result represents successful computation vs error.
        
        Delegates to ResultValidator for success detection logic.
        
        Args:
            value: The result value to check for success
            
        Returns:
            bool: True if result is successful, False for errors or empty values
        """
        return ResultValidator.is_successful_result(value)