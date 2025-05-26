"""
MatHud Mathematical Expression Evaluator

Handles evaluation of mathematical expressions and function calls within the canvas system.
Supports both numeric computations and custom function evaluation with error handling.

Evaluation Types:
    - Numeric expressions: Basic mathematical operations with variables
    - Function expressions: Canvas-defined functions with argument substitution
    - Hybrid evaluation: Fallback from numeric to function evaluation

Dependencies:
    - utils.math_utils: Core mathematical computation engine
    - re: Regular expression parsing for function calls
"""

from utils.math_utils import MathUtils
import re


class ExpressionEvaluator:
    """Handles evaluation of mathematical expressions and custom functions.
    
    Provides static methods for evaluating numeric expressions and canvas-defined
    functions with comprehensive error handling and type consistency.
    """
    
    @staticmethod
    def evaluate_numeric_expression(expression, variables):
        """Evaluates a numeric mathematical expression with variable substitution.
        
        Uses MathUtils for computation and ensures consistent float return type.
        
        Args:
            expression (str): Mathematical expression string to evaluate
            variables (dict): Dictionary of variable names to values for substitution
            
        Returns:
            float: The computed numeric result
        """
        result = MathUtils.evaluate(expression, variables)
        print(f"Evaluated numeric expression: {expression} = {result}")   # DEBUG
        # Convert numeric results to float for consistency
        if isinstance(result, (int, float)):
            result = float(result)
        return result
    
    @staticmethod
    def evaluate_function(expression, canvas):
        """Evaluates a function expression using canvas-defined functions.
        
        Parses function call syntax and evaluates using functions stored in the canvas.
        
        Args:
            expression (str): Function expression in format "function_name(argument)"
            canvas (Canvas): Canvas instance containing function definitions
            
        Returns:
            float: The computed function result
            
        Raises:
            ValueError: If canvas is None, expression format is invalid, or function not found
        """
        print(f"Evaluating function with expression: {expression}")   # DEBUG
        if canvas is None:
            raise ValueError("Cannot evaluate function: no canvas available")
        
        functions = canvas.get_drawables_by_class_name('Function')
        # Split the expression into function name and argument
        match = re.match(r'(\w+)\((.+)\)', expression)
        if match:
            function_name, argument = match.groups()
            print(f"Function name: {function_name}, argument: {argument}")   # DEBUG
        else:
            raise ValueError(f"Invalid function expression: {expression}")
        
        for function in functions:
            if function.name.lower() == function_name.lower():
                # If the function name matches, evaluate the function
                print(f"Found function: {function.name} = {function.function_string}")   # DEBUG
                try:
                    argument = float(argument)  # Convert argument to float
                    result = function.function(argument)
                    # Convert result to float for consistency
                    if isinstance(result, (int, float)):
                        result = float(result)
                    return result
                except ValueError:
                    raise ValueError(f"Invalid argument for function: {argument}")
        
        # If we get here, no matching function was found
        raise ValueError(f"No function found with name: {function_name}")
    
    @staticmethod
    def evaluate_expression(expression, variables=None, canvas=None):
        """Main method to evaluate expressions with fallback from numeric to function evaluation.
        
        First attempts numeric evaluation, then falls back to function evaluation if available.
        Provides comprehensive error handling with user-friendly messages.
        
        Args:
            expression (str): Expression to evaluate (numeric or function call)
            variables (dict, optional): Variable substitutions for numeric expressions
            canvas (Canvas, optional): Canvas instance for function evaluation
            
        Returns:
            float or str: Computed result or error message
        """
        try:
            # First, try to evaluate the expression as a numeric expression
            result = ExpressionEvaluator.evaluate_numeric_expression(expression, variables)
            if not result or (isinstance(result, str) and "Error" in result):
                raise ValueError(f"Error evaluating numeric expression")
            return result
        except Exception as e:
            bad_result_msg = "Sorry, that's not a supported mathematical expression."
            try:
                # If numeric evaluation fails and we have a canvas, try to evaluate as a function
                if canvas is not None:
                    result = ExpressionEvaluator.evaluate_function(expression, canvas)
                    if not result:
                        return bad_result_msg
                    return result
                else:
                    # If no canvas is available, just return the numeric evaluation error
                    return f"{bad_result_msg} Error: {str(e)}"
            except Exception as e:
                exception_details = str(e).split(":", 1)[0]
                result = f"{bad_result_msg} Exception for ({expression}): {exception_details}."
                return result 