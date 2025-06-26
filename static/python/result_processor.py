"""
MatHud Function Call Result Processing

Handles execution of AI function calls and aggregation of their results.
Manages state archiving, computation tracking, and error handling for function execution.

Key Features:
    - Function call validation and execution
    - Result formatting and type consistency
    - State archiving for undoable operations
    - Computation history integration
    - Error handling and exception management
    - Expression evaluation result processing

Processing Flow:
    1. Input validation for function calls and available functions
    2. State archiving for undoable operations
    3. Individual function call execution with error handling
    4. Result formatting and key generation
    5. Computation history integration (for mathematical operations)
    6. Result aggregation into structured dictionary

Dependencies:
    - constants: Success message definitions for result formatting
"""

from constants import successful_call_message

class ResultProcessor:
    """Handles the processing of function calls and their results.
    
    Executes AI function calls, manages state for undoable operations, and aggregates
    results with proper formatting and error handling. Integrates with canvas
    computation history for mathematical operations.
    """
    
    @staticmethod
    def get_results(calls, available_functions, undoable_functions, canvas):
        """
        Process function calls and collect their results.
        
        Args:
            calls: List of function call dictionaries
            available_functions: Dictionary mapping function names to implementations
            undoable_functions: Tuple of function names that are undoable
            canvas: Canvas instance for archiving state and adding computations
            
        Returns:
            Dictionary mapping function call strings to their results
        """
        ResultProcessor._validate_inputs(calls, available_functions, undoable_functions)
        
        results = {}  # Use a dictionary for results
        non_computation_functions, unformattable_functions = ResultProcessor._prepare_helper_variables(undoable_functions)
        
        # Archive once at the start and then suspend archiving while calling undoable functions
        contains_undoable_function = any(call.get('function_name', '') in undoable_functions for call in calls)
        if contains_undoable_function:
            canvas.archive()
            
        # Process each function call
        for call in calls:
            try:
                ResultProcessor._process_function_call(call, available_functions, 
                                                     non_computation_functions, unformattable_functions, canvas, results)
            except Exception as e:
                function_name = call.get('function_name', '')
                ResultProcessor._handle_exception(e, function_name, results)
        
        return results
    
    @staticmethod
    def _validate_inputs(calls, available_functions, undoable_functions):
        """Validate the input parameters."""
        if not isinstance(calls, list):
            raise ValueError("Invalid input for calls.")
        if not isinstance(available_functions, dict):
            raise ValueError("Invalid input for available_functions.")
        if not isinstance(undoable_functions, tuple):
            raise ValueError("Invalid input for undoable_functions.")
    
    @staticmethod
    def _prepare_helper_variables(undoable_functions):
        """Prepare helper variables needed for processing."""
        unformattable_functions = undoable_functions + ('undo', 'redo')
        non_computation_functions = unformattable_functions + ('run_tests', 'list_workspaces', 
                                                              'save_workspace', 'load_workspace', 
                                                              'delete_workspace')
        return non_computation_functions, unformattable_functions
    
    @staticmethod
    def _process_function_call(call, available_functions, 
                              non_computation_functions, unformattable_functions, canvas, results):
        """
        Process a single function call and update results.
        
        Args:
            call: Dictionary containing function call information
            available_functions: Dictionary mapping function names to implementations
            non_computation_functions: Tuple of function names that don't add computations
            unformattable_functions: Tuple of function names that return standard success message
            canvas: Canvas instance for adding computations
            results: Dictionary to update with the results
        """
        function_name = call.get('function_name', '')
        
        # Check if function exists
        if not ResultProcessor._is_function_available(function_name, available_functions, results):
            return
                
        # Execute the function
        args = call.get('arguments', {})
        result = ResultProcessor._execute_function(function_name, args, available_functions)
        
        # Format the key for results dictionary
        key = ResultProcessor._generate_result_key(function_name, args)
        
        # Process the result based on function type
        ResultProcessor._process_result(function_name, args, result, key, unformattable_functions,
                                       non_computation_functions, canvas, results)
    
    @staticmethod
    def _is_function_available(function_name, available_functions, results):
        """Check if the function exists and update results if not."""
        if function_name not in available_functions:
            error_msg = f"Error: function {function_name} not found."
            print(error_msg)  # DEBUG
            results[function_name] = error_msg
            return False
        return True
    
    @staticmethod
    def _execute_function(function_name, args, available_functions):
        """Execute the function with the provided arguments."""
        result = available_functions[function_name](**args)
        print(f"Called function {function_name} with args {args}. Result: {result}")   # DEBUG
        return result
    
    @staticmethod
    def _generate_result_key(function_name, args):
        """Generate a consistent key format for the results dictionary."""
        formatted_args = ResultProcessor._format_arguments(args)
        return f"{function_name}({formatted_args})"
    
    @staticmethod
    def _process_result(function_name, args, result, key, unformattable_functions,
                       non_computation_functions, canvas, results):
        """Process the result based on function type and update results dictionary."""
        if function_name in unformattable_functions:
            # Handle unformattable functions (return success message)
            ResultProcessor._handle_unformattable_function(key, results)
        elif function_name == 'evaluate_expression' and 'expression' in args:
            # Handle expression evaluation
            ResultProcessor._handle_expression_evaluation(args, result, function_name, 
                                                         non_computation_functions, canvas, results)
        else:
            # Handle regular functions
            ResultProcessor._handle_regular_function(key, result, function_name, 
                                                   non_computation_functions, canvas, results)
    
    @staticmethod
    def _handle_unformattable_function(key, results):
        """Handle result for unformattable functions."""
        results[key] = successful_call_message
    
    @staticmethod
    def _handle_regular_function(key, result, function_name, non_computation_functions, canvas, results):
        """Handle result for regular functions."""
        # Save computation to canvas state if it's not a non-computation function
        # DISABLED: Saving basic calculations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions, key, canvas)
        
        print(f"Appending result for {key}: {result}")  # DEBUG
        results[key] = result
    
    @staticmethod
    def _format_arguments(args):
        """Format function arguments for display."""
        return ', '.join(f"{k}:{v}" for k, v in args.items() if k != 'canvas')
    
    @staticmethod
    def _add_computation_if_needed(result, function_name, non_computation_functions, 
                                  expression, canvas):
        """Add the computation to canvas if it's not a non-computation function and succeeded."""
        if (not isinstance(result, str) or not result.startswith("Error:")) and \
           function_name not in non_computation_functions:
            canvas.add_computation(
                expression=expression,
                result=result
            )
    
    @staticmethod
    def _handle_exception(exception, function_name, results):
        """
        Handle exceptions during function calls.
        
        Args:
            exception: The exception that was raised
            function_name: Name of the function that caused the exception
            results: Dictionary to update with the error information
        """
        error_message = f"Error calling function {function_name}: {exception}"
        print(error_message)  # DEBUG
        
        # Use the function name as the key for storing the error
        key = function_name
        
        # Store the error message as the result value
        results[key] = f"Error: {str(exception)}"
    
    @staticmethod
    def _handle_expression_evaluation(args, result, function_name, non_computation_functions, canvas, results):
        """
        Handle the special case of expression evaluation results.
        
        Args:
            args: Arguments dictionary for the function call
            result: Result of the function call
            function_name: Name of the function that was called
            non_computation_functions: List of functions that shouldn't be added to computations
            canvas: Canvas instance for adding computations
            results: Dictionary to update with the result
        """
        expression = args.get('expression').replace(' ', '')
        key = ResultProcessor._format_expression_key(expression, args)
            
        # DISABLED: Saving expression evaluation computations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions, 
        #                                            expression, canvas)
        
        print(f"Appending result for {key}: {result}")  # DEBUG
        results[key] = result
    
    @staticmethod
    def _format_expression_key(expression, args):
        """Format a key for expression evaluation results."""
        if 'variables' in args:
            variables_dict = args.get('variables', {})
            if not isinstance(variables_dict, dict):
                variables_dict = {}
            variables = ', '.join(f"{k}:{v}" for k, v in variables_dict.items())
            return f"{expression} for {variables}"
        else:
            return expression 