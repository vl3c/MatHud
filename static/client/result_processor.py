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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from browser import window

from constants import successful_call_message

if TYPE_CHECKING:
    from canvas import Canvas

    TracedCall = Dict[str, Any]
    """Per-call trace record: seq, function_name, arguments, result, is_error, duration_ms."""


class ResultProcessor:
    """Handles the processing of function calls and their results.

    Executes AI function calls, manages state for undoable operations, and aggregates
    results with proper formatting and error handling. Integrates with canvas
    computation history for mathematical operations.
    """

    @staticmethod
    def get_results(calls: List[Dict[str, Any]], available_functions: Dict[str, Any], undoable_functions: Tuple[str, ...], canvas: "Canvas") -> Dict[str, Any]:
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

        results: Dict[str, Any] = {}  # Use a dictionary for results
        non_computation_functions: Tuple[str, ...]
        unformattable_functions: Tuple[str, ...]
        non_computation_functions, unformattable_functions = ResultProcessor._prepare_helper_variables(undoable_functions)

        # Archive once at the start and then suspend archiving while calling undoable functions
        contains_undoable_function: bool = any(call.get('function_name', '') in undoable_functions for call in calls)
        if contains_undoable_function:
            canvas.archive()

        # Process each function call
        for call in calls:
            try:
                ResultProcessor._process_function_call(call, available_functions,
                                                     non_computation_functions, unformattable_functions, canvas, results)
            except Exception as e:
                function_name: str = call.get('function_name', '')
                ResultProcessor._handle_exception(e, function_name, results)

        return results

    @staticmethod
    def get_results_traced(
        calls: List[Dict[str, Any]],
        available_functions: Dict[str, Any],
        undoable_functions: Tuple[str, ...],
        canvas: "Canvas",
    ) -> Tuple[Dict[str, Any], List["TracedCall"]]:
        """Execute tool calls and return (results_dict, traced_calls_list).

        Same execution semantics as get_results() but additionally records
        per-call timing and result metadata for action tracing.

        Args:
            calls: List of function call dictionaries
            available_functions: Dictionary mapping function names to implementations
            undoable_functions: Tuple of function names that are undoable
            canvas: Canvas instance for archiving state and adding computations

        Returns:
            Tuple of (results dict, list of traced call records)
        """
        ResultProcessor._validate_inputs(calls, available_functions, undoable_functions)

        results: Dict[str, Any] = {}
        traced_calls: List[TracedCall] = []
        non_computation_functions: Tuple[str, ...]
        unformattable_functions: Tuple[str, ...]
        non_computation_functions, unformattable_functions = ResultProcessor._prepare_helper_variables(undoable_functions)

        # Archive once at the start (same as get_results)
        contains_undoable_function: bool = any(call.get('function_name', '') in undoable_functions for call in calls)
        if contains_undoable_function:
            canvas.archive()

        for seq, call in enumerate(calls):
            function_name = call.get('function_name', '')
            args = call.get('arguments', {})
            # Sanitize arguments for trace: exclude canvas ref, guard against non-dict
            if isinstance(args, dict):
                sanitized_args = {k: v for k, v in args.items() if k != 'canvas'}
            else:
                sanitized_args = {"_raw": args}

            t0 = window.performance.now()
            is_error = False
            result_value: Any = None
            try:
                snapshot_before = dict(results)
                ResultProcessor._process_function_call(
                    call, available_functions,
                    non_computation_functions, unformattable_functions, canvas, results,
                )
                # Extract result: find the key that was added or changed
                for rk, rv in results.items():
                    if rk not in snapshot_before or snapshot_before[rk] is not rv:
                        result_value = rv
                        break
                else:
                    # No change detected; grab by function name as last resort
                    result_value = results.get(function_name)
                if isinstance(result_value, str) and result_value.startswith("Error"):
                    is_error = True
            except Exception as e:
                ResultProcessor._handle_exception(e, function_name, results)
                result_value = results.get(function_name, str(e))
                is_error = True

            duration_ms = window.performance.now() - t0
            traced_calls.append({
                "seq": seq,
                "function_name": function_name,
                "arguments": sanitized_args,
                "result": result_value,
                "is_error": is_error,
                "duration_ms": round(duration_ms, 2),
            })

        return results, traced_calls

    @staticmethod
    def _validate_inputs(calls: List[Dict[str, Any]], available_functions: Dict[str, Any], undoable_functions: Tuple[str, ...]) -> None:
        """Validate the input parameters."""
        if not isinstance(calls, list):
            raise ValueError("Invalid input for calls.")
        if not isinstance(available_functions, dict):
            raise ValueError("Invalid input for available_functions.")
        if not isinstance(undoable_functions, tuple):
            raise ValueError("Invalid input for undoable_functions.")

    @staticmethod
    def _prepare_helper_variables(undoable_functions: Tuple[str, ...]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Prepare helper variables needed for processing."""
        unformattable_functions: Tuple[str, ...] = undoable_functions + ('undo', 'redo')
        non_computation_functions: Tuple[str, ...] = unformattable_functions + ('run_tests', 'list_workspaces',
                                                              'save_workspace', 'load_workspace',
                                                              'delete_workspace')
        return non_computation_functions, unformattable_functions

    @staticmethod
    def _process_function_call(call: Dict[str, Any], available_functions: Dict[str, Any],
                              non_computation_functions: Tuple[str, ...], unformattable_functions: Tuple[str, ...], canvas: "Canvas", results: Dict[str, Any]) -> None:
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
        function_name: str = call.get('function_name', '')

        # Check if function exists
        if not ResultProcessor._is_function_available(function_name, available_functions, results):
            return

        # Execute the function
        args: Dict[str, Any] = call.get('arguments', {})
        result: Any = ResultProcessor._execute_function(function_name, args, available_functions)

        # Format the key for results dictionary
        key: str = ResultProcessor._generate_result_key(function_name, args)

        # Process the result based on function type
        ResultProcessor._process_result(function_name, args, result, key, unformattable_functions,
                                       non_computation_functions, canvas, results)

    @staticmethod
    def _is_function_available(function_name: str, available_functions: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Check if the function exists and update results if not."""
        if function_name not in available_functions:
            error_msg: str = f"Error: function {function_name} not found."
            print(error_msg)  # DEBUG
            results[function_name] = error_msg
            return False
        return True

    @staticmethod
    def _execute_function(function_name: str, args: Dict[str, Any], available_functions: Dict[str, Any]) -> Any:
        """Execute the function with the provided arguments."""
        result: Any = available_functions[function_name](**args)
        return result

    @staticmethod
    def _generate_result_key(function_name: str, args: Dict[str, Any]) -> str:
        """Generate a consistent key format for the results dictionary."""
        formatted_args: str = ResultProcessor._format_arguments(args)
        return f"{function_name}({formatted_args})"

    @staticmethod
    def _process_result(function_name: str, args: Dict[str, Any], result: Any, key: str, unformattable_functions: Tuple[str, ...],
                       non_computation_functions: Tuple[str, ...], canvas: "Canvas", results: Dict[str, Any]) -> None:
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
    def _handle_unformattable_function(key: str, results: Dict[str, Any]) -> None:
        """Handle result for unformattable functions."""
        results[key] = successful_call_message

    @staticmethod
    def _handle_regular_function(key: str, result: Any, function_name: str, non_computation_functions: Tuple[str, ...], canvas: "Canvas", results: Dict[str, Any]) -> None:
        """Handle result for regular functions."""
        # Save computation to canvas state if it's not a non-computation function
        # DISABLED: Saving basic calculations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions, key, canvas)

        results[key] = result

    @staticmethod
    def _format_arguments(args: Dict[str, Any]) -> str:
        """Format function arguments for display."""
        return ', '.join(f"{k}:{v}" for k, v in args.items() if k != 'canvas')

    @staticmethod
    def _add_computation_if_needed(result: Any, function_name: str, non_computation_functions: Tuple[str, ...],
                                  expression: str, canvas: "Canvas") -> None:
        """Add the computation to canvas if it's not a non-computation function and succeeded."""
        if (not isinstance(result, str) or not result.startswith("Error:")) and \
           function_name not in non_computation_functions:
            canvas.add_computation(
                expression=expression,
                result=result
            )

    @staticmethod
    def _handle_exception(exception: Exception, function_name: str, results: Dict[str, Any]) -> None:
        """
        Handle exceptions during function calls.

        Args:
            exception: The exception that was raised
            function_name: Name of the function that caused the exception
            results: Dictionary to update with the error information
        """
        error_message: str = f"Error calling function {function_name}: {exception}"
        print(error_message)  # DEBUG

        # Use the function name as the key for storing the error
        key: str = function_name

        # Store the error message as the result value
        results[key] = f"Error: {str(exception)}"

    @staticmethod
    def _handle_expression_evaluation(args: Dict[str, Any], result: Any, function_name: str, non_computation_functions: Tuple[str, ...], canvas: "Canvas", results: Dict[str, Any]) -> None:
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
        expression: str = args.get('expression', '')
        if not expression:
            return
        expression = expression.replace(' ', '')
        key: str = ResultProcessor._format_expression_key(expression, args)

        # DISABLED: Saving expression evaluation computations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions,
        #                                            expression, canvas)

        results[key] = result

    @staticmethod
    def _format_expression_key(expression: str, args: Dict[str, Any]) -> str:
        """Format a key for expression evaluation results."""
        if 'variables' in args:
            variables_dict: Any = args.get('variables', {})
            if not isinstance(variables_dict, dict):
                variables_dict = {}
            variables: str = ', '.join(f"{k}:{v}" for k, v in variables_dict.items())
            return f"{expression} for {variables}"
        else:
            return expression
