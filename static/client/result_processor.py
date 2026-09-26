"""
MatHud Function Call Result Processing

Handles execution of AI function calls and aggregation of their results.
Manages state archiving, computation tracking, and error handling for function execution.

Key Features:
    - Function call validation and execution
    - Result formatting and type consistency
    - One undo step per batch of calls
    - Computation history integration
    - Error handling and exception management
    - Expression evaluation result processing

Processing Flow:
    1. Input validation for function calls and available functions
    2. An undo batch around calls that can change the canvas, so the batch is one undo step
    3. Individual function call execution with error handling
    4. Result formatting and key generation
    5. Computation history integration (for mathematical operations)
    6. Result aggregation into structured dictionary

Dependencies:
    - constants: Success message definitions for result formatting
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from browser import window

from constants import (
    nothing_deleted_message,
    nothing_to_redo_message,
    nothing_to_undo_message,
    successful_call_message,
)
from no_change_result import NoChangeResult

# Largest JSON-serialized return value of a canvas-mutating tool passed back to the model;
# larger values are replaced by the success message to keep token usage bounded.
MAX_PASSTHROUGH_RESULT_CHARS = 2000

if TYPE_CHECKING:
    from canvas import Canvas

    TracedCall = Dict[str, Any]
    """Per-call trace record: seq, function_name, arguments, result_key, result, is_error, duration_ms."""


class ResultProcessor:
    """Handles the processing of function calls and their results.

    Executes AI function calls, manages state for undoable operations, and aggregates
    results with proper formatting and error handling. Integrates with canvas
    computation history for mathematical operations.
    """

    @staticmethod
    def get_results(
        calls: List[Dict[str, Any]],
        available_functions: Dict[str, Any],
        undoable_functions: Tuple[str, ...],
        canvas: "Canvas",
    ) -> Dict[str, Any]:
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
        results, _ = ResultProcessor.get_results_traced(calls, available_functions, undoable_functions, canvas)
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
        non_computation_functions, unformattable_functions = ResultProcessor._prepare_helper_variables(
            undoable_functions
        )

        # One tool batch is one undo step: inside the batch, archives only mark it as changed
        batch_is_undoable: bool = ResultProcessor._contains_undoable_function(calls, undoable_functions)
        if batch_is_undoable:
            canvas.begin_undo_batch()
        try:
            for seq, call in enumerate(calls):
                changed_before: bool = batch_is_undoable and canvas.is_undo_batch_changed()
                traced_call, changed_nothing = ResultProcessor._run_traced_call(
                    seq, call, available_functions, non_computation_functions, unformattable_functions, canvas, results
                )
                traced_calls.append(traced_call)
                if batch_is_undoable:
                    ResultProcessor._record_batch_change(
                        traced_call, changed_nothing, changed_before, undoable_functions, canvas
                    )
        finally:
            if batch_is_undoable:
                canvas.end_undo_batch()

        return results, traced_calls

    @staticmethod
    def _contains_undoable_function(calls: List[Dict[str, Any]], undoable_functions: Tuple[str, ...]) -> bool:
        """Return True when any call in the batch can change the canvas."""
        return any(call.get("function_name", "") in undoable_functions for call in calls)

    @staticmethod
    def _run_traced_call(
        seq: int,
        call: Dict[str, Any],
        available_functions: Dict[str, Any],
        non_computation_functions: Tuple[str, ...],
        unformattable_functions: Tuple[str, ...],
        canvas: "Canvas",
        results: Dict[str, Any],
    ) -> Tuple["TracedCall", bool]:
        """Execute one call, add its result to ``results`` and return its trace record.

        The flag is True when the call reported that it changed nothing (see ``_changed_nothing``).
        """
        function_name = call.get("function_name", "")
        args = call.get("arguments", {})
        # Sanitize arguments for trace: exclude canvas ref, guard against non-dict
        if isinstance(args, dict):
            sanitized_args = {k: v for k, v in args.items() if k != "canvas"}
        else:
            sanitized_args = {"_raw": args}

        t0 = window.performance.now()
        # Collect this call's result separately so it can be reported per call
        call_results: Dict[str, Any] = {}
        raw_result: Any = None
        try:
            raw_result = ResultProcessor._process_function_call(
                call,
                available_functions,
                non_computation_functions,
                unformattable_functions,
                canvas,
                call_results,
            )
        except Exception as e:
            ResultProcessor._handle_exception(e, function_name, call_results, args)
        results.update(call_results)
        result_key, result_value = next(iter(call_results.items()), (function_name, None))
        is_error = ResultProcessor.is_error_result(result_value)

        duration_ms = window.performance.now() - t0
        traced_call: TracedCall = {
            "seq": seq,
            "function_name": function_name,
            "arguments": sanitized_args,
            "result_key": result_key,
            "result": result_value,
            "is_error": is_error,
            "duration_ms": round(duration_ms, 2),
        }
        return traced_call, ResultProcessor._changed_nothing(function_name, raw_result)

    @staticmethod
    def _changed_nothing(function_name: str, raw_result: Any) -> bool:
        """True for a NoChangeResult, or a delete, undo or redo that returned False."""
        return (
            isinstance(raw_result, NoChangeResult)
            or ResultProcessor._no_op_message(function_name, raw_result) is not None
        )

    @staticmethod
    def is_error_result(value: Any) -> bool:
        """Return True for a failed call's result.

        Failures come back as ``"Error..."`` strings, as dicts with a non-empty ``"error"``
        field (e.g. ``analyze_graph`` on a missing graph), or as ``{"type": "error", ...}``
        payloads. A dict whose ``"error"`` field is empty (``search_tools``) is a success.
        """
        if isinstance(value, str):
            return value.startswith("Error") or ResultProcessor._is_json_error_string(value)
        if isinstance(value, dict):
            return ResultProcessor._is_error_dict(value)
        return False

    @staticmethod
    def _is_error_dict(value: Dict[str, Any]) -> bool:
        return bool(value.get("error")) or value.get("type") == "error"

    @staticmethod
    def _is_json_error_string(value: str) -> bool:
        """True for a JSON object string with a non-empty "error" field (``solve_numeric`` failures)."""
        if not value.lstrip().startswith("{"):
            return False
        try:
            parsed = json.loads(value)
        except Exception:
            return False
        return isinstance(parsed, dict) and ResultProcessor._is_error_dict(parsed)

    @staticmethod
    def _record_batch_change(
        traced_call: "TracedCall",
        changed_nothing: bool,
        changed_before: bool,
        undoable_functions: Tuple[str, ...],
        canvas: "Canvas",
    ) -> None:
        """Update the undo batch's change mark after one call.

        A successful undoable call marks the batch changed even if it did not archive itself
        (inside the batch ``canvas.archive()`` only sets the mark). After a call that failed or
        changed nothing, the batch is marked changed only if it already was, or if the canvas
        no longer matches the batch baseline: a manager that archived and then failed without
        changing anything adds no entry, while a change left behind by a failed call stays
        undoable.
        """
        if traced_call["is_error"] or changed_nothing:
            ResultProcessor._reset_mark_after_unchanged_call(changed_before, canvas)
        elif traced_call["function_name"] in undoable_functions:
            canvas.archive()

    @staticmethod
    def _reset_mark_after_unchanged_call(changed_before: bool, canvas: "Canvas") -> None:
        """Keep the mark only if the batch changed earlier or the call left a partial change behind.

        With no change recorded before the call, the batch baseline is the state before the call.
        """
        canvas.set_undo_batch_changed(changed_before or canvas.state_differs_from_undo_batch_baseline())

    @staticmethod
    def build_tool_call_results(calls: List[Dict[str, Any]], traced_calls: List["TracedCall"]) -> List[Dict[str, Any]]:
        """Pair each traced call with its tool-call id, in call order, for the server.

        Each entry is ``{"tool_call_id": id_or_None, "result": {result_key: value}}`` so the
        provider can answer every parallel tool call with its own result.
        """
        entries: List[Dict[str, Any]] = []
        for call, traced in zip(calls, traced_calls):
            tool_call_id = call.get("id") if isinstance(call, dict) else None
            entries.append({"tool_call_id": tool_call_id, "result": {traced["result_key"]: traced["result"]}})
        return entries

    @staticmethod
    def _validate_inputs(
        calls: List[Dict[str, Any]], available_functions: Dict[str, Any], undoable_functions: Tuple[str, ...]
    ) -> None:
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
        unformattable_functions: Tuple[str, ...] = undoable_functions + ("undo", "redo")
        non_computation_functions: Tuple[str, ...] = unformattable_functions + (
            "run_tests",
            "list_workspaces",
            "save_workspace",
            "load_workspace",
            "delete_workspace",
        )
        return non_computation_functions, unformattable_functions

    @staticmethod
    def _process_function_call(
        call: Dict[str, Any],
        available_functions: Dict[str, Any],
        non_computation_functions: Tuple[str, ...],
        unformattable_functions: Tuple[str, ...],
        canvas: "Canvas",
        results: Dict[str, Any],
    ) -> Any:
        """
        Process a single function call and update results.

        Args:
            call: Dictionary containing function call information
            available_functions: Dictionary mapping function names to implementations
            non_computation_functions: Tuple of function names that don't add computations
            unformattable_functions: Tuple of function names that return standard success message
            canvas: Canvas instance for adding computations
            results: Dictionary to update with the results

        Returns:
            The function's raw return value (None when the function does not exist)
        """
        function_name: str = call.get("function_name", "")

        # Check if function exists
        if not ResultProcessor._is_function_available(function_name, available_functions, results):
            return None

        # Execute the function
        args: Dict[str, Any] = call.get("arguments", {})
        result: Any = ResultProcessor._execute_function(function_name, args, available_functions)

        # Format the key for results dictionary
        key: str = ResultProcessor.generate_result_key(function_name, args)

        # Process the result based on function type
        ResultProcessor._process_result(
            function_name, args, result, key, unformattable_functions, non_computation_functions, canvas, results
        )
        return result

    @staticmethod
    def _is_function_available(
        function_name: str, available_functions: Dict[str, Any], results: Dict[str, Any]
    ) -> bool:
        """Check if the function exists and update results if not."""
        if function_name not in available_functions:
            error_msg: str = f"Error: function {function_name} not found."
            results[function_name] = error_msg
            return False
        return True

    @staticmethod
    def _execute_function(function_name: str, args: Dict[str, Any], available_functions: Dict[str, Any]) -> Any:
        """Execute the function with the provided arguments."""
        result: Any = available_functions[function_name](**args)
        return result

    @staticmethod
    def generate_result_key(function_name: str, args: Dict[str, Any]) -> str:
        """Generate a consistent key format for the results dictionary."""
        formatted_args: str = ResultProcessor._format_arguments(args)
        return f"{function_name}({formatted_args})"

    @staticmethod
    def _process_result(
        function_name: str,
        args: Dict[str, Any],
        result: Any,
        key: str,
        unformattable_functions: Tuple[str, ...],
        non_computation_functions: Tuple[str, ...],
        canvas: "Canvas",
        results: Dict[str, Any],
    ) -> None:
        """Process the result based on function type and update results dictionary."""
        if function_name in unformattable_functions:
            # Handle unformattable functions (return success message)
            ResultProcessor._handle_unformattable_function(function_name, key, result, results)
        elif function_name == "evaluate_expression" and "expression" in args:
            # Handle expression evaluation
            ResultProcessor._handle_expression_evaluation(
                args, result, function_name, non_computation_functions, canvas, results
            )
        else:
            # Handle regular functions
            ResultProcessor._handle_regular_function(
                key, result, function_name, non_computation_functions, canvas, results
            )

    @staticmethod
    def _handle_unformattable_function(function_name: str, key: str, result: Any, results: Dict[str, Any]) -> None:
        """Handle result for unformattable functions.

        A NoChangeResult reports its message, and a delete, undo or redo that returned False
        changed nothing and says so. Small string/dict return values (e.g. generated names or
        graph state) are passed through so the model can use them; anything else becomes the
        success message.
        """
        no_op_message = ResultProcessor._no_op_message(function_name, result)
        if isinstance(result, NoChangeResult):
            results[key] = result.message
        elif no_op_message is not None:
            results[key] = no_op_message
        elif ResultProcessor._is_small_passthrough_result(result):
            results[key] = result
        else:
            results[key] = successful_call_message

    @staticmethod
    def _no_op_message(function_name: str, result: Any) -> Optional[str]:
        """Message for a delete, undo or redo whose False return means nothing happened."""
        if result is not False:
            return None
        if function_name == "undo":
            return nothing_to_undo_message
        if function_name == "redo":
            return nothing_to_redo_message
        if function_name.startswith("delete_"):
            return nothing_deleted_message
        return None

    @staticmethod
    def _is_small_passthrough_result(result: Any) -> bool:
        """Return True for non-empty strings/dicts whose JSON form fits the size cap."""
        if not isinstance(result, (str, dict)) or not result:
            return False
        try:
            serialized: str = json.dumps(result)
        except Exception:
            return False
        return len(serialized) <= MAX_PASSTHROUGH_RESULT_CHARS

    @staticmethod
    def _handle_regular_function(
        key: str,
        result: Any,
        function_name: str,
        non_computation_functions: Tuple[str, ...],
        canvas: "Canvas",
        results: Dict[str, Any],
    ) -> None:
        """Handle result for regular functions."""
        # Save computation to canvas state if it's not a non-computation function
        # DISABLED: Saving basic calculations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions, key, canvas)

        results[key] = result

    @staticmethod
    def _format_arguments(args: Dict[str, Any]) -> str:
        """Format function arguments for display."""
        return ", ".join(f"{k}:{v}" for k, v in args.items() if k != "canvas")

    @staticmethod
    def _add_computation_if_needed(
        result: Any, function_name: str, non_computation_functions: Tuple[str, ...], expression: str, canvas: "Canvas"
    ) -> None:
        """Add the computation to canvas if it's not a non-computation function and succeeded."""
        if (
            not isinstance(result, str) or not result.startswith("Error:")
        ) and function_name not in non_computation_functions:
            canvas.add_computation(expression=expression, result=result)

    @staticmethod
    def _handle_exception(
        exception: Exception, function_name: str, results: Dict[str, Any], args: Any = None
    ) -> None:
        """
        Handle exceptions during function calls.

        Args:
            exception: The exception that was raised
            function_name: Name of the function that caused the exception
            results: Dictionary to update with the error information
            args: Arguments of the failed call, used to key the error like a success
        """
        # Key errors like successes so failures of the same tool don't overwrite each other
        key: str = ResultProcessor.generate_result_key(function_name, args) if isinstance(args, dict) else function_name

        # Store the error message as the result value
        results[key] = f"Error: {str(exception)}"

    @staticmethod
    def _handle_expression_evaluation(
        args: Dict[str, Any],
        result: Any,
        function_name: str,
        non_computation_functions: Tuple[str, ...],
        canvas: "Canvas",
        results: Dict[str, Any],
    ) -> None:
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
        expression: str = args.get("expression", "")
        if not expression:
            return
        expression = expression.replace(" ", "")
        key: str = ResultProcessor._format_expression_key(expression, args)

        # DISABLED: Saving expression evaluation computations to canvas state (takes up too many tokens, not useful info to store)
        # ResultProcessor._add_computation_if_needed(result, function_name, non_computation_functions,
        #                                            expression, canvas)

        results[key] = result

    @staticmethod
    def _format_expression_key(expression: str, args: Dict[str, Any]) -> str:
        """Format a key for expression evaluation results."""
        if "variables" in args:
            variables_dict: Any = args.get("variables", {})
            if not isinstance(variables_dict, dict):
                variables_dict = {}
            variables: str = ", ".join(f"{k}:{v}" for k, v in variables_dict.items())
            return f"{expression} for {variables}"
        else:
            return expression
