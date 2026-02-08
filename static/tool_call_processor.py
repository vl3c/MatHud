"""
MatHud Tool Call Processing Utilities

Converts OpenAI tool call objects into JSON-serializable format for client processing.
Handles argument parsing and error recovery for malformed tool calls.

Dependencies:
    - json: JSON parsing and serialization
    - logging: Error logging for malformed arguments
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Protocol, Sequence, TypedDict


class ToolCallFunction(Protocol):
    """Protocol for OpenAI tool call function attribute."""

    name: str
    arguments: str


class ToolCallObject(Protocol):
    """Protocol for OpenAI tool call object structure."""

    function: ToolCallFunction


class ProcessedToolCall(TypedDict):
    """Processed tool call format sent to client."""

    function_name: str
    arguments: Dict[str, Any]


class ToolCallProcessor:
    """Static class for processing tool calls into JSON-serializable format.

    Provides utilities to convert OpenAI tool call objects into simplified
    dictionaries that can be sent to the client-side function processing system.
    """

    @staticmethod
    def jsonify_tool_call(tool_call: ToolCallObject) -> ProcessedToolCall:
        """Convert a single tool call into a simplified JSON-serializable dictionary representation.

        Args:
            tool_call: OpenAI tool call object with function name and arguments

        Returns:
            ProcessedToolCall: Simplified tool call with 'function_name' and 'arguments' keys
        """
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                arguments = {}
                logging.error(f"Arguments for function {function_name} are not a JSON object.")
        except json.JSONDecodeError:
            arguments = {}
            logging.error(f"Failed to decode arguments for function {function_name}.")

        processed_call: ProcessedToolCall = {"function_name": function_name, "arguments": arguments}
        return processed_call

    @staticmethod
    def jsonify_tool_calls(tool_calls: Sequence[ToolCallObject]) -> List[ProcessedToolCall]:
        """Convert a list of tool calls into a simplified JSON-serializable format.

        Args:
            tool_calls: Sequence of OpenAI tool call objects

        Returns:
            List[ProcessedToolCall]: List of simplified tool call dictionaries
        """
        simplified_calls = [ToolCallProcessor.jsonify_tool_call(tool_call) for tool_call in tool_calls]
        return simplified_calls
