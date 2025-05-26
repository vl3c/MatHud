"""
MatHud Tool Call Processing Utilities

Converts OpenAI tool call objects into JSON-serializable format for client processing.
Handles argument parsing and error recovery for malformed tool calls.

Dependencies:
    - json: JSON parsing and serialization
    - logging: Error logging for malformed arguments
"""

import json
import logging


class ToolCallProcessor:
    """Static class for processing tool calls into JSON-serializable format.
    
    Provides utilities to convert OpenAI tool call objects into simplified
    dictionaries that can be sent to the client-side function processing system.
    """

    @staticmethod
    def jsonify_tool_call(tool_call):
        """Convert a single tool call into a simplified JSON-serializable dictionary representation.
        
        Args:
            tool_call: OpenAI tool call object with function name and arguments
            
        Returns:
            dict: Simplified tool call with 'function_name' and 'arguments' keys
        """
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}
            logging.error(f"Failed to decode arguments for function {function_name}.")
        
        processed_call = {"function_name": function_name, "arguments": arguments}
        return processed_call

    @staticmethod
    def jsonify_tool_calls(tool_calls):
        """Convert a list of tool calls into a simplified JSON-serializable format.
        
        Args:
            tool_calls: List of OpenAI tool call objects
            
        Returns:
            list: List of simplified tool call dictionaries
        """
        simplified_calls = [ToolCallProcessor.jsonify_tool_call(tool_call) for tool_call in tool_calls]
        return simplified_calls 