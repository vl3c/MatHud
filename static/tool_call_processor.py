import json
import logging


class ToolCallProcessor:
    """Static class for processing tool calls into JSON-serializable format."""

    @staticmethod
    def jsonify_tool_call(tool_call):
        """Convert a single tool call into a simplified JSON-serializable dictionary representation."""
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
        """Convert a list of tool calls into a simplified JSON-serializable format."""
        simplified_calls = [ToolCallProcessor.jsonify_tool_call(tool_call) for tool_call in tool_calls]
        return simplified_calls 