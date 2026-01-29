"""
MatHud Tool Search Service

Provides semantic tool discovery using AI-powered matching.
Given a user's description of what they want to accomplish, searches through
available tool definitions and returns the most relevant matches.

Dependencies:
    - static.ai_model: AI model configuration
    - static.functions_definitions: Tool definitions to search through
    - openai: OpenAI API client for semantic matching
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS, FunctionDefinition

_logger = logging.getLogger("mathud")

# Tools to exclude from search results (meta-tools that shouldn't be recommended)
EXCLUDED_FROM_SEARCH = frozenset({"search_tools"})


class ToolSearchService:
    """Service for semantic tool discovery using AI-powered matching.
    
    Uses the app's AI model to find the most relevant tools for a given query
    by analyzing tool names and descriptions.
    """

    # System prompt for tool selection
    TOOL_SELECTOR_PROMPT = """You are a tool selector. Given a user's description of what they want to accomplish, select the most relevant tools from the list below. Return ONLY a JSON array of tool names, ordered by relevance (most relevant first).

Available tools:
{tool_descriptions}

User query: "{query}"

Return a JSON array of up to {max_results} tool names. Example: ["create_circle", "create_point"]"""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        default_model: Optional[AIModel] = None,
    ) -> None:
        """Initialize the tool search service.

        Args:
            client: Optional OpenAI-compatible client. If not provided, creates a new one.
            default_model: Optional default model to use for search. If not provided,
                uses gpt-4.1-mini for OpenAI or the client's configured model for local LLMs.
        """
        if client is not None:
            self.client = client
        else:
            api_key = self._initialize_api_key()
            self.client = OpenAI(api_key=api_key)

        self.default_model = default_model

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        dotenv_path = ".env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")

        return api_key

    @staticmethod
    def get_all_tools() -> List[FunctionDefinition]:
        """Get all available tool definitions.
        
        Returns:
            List of all function definitions.
        """
        return list(FUNCTIONS)

    @staticmethod
    def build_tool_descriptions(exclude_meta_tools: bool = True) -> str:
        """Build a compact string of all tool names and descriptions.
        
        Args:
            exclude_meta_tools: If True, excludes meta-tools like search_tools.
        
        Returns:
            Formatted string with tool names and descriptions.
        """
        lines: List[str] = []
        for tool in FUNCTIONS:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            # Skip meta-tools to avoid recursive suggestions
            if exclude_meta_tools and name in EXCLUDED_FROM_SEARCH:
                continue
            description = func.get("description", "No description")
            # Truncate long descriptions for the prompt
            if len(description) > 150:
                description = description[:147] + "..."
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    @staticmethod
    def get_tool_by_name(name: str) -> Optional[FunctionDefinition]:
        """Get a tool definition by its name.
        
        Args:
            name: The tool name to look up.
            
        Returns:
            The tool definition if found, None otherwise.
        """
        for tool in FUNCTIONS:
            func = tool.get("function", {})
            if func.get("name") == name:
                return tool
        return None

    def search_tools(
        self,
        query: str,
        model: Optional[AIModel] = None,
        max_results: int = 10,
    ) -> List[FunctionDefinition]:
        """Search for tools matching a query description.
        
        Uses AI to semantically match the query against tool descriptions
        and return the most relevant tool definitions.
        
        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching. Defaults to gpt-4.1-mini.
            max_results: Maximum number of tools to return (1-20).
            
        Returns:
            List of matching tool definitions, ordered by relevance.
        """
        if not query or not query.strip():
            _logger.warning("Tool search called with empty query")
            return []

        # Clamp max_results to valid range
        max_results = max(1, min(20, max_results))

        # Use provided model, instance default, or fallback to gpt-4.1-mini
        if model is None:
            model = self.default_model or AIModel.from_identifier("gpt-4.1-mini")

        # Build the prompt
        tool_descriptions = self.build_tool_descriptions()
        prompt = self.TOOL_SELECTOR_PROMPT.format(
            tool_descriptions=tool_descriptions,
            query=query,
            max_results=max_results,
        )

        try:
            # Call the AI model
            response = self.client.chat.completions.create(
                model=model.id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistent results
                max_tokens=500,  # Tool names are short
            )

            # Extract the response content
            content = response.choices[0].message.content
            if not content:
                _logger.warning("Tool search returned empty response")
                return []

            # Parse the JSON array of tool names
            tool_names = self._parse_tool_names(content)

            # Look up full definitions for each tool name (excluding meta-tools)
            matched_tools: List[FunctionDefinition] = []
            for name in tool_names:
                if name in EXCLUDED_FROM_SEARCH:
                    continue
                tool = self.get_tool_by_name(name)
                if tool is not None:
                    matched_tools.append(tool)
                    if len(matched_tools) >= max_results:
                        break

            _logger.info(f"Tool search for '{query}' found {len(matched_tools)} tools")
            return matched_tools

        except Exception as e:
            _logger.error(f"Tool search failed: {e}")
            return []

    @staticmethod
    def _extract_list_from_parsed(parsed: Any) -> List[str]:
        """Extract a list of strings from a parsed JSON value.
        
        Handles both direct arrays and objects with 'tools' key.
        
        Args:
            parsed: The parsed JSON value.
            
        Returns:
            List of tool name strings.
        """
        # Direct array
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str)]
        # Object with 'tools' key (e.g., {"tools": ["create_circle", ...]})
        if isinstance(parsed, dict):
            tools = parsed.get("tools")
            if isinstance(tools, list):
                return [str(item) for item in tools if isinstance(item, str)]
        return []

    @staticmethod
    def _parse_tool_names(content: str) -> List[str]:
        """Parse tool names from AI response.
        
        Handles various response formats including:
        - JSON arrays: ["tool1", "tool2"]
        - JSON objects: {"tools": ["tool1", "tool2"]}
        - Markdown code blocks with JSON
        
        Args:
            content: The AI response content.
            
        Returns:
            List of tool names extracted from the response.
        """
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    content = part
                    break

        # Try to parse as JSON (array or object)
        try:
            parsed = json.loads(content)
            result = ToolSearchService._extract_list_from_parsed(parsed)
            if result:
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON array in the content
        start_idx = content.find("[")
        end_idx = content.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = content[start_idx : end_idx + 1]
                parsed = json.loads(json_str)
                result = ToolSearchService._extract_list_from_parsed(parsed)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: try to find JSON object in the content
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = content[start_idx : end_idx + 1]
                parsed = json.loads(json_str)
                result = ToolSearchService._extract_list_from_parsed(parsed)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        _logger.warning(f"Could not parse tool names from: {content[:100]}")
        return []

    def search_tools_formatted(
        self,
        query: str,
        model: Optional[AIModel] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search for tools and return formatted result for AI consumption.
        
        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching.
            max_results: Maximum number of tools to return.
            
        Returns:
            Dict with 'tools' list and 'count' for the AI to use.
        """
        tools = self.search_tools(query, model, max_results)
        return {
            "tools": tools,
            "count": len(tools),
            "query": query,
        }
