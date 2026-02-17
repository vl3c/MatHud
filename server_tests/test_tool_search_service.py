"""
Tests for the Tool Search Service.

Tests semantic tool discovery, prompt construction, result parsing,
and edge case handling.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Setup path for imports - add static directory to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from static.tool_search_service import ToolSearchService
from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS


class TestToolSearchServiceBasics:
    """Test basic ToolSearchService functionality."""

    def test_get_all_tools_returns_list(self) -> None:
        """get_all_tools should return a non-empty list."""
        tools = ToolSearchService.get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_all_tools_matches_functions(self) -> None:
        """get_all_tools should return all FUNCTIONS."""
        tools = ToolSearchService.get_all_tools()
        assert len(tools) == len(FUNCTIONS)

    def test_build_tool_descriptions_format(self) -> None:
        """build_tool_descriptions should return properly formatted string."""
        descriptions = ToolSearchService.build_tool_descriptions()
        assert isinstance(descriptions, str)
        # Should have multiple lines (one per tool)
        lines = descriptions.strip().split("\n")
        assert len(lines) > 0
        # Each line should start with "- "
        for line in lines:
            assert line.startswith("- "), f"Line should start with '- ': {line}"

    def test_build_tool_descriptions_contains_tools(self) -> None:
        """build_tool_descriptions should contain known tool names."""
        descriptions = ToolSearchService.build_tool_descriptions()
        # Check for some known tools
        assert "create_point" in descriptions
        assert "create_circle" in descriptions
        assert "undo" in descriptions
        assert "redo" in descriptions

    def test_get_tool_by_name_found(self) -> None:
        """get_tool_by_name should return the tool when found."""
        tool = ToolSearchService.get_tool_by_name("create_point")
        assert tool is not None
        assert tool["function"]["name"] == "create_point"

    def test_get_tool_by_name_not_found(self) -> None:
        """get_tool_by_name should return None for unknown tools."""
        tool = ToolSearchService.get_tool_by_name("nonexistent_tool_xyz")
        assert tool is None

    def test_get_tool_by_name_all_tools(self) -> None:
        """get_tool_by_name should find all tools by their names."""
        for func in FUNCTIONS:
            name = func.get("function", {}).get("name")
            if name:
                tool = ToolSearchService.get_tool_by_name(name)
                assert tool is not None, f"Tool '{name}' should be found"


class TestToolNameParsing:
    """Test the _parse_tool_names static method."""

    def test_parse_simple_json_array(self) -> None:
        """Should parse a simple JSON array."""
        content = '["create_circle", "create_point"]'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_json_with_whitespace(self) -> None:
        """Should handle JSON with extra whitespace."""
        content = '  ["create_circle",  "create_point"]  '
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_json_in_code_block(self) -> None:
        """Should extract JSON from markdown code blocks."""
        content = '```json\n["create_circle", "create_point"]\n```'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_json_in_plain_code_block(self) -> None:
        """Should extract JSON from plain code blocks."""
        content = '```\n["create_circle", "create_point"]\n```'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_json_with_surrounding_text(self) -> None:
        """Should extract JSON array from surrounding text."""
        content = 'Here are the tools: ["create_circle", "create_point"] that match.'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_empty_array(self) -> None:
        """Should handle empty arrays."""
        content = "[]"
        result = ToolSearchService._parse_tool_names(content)
        assert result == []

    def test_parse_invalid_json(self) -> None:
        """Should return empty list for invalid JSON."""
        content = "not valid json at all"
        result = ToolSearchService._parse_tool_names(content)
        assert result == []

    def test_parse_filters_non_strings(self) -> None:
        """Should filter out non-string items."""
        content = '["create_circle", 123, "create_point", null]'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_single_item(self) -> None:
        """Should handle single-item arrays."""
        content = '["create_circle"]'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle"]


class TestSearchToolsWithMock:
    """Test search_tools with mocked OpenAI client."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def service(self, mock_client: MagicMock) -> ToolSearchService:
        """Create a ToolSearchService with mocked client."""
        return ToolSearchService(client=mock_client)

    def _setup_mock_response(self, mock_client: MagicMock, content: str) -> None:
        """Configure mock client to return specific content."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

    def test_search_returns_tools(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should return matching tool definitions."""
        self._setup_mock_response(mock_client, '["create_circle", "create_point"]')

        result = service.search_tools("draw a circle")

        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "create_circle" in names
        assert "create_point" in names

    def test_search_respects_max_results(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should limit results to max_results."""
        self._setup_mock_response(
            mock_client,
            '["create_circle", "create_point", "create_segment", "create_vector"]',
        )

        result = service.search_tools("draw shapes", max_results=2)

        assert len(result) == 2

    def test_search_empty_query_returns_empty(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should return empty list for empty query."""
        result = service.search_tools("")
        assert result == []
        # Should not call the API
        mock_client.chat.completions.create.assert_not_called()

    def test_search_whitespace_query_returns_empty(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should return empty list for whitespace-only query."""
        result = service.search_tools("   ")
        assert result == []
        mock_client.chat.completions.create.assert_not_called()

    def test_search_clamps_max_results_low(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should clamp max_results to minimum of 1."""
        self._setup_mock_response(mock_client, '["create_circle"]')

        result = service.search_tools("draw", max_results=0)

        # Should still work with at least 1 result
        assert len(result) <= 1

    def test_search_clamps_max_results_high(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should clamp max_results to maximum of 20."""
        self._setup_mock_response(mock_client, '["create_circle", "create_point"]')

        # Passing 100 should be clamped to 20
        service.search_tools("draw", max_results=100)

        # Verify the prompt contains max 20
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        prompt = messages[0]["content"] if messages else ""
        assert "up to 20 tool names" in prompt

    def test_search_filters_unknown_tools(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should filter out unknown tool names from response."""
        self._setup_mock_response(
            mock_client,
            '["create_circle", "nonexistent_tool", "create_point"]',
        )

        result = service.search_tools("draw")

        # Only valid tools should be returned
        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "nonexistent_tool" not in names

    def test_search_handles_api_error(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should use fallback ranking on API error."""
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = service.search_tools("draw a circle")

        assert len(result) > 0
        names = [t["function"]["name"] for t in result]
        assert "create_circle" in names

    def test_search_handles_empty_response(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should handle empty API response."""
        self._setup_mock_response(mock_client, "")

        result = service.search_tools("draw")

        assert result == []

    def test_search_uses_correct_model(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should use the specified model."""
        self._setup_mock_response(mock_client, '["create_circle"]')
        model = AIModel.from_identifier("gpt-4.1")

        service.search_tools("draw", model=model)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("model") == "gpt-4.1"

    def test_search_uses_default_model_when_none(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools should use gpt-4.1-mini when no model specified."""
        self._setup_mock_response(mock_client, '["create_circle"]')

        service.search_tools("draw")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("model") == "gpt-4.1-mini"


class TestSearchToolsFormatted:
    """Test the search_tools_formatted method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def service(self, mock_client: MagicMock) -> ToolSearchService:
        """Create a ToolSearchService with mocked client."""
        return ToolSearchService(client=mock_client)

    def _setup_mock_response(self, mock_client: MagicMock, content: str) -> None:
        """Configure mock client to return specific content."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

    def test_formatted_returns_dict(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools_formatted should return a dict."""
        self._setup_mock_response(mock_client, '["create_circle"]')

        result = service.search_tools_formatted("draw")

        assert isinstance(result, dict)

    def test_formatted_contains_required_keys(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """search_tools_formatted result should have tools, count, and query."""
        self._setup_mock_response(mock_client, '["create_circle"]')

        result = service.search_tools_formatted("draw a circle")

        assert "tools" in result
        assert "count" in result
        assert "query" in result

    def test_formatted_count_matches_tools(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """count should match the number of tools returned."""
        self._setup_mock_response(mock_client, '["create_circle", "create_point"]')

        result = service.search_tools_formatted("draw shapes")

        assert result["count"] == len(result["tools"])
        assert result["count"] == 2

    def test_formatted_preserves_query(self, service: ToolSearchService, mock_client: MagicMock) -> None:
        """query field should contain the original query."""
        self._setup_mock_response(mock_client, '["create_circle"]')
        query = "draw a beautiful circle"

        result = service.search_tools_formatted(query)

        assert result["query"] == query


class TestToolModeConfiguration:
    """Test tool mode configuration in OpenAIAPIBase."""

    def test_search_mode_tools_exist(self) -> None:
        """SEARCH_MODE_TOOLS should be defined and non-empty."""
        from static.openai_api_base import SEARCH_MODE_TOOLS

        assert isinstance(SEARCH_MODE_TOOLS, list)
        assert len(SEARCH_MODE_TOOLS) > 0

    def test_search_mode_contains_search_tools(self) -> None:
        """SEARCH_MODE_TOOLS should contain search_tools."""
        from static.openai_api_base import SEARCH_MODE_TOOLS

        names = [t.get("function", {}).get("name") for t in SEARCH_MODE_TOOLS]
        assert "search_tools" in names

    def test_search_mode_contains_essential_tools(self) -> None:
        """SEARCH_MODE_TOOLS should contain essential tools."""
        from static.openai_api_base import SEARCH_MODE_TOOLS, ESSENTIAL_TOOLS

        names = set(t.get("function", {}).get("name") for t in SEARCH_MODE_TOOLS)
        for essential in ESSENTIAL_TOOLS:
            assert essential in names, f"Essential tool '{essential}' missing"

    def test_search_mode_is_minimal(self) -> None:
        """SEARCH_MODE_TOOLS should be smaller than full FUNCTIONS."""
        from static.openai_api_base import SEARCH_MODE_TOOLS
        from static.functions_definitions import FUNCTIONS

        assert len(SEARCH_MODE_TOOLS) < len(FUNCTIONS)

    def test_essential_tools_defined(self) -> None:
        """ESSENTIAL_TOOLS should contain expected tools."""
        from static.openai_api_base import ESSENTIAL_TOOLS

        assert "undo" in ESSENTIAL_TOOLS
        assert "redo" in ESSENTIAL_TOOLS
        assert "get_current_canvas_state" in ESSENTIAL_TOOLS


class TestOpenAIAPIBaseToolModes:
    """Test OpenAIAPIBase tool mode methods."""

    def test_default_tool_mode_is_full(self) -> None:
        """Default tool mode should be 'full'."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()
                assert api.get_tool_mode() == "full"

    def test_can_initialize_with_search_mode(self) -> None:
        """Should be able to initialize with search mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase, SEARCH_MODE_TOOLS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase(tool_mode="search")
                assert api.get_tool_mode() == "search"
                assert len(api.tools) == len(SEARCH_MODE_TOOLS)

    def test_set_tool_mode_to_search(self) -> None:
        """set_tool_mode should switch to search mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase, SEARCH_MODE_TOOLS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()
                api.set_tool_mode("search")
                assert api.get_tool_mode() == "search"
                assert len(api.tools) == len(SEARCH_MODE_TOOLS)

    def test_set_tool_mode_to_full(self) -> None:
        """set_tool_mode should switch back to full mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase
            from static.functions_definitions import FUNCTIONS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase(tool_mode="search")
                api.set_tool_mode("full")
                assert api.get_tool_mode() == "full"
                assert len(api.tools) == len(FUNCTIONS)

    def test_set_tool_mode_invalid_raises(self) -> None:
        """set_tool_mode should raise for invalid mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()
                with pytest.raises(ValueError):
                    api.set_tool_mode("invalid")  # type: ignore

    def test_custom_tools_override_mode(self) -> None:
        """Custom tools should override tool mode selection."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase

            custom_tools = [{"type": "function", "function": {"name": "custom"}}]
            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase(tools=custom_tools, tool_mode="search")
                # Should use custom tools, not search mode tools
                assert len(api.tools) == 1
                assert api.tools[0]["function"]["name"] == "custom"


class TestSearchToolsFunctionDefinition:
    """Test that search_tools function is properly defined."""

    def test_search_tools_in_functions(self) -> None:
        """search_tools should be in FUNCTIONS list."""
        from static.functions_definitions import FUNCTIONS

        names = [f.get("function", {}).get("name") for f in FUNCTIONS]
        assert "search_tools" in names

    def test_search_tools_has_required_schema(self) -> None:
        """search_tools should have proper function schema."""
        tool = ToolSearchService.get_tool_by_name("search_tools")
        assert tool is not None

        func = tool["function"]
        assert func["name"] == "search_tools"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]
        assert "query" in params["required"]

    def test_numeric_integrate_has_required_schema(self) -> None:
        """numeric_integrate should expose method and step controls."""
        tool = ToolSearchService.get_tool_by_name("numeric_integrate")
        assert tool is not None

        func = tool["function"]
        assert func["name"] == "numeric_integrate"
        params = func["parameters"]
        assert params["type"] == "object"
        assert "expression" in params["properties"]
        assert "variable" in params["properties"]
        assert "lower_bound" in params["properties"]
        assert "upper_bound" in params["properties"]
        assert "method" in params["properties"]
        assert "steps" in params["properties"]

    def test_evaluate_expression_description_mentions_series_helpers(self) -> None:
        tool = ToolSearchService.get_tool_by_name("evaluate_expression")
        assert tool is not None
        description = tool["function"].get("description", "")
        assert "summation(" in description
        assert "product(" in description


class TestSearchToolsExclusion:
    """Test that search_tools is excluded from search results."""

    def test_search_tools_excluded_from_descriptions(self) -> None:
        """build_tool_descriptions should exclude search_tools by default."""
        descriptions = ToolSearchService.build_tool_descriptions()
        assert "- search_tools:" not in descriptions

    def test_search_tools_included_when_requested(self) -> None:
        """build_tool_descriptions can include search_tools if requested."""
        descriptions = ToolSearchService.build_tool_descriptions(exclude_meta_tools=False)
        assert "- search_tools:" in descriptions

    def test_search_filters_out_search_tools(self) -> None:
        """search_tools should not appear in search results."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '["search_tools", "create_circle", "create_point"]'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        service = ToolSearchService(client=mock_client)
        result = service.search_tools("find tools")

        names = [t["function"]["name"] for t in result]
        assert "search_tools" not in names
        assert "create_circle" in names
        assert "create_point" in names


class TestObjectFormatParsing:
    """Test parsing of object format responses like {"tools": [...]}."""

    def test_parse_object_with_tools_key(self) -> None:
        """Should parse object with 'tools' key."""
        content = '{"tools": ["create_circle", "create_point"]}'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "create_point"]

    def test_parse_object_in_code_block(self) -> None:
        """Should extract object from code block."""
        content = '```json\n{"tools": ["create_circle"]}\n```'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle"]

    def test_parse_object_with_surrounding_text(self) -> None:
        """Should extract object from surrounding text."""
        content = 'Here are the tools: {"tools": ["create_circle", "undo"]} end.'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "undo"]

    def test_parse_object_without_tools_key(self) -> None:
        """Object without 'tools' key falls back to array extraction."""
        # The parser will find the embedded array as a fallback
        content = '{"result": ["create_circle"]}'
        result = ToolSearchService._parse_tool_names(content)
        # Falls back to extracting the array inside
        assert result == ["create_circle"]

    def test_parse_object_with_no_arrays(self) -> None:
        """Should return empty for object with no extractable arrays."""
        content = '{"message": "no tools found"}'
        result = ToolSearchService._parse_tool_names(content)
        assert result == []

    def test_parse_prefers_direct_array(self) -> None:
        """Should still work with direct array format."""
        content = '["create_circle", "undo"]'
        result = ToolSearchService._parse_tool_names(content)
        assert result == ["create_circle", "undo"]


class TestDynamicToolInjection:
    """Test inject_tools and reset_tools methods."""

    def test_inject_tools_replaces_tool_set(self) -> None:
        """inject_tools should replace the current tool set."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()
                original_count = len(api.tools)

                # Inject just 2 tools
                custom_tools = [
                    {"type": "function", "function": {"name": "create_circle"}},
                    {"type": "function", "function": {"name": "create_point"}},
                ]
                api.inject_tools(custom_tools, include_essentials=False)

                assert len(api.tools) == 2
                assert len(api.tools) < original_count

    def test_inject_tools_includes_essentials_by_default(self) -> None:
        """inject_tools should include essential tools by default."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase, ESSENTIAL_TOOLS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()

                custom_tools = [
                    {"type": "function", "function": {"name": "create_circle"}},
                ]
                api.inject_tools(custom_tools, include_essentials=True)

                names = {t.get("function", {}).get("name") for t in api.tools}
                assert "create_circle" in names
                for essential in ESSENTIAL_TOOLS:
                    assert essential in names

    def test_inject_tools_no_duplicate_essentials(self) -> None:
        """inject_tools should not duplicate essentials if already present."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase()

                # Include undo in the injected tools
                custom_tools = [
                    {"type": "function", "function": {"name": "create_circle"}},
                    {"type": "function", "function": {"name": "undo"}},
                ]
                api.inject_tools(custom_tools, include_essentials=True)

                names = [t.get("function", {}).get("name") for t in api.tools]
                # undo should appear only once
                assert names.count("undo") == 1

    def test_reset_tools_restores_full_mode(self) -> None:
        """reset_tools should restore tools based on current mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase
            from static.functions_definitions import FUNCTIONS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase(tool_mode="full")

                # Inject minimal tools
                api.inject_tools([{"type": "function", "function": {"name": "test"}}], include_essentials=False)
                assert len(api.tools) == 1

                # Reset
                api.reset_tools()
                assert len(api.tools) == len(FUNCTIONS)

    def test_reset_tools_restores_search_mode(self) -> None:
        """reset_tools should restore search mode tools if in search mode."""
        with patch("static.openai_api_base.OpenAI"):
            from static.openai_api_base import OpenAIAPIBase, SEARCH_MODE_TOOLS

            with patch.object(OpenAIAPIBase, "_initialize_api_key", return_value="test-key"):
                api = OpenAIAPIBase(tool_mode="search")

                # Inject different tools
                api.inject_tools([{"type": "function", "function": {"name": "test"}}], include_essentials=False)

                # Reset
                api.reset_tools()
                assert len(api.tools) == len(SEARCH_MODE_TOOLS)


class TestExtractListFromParsed:
    """Test the _extract_list_from_parsed helper method."""

    def test_extract_from_list(self) -> None:
        """Should extract strings from a list."""
        result = ToolSearchService._extract_list_from_parsed(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_extract_from_dict_with_tools(self) -> None:
        """Should extract from dict with 'tools' key."""
        result = ToolSearchService._extract_list_from_parsed({"tools": ["a", "b"]})
        assert result == ["a", "b"]

    def test_extract_filters_non_strings(self) -> None:
        """Should filter out non-string items."""
        result = ToolSearchService._extract_list_from_parsed(["a", 1, "b", None])
        assert result == ["a", "b"]

    def test_extract_from_empty_list(self) -> None:
        """Should return empty for empty list."""
        result = ToolSearchService._extract_list_from_parsed([])
        assert result == []

    def test_extract_from_dict_without_tools(self) -> None:
        """Should return empty for dict without 'tools' key."""
        result = ToolSearchService._extract_list_from_parsed({"other": ["a"]})
        assert result == []

    def test_extract_from_string(self) -> None:
        """Should return empty for non-list/dict types."""
        result = ToolSearchService._extract_list_from_parsed("not a list")
        assert result == []

    def test_extract_from_none(self) -> None:
        """Should return empty for None."""
        result = ToolSearchService._extract_list_from_parsed(None)
        assert result == []
