"""
Tests for search-first tool exposure: the MATHUD_TOOL_EXPOSURE setting and the
system-prompt sentence that explains search_tools to the model.
"""

from __future__ import annotations

import os
import unittest
from typing import Optional
from unittest.mock import patch

from static.ai_model import AIModel
from static.app_manager import AppManager
from static.functions_definitions import FUNCTIONS
from static.openai_api_base import SEARCH_MODE_TOOLS, OpenAIAPIBase, get_configured_tool_mode
from static.providers.anthropic_api import AnthropicAPI
from static.providers.local.local_agent_api import LocalAgentAPI

SEARCH_MSG = OpenAIAPIBase.SEARCH_MODE_MSG


class TestConfiguredToolMode(unittest.TestCase):
    def test_defaults_to_search(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATHUD_TOOL_EXPOSURE", None)
            self.assertEqual(get_configured_tool_mode(), "search")

    def test_full_is_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"MATHUD_TOOL_EXPOSURE": " Full "}):
            self.assertEqual(get_configured_tool_mode(), "full")

    def test_unknown_value_falls_back_to_search(self) -> None:
        with patch.dict(os.environ, {"MATHUD_TOOL_EXPOSURE": "everything"}):
            self.assertEqual(get_configured_tool_mode(), "search")


class TestSearchModeSystemPrompt(unittest.TestCase):
    def _api(self, **kwargs: object) -> OpenAIAPIBase:
        with patch("static.openai_api_base.OpenAI"):
            return OpenAIAPIBase(**kwargs)  # type: ignore[arg-type]

    def test_search_mode_explains_search_tools(self) -> None:
        api = self._api(tool_mode="search")
        self.assertIn(SEARCH_MSG, api.messages[0]["content"])
        self.assertIn(OpenAIAPIBase.DEV_MSG, api.messages[0]["content"])

    def test_full_mode_omits_search_sentence(self) -> None:
        api = self._api(tool_mode="full")
        self.assertEqual(api.messages[0]["content"], OpenAIAPIBase.DEV_MSG)

    def test_custom_tools_omit_search_sentence(self) -> None:
        api = self._api(tool_mode="search", tools=[])
        self.assertEqual(api.messages[0]["content"], OpenAIAPIBase.DEV_MSG)

    def test_set_tool_mode_refreshes_prompt_and_reset_keeps_it(self) -> None:
        api = self._api(tool_mode="full")
        api.set_tool_mode("search")
        self.assertIn(SEARCH_MSG, api.messages[0]["content"])
        api.reset_conversation()
        self.assertIn(SEARCH_MSG, api.messages[0]["content"])
        api.set_tool_mode("full")
        self.assertEqual(api.messages[0]["content"], OpenAIAPIBase.DEV_MSG)

    def test_anthropic_and_local_system_prompts_follow_mode(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            anthropic_api = AnthropicAPI(model=AIModel.from_identifier("claude-haiku-4-5"), tool_mode="search")
        self.assertIn(SEARCH_MSG, anthropic_api._build_system_prompt())
        local_api = LocalAgentAPI(model=AIModel.from_identifier("local-model"))
        self.assertIn(SEARCH_MSG, local_api.messages[0]["content"])

    def test_search_tools_description_says_to_call_it_first(self) -> None:
        search_tool = next(t for t in FUNCTIONS if t["function"]["name"] == "search_tools")
        description = search_tool["function"]["description"]
        self.assertIn("before any other tool", description)
        self.assertNotIn("unsure", description)


class TestToolExposureWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.original_require_auth: Optional[str] = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"

    def tearDown(self) -> None:
        if self.original_require_auth is not None:
            os.environ["REQUIRE_AUTH"] = self.original_require_auth
        else:
            os.environ.pop("REQUIRE_AUTH", None)

    def test_full_exposure_gives_every_provider_all_tools(self) -> None:
        from static.routes import get_provider_for_model

        with patch.dict(os.environ, {"MATHUD_TOOL_EXPOSURE": "full", "ANTHROPIC_API_KEY": "test-key"}):
            app = AppManager.create_app()
            anthropic_provider = get_provider_for_model(app, "claude-haiku-4-5")
            local_api = LocalAgentAPI(model=AIModel.from_identifier("local-model"))

        for api in (app.ai_api, app.responses_api, anthropic_provider, local_api):
            self.assertEqual(api.get_tool_mode(), "full")
            self.assertEqual(len(api.tools), len(FUNCTIONS))
        self.assertEqual(app.ai_api.messages[0]["content"], OpenAIAPIBase.DEV_MSG)

    def test_default_exposure_is_search_first(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATHUD_TOOL_EXPOSURE", None)
            app = AppManager.create_app()
        self.assertEqual(app.ai_api.get_tool_mode(), "search")
        self.assertEqual(len(app.ai_api.tools), len(SEARCH_MODE_TOOLS))
        self.assertIn(SEARCH_MSG, app.ai_api.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
