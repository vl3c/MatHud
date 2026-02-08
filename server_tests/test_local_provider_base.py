"""
Tests for local LLM provider base module.

Tests shared functionality for local LLM providers including
tool capability detection and model name normalization.
"""


from static.providers.local import (
    TOOL_CAPABLE_MODEL_FAMILIES,
    LocalProviderRegistry,
    normalize_model_name,
    supports_tools,
)


class TestNormalizeModelName:
    """Tests for model name normalization."""

    def test_simple_name(self) -> None:
        """Simple model name without tag."""
        assert normalize_model_name("llama3.1") == "llama3.1"

    def test_name_with_tag(self) -> None:
        """Model name with size tag."""
        assert normalize_model_name("llama3.1:8b") == "llama3.1"

    def test_name_with_instruct_tag(self) -> None:
        """Model name with instruct tag."""
        assert normalize_model_name("qwen2.5:7b-instruct") == "qwen2.5"

    def test_name_with_latest_tag(self) -> None:
        """Model name with latest tag."""
        assert normalize_model_name("mistral:latest") == "mistral"

    def test_uppercase_name(self) -> None:
        """Uppercase model name gets lowercased."""
        assert normalize_model_name("Llama3.1:8B") == "llama3.1"

    def test_whitespace_handling(self) -> None:
        """Whitespace around name is stripped."""
        assert normalize_model_name("  llama3.1:8b  ") == "llama3.1"


class TestSupportsTools:
    """Tests for tool capability detection."""

    def test_llama31_supports_tools(self) -> None:
        """Llama 3.1 models support tools."""
        assert supports_tools("llama3.1:8b") is True
        assert supports_tools("llama3.1:70b") is True

    def test_llama32_supports_tools(self) -> None:
        """Llama 3.2 models support tools."""
        assert supports_tools("llama3.2:3b") is True

    def test_llama33_supports_tools(self) -> None:
        """Llama 3.3 models support tools."""
        assert supports_tools("llama3.3:70b") is True

    def test_qwen25_supports_tools(self) -> None:
        """Qwen 2.5 models support tools."""
        assert supports_tools("qwen2.5:7b") is True
        assert supports_tools("qwen2.5-coder:7b") is True

    def test_mistral_supports_tools(self) -> None:
        """Mistral models support tools."""
        assert supports_tools("mistral:7b") is True
        assert supports_tools("mistral-nemo:12b") is True
        assert supports_tools("mixtral:8x7b") is True

    def test_command_r_supports_tools(self) -> None:
        """Command R models support tools."""
        assert supports_tools("command-r:35b") is True
        assert supports_tools("command-r-plus:104b") is True

    def test_nemotron_supports_tools(self) -> None:
        """Nemotron models support tools."""
        assert supports_tools("nemotron:70b") is True

    def test_granite3_supports_tools(self) -> None:
        """Granite 3 models support tools."""
        assert supports_tools("granite3-dense:8b") is True
        assert supports_tools("granite3-moe:3b") is True

    def test_gpt_oss_supports_tools(self) -> None:
        """GPT-OSS models support tools."""
        assert supports_tools("gpt-oss:20b") is True

    def test_llama2_does_not_support_tools(self) -> None:
        """Llama 2 models do not support tools."""
        assert supports_tools("llama2:7b") is False

    def test_phi_does_not_support_tools(self) -> None:
        """Phi models do not support tools."""
        assert supports_tools("phi:3b") is False

    def test_codellama_does_not_support_tools(self) -> None:
        """Code Llama models do not support tools."""
        assert supports_tools("codellama:7b") is False

    def test_gemma_does_not_support_tools(self) -> None:
        """Gemma models do not support tools."""
        assert supports_tools("gemma:7b") is False

    def test_unknown_model_does_not_support_tools(self) -> None:
        """Unknown models do not support tools."""
        assert supports_tools("some-unknown-model:latest") is False

    def test_case_insensitive(self) -> None:
        """Tool support check is case-insensitive."""
        assert supports_tools("Llama3.1:8B") is True
        assert supports_tools("QWEN2.5:7B") is True


class TestToolCapableModelFamilies:
    """Tests for the TOOL_CAPABLE_MODEL_FAMILIES constant."""

    def test_contains_llama_family(self) -> None:
        """Contains Llama 3.x family."""
        assert "llama3.1" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "llama3.2" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "llama3.3" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_qwen_family(self) -> None:
        """Contains Qwen 2.5 family."""
        assert "qwen2.5" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "qwen2.5-coder" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_mistral_family(self) -> None:
        """Contains Mistral family."""
        assert "mistral" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "mistral-nemo" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "mixtral" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_command_r_family(self) -> None:
        """Contains Command R family."""
        assert "command-r" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "command-r-plus" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_nvidia_family(self) -> None:
        """Contains NVIDIA Nemotron."""
        assert "nemotron" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_ibm_family(self) -> None:
        """Contains IBM Granite 3 family."""
        assert "granite3-dense" in TOOL_CAPABLE_MODEL_FAMILIES
        assert "granite3-moe" in TOOL_CAPABLE_MODEL_FAMILIES

    def test_contains_gpt_oss(self) -> None:
        """Contains GPT-OSS."""
        assert "gpt-oss" in TOOL_CAPABLE_MODEL_FAMILIES


class TestLocalProviderRegistry:
    """Tests for LocalProviderRegistry."""

    def test_get_registered_providers_empty_initially(self) -> None:
        """Registry starts with providers registered on import."""
        # The ollama_api module registers itself on import
        providers = LocalProviderRegistry.get_registered_providers()
        # Should have at least ollama registered
        assert "ollama" in providers

    def test_get_provider_class_ollama(self) -> None:
        """Can retrieve Ollama provider class."""
        from static.providers.local.ollama_api import OllamaAPI
        provider_class = LocalProviderRegistry.get_provider_class("ollama")
        assert provider_class is OllamaAPI

    def test_get_provider_class_unknown(self) -> None:
        """Returns None for unknown provider."""
        provider_class = LocalProviderRegistry.get_provider_class("unknown-provider")
        assert provider_class is None
