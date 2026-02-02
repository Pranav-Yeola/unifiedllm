"""
Tests for the LLM client class.
"""

import pytest
from unittest.mock import Mock, MagicMock

from unifiedllm.client import LLM
from unifiedllm.types import ChatResponse
from unifiedllm.errors import ProviderNotSupportedError


class TestLLMInitialization:
    """Tests for LLM client initialization."""

    def test_supported_provider_initializes(self):
        """Test that supported provider names initialize correctly."""
        # All three supported providers should work
        for provider_name in ["openai", "anthropic", "gemini"]:
            llm = LLM(provider=provider_name, model="test-model", api_key="test-key")
            assert llm.provider_name == provider_name
            assert llm.model == "test-model"
            llm.close()

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ProviderNotSupportedError."""
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            LLM(provider="invalid-provider", model="test-model", api_key="test-key")

        assert "invalid-provider" in str(exc_info.value).lower()
        assert "not supported" in str(exc_info.value).lower()

    def test_provider_name_is_normalized(self):
        """Test that provider name is lowercased and stripped."""
        llm = LLM(provider="  OpenAI  ", model="test-model", api_key="test-key")
        assert llm.provider_name == "openai"
        llm.close()


class TestLLMChatNormalization:
    """Tests for message normalization in chat method."""

    def test_prompt_converted_to_messages(self, monkeypatch):
        """Test that prompt is converted to messages format."""
        mock_provider = Mock()
        mock_provider.chat.return_value = ChatResponse(text="response")

        def mock_provider_init(self, model, api_key, timeout):
            return None

        # Mock the provider class
        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        llm.chat(prompt="Hello, world!")

        # Verify provider.chat was called with normalized messages
        mock_provider.chat.assert_called_once()
        call_args = mock_provider.chat.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    def test_messages_passed_through(self, monkeypatch):
        """Test that messages are passed through as-is."""
        mock_provider = Mock()
        mock_provider.chat.return_value = ChatResponse(text="response")

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        test_messages = [
            {"role": "user", "content": "First message"},
            {"role": "model", "content": "Response"},
            {"role": "user", "content": "Follow-up"},
        ]

        llm.chat(messages=test_messages)

        mock_provider.chat.assert_called_once()
        call_args = mock_provider.chat.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 3
        assert messages[0]["content"] == "First message"
        assert messages[1]["content"] == "Response"
        assert messages[2]["content"] == "Follow-up"

    def test_neither_prompt_nor_messages_raises_error(self):
        """Test that providing neither prompt nor messages raises ValueError."""
        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat()

        assert "exactly one" in str(exc_info.value).lower()
        llm.close()

    def test_both_prompt_and_messages_raises_error(self):
        """Test that providing both prompt and messages raises ValueError."""
        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat(prompt="Hello", messages=[{"role": "user", "content": "Hi"}])

        assert "exactly one" in str(exc_info.value).lower()
        llm.close()

    def test_chat_with_role_enum(self, monkeypatch):
        """Test that Role enum members work in messages."""
        from unifiedllm.enums import Role

        mock_provider = Mock()
        mock_provider.chat.return_value = ChatResponse(text="response")

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        # Use Role enum objects instead of strings
        messages = [
            {"role": Role.USER, "content": "Hello"},
            {"role": Role.MODEL, "content": "Hi"},
        ]

        llm.chat(messages=messages)

        mock_provider.chat.assert_called_once()


class TestLLMConfigurationMethods:
    """Tests for configuration methods."""

    def test_config_returns_self_for_chaining(self, monkeypatch):
        """Test that config() returns self for method chaining."""
        mock_provider = Mock()

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        result = llm.config(temperature=0.7, max_tokens=100)

        assert result is llm
        mock_provider.config.assert_called_once_with(
            temperature=0.7, top_p=None, max_tokens=100, stop=None, custom=None
        )

    def test_system_prompt_returns_self_for_chaining(self, monkeypatch):
        """Test that system_prompt() returns self for method chaining."""
        mock_provider = Mock()

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        result = llm.system_prompt("You are a helpful assistant")

        assert result is llm
        mock_provider.system_prompt.assert_called_once_with(
            "You are a helpful assistant"
        )

    def test_method_chaining_works(self, monkeypatch):
        """Test that methods can be chained together."""
        mock_provider = Mock()
        mock_provider.chat.return_value = ChatResponse(text="response")

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        # Chain multiple configuration calls
        response = (
            llm.config(temperature=0.5).system_prompt("Be helpful").chat(prompt="Hello")
        )

        assert isinstance(response, ChatResponse)
        mock_provider.config.assert_called_once()
        mock_provider.system_prompt.assert_called_once()
        mock_provider.chat.assert_called_once()

    def test_config_passes_all_parameters(self, monkeypatch):
        """Test that config passes all parameters to provider."""
        mock_provider = Mock()

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        custom_params = {"frequency_penalty": 0.5}
        llm.config(
            temperature=0.8,
            top_p=0.9,
            max_tokens=200,
            stop=["END", "STOP"],
            custom=custom_params,
        )

        mock_provider.config.assert_called_once_with(
            temperature=0.8,
            top_p=0.9,
            max_tokens=200,
            stop=["END", "STOP"],
            custom=custom_params,
        )


class TestLLMContextManager:
    """Tests for context manager support."""

    def test_context_manager_calls_close(self, monkeypatch):
        """Test that exiting context manager calls close()."""
        mock_provider = Mock()

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        with LLM(provider="openai", model="gpt-4", api_key="test-key") as llm:
            llm._provider = mock_provider
            assert llm is not None

        # After exiting context, close should have been called
        mock_provider.close.assert_called_once()

    def test_context_manager_returns_self(self):
        """Test that entering context manager returns the LLM instance."""
        llm_instance = LLM(provider="openai", model="gpt-4", api_key="test-key")

        with llm_instance as llm:
            assert llm is llm_instance

        llm_instance.close()


class TestLLMClose:
    """Tests for closing LLM resources."""

    def test_close_calls_provider_close(self, monkeypatch):
        """Test that LLM.close() calls the provider's close method."""
        mock_provider = Mock()

        def mock_provider_init(self, model, api_key, timeout):
            return None

        from unifiedllm.providers import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_provider_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")
        llm._provider = mock_provider

        llm.close()

        mock_provider.close.assert_called_once()


class TestLLMProviderSelection:
    """Tests for provider selection logic."""

    def test_openai_provider_selected(self, monkeypatch):
        """Test that 'openai' selects OpenAIProvider."""
        from unifiedllm.providers import OpenAIProvider

        init_called = []
        original_init = OpenAIProvider.__init__

        def mock_init(self, model, api_key, timeout):
            init_called.append(True)
            # Need to initialize base properly
            original_init(self, model=model, api_key=api_key, timeout=timeout)

        monkeypatch.setattr(OpenAIProvider, "__init__", mock_init)

        llm = LLM(provider="openai", model="gpt-4", api_key="test-key")

        assert len(init_called) == 1
        assert isinstance(llm._provider, OpenAIProvider)
        llm.close()

    def test_anthropic_provider_selected(self, monkeypatch):
        """Test that 'anthropic' selects AnthropicProvider."""
        from unifiedllm.providers import AnthropicProvider

        init_called = []
        original_init = AnthropicProvider.__init__

        def mock_init(self, model, api_key, timeout):
            init_called.append(True)
            original_init(self, model=model, api_key=api_key, timeout=timeout)

        monkeypatch.setattr(AnthropicProvider, "__init__", mock_init)

        llm = LLM(provider="anthropic", model="claude-3", api_key="test-key")

        assert len(init_called) == 1
        assert isinstance(llm._provider, AnthropicProvider)
        llm.close()

    def test_gemini_provider_selected(self, monkeypatch):
        """Test that 'gemini' selects GeminiProvider."""
        from unifiedllm.providers import GeminiProvider

        init_called = []
        original_init = GeminiProvider.__init__

        def mock_init(self, model, api_key, timeout):
            init_called.append(True)
            original_init(self, model=model, api_key=api_key, timeout=timeout)

        monkeypatch.setattr(GeminiProvider, "__init__", mock_init)

        llm = LLM(provider="gemini", model="gemini-pro", api_key="test-key")

        assert len(init_called) == 1
        assert isinstance(llm._provider, GeminiProvider)
        llm.close()
