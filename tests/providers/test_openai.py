"""
Tests for the OpenAI provider.
"""

import pytest

from unifiedllm.providers.openai import OpenAIProvider
from unifiedllm.types import ChatResponse
from unifiedllm.errors import (
    ProviderHTTPError,
    ProviderAPIError,
    ProviderParseError,
    MissingAPIKeyError,
    HTTPTimeoutError,
    HTTPNetworkError,
)
from conftest import OPENAI_SUCCESS_RESPONSE, OPENAI_ERROR_RESPONSE


class TestOpenAIProviderSuccess:
    """Tests for successful OpenAI API responses."""

    def test_success_response_with_all_fields(self, mock_http_success):
        """Test successful response with all fields populated."""
        headers = {"x-request-id": "req-openai-123"}
        mock_http_success(
            json_data=OPENAI_SUCCESS_RESPONSE, headers=headers, latency_ms=42.5
        )

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, ChatResponse)
        assert response.text == "Hello! How can I help you today?"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.status_code == 200
        assert response.latency_ms == 42.5
        assert response.request_id == "req-openai-123"
        assert response.raw is not None
        assert response.raw == OPENAI_SUCCESS_RESPONSE

        # Check usage
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    def test_success_with_request_id_fallback(self, mock_http_success):
        """Test that request_id falls back to 'request-id' header."""
        headers = {"request-id": "req-fallback-456"}
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE, headers=headers)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.request_id == "req-fallback-456"

    def test_success_with_empty_choices(self, mock_http_success):
        """Test response with empty choices array returns empty text."""
        response_data = {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        mock_http_success(json_data=response_data)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == ""
        assert response.usage.total_tokens == 5

    def test_success_without_usage(self, mock_http_success):
        """Test response without usage field."""
        response_data = {
            "choices": [{"message": {"role": "assistant", "content": "Response text"}}]
        }
        mock_http_success(json_data=response_data)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == "Response text"
        assert response.usage is None

    def test_success_with_system_prompt(self, mock_http_success, monkeypatch):
        """Test that system prompt is added to messages."""
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE)

        # Track the payload sent
        sent_payload = {}

        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock

            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = OPENAI_SUCCESS_RESPONSE
            resp.headers = {}
            resp.text = str(OPENAI_SUCCESS_RESPONSE)
            import httpx

            resp.headers = httpx.Headers({})
            return resp, 10.0

        from unifiedllm.http import HTTPClient

        monkeypatch.setattr(HTTPClient, "post", capture_post)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        provider.system_prompt("You are a helpful assistant")
        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        # System message should be first
        assert len(sent_payload["messages"]) == 2
        assert sent_payload["messages"][0]["role"] == "system"
        assert sent_payload["messages"][0]["content"] == "You are a helpful assistant"
        assert sent_payload["messages"][1]["role"] == "user"


class TestOpenAIProviderAPIErrors:
    """Tests for OpenAI API error responses."""

    def test_api_error_with_full_details(self, mock_http_status_error):
        """Test API error with type, code, and message."""
        headers = {"x-request-id": "req-error-789"}
        mock_http_status_error(
            json_data=OPENAI_ERROR_RESPONSE, status_code=401, headers=headers
        )

        provider = OpenAIProvider(model="gpt-4", api_key="bad-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "openai"
        assert error.display_name == "OpenAI"
        assert error.model == "gpt-4"
        assert error.status_code == 401
        assert error.error_type == "invalid_request_error"
        assert error.code == "invalid_api_key"
        assert "Invalid API key" in error.detail
        assert error.request_id == "req-error-789"
        assert error.raw == OPENAI_ERROR_RESPONSE

    def test_api_error_status_500(self, mock_http_status_error):
        """Test server error (500)."""
        error_response = {
            "error": {"message": "Internal server error", "type": "server_error"}
        }
        mock_http_status_error(json_data=error_response, status_code=500)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    def test_api_error_without_structured_error(self, mock_http_status_error):
        """Test error response without standard error structure."""
        error_response = {"message": "Something went wrong"}
        mock_http_status_error(json_data=error_response, status_code=400)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.status_code == 400
        # Should extract message from top level
        assert exc_info.value.detail == "Something went wrong"


class TestOpenAIProviderTransportErrors:
    """Tests for network/timeout errors."""

    def test_timeout_error_raises_provider_http_error(self, mock_http_error):
        """Test that timeout is converted to ProviderHTTPError."""
        mock_http_error(HTTPTimeoutError, "Request timed out")

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "openai"
        assert error.display_name == "OpenAI"
        assert error.model == "gpt-4"
        assert "OpenAI" in str(error)

    def test_network_error_raises_provider_http_error(self, mock_http_error):
        """Test that network error is converted to ProviderHTTPError."""
        mock_http_error(HTTPNetworkError, "Network error")

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.provider == "openai"


class TestOpenAIProviderParseErrors:
    """Tests for response parsing errors."""

    def test_missing_choices_raises_parse_error(self, mock_http_success):
        """Test that missing 'choices' field raises ProviderParseError."""
        mock_http_success(json_data={"usage": {"total_tokens": 10}})

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "openai"
        assert "choices" in error.detail.lower()
        assert "missing" in error.detail.lower()

    def test_choices_not_a_list_raises_parse_error(self, mock_http_success):
        """Test that 'choices' not being a list raises ProviderParseError."""
        mock_http_success(json_data={"choices": "not-a-list"})

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert "choices" in exc_info.value.detail.lower()
        assert "not a list" in exc_info.value.detail.lower()


class TestOpenAIProviderMissingAPIKey:
    """Tests for missing API key."""

    def test_empty_api_key_raises_error(self, monkeypatch):
        """Test that empty API key raises MissingAPIKeyError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError) as exc_info:
            OpenAIProvider(model="gpt-4", api_key="", timeout=30.0)

        error = exc_info.value
        assert error.provider == "openai"
        assert error.display_name == "OpenAI"
        assert error.model == "gpt-4"
        assert "OPENAI_API_KEY" in str(error)

    def test_default_api_key_without_env_raises_error(self, monkeypatch):
        """Test that api_key='default' without env var raises error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            OpenAIProvider(model="gpt-4", api_key="default", timeout=30.0)


class TestOpenAIProviderRoleValidation:
    """Tests for message role validation."""

    def test_invalid_role_system_raises_error(self):
        """Test that 'system' role in messages raises ValueError."""
        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "system", "content": "System prompt"}])

        assert "unsupported role" in str(exc_info.value).lower()
        assert "system" in str(exc_info.value).lower()

    def test_invalid_role_custom_raises_error(self):
        """Test that custom role raises ValueError."""
        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "banana", "content": "Test"}])

        assert "unsupported role" in str(exc_info.value).lower()
        assert "banana" in str(exc_info.value).lower()

    def test_valid_role_user(self, mock_http_success):
        """Test that 'user' role is valid."""
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response.text is not None

    def test_valid_role_model(self, mock_http_success):
        """Test that 'model' role is valid and converted to 'assistant'."""
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        # Should not raise - 'model' should be converted to 'assistant'
        response = provider.chat(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi there"},
            ]
        )
        assert response.text is not None


class TestOpenAIProviderConfigValidation:
    """Tests for configuration parameter validation."""

    def test_unsupported_custom_parameter_raises_error(self):
        """Test that unsupported custom parameter raises ValueError."""
        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.config(custom={"invalid_param": "value", "another_invalid": 123})

        error_msg = str(exc_info.value).lower()
        assert "unsupported" in error_msg
        assert "openai" in error_msg
        assert "invalid_param" in error_msg or "another_invalid" in error_msg

    def test_supported_custom_parameters_work(self, mock_http_success):
        """Test that supported custom parameters don't raise errors."""
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)

        # These are in SUPPORTED_CUSTOM_KEYS
        provider.config(
            custom={"presence_penalty": 0.5, "frequency_penalty": 0.3, "seed": 42}
        )

        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None

    def test_standard_config_parameters(self, mock_http_success):
        """Test that standard config parameters work."""
        mock_http_success(json_data=OPENAI_SUCCESS_RESPONSE)

        provider = OpenAIProvider(model="gpt-4", api_key="test-key", timeout=30.0)
        provider.config(temperature=0.7, top_p=0.9, max_tokens=100, stop=["END"])

        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None
