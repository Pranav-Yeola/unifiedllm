"""
Tests for the Anthropic provider.
"""
import pytest

from unifiedllm.providers.anthropic import AnthropicProvider
from unifiedllm.types import ChatResponse
from unifiedllm.errors import (
    ProviderHTTPError,
    ProviderAPIError,
    ProviderParseError,
    MissingAPIKeyError,
    HTTPTimeoutError,
    HTTPNetworkError,
)
from conftest import ANTHROPIC_SUCCESS_RESPONSE, ANTHROPIC_ERROR_RESPONSE


class TestAnthropicProviderSuccess:
    """Tests for successful Anthropic API responses."""
    
    def test_success_response_with_all_fields(self, mock_http_success):
        """Test successful response with all fields populated."""
        headers = {"request-id": "req-anthropic-abc123"}
        mock_http_success(
            json_data=ANTHROPIC_SUCCESS_RESPONSE,
            headers=headers,
            latency_ms=35.7
        )
        
        provider = AnthropicProvider(model="claude-3-opus-20240229", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Hello"}])
        
        assert isinstance(response, ChatResponse)
        assert response.text == "Hello! How can I assist you today?"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-opus-20240229"
        assert response.status_code == 200
        assert response.latency_ms == 35.7
        assert response.request_id == "req-anthropic-abc123"
        assert response.raw is not None
        assert response.raw == ANTHROPIC_SUCCESS_RESPONSE
        
        # Check usage
        assert response.usage is not None
        assert response.usage.prompt_tokens == 15
        assert response.usage.completion_tokens == 25
        assert response.usage.total_tokens == 40  # computed by provider
    
    def test_success_with_x_request_id_header(self, mock_http_success):
        """Test that request_id uses x-request-id as fallback."""
        headers = {"x-request-id": "req-x-456"}
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE, headers=headers)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert response.request_id == "req-x-456"
    
    def test_success_with_multiple_text_blocks(self, mock_http_success):
        """Test response with multiple text content blocks."""
        response_data = {
            "content": [
                {"type": "text", "text": "First part. "},
                {"type": "text", "text": "Second part."}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15}
        }
        mock_http_success(json_data=response_data)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert response.text == "First part. Second part."
    
    def test_success_with_empty_content(self, mock_http_success):
        """Test response with empty content array returns empty text."""
        response_data = {
            "content": [],
            "usage": {"input_tokens": 5, "output_tokens": 0}
        }
        mock_http_success(json_data=response_data)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert response.text == ""
    
    def test_success_with_non_text_blocks(self, mock_http_success):
        """Test that non-text blocks are skipped."""
        response_data = {
            "content": [
                {"type": "image", "source": "..."},
                {"type": "text", "text": "Text content"},
                {"type": "tool_use", "id": "123"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        mock_http_success(json_data=response_data)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert response.text == "Text content"
    
    def test_success_without_usage(self, mock_http_success):
        """Test response without usage field."""
        response_data = {
            "content": [{"type": "text", "text": "Response"}]
        }
        mock_http_success(json_data=response_data)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert response.text == "Response"
        assert response.usage is None
    
    def test_success_with_system_prompt(self, mock_http_success, monkeypatch):
        """Test that system prompt is added to payload."""
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE)
        
        # Track the payload sent
        sent_payload = {}
        
        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock
            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = ANTHROPIC_SUCCESS_RESPONSE
            resp.text = str(ANTHROPIC_SUCCESS_RESPONSE)
            import httpx
            resp.headers = httpx.Headers({})
            return resp, 10.0
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", capture_post)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        provider.system_prompt("You are a helpful assistant")
        provider.chat(messages=[{"role": "user", "content": "Hello"}])
        
        # System prompt should be in payload
        assert "system" in sent_payload
        assert sent_payload["system"] == "You are a helpful assistant"
    
    def test_default_max_tokens_applied(self, mock_http_success, monkeypatch):
        """Test that DEFAULT_MAX_TOKENS is applied when not configured."""
        sent_payload = {}
        
        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock
            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = ANTHROPIC_SUCCESS_RESPONSE
            resp.text = str(ANTHROPIC_SUCCESS_RESPONSE)
            import httpx
            resp.headers = httpx.Headers({})
            return resp, 10.0
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", capture_post)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        provider.chat(messages=[{"role": "user", "content": "Hello"}])
        
        # Default max_tokens should be set
        assert "max_tokens" in sent_payload
        assert sent_payload["max_tokens"] == AnthropicProvider.DEFAULT_MAX_TOKENS


class TestAnthropicProviderAPIErrors:
    """Tests for Anthropic API error responses."""
    
    def test_api_error_with_full_details(self, mock_http_status_error):
        """Test API error with type and message."""
        headers = {"request-id": "req-error-xyz"}
        mock_http_status_error(
            json_data=ANTHROPIC_ERROR_RESPONSE,
            status_code=401,
            headers=headers
        )
        
        provider = AnthropicProvider(model="claude-3", api_key="bad-key", timeout=30.0)
        
        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        error = exc_info.value
        assert error.provider == "anthropic"
        assert error.display_name == "Anthropic"
        assert error.model == "claude-3"
        assert error.status_code == 401
        assert error.error_type == "authentication_error"
        assert "Invalid API key" in error.detail
        assert error.request_id == "req-error-xyz"
        assert error.raw == ANTHROPIC_ERROR_RESPONSE
    
    def test_api_error_with_request_id_in_body(self, mock_http_status_error):
        """Test that request_id from body is used when header is missing."""
        error_response = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            },
            "request_id": "req-from-body-123"
        }
        mock_http_status_error(json_data=error_response, status_code=429, headers={})
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        # Should extract request_id from body when header is missing
        assert exc_info.value.request_id == "req-from-body-123"
    
    def test_api_error_without_error_object(self, mock_http_status_error):
        """Test error response without nested error object."""
        error_response = {"message": "Something went wrong"}
        mock_http_status_error(json_data=error_response, status_code=500)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Something went wrong"
    
    def test_api_error_status_429(self, mock_http_status_error):
        """Test rate limit error (429)."""
        error_response = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }
        mock_http_status_error(json_data=error_response, status_code=429)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_type == "rate_limit_error"


class TestAnthropicProviderTransportErrors:
    """Tests for network/timeout errors."""
    
    def test_timeout_error_raises_provider_http_error(self, mock_http_error):
        """Test that timeout is converted to ProviderHTTPError."""
        mock_http_error(HTTPTimeoutError, "Request timed out")
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        error = exc_info.value
        assert error.provider == "anthropic"
        assert error.display_name == "Anthropic"
        assert error.model == "claude-3"
        assert "Anthropic" in str(error)
    
    def test_network_error_raises_provider_http_error(self, mock_http_error):
        """Test that network error is converted to ProviderHTTPError."""
        mock_http_error(HTTPNetworkError, "Network error")
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert exc_info.value.provider == "anthropic"


class TestAnthropicProviderParseErrors:
    """Tests for response parsing errors."""
    
    def test_missing_content_raises_parse_error(self, mock_http_success):
        """Test that missing 'content' field raises ProviderParseError."""
        mock_http_success(json_data={"usage": {"input_tokens": 10}})
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        error = exc_info.value
        assert error.provider == "anthropic"
        assert "content" in error.detail.lower()
    
    def test_content_not_a_list_raises_parse_error(self, mock_http_success):
        """Test that 'content' not being a list raises ProviderParseError."""
        mock_http_success(json_data={"content": "not-a-list"})
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        assert "content" in exc_info.value.detail.lower()
        assert "not a list" in exc_info.value.detail.lower()


class TestAnthropicProviderMissingAPIKey:
    """Tests for missing API key."""
    
    def test_empty_api_key_raises_error(self, monkeypatch):
        """Test that empty API key raises MissingAPIKeyError."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        with pytest.raises(MissingAPIKeyError) as exc_info:
            AnthropicProvider(model="claude-3", api_key="", timeout=30.0)
        
        error = exc_info.value
        assert error.provider == "anthropic"
        assert error.display_name == "Anthropic"
        assert error.model == "claude-3"
        assert "ANTHROPIC_API_KEY" in str(error)
    
    def test_default_api_key_without_env_raises_error(self, monkeypatch):
        """Test that api_key='default' without env var raises error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        with pytest.raises(MissingAPIKeyError):
            AnthropicProvider(model="claude-3", api_key="default", timeout=30.0)


class TestAnthropicProviderRoleValidation:
    """Tests for message role validation."""
    
    def test_invalid_role_system_raises_error(self):
        """Test that 'system' role in messages raises ValueError."""
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "system", "content": "System prompt"}])
        
        assert "unsupported role" in str(exc_info.value).lower()
        assert "system" in str(exc_info.value).lower()
    
    def test_invalid_role_assistant_raises_error(self):
        """Test that 'assistant' role raises ValueError (should use 'model')."""
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "assistant", "content": "Test"}])
        
        assert "unsupported role" in str(exc_info.value).lower()
    
    def test_invalid_role_custom_raises_error(self):
        """Test that custom role raises ValueError."""
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "banana", "content": "Test"}])
        
        assert "unsupported role" in str(exc_info.value).lower()
        assert "banana" in str(exc_info.value).lower()
    
    def test_valid_role_user(self, mock_http_success):
        """Test that 'user' role is valid."""
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response.text is not None
    
    def test_valid_role_model(self, mock_http_success):
        """Test that 'model' role is valid and converted to 'assistant'."""
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        # Should not raise - 'model' should be converted to 'assistant'
        response = provider.chat(messages=[
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there"}
        ])
        assert response.text is not None


class TestAnthropicProviderConfigValidation:
    """Tests for configuration parameter validation."""
    
    def test_unsupported_custom_parameter_raises_error(self):
        """Test that unsupported custom parameter raises ValueError."""
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        with pytest.raises(ValueError) as exc_info:
            provider.config(custom={"invalid_param": "value", "another_bad": 123})
        
        error_msg = str(exc_info.value).lower()
        assert "unsupported" in error_msg
        assert "anthropic" in error_msg
    
    def test_supported_custom_parameters_work(self, mock_http_success):
        """Test that supported custom parameters don't raise errors."""
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        
        # These are in SUPPORTED_CUSTOM_KEYS
        provider.config(custom={
            "top_k": 40,
            "metadata": {"user_id": "user-123"}
        })
        
        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None
    
    def test_standard_config_parameters(self, mock_http_success):
        """Test that standard config parameters work."""
        mock_http_success(json_data=ANTHROPIC_SUCCESS_RESPONSE)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        provider.config(
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            stop=["END"]
        )
        
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None
    
    def test_stop_mapped_to_stop_sequences(self, mock_http_success, monkeypatch):
        """Test that 'stop' parameter is mapped to 'stop_sequences' for Anthropic."""
        sent_payload = {}
        
        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock
            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = ANTHROPIC_SUCCESS_RESPONSE
            resp.text = str(ANTHROPIC_SUCCESS_RESPONSE)
            import httpx
            resp.headers = httpx.Headers({})
            return resp, 10.0
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", capture_post)
        
        provider = AnthropicProvider(model="claude-3", api_key="test-key", timeout=30.0)
        provider.config(stop=["STOP", "END"])
        provider.chat(messages=[{"role": "user", "content": "Test"}])
        
        # 'stop' should be mapped to 'stop_sequences'
        assert "stop_sequences" in sent_payload
        assert sent_payload["stop_sequences"] == ["STOP", "END"]
