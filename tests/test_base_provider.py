"""
Tests for the BaseProvider class.
"""
import pytest
from typing import Any, Optional

from unifiedllm.providers.base import BaseProvider
from unifiedllm.types import ChatResponse, Messages, HTTPResponse, LLMUsage, APIErrorDetails
from unifiedllm.errors import (
    ProviderHTTPError,
    ProviderAPIError,
    ProviderParseError,
    MissingAPIKeyError,
    HTTPTimeoutError,
    HTTPNetworkError,
)


# ============================================================================
# Concrete Test Provider Implementation
# ============================================================================

class ConcreteTestProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""
    
    name = "test"
    display_name = "Test Provider"
    base_url = "https://api.test.com"
    chat_endpoint = "/v1/chat"
    env_key_name = "TEST_API_KEY"
    
    def config(self, *, temperature=None, top_p=None, max_tokens=None, stop=None, custom=None):
        """Simple config implementation."""
        self._config = {}
        if temperature is not None:
            self._config["temperature"] = temperature
    
    def system_prompt(self, text: str):
        """Simple system prompt implementation."""
        self._system_prompt = text
    
    def chat(self, *, messages: Messages) -> ChatResponse:
        """Simple chat implementation."""
        payload = {"messages": list(messages)}
        http_resp = self._get_response(
            url=self._get_chat_url(),
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload
        )
        return self._parse_chat_response(http_resp)
    
    def _extract_error_details(self, resp) -> APIErrorDetails:
        """Simple error extraction."""
        try:
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                return APIErrorDetails(
                    message=data["error"].get("message", "Unknown error"),
                    error_type=data["error"].get("type"),
                    code=data["error"].get("code"),
                    raw=data
                )
        except Exception:
            pass
        return APIErrorDetails(message=resp.text, raw=resp.text)
    
    def _extract_text(self, data: dict[str, Any]) -> str:
        """Simple text extraction."""
        if "text" not in data:
            raise ProviderParseError(
                provider=self.name,
                display_name=self.display_name,
                model=self.model,
                detail="Missing 'text' field in response"
            )
        return str(data["text"])
    
    def _extract_usage(self, data: dict[str, Any]) -> Optional[LLMUsage]:
        """Simple usage extraction."""
        if "usage" in data and isinstance(data["usage"], dict):
            u = data["usage"]
            return LLMUsage(
                prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"),
                total_tokens=u.get("total_tokens")
            )
        return None
    
    def _extract_request_id(self, http_resp: HTTPResponse) -> Optional[str]:
        """Simple request ID extraction."""
        rid = http_resp.headers.get("x-request-id")
        return rid if rid else None


# ============================================================================
# Tests
# ============================================================================

class TestBaseProviderInitialization:
    """Tests for BaseProvider initialization."""
    
    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that initializing with empty API key raises MissingAPIKeyError."""
        # Ensure env var is not set
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        
        with pytest.raises(MissingAPIKeyError) as exc_info:
            ConcreteTestProvider(model="test-model", api_key="", timeout=30.0)
        
        assert exc_info.value.provider == "test"
        assert exc_info.value.display_name == "Test Provider"
        assert exc_info.value.model == "test-model"
        assert "TEST_API_KEY" in str(exc_info.value)
    
    def test_missing_api_key_with_default_raises_error(self, monkeypatch):
        """Test that api_key='default' with no env var raises MissingAPIKeyError."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        
        with pytest.raises(MissingAPIKeyError):
            ConcreteTestProvider(model="test-model", api_key="default", timeout=30.0)
    
    def test_valid_api_key_succeeds(self):
        """Test that valid API key allows initialization."""
        provider = ConcreteTestProvider(model="test-model", api_key="valid-key", timeout=30.0)
        
        assert provider.api_key == "valid-key"
        assert provider.model == "test-model"
        assert provider.timeout == 30.0
    
    def test_loads_api_key_from_env(self, monkeypatch):
        """Test that api_key='default' loads from environment variable."""
        monkeypatch.setenv("TEST_API_KEY", "env-key-value")
        
        provider = ConcreteTestProvider(model="test-model", api_key="default", timeout=30.0)
        
        assert provider.api_key == "env-key-value"


class TestBaseProviderErrorMapping:
    """Tests for HTTP error to Provider error mapping."""
    
    def test_timeout_error_becomes_provider_http_error(self, mock_http_error):
        """Test that HTTPTimeoutError is mapped to ProviderHTTPError."""
        mock_http_error(HTTPTimeoutError, "Request timed out")
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert exc_info.value.provider == "test"
        assert exc_info.value.display_name == "Test Provider"
        assert exc_info.value.model == "test-model"
    
    def test_network_error_becomes_provider_http_error(self, mock_http_error):
        """Test that HTTPNetworkError is mapped to ProviderHTTPError."""
        mock_http_error(HTTPNetworkError, "Network error")
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert "Test Provider" in str(exc_info.value)
    
    def test_status_error_becomes_provider_api_error(self, mock_http_status_error):
        """Test that HTTPStatusError is mapped to ProviderAPIError."""
        error_json = {
            "error": {
                "message": "Invalid request",
                "type": "invalid_request",
                "code": "400"
            }
        }
        mock_http_status_error(json_data=error_json, status_code=400)
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert exc_info.value.provider == "test"
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_type == "invalid_request"
        assert exc_info.value.code == "400"
        assert "Invalid request" in exc_info.value.detail


class TestBaseProviderParseErrors:
    """Tests for response parsing errors."""
    
    def test_invalid_json_raises_provider_parse_error(self, monkeypatch, fake_httpx_response):
        """Test that invalid JSON raises ProviderParseError."""
        # Create a response that will fail json parsing
        resp = fake_httpx_response({}, status_code=200)
        resp.json.side_effect = ValueError("Invalid JSON")
        resp.text = "not valid json"
        
        def mock_post(self, *, url, headers, payload):
            return resp, 10.0
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", mock_post)
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert exc_info.value.provider == "test"
        assert exc_info.value.display_name == "Test Provider"
        assert "not valid JSON" in exc_info.value.detail
        assert exc_info.value.raw == "not valid json"
    
    def test_non_dict_json_raises_provider_parse_error(self, mock_http_success):
        """Test that JSON array (not object) raises ProviderParseError."""
        mock_http_success(json_data=["array", "instead", "of", "object"])
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert "Unexpected JSON shape" in exc_info.value.detail
        assert "expected object" in exc_info.value.detail
    
    def test_missing_required_field_raises_provider_parse_error(self, mock_http_success):
        """Test that missing required field raises ProviderParseError from provider."""
        # ConcreteTestProvider expects 'text' field
        mock_http_success(json_data={"no_text_field": "value"})
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        
        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert "Missing 'text' field" in exc_info.value.detail


class TestBaseProviderSuccessfulResponse:
    """Tests for successful response handling."""
    
    def test_successful_response_returns_chat_response(self, mock_http_success):
        """Test that successful HTTP response is parsed into ChatResponse."""
        json_data = {
            "text": "Hello, world!",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        headers = {"x-request-id": "req-123"}
        mock_http_success(json_data=json_data, headers=headers, latency_ms=25.5)
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert isinstance(response, ChatResponse)
        assert response.text == "Hello, world!"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.request_id == "req-123"
        assert response.latency_ms == 25.5
        assert response.provider == "test"
        assert response.model == "test-model"
        assert response.raw == json_data
    
    def test_successful_response_without_usage(self, mock_http_success):
        """Test that response without usage field works."""
        json_data = {"text": "Response text"}
        mock_http_success(json_data=json_data)
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert response.text == "Response text"
        assert response.usage is None
    
    def test_successful_response_without_request_id(self, mock_http_success):
        """Test that response without request_id works."""
        json_data = {"text": "Response text"}
        mock_http_success(json_data=json_data, headers={})
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "test"}])
        
        assert response.text == "Response text"
        assert response.request_id is None


class TestBaseProviderClose:
    """Tests for closing provider resources."""
    
    def test_close_calls_http_client_close(self, monkeypatch):
        """Test that provider.close() closes the underlying HTTP client."""
        from unittest.mock import Mock
        
        mock_http_close = Mock()
        
        def mock_http_init(self, timeout):
            self.close = mock_http_close
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "__init__", mock_http_init)
        
        provider = ConcreteTestProvider(model="test-model", api_key="key", timeout=30.0)
        provider.close()
        
        mock_http_close.assert_called_once()


class TestBaseProviderImplementation:
    """Tests for BaseProvider implementation details and checks."""

    def test_missing_class_attributes_raises_error(self):
        """Test that missing required class attributes raises NotImplementedError."""
        class IncompleteProvider(BaseProvider):
            # Missing name, base_url, etc.
            def config(self, **kwargs): pass
            def system_prompt(self, text): pass
            def chat(self, *, messages): pass
            def _extract_error_details(self, resp): pass
            def _extract_text(self, data): pass
            def _extract_usage(self, data): pass
            def _extract_request_id(self, resp): pass

        with pytest.raises(NotImplementedError) as exc_info:
            IncompleteProvider(model="test", api_key="key", timeout=30)
        
        assert "must define class attribute" in str(exc_info.value)

    def test_get_chat_url_missing_param_raises_error(self):
        """Test that missing path parameter in URL generation raises ValueError."""
        class ParamProvider(ConcreteTestProvider):
            chat_endpoint = "/v1/{custom_param}/chat"

        provider = ParamProvider(model="test", api_key="key", timeout=30)
        
        # Should raise ValueError because custom_param is not provided
        with pytest.raises(ValueError) as exc_info:
            provider._get_chat_url(other="value")
        
        assert "Missing path parameter 'custom_param'" in str(exc_info.value)
