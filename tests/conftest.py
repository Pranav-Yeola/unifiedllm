"""
Shared fixtures and test utilities for unifiedllm test suite.
"""
import pytest
import httpx
from unittest.mock import Mock


# ============================================================================
# Sample Provider Response Payloads
# ============================================================================

OPENAI_SUCCESS_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

OPENAI_ERROR_RESPONSE = {
    "error": {
        "message": "Invalid API key provided",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
    }
}

ANTHROPIC_SUCCESS_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! How can I assist you today?"
        }
    ],
    "model": "claude-3-opus-20240229",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 25
    }
}

ANTHROPIC_ERROR_RESPONSE = {
    "error": {
        "type": "authentication_error",
        "message": "Invalid API key"
    },
    "request_id": "req_abc123"
}

GEMINI_SUCCESS_RESPONSE = {
    "responseId": "resp_xyz789",
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Hello! I'm here to help."}
                ],
                "role": "model"
            },
            "finishReason": "STOP"
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 18,
        "totalTokenCount": 30
    }
}

GEMINI_ERROR_RESPONSE = {
    "error": {
        "code": 400,
        "message": "API key not valid",
        "status": "INVALID_ARGUMENT"
    }
}


# ============================================================================
# Fixtures for Creating Mock HTTP Responses
# ============================================================================

@pytest.fixture
def fake_httpx_response():
    """
    Factory fixture to create fake httpx.Response objects.
    
    Usage:
        resp = fake_httpx_response({"key": "value"}, status_code=200, headers={"X-Request-ID": "123"})
    """
    def _make(json_data, status_code=200, headers=None):
        resp = Mock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.text = str(json_data)
        resp.headers = httpx.Headers(headers or {})
        return resp
    return _make


# ============================================================================
# Fixtures for Mocking HTTPClient.post()
# ============================================================================

@pytest.fixture
def mock_http_success(monkeypatch, fake_httpx_response):
    """
    Mock HTTPClient.post() to return a successful response.
    
    Usage:
        mock_http_success(json_data={"result": "ok"}, headers={"X-ID": "123"}, latency_ms=15.5)
    """
    def _mock(json_data, headers=None, latency_ms=10.0):
        resp = fake_httpx_response(json_data, 200, headers)
        
        def mock_post(self, *, url, headers, payload):
            return resp, latency_ms
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", mock_post)
        return resp
    return _mock


@pytest.fixture
def mock_http_error(monkeypatch, fake_httpx_response):
    """
    Mock HTTPClient.post() to raise an HTTP error.
    
    Usage:
        # For timeout/network errors:
        mock_http_error(HTTPTimeoutError, "Request timed out")
        
        # For status errors:
        mock_http_error(HTTPStatusError, response=fake_response)
    """
    def _mock(error_class, *args, **kwargs):
        def mock_post(self, *, url, headers, payload):
            raise error_class(*args, **kwargs)
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", mock_post)
    return _mock


@pytest.fixture
def mock_http_status_error(monkeypatch, fake_httpx_response):
    """
    Mock HTTPClient.post() to raise HTTPStatusError with a given response.
    
    Usage:
        mock_http_status_error(json_data={"error": "..."}, status_code=400, headers={...})
    """
    def _mock(json_data, status_code=400, headers=None):
        from unifiedllm.errors import HTTPStatusError
        resp = fake_httpx_response(json_data, status_code, headers)
        
        def mock_post(self, *, url, headers, payload):
            raise HTTPStatusError(response=resp)
        
        from unifiedllm.http import HTTPClient
        monkeypatch.setattr(HTTPClient, "post", mock_post)
        return resp
    return _mock


# ============================================================================
# Fixtures for Mocking httpx.Client (for HTTPClient tests)
# ============================================================================

@pytest.fixture
def mock_httpx_client(monkeypatch):
    """
    Mock httpx.Client for testing HTTPClient itself.
    
    Usage:
        mock_httpx_client(response=fake_response, error=None)
    """
    def _mock(response=None, error=None):
        mock_client = Mock()
        
        if error:
            mock_client.post.side_effect = error
        else:
            mock_client.post.return_value = response
        
        def mock_init(self, timeout):
            self._client = mock_client
        
        import httpx
        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_client))
        return mock_client
    return _mock
