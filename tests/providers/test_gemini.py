"""
Tests for the Gemini (Google) provider.
"""

import pytest

from unifiedllm.providers.google import GeminiProvider
from unifiedllm.types import ChatResponse
from unifiedllm.errors import (
    ProviderHTTPError,
    ProviderAPIError,
    ProviderParseError,
    MissingAPIKeyError,
    HTTPTimeoutError,
    HTTPNetworkError,
)
from conftest import GEMINI_SUCCESS_RESPONSE, GEMINI_ERROR_RESPONSE


class TestGeminiProviderSuccess:
    """Tests for successful Gemini API responses."""

    def test_success_response_with_all_fields(self, mock_http_success):
        """Test successful response with all fields populated."""
        mock_http_success(
            json_data=GEMINI_SUCCESS_RESPONSE,
            headers={},  # Gemini uses body for request_id, not headers
            latency_ms=52.3,
        )

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, ChatResponse)
        assert response.text == "Hello! I'm here to help."
        assert response.provider == "gemini"
        assert response.model == "gemini-pro"
        assert response.status_code == 200
        assert response.latency_ms == 52.3
        assert response.request_id == "resp_xyz789"  # From body, not headers!
        assert response.raw is not None
        assert response.raw == GEMINI_SUCCESS_RESPONSE

        # Check usage
        assert response.usage is not None
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 18
        assert response.usage.total_tokens == 30

    def test_request_id_from_body_not_headers(self, mock_http_success):
        """Test that request_id comes from body responseId, not headers."""
        # Include header that should be ignored
        headers = {"x-request-id": "should-be-ignored"}
        response_data = {
            "responseId": "body-request-id-123",
            "candidates": [
                {"content": {"parts": [{"text": "Response"}], "role": "model"}}
            ],
        }
        mock_http_success(json_data=response_data, headers=headers)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        # Should use body responseId, not header
        assert response.request_id == "body-request-id-123"

    def test_success_with_multiple_parts(self, mock_http_success):
        """Test response with multiple text parts."""
        response_data = {
            "responseId": "resp-multi",
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Part 1. "},
                            {"text": "Part 2. "},
                            {"text": "Part 3."},
                        ],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        }
        mock_http_success(json_data=response_data)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == "Part 1. Part 2. Part 3."

    def test_success_with_no_candidates(self, mock_http_success):
        """Test response with no candidates returns empty text."""
        response_data = {"responseId": "resp-empty", "candidates": []}
        mock_http_success(json_data=response_data)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == ""

    def test_success_with_missing_candidates(self, mock_http_success):
        """Test response with missing candidates field returns empty text."""
        response_data = {"responseId": "resp-no-candidates"}
        mock_http_success(json_data=response_data)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == ""

    def test_success_without_usage(self, mock_http_success):
        """Test response without usageMetadata field."""
        response_data = {
            "responseId": "resp-no-usage",
            "candidates": [
                {"content": {"parts": [{"text": "Response"}], "role": "model"}}
            ],
        }
        mock_http_success(json_data=response_data)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert response.text == "Response"
        assert response.usage is None

    def test_success_with_system_prompt(self, mock_http_success, monkeypatch):
        """Test that system prompt is added to payload as systemInstruction."""
        mock_http_success(json_data=GEMINI_SUCCESS_RESPONSE)

        # Track the payload sent
        sent_payload = {}

        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock

            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = GEMINI_SUCCESS_RESPONSE
            resp.text = str(GEMINI_SUCCESS_RESPONSE)
            import httpx

            resp.headers = httpx.Headers({})
            return resp, 10.0

        from unifiedllm.http import HTTPClient

        monkeypatch.setattr(HTTPClient, "post", capture_post)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        provider.system_prompt("You are a helpful assistant")
        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        # System prompt should be in systemInstruction
        assert "systemInstruction" in sent_payload
        assert sent_payload["systemInstruction"] == {
            "parts": [{"text": "You are a helpful assistant"}]
        }


class TestGeminiProviderAPIErrors:
    """Tests for Gemini API error responses."""

    def test_api_error_with_full_details(self, mock_http_status_error):
        """Test API error with code, status, and message."""
        mock_http_status_error(
            json_data=GEMINI_ERROR_RESPONSE, status_code=400, headers={}
        )

        provider = GeminiProvider(model="gemini-pro", api_key="bad-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "gemini"
        assert error.display_name == "Gemini"
        assert error.model == "gemini-pro"
        assert error.status_code == 400
        assert error.error_type == "INVALID_ARGUMENT"  # From error.status
        assert error.code == "400"  # From error.code, converted to string
        assert "API key not valid" in error.detail
        assert error.raw == GEMINI_ERROR_RESPONSE

    def test_api_error_status_401(self, mock_http_status_error):
        """Test authentication error (401)."""
        error_response = {
            "error": {
                "code": 401,
                "message": "Request is missing required authentication credential",
                "status": "UNAUTHENTICATED",
            }
        }
        mock_http_status_error(json_data=error_response, status_code=401)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_type == "UNAUTHENTICATED"
        assert exc_info.value.code == "401"

    def test_api_error_status_429(self, mock_http_status_error):
        """Test rate limit error (429)."""
        error_response = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
        mock_http_status_error(json_data=error_response, status_code=429)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.status_code == 429
        assert exc_info.value.error_type == "RESOURCE_EXHAUSTED"

    def test_api_error_without_structured_error(self, mock_http_status_error):
        """Test error response without standard error structure."""
        error_response = {"message": "Something went wrong"}
        mock_http_status_error(json_data=error_response, status_code=500)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderAPIError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Something went wrong"


class TestGeminiProviderTransportErrors:
    """Tests for network/timeout errors."""

    def test_timeout_error_raises_provider_http_error(self, mock_http_error):
        """Test that timeout is converted to ProviderHTTPError."""
        mock_http_error(HTTPTimeoutError, "Request timed out")

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "gemini"
        assert error.display_name == "Gemini"
        assert error.model == "gemini-pro"
        assert "Gemini" in str(error)

    def test_network_error_raises_provider_http_error(self, mock_http_error):
        """Test that network error is converted to ProviderHTTPError."""
        mock_http_error(HTTPNetworkError, "Network error")

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderHTTPError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        assert exc_info.value.provider == "gemini"


class TestGeminiProviderParseErrors:
    """Tests for response parsing errors."""

    def test_candidates_not_a_list_raises_parse_error(self, mock_http_success):
        """Test that 'candidates' not being a list raises ProviderParseError."""
        mock_http_success(json_data={"candidates": "not-a-list"})

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ProviderParseError) as exc_info:
            provider.chat(messages=[{"role": "user", "content": "Test"}])

        error = exc_info.value
        assert error.provider == "gemini"
        assert "candidates" in error.detail.lower()
        assert "not a list" in error.detail.lower()


class TestGeminiProviderMissingAPIKey:
    """Tests for missing API key."""

    def test_empty_api_key_raises_error(self, monkeypatch):
        """Test that empty API key raises MissingAPIKeyError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError) as exc_info:
            GeminiProvider(model="gemini-pro", api_key="", timeout=30.0)

        error = exc_info.value
        assert error.provider == "gemini"
        assert error.display_name == "Gemini"
        assert error.model == "gemini-pro"
        assert "GEMINI_API_KEY" in str(error)

    def test_default_api_key_without_env_raises_error(self, monkeypatch):
        """Test that api_key='default' without env var raises error."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            GeminiProvider(model="gemini-pro", api_key="default", timeout=30.0)


class TestGeminiProviderRoleValidation:
    """Tests for message role validation."""

    def test_invalid_role_system_raises_error(self):
        """Test that 'system' role in messages raises ValueError."""
        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "system", "content": "System prompt"}])

        assert "unsupported role" in str(exc_info.value).lower()
        assert "system" in str(exc_info.value).lower()

    def test_invalid_role_assistant_raises_error(self):
        """Test that 'assistant' role raises ValueError (Gemini uses 'model')."""
        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "assistant", "content": "Test"}])

        assert "unsupported role" in str(exc_info.value).lower()

    def test_invalid_role_custom_raises_error(self):
        """Test that custom role raises ValueError."""
        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.chat(messages=[{"role": "banana", "content": "Test"}])

        assert "unsupported role" in str(exc_info.value).lower()
        assert "banana" in str(exc_info.value).lower()

    def test_valid_role_user(self, mock_http_success):
        """Test that 'user' role is valid."""
        mock_http_success(json_data=GEMINI_SUCCESS_RESPONSE)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response.text is not None

    def test_valid_role_model(self, mock_http_success):
        """Test that 'model' role is valid (Gemini keeps it as 'model')."""
        mock_http_success(json_data=GEMINI_SUCCESS_RESPONSE)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        # Should not raise - 'model' stays as 'model' for Gemini
        response = provider.chat(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi there"},
            ]
        )
        assert response.text is not None

    def test_role_mapping_preserves_model(self, mock_http_success, monkeypatch):
        """Test that Gemini preserves 'model' role (doesn't convert to 'assistant')."""
        sent_payload = {}

        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock

            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = GEMINI_SUCCESS_RESPONSE
            resp.text = str(GEMINI_SUCCESS_RESPONSE)
            import httpx

            resp.headers = httpx.Headers({})
            return resp, 10.0

        from unifiedllm.http import HTTPClient

        monkeypatch.setattr(HTTPClient, "post", capture_post)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        provider.chat(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi"},
            ]
        )

        # Check that roles are preserved in Gemini format
        contents = sent_payload["contents"]
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"  # Not 'assistant'!


class TestGeminiProviderConfigValidation:
    """Tests for configuration parameter validation."""

    def test_unsupported_custom_parameter_raises_error(self):
        """Test that unsupported custom parameter raises ValueError."""
        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        with pytest.raises(ValueError) as exc_info:
            provider.config(custom={"invalid_param": "value", "bad_param": 123})

        error_msg = str(exc_info.value).lower()
        assert "unsupported" in error_msg
        assert "gemini" in error_msg

    def test_supported_custom_parameters_work(self, mock_http_success):
        """Test that supported custom parameters don't raise errors."""
        mock_http_success(json_data=GEMINI_SUCCESS_RESPONSE)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)

        # These are in SUPPORTED_CUSTOM_KEYS
        provider.config(custom={"topK": 40, "candidateCount": 1})

        # Should not raise
        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None

    def test_config_parameters_mapped_correctly(self, mock_http_success, monkeypatch):
        """Test that config parameters are mapped to Gemini's naming."""
        sent_payload = {}

        def capture_post(self, *, url, headers, payload):
            sent_payload.update(payload)
            from unittest.mock import Mock

            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = GEMINI_SUCCESS_RESPONSE
            resp.text = str(GEMINI_SUCCESS_RESPONSE)
            import httpx

            resp.headers = httpx.Headers({})
            return resp, 10.0

        from unifiedllm.http import HTTPClient

        monkeypatch.setattr(HTTPClient, "post", capture_post)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        provider.config(
            temperature=0.7, top_p=0.9, max_tokens=500, stop=["STOP", "END"]
        )
        provider.chat(messages=[{"role": "user", "content": "Test"}])

        # Check generationConfig mapping
        gen_config = sent_payload.get("generationConfig", {})
        assert gen_config["temperature"] == 0.7
        assert gen_config["topP"] == 0.9  # Mapped from top_p
        assert gen_config["maxOutputTokens"] == 500  # Mapped from max_tokens
        assert gen_config["stopSequences"] == ["STOP", "END"]  # Mapped from stop

    def test_standard_config_parameters(self, mock_http_success):
        """Test that standard config parameters work."""
        mock_http_success(json_data=GEMINI_SUCCESS_RESPONSE)

        provider = GeminiProvider(model="gemini-pro", api_key="test-key", timeout=30.0)
        provider.config(temperature=0.8, top_p=0.95, max_tokens=1000)

        response = provider.chat(messages=[{"role": "user", "content": "Test"}])
        assert response is not None


class TestGeminiProviderURLGeneration:
    """Tests for URL generation with model parameter."""

    def test_model_name_in_endpoint(self, mock_http_success, monkeypatch):
        """Test that model name is included in the endpoint URL."""
        sent_url = []

        def capture_post(self, *, url, headers, payload):
            sent_url.append(url)
            from unittest.mock import Mock

            resp = Mock()
            resp.status_code = 200
            resp.json.return_value = GEMINI_SUCCESS_RESPONSE
            resp.text = str(GEMINI_SUCCESS_RESPONSE)
            import httpx

            resp.headers = httpx.Headers({})
            return resp, 10.0

        from unifiedllm.http import HTTPClient

        monkeypatch.setattr(HTTPClient, "post", capture_post)

        provider = GeminiProvider(
            model="gemini-1.5-pro", api_key="test-key", timeout=30.0
        )
        provider.chat(messages=[{"role": "user", "content": "Test"}])

        # URL should contain model name
        assert len(sent_url) == 1
        assert "gemini-1.5-pro" in sent_url[0]
        assert "generateContent" in sent_url[0]
