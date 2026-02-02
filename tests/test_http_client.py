"""
Tests for the HTTPClient class (HTTP transport layer).
"""

import pytest
import httpx
from unittest.mock import Mock

from unifiedllm.http import HTTPClient
from unifiedllm.errors import HTTPTimeoutError, HTTPNetworkError, HTTPStatusError


class TestHTTPClientSuccess:
    """Tests for successful HTTP requests."""

    def test_post_returns_response_and_latency(self, fake_httpx_response, monkeypatch):
        """Test that post() returns both response and latency_ms."""
        json_data = {"result": "success"}
        fake_resp = fake_httpx_response(json_data, status_code=200)

        # Mock httpx.Client
        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)
        resp, latency_ms = client.post(
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer token"},
            payload={"message": "hello"},
        )

        assert resp == fake_resp
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0

    def test_post_latency_is_measured(self, fake_httpx_response, monkeypatch):
        """Test that latency is actually measured in milliseconds."""
        fake_resp = fake_httpx_response({"ok": True}, status_code=200)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)
        _, latency_ms = client.post(
            url="https://api.example.com/test", headers={}, payload={}
        )

        # Latency should be a small positive number (milliseconds)
        assert 0 <= latency_ms < 1000  # Should complete in under 1 second for mock

    def test_post_passes_correct_parameters(self, fake_httpx_response, monkeypatch):
        """Test that post() passes url, headers, and json payload correctly."""
        fake_resp = fake_httpx_response({"ok": True}, status_code=200)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)
        test_url = "https://api.example.com/endpoint"
        test_headers = {
            "Authorization": "Bearer token",
            "Content-Type": "application/json",
        }
        test_payload = {"key": "value", "number": 42}

        client.post(url=test_url, headers=test_headers, payload=test_payload)

        # Verify the httpx client was called with correct parameters
        mock_httpx_client.post.assert_called_once_with(
            test_url, headers=test_headers, json=test_payload
        )


class TestHTTPClientTimeoutError:
    """Tests for timeout errors."""

    def test_timeout_raises_http_timeout_error(self, monkeypatch):
        """Test that httpx.TimeoutException is converted to HTTPTimeoutError."""
        mock_httpx_client = Mock()
        mock_httpx_client.post.side_effect = httpx.TimeoutException(
            "Connection timeout"
        )

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        with pytest.raises(HTTPTimeoutError) as exc_info:
            client.post(url="https://api.example.com/test", headers={}, payload={})

        assert "timed out" in str(exc_info.value).lower()


class TestHTTPClientNetworkError:
    """Tests for network/connection errors."""

    def test_network_error_raises_http_network_error(self, monkeypatch):
        """Test that httpx.RequestError is converted to HTTPNetworkError."""
        mock_httpx_client = Mock()
        mock_httpx_client.post.side_effect = httpx.RequestError("Connection failed")

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        with pytest.raises(HTTPNetworkError) as exc_info:
            client.post(url="https://api.example.com/test", headers={}, payload={})

        assert "network" in str(exc_info.value).lower()


class TestHTTPClientStatusError:
    """Tests for HTTP status errors (4xx, 5xx)."""

    def test_status_400_raises_http_status_error(
        self, fake_httpx_response, monkeypatch
    ):
        """Test that 400 status code raises HTTPStatusError."""
        fake_resp = fake_httpx_response({"error": "bad request"}, status_code=400)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        with pytest.raises(HTTPStatusError) as exc_info:
            client.post(url="https://api.example.com/test", headers={}, payload={})

        assert exc_info.value.response == fake_resp
        assert "400" in str(exc_info.value)

    def test_status_401_raises_http_status_error(
        self, fake_httpx_response, monkeypatch
    ):
        """Test that 401 status code raises HTTPStatusError."""
        fake_resp = fake_httpx_response({"error": "unauthorized"}, status_code=401)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        with pytest.raises(HTTPStatusError) as exc_info:
            client.post(url="https://api.example.com/test", headers={}, payload={})

        assert exc_info.value.response.status_code == 401

    def test_status_500_raises_http_status_error(
        self, fake_httpx_response, monkeypatch
    ):
        """Test that 500 status code raises HTTPStatusError."""
        fake_resp = fake_httpx_response({"error": "server error"}, status_code=500)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        with pytest.raises(HTTPStatusError) as exc_info:
            client.post(url="https://api.example.com/test", headers={}, payload={})

        assert exc_info.value.response.status_code == 500
        assert "500" in str(exc_info.value)

    def test_status_200_does_not_raise(self, fake_httpx_response, monkeypatch):
        """Test that 200 status code does not raise an error."""
        fake_resp = fake_httpx_response({"success": True}, status_code=200)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        # Should not raise
        resp, _ = client.post(
            url="https://api.example.com/test", headers={}, payload={}
        )
        assert resp.status_code == 200

    def test_status_201_does_not_raise(self, fake_httpx_response, monkeypatch):
        """Test that 201 status code does not raise an error."""
        fake_resp = fake_httpx_response({"created": True}, status_code=201)

        mock_httpx_client = Mock()
        mock_httpx_client.post.return_value = fake_resp

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)

        # Should not raise
        resp, _ = client.post(
            url="https://api.example.com/test", headers={}, payload={}
        )
        assert resp.status_code == 201


class TestHTTPClientClose:
    """Tests for closing the HTTP client."""

    def test_close_calls_underlying_client_close(self, monkeypatch):
        """Test that close() calls the underlying httpx.Client.close()."""
        mock_httpx_client = Mock()

        monkeypatch.setattr(httpx, "Client", Mock(return_value=mock_httpx_client))

        client = HTTPClient(timeout=30.0)
        client.close()

        mock_httpx_client.close.assert_called_once()
