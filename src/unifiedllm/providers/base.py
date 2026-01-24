from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import httpx


from ..http import HTTPClient
from ..types import ChatResponse, Messages, HTTPResponse, APIErrorDetails, LLMUsage
from ..errors import (
    ProviderAPIError,
    ProviderHTTPError,
    ProviderParseError,
    MissingAPIKeyError,
    HTTPTimeoutError,
    HTTPNetworkError,
    HTTPStatusError,
)

# from ..enums import HTTPErrorCode  # adjust import path if needed


class BaseProvider(ABC):
    """
    Base class for providers.
    """

    name: str | None = None
    display_name: str | None = None
    base_url: str | None = None
    chat_endpoint: str | None = None
    env_key_name: str | None = None

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float,
    ) -> None:
        self._check_class_vars()
        self.model = model
        self.timeout = timeout
        self._system_prompt: Optional[str] = None
        self._config: dict[str, Any] = {}
        self._http = HTTPClient(timeout=self.timeout)
        self.api_key = self.load_env_api_key() if api_key == "default" else api_key

        if not self.api_key:
            raise MissingAPIKeyError(
                provider=self.name,
                display_name=self.display_name,
                model=self.model,
                suggestion=f"Set the {self.env_key_name} environment variable or pass api_key explicitly.",
            )

    def _check_class_vars(self) -> None:
        for attr in ("name", "base_url", "chat_endpoint", "env_key_name"):
            if getattr(self, attr, None) is None:
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define class attribute '{attr}'"
                )

    def _get_chat_url(self, **path_params) -> str:
        try:
            endpoint = self.chat_endpoint.format(**path_params)
        except KeyError as e:
            raise ValueError(
                f"Missing path parameter '{e.args[0]}' for provider '{self.name}'"
            ) from None

        base = str(self.base_url).rstrip("/")
        endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        return base + endpoint

    def close(self) -> None:
        self._http.close()

    @classmethod
    def load_env_api_key(cls) -> Optional[str]:
        return os.getenv(cls.env_key_name)

    def _get_response(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> HTTPResponse:
        provider = str(self.name or self.__class__.__name__)

        # API connection
        try:
            resp, latency_ms = self._http.post(
                url=url, headers=headers, payload=payload
            )
        except (HTTPTimeoutError, HTTPNetworkError) as e:
            raise ProviderHTTPError(
                provider=provider,
                display_name=self.display_name,
                model=self.model,
            ) from e
        except HTTPStatusError as e:
            error_details = self._extract_error_details(e.response)
            raise ProviderAPIError(
                provider=provider,
                display_name=self.display_name,
                model=self.model,
                status_code=e.response.status_code,
                error_type=error_details.error_type,
                code=error_details.code,
                request_id=error_details.request_id,
                detail=error_details.message,
                raw=error_details.raw,
            ) from e

        # Parse JSON on success
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderParseError(
                provider=provider,
                display_name=self.display_name,
                model=self.model,
                status_code=resp.status_code,
                detail="Response was not valid JSON.",
                raw=resp.text,
                suggestion="Try again. If this persists, the provider may be having an outage.",
            ) from e

        if not isinstance(data, dict):
            raise ProviderParseError(
                provider=provider,
                display_name=self.display_name,
                model=self.model,
                status_code=resp.status_code,
                detail="Unexpected JSON shape (expected object).",
                raw=data,
                suggestion="Try again. If this persists, report the raw response.",
            )

        return HTTPResponse(
            data=data,
            status_code=resp.status_code,
            latency_ms=latency_ms,
            headers=resp.headers,
        )

    def _parse_chat_response(self, http_resp: HTTPResponse) -> ChatResponse:
        data = http_resp.data

        text = self._extract_text(data)
        usage = self._extract_usage(data)
        request_id = self._extract_request_id(http_resp)

        return ChatResponse(
            text=text,
            raw=data,
            usage=usage,
            provider=self.name,
            model=self.model,
            status_code=http_resp.status_code,
            latency_ms=http_resp.latency_ms,
            request_id=request_id,
        )

    @abstractmethod
    def config(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        custom: Optional[dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def system_prompt(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def chat(self, *, messages: Messages) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    def _extract_error_details(self, resp: httpx.Response) -> APIErrorDetails:
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, data: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _extract_usage(self, data: dict[str, Any]) -> Optional[LLMUsage]:
        raise NotImplementedError

    @abstractmethod
    def _extract_request_id(self, http_resp: HTTPResponse) -> Optional[str]:
        raise NotImplementedError
