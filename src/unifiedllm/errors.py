from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import httpx

# HTTP Errors


class HTTPClientError(Exception):
    """Base transport-level error raised by HTTPClient"""

    pass


class HTTPTimeoutError(HTTPClientError):
    """Request timed out before receiving a response."""

    pass


class HTTPNetworkError(HTTPClientError):
    """Network / connection / DNS error."""

    pass


@dataclass
class HTTPStatusError(HTTPClientError):
    """HTTP response was received, but it indicated an error (4xx or 5xx)."""

    response: httpx.Response

    def __str__(self) -> str:
        return f"HTTP status error: {self.response.status_code}"


# Provider errors


@dataclass
class ProviderError(Exception):
    provider: str
    display_name: str
    model: Optional[str] = None
    status_code: Optional[int] = None
    request_id: Optional[str] = None
    raw: Optional[Any] = None

    detail: Optional[str] = None  # optional provider-specific detail
    suggestion: Optional[str] = None


@dataclass
class ProviderHTTPError(ProviderError):
    def __str__(self) -> str:
        s = f"Failed to connect to {self.display_name}"
        if self.model:
            s += f" ({self.model})"
        if self.detail:  # optional: timeout vs network details if you ever want
            s += f": {self.detail}"
        return s


@dataclass
class ProviderAPIError(ProviderError):
    error_type: Optional[str] = None
    code: Optional[str] = None

    def __str__(self) -> str:
        msg = f"{self.display_name} API request failed [HTTP {self.status_code}]"
        meta = []
        if self.error_type:
            meta.append(self.error_type)
        if self.code:
            meta.append(self.code)
        if meta:
            msg += f" ({' / '.join(meta)})"

        if self.detail:
            msg += f": {self.detail}"

        return msg


@dataclass
class MissingAPIKeyError(ProviderError):
    """API key is missing or not configured."""

    def __str__(self) -> str:
        msg = f"Missing API key for {self.display_name}"
        if self.model:
            msg += f" ({self.model})"
        if self.suggestion:
            msg += f". {self.suggestion}"
        return msg


@dataclass
class ProviderParseError(ProviderError):
    """HTTP response exists, but payload is invalid/unexpected (parse/shape)."""

    def __str__(self) -> str:
        s = f"Invalid response from {self.display_name}"
        if self.model:
            s += f" ({self.model})"
        if self.status_code is not None:
            s += f" [HTTP {self.status_code}]"
        if self.detail:
            s += f": {self.detail}"
        return s


class ProviderNotSupportedError(Exception):
    """Provider is not supported."""

    pass
