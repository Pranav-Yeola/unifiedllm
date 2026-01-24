from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any, Dict, Optional, Sequence, TypedDict
import httpx

from .enums import Role


class Message(TypedDict):
    role: Role | Literal["user","model"]
    content: str


@dataclass(frozen=True)
class HTTPResponse:
    data: dict[str, Any]
    status_code: int
    latency_ms: float
    headers: httpx.Headers


@dataclass(frozen=True)
class APIErrorDetails:
    """
    Provider-extracted error information from an HTTP error response.
    """

    message: Optional[str] = None
    error_type: Optional[str] = None
    code: Optional[str] = None
    raw: Optional[Any] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class ChatResponse:
    text: str
    raw: Optional[Any] = None
    usage: Optional[LLMUsage] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    request_id: Optional[str] = None


Params = Dict[str, Any]
Messages = Sequence[Message]
