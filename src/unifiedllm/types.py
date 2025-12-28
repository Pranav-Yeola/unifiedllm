from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Optional[LLMUsage] = None
    raw: Optional[Any] = None
    provider: Optional[str] = None
    model: Optional[str] = None


Params = Dict[str, Any]
Messages = Sequence[Message]
