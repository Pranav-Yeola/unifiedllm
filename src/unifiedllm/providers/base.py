from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import LLMResponse, Messages, Params


class BaseProvider(ABC):
    """
    Provider adapter interface.

    Providers should:
    - Read api_key from env var (or accept explicit override)
    - Translate unifiedllm Messages -> provider format
    - Call provider SDK/API
    - Return LLMResponse
    """

    name: str

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
    ) -> None:
        self.api_key = api_key or self.default_api_key()
        self.base_url = base_url
        self.timeout = timeout

    @abstractmethod
    def default_api_key(self) -> Optional[str]:
        """Return API key from environment for this provider."""
        raise NotImplementedError

    @abstractmethod
    def chat(self, *, model: str, messages: Messages, params: Params) -> LLMResponse:
        """Perform a chat request and return a normalized response."""
        raise NotImplementedError
