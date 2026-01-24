from __future__ import annotations

from typing import Any, Optional

from .errors import ProviderNotSupportedError
from .types import ChatResponse, Messages

from .providers import BaseProvider, AnthropicProvider, GeminiProvider, OpenAIProvider


class LLM:
    """
    Gateway to a specific provider + model.

    - Owns the underlying httpx.Client (connection pool).
    - Exposes close() and supports context manager usage.
    """

    PROVIDERS = {
        GeminiProvider.name: GeminiProvider,
        OpenAIProvider.name: OpenAIProvider,
        AnthropicProvider.name: AnthropicProvider,
    }

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str = "default",
        timeout: float = 60.0,
    ) -> None:
        self.provider_name = provider.lower().strip()
        self.model = model
        self.timeout = timeout

        if self.provider_name not in LLM.PROVIDERS:
            raise ProviderNotSupportedError(
                f"Provider '{provider}' not supported. Supported: {sorted(self.PROVIDERS.keys())}"
            )

        self._provider: BaseProvider = LLM.PROVIDERS[self.provider_name](
            model=self.model, api_key=api_key, timeout=self.timeout
        )

    def config(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        custom: dict[str, Any] | None = None,
    ) -> "LLM":
        self._provider.config(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            custom=custom,
        )
        return self

    def system_prompt(self, text: str) -> "LLM":
        self._provider.system_prompt(text)
        return self

    def chat(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
    ) -> ChatResponse:
        """
        Provide either prompt=... OR messages=[{"role": ..., "content": ...}].
        """
        if (prompt is None) == (messages is None):
            raise ValueError("Provide exactly one of: prompt=... or messages=[...]")

        normalized: Messages
        if messages is not None:
            normalized = list(messages)
        else:
            normalized = [{"role": "user", "content": str(prompt)}]

        return self._provider.chat(messages=normalized)

    def close(self) -> None:
        """Close underlying HTTP resources."""
        self._provider.close()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
