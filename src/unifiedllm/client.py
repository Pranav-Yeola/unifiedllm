from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .errors import MissingAPIKeyError, ProviderNotSupportedError
from .types import LLMResponse, Message, Messages, Params
from .providers import BaseProvider, GeminiProvider, OpenAIProvider


class LLMClient:
    """
    Public, provider-agnostic API.

    v1 supports:
    - chat(prompt=...)
    - chat(messages=[...])
    """

    ProvidersClient: Dict[str, Type[BaseProvider]] = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
    }

    def __init__(
        self,
        *, 
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        
        self.provider = provider.strip().lower()
        self.model = model.strip()
        self.timeout = timeout
        self.default_params: Dict[str, Any] = default_params or {}

        if not self.provider:
            raise ValueError("provider must be a non-empty string")
        if not self.model:
            raise ValueError("model must be a non-empty string")

        provider_cls = self.ProvidersClient.get(self.provider)
        if provider_cls is None:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider}' is not supported. Supported: {sorted(self.ProvidersClient.keys())}"
            )

        self._backend = provider_cls(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

        if not self._backend.api_key:
            raise MissingAPIKeyError(
                f"Missing API key for provider '{self.provider}'. "
                f"Set the provider env var or pass api_key explicitly."
            )

    def chat(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        system: Optional[str] = None,
        **params: Any,
    ) -> LLMResponse:
        normalized = self._normalize_messages(
            prompt=prompt, messages=messages, system=system
        )

        merged: Params = dict(self.default_params)
        merged.update(params)

        return self._backend.chat(model=self.model, messages=normalized, params=merged)

    @staticmethod
    def _normalize_messages(
        *,
        prompt: Optional[str],
        messages: Optional[List[Message]],
        system: Optional[str],
    ) -> Messages:
        if prompt is None and not messages:
            raise ValueError("Provide either prompt=... or messages=[...]")

        if prompt is not None and messages:
            raise ValueError(
                "Provide only one of prompt=... or messages=[...], not both"
            )

        out: List[Message] = []

        if system:
            out.append({"role": "system", "content": system})

        if prompt is not None:
            out.append({"role": "user", "content": prompt})
            return out

        assert messages is not None
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list of {role, content}")

        for i, m in enumerate(messages):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not role:
                raise ValueError(f"messages[{i}].role must be non-empty")
            if content == "":
                raise ValueError(f"messages[{i}].content must be non-empty")
            out.append({"role": role, "content": content})

        return out
