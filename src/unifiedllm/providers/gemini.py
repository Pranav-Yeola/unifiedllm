import os
from typing import Optional

from .base import BaseProvider
from ..types import LLMResponse, Messages, Params


class GeminiProvider(BaseProvider):
    name = "gemini"

    def default_api_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY")

    def chat(self, *, model: str, messages: Messages, params: Params) -> LLMResponse:
        # Stub implementation: replace with real Gemini SDK call next.
        last = messages[-1]["content"] if messages else ""
        return LLMResponse(
            text=f"[stub:{self.name}:{model}] {last}",
            raw={"messages": list(messages), "params": params},
            provider=self.name,
            model=model,
        )
