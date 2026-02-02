from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .base import BaseProvider
from ..errors import ProviderParseError
from ..types import ChatResponse, Messages, HTTPResponse, LLMUsage, APIErrorDetails


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    display_name = "Anthropic"
    base_url = "https://api.anthropic.com"
    chat_endpoint = "/v1/messages"
    env_key_name = "ANTHROPIC_API_KEY"

    SUPPORTED_CUSTOM_KEYS = {
        "top_k",
        "stop_sequences",
        "metadata",
        "stream",
        "user_id",
    }

    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "default",
        timeout: float,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._url = self._get_chat_url()
        self._headers = self._get_headers()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def config(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        custom: Optional[dict[str, Any]] = None,
    ) -> None:
        cfg: Dict[str, Any] = {}

        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["top_p"] = top_p
        if max_tokens is not None:
            cfg["max_tokens"] = max_tokens
        if stop is not None:
            cfg["stop_sequences"] = stop

        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError("custom must be a dict[str, Any]")

            unknown = set(custom) - self.SUPPORTED_CUSTOM_KEYS
            if unknown:
                raise ValueError(
                    f"Unsupported Anthropic custom parameters: {sorted(unknown)}. Supported Parameters: {self.SUPPORTED_CUSTOM_KEYS}"
                )
            cfg.update(custom)

        self._config.update(cfg)

    def system_prompt(self, text: str) -> None:
        self._system_prompt = text

    def chat(self, *, messages: Messages) -> ChatResponse:
        anthropic_messages = self._convert_messages(messages)

        cfg = dict(self._config)
        cfg.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            **cfg,
        }

        if self._system_prompt is not None:
            payload["system"] = self._system_prompt

        http_resp = self._get_response(
            url=self._url, headers=self._headers, payload=payload
        )
        return self._parse_chat_response(http_resp)

    def _extract_request_id(self, http_resp: HTTPResponse) -> Optional[str]:
        rid: str = http_resp.headers.get("request-id") or http_resp.headers.get(
            "x-request-id"
        )
        return rid.strip() if rid else None

    def _extract_text(self, data: Dict[str, Any]) -> str:
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderParseError(
                provider=self.name,
                display_name=self.display_name,
                model=self.model,
                detail="Unexpected response shape: 'content' missing or not a list.",
                raw=data,
            )

        if not blocks:
            return ""

        parts: List[str] = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    parts.append(t)

        return "".join(parts)

    def _extract_usage(self, data: Dict[str, Any]) -> Optional[LLMUsage]:
        usage_obj = data.get("usage")
        if not isinstance(usage_obj, dict):
            return None

        prompt_tokens = usage_obj.get("input_tokens")
        completion_tokens = usage_obj.get("output_tokens")

        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _extract_error_details(self, resp: httpx.Response) -> APIErrorDetails:
        request_id = resp.headers.get("request-id") or resp.headers.get("x-request-id")

        try:
            payload = resp.json()
        except Exception:
            return APIErrorDetails(
                message=resp.text,
                raw=resp.text,
                request_id=request_id,
            )

        if isinstance(payload, dict):
            rid2 = payload.get("request_id")
            if isinstance(rid2, str) and not request_id:
                request_id = rid2

            err = payload.get("error")
            if isinstance(err, dict):
                return APIErrorDetails(
                    message=err.get("message") or resp.text,
                    error_type=err.get("type"),
                    raw=payload,
                    request_id=request_id,
                )

            msg = payload.get("message")
            return APIErrorDetails(
                message=msg if isinstance(msg, str) else resp.text,
                raw=payload,
                request_id=request_id,
            )

        return APIErrorDetails(
            message=resp.text,
            raw=payload,
            request_id=request_id,
        )

    def _convert_messages(self, messages: Messages) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            if role not in {"user", "model"}:
                raise ValueError(f"Unsupported role '{role}'. Use 'user' or 'model'.")

            content = str(m.get("content") or "")
            out.append(
                {"role": "user" if role == "user" else "assistant", "content": content}
            )
        return out
