from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .base import BaseProvider
from ..errors import ProviderParseError
from ..types import ChatResponse, Messages, HTTPResponse, LLMUsage, APIErrorDetails


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"
    base_url = "https://api.openai.com"
    chat_endpoint = "/v1/chat/completions"
    env_key_name = "OPENAI_API_KEY"

    SUPPORTED_CUSTOM_KEYS = {
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "n",
        "user",
        "store",
        "service_tier",
        "response_format",
        "max_completion_tokens",
    }

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
            cfg["stop"] = stop

        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError("custom must be a dict[str, Any]")

            unknown = set(custom) - self.SUPPORTED_CUSTOM_KEYS
            if unknown:
                raise ValueError(
                    f"Unsupported OpenAI custom parameters: {sorted(unknown)}. Supported Parameters: {self.SUPPORTED_CUSTOM_KEYS}"
                )
            cfg.update(custom)

        self._config.update(cfg)

    def system_prompt(self, text: str) -> None:
        self._system_prompt = text

    def chat(self, *, messages: Messages) -> ChatResponse:
        openai_messages = self._convert_messages(messages)

        if self._system_prompt is not None:
            openai_messages.insert(
                0, {"role": "system", "content": self._system_prompt}
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        if self._config:
            payload.update(self._config)

        http_resp = self._get_response(
            url=self._url, headers=self._headers, payload=payload
        )
        return self._parse_chat_response(http_resp)

    def _extract_error_details(self, resp: httpx.Response) -> APIErrorDetails:
        request_id = resp.headers.get("x-request-id") or resp.headers.get("request-id")

        try:
            payload = resp.json()
        except Exception:
            return APIErrorDetails(
                message=resp.text, raw=resp.text, request_id=request_id
            )

        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            err: dict[str, Any] = payload["error"]
            return APIErrorDetails(
                message=err.get("message") or resp.text,
                error_type=err.get("type"),
                code=err.get("code"),
                raw=payload,
                request_id=request_id,
            )

        return APIErrorDetails(
            message=payload.get("message") if isinstance(payload, dict) else resp.text,
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

    def _extract_request_id(self, http_resp: HTTPResponse) -> Optional[str]:
        rid: str = http_resp.headers.get("x-request-id") or http_resp.headers.get("request-id")
        return rid.strip() if rid else None

    def _extract_text(self, data: Dict[str, Any]) -> str:
        choices: List[Dict] = data.get("choices")
        if not isinstance(choices, list):
            raise ProviderParseError(
                provider=self.name,
                display_name=self.display_name,
                model=self.model,
                detail="Unexpected response shape: 'choices' {}.".format(
                    "missing" if choices is None else "not a list"
                ),
                raw=data,
            )

        if not choices:  # valid shape but empty -> treat as no text
            return ""

        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        text = content if isinstance(content, str) else ""

        return text

    def _extract_usage(self, data: Dict[str, Any]) -> Optional[LLMUsage]:
        usage_obj = data.get("usage")
        if not isinstance(usage_obj, dict):
            return None

        return LLMUsage(
            prompt_tokens=usage_obj.get("prompt_tokens"),
            completion_tokens=usage_obj.get("completion_tokens"),
            total_tokens=usage_obj.get("total_tokens"),
        )
