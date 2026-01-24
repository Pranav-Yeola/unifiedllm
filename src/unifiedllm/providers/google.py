from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .base import BaseProvider
from ..errors import ProviderParseError
from ..types import Messages, HTTPResponse, LLMUsage, APIErrorDetails


class GeminiProvider(BaseProvider):
    name = "gemini"
    display_name = "Gemini"
    base_url = "https://generativelanguage.googleapis.com"
    chat_endpoint = "/v1beta/models/{model}:generateContent"
    env_key_name = "GEMINI_API_KEY"

    SUPPORTED_CUSTOM_KEYS = {
        "topK",
        "candidateCount",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "default",
        timeout: float,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._url = self._get_chat_url(model=self.model)
        self._headers = self._get_headers()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
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

        # generationConfig fields (Gemini REST)
        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["topP"] = top_p
        if max_tokens is not None:
            cfg["maxOutputTokens"] = max_tokens
        if stop is not None:
            cfg["stopSequences"] = stop

        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError("custom must be a dict[str, Any]")

            unknown = set(custom) - self.SUPPORTED_CUSTOM_KEYS
            if unknown:
                raise ValueError(
                    f"Unsupported Gemini custom parameters: {sorted(unknown)}. Supported Parameters: {self.SUPPORTED_CUSTOM_KEYS}"
                )
            cfg.update(custom)

        self._config.update(cfg)

    def system_prompt(self, text: str) -> None:
        self._system_prompt = text

    def chat(self, *, messages: Messages):
        payload: Dict[str, Any] = {
            "contents": self._convert_messages(messages),
        }

        if self._system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": self._system_prompt}]}

        if self._config:
            payload["generationConfig"] = self._config

        http_resp = self._get_response(
            url=self._url, headers=self._headers, payload=payload
        )
        return self._parse_chat_response(http_resp)

    def _extract_request_id(self, http_resp: HTTPResponse) -> Optional[str]:
        rid = http_resp.data.get("responseId")
        return rid.strip() if isinstance(rid, str) and rid.strip() else None

    def _extract_text(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates")

        if candidates is None:
            return ""

        if not isinstance(candidates, list):
            raise ProviderParseError(
                provider=self.name,
                display_name=self.display_name,
                model=self.model,
                detail="Unexpected response shape: 'candidates' missing or not a list.",
                raw=data,
            )

        if not candidates:
            return ""

        c0 = candidates[0]
        if not isinstance(c0, dict):
            return ""

        content = c0.get("content")
        if not isinstance(content, dict):
            return ""

        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""

        texts: List[str] = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    texts.append(t)

        return "".join(texts)

    def _extract_usage(self, data: Dict[str, Any]) -> Optional[LLMUsage]:
        # Gemini REST returns usageMetadata with token counts
        usage = data.get("usageMetadata")
        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _extract_error_details(self, resp: httpx.Response) -> APIErrorDetails:
        # best-effort request id from headers
        request_id = (
            resp.headers.get("x-goog-request-id")
            or resp.headers.get("x-request-id")
            or resp.headers.get("request-id")
        )

        try:
            payload = resp.json()
        except Exception:
            return APIErrorDetails(
                message=resp.text,
                raw=resp.text,
                request_id=request_id,
            )

        # Google-style error shape: {"error": {"code": <int>, "message": <str>, "status": <str>, ...}}
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or resp.text
                status = err.get("status")
                code = err.get("code")

                return APIErrorDetails(
                    message=msg if isinstance(msg, str) else resp.text,
                    error_type=status if isinstance(status, str) else None,
                    code=str(code) if code is not None else None,
                    raw=payload,
                    request_id=request_id,
                )

            # fallback
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
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            if role not in {"user", "model"}:
                raise ValueError(f"Unsupported role '{role}'. Use 'user' or 'model'.")

            content = str(m.get("content") or "")
            contents.append({"role": role, "parts": [{"text": content}]})

        return contents
