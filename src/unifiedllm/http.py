from __future__ import annotations

import time
from typing import Any

import httpx
from .errors import HTTPTimeoutError, HTTPNetworkError, HTTPStatusError


class HTTPClient:
    def __init__(self, *, timeout: float) -> None:
        self._client = httpx.Client(timeout=timeout)

    def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[httpx.Response, float]:
        try:
            t0 = time.perf_counter()
            resp = self._client.post(url, headers=headers, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except httpx.TimeoutException as e:
            raise HTTPTimeoutError("Request timed out") from e
        except httpx.RequestError as e:
            raise HTTPNetworkError("Network error") from e

        if resp.status_code >= 400:
            raise HTTPStatusError(response=resp)

        return resp, latency_ms

    def close(self) -> None:
        self._client.close()
