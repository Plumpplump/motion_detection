"""OpenAI-compatible vision backend.

One adapter covers OpenRouter, Ollama's /v1 endpoint, LM Studio, vLLM,
llama.cpp server and any other provider that speaks the chat-completions
protocol with image content parts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .base import VisionBackend
from .image import prepare_jpeg, to_data_url
from .parsing import parse_detection_reply
from .prompt import build_messages
from .types import DetectionResult, VisionError


class OpenAICompatibleBackend(VisionBackend):
    key = "remote"
    label = "Remote model"
    local = False

    def __init__(
        self,
        *,
        key: str,
        label: str,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 45.0,
        requires_key: bool = False,
        extra_headers: dict[str, str] | None = None,
        json_mode: bool = True,
        local: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self.local = local
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.requires_key = requires_key
        self.extra_headers = extra_headers or {}
        self.json_mode = json_mode
        self._client = client
        self._owns_client = client is None

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.model and (self.api_key or not self.requires_key))

    async def prepare(self) -> None:
        if not self.base_url:
            raise VisionError(
                "No base URL configured for this backend.",
                kind="config",
            )
        if not self.model:
            raise VisionError(
                "No model name configured for this backend.",
                kind="config",
            )
        if self.requires_key and not self.api_key:
            raise VisionError(
                "This provider needs an API key. Add it in AI settings.",
                kind="auth",
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                **self.extra_headers,
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                follow_redirects=True,
            )
        return self._client

    async def check(
        self,
        image_jpeg: bytes,
        target: str,
        *,
        threshold: float = 0.35,
    ) -> DetectionResult:
        del threshold  # remote models express confidence in their own way
        started = time.perf_counter()
        prepared = await asyncio.to_thread(prepare_jpeg, image_jpeg, 768, 85)
        messages = build_messages(target, to_data_url(prepared))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": False,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._post(payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise VisionError(
                "The provider returned a non-JSON HTTP response.",
                kind="response",
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise VisionError(
                "The provider returned no choices. Check the model name.",
                kind="response",
            )
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = str(content or "")

        present, confidence, reason, raw = parse_detection_reply(text, target)
        return DetectionResult(
            present=present,
            label=target,
            confidence=confidence,
            reason=reason,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend=self.key,
            target=target,
        )

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        client = await self._get_client()
        url = f"{self.base_url}/chat/completions"

        for attempt in range(2):
            try:
                response = await client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                raise VisionError(
                    f"The AI model took longer than {self.timeout:.0f}s to answer.",
                    kind="timeout",
                ) from exc
            except httpx.ConnectError as exc:
                raise VisionError(
                    f"Could not connect to {self.base_url}. "
                    "Is the local server running?",
                    kind="network",
                ) from exc
            except httpx.HTTPError as exc:
                raise VisionError(
                    f"Network error while calling the model: {exc}",
                    kind="network",
                ) from exc

            if (
                response.status_code in (400, 422)
                and "response_format" in payload
                and attempt == 0
            ):
                # Some OpenAI-compatible servers do not implement JSON mode.
                payload.pop("response_format", None)
                continue

            if response.status_code >= 400:
                self._raise_status(response)
            return response

        raise VisionError("The provider rejected the request.", kind="http")

    def _raise_status(self, response: httpx.Response) -> None:
        status = response.status_code
        detail = ""
        try:
            body = response.json()
            detail = (
                body.get("error", {}).get("message")
                if isinstance(body, dict) and isinstance(body.get("error"), dict)
                else str(body)[:300]
            )
        except Exception:
            detail = response.text[:300]

        messages = {
            400: "The provider rejected the request",
            401: "API key rejected",
            402: "No remaining provider credits",
            403: "Access denied by the provider",
            404: "Model or endpoint not found",
            422: "The provider could not process this request",
            429: "Rate limit reached — wait a moment or switch models",
        }
        message = messages.get(status, f"Provider error (HTTP {status})")
        if detail:
            message = f"{message}: {detail}"
        kind = {
            401: "auth",
            402: "credits",
            403: "auth",
            404: "model",
            429: "rate_limit",
        }.get(status, "http")
        raise VisionError(message, kind=kind, status_code=status)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None