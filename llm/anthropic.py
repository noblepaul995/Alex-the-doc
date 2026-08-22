"""
Anthropic provider — talks to the native `/v1/messages` API directly
over httpx (no SDK dependency required), since the wire format diverges
from the OpenAI convention (system prompt is a top-level field, not a
role in the messages array).

Anthropic has no first-party text-embeddings endpoint, so `embed()`
raises `ProviderError`; use another provider for the embedding stage.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx

from config.providers import ProviderConfig
from llm.base import (
    BaseProvider,
    ChatMessage,
    EmbeddingResult,
    GenerationResult,
    HealthStatus,
    ModelInfo,
    ProviderError,
    Role,
)
from utils.logger import get_logger

log = get_logger(__name__)

_API_VERSION = "2023-06-01"
_DEFAULT_ENDPOINT = "https://api.anthropic.com"


class AnthropicProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise ProviderError("Anthropic requires an API key.", provider="anthropic")

        self._client = httpx.AsyncClient(
            base_url=(config.endpoint or _DEFAULT_ENDPOINT).rstrip("/"),
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
            timeout=config.timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _split_system(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        conversation = [
            {"role": m.role.value, "content": m.content} for m in messages if m.role != Role.SYSTEM
        ]
        return ("\n\n".join(system_parts) or None, conversation)

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        system, conversation = self._split_system(messages)
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            payload["system"] = system

        try:
            response = await self._client.post("/v1/messages", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}", provider="anthropic", retryable=True) from exc

        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return GenerationResult(
            text=text,
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import json as _json

        system, conversation = self._split_system(messages)
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self._client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                event = _json.loads(line.removeprefix("data: "))
                if event.get("type") == "content_block_delta":
                    if text := event.get("delta", {}).get("text"):
                        yield text

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        raise ProviderError(
            "Anthropic has no first-party embeddings endpoint. Configure a different "
            "provider for the embedding stage (e.g. Ollama with nomic-embed-text).",
            provider="anthropic",
        )

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/messages",
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return HealthStatus(
                    healthy=False, provider="anthropic", model=self.config.model,
                    detail=f"HTTP {response.status_code}", latency_ms=latency,
                )
            return HealthStatus(healthy=True, provider="anthropic", model=self.config.model, latency_ms=latency)
        except httpx.HTTPError as exc:
            return HealthStatus(healthy=False, provider="anthropic", model=self.config.model, detail=str(exc))

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="anthropic",
            model=self.config.model,
            supports_streaming=True,
            supports_embeddings=False,
        )
