"""
Provider implementation for any endpoint speaking the OpenAI Chat
Completions wire format: OpenAI itself, Groq, OpenRouter, LM Studio, and
arbitrary custom OpenAI-compatible servers.

Kept as a single class (parametrized by base URL + API key) rather than
one class per provider, since the wire protocol is identical — this is
exactly the "provider-independent" principle from the spec applied one
level down.
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
)
from utils.helpers import with_retry
from utils.logger import get_logger

log = get_logger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider(BaseProvider):
    """Chat/embeddings client for OpenAI-compatible HTTP APIs."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.endpoint:
            raise ProviderError(
                "An endpoint is required for OpenAI-compatible providers.",
                provider=config.provider.value,
            )
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._client = httpx.AsyncClient(
            base_url=config.endpoint.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _messages_payload(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        payload = {
            "model": self.config.model,
            "messages": self._messages_payload(messages),
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async for attempt in with_retry(max_attempts=self.config.max_retries, exceptions=(ProviderError,)):
            with attempt:
                response = await self._client.post("/chat/completions", json=payload)
                self._raise_for_retryable(response)
                response.raise_for_status()
                data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return GenerationResult(
            text=choice["message"]["content"],
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
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

        payload = {
            "model": self.config.model,
            "messages": self._messages_payload(messages),
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            self._raise_for_retryable(response)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line.removeprefix("data: ").strip()
                if data_str == "[DONE]":
                    break
                chunk = _json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        payload = {"model": self.config.embedding_model or self.config.model, "input": texts}
        response = await self._client.post("/embeddings", json=payload)
        self._raise_for_retryable(response)
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        return EmbeddingResult(
            vectors=vectors,
            model=data.get("model", payload["model"]),
            dimensions=len(vectors[0]) if vectors else 0,
        )

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            response = await self._client.get("/models")
            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return HealthStatus(
                    healthy=False,
                    provider=self.config.provider.value,
                    model=self.config.model,
                    detail=f"HTTP {response.status_code}",
                    latency_ms=latency,
                )
            return HealthStatus(
                healthy=True,
                provider=self.config.provider.value,
                model=self.config.model,
                latency_ms=latency,
            )
        except httpx.HTTPError as exc:
            return HealthStatus(
                healthy=False,
                provider=self.config.provider.value,
                model=self.config.model,
                detail=str(exc),
            )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.config.provider.value,
            model=self.config.model,
            supports_streaming=True,
            supports_embeddings=True,
        )

    def _raise_for_retryable(self, response: httpx.Response) -> None:
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise ProviderError(
                f"Retryable HTTP {response.status_code} from {self.config.provider.value}",
                provider=self.config.provider.value,
                retryable=True,
            )
