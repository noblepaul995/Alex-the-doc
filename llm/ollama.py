"""
Ollama provider — talks to a local (or remote) Ollama server over its
native HTTP API. This is the zero-config, zero-API-key default so
alex works fully offline out of the box.
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
from utils.logger import get_logger

log = get_logger(__name__)


class OllamaProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        endpoint = config.endpoint or "http://localhost:11434"
        self._client = httpx.AsyncClient(base_url=endpoint.rstrip("/"), timeout=config.timeout_seconds)

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
            "stream": False,
            # Reasoning-capable models (e.g. qwen3.x "thinking" variants) support
            # a top-level `think` flag. Documentation summaries don't benefit
            # from a visible chain-of-thought pass — it's pure extra latency
            # and token cost here — so we disable it explicitly. Non-thinking
            # models simply ignore this field.
            "think": False,
            "options": {"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}", provider="ollama", retryable=True) from exc

        data = response.json()
        return GenerationResult(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.config.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason="stop" if data.get("done") else None,
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
            "stream": True,
            "think": False,
            "options": {"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})},
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = _json.loads(line)
                if content := chunk.get("message", {}).get("content"):
                    yield content
                if chunk.get("done"):
                    break

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        model = self.config.embedding_model or "nomic-embed-text"
        vectors: list[list[float]] = []
        for text in texts:
            response = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return EmbeddingResult(vectors=vectors, model=model, dimensions=len(vectors[0]) if vectors else 0)

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            response = await self._client.get("/api/tags")
            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return HealthStatus(
                    healthy=False, provider="ollama", model=self.config.model,
                    detail=f"HTTP {response.status_code}", latency_ms=latency,
                )
            return HealthStatus(healthy=True, provider="ollama", model=self.config.model, latency_ms=latency)
        except httpx.HTTPError as exc:
            return HealthStatus(healthy=False, provider="ollama", model=self.config.model, detail=str(exc))

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="ollama",
            model=self.config.model,
            supports_streaming=True,
            supports_embeddings=True,
        )
