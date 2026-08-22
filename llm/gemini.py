"""
Google Gemini provider — talks to the native `generativelanguage.googleapis.com`
REST API directly over httpx, since Gemini's request shape (`contents` with
`parts`, `role` of "model" instead of "assistant") differs from both the
OpenAI and Anthropic conventions.
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

_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise ProviderError("Gemini requires an API key.", provider="gemini")

        self._client = httpx.AsyncClient(
            base_url=(config.endpoint or _DEFAULT_ENDPOINT).rstrip("/"),
            timeout=config.timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_contents(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, object]]]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        contents = [
            {
                "role": "model" if m.role == Role.ASSISTANT else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != Role.SYSTEM
        ]
        return ("\n\n".join(system_parts) or None, contents)

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        system, contents = self._to_contents(messages)
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                **({"maxOutputTokens": max_tokens} if max_tokens else {}),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            response = await self._client.post(
                f"/models/{self.config.model}:generateContent",
                params={"key": self.config.api_key},
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {exc}", provider="gemini", retryable=True) from exc

        data = response.json()
        candidate = data["candidates"][0]
        text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
        usage = data.get("usageMetadata", {})
        return GenerationResult(
            text=text,
            model=self.config.model,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            finish_reason=candidate.get("finishReason"),
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

        system, contents = self._to_contents(messages)
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                **({"maxOutputTokens": max_tokens} if max_tokens else {}),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with self._client.stream(
            "POST",
            f"/models/{self.config.model}:streamGenerateContent",
            params={"key": self.config.api_key, "alt": "sse"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk = _json.loads(line.removeprefix("data: "))
                for candidate in chunk.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if text := part.get("text"):
                            yield text

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        model = self.config.embedding_model or "text-embedding-004"
        vectors: list[list[float]] = []
        for text in texts:
            response = await self._client.post(
                f"/models/{model}:embedContent",
                params={"key": self.config.api_key},
                json={"content": {"parts": [{"text": text}]}},
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"]["values"])
        return EmbeddingResult(vectors=vectors, model=model, dimensions=len(vectors[0]) if vectors else 0)

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            response = await self._client.get("/models", params={"key": self.config.api_key})
            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return HealthStatus(
                    healthy=False, provider="gemini", model=self.config.model,
                    detail=f"HTTP {response.status_code}", latency_ms=latency,
                )
            return HealthStatus(healthy=True, provider="gemini", model=self.config.model, latency_ms=latency)
        except httpx.HTTPError as exc:
            return HealthStatus(healthy=False, provider="gemini", model=self.config.model, detail=str(exc))

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="gemini",
            model=self.config.model,
            supports_streaming=True,
            supports_embeddings=True,
        )
