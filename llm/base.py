"""
The provider abstraction boundary.

Nothing outside `llm/` should ever know whether it's talking to Ollama,
OpenAI, Anthropic, Gemini, Groq, OpenRouter, LM Studio, or a bespoke
internal endpoint. Everything upstream (agents, graph nodes) talks only
to this interface.

Every concrete provider implements:

    generate()    -- single-shot chat completion
    stream()      -- token-streamed chat completion
    embed()       -- text -> vector(s)
    health()      -- cheap reachability/auth check, used by `doctor`
    model_info()  -- static metadata about the configured model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel

from config.providers import ProviderConfig


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class GenerationResult(BaseModel):
    """The result of a single-shot `generate()` call."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict | None = None


class EmbeddingResult(BaseModel):
    """The result of an `embed()` call — one vector per input text."""

    vectors: list[list[float]]
    model: str
    dimensions: int


class ModelInfo(BaseModel):
    provider: str
    model: str
    supports_streaming: bool
    supports_embeddings: bool
    context_window: int | None = None


class HealthStatus(BaseModel):
    healthy: bool
    provider: str
    model: str
    detail: str = ""
    latency_ms: float | None = None


class ProviderError(Exception):
    """Raised for any provider-side failure (auth, network, rate limit)."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class BaseProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Single-shot chat completion."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Token-streamed chat completion. Yields text deltas."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed one or more texts. Raises `ProviderError` if unsupported."""
        ...

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Cheap reachability/auth check, used by the `doctor` CLI command."""
        ...

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Static metadata about the currently configured model."""
        ...

    async def aclose(self) -> None:
        """Release any held resources (HTTP clients, etc.). Optional override."""
        return None

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()
