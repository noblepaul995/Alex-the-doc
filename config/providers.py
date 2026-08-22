"""
Provider-independent configuration layer.

Users only ever need to supply three things:

    model     -- e.g. "gpt-4o-mini", "claude-sonnet-4-6", "llama3.1:8b"
    api_key   -- optional, required for hosted providers
    endpoint  -- optional, overrides the provider's default base URL

Everything else (auth headers, request shape, streaming behaviour) is
resolved by the concrete provider implementation in `llm/`. This module
only describes *what* a provider needs, never *how* it talks to it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings


class ProviderType(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    LMSTUDIO = "lmstudio"
    CUSTOM = "custom"


class ProviderConfig(BaseModel):
    """Everything a provider implementation needs to make a request."""

    provider: ProviderType
    model: str
    api_key: str | None = None
    endpoint: str | None = None
    embedding_model: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3

    model_config = {"frozen": True}


# Providers that speak the OpenAI Chat Completions wire format. This lets a
# single provider implementation (`llm.openai.OpenAICompatibleProvider`)
# serve all of them, distinguished only by base URL + API key.
OPENAI_COMPATIBLE_PROVIDERS: frozenset[ProviderType] = frozenset(
    {
        ProviderType.OPENAI,
        ProviderType.GROQ,
        ProviderType.OPENROUTER,
        ProviderType.LMSTUDIO,
        ProviderType.CUSTOM,
    }
)


def resolve_provider_config(
    *,
    provider: str | ProviderType | None = None,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    embedding_model: str | None = None,
    settings: Settings | None = None,
) -> ProviderConfig:
    """
    Build a fully-resolved `ProviderConfig` from explicit user input,
    falling back to environment-derived defaults from `Settings`.

    This is the single choke point the rest of the app should go through
    to figure out "which provider, with which credentials, talking to
    which endpoint" — CLI flags, config files, and defaults all funnel
    through here.
    """
    settings = settings or get_settings()

    provider_type = ProviderType(provider) if provider else ProviderType(settings.default_provider)
    resolved_model = model or settings.default_model
    resolved_embedding_model = embedding_model or settings.default_embedding_model

    default_endpoint, default_key = _defaults_for(provider_type, settings)

    return ProviderConfig(
        provider=provider_type,
        model=resolved_model,
        api_key=api_key or default_key,
        endpoint=endpoint or default_endpoint,
        embedding_model=resolved_embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )


def resolve_synthesis_provider_config(settings: Settings | None = None) -> ProviderConfig:
    """
    Like `resolve_provider_config()`, but for the handful of whole-repository
    synthesis calls (Repository, Architecture, README, Review Agents) rather
    than the per-chunk/per-file calls. Applies `settings.synthesis_provider`
    / `settings.synthesis_model` on top of the usual defaults when set —
    falls back to exactly the same provider/model as everything else when
    they're not (`None` for both is the default, meaning "no override").

    This is the single choke point those four agents should go through,
    rather than each separately checking `settings.synthesis_provider` —
    see `settings.synthesis_provider`'s docstring for the reasoning.
    """
    settings = settings or get_settings()
    return resolve_provider_config(
        provider=settings.synthesis_provider,
        model=settings.synthesis_model,
        settings=settings,
    )


def _defaults_for(provider_type: ProviderType, settings: Settings) -> tuple[str | None, str | None]:
    """Return (endpoint, api_key) defaults for a given provider type."""
    mapping: dict[ProviderType, tuple[str | None, str | None]] = {
        ProviderType.OLLAMA: (settings.ollama_endpoint, None),
        ProviderType.OPENAI: (settings.openai_endpoint, settings.openai_api_key),
        ProviderType.ANTHROPIC: (None, settings.anthropic_api_key),
        ProviderType.GEMINI: (None, settings.gemini_api_key),
        ProviderType.GROQ: (settings.groq_endpoint, settings.groq_api_key),
        ProviderType.OPENROUTER: (settings.openrouter_endpoint, settings.openrouter_api_key),
        ProviderType.LMSTUDIO: (settings.lmstudio_endpoint, None),
        ProviderType.CUSTOM: (None, None),
    }
    return mapping.get(provider_type, (None, None))


class ProviderRegistryEntry(BaseModel):
    """Static metadata about a provider, used for CLI help / `doctor`."""

    name: str
    requires_api_key: bool
    supports_embeddings: bool
    default_endpoint: str | None = None
    notes: str = ""


PROVIDER_REGISTRY: dict[ProviderType, ProviderRegistryEntry] = {
    ProviderType.OLLAMA: ProviderRegistryEntry(
        name="Ollama",
        requires_api_key=False,
        supports_embeddings=True,
        default_endpoint="http://localhost:11434",
        notes="Local models. No API key needed.",
    ),
    ProviderType.OPENAI: ProviderRegistryEntry(
        name="OpenAI",
        requires_api_key=True,
        supports_embeddings=True,
        default_endpoint="https://api.openai.com/v1",
    ),
    ProviderType.ANTHROPIC: ProviderRegistryEntry(
        name="Anthropic",
        requires_api_key=True,
        supports_embeddings=False,
        notes="No first-party embeddings endpoint; use another provider for embeddings.",
    ),
    ProviderType.GEMINI: ProviderRegistryEntry(
        name="Google Gemini",
        requires_api_key=True,
        supports_embeddings=True,
    ),
    ProviderType.GROQ: ProviderRegistryEntry(
        name="Groq",
        requires_api_key=True,
        supports_embeddings=False,
        default_endpoint="https://api.groq.com/openai/v1",
    ),
    ProviderType.OPENROUTER: ProviderRegistryEntry(
        name="OpenRouter",
        requires_api_key=True,
        supports_embeddings=False,
        default_endpoint="https://openrouter.ai/api/v1",
    ),
    ProviderType.LMSTUDIO: ProviderRegistryEntry(
        name="LM Studio",
        requires_api_key=False,
        supports_embeddings=True,
        default_endpoint="http://localhost:1234/v1",
    ),
    ProviderType.CUSTOM: ProviderRegistryEntry(
        name="Custom OpenAI-compatible endpoint",
        requires_api_key=False,
        supports_embeddings=True,
    ),
}
