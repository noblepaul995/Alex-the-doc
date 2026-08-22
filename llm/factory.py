"""
Provider factory.

The single place in the codebase that maps a `ProviderType` to a
concrete `BaseProvider` subclass. Agents never import a specific
provider module directly — they call `create_provider(config)` and get
back something implementing the `BaseProvider` interface.
"""

from __future__ import annotations

from config.providers import OPENAI_COMPATIBLE_PROVIDERS, ProviderConfig, ProviderType
from llm.anthropic import AnthropicProvider
from llm.base import BaseProvider, ProviderError
from llm.gemini import GeminiProvider
from llm.ollama import OllamaProvider
from llm.openai import OpenAICompatibleProvider
from llm.openrouter import OpenRouterProvider


def create_provider(config: ProviderConfig) -> BaseProvider:
    """Instantiate the correct provider implementation for `config`."""
    if config.provider == ProviderType.OLLAMA:
        return OllamaProvider(config)
    if config.provider == ProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    if config.provider == ProviderType.GEMINI:
        return GeminiProvider(config)
    if config.provider == ProviderType.OPENROUTER:
        return OpenRouterProvider(config)
    if config.provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleProvider(config)

    raise ProviderError(f"Unknown provider type: {config.provider}", provider=str(config.provider))
