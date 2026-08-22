"""
OpenRouter provider.

OpenRouter speaks the OpenAI Chat Completions format, so behaviourally
this is `OpenAICompatibleProvider` — this subclass exists only to attach
OpenRouter's recommended attribution headers (`HTTP-Referer`, `X-Title`),
which affect routing/analytics on their side but aren't part of the
generic OpenAI-compatible contract.
"""

from __future__ import annotations

from config.providers import ProviderConfig
from llm.openai import OpenAICompatibleProvider

_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig) -> None:
        if not config.endpoint:
            config = config.model_copy(update={"endpoint": _DEFAULT_ENDPOINT})
        super().__init__(config)
        self._client.headers.update(
            {
                "HTTP-Referer": "https://github.com/alex",
                "X-Title": "alex",
            }
        )
