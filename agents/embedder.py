"""
Embedding Agent — embeds summaries (never raw code) into vector space.

Runs after the File Documentation Agent. Gathers every successful
`ChunkDocumentation.summary` and `FileDocumentation.summary` produced
so far, and embeds them via the configured provider, populating
`state.vectors`. Only summaries are embedded, deliberately: embedding
raw source code would mean re-deriving semantic meaning the LLM has
already extracted, at higher token cost and lower signal-to-noise than
embedding the summary that already distills it.

Structured like the two documentation agents before it: the
provider-calling core (`embeddings.embed.embed_items`) takes an
already-built `BaseProvider`, so it's unit-tested against an in-memory
fake with zero network access, and `embedder_node` is the only place
that touches `create_provider`.

One capability check this stage adds that the documentation agents
didn't need: not every provider supports embeddings at all (Anthropic,
Groq, and OpenRouter don't have a first-party embeddings endpoint —
see `PROVIDER_REGISTRY` in `config/providers.py`). If the configured
provider doesn't support embeddings, this stage records one clear error
and skips embedding entirely, rather than firing dozens of calls that
are all guaranteed to fail identically.
"""

from __future__ import annotations

from config.providers import PROVIDER_REGISTRY, resolve_provider_config
from config.settings import get_settings
from embeddings.embed import EmbeddingItem, embed_items
from embeddings.vector import RecordType, VectorCollection
from graph.state import RepositoryState
from llm.factory import create_provider
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)


async def embedder_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: embed every successful chunk/file summary into `state.vectors`."""
    items = _collect_items(state)
    if not items:
        return {"vectors": VectorCollection(), "statistics": state.statistics, "errors": state.errors}

    settings = get_settings()
    config = resolve_provider_config()

    registry_entry = PROVIDER_REGISTRY.get(config.provider)
    if registry_entry is not None and not registry_entry.supports_embeddings:
        state.record_error(
            "embed",
            f"Provider {config.provider.value!r} has no embeddings endpoint "
            f"({registry_entry.notes or 'not supported'}); configure a different provider for embeddings.",
        )
        return {"vectors": VectorCollection(), "statistics": state.statistics, "errors": state.errors}

    with Stopwatch("embed") as sw:
        async with create_provider(config) as provider:
            collection = await embed_items(
                items, provider, batch_size=settings.embedding_batch_size, max_concurrency=settings.max_concurrent_requests
            )

    for record in collection.failed():
        state.record_error("embed", f"{record.file_path} ({record.id}): {record.error}", file_path=record.file_path)

    state.statistics.embeddings_generated = len([r for r in collection.records if r.error is None])

    log.info(
        "Embedding Agent: %d/%d summary(ies) embedded (%d failed) in %.2fs",
        state.statistics.embeddings_generated,
        len(collection.records),
        len(collection.failed()),
        sw.elapsed_seconds,
    )

    return {"vectors": collection, "statistics": state.statistics, "errors": state.errors}


def _collect_items(state: RepositoryState) -> list[EmbeddingItem]:
    items = [
        EmbeddingItem(id=doc.chunk_id, record_type=RecordType.CHUNK, file_path=doc.file_path, text=doc.summary)
        for doc in state.chunk_docs.docs
        if doc.error is None and doc.summary
    ]
    items += [
        EmbeddingItem(id=doc.file_path, record_type=RecordType.FILE, file_path=doc.file_path, text=doc.summary)
        for doc in state.file_docs.docs
        if doc.error is None and doc.summary
    ]
    return items
