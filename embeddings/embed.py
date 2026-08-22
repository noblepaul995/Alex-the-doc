"""
Embedding orchestration — batches summaries through the configured provider.

Takes a flat list of `EmbeddingItem`s (never raw source code — see
`agents/embedder.py`'s module docstring) and turns them into a
`VectorCollection`, batching multiple texts into each `provider.embed()`
call rather than one call per item. This matters for both provider
families: OpenAI-compatible providers accept a true batch request (one
HTTP call for the whole list), and Ollama's `embed()` loops one request
per text internally regardless — either way, batching at this layer
bounds how many concurrent `embed()` calls are in flight rather than
firing one per chunk/file unboundedly.

Order within a batch is trusted to come back aligned with the input
list (`vectors[i]` corresponds to `texts[i]`) — this is what every
current provider implementation in `llm/` actually does, and is also
what the OpenAI Embeddings API guarantees, so it's not a new
assumption introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

from embeddings.vector import RecordType, VectorCollection, VectorRecord
from llm.base import BaseProvider, ProviderError
from utils.helpers import gather_with_concurrency, with_retry
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EmbeddingItem:
    """One thing to embed: an id/file_path pair plus the summary text (never raw code)."""

    id: str
    record_type: RecordType
    file_path: str
    text: str


async def embed_items(items: list[EmbeddingItem], provider: BaseProvider, *, batch_size: int, max_concurrency: int) -> VectorCollection:
    """
    Embed every item in `items`, batching `batch_size` texts per
    `provider.embed()` call and running up to `max_concurrency` batches
    concurrently. A batch failing after retries doesn't stop the
    others — every item in that batch is recorded with `error` set
    (a batch-level HTTP failure can't be attributed to one item within
    it, so all items in the failed batch share the same error).
    """
    if not items:
        return VectorCollection()

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results = await gather_with_concurrency(max_concurrency, *(_embed_batch(batch, provider) for batch in batches))

    records = [record for batch_records in results for record in batch_records]
    return VectorCollection(records=records)


async def _embed_batch(batch: list[EmbeddingItem], provider: BaseProvider) -> list[VectorRecord]:
    texts = [item.text for item in batch]

    try:
        async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
            with attempt:
                result = await provider.embed(texts)
    except ProviderError as exc:
        model = provider.model_info().model
        return [
            VectorRecord(id=item.id, record_type=item.record_type, file_path=item.file_path, text=item.text, model=model, error=str(exc))
            for item in batch
        ]

    if len(result.vectors) != len(batch):
        # Defensive: a provider returning a mismatched vector count is a provider bug, not
        # something we can recover an item-to-vector mapping from. Fail the whole batch rather
        # than silently zipping mismatched items to vectors.
        message = f"provider returned {len(result.vectors)} vector(s) for {len(batch)} text(s)"
        return [
            VectorRecord(id=item.id, record_type=item.record_type, file_path=item.file_path, text=item.text, model=result.model, error=message)
            for item in batch
        ]

    return [
        VectorRecord(
            id=item.id,
            record_type=item.record_type,
            file_path=item.file_path,
            text=item.text,
            vector=vector,
            model=result.model,
            dimensions=result.dimensions,
        )
        for item, vector in zip(batch, result.vectors)
    ]
