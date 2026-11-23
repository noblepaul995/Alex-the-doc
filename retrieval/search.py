"""Semantic search over the LanceDB vector store."""

from __future__ import annotations

from pathlib import Path

from embeddings.lancedb import LanceVectorStore
from embeddings.vector import RecordType, VectorSearchResult
from llm.base import BaseProvider
from retrieval.ranking import rerank_results


async def semantic_search(
    query: str,
    vector_dir: Path,
    provider: BaseProvider,
    *,
    limit: int = 10,
    record_type: RecordType | None = None,
) -> list[VectorSearchResult]:
    """
    Embed `query`, search the LanceDB vector store at `vector_dir`,
    and return re-ranked `VectorSearchResult` matches.
    """
    store = await LanceVectorStore.connect(vector_dir)
    if await store.count() == 0:
        return []

    embedding_result = await provider.embed([query])
    if not embedding_result.vectors:
        return []

    raw = await store.search(
        embedding_result.vectors[0],
        limit=max(limit * 2, limit),
        record_type=record_type,
    )
    return rerank_results(raw, query, limit=limit)
