"""Result re-ranking for retrieved vector search summaries."""

from __future__ import annotations

import re

from embeddings.vector import RecordType, VectorSearchResult

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def rerank_results(
    results: list[VectorSearchResult],
    query: str,
    *,
    limit: int = 10,
) -> list[VectorSearchResult]:
    """
    Re-rank `VectorSearchResult` items by blending vector similarity with
    lexical overlap on `file_path`, `symbol_names`, and `summary`.
    """
    if not results:
        return []

    query_tokens = _tokens(query)

    def _score(item: VectorSearchResult) -> float:
        # LanceDB returns L2/cosine distance where smaller is closer.
        sim = 1.0 / (1.0 + max(item.distance, 0.0))
        if not query_tokens:
            return sim

        haystack_tokens = _tokens(f"{item.file_path} {item.text}")
        overlap = len(query_tokens & haystack_tokens) / len(query_tokens)
        type_boost = 0.03 if item.record_type == RecordType.FILE else 0.0
        return sim + 0.25 * overlap + type_boost

    ranked = sorted(results, key=_score, reverse=True)
    return ranked[:limit]
