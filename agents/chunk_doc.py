"""
Chunk Documentation Agent — per-chunk documentation via LLM.

Runs after the Chunk Agent. For every `Chunk` in `state.chunks`, asks
the configured LLM provider for a short, factual summary of what that
chunk does, producing a `ChunkDocumentation` tied back to the chunk's
id/file/symbols. This is the first stage in the pipeline that makes a
network call, so it's also the first node in the graph that has to be
`async def` — see `graph/graph.py`'s docstring for what that changes
about how the graph is invoked.

Now backed by the Memory System, at file granularity, matched at chunk
granularity: `RepositoryMemory` caches one file's *complete* list of
`ChunkDocumentation`s together (keyed by that file's content hash), but
reuse is still decided per chunk id, not wholesale per file. That
distinction matters because a file's content hash staying the same
doesn't guarantee its *chunk boundaries* did — if `ALEX_CHUNK_MAX_TOKENS`
changed between runs, or the Chunker's packing behavior changed, the
same file could produce a different set of chunk ids even though
nothing in the source changed. Matching by chunk id means only chunks
that are genuinely still the same get reused; anything else falls back
to a fresh LLM call rather than risk handing back a stale or
misattributed summary. Only a file whose *entire* fresh chunk set
documented successfully gets cached — a partially-failed set is never
written back, so a later run can't mistake it for complete.

The provider-calling core (`_document_chunks`) takes an already-built
`BaseProvider` (and an optional already-built `RepositoryMemory`)
rather than resolving either itself, specifically so it can be
unit-tested against fakes with zero network or disk access —
`chunk_doc_node` (the actual LangGraph node) is the only place that
touches `create_provider`/`resolve_provider_config`/`RepositoryMemory`.

"Structured" documentation here means "one `ChunkDocumentation` record
per chunk, carrying its own id/file/symbol metadata" rather than "the
LLM is forced into a JSON schema": the prompt asks for a short plain-text
summary. Local models (the zero-config Ollama default in particular)
follow a plain-text instruction far more reliably than a strict JSON
schema without grammar-constrained decoding, and a summary is all this
stage actually needs — forcing JSON here would trade reliability for a
structure with no immediate consumer.
"""

from __future__ import annotations

from collections import defaultdict

from config.prompts import render_prompt
from config.providers import resolve_provider_config
from config.settings import get_settings
from graph.chunk import Chunk, ChunkDocumentation, ChunkDocumentationCollection
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from memory.repository import RepositoryMemory
from utils.helpers import gather_with_concurrency, with_retry
from utils.logger import get_logger
from utils.progress import report_progress
from utils.timers import Stopwatch

log = get_logger(__name__)

_MAX_PROMPT_CHARS = 8000
"""Hard safety cap on chunk content embedded in the prompt, independent of the Chunker's own token budget."""


async def chunk_doc_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate a `ChunkDocumentation` for every chunk in `state.chunks`."""
    chunks = state.chunks.chunks
    if not chunks:
        return {"chunk_docs": ChunkDocumentationCollection(), "statistics": state.statistics, "errors": state.errors}

    settings = get_settings()
    config = resolve_provider_config()
    memory = RepositoryMemory(settings.db_path) if settings.memory_enabled else None
    repository_key = str(state.repository_path.resolve())
    content_hashes = {path: pf.content_hash for path, pf in state.file_metadata.items()}

    with Stopwatch("chunk_doc") as sw:
        async with create_provider(config) as provider:
            collection = await _document_chunks(
                chunks,
                provider,
                max_concurrency=settings.max_concurrent_requests,
                memory=memory,
                repository_key=repository_key,
                content_hashes=content_hashes,
            )

    for doc in collection.failed():
        state.record_error("chunk_doc", f"{doc.file_path} ({doc.chunk_id}): {doc.error}", file_path=doc.file_path)

    state.statistics.chunks_documented = len([d for d in collection.docs if d.error is None])
    state.statistics.tokens_used += sum((d.prompt_tokens or 0) + (d.completion_tokens or 0) for d in collection.docs)

    log.info(
        "Chunk Documentation Agent: %d/%d chunk(s) documented (%d failed) in %.2fs",
        state.statistics.chunks_documented,
        len(chunks),
        len(collection.failed()),
        sw.elapsed_seconds,
    )

    return {"chunk_docs": collection, "statistics": state.statistics, "errors": state.errors}


async def _document_chunks(
    chunks: list[Chunk],
    provider: BaseProvider,
    *,
    max_concurrency: int,
    memory: RepositoryMemory | None = None,
    repository_key: str = "",
    content_hashes: dict[str, str] | None = None,
) -> ChunkDocumentationCollection:
    """
    Generate a `ChunkDocumentation` for every chunk in `chunks`, bounded
    to `max_concurrency` concurrent provider calls. A single chunk
    failing after retries doesn't stop the others — it's recorded with
    `error` set instead. Chunks whose id matches a cached entry for
    their file's current content hash are reused instead of re-calling
    the provider — see this module's docstring for why matching happens
    at chunk-id granularity, not just file-hash granularity.
    """
    content_hashes = content_hashes or {}
    cached_by_file: dict[str, dict[str, ChunkDocumentation]] = {}
    if memory is not None:
        for file_path in {c.file_path for c in chunks}:
            content_hash = content_hashes.get(file_path)
            if not content_hash:
                continue
            cached_list = memory.load_chunk_docs(repository_key, file_path, content_hash)
            if cached_list:
                cached_by_file[file_path] = {d.chunk_id: d for d in cached_list}

    async def _document_one(chunk: Chunk) -> ChunkDocumentation:
        cached = cached_by_file.get(chunk.file_path, {}).get(chunk.id)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "chunk",
            file_path=chunk.file_path,
            language=chunk.language,
            symbol_names=", ".join(chunk.symbol_names) or "(module-level code)",
            code=_truncate(chunk.content),
        )
        messages = [ChatMessage(role=Role.USER, content=prompt)]

        try:
            async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
                with attempt:
                    result = await provider.generate(messages, temperature=0.1, max_tokens=450)
        except ProviderError as exc:
            return ChunkDocumentation(
                chunk_id=chunk.id,
                file_path=chunk.file_path,
                summary="",
                symbol_names=chunk.symbol_names,
                model=provider.model_info().model,
                error=str(exc),
            )

        return ChunkDocumentation(
            chunk_id=chunk.id,
            file_path=chunk.file_path,
            summary=result.text.strip(),
            symbol_names=chunk.symbol_names,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    total = len(chunks)
    completed = 0
    report_progress("chunk_doc", 0, total)

    def _tick() -> None:
        nonlocal completed
        completed += 1
        report_progress("chunk_doc", completed, total)

    docs = await gather_with_concurrency(
        max_concurrency, *(_document_one(c) for c in chunks), on_complete=_tick
    )
    collection = ChunkDocumentationCollection(docs=list(docs))

    if memory is not None:
        docs_by_file: dict[str, list[ChunkDocumentation]] = defaultdict(list)
        for doc in collection.docs:
            docs_by_file[doc.file_path].append(doc)
        for file_path, file_docs in docs_by_file.items():
            content_hash = content_hashes.get(file_path)
            if content_hash and all(d.error is None for d in file_docs):
                memory.save_chunk_docs(repository_key, file_path, content_hash, file_docs)

    return collection


def _truncate(content: str) -> str:
    if len(content) <= _MAX_PROMPT_CHARS:
        return content
    return content[:_MAX_PROMPT_CHARS] + "\n... (truncated)"