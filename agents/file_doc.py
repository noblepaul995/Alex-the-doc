"""
File Documentation Agent — combines chunk docs into file-level documentation.

Runs after the Chunk Documentation Agent. For every file that has at
least one successfully-documented chunk, asks the configured LLM
provider to synthesize those chunk summaries (plus the file's top-level
symbol names) into one whole-file `FileDocumentation`.

Now backed by the Memory System: a file whose current content hash
matches a cached `FileDocumentation` reuses it directly, skipping the
LLM call entirely — simpler than the Chunk Documentation Agent's
per-chunk-id matching, since a `FileDocumentation` is already exactly
one record per file, not a list that could partially mismatch. Only
successful results get cached, so a cache hit is always a complete,
previously-successful summary, never a stale error.

Structured the same way as `agents/chunk_doc.py`: the provider-calling
core (`_document_files`) takes an already-built `BaseProvider` (and an
optional already-built `RepositoryMemory`), so it's unit-testable
against fakes with zero network or disk access, and `file_doc_node`
(the actual LangGraph node) is the only place that touches
`create_provider`/`RepositoryMemory`.

A file whose chunks *all* failed to document skips the LLM call
entirely and is recorded with `error` set directly — there's nothing
usable to synthesize, so making that call would just be a guaranteed
failure (or worse, a summary hallucinated from the file path and symbol
names alone, with no grounding in the actual chunk content).
"""

from __future__ import annotations

from collections import defaultdict

from config.prompts import render_prompt
from config.providers import resolve_provider_config
from config.settings import get_settings
from graph.chunk import Chunk, ChunkDocumentation, ChunkDocumentationCollection
from graph.file_doc import FileDocumentation, FileDocumentationCollection
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from memory.repository import RepositoryMemory
from parsers.metadata import ParseResult
from utils.helpers import gather_with_concurrency, with_retry
from utils.logger import get_logger
from utils.progress import report_progress
from utils.timers import Stopwatch

log = get_logger(__name__)


async def file_doc_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate a `FileDocumentation` for every file with at least one chunk doc."""
    if not state.chunk_docs.docs:
        return {"file_docs": FileDocumentationCollection(), "statistics": state.statistics, "errors": state.errors}

    settings = get_settings()
    config = resolve_provider_config()
    memory = RepositoryMemory(settings.db_path) if settings.memory_enabled else None
    repository_key = str(state.repository_path.resolve())
    content_hashes = {path: pf.content_hash for path, pf in state.file_metadata.items()}

    with Stopwatch("file_doc") as sw:
        async with create_provider(config) as provider:
            collection = await _document_files(
                state.chunk_docs,
                state.chunks.chunks,
                state.parse_results,
                provider,
                max_concurrency=settings.max_concurrent_requests,
                memory=memory,
                repository_key=repository_key,
                content_hashes=content_hashes,
            )

    for doc in collection.failed():
        state.record_error("file_doc", f"{doc.file_path}: {doc.error}", file_path=doc.file_path)

    state.statistics.tokens_used += sum((d.prompt_tokens or 0) + (d.completion_tokens or 0) for d in collection.docs)

    log.info(
        "File Documentation Agent: %d/%d file(s) documented (%d failed) in %.2fs",
        len([d for d in collection.docs if d.error is None]),
        len(collection.docs),
        len(collection.failed()),
        sw.elapsed_seconds,
    )

    return {"file_docs": collection, "statistics": state.statistics, "errors": state.errors}


async def _document_files(
    chunk_docs: ChunkDocumentationCollection,
    chunks: list[Chunk],
    parse_results: dict[str, ParseResult],
    provider: BaseProvider,
    *,
    max_concurrency: int,
    memory: RepositoryMemory | None = None,
    repository_key: str = "",
    content_hashes: dict[str, str] | None = None,
) -> FileDocumentationCollection:
    """
    Generate a `FileDocumentation` for every file represented in
    `chunk_docs`, bounded to `max_concurrency` concurrent provider
    calls. A file failing after retries doesn't stop the others. A file
    whose current content hash matches a cached entry reuses it instead
    of calling the provider.
    """
    content_hashes = content_hashes or {}
    chunk_start_line: dict[str, int] = {c.id: c.start_line for c in chunks}

    docs_by_file: dict[str, list[ChunkDocumentation]] = defaultdict(list)
    for doc in chunk_docs.docs:
        docs_by_file[doc.file_path].append(doc)

    async def _document_one(file_path: str, docs: list[ChunkDocumentation]) -> FileDocumentation:
        if memory is not None:
            content_hash = content_hashes.get(file_path)
            if content_hash:
                cached = memory.load_file_doc(repository_key, file_path, content_hash)
                if cached is not None:
                    return cached

        successful = sorted((d for d in docs if d.error is None), key=lambda d: chunk_start_line.get(d.chunk_id, 0))
        parse_result = parse_results.get(file_path)
        language = parse_result.language if parse_result else "unknown"
        symbol_names = [s.name for s in parse_result.symbols if s.parent is None] if parse_result else []

        if not successful:
            return FileDocumentation(
                file_path=file_path,
                language=language,
                summary="",
                symbol_names=symbol_names,
                chunk_count=len(docs),
                model=provider.model_info().model,
                error="All chunks in this file failed to document; nothing to synthesize.",
            )

        chunk_summaries = "\n".join(f"- {d.summary}" for d in successful)
        prompt = render_prompt(
            "file",
            file_path=file_path,
            language=language,
            symbol_names=", ".join(symbol_names) or "(none)",
            chunk_summaries=chunk_summaries,
        )
        messages = [ChatMessage(role=Role.USER, content=prompt)]

        try:
            async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
                with attempt:
                    result = await provider.generate(messages, temperature=0.1, max_tokens=650)
        except ProviderError as exc:
            return FileDocumentation(
                file_path=file_path,
                language=language,
                summary="",
                symbol_names=symbol_names,
                chunk_count=len(docs),
                model=provider.model_info().model,
                error=str(exc),
            )

        return FileDocumentation(
            file_path=file_path,
            language=language,
            summary=result.text.strip(),
            symbol_names=symbol_names,
            chunk_count=len(docs),
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    total = len(docs_by_file)
    completed = 0
    report_progress("file_doc", 0, total)

    def _tick() -> None:
        nonlocal completed
        completed += 1
        report_progress("file_doc", completed, total)

    docs = await gather_with_concurrency(
        max_concurrency, *(_document_one(fp, ds) for fp, ds in docs_by_file.items()), on_complete=_tick
    )
    collection = FileDocumentationCollection(docs=list(docs))

    if memory is not None:
        for doc in collection.docs:
            content_hash = content_hashes.get(doc.file_path)
            if content_hash and doc.error is None:
                memory.save_file_doc(repository_key, doc.file_path, content_hash, doc)

    return collection