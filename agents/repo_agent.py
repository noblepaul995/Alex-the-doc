"""
Repository Agent — synthesizes a whole-repository summary.

Runs after the Knowledge Graph Agent in the full pipeline. Builds one
"digest" of the repository — statistics plus the summaries of its most
architecturally important files — and asks the configured LLM provider
for a single narrative `state.repo_summary` describing what the
repository does and how it's structured.

"Most important" is decided by dependents count from
`state.dependency_graph` (how many other files import this one), not
by file size, alphabetical order, or chunk count — a file many other
files depend on is a reasonable proxy for architectural centrality (a
core module, a shared data model, a config layer), which is exactly
what a whole-repository summary should be grounded in when not every
file summary can fit in one prompt.

Structured like the other single-shot synthesis stage (File Documentation):
the provider-calling core (`summarize_repository`) takes an
already-built `BaseProvider`, so it's unit-tested against an in-memory
fake with zero network access, and `repository_understanding_node` is
the only place that touches `create_provider`. Unlike Chunk/File
Documentation, there's exactly one LLM call for the whole run here, not
one per item — a repository has one summary, not one per file — so there's
no batching/concurrency to manage, just a single retried call.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.prompts import render_prompt
from config.providers import resolve_synthesis_provider_config
from config.settings import get_settings
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from utils.helpers import with_retry
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)


@dataclass
class RepoDigest:
    """The condensed repository content that goes into the summary prompt."""

    file_count: int
    symbol_count: int
    language_breakdown: str
    file_digest: str
    dependency_edges: str
    truncated: bool


async def repository_understanding_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: synthesize `state.repo_summary` from `state.file_docs` and `state.dependency_graph`."""
    digest = build_digest(state, max_files=get_settings().repo_digest_max_files)
    if digest is None:
        return {"repo_summary": None, "statistics": state.statistics, "errors": state.errors}

    config = resolve_synthesis_provider_config()

    with Stopwatch("repository_understanding") as sw:
        async with create_provider(config) as provider:
            try:
                summary, prompt_tokens, completion_tokens = await summarize_repository(digest, provider)
            except ProviderError as exc:
                state.record_error("repository_understanding", f"Failed to synthesize repository summary: {exc}", exception=exc)
                return {"repo_summary": None, "statistics": state.statistics, "errors": state.errors}

    state.statistics.tokens_used += (prompt_tokens or 0) + (completion_tokens or 0)
    log.info("Repository Agent: repository summary generated in %.2fs", sw.elapsed_seconds)

    return {"repo_summary": summary, "statistics": state.statistics, "errors": state.errors}


def build_digest(state: RepositoryState, *, max_files: int) -> RepoDigest | None:
    """
    Build a `RepoDigest` from `state.file_docs` and `state.dependency_graph`.
    Returns `None` if there are no successful file docs to summarize —
    nothing grounded to synthesize from.
    """
    successful = [d for d in state.file_docs.docs if d.error is None]
    if not successful:
        return None

    language_counts: dict[str, int] = {}
    symbol_count = 0
    for parse_result in state.parse_results.values():
        language_counts[parse_result.language] = language_counts.get(parse_result.language, 0) + 1
        symbol_count += len(parse_result.symbols)
    language_breakdown = ", ".join(f"{lang}: {count}" for lang, count in sorted(language_counts.items(), key=lambda kv: -kv[1]))

    truncated = len(successful) > max_files
    if truncated:
        selected = sorted(successful, key=lambda d: len(state.dependency_graph.dependents_of(d.file_path)), reverse=True)[:max_files]
        selected.sort(key=lambda d: d.file_path)
    else:
        selected = sorted(successful, key=lambda d: d.file_path)

    file_digest = "\n".join(f"- {d.file_path} ({d.language}): {d.summary}" for d in selected)

    # Real, verified import relationships among the selected files — computed
    # from the actual dependency graph, not inferred. This is structural
    # ("imports"), not temporal ("runs before/after" or "in parallel with"):
    # it exists so prompts consuming this digest have *something concrete* to
    # ground relationship claims in, rather than the model inventing execution
    # order or concurrency that no summary here actually states.
    selected_paths = {d.file_path for d in selected}
    edge_lines: list[str] = []
    for d in selected:
        deps = sorted(set(state.dependency_graph.dependencies_of(d.file_path)) & selected_paths)
        if deps:
            edge_lines.append(f"- {d.file_path} imports: {', '.join(deps)}")
    dependency_edges = "\n".join(edge_lines) if edge_lines else "(no internal import relationships found among the selected files)"

    return RepoDigest(
        file_count=len(successful),
        symbol_count=symbol_count,
        language_breakdown=language_breakdown or "(none)",
        file_digest=file_digest,
        dependency_edges=dependency_edges,
        truncated=truncated,
    )


async def summarize_repository(digest: RepoDigest, provider: BaseProvider) -> tuple[str, int | None, int | None]:
    """
    Generate the whole-repository summary for `digest`. Returns
    `(summary, prompt_tokens, completion_tokens)`. Raises `ProviderError`
    if generation fails after retries — the caller decides how to
    record that, matching the pattern used elsewhere in this codebase
    rather than swallowing the error here.
    """
    truncation_note = ""
    if digest.truncated:
        truncation_note = f"(showing the {len(digest.file_digest.splitlines())} most depended-upon of {digest.file_count} total files)\n"

    prompt = render_prompt(
        "repository",
        file_count=digest.file_count,
        symbol_count=digest.symbol_count,
        language_breakdown=digest.language_breakdown,
        truncation_note=truncation_note,
        file_digest=digest.file_digest,
    )
    messages = [ChatMessage(role=Role.USER, content=prompt)]

    async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
        with attempt:
            result = await provider.generate(messages, temperature=0.1, max_tokens=600)

    return result.text.strip(), result.prompt_tokens, result.completion_tokens
