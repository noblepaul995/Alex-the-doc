"""
Review Agent — checks each generated document against the grounding
context it was produced from, flagging grammar issues, internal
inconsistencies, and claims the grounding doesn't support.

Deliberately report-only (see `graph/review.py`'s module docstring for
why): this stage never rewrites `state.generated_docs`, it only
populates `state.review` with findings.

Grounding is picked per document type, matching what each Documentation
Generator agent actually used to produce it. "architecture" and "readme"
were both grounded in `state.repo_summary` *plus* the per-file digest
(`RepoDigest.file_digest` / `dependency_edges`) that summary itself was
built from — see `readme_agent.generate_readme` and
`architecture_agent.generate_architecture_narrative`. Reviewing against
`repo_summary` alone was tried first and dropped: `repo_summary` is
itself a compressed narrative of the digest, so anything the README
pulled straight from a file summary (a specific filename, a helper
function, a library) but that didn't survive the repo-summary
compression reads as an unsupported claim to the reviewer — a false
positive, not a real hallucination. Rebuilding the same `RepoDigest` via
`repo_agent.build_digest` is cheap and deterministic (no LLM call, same
pattern as the "api" digest below), so review grounding now matches
generation grounding exactly. "api" grounding is a freshly-rebuilt
endpoint digest (cheap and deterministic — no LLM call — since
`api_agent.detect_endpoints` just re-reads `state.parse_results`). A
document with no available grounding is skipped rather than reviewed
against nothing, which would just be asking the model to guess whether
text "sounds right" with no ground truth to check it against.

The review response format is plain text with one required sentinel
line (`STATUS: PASS` / `STATUS: ISSUES_FOUND`) rather than JSON — same
reasoning as every other LLM-facing stage in this codebase: local
models follow a simple, explicit format far more reliably than a
strict schema without grammar-constrained decoding. Parsing is
deliberately lenient: a response that doesn't include a recognizable
status line defaults to `passed=True` rather than risk false positives
from a review the model didn't format as asked — documented directly in
`parse_review_response`, not hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.api_agent import build_endpoint_digest, detect_endpoints
from agents.repo_agent import build_digest
from config.prompts import render_prompt
from config.providers import resolve_synthesis_provider_config
from config.settings import get_settings
from graph.review import ReviewCollection, ReviewResult
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from utils.banned_phrases import find_banned_phrases
from utils.file_references import find_nonexistent_file_references
from utils.helpers import gather_with_concurrency, with_retry
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)

_MAX_ISSUES_PER_DOC = 5
"""Matches the "at most 5" instruction in prompts/review.md — see parse_review_response."""


@dataclass
class ReviewItem:
    doc_name: str
    doc_text: str
    grounding: str


async def review_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: review every document in `state.generated_docs`, populating `state.review`."""
    items = build_review_items(state)

    if items:
        settings = get_settings()
        config = resolve_synthesis_provider_config()
        with Stopwatch("review") as sw:
            async with create_provider(config) as provider:
                collection = await review_documents(items, provider, max_concurrency=settings.max_concurrent_requests)
        elapsed = sw.elapsed_seconds
    else:
        collection = ReviewCollection()
        elapsed = 0.0

    collection = _apply_deterministic_checks(collection, state)

    for result in collection.failed_reviews():
        state.record_error("review", f"{result.doc_name}: {result.error}")

    log.info(
        "Review Agent: %d document(s) reviewed (%d flagged, %d review failures) in %.2fs",
        len(collection.results),
        len(collection.flagged_documents()),
        len(collection.failed_reviews()),
        elapsed,
    )

    return {"review": collection, "statistics": state.statistics, "errors": state.errors}


def _apply_deterministic_checks(collection: ReviewCollection, state: RepositoryState) -> ReviewCollection:
    """
    Deterministic backstop for two things prompt-level instructions alone
    have proven unreliable to enforce across real runs:

      - generic stock phrasing ("promotes flexibility and maintainability"),
        which the README/Architecture prompts explicitly ban but a model
        can simply ignore — see `utils/banned_phrases.py`.
      - references to files that don't actually exist in this repository
        (a fabricated path, or a subtly wrong one like a pluralization
        mismatch) — see `utils/file_references.py`.

    Runs over every generated document, not just the ones
    `build_review_items` sent to the LLM reviewer (e.g. "structure" and
    "changelog" never get an LLM review since they're deterministic
    themselves, but can still be checked here), merging any hits into that
    document's `ReviewResult` — creating one if none exists yet.
    """
    real_paths = set(state.file_metadata.keys())
    results_by_doc = {result.doc_name: result for result in collection.results}

    for doc_name, doc_text in state.generated_docs.items():
        issue_lines: list[str] = []

        phrase_hits = find_banned_phrases(doc_text)
        if phrase_hits:
            issue_lines.extend(f'Contains banned generic phrase: "{phrase}"' for phrase in sorted(set(phrase_hits), key=str.lower))

        missing_files = find_nonexistent_file_references(doc_text, real_paths, repository_root=state.repository_path)
        if missing_files:
            issue_lines.extend(f'References a file that doesn\'t exist in this repository: "{path}"' for path in missing_files)

        if not issue_lines:
            continue

        existing = results_by_doc.get(doc_name)
        if existing is not None and existing.error is None:
            existing.passed = False
            existing.issues = [*existing.issues, *issue_lines]
        else:
            results_by_doc[doc_name] = ReviewResult(
                doc_name=doc_name, passed=False, issues=issue_lines, model="deterministic-scan"
            )

    return ReviewCollection(results=list(results_by_doc.values()))


def build_review_items(state: RepositoryState) -> list[ReviewItem]:
    """Pair every document in `state.generated_docs` with the grounding context it was produced from, skipping any with none available."""
    items: list[ReviewItem] = []

    if state.repo_summary:
        digest = build_digest(state, max_files=get_settings().repo_digest_max_files)
        digest_text = (
            f"{state.repo_summary}\n\nPer-file summaries this narrative was built from:\n{digest.file_digest}\n\n"
            f"Verified import relationships:\n{digest.dependency_edges}"
            if digest is not None
            else state.repo_summary
        )
        for doc_name in ("architecture", "readme"):
            doc_text = state.generated_docs.get(doc_name)
            if doc_text:
                items.append(ReviewItem(doc_name=doc_name, doc_text=doc_text, grounding=digest_text))

    api_doc = state.generated_docs.get("api")
    if api_doc:
        candidates = detect_endpoints(state.parse_results)
        if candidates:
            items.append(ReviewItem(doc_name="api", doc_text=api_doc, grounding=build_endpoint_digest(candidates)))

    return items


async def review_documents(items: list[ReviewItem], provider: BaseProvider, *, max_concurrency: int) -> ReviewCollection:
    """Review every item in `items`, bounded to `max_concurrency` concurrent provider calls."""

    async def _review_one(item: ReviewItem) -> ReviewResult:
        prompt = render_prompt("review", doc_name=item.doc_name, doc_text=item.doc_text, grounding=item.grounding)
        messages = [ChatMessage(role=Role.USER, content=prompt)]

        try:
            async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
                with attempt:
                    # 1000, not 400: grounding now includes the full per-file digest (see
                    # build_review_items), so a thorough review can legitimately surface more
                    # issues than the old repo_summary-only grounding did. 400 was cutting
                    # real bullets off mid-word once there was more to report.
                    result = await provider.generate(messages, temperature=0.0, max_tokens=1000)
        except ProviderError as exc:
            return ReviewResult(doc_name=item.doc_name, passed=False, model=provider.model_info().model, error=str(exc))

        passed, issues = parse_review_response(result.text)
        return ReviewResult(doc_name=item.doc_name, passed=passed, issues=issues, model=result.model, raw_response=result.text.strip())

    results = await gather_with_concurrency(max_concurrency, *(_review_one(item) for item in items))
    return ReviewCollection(results=list(results))


def _issue_key_words(issue: str) -> set[str]:
    """Lowercased, punctuation-stripped word set for one issue line, used to compare issues for near-duplication. Deliberately crude (no stemming, no stopword list) — file paths and identifiers, which is what actually distinguishes two issues about different parts of the codebase, survive this intact; grammatical connectors are what fall away, which is exactly the noise that makes two restatements of the same point look different at the string level."""
    return set(re.findall(r"[a-z0-9_./]+", issue.lower()))


def _is_near_duplicate(candidate: set[str], kept: list[set[str]], *, threshold: float = 0.55) -> bool:
    """True if `candidate`'s word set overlaps any already-kept issue's word set by at least `threshold`, using the overlap coefficient (`|A∩B| / min(|A|,|B|)`) rather than Jaccard. Two issues restating the same claim are often very different lengths — one terse, one padded out with a long enumerated list of file paths from the grounding context — and Jaccard's union-based denominator undercounts similarity in exactly that case (a long list of extra file paths dilutes it even though the shorter bullet's entire content is contained in the longer one). The overlap coefficient asks "how much of the smaller bullet is contained in the larger one," which is the right question for "is this a restatement," and isn't diluted by one side simply being longer."""
    for other in kept:
        union = candidate | other
        if not union:
            continue
        smaller = min(len(candidate), len(other))
        if smaller and len(candidate & other) / smaller >= threshold:
            return True
    return False


def parse_review_response(text: str) -> tuple[bool, list[str]]:
    """
    Parse a review response into `(passed, issues)`. Lenient by design:
    a response with no recognizable `STATUS:` line defaults to
    `passed=True` rather than guessing — see this module's docstring.

    Deduplicates and caps issue lines as a deterministic backstop: prompt-
    level instructions not to repeat or over-list have proven unreliable
    on their own (same reasoning as `_apply_deterministic_checks` above)
    — a model can restate the same underlying issue several times with
    different wording, which both pads the report and, since responses
    are token-capped, can crowd out genuinely distinct issues or
    truncate the response mid-sentence. Exact-after-normalization
    catches verbatim repeats; `_is_near_duplicate` additionally catches
    restatements that differ in phrasing but share most of their
    concrete content (see its docstring). The hard cap at
    `_MAX_ISSUES_PER_DOC` is a separate, final backstop against a
    genuinely long list of *distinct* issues overrunning the token
    budget — ranking is left to the model's ordering (it's asked to
    lead with the most significant issues), so capping the list keeps
    the top-ranked ones.
    """
    lines = text.strip().splitlines()
    status_line = next((line for line in lines if line.strip().upper().startswith("STATUS:")), None)

    passed = not (status_line and "ISSUES" in status_line.upper())

    issues: list[str] = []
    seen_exact: set[str] = set()
    seen_words: list[set[str]] = []
    for line in lines:
        if len(issues) >= _MAX_ISSUES_PER_DOC:
            break
        if not line.strip().startswith("-"):
            continue
        issue = line.strip().lstrip("-").strip()
        exact_key = " ".join(issue.lower().split())
        if exact_key in seen_exact:
            continue
        words = _issue_key_words(issue)
        if _is_near_duplicate(words, seen_words):
            continue
        seen_exact.add(exact_key)
        seen_words.append(words)
        issues.append(issue)


    if not passed and not issues:
        issues = ["Review flagged issues but did not list specifics."]

    return passed, issues