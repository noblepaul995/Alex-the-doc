"""
README Agent — drafts a project README, then fact-checks and corrects its
own draft before returning it.

Reuses `agents.repo_agent.build_digest` rather than re-deriving the
same "which files matter most" logic a second time — the README's
grounding needs are the same as the Repository Agent's (language
breakdown, most depended-upon files), so building a second, subtly
different digest would just be duplicated logic with a chance to drift
out of sync.

The prompt is deliberately conservative about anything this pipeline
can't verify: installation commands, exact CLI flags, or usage examples
aren't things any implemented stage extracts (no stage parses
`pyproject.toml` entry points, `package.json` scripts, or argument
parsers into structured data), so the prompt explicitly tells the model
not to invent them. A generated README that confidently states a wrong
`pip install` command is worse than one that's honestly vague about
setup — this stage would rather under-claim than hallucinate commands
a user might actually run.

**Draft, then loop critique/revise up to `MAX_CRITIQUE_ROUNDS` times, not
one shot.** A single-pass draft has no mechanism to catch its own invented
claims — a "don't invent things" instruction in the draft prompt is
necessarily competing with the same model's urge to write something that
reads well, in the same generation, with nothing checking the output
afterward. So this stage runs a loop instead:

  1. **Draft** — one shot at a full README, grounded in the repo summary,
     per-file digest, the real import relationships computed from the
     actual dependency graph, and the real, verified pipeline execution
     order (`graph.stages.STAGE_ORDER` — literal fact about which
     stages run before/after/concurrently with which, not inferred).
  2. **Critique** — a separate call re-reads the current draft against
     only that verified grounding material and flags three specific
     things: invented execution-order/parallelism/timing claims,
     *implied* sequencing (the order components are discussed in prose,
     even without explicit "before/after" wording, checked against the
     real stage order) that contradicts reality, and behavioral claims
     that don't trace back to any file's summary. This call never sees
     its own output being graded — it's strictly draft-in, findings-out.
  3. **Revise** — only runs if the critique found something. Takes the
     current draft plus the exact flagged sentences and is instructed to
     fix only those, leaving everything else untouched, rather than
     regenerating the whole document (which would risk introducing new
     unverified claims of its own).

Steps 2-3 repeat, feeding each revision back into the next critique round,
until a critique pass comes back clean or `MAX_CRITIQUE_ROUNDS` is hit —
whichever comes first. Most runs should exit after one round (draft,
critique, done) or two (draft, critique, revise, critique-clean); the cap
exists so a model that can't converge doesn't loop indefinitely, not
because three rounds is the expected case.

This adds real token cost to README generation (worst case, `1 +
2 * MAX_CRITIQUE_ROUNDS` calls), but README is one of a handful of
one of a handful of whole-repository calls in the entire pipeline (unlike
`chunk_doc`/`file_doc`, which run once per chunk/file) — the cost is
negligible next to what it buys in caught invention.
"""

from __future__ import annotations

from agents.repo_agent import RepoDigest, build_digest
from config.prompts import render_prompt
from config.providers import resolve_synthesis_provider_config
from config.settings import get_settings
from graph.stages import concurrent_stage_groups, render_stage_order
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from utils.banned_phrases import find_banned_phrases
from utils.bullet_dedup import dedupe_and_cap_bullets
from utils.helpers import with_retry
from utils.license_detect import detect_license
from utils.logger import get_logger
from utils.structure_check import check_file_enumeration, count_individual_file_bullets
from utils.ordering_language import find_ordering_language
from utils.timers import Stopwatch

log = get_logger(__name__)

_NO_ISSUES_MARKER = "no issues found"

_MAX_CRITIQUE_ISSUES = 8
"""Cap on distinct issues kept from one critique call — see the dedupe_and_cap_bullets call in generate_readme's critique loop. Higher than review_agent.py's equivalent cap (5) since a critique bullet here is typically a single quoted sentence rather than a fuller explanation, so more of them fit usefully in one revise pass."""


async def readme_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["readme"]`."""
    digest = build_digest(state, max_files=get_settings().readme_digest_max_files)
    if digest is None:
        return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}

    license_info = detect_license(state.repository_path)

    config = resolve_synthesis_provider_config()

    with Stopwatch("readme") as sw:
        async with create_provider(config) as provider:
            try:
                doc, tokens_used, revision_count = await generate_readme(
                    digest, state.repo_summary, provider, license_info=license_info
                )
            except ProviderError as exc:
                state.record_error("readme", f"Failed to generate README: {exc}", exception=exc)
                return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}

    state.statistics.tokens_used += tokens_used
    state.generated_docs["readme"] = doc

    log.info(
        "README Agent: document generated in %.2fs (%d critique/revise round(s) applied)",
        sw.elapsed_seconds,
        revision_count,
    )

    return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}


async def _call(
    provider: BaseProvider, prompt: str, *, temperature: float, max_tokens: int
) -> tuple[str, int | None, int | None, str | None]:
    """One retried generation call. Shared by the draft/critique/revise passes below. Returns `finish_reason` too — see `_warn_if_truncated`, which is what actually uses it."""
    messages = [ChatMessage(role=Role.USER, content=prompt)]
    async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
        with attempt:
            result = await provider.generate(messages, temperature=temperature, max_tokens=max_tokens)
    return result.text.strip(), result.prompt_tokens, result.completion_tokens, result.finish_reason


def _warn_if_truncated(finish_reason: str | None, *, stage: str) -> None:
    """
    Log a clear warning if `finish_reason` indicates the model was cut off
    by `max_tokens` rather than finishing on its own. This is a real,
    provider-reported signal (not a text heuristic guessing from where a
    sentence ends), but the exact string differs per provider —
    `llm/anthropic.py` reports `"max_tokens"`, `llm/openai.py` and
    `llm/gemini.py` report `"length"`/`"MAX_TOKENS"` respectively — hence
    the substring check rather than an exact match. `llm/ollama.py`
    currently never reports a length-cutoff reason at all (see its
    `finish_reason` mapping), so this check is a no-op for Ollama; that's
    a real gap in what this can detect, not a false negative being
    silently swallowed.

    Purely diagnostic for now: a truncated draft/revise output still gets
    returned to the caller either way (this pipeline generally prefers a
    visibly-incomplete result plus a loud log line over silently
    retrying and hoping, which is why `with_retry` above only covers
    `ProviderError` — a cut-off-by-length response isn't an error the
    provider raised, it succeeded at generating less than was asked for).
    """
    if finish_reason and any(marker in finish_reason.lower() for marker in ("length", "max_token")):
        log.warning(
            "README Agent: %s call was cut off by max_tokens (finish_reason=%r) — "
            "the returned text may be incomplete or end mid-sentence",
            stage,
            finish_reason,
        )


MAX_CRITIQUE_ROUNDS = 3


async def generate_readme(
    digest: RepoDigest,
    repo_summary: str | None,
    provider: BaseProvider,
    *,
    license_info: tuple[str | None, str] | None = None,
) -> tuple[str, int, int]:
    """
    Run the draft -> critique -> (conditional) revise chain, looping the
    critique/revise pair up to `MAX_CRITIQUE_ROUNDS` times and stopping as
    soon as a critique pass comes back clean. Raises `ProviderError` after
    retries are exhausted on any individual call.

    `license_info` is `utils.license_detect.detect_license`'s return value
    — real, verified data (or `None`), the same source the GitHub README
    Agent's license badge already uses (see `agents/github_readme_agent.py`).
    Passed through to the draft prompt so the README's own body can state
    a real detected license in a proper `## License` section, rather than
    the license only ever surfacing as a badge on the GitHub-flavored
    wrapper and never in the README text itself.

    Returns (final_document, total_tokens_used, revision_count). A
    `revision_count` of 0 means the first draft passed critique untouched;
    `MAX_CRITIQUE_ROUNDS` means the loop was cut off by the cap, not by a
    clean critique — worth knowing, since it means unresolved issues may
    remain in the returned document.
    """
    total_tokens = 0
    stage_order = render_stage_order()

    if license_info is None:
        license_summary = "(no LICENSE-shaped file or manifest license field found — do not state a license)"
    else:
        spdx_id, filename = license_info
        license_summary = (
            f"{spdx_id}, per `{filename}`"
            if spdx_id is not None
            else f"a LICENSE-shaped file exists (`{filename}`) but its specific type wasn't identified — "
            "you may say a license file exists and point to it by name, but do not name a specific license type"
        )

    draft_prompt = render_prompt(
        "readme",
        repo_summary=repo_summary or "(not yet established)",
        language_breakdown=digest.language_breakdown,
        file_count=digest.file_count,
        file_digest=digest.file_digest,
        dependency_edges=digest.dependency_edges,
        stage_order=stage_order,
        license_summary=license_summary,
    )
    current, prompt_tokens, completion_tokens, finish_reason = await _call(
        provider, draft_prompt, temperature=0.2, max_tokens=4500
    )
    total_tokens += (prompt_tokens or 0) + (completion_tokens or 0)
    _warn_if_truncated(finish_reason, stage="draft")

    current, fix_tokens = await _fix_banned_phrases(current, provider)
    total_tokens += fix_tokens
    current, fix_tokens = await _fix_ordering_language(current, provider)
    total_tokens += fix_tokens

    digest_file_count = len(digest.file_digest.splitlines())
    current, fix_tokens = await _fix_file_enumeration(current, digest_file_count, provider)
    total_tokens += fix_tokens

    revision_count = 0
    for round_number in range(1, MAX_CRITIQUE_ROUNDS + 1):
        critique_prompt = render_prompt(
            "readme_critique",
            repo_summary=repo_summary or "(not yet established)",
            file_digest=digest.file_digest,
            dependency_edges=digest.dependency_edges,
            stage_order=stage_order,
            license_summary=license_summary,
            draft=current,
        )
        critique, prompt_tokens, completion_tokens, finish_reason = await _call(
            provider, critique_prompt, temperature=0.0, max_tokens=1200
        )
        total_tokens += (prompt_tokens or 0) + (completion_tokens or 0)
        _warn_if_truncated(finish_reason, stage="critique")

        llm_found_issues = _NO_ISSUES_MARKER not in critique.strip().lower()[:40]

        # Deterministic structural check, independent of what the LLM's
        # own critique found — see utils/structure_check.py's docstring
        # for why "don't enumerate the file digest" needs this backstop
        # the same way ordering language and banned phrases do. Checked
        # every round against the latest `current`, since a revise pass
        # could in principle reintroduce the pattern while fixing
        # something else.
        structural_finding = check_file_enumeration(current, digest_file_count)

        if not llm_found_issues and structural_finding is None:
            log.info("README Agent: critique round %d/%d found no issues", round_number, MAX_CRITIQUE_ROUNDS)
            break

        # Dedupe/cap before deciding what to do with it: an unbounded
        # critique call has been observed restating the same one or two
        # flagged sentences ten-plus times in a single response (see
        # `utils/bullet_dedup.py`'s docstring) — same failure mode
        # `review_agent.py` needed this fix for first. Rebuilding the
        # critique text from the deduped bullets, rather than passing
        # `critique` through as-is, keeps the noise out of both the log
        # line below and the revise prompt that reads `{critique}` next.
        bullets = dedupe_and_cap_bullets(critique, max_bullets=_MAX_CRITIQUE_ISSUES) if llm_found_issues else []
        if structural_finding is not None:
            # Lead with it: a structural finding calls for reorganizing
            # whole sections, which matters more than any single
            # sentence-level fix and is worth keeping even if the cap
            # would otherwise trim it.
            bullets = [structural_finding.lstrip("- ").strip(), *bullets][:_MAX_CRITIQUE_ISSUES]

        if not bullets:
            log.info("README Agent: critique round %d/%d found no issues", round_number, MAX_CRITIQUE_ROUNDS)
            break
        critique = "\n".join(f"- {b}" for b in bullets)

        log.info(
            "README Agent: critique round %d/%d flagged issue(s), running revision pass:\n%s",
            round_number,
            MAX_CRITIQUE_ROUNDS,
            critique,
        )

        revise_prompt = render_prompt("readme_revise", draft=current, critique=critique)
        current, prompt_tokens, completion_tokens, finish_reason = await _call(
            provider, revise_prompt, temperature=0.2, max_tokens=4500
        )
        total_tokens += (prompt_tokens or 0) + (completion_tokens or 0)
        _warn_if_truncated(finish_reason, stage=f"revise (round {round_number})")
        revision_count += 1

        # Re-run all three dedicated fixers after every revise, in case
        # the revise call reintroduced something one of them catches
        # while fixing something else — cheap no-op scans when the
        # document is already clean of all three.
        current, fix_tokens = await _fix_banned_phrases(current, provider)
        total_tokens += fix_tokens
        current, fix_tokens = await _fix_ordering_language(current, provider)
        total_tokens += fix_tokens
        current, fix_tokens = await _fix_file_enumeration(current, digest_file_count, provider)
        total_tokens += fix_tokens
    else:
        log.info(
            "README Agent: hit the %d-round critique cap without a clean pass; "
            "returning the most recent revision, which may still have unresolved issues",
            MAX_CRITIQUE_ROUNDS,
        )

    return _strip_trailing_dividers(current), total_tokens, revision_count


def _strip_trailing_dividers(text: str) -> str:
    """
    Strip any stray Markdown horizontal rule(s) (`---`, `***`, or `___`,
    the three ways CommonMark spells one) trailing at the very end of the
    document, along with the blank lines around them. Handles more than
    one stacked divider, not just one.

    This exists because both `prompts/readme_critique.md` and
    `prompts/readme_revise.md` used to wrap the input draft in literal
    `---` fences as a plain prompt delimiter (now XML-style `<draft>`
    tags instead, specifically to stop this) — a model asked to "output
    the full corrected README, start to finish" right next to text
    formatted exactly like a Markdown divider had a real chance of
    echoing that formatting back as if it were part of the document.
    Across `MAX_CRITIQUE_ROUNDS` revise rounds, each one echoing its own
    stray divider (and faithfully preserving the prior round's, per the
    "don't touch unflagged content" instruction) stacks into multiple
    dividers by the end — this cleans up the accumulated result
    regardless of how many rounds ran, as a deterministic backstop
    alongside the delimiter fix rather than instead of it, matching how
    every other cross-cutting problem in this pipeline gets both a
    prompt-level fix and a backstop that doesn't depend on the model
    actually following it.
    """
    while True:
        stripped = text.rstrip()
        last_line = stripped.rsplit("\n", 1)[-1].strip()
        if last_line in ("---", "***", "___"):
            text = stripped.rsplit("\n", 1)[0] if "\n" in stripped else ""
        else:
            return stripped + "\n" if stripped else stripped


def _locate_sentence(text: str, phrase: str) -> tuple[int, int, str] | None:
    """Find the rough sentence boundaries around `phrase`'s first occurrence in `text`."""
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return None

    start = 0
    for terminator in (". ", "! ", "? ", "\n"):
        pos = text.rfind(terminator, 0, idx)
        if pos != -1:
            start = max(start, pos + len(terminator))

    end = len(text)
    for terminator in (".", "!", "?"):
        pos = text.find(terminator, idx)
        if pos != -1:
            end = min(end, pos + 1)

    # Fall back to a bare newline as the end boundary only if it comes
    # before any sentence-ending punctuation was found (e.g. bullet points
    # with no terminal punctuation) — the newline itself is never consumed
    # here, so it's preserved as leading whitespace in the caller's `tail`.
    newline_pos = text.find("\n", idx)
    if newline_pos != -1:
        end = min(end, newline_pos)

    return start, end, text[start:end]


async def _fix_banned_phrases(current: str, provider: BaseProvider) -> tuple[str, int]:
    """
    Dedicated, isolated fix pass for banned generic phrases — deliberately
    NOT bundled into the general critique/revise loop above.

    Real evidence from actual runs showed the general revise call, even
    when handed the exact flagged phrase verbatim as part of a larger
    multi-issue critique, failed to remove it: the same two phrases
    ("ensure efficiency and consistency", "seamless") survived unchanged
    across all 3 critique/revise rounds in one observed run. Isolating
    this to "rewrite just this one short sentence, removing just this one
    phrase" is a far smaller, more constrained task than "fix a dozen
    things across the whole document at once" — and if the isolated
    rewrite itself still contains a banned phrase, the original sentence
    is kept rather than risking making things worse, so this can only
    improve the document, never silently corrupt it.
    """
    total_tokens = 0
    seen_phrases: set[str] = set()

    for phrase in find_banned_phrases(current):
        key = phrase.lower()
        if key in seen_phrases:
            continue
        seen_phrases.add(key)

        span = _locate_sentence(current, phrase)
        if span is None:
            continue
        start, end, sentence = span

        fix_prompt = render_prompt("phrase_micro_fix", sentence=sentence, phrase=phrase)
        messages = [ChatMessage(role=Role.USER, content=fix_prompt)]
        try:
            async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
                with attempt:
                    result = await provider.generate(messages, temperature=0.1, max_tokens=150)
        except ProviderError as exc:
            log.info('README Agent: micro-fix for phrase "%s" failed (%s), leaving original sentence', phrase, exc)
            continue

        total_tokens += (result.prompt_tokens or 0) + (result.completion_tokens or 0)
        replacement = result.text.strip()

        if replacement and not find_banned_phrases(replacement):
            tail = current[end:]
            # Preserve whatever whitespace originally followed the sentence
            # (a paragraph break, a newline, etc.) exactly — only insert a
            # bridging space if the tail doesn't already start with
            # whitespace and the replacement doesn't already end with it.
            # Stripping the tail's leading whitespace here previously
            # collapsed paragraph breaks (e.g. before a "## Heading"),
            # gluing sections together on one line.
            separator = "" if not tail or tail[0].isspace() or replacement.endswith((" ", "\n")) else " "
            current = current[:start] + replacement + separator + tail
            log.info('README Agent: micro-fixed banned phrase "%s"', phrase)
        else:
            log.info('README Agent: micro-fix for phrase "%s" still contained banned phrasing, keeping original', phrase)

    return current, total_tokens


async def _fix_ordering_language(current: str, provider: BaseProvider) -> tuple[str, int]:
    """
    Dedicated, isolated fix pass for the small set of high-precision
    ordering/timing words `utils/ordering_language.py` detects — same
    isolated-micro-fix structure as `_fix_banned_phrases` above, and for
    the same reason: a flagged phrase merged into the general critique's
    bullet list isn't reliably acted on, even verbatim, across multiple
    rounds. See `utils/ordering_language.py`'s docstring for the concrete
    run that motivated this, and for why this only covers five words
    ("simultaneously," "concurrently," "in parallel," "parallel to,"
    "meanwhile") rather than rule 1's full list.

    Passes `concurrent_stage_groups()` through so a sentence correctly
    describing the real fan-out group isn't mechanically stripped the
    same way an invented claim would be — see that function's docstring.
    """
    total_tokens = 0
    seen_phrases: set[str] = set()

    for phrase in find_ordering_language(current, known_concurrent_groups=concurrent_stage_groups()):
        key = phrase.lower()
        if key in seen_phrases:
            continue
        seen_phrases.add(key)

        span = _locate_sentence(current, phrase)
        if span is None:
            continue
        start, end, sentence = span

        fix_prompt = render_prompt("ordering_micro_fix", sentence=sentence, phrase=phrase)
        messages = [ChatMessage(role=Role.USER, content=fix_prompt)]
        try:
            async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
                with attempt:
                    result = await provider.generate(messages, temperature=0.1, max_tokens=150)
        except ProviderError as exc:
            log.info('README Agent: micro-fix for ordering word "%s" failed (%s), leaving original sentence', phrase, exc)
            continue

        total_tokens += (result.prompt_tokens or 0) + (result.completion_tokens or 0)
        replacement = result.text.strip()

        if replacement and not find_ordering_language(replacement, known_concurrent_groups=concurrent_stage_groups()):
            tail = current[end:]
            separator = "" if not tail or tail[0].isspace() or replacement.endswith((" ", "\n")) else " "
            current = current[:start] + replacement + separator + tail
            log.info('README Agent: micro-fixed ordering word "%s"', phrase)
        else:
            log.info('README Agent: micro-fix for ordering word "%s" still contained it, keeping original', phrase)

    return current, total_tokens


async def _fix_file_enumeration(current: str, digest_file_count: int, provider: BaseProvider) -> tuple[str, int]:
    """
    Dedicated, isolated fix pass for `utils/structure_check.py`'s
    file-enumeration finding — unlike `_fix_banned_phrases` and
    `_fix_ordering_language` above, this isn't a single-sentence
    micro-fix (consolidating a dozen file-by-file bullets into a few
    grouped ones is a restructuring, not a word swap), so it rewrites
    the whole document via `prompts/readme_structure_fix.md` rather than
    one located sentence.

    Isolated for the same reason as the other two, with direct evidence
    for this specific case: merged into the general critique/revise
    loop's bullet list, this finding was observed persisting completely
    unchanged (identical file count, identical percentage) across all
    `MAX_CRITIQUE_ROUNDS` rounds of a real run — the model was
    consistently under-investing effort in the one genuinely hard,
    time-consuming item on a list otherwise made of small one-sentence
    fixes. Giving it a call with no competing smaller asks is the fix.

    Runs once per call (not a retry-until-clean loop): if the rewrite
    doesn't fully resolve it, the in-loop check in `generate_readme`'s
    critique loop still catches whatever's left as a backstop, the same
    role it already played before this dedicated pass existed.
    """
    finding = check_file_enumeration(current, digest_file_count)
    if finding is None:
        return current, 0

    before_count = count_individual_file_bullets(current)
    ratio = before_count / digest_file_count if digest_file_count else 0.0

    fix_prompt = render_prompt(
        "readme_structure_fix",
        draft=current,
        named_count=before_count,
        digest_file_count=digest_file_count,
        ratio=f"{ratio:.0%}",
    )
    messages = [ChatMessage(role=Role.USER, content=fix_prompt)]
    try:
        async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
            with attempt:
                result = await provider.generate(messages, temperature=0.2, max_tokens=4500)
    except ProviderError as exc:
        log.info("README Agent: structural consolidation pass failed (%s), leaving draft as-is", exc)
        return current, 0

    total_tokens = (result.prompt_tokens or 0) + (result.completion_tokens or 0)
    rewritten = result.text.strip()
    if not rewritten:
        log.info("README Agent: structural consolidation pass returned nothing usable, leaving draft as-is")
        return current, total_tokens

    after_count = count_individual_file_bullets(rewritten)
    log.info(
        "README Agent: structural consolidation pass: %d -> %d individually-named files (of %d in digest)",
        before_count,
        after_count,
        digest_file_count,
    )
    return rewritten, total_tokens