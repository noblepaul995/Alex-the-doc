"""
Shared deduplication/capping for LLM-generated bullet lists — extracted
from `agents/review_agent.py`'s `parse_review_response`, which needed
this first (see git history), so that `agents/readme_agent.py`'s
critique loop can get the identical protection instead of a second,
independently-drifting copy of the same logic.

Both callers hit the same underlying failure mode: an LLM asked to list
"the problems you find" has no self-limiting instinct on its own —
across a critique/revise loop, both `review_agent.py`'s review calls and
`readme_agent.py`'s critique calls have been observed restating the same
one or two points ten-plus times in a single response before running out
of room, which both wastes the token budget that could've gone to
genuinely distinct issues and inflates the downstream fix pass with
noise. See `parse_review_response` in `review_agent.py` for the original
diagnosis; this module is the fix, made reusable.
"""

from __future__ import annotations

import re


def _issue_key_words(issue: str) -> set[str]:
    """Lowercased, punctuation-stripped word set for one issue line, used to compare issues for near-duplication. Deliberately crude (no stemming, no stopword list) — file paths and identifiers, which is what actually distinguishes two issues about different parts of the codebase, survive this intact; grammatical connectors are what fall away, which is exactly the noise that makes two restatements of the same point look different at the string level."""
    return set(re.findall(r"[a-z0-9_./]+", issue.lower()))


def _is_near_duplicate(candidate: set[str], kept: list[set[str]], *, threshold: float) -> bool:
    """True if `candidate`'s word set overlaps any already-kept issue's word set by at least `threshold`, using the overlap coefficient (`|A∩B| / min(|A|,|B|)`) rather than Jaccard. Two issues restating the same claim are often very different lengths — one terse, one padded out with a long enumerated list of file paths or details — and Jaccard's union-based denominator undercounts similarity in exactly that case. The overlap coefficient asks "how much of the smaller bullet is contained in the larger one," which is the right question for "is this a restatement," and isn't diluted by one side simply being longer."""
    for other in kept:
        union = candidate | other
        if not union:
            continue
        smaller = min(len(candidate), len(other))
        if smaller and len(candidate & other) / smaller >= threshold:
            return True
    return False


def dedupe_and_cap_bullets(text: str, *, max_bullets: int, near_dup_threshold: float = 0.55) -> list[str]:
    """
    Extract `-`-prefixed bullet lines from `text`, dropping exact and
    near-duplicate restatements and capping the result at `max_bullets`,
    keeping the model's own ordering (so a caller that asks the model to
    lead with its most significant findings keeps the top-ranked ones
    when the cap trims the list). Non-bullet lines (e.g. a leading
    `STATUS: ...` marker) are ignored — callers that need one should
    read it from the original `text` separately before or after calling
    this.
    """
    bullets: list[str] = []
    seen_exact: set[str] = set()
    seen_words: list[set[str]] = []

    for line in text.strip().splitlines():
        if len(bullets) >= max_bullets:
            break
        if not line.strip().startswith("-"):
            continue
        bullet = line.strip().lstrip("-").strip()
        exact_key = " ".join(bullet.lower().split())
        if exact_key in seen_exact:
            continue
        words = _issue_key_words(bullet)
        if _is_near_duplicate(words, seen_words, threshold=near_dup_threshold):
            continue
        seen_exact.add(exact_key)
        seen_words.append(words)
        bullets.append(bullet)

    return bullets