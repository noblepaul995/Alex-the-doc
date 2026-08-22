"""
Deterministic banned-phrase detection for generated documentation.

Every prompt in this pipeline that generates prose (chunk.md, file.md,
readme.md, architecture.md) already tells the model not to use generic
justificatory filler — "promotes flexibility," "ensures maintainability,"
and similar stock phrases that sound informative but state nothing
specific. In practice, across repeated runs, that "don't say X" prompt
instruction gets ignored more often than not: it's competing against the
same model's default reach for wrap-up language, in the same generation,
with nothing checking the output afterward.

So detection here is a plain, case-insensitive substring match — no model
judgment call involved, which means it can't be "convinced" to miss an
occurrence the way a critique LLM call can. This is deliberately a
*detector*, not a *fixer*: matched phrases get handed to the existing
critique/revise loop (README) or a dedicated fix pass (Architecture) so an
LLM does the actual rewrite — regex-deleting "ensures maintainability"
from a sentence usually leaves grammatically broken text behind, which is
worse than the phrase itself.
"""

from __future__ import annotations

import re

# Verbs and abstract nouns that combine into the stock phrases we want to
# catch — built as a single pattern (rather than one entry per combination)
# so a compound like "promotes flexibility and maintainability" is caught as
# one match, not silently missed because only "promotes flexibility" alone
# was listed.
_VERBS = [r"promotes?", r"promoting", r"ensures?", r"ensuring", r"improves?", r"improving", r"enhances?", r"enhancing"]
_ABSTRACT_NOUNS = [r"flexibility", r"maintainability", r"scalability", r"reliability", r"efficiency", r"consistency"]

_verb_noun_pattern = (
    rf"(?:{'|'.join(_VERBS)})\s+(?:{'|'.join(_ABSTRACT_NOUNS)})"
    rf"(?:\s*,?\s*(?:and\s+)?(?:{'|'.join(_ABSTRACT_NOUNS)}))*"
)

_BANNED_PHRASES: list[str] = [
    _verb_noun_pattern,
    r"significantly improves? performance",
    r"robust and (?:reliable|scalable|maintainable)",
    r"seamless(?:ly)?",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _BANNED_PHRASES]


def find_banned_phrases(text: str) -> list[str]:
    """
    Scan `text` and return the exact matched substrings (as they actually
    appear, preserving original casing/inflection) for every banned-phrase
    hit, in order of appearance. Empty list means clean.
    """
    matches: list[str] = []
    for pattern in _COMPILED:
        for match in pattern.finditer(text):
            matches.append(match.group(0))
    return matches


def render_banned_phrase_findings(matches: list[str]) -> str:
    """
    Format `find_banned_phrases()` output as critique-style bullet lines,
    suitable for merging directly into an LLM critique's findings before a
    revise pass — same shape as the model's own flagged-sentence bullets,
    so the revise prompt doesn't need to know these came from a different
    source.
    """
    unique = sorted(set(matches), key=str.lower)
    return "\n".join(f'- Generic stock phrase, restate concretely or remove: "{m}"' for m in unique)
