"""
Deterministic backstop for a small, high-precision subset of the
ordering/timing language `prompts/readme.md`'s rule 1 forbids.

Same motivation as `utils/banned_phrases.py` (see its docstring): a
"don't say X" prompt instruction competes against the same model's own
generation, with nothing checking the output afterward, and in practice
gets ignored often enough to matter — including, concretely, surviving
multiple critique/revise rounds in a single run (a document reaching
`"Parallel to structural analysis..."` and `"...simultaneously ingesting
code..."` in its intro paragraph, unflagged, after going through the
critique loop's `ordering`/`sequence` check more than once).

This is scoped narrower than rule 1's full word list on purpose. Rule 1
also forbids "before," "after," "then," and "once X completes" — those
are deliberately *not* included here, because they're common enough in
ordinary, non-temporal-claim English ("the config format changed after
version 2," "validates the schema, then normalizes it" as a description
of code structure rather than runtime timing) that a blind substring
match would flag legitimate sentences constantly, unlike the words below.
"simultaneously," "concurrently," "in parallel," and "meanwhile" are
different: outside of describing runtime concurrency, they essentially
never come up, so matching them is high-precision the same way matching
"seamlessly" is in `banned_phrases.py` — a word that's virtually always
the violation whenever it appears at all. "parallel to" (the sentence-
construction "Parallel to X, the system does Y") is included alongside
"in parallel" for the same reason, but a bare adjective "parallel" is
deliberately excluded — "a parallel implementation" can genuinely mean
"an alternative/similar one," not a concurrency claim, so that's exactly
the ambiguous case this module stays out of. The higher-recall, harder-
to-regex-safely cases stay the critique LLM's job alone.

One more wrinkle this module has to handle: `prompts/readme.md` now
explicitly *permits* describing `graph/stages.py`'s real concurrent
fan-out group as concurrent (it's verified ground truth, not an
invented claim — see that prompt's input section), so a bare word match
can no longer mean "always wrong." Observed in practice: a sentence
correctly saying the real fan-out group's agents "run concurrently" got
mechanically stripped by this module same as an actually-invented claim
would, and the model's next attempt reached for an unlisted synonym
("parallelism") to keep expressing the same true fact — a whack-a-mole
loss, not a fix. `known_concurrent_groups`, when passed, lets a match
survive if the text around it names two or more members of the same
real concurrent group, on the theory that a sentence naming several of
those specific things together is almost certainly describing that real
group rather than inventing an unrelated one.
"""

from __future__ import annotations

import re

_ORDERING_WORDS = [
    r"simultaneously",
    r"concurrently",
    r"in parallel",
    r"parallel to",
    r"meanwhile",
]

_COMPILED = [re.compile(rf"\b{pattern}\b", re.IGNORECASE) for pattern in _ORDERING_WORDS]


def find_ordering_language(
    text: str,
    *,
    known_concurrent_groups: list[set[str]] | None = None,
    context_window: int = 200,
) -> list[str]:
    """
    Scan `text` and return the exact matched substrings (preserving
    original casing) for every high-precision ordering/timing word hit,
    in order of appearance. Empty list means clean.

    Deliberately just a detector, same reasoning as
    `banned_phrases.find_banned_phrases`: a matched word gets handed to
    `agents/readme_agent.py._fix_ordering_language`, an isolated
    micro-fix pass, rather than fixed here directly — regex-deleting a
    word like "simultaneously" out of a sentence usually leaves
    grammatically broken text behind, which is worse than the word
    itself.

    `known_concurrent_groups` (typically `graph.stages.concurrent_stage_groups()`)
    lets a match through uncounted if two or more members of the same
    real concurrent group are named within `context_window` characters
    of it — see this module's docstring for why a bare word match can no
    longer safely mean "always wrong" now that describing a real
    concurrent group is explicitly permitted. Matching is a simple
    case-insensitive substring check per group member, not word-boundary
    matched, so short stage names can in principle match inside an
    unrelated word — the failure mode that risks is a missed real
    violation, not a wrongly-stripped true claim, which is the direction
    this module should err given what's already gone wrong the other way.
    """
    matches: list[str] = []
    for pattern in _COMPILED:
        for match in pattern.finditer(text):
            if known_concurrent_groups and _names_a_real_concurrent_group(
                text, match.start(), match.end(), known_concurrent_groups, context_window
            ):
                continue
            matches.append(match.group(0))
    return matches


def _names_a_real_concurrent_group(
    text: str, start: int, end: int, groups: list[set[str]], window: int
) -> bool:
    context = text[max(0, start - window) : end + window].lower()
    return any(sum(1 for name in group if name.lower() in context) >= 2 for group in groups)