"""
Deterministic backstop for one specific structural failure mode:
`prompts/readme.md`'s "What this document is (and isn't)" section
explicitly tells the model not to walk through most of `file_digest`,
naming most of its files individually — a real, observed failure mode
(a README that reads as a file-by-file inventory rather than a
synthesized overview) that survived that prompt instruction on its own
across more than one run, whether the model expressed it as one bullet
per file, several files crammed into one bullet's bold list, or —
observed later, after the first two were caught — the same
near-total enumeration moved into flowing prose paragraphs instead of
bullets at all (`repo_agent.py acts as a central coordinator... The
review_agent.py imports utils/banned_phrases.py and
utils/file_references.py...`, semicolon after semicolon, one clause per
file). That last one is why this module counts path-like backtick
tokens across the *entire* document body, not just Markdown list-item
lines — an earlier, list-scoped version of this check was evaded
exactly that way: the underlying enumeration problem was completely
unchanged, but reformatting it as prose was enough to make the narrower
check report a false 0. Same reasoning as `utils/ordering_language.py`
and `utils/banned_phrases.py` generally: a "don't do X" instruction
competes against the same model's own generation with nothing checking
the result afterward, and in practice doesn't reliably hold on its own
— and a check narrow enough to have a blind spot is close to as
unreliable as no check, once something is actually looking for the gap.

Unlike those two modules, this isn't fixable with an isolated
sentence-level rewrite — consolidating a dozen individually-named files
into two or three grouped points is a restructuring, not a word swap. So
this doesn't get its own `_fix_*` pass; instead `check_file_enumeration`'s
finding is merged into the normal critique/revise loop in
`agents/readme_agent.py`, forcing a revision round even when the model's
own critique pass came back clean, since only a full revise call has
the room to actually restructure a section.
"""

from __future__ import annotations

import re

_BACKTICK_TOKEN = re.compile(r"`([^`\n]+)`")

_PATH_LIKE_EXTENSIONS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
)


def _looks_like_file_path(token: str) -> bool:
    """True if `token` (already stripped of its surrounding backticks) looks like a real file path rather than some other backtick-quoted term the document might legitimately contain (a class name, a CLI flag, a constant)."""
    return "/" in token or token.endswith(_PATH_LIKE_EXTENSIONS)


def count_individual_file_bullets(text: str) -> int:
    """
    Count distinct path-like backtick-quoted tokens appearing anywhere
    in `text` — not scoped to Markdown list items, and deliberately not
    restricted to "one path immediately after a bullet marker" either.
    Both narrower versions of this check were tried and both were
    evaded by a reformatting that changed nothing about the underlying
    problem: many files crammed into one bullet's bold list, and — the
    version that motivated dropping the list-item restriction entirely
    — the same enumeration moved into flowing prose. Counting distinct
    paths across the whole document, regardless of surrounding
    formatting, means there's no format left to hide the count behind.
    Counting distinct paths rather than raw occurrences means the same
    file mentioned in two different places (e.g. once in a components
    section, once in a design-pattern paragraph) is counted once, not
    twice — repetition isn't what this check is measuring.
    """
    return len({token for token in _BACKTICK_TOKEN.findall(text) if _looks_like_file_path(token)})


def check_file_enumeration(text: str, digest_file_count: int, *, threshold: float = 0.4) -> str | None:
    """
    Return a critique-style finding string if `text` names more than
    `threshold` of the files actually in the digest as individually
    called-out files anywhere in the document, or `None` if it's
    comfortably below that (the normal case for a document that's
    actually synthesizing rather than enumerating). `digest_file_count`
    should be the number of files in the specific `file_digest` this
    document was drafted from (e.g. `len(digest.file_digest.splitlines())`),
    not a repository-wide file count — the digest is usually a curated
    subset, and that subset is the right base rate to compare against.
    """
    if digest_file_count <= 0:
        return None

    named_count = count_individual_file_bullets(text)
    ratio = named_count / digest_file_count
    if ratio <= threshold:
        return None

    return (
        f'- [structure] The document individually names {named_count} of the {digest_file_count} files in the '
        f"digest ({ratio:.0%}) — whether as one bullet per file, several files crammed into one bullet's bold "
        "list, or the same thing said as flowing prose instead of bullets, this reads as a file-by-file "
        "inventory, which is what the separate structure/file reference document is already for, not what "
        "this document is for. Consolidate: group files that share a role or subsystem into one bullet or "
        "short paragraph that describes the group's purpose, rather than calling out most of the individual "
        "files by name — reformatting the enumeration into a different shape (bullets into prose, or vice "
        "versa) without actually reducing how many files are individually named does not fix this. This is a "
        "restructuring, not a wording or formatting fix — some sections may need to be reorganized, not just "
        "have individual sentences edited."
    )