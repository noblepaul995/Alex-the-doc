"""
Deterministic file-existence checking for generated documentation.

Every prompt in this pipeline that names a specific file (chunk.md,
file.md, readme.md, architecture.md, structure_agent.py) asks for it in
backtick-quoted form, e.g. `` `agents/repo_agent.py` ``. That convention is
what makes this check possible: rather than trying to parse arbitrary
prose for anything that might be a path (extremely false-positive-prone —
"the config module" isn't a path, "config.py" is), this only inspects
backtick-quoted spans that already look like a path, and checks each one
against the real, scanned file list.

Same reasoning as `utils/banned_phrases.py`: this is a plain, deterministic
check, not a model judgment call, so it can't be talked out of flagging a
genuine mismatch the way an LLM review pass sometimes is.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

_BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")

# Common source/config/doc extensions worth checking. Deliberately narrow —
# a backtick span like `state.generated_docs` or `ChatMessage` shouldn't be
# treated as a file-path claim just because it's in backticks; only things
# that actually look like a path (contain a slash) or end in one of these
# extensions are candidates.
_PATH_LIKE_EXTENSIONS = (
    ".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".md", ".json",
    ".toml", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".sql",
)

# This pipeline's own generated output filenames — legitimately referenced
# in prose (e.g. the Table of Contents links to them) but never present in
# `state.file_metadata`, since they're outputs of this run, not scanned
# source files. Excluded from checking rather than always-flagged.
_KNOWN_OUTPUT_FILENAMES = frozenset(
    {"README.md", "ARCHITECTURE.md", "API.md", "STRUCTURE.md", "CHANGELOG.md", "DEPENDENCIES.md", "githubReadme.md", "REVIEW.md"}
)


def _looks_like_file_path(token: str) -> bool:
    """
    A backtick-quoted token is worth checking if it ends in a recognized file
    extension, or ends with a trailing slash (a directory reference, e.g.
    from the README's `Location:` lines).

    Deliberately does NOT treat "contains a slash" alone as sufficient —
    that's too loose: `/v1/messages` (an API endpoint path mentioned in a
    file summary) or similar URL-shaped fragments also contain a slash
    without being a repository file/directory reference at all, and would
    otherwise get flagged as "nonexistent" even though they were never
    claiming to be a real path in the first place.
    """
    token = token.strip()
    if not token or " " in token or "(" in token or "`" in token:
        return False  # code snippets, function calls, prose fragments — not a bare path
    if token.endswith("/"):
        return True  # directory reference
    return token.lower().endswith(_PATH_LIKE_EXTENSIONS)


def find_nonexistent_file_references(
    text: str, real_paths: set[str], *, repository_root: Path | None = None
) -> list[str]:
    """
    Scan `text` for backtick-quoted, path-like tokens and return every one
    that doesn't correspond to something real, checked two ways:

      - A **file** reference (no trailing slash) is checked against
        `real_paths` (a set of scanned relative paths, e.g. from
        `state.file_metadata`) — either as a full relative path, or, for a
        bare filename with no slash, against any real file's basename (a
        doc can legitimately say `` `parser.py` `` without the full
        `agents/parser.py` path).

      - A **directory** reference (trailing slash, e.g. from README's
        `Location:` lines) is checked against the real filesystem via
        `repository_root`, not just `real_paths`. A directory can be real
        and legitimately mentioned (e.g. `prompts/`) even if nothing under
        it was scanned as a documented source file — `.md` templates
        aren't a parsed source language, but the directory still exists.
        If `repository_root` isn't given, directory references are skipped
        entirely rather than guessed at (never flag without being able to
        actually check).

    Returns matches in order of appearance, without duplicates.
    """
    real_basenames = {PurePosixPath(p).name for p in real_paths}
    seen: set[str] = set()
    missing: list[str] = []

    for match in _BACKTICK_SPAN.finditer(text):
        token = match.group(1).strip().lstrip("./")
        if not _looks_like_file_path(token) or token in seen:
            continue
        seen.add(token)

        if token in _KNOWN_OUTPUT_FILENAMES:
            continue

        if token.endswith("/"):
            if repository_root is None:
                continue  # can't verify a directory claim without the real filesystem — don't guess
            if (repository_root / token.rstrip("/")).is_dir():
                continue
            missing.append(token)
            continue

        if token in real_paths:
            continue
        if "/" not in token and token in real_basenames:
            continue

        missing.append(token)

    return missing


def render_missing_file_findings(missing: list[str]) -> str:
    """Format `find_nonexistent_file_references()` output as review-style issue lines."""
    return "\n".join(f'References a file that doesn\'t exist in this repository: "{path}"' for path in missing)