"""
Changelog Agent — deterministically renders `state.generated_docs["changelog"]`:
a "Recent Changes" section built from the target repository's real `git log`.

Deliberately *not* an LLM call, for the same reason as the Structure Agent
(agents/structure_agent.py): commit history is already exact, verified data
— a hash, an author date, a commit subject someone actually wrote. There's
nothing here for a model to add, and real risk in letting one paraphrase
or summarize commit messages, since that's exactly the kind of "confident
but unverifiable" rewrite this pipeline is built to avoid everywhere else.

If the target repository isn't a git repository, or has no commits yet, or
`git` isn't available at all, this agent skips cleanly (no error recorded,
no changelog produced) rather than treating any of that as a pipeline
failure — plenty of legitimate repositories to document aren't (yet, or
ever) under git.
"""

from __future__ import annotations

import re
import subprocess

from config.settings import get_settings
from graph.state import RepositoryState
from utils.logger import get_logger

log = get_logger(__name__)

# %h  = abbreviated commit hash
# %ad = author date (formatted below via --date=short -> YYYY-MM-DD)
# %s  = subject line (first line of the commit message only — never the
#       full body, which could contain arbitrary, unverified free text)
_LOG_FORMAT = "%h\x1f%ad\x1f%s"
_FIELD_SEP = "\x1f"


def changelog_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["changelog"]`."""
    settings = get_settings()
    entries = _read_git_log(
        state.repository_path,
        max_commits=settings.changelog_max_commits,
        timeout_seconds=settings.changelog_git_timeout_seconds,
    )

    if entries is None:
        log.info("Changelog Agent: repository is not a git repository (or has no commits), skipping")
        return {"generated_docs": state.generated_docs}

    if not entries:
        log.info("Changelog Agent: git repository has no commits yet, skipping")
        return {"generated_docs": state.generated_docs}

    lines = [
        "# Recent Changes",
        "",
        f"The {len(entries)} most recent commits, from `git log` — every entry below is a real "
        "commit that actually exists in this repository's history, not a summary.",
        "",
    ]
    for commit_hash, date, subject in entries:
        stat = _commit_file_count(state.repository_path, commit_hash, timeout_seconds=settings.changelog_git_timeout_seconds)
        suffix = f" ({stat} file{'s' if stat != 1 else ''} changed)" if stat is not None else ""
        lines.append(f"- `{commit_hash}` ({date}) {subject}{suffix}")

    state.generated_docs["changelog"] = "\n".join(lines).rstrip() + "\n"
    log.info("Changelog Agent: %d commit(s) from git log", len(entries))
    return {"generated_docs": state.generated_docs}


def _commit_file_count(repository_path, commit_hash: str, *, timeout_seconds: float) -> int | None:
    """
    Return the number of files a single commit touched, or `None` if it
    can't be determined. Exact data from `git show --stat`, same
    reasoning as `_read_git_log`'s docstring: a file count is a fact
    about the commit, not a rewrite of it, so it carries no risk of the
    "confident but unverifiable" content this pipeline avoids elsewhere.

    This exists because commit subjects are frequently low-signal on
    their own ("Changes", "Updates", two separate commits both titled
    "first commit") — a file count doesn't explain *what* changed, but
    it's real, verified context that at least distinguishes commits from
    each other and hints at scope, without inventing a description the
    subject itself doesn't provide.
    """
    try:
        result = subprocess.run(
            ["git", "show", "--stat", "--format=", commit_hash],
            cwd=repository_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    # The stat summary is the last non-empty line, e.g.
    # " 3 files changed, 42 insertions(+), 7 deletions(-)" or, for a
    # single-file commit, " 1 file changed, 2 insertions(+)".
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    match = re.search(r"(\d+) files? changed", lines[-1])
    return int(match.group(1)) if match else None


def _read_git_log(
    repository_path, *, max_commits: int, timeout_seconds: float
) -> list[tuple[str, str, str]] | None:
    """
    Return `(hash, date, subject)` for up to `max_commits` recent commits,
    or `None` if `repository_path` isn't a git repository / git isn't
    available at all. Returns an empty list (not `None`) for a valid git
    repository that simply has zero commits yet.

    Every subprocess call here is a fixed argument list (never `shell=True`,
    never string-interpolated into a shell command), so there's no command
    injection surface regardless of what `repository_path` contains.
    """
    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repository_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.info("Changelog Agent: git not available or check failed (%s)", exc)
        return None

    if check.returncode != 0 or check.stdout.strip() != "true":
        return None

    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}", "--date=short", f"--pretty=format:{_LOG_FORMAT}"],
            cwd=repository_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.info("Changelog Agent: git log failed (%s)", exc)
        return None

    if result.returncode != 0:
        # A git repo with zero commits yet returns non-zero here (no HEAD) —
        # that's a legitimate, non-error state, not a failure to log loudly.
        return []

    entries: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(_FIELD_SEP, maxsplit=2)
        if len(parts) == 3:
            entries.append((parts[0], parts[1], parts[2]))
    return entries