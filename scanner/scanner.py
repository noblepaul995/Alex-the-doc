"""
The Scanner Agent's core logic: walk a repository and produce a
`ProjectFile` for every file worth documenting.

Responsibilities (per the project spec):
    - walk repository
    - obey .gitignore
    - ignore binaries
    - detect language
    - compute SHA256
    - collect metadata

Change detection ("detect changes") is intentionally a separate step —
see `scanner/hashes.py:diff_hashes` — since it needs a previous run's
hashes to compare against, which don't exist until the Memory System
stage is built. This module always returns the *complete* current state
of the repository; callers decide what's "new" relative to that.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from graph.state import ProjectFile
from scanner.hashes import compute_file_hash
from scanner.ignore import IgnoreRules, is_binary, is_secret_file
from scanner.language import detect_language
from utils.logger import get_logger

log = get_logger(__name__)

_PEEK_SIZE = 8192


class ScanResult(BaseModel):
    """Everything the Scanner Agent produced from a single repository walk."""

    files: list[ProjectFile] = []
    skipped_binary: int = 0
    skipped_ignored: int = 0
    skipped_secret: int = 0
    errors: list[str] = []

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def by_language(self) -> dict[str, int]:
        """Count of files per detected language (None grouped as 'unknown')."""
        counts: dict[str, int] = {}
        for f in self.files:
            key = f.language or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts


def scan_repository(root: Path, *, extra_ignore_patterns: list[str] | None = None) -> ScanResult:
    """
    Walk `root`, applying ignore rules, and return a `ScanResult`.

    Directories matched by `.gitignore` or the hard-coded always-ignore
    set are pruned entirely (never descended into), which matters for
    performance on repositories with huge ignored trees (e.g.
    `node_modules`, `.venv`).
    """
    root = root.resolve()
    result = ScanResult()

    if not root.exists():
        result.errors.append(f"Repository path does not exist: {root}")
        return result
    if not root.is_dir():
        result.errors.append(f"Repository path is not a directory: {root}")
        return result

    ignore_rules = IgnoreRules(root, extra_patterns=extra_ignore_patterns)

    for file_path in _walk(root, ignore_rules):
        try:
            _process_file(file_path, root, ignore_rules, result)
        except OSError as exc:
            result.errors.append(f"{file_path}: {exc}")
            log.warning("Failed to process %s: %s", file_path, exc)

    log.info(
        "Scan complete: %d files, %d skipped (binary), %d skipped (secret), "
        "%d skipped (ignored), %d errors",
        len(result.files),
        result.skipped_binary,
        result.skipped_secret,
        result.skipped_ignored,
        len(result.errors),
    )
    return result


def _walk(root: Path, ignore_rules: IgnoreRules):
    """Yield every non-ignored file path under `root`, pruning ignored directories."""
    stack = [root]
    while stack:
        current_dir = stack.pop()
        try:
            entries = sorted(current_dir.iterdir())
        except OSError as exc:
            log.warning("Cannot list directory %s: %s", current_dir, exc)
            continue

        for entry in entries:
            if entry.is_dir():
                if not ignore_rules.is_ignored_dir(entry):
                    stack.append(entry)
            elif entry.is_file():
                yield entry


def _process_file(file_path: Path, root: Path, ignore_rules: IgnoreRules, result: ScanResult) -> None:
    relative_path = file_path.relative_to(root).as_posix()

    if is_secret_file(relative_path):
        result.skipped_secret += 1
        log.debug("Skipping secret-pattern file: %s", relative_path)
        return

    if ignore_rules.is_ignored_file(file_path):
        result.skipped_ignored += 1
        return

    if is_binary(file_path):
        result.skipped_binary += 1
        return

    stat = file_path.stat()
    with open(file_path, "rb") as handle:
        peek = handle.read(_PEEK_SIZE)

    result.files.append(
        ProjectFile(
            path=file_path,
            relative_path=relative_path,
            size_bytes=stat.st_size,
            content_hash=compute_file_hash(file_path),
            language=detect_language(file_path, peek_bytes=peek),
            mtime=stat.st_mtime,
        )
    )
