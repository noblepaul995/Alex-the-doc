"""
Change detection.

This module owns "given what we scanned before and what we see now,
what changed?" as a pure function over hash maps — deliberately
decoupled from *where* the previous hashes came from. Today the caller
(CLI) has no previous run to compare against, so everything shows up as
"added". Once the Memory System stage exists, it will load the previous
run's hashes from SQLite and pass them in here unchanged — this
function doesn't need to change at all.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from utils.hashing import hash_file
from graph.state import ProjectFile


class ChangeStatus(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


class ChangeSet(BaseModel):
    """The result of diffing two {relative_path: content_hash} snapshots."""

    added: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    deleted: list[str] = []

    @property
    def changed(self) -> list[str]:
        """Paths that need (re-)documentation: added + modified."""
        return [*self.added, *self.modified]

    def status_of(self, relative_path: str) -> ChangeStatus | None:
        for status, paths in (
            (ChangeStatus.ADDED, self.added),
            (ChangeStatus.MODIFIED, self.modified),
            (ChangeStatus.UNCHANGED, self.unchanged),
            (ChangeStatus.DELETED, self.deleted),
        ):
            if relative_path in paths:
                return status
        return None


def compute_file_hash(path) -> str:  # noqa: ANN001 - Path | str, kept loose to match utils.hashing
    """Thin, discoverable wrapper around `utils.hashing.hash_file` for scanner callers."""
    return hash_file(path)


def diff_hashes(
    *,
    current: dict[str, ProjectFile],
    previous: dict[str, str] | None,
) -> ChangeSet:
    """
    Compare the current scan's file hashes against a previous run's.

    `previous` is `None` on a first-ever run (or when no Memory System
    is wired up yet) — in that case every file is reported as `added`.
    """
    if previous is None:
        return ChangeSet(added=sorted(current.keys()))

    added, modified, unchanged = [], [], []
    for relative_path, project_file in current.items():
        previous_hash = previous.get(relative_path)
        if previous_hash is None:
            added.append(relative_path)
        elif previous_hash != project_file.content_hash:
            modified.append(relative_path)
        else:
            unchanged.append(relative_path)

    deleted = sorted(set(previous.keys()) - set(current.keys()))

    return ChangeSet(
        added=sorted(added),
        modified=sorted(modified),
        unchanged=sorted(unchanged),
        deleted=deleted,
    )
