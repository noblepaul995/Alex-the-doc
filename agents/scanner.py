"""
Scanner Agent — the first real LangGraph pipeline node.

Wraps `scanner.scanner.scan_repository()`, translating its result into
`RepositoryState` updates: populated `file_metadata`, a `changed_files`
list, and run statistics.

Now backed by the Memory System (`memory.repository.RepositoryMemory`):
loads the previous run's file hashes for this repository (if any) and
passes them to `diff_hashes`, so `changed_files` is a genuine subset
(added + modified) rather than always "everything," and then persists
the current hashes back for next time. Disabled via
`ALEX_MEMORY_ENABLED=false`, in which case `previous_hashes` stays
`None` and every file is reported as changed, matching the pre-Memory-
System behavior exactly.
"""

from __future__ import annotations

from config.settings import get_settings
from graph.state import RepositoryState
from memory.repository import RepositoryMemory
from scanner.hashes import diff_hashes
from scanner.scanner import scan_repository
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)


def scan_repository_node(state: RepositoryState) -> dict[str, object]:
    """
    LangGraph node: walk `state.repository_path` and populate file metadata.

    Runs even if `initialize` already recorded an error for a missing
    path — `scan_repository` performs its own existence check and will
    simply report zero files rather than raising, keeping the graph
    resilient to a bad path instead of crashing mid-run.
    """
    with Stopwatch("scan") as sw:
        result = scan_repository(state.repository_path)

    for error in result.errors:
        state.record_error("scan", error)

    file_metadata = {f.relative_path: f for f in result.files}

    settings = get_settings()
    repository_key = str(state.repository_path.resolve())
    memory = RepositoryMemory(settings.db_path) if settings.memory_enabled else None

    previous_hashes = memory.load_previous_hashes(repository_key) if memory else None
    change_set = diff_hashes(current=file_metadata, previous=previous_hashes)

    if memory:
        for relative_path, project_file in file_metadata.items():
            memory.save_file_hash(repository_key, relative_path, project_file.content_hash, project_file.language)

    state.statistics.files_total = len(file_metadata)
    state.statistics.files_changed = len(change_set.changed)
    state.statistics.files_skipped = result.skipped_binary + result.skipped_ignored + result.skipped_secret
    state.statistics.files_skipped_secret = result.skipped_secret

    log.info(
        "Scanner Agent: %d files found (%d changed, %d binary skipped, %d secret skipped, %d ignored) in %.2fs",
        len(file_metadata),
        len(change_set.changed),
        result.skipped_binary,
        result.skipped_secret,
        result.skipped_ignored,
        sw.elapsed_seconds,
    )

    return {
        "file_metadata": file_metadata,
        "changed_files": change_set.changed,
        "statistics": state.statistics,
        "errors": state.errors,
    }
