"""
Progress reporting hook for long-running, per-item pipeline stages.

`chunk_doc` and `file_doc` are the stages that dominate wall-clock time on
a medium/large repo (one LLM call per chunk / per file), and they're also
the stages where "is this actually working or stuck?" matters most. This
module lets them report incremental progress without coupling them to the
CLI's rich progress bars — outside the CLI (tests, library use), reporting
is a silent no-op.

Usage from a node:

    from utils.progress import report_progress

    total = len(chunks)
    completed = 0

    def _tick() -> None:
        nonlocal completed
        completed += 1
        report_progress("chunk_doc", completed, total)

Usage from the CLI:

    with progress_scope(my_callback):
        result = await graph.ainvoke(...)
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager

# Signature: (stage: str, completed: int, total: int) -> None
ProgressCallback = Callable[[str, int, int], None]

_current: contextvars.ContextVar[ProgressCallback | None] = contextvars.ContextVar(
    "progress_callback", default=None
)


def report_progress(stage: str, completed: int, total: int) -> None:
    """Report that `completed` of `total` items are done for `stage`. No-op if nothing is listening."""
    callback = _current.get()
    if callback is not None:
        callback(stage, completed, total)


@contextmanager
def progress_scope(callback: ProgressCallback) -> Iterator[None]:
    """Install `callback` as the active progress reporter for the duration of the `with` block."""
    token = _current.set(callback)
    try:
        yield
    finally:
        _current.reset(token)
