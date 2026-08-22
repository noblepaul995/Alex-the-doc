"""
Lightweight timing helpers.

`Stopwatch` doubles as a sync/async-friendly context manager so any
agent can measure its own wall-clock time without pulling in a
profiling dependency, and report it into `RepositoryState.statistics`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType


@dataclass
class Stopwatch:
    label: str
    _start: float = field(default=0.0, init=False, repr=False)
    elapsed_seconds: float = field(default=0.0, init=False)

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start

    async def __aenter__(self) -> "Stopwatch":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)
