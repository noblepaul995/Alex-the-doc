"""
Small, generic helpers that don't warrant their own module.

`with_retry` centralizes the "retry with exponential backoff" and
"provider failover" requirements from the project spec so every LLM
provider implementation gets consistent, tested retry behaviour instead
of hand-rolling its own loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Iterator
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


def with_retry(
    *,
    max_attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    min_wait: float = 1.0,
    max_wait: float = 20.0,
) -> AsyncRetrying:
    """
    Build a configured `AsyncRetrying` controller for use as:

        async for attempt in with_retry(max_attempts=3, exceptions=(httpx.HTTPError,)):
            with attempt:
                result = await do_the_call()

    Exponential backoff between attempts, capped at `max_wait` seconds.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive `size`-sized chunks from `items`."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def gather_with_concurrency(
    limit: int,
    *coroutines: Awaitable[T],
    on_complete: Callable[[], None] | None = None,
) -> list[T]:
    """
    Run coroutines concurrently, bounded by a semaphore of size `limit`.

    `on_complete`, if given, is called synchronously right after each
    individual coroutine finishes (success or failure that already handled
    its own exception) — not after the whole batch. That makes it safe to
    use as a per-item progress tick even though `asyncio.gather` itself
    only returns once everything is done.
    """
    import asyncio

    semaphore = asyncio.Semaphore(limit)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            result = await coro
            if on_complete is not None:
                on_complete()
            return result

    return await asyncio.gather(*(_bounded(c) for c in coroutines))
