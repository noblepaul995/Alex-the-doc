"""
Centralized logging configuration.

Uses `rich.logging.RichHandler` for pleasant console output while keeping
the standard library `logging` module as the actual backbone, so any
third-party library that logs normally integrates without extra work.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Scanning %s files", count)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure the root logger. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    install_rich_traceback(show_locals=False)

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )

    # Keep noisy third-party libraries at WARNING unless we're in DEBUG.
    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
