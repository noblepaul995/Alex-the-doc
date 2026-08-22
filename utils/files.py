"""
Generic filesystem helpers.

Deliberately minimal in the foundation stage — repository-walking, ignore
rules, and language detection belong to the Scanner Agent and are added
in a later stage. This module only holds primitives that many later
modules will need regardless of what they're for.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """Create `path` (and parents) if it doesn't exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """
    Write text to `path` atomically: write to a temp file in the same
    directory, then rename over the destination. Prevents readers from
    ever seeing a half-written file (important once we're running many
    async writers concurrently).
    """
    destination = Path(path)
    ensure_dir(destination.parent)

    fd, tmp_path = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
        os.replace(tmp_path, destination)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def safe_read_text(path: Path | str, encoding: str = "utf-8") -> str | None:
    """Read text from `path`, returning None instead of raising if unreadable."""
    try:
        return Path(path).read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError):
        return None
