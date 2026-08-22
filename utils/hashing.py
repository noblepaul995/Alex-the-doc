"""
Content hashing primitives.

Used throughout the pipeline (Scanner Agent, Memory System) to detect
whether a file's content has changed since the last run, so that
unchanged files can be skipped entirely. Kept as a standalone utility
because "hash this content" has zero dependency on *what* the content
is or *why* it's being hashed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def hash_text(content: str) -> str:
    """Return the SHA256 hex digest of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_bytes(content: bytes) -> str:
    """Return the SHA256 hex digest of raw bytes."""
    return hashlib.sha256(content).hexdigest()


def hash_file(path: Path | str) -> str:
    """
    Return the SHA256 hex digest of a file's contents, streamed in
    chunks so multi-gigabyte files never need to be loaded into memory.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
