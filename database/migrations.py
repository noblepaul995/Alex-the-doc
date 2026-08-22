"""
Schema setup for the Memory System.

Intentionally minimal: one table (`database/models.py`'s `CacheRecord`),
created via `Base.metadata.create_all()` in `create_memory_engine()`
rather than a full Alembic migration chain. A single-table cache with
no relational structure between rows has no schema evolution story
that `create_all()` can't already handle (a genuinely new column would
still need real migration tooling, but adding one hasn't been necessary
for this table's shape — see `database/models.py`'s module docstring
for why). Introducing Alembic for one table would be meaningfully more
machinery than this system currently needs.
"""

from __future__ import annotations

from database.sqlite import create_memory_engine

__all__ = ["create_memory_engine"]
