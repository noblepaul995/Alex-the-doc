"""
SQLite engine and session setup for the Memory System.

Deliberately synchronous SQLAlchemy, not the async variant. `scan`,
`parse`, and `chunk` — the nodes that need to read/write this cache
most — are plain synchronous functions in `build_scan_graph()`, which
must stay `.invoke()`-compatible (see `graph/graph.py`'s module
docstring for why that split exists). Making the cache layer async
would force those nodes async too, which would force `build_scan_graph()`
itself to require `ainvoke()`, defeating the entire point of that split.
The two async nodes that also touch this cache (`chunk_doc`, `file_doc`)
can call synchronous SQLite I/O from an `async def` function without
issue — it's a local file, not a network call, so there's no event
loop to block in any way that matters here.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base


def create_memory_engine(db_path: Path) -> Engine:
    """Create (and ensure the schema exists for) a SQLite engine at `db_path`."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
