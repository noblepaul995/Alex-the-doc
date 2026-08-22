"""
Vector Database Agent — persists `state.vectors` to LanceDB.

Runs after the Embedding Agent. Not present as a separate placeholder
in the original scaffold (`embeddings/lancedb.py` was the only file
earmarked for this stage) — added here as its own node to match the
one-node-per-pipeline-stage pattern every other stage already follows,
rather than folding storage into `agents/embedder.py` and giving that
node two responsibilities.

Writes only successful vector records (`VectorRecord.error is None`) —
a record whose embedding failed carries an empty vector, and LanceDB's
vector column can't usefully hold that. Uses `LanceVectorStore.write_records`
(a full overwrite), not `upsert_records` — see `embeddings/lancedb.py`'s
module docstring for why an overwrite is the correct behavior until the
Scanner Agent tracks incremental change state.
"""

from __future__ import annotations

from config.settings import get_settings
from embeddings.lancedb import LanceVectorStore
from graph.state import RepositoryState
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)


async def vector_db_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: write every successful vector record in `state.vectors` to LanceDB."""
    records = [r for r in state.vectors.records if r.error is None]
    if not records:
        return {"statistics": state.statistics, "errors": state.errors}

    settings = get_settings()
    settings.ensure_directories()

    with Stopwatch("vector_db") as sw:
        try:
            store = await LanceVectorStore.connect(settings.vector_dir)
            written = await store.write_records(records)
        except Exception as exc:  # noqa: BLE001 — a storage failure shouldn't crash the whole run
            state.record_error("vector_db", f"Failed to write vector database: {exc}", exception=exc)
            return {"statistics": state.statistics, "errors": state.errors}

    state.statistics.vectors_persisted = written

    log.info("Vector Database Agent: %d vector(s) written to %s in %.2fs", written, settings.vector_dir, sw.elapsed_seconds)

    return {"statistics": state.statistics, "errors": state.errors}
