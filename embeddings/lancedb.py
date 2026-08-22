"""
LanceDB client wrapper — persists `VectorRecord`s and supports
similarity search over them.

Every API call this module makes was verified directly against the
installed `lancedb` package (0.34.0) before being used here, not
assumed from documentation or training data — the async API differs
from what most examples online show for older versions:
    - `lancedb.connect_async()`, not `lancedb.connect()`, for an
      awaitable connection.
    - `table.query()` itself is *not* awaitable; only the terminal
      `.to_list()` (after `.nearest_to()`/`.where()`/`.limit()`) is.
    - `db.list_tables()` returns a `ListTablesResponse` object with a
      `.tables` attribute, not a plain list.
    - `db.open_table()` on a name that doesn't exist raises
      `ValueError`, not a LanceDB-specific exception type.
    - `table.add()` always appends — it is not an upsert. Genuine
      upsert-by-key needs `table.merge_insert(key).when_matched_update_all()
      .when_not_matched_insert_all().execute(rows)`.

One `vectors` table holds both chunk and file records, distinguished by
the `record_type` column, rather than separate `chunks`/`files` tables
— `VectorRecord.record_type` already exists for exactly this filtering,
and one table is simpler to manage than several with (currently)
identical schemas. Per-kind tables can be introduced later if a future
collection (architecture, concepts) genuinely needs a different schema.

Every write in this stage is a full overwrite of the table, not an
incremental upsert: the Scanner Agent doesn't yet track file-level
change state across runs (every file is reported as "changed" every
run — see `agents/scanner.py`), so there's nothing meaningful to
preserve from a prior write yet. `merge_insert` is available on
`LanceVectorStore` for when incremental tracking lands, but
`write_records()` itself uses `mode="overwrite"`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb

from embeddings.vector import RecordType, VectorRecord, VectorSearchResult
from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_TABLE_NAME = "vectors"


class LanceVectorStore:
    """Thin async wrapper around a LanceDB database directory."""

    def __init__(self, connection: Any) -> None:
        self._db = connection

    @classmethod
    async def connect(cls, uri: str | Path) -> LanceVectorStore:
        db = await lancedb.connect_async(str(uri))
        return cls(db)

    async def write_records(self, records: list[VectorRecord], *, table_name: str = DEFAULT_TABLE_NAME) -> int:
        """
        Overwrite `table_name` with `records`. Returns the count written.
        A no-op (table left untouched) if `records` is empty — an empty
        write would otherwise still recreate the table with zero rows,
        discarding a previous real write for no reason.
        """
        if not records:
            return 0
        rows = [_record_to_row(r) for r in records]
        await self._db.create_table(table_name, data=rows, mode="overwrite")
        return len(rows)

    async def upsert_records(self, records: list[VectorRecord], *, table_name: str = DEFAULT_TABLE_NAME) -> int:
        """
        Upsert `records` into `table_name` by `id`, creating the table
        first if it doesn't exist yet. Not used by the current
        `vector_db_node` (see module docstring for why), but available
        for when incremental change tracking makes upsert meaningful.
        """
        if not records:
            return 0
        rows = [_record_to_row(r) for r in records]
        try:
            table = await self._db.open_table(table_name)
        except ValueError:
            await self._db.create_table(table_name, data=rows, mode="create")
            return len(rows)

        merge = table.merge_insert("id")
        merge = merge.when_matched_update_all().when_not_matched_insert_all()
        await merge.execute(rows)
        return len(rows)

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        record_type: RecordType | None = None,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> list[VectorSearchResult]:
        try:
            table = await self._db.open_table(table_name)
        except ValueError:
            return []  # table doesn't exist yet — nothing to search.

        query = table.query().nearest_to(query_vector).limit(limit)
        if record_type is not None:
            query = query.where(f"record_type = '{record_type.value}'")

        rows = await query.to_list()
        return [_row_to_search_result(row) for row in rows]

    async def count(self, *, table_name: str = DEFAULT_TABLE_NAME) -> int:
        try:
            table = await self._db.open_table(table_name)
        except ValueError:
            return 0
        return await table.count_rows()


def _record_to_row(record: VectorRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "record_type": record.record_type.value,
        "file_path": record.file_path,
        "text": record.text,
        "model": record.model,
        "vector": record.vector,
    }


def _row_to_search_result(row: dict[str, Any]) -> VectorSearchResult:
    return VectorSearchResult(
        id=row["id"],
        record_type=RecordType(row["record_type"]),
        file_path=row["file_path"],
        text=row["text"],
        distance=row.get("_distance", 0.0),
    )
