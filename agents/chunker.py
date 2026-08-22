"""
Chunk Agent — AST-aware, context-preserving code chunking.

Runs after the Dependency Agent. For every file in `state.parse_results`,
breaks the source into `Chunk`s sized for downstream documentation
agents: each chunk holds one or more *complete* top-level symbols
wherever possible, never an arbitrary line range that might cut a
function in half.

"Top-level" here is determined geometrically (by byte-span containment),
not by each `Symbol.parent` field: Python/JS methods are lexically
nested inside their class's span, so the class is the containing unit
and methods are its children. Go and Rust methods are *not* nested in
their struct/impl's span — `func (p *Point) Move()` is a standalone
top-level declaration elsewhere in the file even though `Symbol.parent`
records which struct it logically belongs to (verified against real
parser output before writing this, not assumed) — so they're correctly
treated as their own top-level chunking units.

When a single top-level symbol is too large for one chunk on its own:
    - if it has geometrically-nested children (a large Python/JS class),
      split by packing those children instead;
    - otherwise (a very long function, or a class with no children small
      enough to help), fall back to a line-aligned raw split.
Either way the result is marked `is_partial=True` with `parent_symbol`
set, so documentation agents downstream know they're seeing one piece
of a larger whole rather than a complete unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import get_settings
from graph.chunk import Chunk, ChunkCollection
from graph.state import RepositoryState
from parsers.metadata import ParseResult, Symbol
from utils.logger import get_logger
from utils.timers import Stopwatch
from utils.tokens import estimate_tokens

log = get_logger(__name__)


def chunk_repository_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: chunk every parsed file into `state.chunks`."""
    max_tokens = get_settings().chunk_max_tokens
    chunks: list[Chunk] = []

    with Stopwatch("chunk") as sw:
        for relative_path, parse_result in state.parse_results.items():
            project_file = state.file_metadata.get(relative_path)
            if project_file is None:
                continue

            try:
                source = project_file.path.read_bytes()
            except OSError as exc:
                state.record_error("chunk", f"Could not read {relative_path}: {exc}", exception=exc, file_path=relative_path)
                continue

            try:
                chunks.extend(_chunk_file(parse_result, source, max_tokens))
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop the run
                state.record_error("chunk", f"Failed to chunk {relative_path}: {exc}", exception=exc, file_path=relative_path)

    collection = ChunkCollection(chunks=chunks)
    log.info(
        "Chunk Agent: %d chunk(s) from %d file(s) (%d partial) in %.2fs",
        len(chunks),
        len(state.parse_results),
        len(collection.partial_chunks()),
        sw.elapsed_seconds,
    )

    return {"chunks": collection, "errors": state.errors}


@dataclass
class _Unit:
    """One packable piece of a file: either a whole top-level symbol, a gap between symbols, or a split piece of an oversized symbol."""

    start_byte: int
    end_byte: int
    tokens: int
    symbol_names: list[str] = field(default_factory=list)
    parent_symbol: str | None = None
    is_partial: bool = False
    context_header: str | None = None


def _chunk_file(result: ParseResult, source: bytes, max_tokens: int) -> list[Chunk]:
    roots = _top_level_symbols(result.symbols)
    units = _build_units(roots, result.symbols, source, max_tokens)
    return [_finalize(u, result.file_path, result.language, source) for u in _pack_units(units, max_tokens)]


def _top_level_symbols(symbols: list[Symbol]) -> list[Symbol]:
    """
    Symbols whose byte span isn't contained within any other symbol's
    span — the geometric definition of "top-level" that works uniformly
    across languages (see module docstring for why `Symbol.parent`
    alone isn't sufficient).
    """
    roots = [s for s in symbols if not any(other is not s and _contains(other, s) for other in symbols)]
    return sorted(roots, key=lambda s: s.span.start_byte)


def _contained_children(root: Symbol, symbols: list[Symbol]) -> list[Symbol]:
    """Symbols (other than `root` itself) whose span is contained within `root`'s span."""
    children = [s for s in symbols if s is not root and _contains(root, s)]
    return sorted(children, key=lambda s: s.span.start_byte)


def _contains(outer: Symbol, inner: Symbol) -> bool:
    return outer.span.start_byte <= inner.span.start_byte and inner.span.end_byte <= outer.span.end_byte


def _build_units(roots: list[Symbol], all_symbols: list[Symbol], source: bytes, max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    cursor = 0

    for root in roots:
        if root.span.start_byte > cursor:
            _append_gap(units, source, cursor, root.span.start_byte)

        text = source[root.span.start_byte : root.span.end_byte]
        tokens = estimate_tokens(text.decode("utf-8", errors="replace"))
        if tokens <= max_tokens:
            units.append(_Unit(root.span.start_byte, root.span.end_byte, tokens, symbol_names=[root.name]))
        else:
            units.extend(_split_oversized_symbol(root, all_symbols, source, max_tokens))

        cursor = max(cursor, root.span.end_byte)

    if cursor < len(source):
        _append_gap(units, source, cursor, len(source))

    return units


def _append_gap(units: list[_Unit], source: bytes, start: int, end: int) -> None:
    text = source[start:end]
    if not text.strip():
        return  # whitespace-only gap between symbols; not worth its own chunk.
    units.append(_Unit(start, end, estimate_tokens(text.decode("utf-8", errors="replace"))))


def _split_oversized_symbol(root: Symbol, all_symbols: list[Symbol], source: bytes, max_tokens: int) -> list[_Unit]:
    children = _contained_children(root, all_symbols)
    header = f"# {root.signature or root.name} (continued)" if root.signature else f"# {root.name} (continued)"

    if children:
        pieces: list[_Unit] = []
        cursor = root.span.start_byte
        for child in children:
            # Any gap between the container's own header/fields and its first child
            # (or between children) gets folded into whichever packed piece follows,
            # by simply extending that child's own span backward to include it —
            # simpler than emitting separate untagged gap units inside a partial symbol.
            start = cursor if cursor < child.span.start_byte else child.span.start_byte
            text = source[start : child.span.end_byte]
            pieces.append(
                _Unit(
                    start,
                    child.span.end_byte,
                    estimate_tokens(text.decode("utf-8", errors="replace")),
                    symbol_names=[child.name],
                    parent_symbol=root.name,
                    is_partial=True,
                    context_header=header,
                )
            )
            cursor = child.span.end_byte
        return _coalesce_partials(pieces, max_tokens)

    return _raw_line_split(root.span.start_byte, root.span.end_byte, source, max_tokens, parent_symbol=root.name, header=header)


def _coalesce_partials(pieces: list[_Unit], max_tokens: int) -> list[_Unit]:
    """Greedily merge adjacent small partial pieces (e.g. several tiny methods) up to the token budget."""
    merged: list[_Unit] = []
    buffer: list[_Unit] = []
    buffer_tokens = 0

    def flush() -> None:
        if not buffer:
            return
        merged.append(
            _Unit(
                buffer[0].start_byte,
                buffer[-1].end_byte,
                buffer_tokens,
                symbol_names=[n for u in buffer for n in u.symbol_names],
                parent_symbol=buffer[0].parent_symbol,
                is_partial=True,
                context_header=buffer[0].context_header,
            )
        )

    for piece in pieces:
        if piece.tokens > max_tokens:
            flush()
            buffer, buffer_tokens = [], 0
            merged.append(piece)  # a single child still too big on its own; emit as-is rather than split further.
            continue
        if buffer and buffer_tokens + piece.tokens > max_tokens:
            flush()
            buffer, buffer_tokens = [], 0
        buffer.append(piece)
        buffer_tokens += piece.tokens

    flush()
    return merged


def _raw_line_split(start_byte: int, end_byte: int, source: bytes, max_tokens: int, *, parent_symbol: str | None, header: str) -> list[_Unit]:
    """Last-resort split for a symbol with no usable children: break on line boundaries, packing lines up to the budget."""
    budget_chars = max_tokens * 4
    lines = source[start_byte:end_byte].splitlines(keepends=True)

    units: list[_Unit] = []
    piece_start = start_byte
    offset = start_byte
    piece_chars = 0

    for line in lines:
        if piece_chars > 0 and piece_chars + len(line) > budget_chars:
            units.append(
                _Unit(
                    piece_start,
                    offset,
                    estimate_tokens(source[piece_start:offset].decode("utf-8", errors="replace")),
                    parent_symbol=parent_symbol,
                    is_partial=True,
                    context_header=header,
                )
            )
            piece_start = offset
            piece_chars = 0
        offset += len(line)
        piece_chars += len(line)

    if piece_chars > 0:
        units.append(
            _Unit(
                piece_start,
                offset,
                estimate_tokens(source[piece_start:offset].decode("utf-8", errors="replace")),
                parent_symbol=parent_symbol,
                is_partial=True,
                context_header=header,
            )
        )

    if parent_symbol:
        for u in units:
            u.symbol_names = [parent_symbol]

    return units


def _pack_units(units: list[_Unit], max_tokens: int) -> list[_Unit]:
    """Greedily merge consecutive whole (non-partial) units up to the token budget. Partial units are never merged with neighbors — they already carry split-specific metadata."""
    packed: list[_Unit] = []
    buffer: list[_Unit] = []
    buffer_tokens = 0

    def flush() -> None:
        if not buffer:
            return
        packed.append(
            _Unit(
                buffer[0].start_byte,
                buffer[-1].end_byte,
                buffer_tokens,
                symbol_names=[n for u in buffer for n in u.symbol_names],
            )
        )

    for unit in units:
        if unit.is_partial:
            flush()
            buffer, buffer_tokens = [], 0
            packed.append(unit)
            continue
        if buffer and buffer_tokens + unit.tokens > max_tokens:
            flush()
            buffer, buffer_tokens = [], 0
        buffer.append(unit)
        buffer_tokens += unit.tokens

    flush()
    return packed


def _finalize(unit: _Unit, file_path: str, language: str, source: bytes) -> Chunk:
    start_line = source.count(b"\n", 0, unit.start_byte) + 1
    end_line = source.count(b"\n", 0, max(unit.end_byte - 1, 0)) + 1

    body = source[unit.start_byte : unit.end_byte].decode("utf-8", errors="replace")
    content = f"{unit.context_header}\n\n{body}" if unit.context_header else body

    return Chunk(
        id=f"{file_path}:{start_line}-{end_line}",
        file_path=file_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        start_byte=unit.start_byte,
        end_byte=unit.end_byte,
        content=content,
        symbol_names=unit.symbol_names,
        parent_symbol=unit.parent_symbol,
        is_partial=unit.is_partial,
        context_header=unit.context_header,
        token_estimate=estimate_tokens(content),
    )
