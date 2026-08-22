"""
Parser Agent — Tree-sitter based AST extraction.

Runs after the Scanner Agent. For every file in `state.changed_files`
whose detected language has a registered extractor, reads the file and
produces a `ParseResult` (symbols, imports, syntax-error flag).

Now backed by the Memory System: every freshly-parsed result is
persisted (`RepositoryMemory.save_parse_result`), and every file the
Scanner Agent marked *unchanged* has its previous `ParseResult` reloaded
from the cache instead of being silently dropped. That reload matters
more than it might look: `chunk`, `resolve_dependencies`, and
`knowledge_graph` all iterate the *complete* `state.parse_results`, not
`state.changed_files` — if `changed_files` becomes a genuine subset
(which it now is, once the Memory System has a previous run to diff
against) and unchanged files were simply never re-parsed, every one of
those downstream stages would silently lose coverage of every
unchanged file on every incremental run. A cache miss for an unchanged
file (e.g. the very first run after enabling the Memory System) falls
back to parsing it fresh rather than leaving it missing — the cache is
strictly an optimization, never a source of truth about which files
exist.

All grammars registered in `parsers/tree_sitter.py` now have a working
extractor: `python_parser.py` (the reference implementation),
`ts_parser.py` (javascript/jsx/typescript/tsx), `go_parser.py`, and
`rust_parser.py`. Languages with no extractor, or no grammar registered
at all, are skipped (not errored) — an unsupported language is an
expected, common case, not a failure.
"""

from __future__ import annotations

from collections.abc import Callable

from config.settings import get_settings
from graph.state import RepositoryState
from memory.repository import RepositoryMemory
from parsers.go_parser import parse_go_file
from parsers.metadata import ParseResult
from parsers.python_parser import parse_python_file
from parsers.rust_parser import parse_rust_file
from parsers.tree_sitter import supported_languages
from parsers.ts_parser import parse_javascript_file, parse_jsx_file, parse_tsx_file, parse_typescript_file
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)

# Canonical language name -> (file_path, source_bytes) -> ParseResult.
_EXTRACTORS: dict[str, Callable[[str, bytes], ParseResult]] = {
    "python": parse_python_file,
    "javascript": parse_javascript_file,
    "jsx": parse_jsx_file,
    "typescript": parse_typescript_file,
    "tsx": parse_tsx_file,
    "go": parse_go_file,
    "rust": parse_rust_file,
}


def parse_repository_node(state: RepositoryState) -> dict[str, object]:
    """
    LangGraph node: parse every changed, supported-language file into
    `state.parse_results`, reloading cached results for unchanged files.

    Reads files directly from disk (not from `RepositoryState`, which
    only carries metadata) using `file_metadata[path].path`, the
    absolute path the Scanner Agent already resolved.
    """
    settings = get_settings()
    repository_key = str(state.repository_path.resolve())
    memory = RepositoryMemory(settings.db_path) if settings.memory_enabled else None

    parse_results: dict[str, ParseResult] = dict(state.parse_results)
    files_parsed = 0
    files_with_errors = 0
    files_reused = 0
    changed_set = set(state.changed_files)

    with Stopwatch("parse") as sw:
        for relative_path, project_file in state.file_metadata.items():
            language = project_file.language
            extractor = _EXTRACTORS.get(language) if language else None
            if extractor is None:
                continue

            if relative_path not in changed_set and memory is not None:
                cached = memory.load_parse_result(repository_key, relative_path, project_file.content_hash)
                if cached is not None:
                    parse_results[relative_path] = cached
                    files_reused += 1
                    continue

            try:
                source = project_file.path.read_bytes()
            except OSError as exc:
                state.record_error("parse", f"Could not read {relative_path}: {exc}", exception=exc, file_path=relative_path)
                continue

            try:
                result = extractor(relative_path, source)
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop the run
                state.record_error("parse", f"Failed to parse {relative_path}: {exc}", exception=exc, file_path=relative_path)
                continue

            parse_results[relative_path] = result
            files_parsed += 1
            if result.has_syntax_errors:
                files_with_errors += 1
                for parse_error in result.errors:
                    state.record_error("parse", f"{relative_path}: {parse_error.message}", file_path=relative_path)

            if memory is not None:
                memory.save_parse_result(repository_key, relative_path, project_file.content_hash, result)

    log.info(
        "Parser Agent: %d file(s) parsed (%d with syntax errors), %d reused from cache, %d supported language(s) registered, in %.2fs",
        files_parsed,
        files_with_errors,
        files_reused,
        len(supported_languages()),
        sw.elapsed_seconds,
    )

    return {
        "parse_results": parse_results,
        "errors": state.errors,
    }
