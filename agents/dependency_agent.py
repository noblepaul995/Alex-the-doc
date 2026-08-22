"""
Dependency Agent — resolves the `Import`s the Parser Agent found into
an actual file-to-file `DependencyGraph`.

Runs after the Parser Agent. For every `Import` in
`state.parse_results[*].imports`, a language-specific resolver tries to
map the raw module string (`os.path`, `./utils`, `crate::foo`, ...) to
relative_path(s) elsewhere in `state.file_metadata`. An import that
can't be resolved this way (stdlib, third-party, or a local pattern
this best-effort resolver doesn't handle) becomes a `DependencyEdge`
with an empty `targets` list rather than being dropped — "this file
imports something outside the repo" is still useful signal for later
stages (e.g. the Dependency Graph exporter, or flagging unused deps).

Resolution is necessarily language-specific, but every resolver takes
the same shape: `(Import, source_path) -> list[str]` of resolved
relative paths, closing over whatever lookup structure that language
needs (built once per run in `dependency_graph_node`, not per import).
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable
from pathlib import PurePosixPath

from graph.dependency import DependencyEdge, DependencyGraph
from graph.state import ProjectFile, RepositoryState
from parsers.metadata import Import
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)

_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_RUST_ENTRY_POINTS = ("src/lib.rs", "src/main.rs")


def dependency_graph_node(state: RepositoryState) -> dict[str, object]:
    """
    LangGraph node: resolve every import in `state.parse_results` into
    `state.dependency_graph`.

    Lookup structures (Python's dotted-module map, Go's package
    directories, ...) are rebuilt once per run from the *complete*
    `state.file_metadata`, not just `changed_files` — a changed file's
    import may resolve to an unchanged file elsewhere in the repo.
    """
    edges: list[DependencyEdge] = []

    with Stopwatch("dependency_graph") as sw:
        context = _ResolverContext.build(state)

        for source_path, parse_result in state.parse_results.items():
            resolver = _RESOLVERS.get(parse_result.language)
            if resolver is None:
                continue

            for imp in parse_result.imports:
                try:
                    targets = resolver(imp, source_path, context)
                except Exception as exc:  # noqa: BLE001 — one bad import must not stop the run
                    state.record_error(
                        "dependency_graph",
                        f"Failed to resolve import {imp.module!r} in {source_path}: {exc}",
                        exception=exc,
                        file_path=source_path,
                    )
                    targets = []

                edges.append(
                    DependencyEdge(
                        source=source_path,
                        module=imp.module,
                        targets=targets,
                        imported_names=imp.imported_names,
                        is_relative=imp.is_relative,
                    )
                )

    graph = DependencyGraph(edges=edges)
    internal = len(graph.internal_edges())
    log.info(
        "Dependency Agent: %d import(s) resolved (%d internal, %d external/unresolved) in %.2fs",
        len(edges),
        internal,
        len(edges) - internal,
        sw.elapsed_seconds,
    )

    return {"dependency_graph": graph, "errors": state.errors}


class _ResolverContext:
    """Lookup structures shared across all resolvers for a single run, built once."""

    def __init__(self, file_set: set[str], python_modules: dict[str, str], go_module: str | None, go_packages: dict[str, list[str]]) -> None:
        self.file_set = file_set
        self.python_modules = python_modules
        self.go_module = go_module
        self.go_packages = go_packages

    @classmethod
    def build(cls, state: RepositoryState) -> _ResolverContext:
        return cls(
            file_set=set(state.file_metadata),
            python_modules=_build_python_module_map(state.file_metadata),
            go_module=_read_go_module_name(state.repository_path),
            go_packages=_build_go_packages(state.file_metadata),
        )


# --- Python -------------------------------------------------------------------


def _build_python_module_map(file_metadata: dict[str, ProjectFile]) -> dict[str, str]:
    """Dotted module path -> relative_path, for every Python file in the repo."""
    mapping: dict[str, str] = {}
    for relative_path, project_file in file_metadata.items():
        if project_file.language != "python":
            continue
        posix = PurePosixPath(relative_path)
        parts = posix.parts[:-1] if posix.name == "__init__.py" else (*posix.parts[:-1], posix.stem)
        mapping[".".join(parts)] = relative_path
    return mapping


def _resolve_python(imp: Import, source_path: str, ctx: _ResolverContext) -> list[str]:
    if not imp.is_relative:
        # `from pkg import utils` is ambiguous from the import alone: `utils` might be a
        # submodule (pkg/utils.py) or just a name defined in pkg/__init__.py. Prefer the
        # submodule reading when one exists, since that's what the Dependency Agent's
        # consumers (call graphs, "what does this file touch") actually care about.
        submodule_targets = [ctx.python_modules[f"{imp.module}.{n}"] for n in imp.imported_names if f"{imp.module}.{n}" in ctx.python_modules]
        if submodule_targets:
            return submodule_targets

        target = ctx.python_modules.get(imp.module)
        return [target] if target else []

    # `imp.module` for relative imports is just the leading dots (e.g. ".", "..", "...pkg")
    # per parsers/python_parser.py's `_extract_imports`.
    dots = len(imp.module) - len(imp.module.lstrip("."))
    remainder = imp.module[dots:]

    source_dir_parts = list(PurePosixPath(source_path).parent.parts)
    climb = dots - 1  # one dot = same package as the importing file
    base_parts = source_dir_parts[: len(source_dir_parts) - climb] if climb > 0 else source_dir_parts

    if remainder:
        candidate = ".".join([*base_parts, *remainder.split(".")])
        target = ctx.python_modules.get(candidate)
        return [target] if target else []

    # `from . import a, b` — each imported name may itself be a submodule.
    package_module = ".".join(base_parts)
    targets: list[str] = []
    for name in imp.imported_names:
        candidate = f"{package_module}.{name}" if package_module else name
        target = ctx.python_modules.get(candidate)
        if target:
            targets.append(target)
    return targets


# --- JavaScript / TypeScript ---------------------------------------------------


def _resolve_js(imp: Import, source_path: str, ctx: _ResolverContext) -> list[str]:
    if not imp.is_relative:
        return []  # bare specifiers ("react", "lodash") are node_modules/external, not resolved here.

    base_dir = PurePosixPath(source_path).parent
    normalized = posixpath.normpath(str(base_dir / imp.module))

    for ext in ("", *_JS_EXTENSIONS):
        candidate = f"{normalized}{ext}"
        if candidate in ctx.file_set:
            return [candidate]

    for ext in _JS_EXTENSIONS:
        candidate = f"{normalized}/index{ext}"
        if candidate in ctx.file_set:
            return [candidate]

    return []


# --- Go --------------------------------------------------------------------


def _read_go_module_name(repository_path) -> str | None:
    """Module name declared in `go.mod`'s `module` directive, if present."""
    go_mod_path = repository_path / "go.mod"
    if not go_mod_path.exists():
        return None
    try:
        for line in go_mod_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("module "):
                return stripped.removeprefix("module").strip()
    except OSError:
        pass
    return None


def _build_go_packages(file_metadata: dict[str, ProjectFile]) -> dict[str, list[str]]:
    """Package directory (relative to repo root) -> [.go files in it]."""
    packages: dict[str, list[str]] = {}
    for relative_path, project_file in file_metadata.items():
        if project_file.language != "go":
            continue
        package_dir = str(PurePosixPath(relative_path).parent)
        package_dir = "" if package_dir == "." else package_dir
        packages.setdefault(package_dir, []).append(relative_path)
    return packages


def _resolve_go(imp: Import, source_path: str, ctx: _ResolverContext) -> list[str]:
    if ctx.go_module is None:
        return []

    if imp.module == ctx.go_module:
        package_dir = ""
    elif imp.module.startswith(ctx.go_module + "/"):
        package_dir = imp.module[len(ctx.go_module) + 1 :]
    else:
        return []  # stdlib or third-party import path.

    return list(ctx.go_packages.get(package_dir, []))


# --- Rust --------------------------------------------------------------------


def _resolve_rust(imp: Import, source_path: str, ctx: _ResolverContext) -> list[str]:
    if imp.module == "crate" or imp.module.startswith("crate::"):
        remainder = imp.module.removeprefix("crate").removeprefix("::")
        return _resolve_rust_crate_path(remainder, ctx.file_set)

    if imp.module in ("self", "super") or imp.module.startswith(("self::", "super::")):
        base_dir = PurePosixPath(source_path).parent
        remainder = imp.module.split("::", 1)[1] if "::" in imp.module else ""
        candidate_base = str(base_dir) if not remainder else f"{base_dir}/{remainder.replace('::', '/')}"
        return _try_rust_suffixes(candidate_base, ctx.file_set)

    return []  # external crate or std/core/alloc.


def _resolve_rust_crate_path(remainder: str, file_set: set[str]) -> list[str]:
    if not remainder:
        return [ep for ep in _RUST_ENTRY_POINTS if ep in file_set][:1]
    candidate_base = "src/" + remainder.replace("::", "/")
    return _try_rust_suffixes(candidate_base, file_set)


def _try_rust_suffixes(candidate_base: str, file_set: set[str]) -> list[str]:
    for suffix in (".rs", "/mod.rs"):
        candidate = f"{candidate_base}{suffix}"
        if candidate in file_set:
            return [candidate]
    return []


_RESOLVERS: dict[str, Callable[[Import, str, _ResolverContext], list[str]]] = {
    "python": _resolve_python,
    "javascript": _resolve_js,
    "jsx": _resolve_js,
    "typescript": _resolve_js,
    "tsx": _resolve_js,
    "go": _resolve_go,
    "rust": _resolve_rust,
}
