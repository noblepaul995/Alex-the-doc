"""
Language detection.

Deliberately simple: extension-based lookup first, with a small shebang
fallback for extensionless scripts. This is enough to route files to the
right Tree-sitter grammar later and to give the Scanner Agent a
`language` field to report — it is not a full content-based classifier,
and doesn't need to be. Extend `EXTENSION_LANGUAGE_MAP` as new language
plugins are added (see the "Future Plugin System" section of the spec).
"""

from __future__ import annotations

from pathlib import Path

# Extension (lowercase, including the dot) -> canonical language name.
# Canonical names match what the Parser Agent / plugin system will key on.
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".m": "objective-c",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".mdx": "markdown",
    ".dockerfile": "dockerfile",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".lua": "lua",
    ".r": "r",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".elm": "elm",
    ".hs": "haskell",
    ".jl": "julia",
    ".zig": "zig",
}

# Exact (case-sensitive) filenames that don't carry a meaningful extension.
FILENAME_LANGUAGE_MAP: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "Rakefile": "ruby",
    "Gemfile": "ruby",
    "CMakeLists.txt": "cmake",
}

_SHEBANG_LANGUAGE_MAP: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "node": "javascript",
    "bash": "shell",
    "sh": "shell",
    "zsh": "shell",
    "ruby": "ruby",
    "perl": "perl",
}


def detect_language(path: Path, *, peek_bytes: bytes | None = None) -> str | None:
    """
    Best-effort language detection for a single file.

    `peek_bytes`, if provided, should be the first ~256 bytes of the file
    (the Scanner Agent already reads a chunk to check for binary content,
    so it's cheap to reuse for shebang sniffing rather than re-opening
    the file).
    """
    if path.name in FILENAME_LANGUAGE_MAP:
        return FILENAME_LANGUAGE_MAP[path.name]

    suffix = path.suffix.lower()
    if suffix in EXTENSION_LANGUAGE_MAP:
        return EXTENSION_LANGUAGE_MAP[suffix]

    if peek_bytes and peek_bytes.startswith(b"#!"):
        return _detect_from_shebang(peek_bytes)

    return None


def _detect_from_shebang(peek_bytes: bytes) -> str | None:
    try:
        first_line = peek_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    except Exception:
        return None

    interpreter_line = first_line.removeprefix("#!").strip()
    interpreter = interpreter_line.split("/")[-1].split()[0] if interpreter_line else ""
    # Handle `#!/usr/bin/env python3` style shebangs.
    if interpreter == "env" and len(interpreter_line.split()) > 1:
        interpreter = interpreter_line.split()[1]

    return _SHEBANG_LANGUAGE_MAP.get(interpreter)
