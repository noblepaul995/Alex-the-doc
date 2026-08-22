"""
Ignore rules: what the Scanner Agent should never walk into or document.

Three independent layers:

1. Path-based ignoring (`IgnoreRules`) — `.gitignore` files (root and
   nested, matching real Git semantics via `pathspec`) plus a hard-coded
   set of directories/files that should never be scanned regardless of
   `.gitignore` (`.git`, `node_modules`, virtualenvs, build output, ...).
   These "always ignore" entries exist because plenty of real repos
   don't bother excluding them (e.g. a `.git` directory is implicit).

2. Secret-file denylisting (`SECRET_FILE_PATTERNS`) — `.env`, private
   keys, credential files, etc. are *always* skipped, independent of
   whatever the repo's own `.gitignore` says. A repo's `.gitignore` is
   not a trustworthy signal here: it's entirely possible (and observed
   in practice) for `.env` to not be gitignored at all. Since this
   engine's whole purpose is eventually feeding file contents to an LLM
   and into generated documentation, silently trusting the repo to have
   gitignored its own secrets is not an acceptable default.

3. Content-based binary detection (`is_binary`) — a fast extension
   blocklist first, falling back to sniffing the first chunk of the file
   for a NUL byte, which is a cheap and reliable binary/text heuristic.
"""

from __future__ import annotations

from pathlib import Path

import pathspec

# Directories that are never worth scanning, regardless of .gitignore.
ALWAYS_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".turbo",
        ".idea",
        ".vscode",
        "vendor",
        "coverage",
        ".alex",
        "egg-info",
    }
)

# Files that are always skipped, regardless of .gitignore, because they
# routinely contain live secrets (API keys, tokens, private keys,
# passwords). Negated patterns (`!...`) carve out obviously-safe
# templates that should still be scanned/documented normally.
SECRET_FILE_PATTERNS: list[str] = [
    ".env",
    ".env.*",
    "!.env.example",
    "!.env.sample",
    "!.env.template",
    "!.env.defaults",
    "*.pem",
    "*.key",
    "!*.pub",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_rsa.*",
    "!id_rsa.pub",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service-account*.json",
    ".npmrc",
    ".netrc",
    ".pgpass",
    ".git-credentials",
    "secrets.yml",
    "secrets.yaml",
    "secrets.json",
    "*.secrets",
]

_SECRET_SPEC = pathspec.PathSpec.from_lines("gitignore", SECRET_FILE_PATTERNS)

# Extensions that are essentially always binary — checked before ever
# opening the file, since opening thousands of images/archives in a
# large repo just to sniff bytes would be wasteful.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svg",
        # Audio / video
        ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".mov", ".avi", ".mkv", ".webm",
        # Archives
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".whl", ".jar", ".war",
        # Documents
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # Fonts
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        # Compiled / binaries
        ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".class",
        # Databases / misc binary data
        ".db", ".sqlite", ".sqlite3", ".lock", ".pack", ".idx",
    }
)

_BINARY_SNIFF_SIZE = 8192


def is_binary(path: Path) -> bool:
    """
    Return True if `path` looks like a binary file.

    Fast path: known binary extension. Slow path: read a small chunk and
    check for a NUL byte, which text files essentially never contain.
    """
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    try:
        with open(path, "rb") as handle:
            chunk = handle.read(_BINARY_SNIFF_SIZE)
    except OSError:
        # Unreadable files are treated as binary/skippable rather than
        # raising here — the caller records it as a scan error separately.
        return True

    return b"\x00" in chunk


def is_secret_file(relative_posix_path: str) -> bool:
    """Return True if `relative_posix_path` matches the secret-file denylist."""
    return _SECRET_SPEC.match_file(relative_posix_path)


class IgnoreRules:
    """
    Combines `.gitignore` semantics, the hard-coded always-ignore set,
    and the secret-file denylist.

    Only reads `.gitignore` files that already exist under `root` at
    construction time (nested `.gitignore` files are all loaded upfront
    and their patterns are rooted relative to their own directory, per
    Git's actual behaviour).
    """

    def __init__(self, root: Path, *, extra_patterns: list[str] | None = None) -> None:
        self.root = root
        self._specs: list[tuple[Path, pathspec.PathSpec]] = []

        for gitignore_path in root.rglob(".gitignore"):
            if any(part in ALWAYS_IGNORE_DIRS for part in gitignore_path.parts):
                continue
            lines = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            spec = pathspec.PathSpec.from_lines("gitignore", lines)
            self._specs.append((gitignore_path.parent, spec))

        self._extra_spec = (
            pathspec.PathSpec.from_lines("gitignore", extra_patterns) if extra_patterns else None
        )

    def is_ignored_dir(self, dir_path: Path) -> bool:
        """Should the Scanner Agent avoid descending into `dir_path` at all?"""
        if dir_path.name in ALWAYS_IGNORE_DIRS:
            return True
        return self._matches_gitignore(dir_path, is_dir=True)

    def is_ignored_file(self, file_path: Path) -> bool:
        """Should the Scanner Agent skip `file_path`?"""
        relative = file_path.relative_to(self.root).as_posix()
        if is_secret_file(relative):
            return True
        return self._matches_gitignore(file_path, is_dir=False)

    def _matches_gitignore(self, path: Path, *, is_dir: bool) -> bool:
        if self._extra_spec is not None:
            relative = path.relative_to(self.root).as_posix()
            if self._extra_spec.match_file(relative + "/" if is_dir else relative):
                return True

        for base, spec in self._specs:
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                continue
            if relative.startswith(".."):
                continue
            if spec.match_file(relative + "/" if is_dir else relative):
                return True
        return False
