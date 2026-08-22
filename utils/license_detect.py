"""
Deterministic license detection. Two real, verified sources, checked in
order — never a guess:

  1. A `LICENSE`-shaped file in the repository root, matched against a
     small set of known license signatures (full text comparison).
  2. Failing that, a license declared in a package manifest —
     `pyproject.toml`'s `[project.license]` or `package.json`'s
     `"license"` field. A maintainer-declared field is trusted as-is
     (even an unusual value): unlike matching prose against a signature,
     there's no interpretation step here — the field *is* the claim,
     verbatim, made by whoever owns the repository. This exists because
     it's common for a project to declare its license in its manifest
     and never add a separate `LICENSE` file at all.

No LLM call either way. `detect_license` returns `None` whenever it
can't point to one of these two verified sources, even if the repository
"looks like" it has a license some other way: a badge that says "see
LICENSE file" (or omits the badge entirely) is honest; a badge that
guesses "MIT" for text or a field this module has never actually
confirmed is not, no matter how MIT-shaped it looks.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 — pyproject.toml detection is skipped, not guessed at.
    tomllib = None  # type: ignore[assignment]

# Ordered most-specific-first: some texts (e.g. any BSD variant) share a
# prefix, so the more specific signature must be checked before the more
# general one this repo also uses.
_SIGNATURES: list[tuple[str, tuple[str, ...]]] = [
    ("Apache-2.0", ("Apache License", "Version 2.0")),
    ("GPL-3.0", ("GNU GENERAL PUBLIC LICENSE", "Version 3")),
    ("GPL-2.0", ("GNU GENERAL PUBLIC LICENSE", "Version 2")),
    ("LGPL-3.0", ("GNU LESSER GENERAL PUBLIC LICENSE", "Version 3")),
    ("MPL-2.0", ("Mozilla Public License", "Version 2.0")),
    ("BSD-3-Clause", ("Redistributions of source code must retain", "Neither the name of")),
    ("BSD-2-Clause", ("Redistributions of source code must retain",)),
    ("ISC", ("Permission to use, copy, modify, and/or distribute this software",)),
    ("Unlicense", ("This is free and unencumbered software released into the public domain",)),
    ("MIT", ("Permission is hereby granted, free of charge, to any person obtaining a copy",)),
]

# Common filenames, checked in this order — the first one found on disk is
# read; case variants (e.g. `license.md`) are covered since repos on
# case-insensitive filesystems (macOS, Windows) and Linux both show up in
# the wild.
_CANDIDATE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst", "COPYING")

# A declared manifest field is trusted verbatim, but only up to this
# length — past this, whatever's in the field reads more like an inlined
# full license text than an identifier, and dropping that into a small
# badge would misrepresent it rather than label it.
_MAX_DECLARED_LICENSE_LENGTH = 40


def detect_license(repository_path: Path) -> tuple[str | None, str] | None:
    """
    Try, in order: a `LICENSE`-shaped file's text (matched against known
    signatures), then a declared `license` field in `pyproject.toml` or
    `package.json`.

    Returns:
      - `None` — no LICENSE-shaped file and no declared manifest field.
      - `(None, filename)` — a LICENSE-shaped file exists but its text
        doesn't match any signature this module recognizes; the caller
        can still link to the real file without naming a type.
      - `(license_id, filename)` — either a LICENSE file's text matched a
        known signature, or a manifest declared a license verbatim.
        `filename` is whichever real, on-disk file the claim came from
        (the LICENSE file itself, or the manifest that declared it), so
        the caller can always link to real evidence for the badge.
    """
    from_file = _detect_from_license_file(repository_path)
    if from_file is not None:
        return from_file

    from_manifest = _detect_from_manifest(repository_path)
    if from_manifest is not None:
        return from_manifest

    return None


def _detect_from_license_file(repository_path: Path) -> tuple[str | None, str] | None:
    for name in _CANDIDATE_NAMES:
        path = repository_path / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for spdx_id, markers in _SIGNATURES:
            if all(marker.lower() in text.lower() for marker in markers):
                return spdx_id, name

        # A LICENSE-shaped file exists but matched no known signature —
        # stop here rather than checking other candidate filenames, since
        # finding this file at all means it's almost certainly the
        # repository's real license text, just one this module doesn't
        # recognize yet.
        return None, name

    return None


def _detect_from_manifest(repository_path: Path) -> tuple[str, str] | None:
    pyproject_result = _detect_from_pyproject(repository_path)
    if pyproject_result is not None:
        return pyproject_result

    return _detect_from_package_json(repository_path)


def _detect_from_pyproject(repository_path: Path) -> tuple[str, str] | None:
    if tomllib is None:
        return None

    path = repository_path / "pyproject.toml"
    if not path.is_file():
        return None

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    license_field = data.get("project", {}).get("license")
    declared = _extract_declared_license(license_field)
    return (declared, "pyproject.toml") if declared else None


def _detect_from_package_json(repository_path: Path) -> tuple[str, str] | None:
    path = repository_path / "package.json"
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    declared = _extract_declared_license(data.get("license"))
    return (declared, "package.json") if declared else None


def _extract_declared_license(license_field: object) -> str | None:
    """
    Normalize a manifest's `license` field to a short display string, or
    `None` if the field is absent, empty, or too long to be an
    identifier rather than inlined full text (see
    `_MAX_DECLARED_LICENSE_LENGTH`).

    Handles both the modern PEP 639 form (a bare SPDX expression string,
    e.g. `license = "MIT"`) and the older PEP 621 form (a table, e.g.
    `license = { text = "MIT" }`) that `pyproject.toml` may use, plus
    `package.json`'s plain string form. A `{ file = "..." }` table (the
    other legacy PEP 621 form) is deliberately not handled here — it
    names a file, not a license, and any such file would already have
    been found by `_detect_from_license_file` if it used one of the
    recognized filenames.
    """
    if isinstance(license_field, str):
        value = license_field.strip()
    elif isinstance(license_field, dict):
        value = str(license_field.get("text", "")).strip()
    else:
        value = ""

    if not value or len(value) > _MAX_DECLARED_LICENSE_LENGTH:
        return None
    return value