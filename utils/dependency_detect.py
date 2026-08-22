"""
Deterministic dependency & installation detection. No LLM call, no
guessing — same rule as `utils/license_detect.py`: every fact reported
here is read directly from a real manifest file, or is a fixed,
unambiguous convention of that manifest's own ecosystem (`pyproject.toml`
is always installed with `pip`; `package.json` is always installed with
one of `npm`/`yarn`/`pnpm`, picked from whichever lockfile is actually
on disk, not guessed). Install *commands* named here are ecosystem
tooling conventions, not claims about this specific repository's
behavior — the same category of fact as "SHA256 produces a 64-character
hex digest," not a claim that could be right or wrong about what this
project's authors intended. Dependency lists, version constraints,
declared scripts, and required runtime versions are always read
verbatim from the manifest; none of that is ever inferred.

Detects: `pyproject.toml` and `requirements*.txt` for Python;
`package.json` for Node; `Cargo.toml` for Rust; `go.mod` for Go. A
repository can have more than one (e.g. a Python backend alongside a
Node frontend) — `detect_manifests` returns all of them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 — pyproject.toml/Cargo.toml detection is skipped, not guessed at.
    tomllib = None  # type: ignore[assignment]


@dataclass
class ManifestInfo:
    ecosystem: str  # "Python", "Node", "Rust", "Go"
    manifest_file: str  # relative filename actually found on disk
    install_command: str  # a fixed ecosystem convention, or a command read verbatim from the manifest
    dependencies: list[str] = field(default_factory=list)  # verbatim declared entries, e.g. "langgraph>=0.2.0"
    optional_dependencies: dict[str, list[str]] = field(default_factory=dict)  # group/extra name -> verbatim entries
    scripts: dict[str, str] = field(default_factory=dict)  # name -> command, real declared entries only
    requires: str | None = None  # e.g. "Python >=3.11", read verbatim from the manifest


def detect_manifests(repository_path: Path) -> list[ManifestInfo]:
    """Detect every real package manifest in the repository root, one `ManifestInfo` per ecosystem found."""
    manifests: list[ManifestInfo] = []
    for detector in (_detect_python, _detect_node, _detect_rust, _detect_go):
        result = detector(repository_path)
        if result is not None:
            manifests.append(result)
    return manifests


def _detect_python(repository_path: Path) -> ManifestInfo | None:
    pyproject_path = repository_path / "pyproject.toml"
    if tomllib is not None and pyproject_path.is_file():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = None
        if data is not None:
            project = data.get("project", {})
            dependencies = project.get("dependencies")
            if dependencies:
                requires_python = project.get("requires-python")
                return ManifestInfo(
                    ecosystem="Python",
                    manifest_file="pyproject.toml",
                    install_command="pip install -e .",
                    dependencies=list(dependencies),
                    optional_dependencies={
                        group: list(deps) for group, deps in project.get("optional-dependencies", {}).items()
                    },
                    scripts=dict(project.get("scripts", {})),
                    requires=f"Python {requires_python}" if requires_python else None,
                )

    for name in ("requirements.txt", "requirements/base.txt"):
        requirements_path = repository_path / name
        if requirements_path.is_file():
            try:
                lines = requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            dependencies = [
                stripped for line in lines if (stripped := line.strip()) and not stripped.startswith("#")
            ]
            if dependencies:
                return ManifestInfo(
                    ecosystem="Python",
                    manifest_file=name,
                    install_command=f"pip install -r {name}",
                    dependencies=dependencies,
                )

    return None


def _detect_node(repository_path: Path) -> ManifestInfo | None:
    package_json_path = repository_path / "package.json"
    if not package_json_path.is_file():
        return None
    try:
        data = json.loads(package_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    dependencies = data.get("dependencies")
    dev_dependencies = data.get("devDependencies")
    if not dependencies and not dev_dependencies:
        return None

    if (repository_path / "pnpm-lock.yaml").is_file():
        install_command = "pnpm install"
    elif (repository_path / "yarn.lock").is_file():
        install_command = "yarn install"
    else:
        install_command = "npm install"

    node_version = data.get("engines", {}).get("node") if isinstance(data.get("engines"), dict) else None

    return ManifestInfo(
        ecosystem="Node",
        manifest_file="package.json",
        install_command=install_command,
        dependencies=[f"{name}@{version}" for name, version in (dependencies or {}).items()],
        optional_dependencies={
            "devDependencies": [f"{name}@{version}" for name, version in (dev_dependencies or {}).items()]
        }
        if dev_dependencies
        else {},
        scripts=dict(data.get("scripts", {})),
        requires=f"Node {node_version}" if node_version else None,
    )


def _detect_rust(repository_path: Path) -> ManifestInfo | None:
    cargo_path = repository_path / "Cargo.toml"
    if tomllib is None or not cargo_path.is_file():
        return None
    try:
        data = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    dependencies = data.get("dependencies")
    if not dependencies:
        return None

    def _format(name: str, spec: object) -> str:
        if isinstance(spec, str):
            return f"{name} = \"{spec}\""
        if isinstance(spec, dict) and "version" in spec:
            return f"{name} = \"{spec['version']}\""
        return name

    rust_edition = data.get("package", {}).get("edition") if isinstance(data.get("package"), dict) else None

    return ManifestInfo(
        ecosystem="Rust",
        manifest_file="Cargo.toml",
        install_command="cargo build",
        dependencies=[_format(name, spec) for name, spec in dependencies.items()],
        requires=f"Rust edition {rust_edition}" if rust_edition else None,
    )


_GO_REQUIRE_LINE = re.compile(r"^\s*([^\s]+)\s+(v[^\s]+)")


def _detect_go(repository_path: Path) -> ManifestInfo | None:
    go_mod_path = repository_path / "go.mod"
    if not go_mod_path.is_file():
        return None
    try:
        text = go_mod_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    go_version_match = re.search(r"^go\s+([\d.]+)", text, flags=re.MULTILINE)

    dependencies: list[str] = []
    in_require_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("require") and stripped.endswith("("):
            in_require_block = True
            continue
        if in_require_block and stripped == ")":
            in_require_block = False
            continue
        target = stripped
        if not in_require_block:
            if not stripped.startswith("require "):
                continue
            target = stripped[len("require ") :].strip()
        match = _GO_REQUIRE_LINE.match(target)
        if match:
            dependencies.append(f"{match.group(1)} {match.group(2)}")

    if not dependencies:
        return None

    return ManifestInfo(
        ecosystem="Go",
        manifest_file="go.mod",
        install_command="go mod download",
        dependencies=dependencies,
        requires=f"Go {go_version_match.group(1)}" if go_version_match else None,
    )