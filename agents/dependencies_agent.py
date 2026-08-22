"""
Dependencies Agent — deterministically renders
`state.generated_docs["dependencies"]`: a "Dependencies & Installation"
section built from the target repository's real package manifest(s).

Deliberately *not* an LLM call, for the same reason as the Changelog and
Structure Agents: a dependency name, a version constraint, a declared
script, a required runtime version — all of this is already exact,
verified data sitting in a manifest file someone actually wrote. There's
nothing here for a model to add, and real risk in letting one summarize
or "helpfully" round off a version constraint. See
`utils/dependency_detect.py` for exactly what's read verbatim versus
what's a fixed ecosystem convention (an install command like `pip
install -e .` for a `pyproject.toml`-based project) rather than a claim
about this specific repository.

If the repository has no manifest this module recognizes, this agent
skips cleanly (no error recorded, no document produced) rather than
treating that as a pipeline failure — plenty of legitimate repositories
(a script collection, a static site with no build step) have no package
manifest at all.
"""

from __future__ import annotations

from graph.state import RepositoryState
from utils.dependency_detect import ManifestInfo, detect_manifests
from utils.logger import get_logger

log = get_logger(__name__)


def dependencies_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["dependencies"]`."""
    manifests = detect_manifests(state.repository_path)

    if not manifests:
        log.info("Dependencies Agent: no recognized package manifest found, skipping")
        return {"generated_docs": state.generated_docs}

    lines = [
        "# Dependencies & Installation",
        "",
        "Every entry below is read directly from this repository's own manifest "
        "file(s) — nothing here is inferred or assumed.",
    ]
    for manifest in manifests:
        lines.append("")
        lines.extend(_render_manifest(manifest))

    state.generated_docs["dependencies"] = "\n".join(lines).rstrip() + "\n"
    log.info(
        "Dependencies Agent: %d manifest(s) found (%s)",
        len(manifests),
        ", ".join(m.ecosystem for m in manifests),
    )
    return {"generated_docs": state.generated_docs}


def _render_manifest(manifest: ManifestInfo) -> list[str]:
    lines = [f"## {manifest.ecosystem} (`{manifest.manifest_file}`)"]

    if manifest.requires:
        lines.append(f"- **Requires:** {manifest.requires}")

    lines.append("")
    lines.append("**Install:**")
    lines.append(f"```bash\n{manifest.install_command}\n```")

    if manifest.dependencies:
        lines.append("")
        lines.append("**Dependencies:**")
        lines.extend(f"- `{dep}`" for dep in manifest.dependencies)

    for group, deps in manifest.optional_dependencies.items():
        if not deps:
            continue
        lines.append("")
        lines.append(f"**{group}:**")
        lines.extend(f"- `{dep}`" for dep in deps)

    if manifest.scripts:
        lines.append("")
        lines.append("**Available commands** (declared in the manifest):")
        lines.extend(f"- `{name}`: `{command}`" for name, command in manifest.scripts.items())

    return lines