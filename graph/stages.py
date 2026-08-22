"""
The real, verified execution order of `build_graph()`'s nodes (see
graph/graph.py) — kept in its own module, rather than inside graph.py
itself, purely so agents can import it as ground-truth data without a
circular import (graph.graph already imports every agent's node function).

This is a plain literal, not derived from the compiled graph at runtime:
`architecture`/`api`/`readme`/`structure`/`changelog`/`package_dependencies`
are only concurrent by construction (LangGraph's fan-out in
`build_graph()`), and a topological sort over the compiled graph's edges
wouldn't reliably distinguish "genuinely concurrent" from "just happens
to have no direct edge between them" the way this explicit grouping does.

**One deliberate imprecision:** `githubReadme` only has a graph edge from
`readme` — it doesn't depend on `architecture`, `api`, `structure`,
`changelog`, or `package_dependencies` at all (see
`agents/github_readme_agent.py`). Placing it in its own group strictly
after the whole fan-out group is a safe upper bound (it is genuinely
never claimed to start before `readme` finishes, and in practice
`review`'s own fan-in means the overall pipeline timing is unaffected
either way) but is not a literal claim that it waits on its five
siblings too — this grouping model has no way to express "depends on one
specific member of the prior group" without losing the simplicity that
makes it trustworthy elsewhere. If a prompt ever needs to make a claim
more precise than "runs after `readme`, no earlier," that claim needs a
real dependency edge here, not an inference from this list.

**If you change `build_graph()`'s wiring, update `STAGE_ORDER` here too.**
It exists so prompts that need real sequencing ground truth (see
agents/readme_agent.py's critique pass) can be given actual fact instead
of leaving a model to infer or invent order from prose summaries that
were never meant to describe execution order.
"""

from __future__ import annotations

# Each inner list runs strictly after the group before it; nodes *within*
# the same group run concurrently, so no ordering claim should ever be
# made between them.
STAGE_ORDER: list[list[str]] = [
    ["initialize"],
    ["scan"],
    ["parse"],
    ["resolve_dependencies"],
    ["chunk"],
    ["chunk_doc"],
    ["file_doc"],
    ["knowledge_graph"],
    ["embed"],
    ["vector_db"],
    ["repository_understanding"],
    ["architecture", "api", "readme", "structure", "changelog", "package_dependencies"],
    ["githubReadme"],
    ["review"],
]


def concurrent_stage_groups() -> list[set[str]]:
    """
    The subset of `STAGE_ORDER` groups that actually contain more than
    one node — i.e. the only stage names any two of which can correctly
    be described as running concurrently with each other. Used by
    `utils/ordering_language.py` to distinguish a legitimate claim about
    one of these real groups from an invented concurrency claim about
    stages that aren't actually in the same group — the same real data
    `prompts/readme.md` is given so the draft can accurately describe
    this group in the first place; the deterministic ordering-language
    check needs the same ground truth to avoid "fixing" a sentence that
    was already correct.
    """
    return [set(group) for group in STAGE_ORDER if len(group) > 1]


def render_stage_order() -> str:
    """Render `STAGE_ORDER` as verified ground truth for a prompt to check narrative claims against."""
    lines = []
    for i, group in enumerate(STAGE_ORDER, start=1):
        if len(group) == 1:
            lines.append(f"{i}. {group[0]}")
        else:
            lines.append(
                f"{i}. {', '.join(group)} (these run concurrently with each other — "
                "no order or parallelism claim between any two of them is supported)"
            )
    return "\n".join(lines)