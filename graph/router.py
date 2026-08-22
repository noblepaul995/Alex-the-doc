"""
Conditional routing between pipeline nodes.

Empty in the foundation stage on purpose: routing decisions ("does this
file need re-documenting?", "did the review agent request another
pass?") depend on stages (Scanner, Review Agent) that don't exist yet.
This module exists now so `graph/graph.py` has a stable place to import
routers from as each stage is added, without churn later.
"""

from __future__ import annotations

from graph.state import RepositoryState


def should_continue(state: RepositoryState) -> str:
    """
    Placeholder conditional-edge router.

    Once the incremental-documentation stage exists, this will inspect
    `state.changed_files` / `state.errors` to decide whether to proceed,
    skip, or halt. For now it always signals completion.
    """
    return "end"
