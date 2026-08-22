"""
Review data model — the output shape of the Review Agent.

A `ReviewResult` reports findings, it never silently rewrites the
document it reviewed. Auto-correcting a generated doc without human
oversight risks introducing a *new*, unreviewed error in the process of
"fixing" the old one — the Review Agent's job is to surface what a
human (or a future, more capable revision stage) should look at, not
to have the last word on the text itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewResult(BaseModel):
    """One document's review findings."""

    doc_name: str
    passed: bool
    """True if the review found nothing worth flagging."""
    issues: list[str] = Field(default_factory=list)
    model: str | None = None
    raw_response: str | None = None
    """The reviewer's full response, kept for debugging when parsing produces something unexpected."""
    error: str | None = None
    """Set if the review call itself failed after retries — distinct from `passed=False`, which means the review succeeded and found problems."""


class ReviewCollection(BaseModel):
    """The complete set of review results for a run."""

    results: list[ReviewResult] = Field(default_factory=list)

    def by_doc(self, doc_name: str) -> ReviewResult | None:
        return next((r for r in self.results if r.doc_name == doc_name), None)

    def failed_reviews(self) -> list[ReviewResult]:
        """Reviews that couldn't complete (provider error), not documents that failed review."""
        return [r for r in self.results if r.error is not None]

    def flagged_documents(self) -> list[ReviewResult]:
        """Documents the review actually completed for and found issues with."""
        return [r for r in self.results if r.error is None and not r.passed]
