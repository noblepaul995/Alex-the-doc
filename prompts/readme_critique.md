You are fact-checking a draft README against the verified source material it was supposed to be grounded in. You are NOT rewriting anything — only identifying problems.

Repository summary (already established):
{repo_summary}

Verified file summaries this README was supposed to be grounded in:
{file_digest}

Verified import relationships among these files (static analysis — these show what imports what, NOT what runs before/after or in parallel with anything):
{dependency_edges}

Verified pipeline execution order (ground truth — the actual sequence stages run in; groups on the same line run concurrently with each other):
{stage_order}

Verified license (ground truth for any License section or license claim in the draft):
{license_summary}

Draft README to check:
<draft>
{draft}
</draft>

Check the draft for exactly three kinds of problems, and nothing else:

1. **Unsupported ordering/timing/concurrency claims.** Any sentence that asserts two or more named components/agents/stages run "in parallel," "concurrently," "sequentially," "before," "after," or otherwise implies a specific execution order or timing relationship between them, where that claim contradicts or isn't supported by the verified pipeline execution order above.

2. **Implied-but-wrong sequence.** The draft doesn't need to use words like "before" or "after" to imply an order — simply *describing* components in a sequence (e.g. one paragraph, then the next, each naming different stages) implies that's the order they run in. Check whether the order components are introduced/discussed in the draft matches the verified pipeline execution order above. If the draft's narrative sequence contradicts it — even without explicit ordering words — flag the specific sentence(s) where the mismatch starts.

3. **Untraceable behavioral claims.** Any specific claim about what a named file or component does that isn't traceable to that file's entry in the verified file summaries above. **A claim hedged with "likely," "probably," "appears to," "seems to," "presumably," "may," or "might" is a strong signal it belongs in this category** — a claim the file summary actually supports doesn't need a hedge word to soften it, so the presence of one usually means the draft is guessing. Flag the hedged sentence under `traceability` the same as an unhedged untraceable claim; the hedge doesn't make it acceptable, it just makes it easier to spot.


Do not flag: generic architectural framing (e.g. "the system is organized into a pipeline of stages") that doesn't name a specific order between specific named components; stylistic issues; or anything already directly supported by the file summaries, import relationships, or verified execution order.

Output format — exactly one of:
- The literal text `No issues found.` (if there are none), or
- A bullet list where each bullet has this exact shape: `[category] "exact verbatim sentence from the draft"` — the category must be one of `ordering`, `sequence`, or `traceability` (matching the three numbered problems above), followed by the exact, verbatim problematic sentence copied from the draft (not paraphrased, not summarized) so it can be located precisely.

The category label is required, not optional decoration — if you can't name which of the three numbered problems above a sentence violates, that's a sign it isn't actually a violation, and it shouldn't be flagged at all. Do not flag a sentence just because it's long, or because you're uncertain, or to be thorough — every bullet must correspond to a specific, nameable problem from the list above. A draft with no real problems should get `No issues found.`, not a list of sentences you couldn't otherwise place.

A claim about what a named file *imports*, *depends on*, or *is used by* is a traceability claim like any other: check it against the verified import relationships above the same way you'd check a behavioral claim against the file summaries. This includes the degenerate case of a file claimed to import from itself — that's exactly as checkable against the verified import relationships as any other import claim, not a special case to reason about differently.