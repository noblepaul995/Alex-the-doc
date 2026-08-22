You are reviewing a piece of auto-generated documentation for a codebase. Check it against the grounding context it was supposed to be based on, and flag anything wrong — do not rewrite the document, only report findings.

Document being reviewed ("{doc_name}"):
{doc_text}

Grounding context it should be consistent with:
{grounding}

Check for:
- Claims in the document that aren't supported by the grounding context (possible hallucination).
- Internal inconsistencies (the document contradicting itself).
- Clear grammar or clarity problems.

Do NOT flag:
- Reasonable synthesis or paraphrasing of the grounding context.
- Stylistic choices you'd have made differently.
- Missing information the grounding context itself doesn't provide.
- A claim just because it's specific — check the *entire* grounding context first (file names, tool names, and implementation details are often present but easy to skim past); only flag it if it is genuinely absent or contradicted.

## How to work

Do this checking silently. Do not show your reasoning, do not think out loud, do not reconsider or walk back a point once you've made it, and do not narrate phrases like "let me re-check" or "actually, looking closer." Reach your conclusion first, then write only the final, settled output below — nothing that precedes it should appear in your response.

If, on reflection, something you were about to flag is actually supported by the grounding context, simply don't flag it — don't include the abandoned thought.

Before writing each bullet, check it against every bullet you've already written. Two bullets are the same issue if they're about the same underlying claim in the document — even if one adds a new angle, a different quoted phrase, or restates it against a slightly different piece of the grounding context. If a new point you're about to make is really just another way the same claim is wrong, fold it into the existing bullet (or drop it) instead of writing a new one. A useful check: if two bullets would both disappear by fixing the same single sentence in the document, they're one issue, not two.

Report at most the 5 most significant, clearly distinct issues — ranked by how much they'd mislead a reader, not in the order you noticed them. If there are more than 5 real problems, report only the 5 that matter most and stop; do not use the rest of your budget restating smaller variations of issues you've already flagged. There is no bonus for a longer list, and having fewer than 5 is a completely normal outcome, not a shortfall to make up.

Every bullet must name something the document gets wrong relative to the grounding context — a claim it makes that the grounding contradicts or doesn't support, or a place it contradicts itself. Before writing a bullet, check whether it actually does that. If what you're about to write is the document and the grounding context agreeing (even if you also add detail the grounding provides beyond what the document said), that is not an issue — do not write the bullet. A bullet that starts by conceding the document is right about something is a sign to stop, not to keep writing to explain the extra context; drop it entirely and move on. It is entirely correct, and expected, for a review to report zero issues, or fewer than 5, when that's genuinely all there is wrong.

Respond in exactly this format, and nothing else:

STATUS: PASS
or
STATUS: ISSUES_FOUND

Then, only if issues were found, one bullet per distinct issue, none repeated:
- <specific issue, quoting the problematic phrase if possible>

Response: