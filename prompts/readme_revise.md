You are correcting a draft README. A fact-checking pass already found specific problems in it — your job is to fix only those, and change nothing else beyond what fixing them requires.

Draft README:
<draft>
{draft}
</draft>

Flagged problems from that pass — each one is one of three kinds, whether or not it's explicitly labeled which:
{critique}

Instructions:
- First sort the flagged problems into two groups by how they need to be fixed:
  - **Sentence-level** (unsupported order/timing claim, or an untraceable behavioral/import claim): fix only the flagged sentence itself, in place, and leave everything else exactly as it was.
    - Figure out which of the two sentence-level kinds it actually is **from the sentence's own content, not from whatever label it arrived with.** A label attached to a flagged sentence has been observed to be wrong often enough that trusting it blindly means applying the wrong fix and leaving the real problem untouched — check yourself: does the sentence actually contain an order/timing word or an implied sequence, or does it not? If a sentence is labeled one way but its content clearly indicates the other, go with what the sentence actually says.
    - An **order/timing claim** — the sentence itself contains words like "in parallel," "concurrently," "before," "after," or a narrative sequence that implies an order — replace it with a neutral, order-free description ("X and Y run in parallel" becomes something like "X and Y are both involved in...").
    - An **untraceable claim** — the sentence makes a specific behavioral or import/dependency claim that isn't actually traceable to a verified file summary or the verified import relationships, with no order/timing language involved at all — replace it with something narrower the file summaries (or the verified import relationships) do support, or cut it if nothing supports it.
  - **Structural** (labeled `[structure]`, about the document naming too many individual files rather than synthesizing) — this one is not a single-sentence fix. Find the section(s) it's describing and actually reorganize them: merge bullets or sentences that each just name one file into fewer bullets or short paragraphs that describe what a group of related files does together, using the group's shared purpose rather than an individual file name as the anchor. Cutting the raw count of named files is the point — a merged bullet naming three files in one sentence because they're genuinely three distinct things worth distinguishing is fine; a section that still marches through every file just with commas instead of bullet points hasn't fixed anything. This is the one case where touching un-flagged sentences and changing section structure is expected and necessary — you cannot consolidate ten bullets into three without rewriting all ten.
- Everything not covered by a flagged problem — including full sections that have no structural finding — is preserved exactly: same wording, same formatting, same order. The structural exception above is scoped to the specific section(s) the `[structure]` finding is about, not a license to rewrite the rest of the document.
- Output the full corrected README, start to finish, **exactly once** — no heading repeated, no section duplicated. Copy each untouched sentence or bullet forward a single time; do not restate it again later in the document, even as part of fixing something else nearby. If you finish making the flagged fixes and find yourself starting to write the document, or a section of it, a second time, stop — you are done; do not write it again. No preamble, no explanation of what you changed.

Corrected README: