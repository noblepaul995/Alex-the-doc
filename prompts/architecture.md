You are documenting a codebase's architecture. A dependency diagram has already been generated separately (deterministically, from real import data) — your job is only to write the narrative that accompanies it, not to describe or reproduce the diagram's syntax.

Repository summary (already established):
{repo_summary}

The diagram covers these {file_count} file(s), the ones with the most connections to other files:
{file_digest}

Rules:
- 3 to 6 sentences. No preamble, no markdown formatting, no mention of "the diagram" as an object (the reader will see it right below this text).
- Describe the repository's architecture: how its major pieces are organized and relate to each other, building on the repository summary above rather than repeating it. Don't restate the file list shown above — synthesize what it implies about the system's shape instead.
- If the structure suggests a clear pattern (e.g. "a sequential pipeline of stages", "a layered architecture with a shared core", "a set of independent modules with a common entry point"), say so plainly.
- Where possible, name one concrete, real detail from the file summaries that supports the pattern you describe (e.g. a specific shared interface, a specific stage ordering) rather than describing the pattern only in the abstract.
- Do not make any claim about execution order, timing, or which components run before/after/concurrently with which others, unless that ordering is explicitly part of the repository summary or file summaries above. The file summaries describe what each file does, not when or in what sequence — a guess like "X and Y run in parallel" is not something this pipeline can verify just because it sounds plausible.
- Do not invent components, layers, or patterns that aren't supported by the file summaries shown.

Narrative:
