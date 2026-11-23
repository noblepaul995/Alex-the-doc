You are documenting a codebase's architecture. A dependency diagram has already been generated separately (deterministically, from real import data) — your job is only to write the narrative that accompanies it, not to describe or reproduce the diagram's syntax.

Repository summary (already established):
{repo_summary}

The diagram covers these {file_count} file(s), the ones with the most connections to other files:
{file_digest}

Rules:
- No preamble, no mention of "the diagram" as an object (the reader will see it right below this text), and do not reproduce Mermaid syntax.
- Structure the narrative as:
  1. A short lead paragraph (2 to 3 sentences) naming the overall architectural pattern (e.g. "a sequential pipeline of stages", "a layered architecture with a shared core", "a set of independent modules with a common entry point") and how the major pieces connect. Do NOT repeat the repository summary's general domain/purpose intro — focus on structural organization and dependency flow.
  2. A bulleted list (`- **Subsystem / Layer**: ...`, 2 to 5 bullets) explaining the key subsystems/boundaries, what concrete interfaces or data shapes pass between them, and at least one real detail from the file summaries (a specific shared interface, class, or boundary) rather than abstract labels.
- Do not restate the full file list shown above — synthesize what it implies about the system's shape instead.
- Do not make any claim about execution order, timing, or which components run before/after/concurrently with which others, unless that ordering is explicitly part of the repository summary or file summaries above. The file summaries describe what each file does, not when or in what sequence — a guess like "X and Y run in parallel" is not something this pipeline can verify just because it sounds plausible.
- Do not invent components, layers, or patterns that aren't supported by the file summaries shown.

Example output shape:
<2 to 3 sentence lead paragraph naming the architectural pattern and how major pieces connect.>

- **<Subsystem or Layer 1>**: <Concrete mechanism, key classes/interfaces, and what data passes across the boundary.>
- **<Subsystem or Layer 2>**: <Concrete mechanism, key classes/interfaces, and what data passes across the boundary.>

Narrative:
