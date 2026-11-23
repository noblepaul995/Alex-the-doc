You are drafting a README.md for a code repository using ONLY the verified facts below. You do not have access to the raw source code — if something is not stated below, treat it as non-existent.

Repository summary:
{repo_summary}

Language breakdown: {language_breakdown}
Total files documented: {file_count}

Key files (by dependent count), with summaries:
{file_digest}

Verified import relationships (static analysis — shows what imports what, not runtime order):
{dependency_edges}

Verified pipeline execution order (ground truth; items on the same line run concurrently):
{stage_order}

Verified setup, CLI, and environment surface (from package manifests, command decorators, and `.env.example`):
{setup_context}

License: {license_summary}

## Ground rules

1. **Strict grounding — zero invention or hedging.** Every behavioral claim must trace to `repo_summary`, `file_digest`, `dependency_edges`, `stage_order`, `setup_context`, or `license_summary`. Do NOT invent installation commands, CLI flags, environment variables, examples, or benchmarks beyond what is explicitly listed above. Never hedge unsupported claims with "likely," "probably," "appears to," "seems to," "presumably," "may," or "might" — cut the claim instead. If `{setup_context}` is `(none detected)`, omit Installation/Usage sections rather than guessing.
2. **No invented execution order.** Import edges do not imply runtime sequence. Never use "before," "after," "then," "once X completes," "in parallel," "concurrently," "simultaneously," or "meanwhile" to connect components unless describing a verified relationship in `{stage_order}`. Describe interface/dependency relationships instead (e.g., "X consumes Y's queue interface").
3. **Synthesize subsystems — never enumerate files.** A separate file-reference document already exists. Never walk through `file_digest` one file at a time, whether as bullets or semicolon-separated prose. Group related files by shared subsystem or workflow stage, and explain *how* they work together mechanically (specific classes, interfaces, data shapes) and *why* at least one observable design pattern exists (e.g., caching invariant, retry/backoff, provider abstraction).
4. **Match length to real detail; never repeat yourself.** Keep sparse repos concise; go deeper when summaries provide rich detail. Say each fact once in its proper section — never repeat headings, bullets, or file names within the same bullet.

## Synthesis example

Bad (file inventory):
- `src/brain/classifier.processor.js` — Classifies articles using an AI service.
- `src/brain/clean.processor.js` — Cleans and normalizes article data.
- `src/brain/deduplicator.processor.js` — Deduplicates articles within 48 hours.

Good (subsystem synthesis):
- **Article Processing Pipeline** (`src/brain/`): Incoming articles pass through cleaning (`clean.processor.js`), AI classification (`classifier.processor.js`), and 48-hour-window deduplication (`deduplicator.processor.js`), sharing `news.service.js` and `ollama.service.js` for persistence and model calls.

## Output format

Produce only raw Markdown (no preamble, no wrapping code fence):
1. **Title** (`# <Project Name>`) and **1–3 short intro paragraphs** explaining what the system does, its domain, and its core approach.
2. **Tailored `##` sections** that fit this repository's shape (e.g., `## Installation & Setup` when `{setup_context}` has manifest/CLI/env data, `## How It Works` / `## Pipeline Stages` following `{stage_order}`, `## Subsystems` with key anchor file paths like `graph/state.py` or `llm/factory.py`, bulleted `## Architecture & Design Decisions` focused on major architectural choices rather than minor utility helpers, and a two-column table `Component | Purpose` only for genuinely flat lists like parsers or exporters).
3. **Closing `## License` section** if a license was provided above (omit if none found).

README:
