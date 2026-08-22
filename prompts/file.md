You are documenting a codebase, one file at a time. Below are summaries of each piece of a single source file, along with the symbols it defines. Write a concise, factual summary of what this file as a whole does.

File: {file_path}
Language: {language}
Top-level symbols: {symbol_names}

Chunk summaries (in file order):
{chunk_summaries}

Rules:
- 2 to 5 sentences is the floor, not a cap — when the chunk summaries give you enough real detail (specific functions/classes involved, what the file's main entry points consume and produce, a notable pattern the chunks reveal), use it; a longer, more specific summary is a better summary here, not a less polished one. Only stay short when the chunk summaries themselves are genuinely thin. No preamble, no restating the file path, no markdown formatting.
- Describe the file's overall purpose and role, not a list of its parts — synthesize, don't just concatenate the chunk summaries. Synthesizing doesn't mean compressing away real detail the chunks gave you: naming the specific mechanism, the specific functions/classes that do the main work, and what the file's main entry points actually consume and produce is what makes a synthesis useful rather than vague.
- If the chunk summaries suggest a clear responsibility (e.g. "this is a data model", "this is a CLI entry point", "this is a test suite"), say so plainly.
- Preserve any explicit formula, algorithm name, exact function/variable name, or magic constant that appears in the chunk summaries verbatim rather than paraphrasing it away.
- Do not invent behavior that isn't supported by the summaries shown.

Summary: