You are documenting a piece of a codebase. Below is one chunk of source code extracted from a larger file. Write a concise, factual summary of what this specific chunk does.

File: {file_path}
Language: {language}
Symbols in this chunk: {symbol_names}

Rules:
- 1 to 3 sentences is the floor, not a cap — if the code shown supports more (a second distinct thing this chunk does, a notable edge case it handles, a specific error condition it guards against), say that too rather than compressing it out. Only stay to one sentence when the chunk genuinely only does one simple thing. No preamble, no restating the file path, no markdown formatting.
- Describe what the code does — the concrete mechanism, not a generic characterization of it. If its role in the file is clear from the code itself (e.g. it's clearly a validation step, a cache lookup, an error handler), you can name that role, but don't speculate about *why* the codebase's authors chose to write it this way or what broader goal it serves — that's not something this chunk of code alone can tell you.
- Preserve any explicit formula, algorithm name, exact function/variable name, or magic constant verbatim rather than paraphrasing it away — e.g. quote `weight * P(action)` rather than describing it as "a weighted calculation."
- If the chunk calls, is called by, reads from, or writes to something else the code makes visible (another function, a specific field on a passed-in object, a specific exception type it raises or catches), name it — that's exactly the kind of concrete detail that makes a summary useful downstream, not detail to compress away.
- If this chunk is a partial view of a larger symbol (see "(continued)" in the code below), make that clear rather than describing it as if it were the whole thing.
- Do not invent behavior that isn't in the code shown.

Code:
```{language}
{code}
```

Summary: