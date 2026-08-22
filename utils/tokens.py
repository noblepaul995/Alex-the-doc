"""
Lightweight token-count estimation.

No tokenizer dependency (tiktoken, transformers, ...) is pulled in for
this — the Chunker only needs a consistent, cheap way to compare a
candidate chunk against a soft budget, not an exact count for any one
model's vocabulary. The standard "~4 characters per token" heuristic
for English/code text is accurate enough for bin-packing decisions;
being off by 10-20% only shifts where a chunk boundary falls, not
whether the pipeline works.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count for `text`, at ~4 characters per token."""
    return max(1, len(text) // _CHARS_PER_TOKEN)
