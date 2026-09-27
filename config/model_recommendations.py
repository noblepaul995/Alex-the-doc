"""
A static, curated table of local (Ollama) models to recommend based on
detected hardware, for `alex setup`'s "I don't have API access, what can
I run locally?" path.

This is deliberately *not* a live query against Ollama's model registry
— there's no simple public search API for it, and adding a dependency
(or a scraper) just for this one setup step isn't worth the fragility.
It's a hand-maintained list of well-known, broadly-recommended model
families as of this writing, organized by the amount of memory a model
of that size roughly needs to run at a usable speed. Treat it as a
reasonable starting point, not an authoritative or benchmarked ranking —
say so plainly in the CLI, since Ollama's library changes over time and
a newer or better-suited model may exist that this table doesn't know
about. `ollama list` / `ollama search` (via the user's own Ollama
installation) is the actual source of truth for what's current.

Sizing rule of thumb this table follows: a quantized local model roughly
needs (parameter count in billions) gigabytes of memory to run
comfortably — a GPU's VRAM if there's a capable one, otherwise system
RAM, with CPU-only inference being far slower but still functional. Tiers
below are deliberately conservative (biased toward "runs smoothly") over
"technically fits."
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.hardware_detect import HardwareInfo


@dataclass
class ModelRecommendation:
    tag: str
    """The exact `ollama pull`/`ollama run` tag."""
    params: str
    """Rough parameter count, for display (e.g. "8B")."""
    blurb: str


@dataclass
class HardwareTier:
    name: str
    min_capability_gb: float
    models: list[ModelRecommendation]


# Ordered smallest-capability-first. `min_capability_gb` is compared
# against whichever is larger of (GPU VRAM, system RAM) — see
# `_effective_capability_gb` — since either can host the model, just at
# very different speeds.
_TIERS: list[HardwareTier] = [
    HardwareTier(
        name="Minimal (under ~8GB)",
        min_capability_gb=0,
        models=[
            ModelRecommendation("qwen2.5:3b", "3B", "Small but capable; a good default for tight memory budgets."),
            ModelRecommendation("llama3.2:3b", "3B", "Meta's small model, solid general instruction-following."),
            ModelRecommendation("gemma2:2b", "2B", "Google's smallest Gemma release; very light footprint."),
            ModelRecommendation("phi3:mini", "3.8B", "Microsoft's small model, punches above its size on reasoning tasks."),
            ModelRecommendation("tinyllama", "1.1B", "The lightest usable option — fastest, least capable of this list."),
        ],
    ),
    HardwareTier(
        name="Standard (~8-16GB)",
        min_capability_gb=8,
        models=[
            ModelRecommendation("llama3.1:8b", "8B", "Widely-used general-purpose model, strong all-around default."),
            ModelRecommendation("mistral:7b", "7B", "Fast, well-regarded for its size; a common local-inference default."),
            ModelRecommendation("qwen2.5:7b", "7B", "Strong instruction-following and code understanding for its size."),
            ModelRecommendation("gemma2:9b", "9B", "A step up from the 2B tier with noticeably better output quality."),
            ModelRecommendation("deepseek-r1:7b", "7B", "Reasoning-focused distilled model; slower per-token but often more careful."),
        ],
    ),
    HardwareTier(
        name="High (~16-32GB)",
        min_capability_gb=16,
        models=[
            ModelRecommendation("qwen2.5:14b", "14B", "Noticeably stronger reasoning and instruction-following than the 7B tier."),
            ModelRecommendation("phi4:14b", "14B", "Microsoft's larger Phi release; strong for its size on structured tasks."),
            ModelRecommendation("mistral-nemo:12b", "12B", "Mistral/NVIDIA collaboration, long context window."),
            ModelRecommendation("deepseek-r1:14b", "14B", "Larger reasoning-distilled variant, more careful multi-step output."),
            ModelRecommendation("gemma2:27b", "27B", "Only comfortable at the top of this tier or with a capable GPU."),
        ],
    ),
    HardwareTier(
        name="Workstation (32GB+ or 16GB+ VRAM)",
        min_capability_gb=32,
        models=[
            ModelRecommendation("qwen2.5:32b", "32B", "One of the stronger open local models available at this size."),
            ModelRecommendation("mixtral:8x7b", "8x7B MoE", "Mixture-of-experts model; strong quality, moderate active-parameter cost."),
            ModelRecommendation("deepseek-r1:32b", "32B", "Top of the reasoning-distilled line short of the full-size model."),
            ModelRecommendation("command-r:35b", "35B", "Cohere's model, built with retrieval/tool-use workloads in mind."),
            ModelRecommendation("llama3.1:70b", "70B", "Only realistic with a strong multi-GPU setup or ample system RAM and patience."),
        ],
    ),
]


def _effective_capability_gb(hardware: HardwareInfo) -> float:
    """The larger of GPU VRAM and system RAM — either can host a model, so the more capable of the two sets the tier."""
    candidates = [value for value in (hardware.total_ram_gb, hardware.gpu_vram_gb) if value is not None]
    return max(candidates) if candidates else 0.0


def recommend_tier(hardware: HardwareInfo) -> HardwareTier:
    """Pick the highest tier this machine's detected capability qualifies for."""
    capability = _effective_capability_gb(hardware)
    matching = [tier for tier in _TIERS if capability >= tier.min_capability_gb]
    return matching[-1] if matching else _TIERS[0]


def top_five(hardware: HardwareInfo) -> list[ModelRecommendation]:
    """The recommended tier's model list — already curated to five entries per tier."""
    return recommend_tier(hardware).models