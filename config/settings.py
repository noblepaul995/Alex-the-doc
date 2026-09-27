"""
Central, strongly-typed application settings.

Everything that can vary between machines, users, or environments lives
here — never hard-coded deeper in the application. Settings are loaded
from environment variables (and an optional `.env` file) via
`pydantic-settings`, so the rest of the codebase can simply do:

    from config.settings import get_settings
    settings = get_settings()

`get_settings()` is memoized so the environment is only parsed once per
process, while still being trivially overridable in tests via
`get_settings.cache_clear()`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application-wide configuration, sourced from env vars / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider defaults --------------------------------------------------
    default_provider: str = Field(default="ollama", alias="ALEX_DEFAULT_PROVIDER")
    default_model: str = Field(default="mistral:latest", alias="ALEX_DEFAULT_MODEL")
    default_embedding_model: str = Field(default="nomic-embed-text", alias="ALEX_DEFAULT_EMBEDDING_MODEL")

    # --- Provider credentials / endpoints -----------------------------------
    ollama_endpoint: str = Field(default="http://localhost:11434", alias="OLLAMA_ENDPOINT")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_endpoint: str = Field(default="https://api.openai.com/v1", alias="OPENAI_ENDPOINT")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_endpoint: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_ENDPOINT")

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_endpoint: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_ENDPOINT")

    lmstudio_endpoint: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_ENDPOINT")

    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    """Personal access token for `alex github` to check access to, and clone, private repositories. Not required for public repos."""

    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    """API key for `utils/web_search.py`. Not required — search is only ever used as an explicit tool an agent chooses to call, never automatically, so its absence just means that tool isn't offered."""

    open_websearch_endpoint: str | None = Field(default=None, alias="OPEN_WEBSEARCH_ENDPOINT")
    """Base URL of an already-running open-websearch daemon (https://github.com/Aas-ee/open-webSearch),
    e.g. `http://localhost:3000`. Preferred over Tavily in `utils/web_search.py` when reachable — no
    API key, self-hosted. Leave unset to let `open_websearch_port` below decide where to look (and,
    if `open_websearch_autostart` is on, where to start one)."""

    open_websearch_port: int = Field(default=3000, ge=1, le=65535, alias="OPEN_WEBSEARCH_PORT")
    """Port used to reach (or start) a local open-websearch daemon when `open_websearch_endpoint`
    isn't set. 3000 is the project's own documented default (`PORT` env var on their side)."""

    open_websearch_autostart: bool = Field(default=True, alias="OPEN_WEBSEARCH_AUTOSTART")
    """If no open-websearch daemon is reachable, let `utils/web_search.py` try to start one itself
    with `npx -y open-websearch@latest` (asking for confirmation first when running interactively).
    Set to `false` to only ever use one that's already running, or to fall back to Tavily/be
    unavailable without ever spawning anything."""

    # --- Paths ---------------------------------------------------------------
    cache_dir: Path = Field(default=Path(".alex/cache"), alias="ALEX_CACHE_DIR")
    db_path: Path = Field(default=Path(".alex/memory.sqlite3"), alias="ALEX_DB_PATH")
    vector_dir: Path = Field(default=Path(".alex/vectors"), alias="ALEX_VECTOR_DIR")
    """LanceDB database directory — one `vectors` table inside it, holding both chunk and file records."""
    docs_dir: Path = Field(default=Path("docs"), alias="ALEX_DOCS_DIR")

    # --- Memory System -----------------------------------------------------------
    memory_enabled: bool = Field(default=True, alias="ALEX_MEMORY_ENABLED")
    """Cross-run incremental caching (file hashes, parse results, chunk/file docs). Disable
    to force every run to treat every file as changed, matching pre-Memory-System behavior."""

    # --- Runtime behaviour -----------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO, alias="ALEX_LOG_LEVEL")
    max_concurrent_requests: int = Field(default=4, ge=1, le=64, alias="ALEX_MAX_CONCURRENCY")
    request_timeout_seconds: float = Field(default=120.0, gt=0, alias="ALEX_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, ge=0, le=10, alias="ALEX_MAX_RETRIES")

    # --- Chunker ---------------------------------------------------------------
    chunk_max_tokens: int = Field(default=1500, gt=0, alias="ALEX_CHUNK_MAX_TOKENS")
    """Soft token budget per chunk. Approximate (chars/4 heuristic, no tokenizer dependency)."""

    # --- Embeddings --------------------------------------------------------------
    embedding_batch_size: int = Field(default=32, gt=0, alias="ALEX_EMBEDDING_BATCH_SIZE")
    """How many summaries go into one `provider.embed()` call. Providers that support true
    batch requests (OpenAI-compatible) send the whole batch in one HTTP call; providers that
    don't (Ollama loops one request per text internally) still benefit from batching at this
    layer since it bounds how many `_embed_batch` coroutines run concurrently."""

    # --- Repository Agent --------------------------------------------------------
    repo_digest_max_files: int = Field(default=40, gt=0, alias="ALEX_REPO_DIGEST_MAX_FILES")
    """Cap on how many individual file summaries go into the single whole-repository prompt.
    Above this, only the most depended-upon files (by dependents count) are included — a
    proxy for architectural importance, not just an arbitrary truncation."""

    # --- README Agent --------------------------------------------------------------
    readme_digest_max_files: int = Field(default=60, gt=0, alias="ALEX_README_DIGEST_MAX_FILES")
    """Same idea as `repo_digest_max_files`, but tracked separately for the README Agent.
    The repo-summary prompt intentionally stays terse (it's meant as compact grounding
    context fed into *other* prompts), but the README itself is a bigger, more elaborate
    document a person actually reads — so it can afford to see more of the repository."""

    # --- Architecture Agent -------------------------------------------------------
    architecture_max_nodes: int = Field(default=25, gt=0, alias="ALEX_ARCHITECTURE_MAX_NODES")
    """Cap on how many files appear as nodes in the generated Mermaid dependency diagram.
    Selected by total degree (dependents + dependencies), same reasoning as repo_digest_max_files."""

    # --- Changelog Agent -----------------------------------------------------------
    changelog_max_commits: int = Field(default=20, gt=0, alias="ALEX_CHANGELOG_MAX_COMMITS")
    """How many of the most recent `git log` entries to include in the generated changelog.
    Purely deterministic — no LLM involved, since commit history is already exact, verified
    data; there's nothing here for a model to add and real risk in letting one paraphrase it."""
    changelog_git_timeout_seconds: float = Field(default=10.0, gt=0, alias="ALEX_CHANGELOG_GIT_TIMEOUT")
    """Timeout for the underlying `git log`/`git rev-parse` subprocess calls, so a misbehaving
    or enormous repository can't hang the whole pipeline run."""

    # --- Synthesis-stage provider override -----------------------------------------
    synthesis_provider: str | None = Field(default=None, alias="ALEX_SYNTHESIS_PROVIDER")
    synthesis_model: str | None = Field(default=None, alias="ALEX_SYNTHESIS_MODEL")
    """Optional separate provider/model for the handful of whole-repository synthesis
    calls (Repository, Architecture, README, Review Agents) as opposed to the many
    per-chunk/per-file calls (Chunk/File Documentation Agents), which always use
    `default_provider`/`default_model`.

    The two kinds of call have very different economics: chunk/file docs run once
    per chunk/file (up to hundreds of calls on a large repo), so speed dominates and
    a small/fast local model is the right choice there. Synthesis calls run once or
    a handful of times per *entire run* regardless of repo size, so a slower but
    stronger model costs almost nothing in wall-clock time while directly reducing
    the invented-claim rate a weaker model produces — attacking the root cause
    rather than relying on the critique/revise loop to catch more of a weak model's
    mistakes after the fact.

    Both default to `None`, meaning "fall back to `default_provider`/`default_model`"
    — set only `ALEX_SYNTHESIS_MODEL` to keep the same provider but use a different
    model on it (e.g. a bigger local Ollama model), or set both to use an entirely
    different provider (e.g. a hosted API) just for synthesis."""

    @field_validator("cache_dir", "vector_dir", "docs_dir", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value.expanduser()

    def ensure_directories(self) -> None:
        """Create all working directories the app depends on, if missing."""
        for path in (self.cache_dir, self.docs_dir, self.db_path.parent, self.vector_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton (memoized)."""
    settings = Settings()
    settings.ensure_directories()
    return settings