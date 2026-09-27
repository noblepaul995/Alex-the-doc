<h1 align="center">alex</h1>
<p align="center">AI documentation engine — understands and documents codebases of any size.</p>
<p align="center">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge"></a>
<img alt="Primary language" src="https://img.shields.io/badge/language-Python-3776AB.svg?style=for-the-badge">
<img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg?style=for-the-badge">
</p>

---

`alex` scans a codebase, parses it into structured representations (Python, JavaScript, TypeScript, Go, Rust — via tree-sitter), builds a knowledge graph of files/symbols/dependencies, embeds it into a local vector store, and runs it through a set of specialized agents to produce a README, architecture overview, API reference, changelog, and dependency docs — grounded in the actual code, not guessed. It also has an agentic Q&A mode: point it at a repository and ask it questions in plain language, and it investigates using real tools (read a file, look up a symbol, trace an import, search the web) across multiple steps before answering, rather than a single retrieve-then-answer pass.

It works against a local path or a GitHub repo directly, and runs against local models via Ollama/LM Studio or hosted providers (OpenAI, Anthropic, Gemini, Groq, OpenRouter) — your choice.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [Web search (Q&A)](#web-search-qa)
- [How it works](#how-it-works)
- [Known issues](#known-issues)
- [Development](#development)
- [License](#license)

## Install

Requires Python 3.11+.

```bash
git clone <this repo>
cd alex
pip install -e ".[dev,providers]"
```

- `providers` pulls in the OpenAI/Anthropic/Gemini SDKs — skip it if you're only using Ollama or LM Studio.
- `dev` pulls in `pytest`/`mypy`/`ruff` for running the test suite and contributing.

If you want to run entirely locally with no API keys, install [Ollama](https://ollama.com) separately and pull a model:

```bash
ollama pull qwen2.5:7b   # or any model with tool-use support — see Known issues below
ollama pull nomic-embed-text  # used for embeddings
```

## Quick start

```bash
alex setup      # first-run: checks providers, recommends a local model for your hardware, pulls & verifies it
alex doctor     # check provider connectivity/auth at any time
alex document .              # run the full pipeline against the current directory
alex export . --format markdown,pdf   # run the pipeline and write results to disk
alex github https://github.com/owner/repo   # clone a GitHub repo, then build docs or ask questions about it
```

## Commands

| Command | Status | What it does |
|---|---|---|
| `alex setup` | ✅ | Interactive first-run flow: checks which providers already work, helps you configure one, and can recommend/pull/verify a local Ollama model based on your hardware. |
| `alex doctor` | ✅ | Checks connectivity/auth for one or all configured providers and reports latency. |
| `alex scan [path]` | ✅ | Dry run: walks the repository and reports what would be scanned, without calling any model. |
| `alex document [path]` | ✅ | Runs the full pipeline (scan → parse → chunk → knowledge graph → embed → README/architecture/API/changelog agents → review) against a local path. |
| `alex export [path] --format markdown,html,pdf,json -o DIR` | ✅ | Same pipeline as `document`, written to disk in one or more formats. |
| `alex github <url\|owner/repo>` | ✅ | Checks repo access (public or private, with a token), clones it under `~/.alex/github-repos` (reusing an existing clone if there is one, with a prompt to refresh), then lets you choose full documentation or the Q&A investigation loop. |
| `alex purge-repos [owner/repo] [--all] [--yes]` | ✅ | Deletes one or all cached clones made by `alex github`. |
| `alex search <query>` | 🚧 | Placeholder — semantic search over documented code (retrieval stage not yet wired to a standalone command). |
| `alex update [path]` | 🚧 | Placeholder — incremental re-documentation of only changed files. |
| `alex serve [--port]` | 🚧 | Placeholder — serve generated docs locally. |
| `alex clear-cache` | ✅ | Clears the local file-hash/parse/doc cache database. |
| `alex rebuild-index` | 🚧 | Placeholder — rebuild the vector index from stored summaries. |
| `alex stats` | 🚧 | Placeholder — statistics about the last run. |

Every command takes `--verbose`/`-v` for debug-level logging. 🚧 commands print a clear "not yet implemented" message rather than doing something silently wrong.

## Configuration

`alex` reads from `.env` in the working directory (see `.env.example` for the full annotated list). The essentials:

```bash
# Which model runs the per-file/per-chunk work (the bulk of the pipeline)
ALEX_DEFAULT_PROVIDER=ollama
ALEX_DEFAULT_MODEL=qwen2.5:7b

# Optional: a stronger model for the reasoning-heavy stages that run once per repo/question
# instead of once per file — README, architecture, review, and the Q&A investigation loop.
# Leave unset to just reuse ALEX_DEFAULT_PROVIDER/MODEL everywhere.
ALEX_SYNTHESIS_PROVIDER=
ALEX_SYNTHESIS_MODEL=

# API keys, only for the provider(s) you actually use
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=

# For alex github against private repos, or to raise your GitHub rate limit
GITHUB_TOKEN=
```

Supported providers: `ollama`, `lmstudio`, `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, or `custom` (any OpenAI-compatible endpoint). `alex doctor` and `alex setup` will tell you what's actually reachable with your current configuration.

## Web search (Q&A)

The Q&A investigation loop (`alex github ... → 2`) can call out to the web for anything the repository itself can't answer — whether a dependency version is still current, what an external API does, general facts not tied to the codebase. Two backends, tried in this order, both optional:

1. **[open-websearch](https://github.com/Aas-ee/open-webSearch)** — free, self-hosted, no API key. If it isn't already running, `alex` offers to install and start it for you the first time a question needs it, via `npx -y open-websearch@latest` — **this downloads and runs a third-party npm package**, so you'll be asked to confirm first, every time, unless it's already running. Requires Node.js. Configure with `OPEN_WEBSEARCH_ENDPOINT` (if you're running one yourself elsewhere), `OPEN_WEBSEARCH_PORT` (default `3000`), and `OPEN_WEBSEARCH_AUTOSTART=false` if you'd rather it never offer to start one.
2. **[Tavily](https://tavily.com)** — hosted, requires `TAVILY_API_KEY`. Used only if open-websearch isn't reachable and can't be started.

If neither is configured, `web_search` just isn't offered as a tool — the Q&A loop still works, answering from repository evidence alone.

## How it works

Pipeline stages, in order:

1. **Scan** — walks the repo, respects `.gitignore`, skips binaries/secrets.
2. **Parse** — tree-sitter parsers for Python, JS, TS, Go, Rust extract structured symbols.
3. **Dependency resolution** — resolves internal imports and flags external ones.
4. **Chunk** — splits large files into model-sized units.
5. **Chunk/File documentation** — an LLM documents each chunk, then each file.
6. **Knowledge graph** — links files, symbols, and dependencies together.
7. **Embed & vector store** — summaries are embedded and written to a local LanceDB store for retrieval.
8. **Repository understanding** — Architecture, API, README, Structure, Changelog, and Package Dependency agents run, each grounded in the graph and real file contents rather than the model's own assumptions.
9. **Review** — a final pass checks generated docs against the grounding data before anything is exported.

The **Q&A loop** is a separate path: rather than a single-shot answer, the model runs an agentic investigation (up to 6 steps) using `search_repository`, `get_file`, `get_symbol`, `list_dependencies`, and optionally `web_search`, before producing a final answer — and it's instructed to say plainly when it couldn't establish something, rather than guess.

## Known issues

- **Small local models can struggle with the Q&A tool-call protocol.** Models under ~10B parameters (tested case: `qwen3.5:9b`) sometimes emit malformed tool calls, narrate reasoning instead of taking an action, or assert an answer after running out of investigation steps instead of saying it couldn't finish. If you hit this, set `ALEX_SYNTHESIS_PROVIDER`/`ALEX_SYNTHESIS_MODEL` to something larger or a hosted provider for the Q&A loop specifically, and check you're on a recent Ollama release (there's [a known Ollama issue](https://github.com/ollama/ollama/issues/14745) around qwen3.5 sometimes printing tool calls as text instead of executing them).
- **`open-websearch` autostart runs a third-party package.** See [Web search](#web-search-qa) above — it always asks first, but know what you're agreeing to.
- Commands marked 🚧 in the table above are intentionally unimplemented placeholders, not bugs.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
mypy .
```

## License

MIT — see [LICENSE](LICENSE).