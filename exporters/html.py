"""
HTML export and live-serve rendering for generated documentation.

Renders Markdown into a full shadcn/ui + Fumadocs-inspired documentation
web application:
- Zinc dark/light design tokens, subtle radial hero gradient, & Lucide SVGs
- Sticky glass navbar with working Cmd/Ctrl+K full-text CommandDialog
  (arrow-key navigable across every page, heading, and file summary)
- Auto-removal of redundant in-body Markdown Table of Contents, replaced by
  a Bento quick-link card grid on the Overview page
- Automatic component upgrades:
  - Numbered pipeline stages -> Vertical Stepper / Timeline component
  - Bold-prefixed bullet lists -> shadcn Card Grid / Feature Callouts
  - CLI command lists -> Copyable terminal command rows
  - Multi-table pages (STRUCTURE.md) -> Unified live search & directory filter pills
  - REVIEW.md status lines -> Verified emerald / destructive Alert banners
  - Code blocks -> Highlight.js syntax highlighting + language header + Copy button
  - Mermaid diagrams -> Pan/zoom + Fullscreen toggle interactive canvas
"""

from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path

import markdown as markdown_lib

_DOC_META: dict[str, tuple[str, str, str, str]] = {
    "README": (
        "Overview",
        "Getting Started",
        "System architecture, installation, CLI commands, and pipeline walkthrough.",
        "book",
    ),
    "DEPENDENCIES": (
        "Dependencies",
        "Getting Started",
        "Verified package manifests, runtime requirements, and installation commands.",
        "package",
    ),
    "ARCHITECTURE": (
        "Architecture",
        "Architecture & Code",
        "Subsystem boundaries, data flow, and deterministic Mermaid dependency graph.",
        "layers",
    ),
    "API": (
        "API Reference",
        "Architecture & Code",
        "Discovered HTTP endpoints, request parameters, and return types.",
        "terminal",
    ),
    "STRUCTURE": (
        "Folder Structure",
        "Architecture & Code",
        "Directory-by-directory reference of every parsed and documented source file.",
        "folder",
    ),
    "REVIEW": (
        "Doc Review",
        "Quality & History",
        "Automated grounding, file-reference, and prose quality audit results.",
        "shield",
    ),
    "CHANGELOG": (
        "Recent Changes",
        "Quality & History",
        "Latest commit activity extracted directly from git history.",
        "git",
    ),
    "githubReadme": (
        "GitHub README",
        "Quality & History",
        "Badge-wrapped GitHub landing page preview.",
        "sparkles",
    ),
}

_CANONICAL_ORDER = [
    "README",
    "DEPENDENCIES",
    "ARCHITECTURE",
    "API",
    "STRUCTURE",
    "REVIEW",
    "CHANGELOG",
    "githubReadme",
]

_ICONS: dict[str, str] = {
    "book": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
    "package": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>',
    "layers": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
    "terminal": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>',
    "folder": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>',
    "shield": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "git": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><line x1="3" x2="9" y1="12" y2="12"/><line x1="15" x2="21" y1="12" y2="12"/></svg>',
    "sparkles": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
}

_MERMAID_BLOCK_RE = re.compile(r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL)
_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — alex docs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css">
<style>
  :root {{
    --bg: #ffffff;
    --fg: #09090b;
    --card: #ffffff;
    --card-hover: #fafafa;
    --muted: #f4f4f5;
    --muted-fg: #71717a;
    --accent: #f4f4f5;
    --accent-fg: #18181b;
    --border: #e4e4e7;
    --border-strong: #d4d4d8;
    --primary: #18181b;
    --primary-fg: #fafafa;
    --brand: #2563eb;
    --brand-soft: rgba(37, 99, 235, 0.08);
    --brand-border: rgba(37, 99, 235, 0.25);
    --code-bg: #09090b;
    --code-header: #121215;
    --success-bg: rgba(16, 185, 129, 0.08);
    --success-fg: #059669;
    --success-border: rgba(16, 185, 129, 0.25);
    --danger-bg: rgba(239, 68, 68, 0.08);
    --danger-fg: #dc2626;
    --danger-border: rgba(239, 68, 68, 0.25);
    --radius: 0.625rem;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 12px -2px rgba(0, 0, 0, 0.06);
    --hero-glow: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(37, 99, 235, 0.09), transparent);
  }}
  html.dark {{
    --bg: #09090b;
    --fg: #fafafa;
    --card: #0c0c0f;
    --card-hover: #121217;
    --muted: #18181b;
    --muted-fg: #a1a1aa;
    --accent: #1f1f24;
    --accent-fg: #fafafa;
    --border: #222228;
    --border-strong: #32323a;
    --primary: #fafafa;
    --primary-fg: #09090b;
    --brand: #60a5fa;
    --brand-soft: rgba(96, 165, 250, 0.12);
    --brand-border: rgba(96, 165, 250, 0.28);
    --code-bg: #070709;
    --code-header: #101014;
    --success-bg: rgba(16, 185, 129, 0.12);
    --success-fg: #34d399;
    --success-border: rgba(16, 185, 129, 0.28);
    --danger-bg: rgba(239, 68, 68, 0.12);
    --danger-fg: #f87171;
    --danger-border: rgba(239, 68, 68, 0.28);
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.35);
    --shadow-md: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    --hero-glow: radial-gradient(ellipse 75% 45% at 50% -15%, rgba(59, 130, 246, 0.14), transparent);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background-color: var(--bg);
    background-image: var(--hero-glow);
    background-repeat: no-repeat;
    color: var(--fg);
    font-size: 15px;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }}

  /* Sticky Top Glass Navbar */
  header.topbar {{
    position: sticky;
    top: 0;
    z-index: 40;
    height: 58px;
    width: 100%;
    border-bottom: 1px solid var(--border);
    background: color-mix(in srgb, var(--bg) 80%, transparent);
    backdrop-filter: blur(14px);
  }}
  .topbar-inner {{
    max-width: 1480px;
    height: 100%;
    margin: 0 auto;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }}
  .topbar-left {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}
  .brand-link {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    color: var(--fg);
    text-decoration: none;
    font-weight: 700;
    font-size: 0.98rem;
    letter-spacing: -0.025em;
  }}
  .brand-icon {{
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: linear-gradient(135deg, var(--primary) 0%, color-mix(in srgb, var(--primary) 75%, var(--brand)) 100%);
    color: var(--primary-fg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.74rem;
    font-weight: 700;
    box-shadow: var(--shadow-sm);
  }}
  .shadcn-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 9999px;
    border: 1px solid var(--border);
    background: var(--muted);
    color: var(--muted-fg);
    padding: 0.12rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 500;
  }}
  .shadcn-badge.brand {{
    background: var(--brand-soft);
    color: var(--brand);
    border-color: var(--brand-border);
  }}
  .topbar-right {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }}
  .cmd-trigger {{
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    height: 36px;
    min-width: 240px;
    padding: 0 0.75rem;
    border-radius: calc(var(--radius) - 2px);
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--muted-fg);
    font-size: 0.82rem;
    cursor: pointer;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s, color 0.15s, background 0.15s;
  }}
  .cmd-trigger:hover {{
    border-color: var(--border-strong);
    background: var(--accent);
    color: var(--fg);
  }}
  .cmd-trigger-left {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .kbd {{
    pointer-events: none;
    display: inline-flex;
    height: 20px;
    align-items: center;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--muted);
    padding: 0 5px;
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    font-weight: 500;
    color: var(--muted-fg);
  }}
  .icon-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: calc(var(--radius) - 2px);
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--fg);
    cursor: pointer;
    box-shadow: var(--shadow-sm);
    transition: background 0.15s, border-color 0.15s;
  }}
  .icon-btn:hover {{
    background: var(--accent);
    border-color: var(--border-strong);
  }}
  .mobile-menu-btn {{ display: none; }}

  /* Three-Column App Shell */
  .shell {{
    max-width: 1480px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 268px minmax(0, 1fr) 240px;
    min-height: calc(100vh - 58px);
  }}

  /* Left Navigation Sidebar */
  aside.sidebar {{
    position: sticky;
    top: 58px;
    height: calc(100vh - 58px);
    overflow-y: auto;
    border-right: 1px solid var(--border);
    padding: 1.5rem 1rem 2rem;
  }}
  .nav-group {{ margin-bottom: 1.5rem; }}
  .nav-group-title {{
    padding: 0 0.65rem;
    margin-bottom: 0.45rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted-fg);
  }}
  .nav-item {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.48rem 0.65rem;
    margin-bottom: 3px;
    border-radius: calc(var(--radius) - 2px);
    color: var(--muted-fg);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    border: 1px solid transparent;
    transition: all 0.12s ease;
  }}
  .nav-item svg {{
    flex-shrink: 0;
    opacity: 0.75;
  }}
  .nav-item:hover {{
    background: var(--accent);
    color: var(--fg);
  }}
  .nav-item.active {{
    background: var(--brand-soft);
    color: var(--brand);
    border-color: var(--brand-border);
    font-weight: 600;
  }}
  .nav-item.active svg {{ opacity: 1; }}

  /* Main Reading Area */
  main.doc-main {{
    padding: 2.5rem 3.75rem 5rem;
    max-width: 920px;
    width: 100%;
  }}
  .hero-header {{
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .breadcrumbs {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    color: var(--muted-fg);
    margin-bottom: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: var(--muted);
    border: 1px solid var(--border);
  }}
  .breadcrumbs span.current {{
    color: var(--fg);
    font-weight: 600;
  }}
  .doc-subtitle {{
    color: var(--muted-fg);
    font-size: 1.02rem;
    line-height: 1.6;
    margin: 0.5rem 0 0;
  }}

  /* Bento Quick-Links Grid (Overview page) */
  .bento-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.85rem;
    margin: 1.5rem 0 2.25rem;
  }}
  .bento-card {{
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    padding: 1rem 1.1rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: var(--shadow-sm);
    text-decoration: none !important;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  }}
  .bento-card:hover {{
    transform: translateY(-2px);
    border-color: var(--brand-border);
    background: var(--card-hover);
  }}
  .bento-card-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--fg);
    font-weight: 600;
    font-size: 0.92rem;
  }}
  .bento-card-head svg {{ color: var(--brand); }}
  .bento-card-desc {{
    color: var(--muted-fg);
    font-size: 0.8rem;
    line-height: 1.45;
  }}

  /* Typography */
  main.doc-main h1 {{
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.18;
    margin: 0;
  }}
  main.doc-main h2 {{
    font-size: 1.38rem;
    font-weight: 600;
    letter-spacing: -0.022em;
    margin: 2.6rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    scroll-margin-top: 78px;
  }}
  main.doc-main h3 {{
    font-size: 1.08rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 1.85rem 0 0.6rem;
    scroll-margin-top: 78px;
  }}
  main.doc-main p {{ margin: 0.9rem 0; }}
  main.doc-main ul, main.doc-main ol {{
    padding-left: 1.35rem;
    margin: 0.85rem 0;
  }}
  main.doc-main li {{ margin: 0.4rem 0; }}
  main.doc-main a {{
    color: var(--brand);
    text-decoration: none;
    font-weight: 500;
  }}
  main.doc-main a:hover {{ text-decoration: underline; }}

  /* Inline Code Pill */
  main.doc-main code {{
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.83em;
    background: var(--muted);
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 0.14rem 0.42rem;
    border-radius: 5px;
    font-weight: 500;
  }}

  /* shadcn Stepper (for <ol> Pipeline Stages) */
  ol.shadcn-stepper {{
    list-style: none;
    padding-left: 0 !important;
    margin: 1.35rem 0 !important;
    position: relative;
  }}
  ol.shadcn-stepper::before {{
    content: "";
    position: absolute;
    left: 15px;
    top: 18px;
    bottom: 18px;
    width: 2px;
    background: var(--border);
  }}
  ol.shadcn-stepper > li {{
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    margin: 0 0 0.65rem 0 !important;
    padding: 0.7rem 0.95rem 0.7rem 0.2rem;
    border-radius: var(--radius);
    transition: background 0.15s;
  }}
  ol.shadcn-stepper > li:hover {{ background: var(--card-hover); }}
  .step-num {{
    position: relative;
    z-index: 2;
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    border-radius: 999px;
    border: 2px solid var(--border-strong);
    background: var(--card);
    color: var(--fg);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .step-body {{
    flex: 1;
    padding-top: 0.2rem;
    font-size: 0.92rem;
  }}

  /* shadcn Feature Cards (for bold-prefixed <ul> lists like Design Decisions / Subsystems) */
  ul.shadcn-card-list {{
    list-style: none;
    padding-left: 0 !important;
    margin: 1.15rem 0 !important;
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
  }}
  ul.shadcn-card-list > li {{
    margin: 0 !important;
    padding: 0.95rem 1.15rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s, background 0.15s;
  }}
  ul.shadcn-card-list > li:hover {{
    border-color: var(--border-strong);
    background: var(--card-hover);
  }}

  /* shadcn Code Card */
  .code-card {{
    margin: 1.25rem 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--code-bg);
    box-shadow: var(--shadow-sm);
  }}
  .code-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.9rem;
    background: var(--code-header);
    border-bottom: 1px solid #222228;
    color: #a1a1aa;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.74rem;
  }}
  .code-dots {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .code-dot {{
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: #27272a;
  }}
  .code-lang {{
    margin-left: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.7rem;
    color: #a1a1aa;
  }}
  .code-copy-btn {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #18181b;
    color: #d4d4d8;
    border: 1px solid #27272a;
    border-radius: 5px;
    padding: 0.22rem 0.6rem;
    font-size: 0.72rem;
    font-family: "Inter", sans-serif;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .code-copy-btn:hover {{
    background: #27272a;
    color: #ffffff;
  }}
  main.doc-main pre {{
    margin: 0;
    padding: 1.05rem 1.2rem;
    overflow-x: auto;
    background: var(--code-bg);
    color: #e4e4e7;
  }}
  main.doc-main pre code {{
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    color: inherit;
    font-size: 0.85rem;
    line-height: 1.65;
  }}

  /* Global Explorer Bar for multi-table pages (STRUCTURE.md) */
  .structure-explorer {{
    position: sticky;
    top: 68px;
    z-index: 25;
    margin: 1.25rem 0 1.75rem;
    padding: 0.85rem 1rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: color-mix(in srgb, var(--card) 90%, transparent);
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow-md);
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
  }}
  .structure-explorer-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }}
  .structure-search-input {{
    flex: 1;
    height: 36px;
    padding: 0 0.85rem;
    border-radius: calc(var(--radius) - 2px);
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
    font-size: 0.86rem;
    outline: none;
  }}
  .structure-search-input:focus {{ border-color: var(--brand); }}
  .structure-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }}
  .folder-pill {{
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--muted);
    color: var(--muted-fg);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.73rem;
    cursor: pointer;
    text-decoration: none !important;
    transition: all 0.12s;
  }}
  .folder-pill:hover, .folder-pill.active {{
    background: var(--brand-soft);
    color: var(--brand);
    border-color: var(--brand-border);
  }}

  /* shadcn Data Table Card */
  .table-card {{
    margin: 1rem 0 1.75rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--card);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
  }}
  .table-scroll {{ overflow-x: auto; }}
  main.doc-main table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }}
  main.doc-main th, main.doc-main td {{
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  main.doc-main td:first-child {{
    white-space: nowrap;
    width: 220px;
  }}
  main.doc-main th {{
    background: var(--muted);
    color: var(--muted-fg);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  main.doc-main tr:last-child td {{ border-bottom: 0; }}
  main.doc-main tbody tr:hover {{ background: var(--card-hover); }}

  /* Mermaid Interactive Card (with Fullscreen & Zoom) */
  .mermaid-card {{
    margin: 1.5rem 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--card);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
  }}
  .mermaid-card.fullscreen {{
    position: fixed;
    inset: 1.5rem;
    z-index: 120;
    margin: 0;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.65);
  }}
  .mermaid-toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.95rem;
    border-bottom: 1px solid var(--border);
    background: var(--muted);
    font-size: 0.78rem;
    color: var(--muted-fg);
    font-weight: 500;
  }}
  .mermaid-actions {{
    display: flex;
    gap: 0.35rem;
  }}
  .mermaid-btn {{
    background: var(--card);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.22rem 0.6rem;
    font-size: 0.74rem;
    font-weight: 500;
    cursor: pointer;
  }}
  .mermaid-btn:hover {{ background: var(--accent); }}
  .mermaid-viewport {{
    overflow: auto;
    padding: 1.75rem;
    text-align: center;
    max-height: 680px;
    flex: 1;
  }}
  .mermaid-card.fullscreen .mermaid-viewport {{ max-height: none; }}
  .mermaid {{
    display: inline-block;
    transform-origin: top center;
    transition: transform 0.15s ease;
  }}

  /* Status Alert Cards (REVIEW.md) */
  .status-alert {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.75rem 1rem;
    border-radius: var(--radius);
    font-size: 0.86rem;
    font-weight: 600;
    margin: 0.65rem 0 1.25rem;
  }}
  .status-pass {{
    background: var(--success-bg);
    color: var(--success-fg);
    border: 1px solid var(--success-border);
  }}
  .status-fail {{
    background: var(--danger-bg);
    color: var(--danger-fg);
    border: 1px solid var(--danger-border);
  }}

  /* Prev / Next Pager Cards */
  .pager {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 3.75rem;
    padding-top: 1.75rem;
    border-top: 1px solid var(--border);
  }}
  .pager-card {{
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.95rem 1.15rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: var(--shadow-sm);
    text-decoration: none !important;
    transition: all 0.15s ease;
  }}
  .pager-card:hover {{
    border-color: var(--brand-border);
    background: var(--card-hover);
    transform: translateY(-1px);
  }}
  .pager-card.next {{ text-align: right; }}
  .pager-dir {{
    font-size: 0.74rem;
    color: var(--muted-fg);
    font-weight: 500;
  }}
  .pager-title {{
    font-size: 0.94rem;
    color: var(--fg);
    font-weight: 600;
  }}

  /* Right Sticky TOC */
  aside.toc {{
    position: sticky;
    top: 58px;
    height: calc(100vh - 58px);
    overflow-y: auto;
    padding: 2.5rem 1.25rem;
    font-size: 0.8rem;
  }}
  .toc-header {{
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-fg);
    margin-bottom: 0.75rem;
  }}
  .toc-list {{
    list-style: none;
    padding: 0;
    margin: 0;
    border-left: 1px solid var(--border);
  }}
  .toc-list a {{
    display: block;
    padding: 0.3rem 0 0.3rem 0.8rem;
    color: var(--muted-fg);
    text-decoration: none;
    line-height: 1.35;
    border-left: 2px solid transparent;
    margin-left: -1px;
    transition: all 0.12s;
  }}
  .toc-list a:hover {{ color: var(--fg); }}
  .toc-list a.active {{
    color: var(--brand);
    font-weight: 600;
    border-left-color: var(--brand);
  }}
  .toc-list a.toc-h3 {{ padding-left: 1.45rem; font-size: 0.77rem; }}

  /* Command Palette Modal (Cmd+K) */
  .cmd-backdrop {{
    display: none;
    position: fixed;
    inset: 0;
    z-index: 150;
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(4px);
    align-items: flex-start;
    justify-content: center;
    padding-top: 11vh;
  }}
  .cmd-backdrop.open {{ display: flex; }}
  .cmd-dialog {{
    width: 100%;
    max-width: 580px;
    border-radius: var(--radius);
    border: 1px solid var(--border-strong);
    background: var(--card);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55);
    overflow: hidden;
  }}
  .cmd-input-wrap {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--muted-fg);
  }}
  .cmd-input {{
    width: 100%;
    height: 48px;
    border: 0;
    background: transparent;
    color: var(--fg);
    font-size: 0.93rem;
    outline: none;
  }}
  .cmd-results {{
    max-height: 360px;
    overflow-y: auto;
    padding: 0.5rem;
  }}
  .cmd-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0.75rem;
    border-radius: calc(var(--radius) - 2px);
    color: var(--fg);
    text-decoration: none;
    font-size: 0.86rem;
  }}
  .cmd-item:hover, .cmd-item.selected {{
    background: var(--brand-soft);
    color: var(--brand);
  }}
  .cmd-item-left {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }}
  .cmd-item-title {{
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cmd-item-snippet {{
    font-size: 0.75rem;
    color: var(--muted-fg);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cmd-item-sub {{
    flex-shrink: 0;
    font-size: 0.72rem;
    color: var(--muted-fg);
    font-family: "JetBrains Mono", monospace;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    background: var(--muted);
    border: 1px solid var(--border);
  }}

  @media (max-width: 1180px) {{
    .shell {{ grid-template-columns: 248px minmax(0, 1fr); }}
    aside.toc {{ display: none; }}
  }}
  @media (max-width: 820px) {{
    .mobile-menu-btn {{ display: inline-flex; }}
    .shell {{ grid-template-columns: 1fr; }}
    aside.sidebar {{
      position: fixed;
      left: -280px;
      top: 58px;
      width: 268px;
      z-index: 50;
      background: var(--bg);
      transition: left 0.2s ease;
    }}
    aside.sidebar.open {{ left: 0; }}
    main.doc-main {{ padding: 1.5rem 1.2rem 3.5rem; }}
    .pager {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="topbar-left">
      <button type="button" class="icon-btn mobile-menu-btn" onclick="document.querySelector('aside.sidebar').classList.toggle('open')" aria-label="Toggle Menu">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/></svg>
      </button>
      <a href="#" class="brand-link">
        <span class="brand-icon">ax</span>
        <span>alex</span>
      </a>
      <span class="shadcn-badge brand">docs</span>
      <span class="shadcn-badge">{category}</span>
    </div>
    <div class="topbar-right">
      <button type="button" class="cmd-trigger" onclick="openCmdPalette()">
        <span class="cmd-trigger-left">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          <span>Search documentation...</span>
        </span>
        <span class="kbd">⌘K</span>
      </button>
      <button type="button" class="icon-btn" onclick="toggleAlexTheme()" title="Toggle light/dark theme" aria-label="Toggle theme">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>
      </button>
    </div>
  </div>
</header>

<div class="shell">
  {sidebar_html}
  <main class="doc-main">
    <div class="hero-header">
      <div class="breadcrumbs">
        <span>Docs</span>
        <span>›</span>
        <span>{category}</span>
        <span>›</span>
        <span class="current">{friendly_title}</span>
      </div>
      <div id="hero-title-slot"></div>
      <p class="doc-subtitle">{subtitle}</p>
    </div>
    {bento_html}
    {body}
    {pager_html}
  </main>
  <aside class="toc" id="on-this-page" aria-label="On this page"></aside>
</div>

<div class="cmd-backdrop" id="cmd-backdrop" onclick="if(event.target===this)closeCmdPalette()">
  <div class="cmd-dialog">
    <div class="cmd-input-wrap">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
      <input type="text" class="cmd-input" id="cmd-input" placeholder="Search pages, sections, or files..." oninput="filterCmdPalette(this.value)">
      <span class="kbd">ESC</span>
    </div>
    <div class="cmd-results" id="cmd-results"></div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<script>
  (function() {{
    const root = document.documentElement;
    const saved = localStorage.getItem('alex-docs-theme');
    if (saved === 'light') root.classList.remove('dark');
    else if (saved === 'dark') root.classList.add('dark');
    window.toggleAlexTheme = function() {{
      const isDark = root.classList.toggle('dark');
      localStorage.setItem('alex-docs-theme', isDark ? 'dark' : 'light');
    }};

    // Move first H1 into hero-title-slot
    const h1 = document.querySelector('main.doc-main > h1');
    const heroSlot = document.getElementById('hero-title-slot');
    if (h1 && heroSlot) heroSlot.appendChild(h1);

    // Remove redundant in-body "## Table of Contents" section injected for raw Markdown readers
    document.querySelectorAll('main.doc-main > h2').forEach(function(h2) {{
      if (h2.textContent.trim().toLowerCase() === 'table of contents') {{
        const next = h2.nextElementSibling;
        if (next && (next.tagName === 'UL' || next.tagName === 'OL')) next.remove();
        h2.remove();
      }}
    }});

    // Rewrite relative .md links inside body to match current routing mode
    const isHtmlExport = window.location.pathname.endsWith('.html');
    document.querySelectorAll('main.doc-main a[href$=".md"]').forEach(function(a) {{
      const raw = a.getAttribute('href');
      if (raw && !raw.startsWith('http')) {{
        const stem = raw.slice(0, -3);
        a.setAttribute('href', isHtmlExport ? stem + '.html' : '/' + stem);
      }}
    }});

    // Upgrade <ol> pipeline steps into a shadcn vertical Stepper
    document.querySelectorAll('main.doc-main ol').forEach(function(ol) {{
      const items = Array.from(ol.children).filter(function(el) {{ return el.tagName === 'LI'; }});
      if (items.length < 3) return;
      ol.classList.add('shadcn-stepper');
      items.forEach(function(li, idx) {{
        const inner = li.innerHTML;
        li.innerHTML = '<span class="step-num">' + String(idx + 1).padStart(2, '0') + '</span>' +
                       '<div class="step-body">' + inner + '</div>';
      }});
    }});

    // Upgrade bold-prefixed <ul> lists into shadcn Feature Card grids
    document.querySelectorAll('main.doc-main > ul').forEach(function(ul) {{
      const items = Array.from(ul.children).filter(function(el) {{ return el.tagName === 'LI'; }});
      if (items.length < 2) return;
      const boldCount = items.filter(function(li) {{
        const first = li.firstElementChild;
        return first && (first.tagName === 'STRONG' || first.tagName === 'CODE');
      }}).length;
      if (boldCount >= Math.ceil(items.length * 0.6)) {{
        ul.classList.add('shadcn-card-list');
      }}
    }});

    // Transform Status: PASS / ISSUES FOUND into shadcn Alert banners
    document.querySelectorAll('main.doc-main p').forEach(function(p) {{
      const txt = p.textContent.trim();
      if (txt === 'Status: PASS') {{
        p.outerHTML = '<div class="status-alert status-pass">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>' +
          '<span>PASS — Verified against grounding context with zero issues</span></div>';
      }} else if (txt.startsWith('Status: ISSUES')) {{
        p.outerHTML = '<div class="status-alert status-fail">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>' +
          '<span>' + txt + '</span></div>';
      }}
    }});

    // Upgrade <pre><code> into shadcn Code Cards + Highlight.js
    document.querySelectorAll('main.doc-main pre').forEach(function(pre) {{
      const code = pre.querySelector('code');
      let lang = 'shell';
      if (code && code.className) {{
        const m = code.className.match(/language-([a-zA-Z0-9_-]+)/);
        if (m) lang = m[1];
      }}
      if (window.hljs && code) {{
        try {{ window.hljs.highlightElement(code); }} catch (e) {{}}
      }}
      const card = document.createElement('div');
      card.className = 'code-card';
      const header = document.createElement('div');
      header.className = 'code-card-header';
      header.innerHTML = '<div class="code-dots"><span class="code-dot"></span><span class="code-dot"></span><span class="code-dot"></span><span class="code-lang">' + lang + '</span></div>';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'code-copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', function() {{
        navigator.clipboard.writeText(pre.innerText).then(function() {{
          btn.textContent = '✓ Copied';
          setTimeout(function() {{ btn.textContent = 'Copy'; }}, 1500);
        }});
      }});
      header.appendChild(btn);
      pre.parentNode.insertBefore(card, pre);
      card.appendChild(header);
      card.appendChild(pre);
    }});

    // Wrap tables in shadcn Table Cards, and if there are multiple tables (e.g. STRUCTURE.md),
    // inject one sticky global Explorer filter bar with folder pills at the top!
    const allTables = Array.from(document.querySelectorAll('main.doc-main table'));
    allTables.forEach(function(tbl) {{
      const card = document.createElement('div');
      card.className = 'table-card';
      const scroll = document.createElement('div');
      scroll.className = 'table-scroll';
      tbl.parentNode.insertBefore(card, tbl);
      scroll.appendChild(tbl);
      card.appendChild(scroll);
    }});

    if (allTables.length > 1) {{
      const explorer = document.createElement('div');
      explorer.className = 'structure-explorer';
      const totalRows = allTables.reduce(function(acc, t) {{ return acc + t.querySelectorAll('tbody tr').length; }}, 0);
      explorer.innerHTML =
        '<div class="structure-explorer-top">' +
        '<input type="text" class="structure-search-input" id="global-table-filter" placeholder="Filter all ' + totalRows + ' files by name or purpose...">' +
        '<span class="shadcn-badge" id="global-table-count">' + totalRows + ' files</span>' +
        '</div><div class="structure-pills" id="folder-pills"></div>';
      const firstHeading = document.querySelector('main.doc-main h3, main.doc-main .table-card');
      if (firstHeading) firstHeading.parentNode.insertBefore(explorer, firstHeading);

      const pillsBox = document.getElementById('folder-pills');
      document.querySelectorAll('main.doc-main h3').forEach(function(h3) {{
        if (!h3.id) h3.id = h3.textContent.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-');
        const pill = document.createElement('a');
        pill.className = 'folder-pill';
        pill.href = '#' + h3.id;
        pill.textContent = h3.textContent.trim();
        if (pillsBox) pillsBox.appendChild(pill);
      }});

      const filterInput = document.getElementById('global-table-filter');
      const countBadge = document.getElementById('global-table-count');
      if (filterInput) {{
        filterInput.addEventListener('input', function() {{
          const q = filterInput.value.toLowerCase();
          let shown = 0;
          allTables.forEach(function(t) {{
            let tableVisible = 0;
            t.querySelectorAll('tbody tr').forEach(function(r) {{
              const match = r.textContent.toLowerCase().includes(q);
              r.style.display = match ? '' : 'none';
              if (match) {{ shown++; tableVisible++; }}
            }});
            const card = t.closest('.table-card');
            const prevH3 = card ? card.previousElementSibling : null;
            if (card) card.style.display = tableVisible > 0 ? '' : 'none';
            if (prevH3 && prevH3.tagName === 'H3') prevH3.style.display = tableVisible > 0 ? '' : 'none';
          }});
          if (countBadge) countBadge.textContent = shown + ' / ' + totalRows + ' files';
        }});
      }}
    }}

    // Build Right TOC with Scroll-Spy
    const toc = document.getElementById('on-this-page');
    const headings = Array.from(document.querySelectorAll('main.doc-main h2, main.doc-main h3'));
    const tocLinks = [];
    if (toc && headings.length > 0) {{
      const title = document.createElement('div');
      title.className = 'toc-header';
      title.textContent = 'On this page';
      toc.appendChild(title);
      const list = document.createElement('div');
      list.className = 'toc-list';
      headings.forEach(function(h) {{
        if (!h.id) {{
          h.id = h.textContent.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
        }}
        if (!h.id) return;
        const a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        if (h.tagName === 'H3') a.className = 'toc-h3';
        list.appendChild(a);
        tocLinks.push({{ id: h.id, el: a }});
      }});
      toc.appendChild(list);

      const obs = new IntersectionObserver(function(entries) {{
        entries.forEach(function(entry) {{
          if (entry.isIntersecting) {{
            tocLinks.forEach(function(item) {{
              item.el.classList.toggle('active', item.id === entry.target.id);
            }});
          }}
        }});
      }}, {{ rootMargin: '-75px 0px -75% 0px' }});
      headings.forEach(function(h) {{ if (h.id) obs.observe(h); }});
    }}

    // Mermaid Zoom & Fullscreen
    window.zoomMermaid = function(btn, delta) {{
      const card = btn.closest('.mermaid-card');
      const m = card ? card.querySelector('.mermaid') : null;
      if (!m) return;
      const cur = parseFloat(m.getAttribute('data-zoom') || '1');
      const next = delta === 0 ? 1 : Math.min(2.6, Math.max(0.4, cur + delta));
      m.setAttribute('data-zoom', String(next));
      m.style.transform = 'scale(' + next + ')';
    }};
    window.toggleMermaidFullscreen = function(btn) {{
      const card = btn.closest('.mermaid-card');
      if (card) card.classList.toggle('fullscreen');
    }};

    // Cross-document Command Palette (Cmd+K / Ctrl+K)
    const paletteItems = {search_index_json};
    if (paletteItems.length === 0) {{
      document.querySelectorAll('aside.sidebar a.nav-item').forEach(function(a) {{
        paletteItems.push({{ title: a.textContent.trim(), href: a.getAttribute('href'), kind: 'Page', snippet: '' }});
      }});
    }}
    headings.forEach(function(h) {{
      if (h.id) paletteItems.push({{ title: h.textContent.trim(), href: '#' + h.id, kind: 'On this page', snippet: '' }});
    }});

    let cmdSelectedIdx = 0;
    let currentCmdMatches = [];

    window.renderCmdItems = function(query) {{
      const box = document.getElementById('cmd-results');
      if (!box) return;
      const q = (query || '').toLowerCase();
      currentCmdMatches = paletteItems.filter(function(it) {{
        return it.title.toLowerCase().includes(q) || (it.snippet && it.snippet.toLowerCase().includes(q));
      }}).slice(0, 25);
      cmdSelectedIdx = 0;
      box.innerHTML = currentCmdMatches.map(function(it, i) {{
        const sel = i === cmdSelectedIdx ? ' selected' : '';
        const snip = it.snippet ? '<span class="cmd-item-snippet">' + it.snippet + '</span>' : '';
        return '<a class="cmd-item' + sel + '" href="' + it.href + '" onclick="closeCmdPalette()">' +
               '<div class="cmd-item-left"><span class="cmd-item-title">' + it.title + '</span>' + snip + '</div>' +
               '<span class="cmd-item-sub">' + it.kind + '</span></a>';
      }}).join('') || '<div style="padding:1.25rem;color:var(--muted-fg);font-size:0.86rem;text-align:center;">No matching results.</div>';
    }};
    window.openCmdPalette = function() {{
      const b = document.getElementById('cmd-backdrop');
      const inp = document.getElementById('cmd-input');
      if (b && inp) {{
        b.classList.add('open');
        inp.value = '';
        window.renderCmdItems('');
        setTimeout(function() {{ inp.focus(); }}, 20);
      }}
    }};
    window.closeCmdPalette = function() {{
      const b = document.getElementById('cmd-backdrop');
      if (b) b.classList.remove('open');
    }};
    window.filterCmdPalette = function(val) {{ window.renderCmdItems(val); }};
    document.addEventListener('keydown', function(e) {{
      const backdrop = document.getElementById('cmd-backdrop');
      const isOpen = backdrop && backdrop.classList.contains('open');
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {{
        e.preventDefault();
        if (isOpen) window.closeCmdPalette();
        else window.openCmdPalette();
      }} else if (isOpen && e.key === 'Escape') {{
        window.closeCmdPalette();
      }} else if (isOpen && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {{
        e.preventDefault();
        const links = document.querySelectorAll('#cmd-results a.cmd-item');
        if (!links.length) return;
        cmdSelectedIdx = (cmdSelectedIdx + (e.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
        links.forEach(function(el, i) {{ el.classList.toggle('selected', i === cmdSelectedIdx); }});
        links[cmdSelectedIdx].scrollIntoView({{ block: 'nearest' }});
      }} else if (isOpen && e.key === 'Enter') {{
        const links = document.querySelectorAll('#cmd-results a.cmd-item');
        if (links[cmdSelectedIdx]) {{
          e.preventDefault();
          links[cmdSelectedIdx].click();
        }}
      }}
    }});
  }})();
</script>
{mermaid_script}
</body>
</html>
"""

_MERMAID_SCRIPT = """<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  const isDark = document.documentElement.classList.contains('dark');
  mermaid.initialize({ startOnLoad: true, theme: isDark ? 'dark' : 'neutral' });
</script>"""


def _doc_info(name: str) -> tuple[str, str, str, str]:
    return _DOC_META.get(name, (name, "Documentation", "", "book"))


def _render_sidebar(nav_items: list[tuple[str, str]] | None, active_name: str | None) -> str:
    if not nav_items:
        return '<aside class="sidebar"><div class="nav-group"><div class="nav-group-title">Documentation</div></div></aside>'

    groups: dict[str, list[str]] = {}
    for name, href in nav_items:
        label, category, _desc, icon_key = _doc_info(name)
        icon_svg = _ICONS.get(icon_key, _ICONS["book"])
        cls = "nav-item active" if name == active_name else "nav-item"
        item_html = (
            f'<a href="{html_lib.escape(href)}" class="{cls}">'
            f"{icon_svg}"
            f"<span>{html_lib.escape(label)}</span>"
            "</a>"
        )
        groups.setdefault(category, []).append(item_html)

    group_blocks = []
    for category, items in groups.items():
        group_blocks.append(
            f'<div class="nav-group"><div class="nav-group-title">{html_lib.escape(category)}</div>'
            + "".join(items)
            + "</div>"
        )
    return f'<aside class="sidebar">{"".join(group_blocks)}</aside>'


def _render_bento(nav_items: list[tuple[str, str]] | None, active_name: str | None) -> str:
    if not nav_items or active_name != "README" or len(nav_items) <= 1:
        return ""
    cards = []
    for name, href in nav_items:
        if name in ("README", "githubReadme"):
            continue
        label, _category, desc, icon_key = _doc_info(name)
        icon_svg = _ICONS.get(icon_key, _ICONS["book"])
        cards.append(
            f'<a class="bento-card" href="{html_lib.escape(href)}">'
            f'<div class="bento-card-head"><span>{html_lib.escape(label)}</span>{icon_svg}</div>'
            f'<div class="bento-card-desc">{html_lib.escape(desc)}</div>'
            "</a>"
        )
    return f'<div class="bento-grid">{"".join(cards)}</div>' if cards else ""


def _render_pager(nav_items: list[tuple[str, str]] | None, active_name: str | None) -> str:
    if not nav_items or not active_name or len(nav_items) < 2:
        return ""
    names = [n for n, _ in nav_items]
    if active_name not in names:
        return ""
    idx = names.index(active_name)
    prev_html = "<div></div>"
    next_html = "<div></div>"
    if idx > 0:
        p_name, p_href = nav_items[idx - 1]
        p_label, _, _, _ = _doc_info(p_name)
        prev_html = (
            f'<a class="pager-card prev" href="{html_lib.escape(p_href)}">'
            '<span class="pager-dir">← Previous</span>'
            f'<span class="pager-title">{html_lib.escape(p_label)}</span></a>'
        )
    if idx + 1 < len(nav_items):
        n_name, n_href = nav_items[idx + 1]
        n_label, _, _, _ = _doc_info(n_name)
        next_html = (
            f'<a class="pager-card next" href="{html_lib.escape(n_href)}">'
            '<span class="pager-dir">Next →</span>'
            f'<span class="pager-title">{html_lib.escape(n_label)}</span></a>'
        )
    return f'<div class="pager">{prev_html}{next_html}</div>'


def _convert_mermaid_blocks(body_html: str) -> tuple[str, bool]:
    has_mermaid = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal has_mermaid
        has_mermaid = True
        return (
            '<div class="mermaid-card">'
            '<div class="mermaid-toolbar">'
            "<span>Dependency Graph (Interactive Canvas)</span>"
            '<div class="mermaid-actions">'
            '<button type="button" class="mermaid-btn" onclick="zoomMermaid(this, -0.2)">−</button>'
            '<button type="button" class="mermaid-btn" onclick="zoomMermaid(this, 0)">100%</button>'
            '<button type="button" class="mermaid-btn" onclick="zoomMermaid(this, 0.2)">+</button>'
            '<button type="button" class="mermaid-btn" onclick="toggleMermaidFullscreen(this)">Fullscreen</button>'
            "</div></div>"
            f'<div class="mermaid-viewport"><div class="mermaid">{match.group(1)}</div></div>'
            "</div>"
        )

    return _MERMAID_BLOCK_RE.sub(_replace, body_html), has_mermaid


def build_search_index(documents: dict[str, str], *, url_pattern: str = "{name}.html") -> list[dict[str, str]]:
    """Build a cross-document search index of pages and headings for the Cmd+K palette."""
    entries: list[dict[str, str]] = []
    for name in ordered_doc_names(list(documents.keys())):
        text = documents[name]
        label, category, desc, _ = _doc_info(name)
        base_href = url_pattern.format(name=name)
        entries.append({"title": label, "href": base_href, "kind": category, "snippet": desc})
        for match in _HEADING_RE.finditer(text):
            heading = match.group(1).strip()
            if heading.lower() == "table of contents":
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            entries.append(
                {
                    "title": heading,
                    "href": f"{base_href}#{slug}",
                    "kind": label,
                    "snippet": f"Section in {label}",
                }
            )
    return entries


def render_docs_page(
    title: str,
    markdown_text: str,
    *,
    nav_items: list[tuple[str, str]] | None = None,
    active_name: str | None = None,
    search_index: list[dict[str, str]] | None = None,
) -> str:
    """Render a Markdown document into the shadcn/ui-styled documentation shell."""
    body = markdown_lib.markdown(markdown_text, extensions=["fenced_code", "tables"])
    body, has_mermaid = _convert_mermaid_blocks(body)
    friendly_title, category, subtitle, _icon = _doc_info(active_name or title)
    sidebar_html = _render_sidebar(nav_items, active_name)
    bento_html = _render_bento(nav_items, active_name)
    pager_html = _render_pager(nav_items, active_name)
    mermaid_script = _MERMAID_SCRIPT if has_mermaid else ""
    return _PAGE_TEMPLATE.format(
        title=html_lib.escape(friendly_title),
        friendly_title=html_lib.escape(friendly_title),
        category=html_lib.escape(category),
        subtitle=html_lib.escape(subtitle),
        sidebar_html=sidebar_html,
        bento_html=bento_html,
        body=body,
        pager_html=pager_html,
        search_index_json=json.dumps(search_index or []),
        mermaid_script=mermaid_script,
    )


def render_html(title: str, markdown_text: str) -> str:
    """Render a single document's Markdown as a complete, styled HTML page."""
    return render_docs_page(title, markdown_text)


def ordered_doc_names(names: list[str]) -> list[str]:
    """Return document names sorted in canonical documentation order."""
    ordered = [n for n in _CANONICAL_ORDER if n in names]
    ordered.extend(sorted(n for n in names if n not in ordered))
    return ordered


def write_html(documents: dict[str, str], output_dir: Path) -> list[Path]:
    """Write each document as `{output_dir}/{name}.html` with shared sidebar navigation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ordered_doc_names(list(documents.keys()))
    nav_items = [(name, f"{name}.html") for name in names]
    search_index = build_search_index(documents, url_pattern="{name}.html")
    written: list[Path] = []
    for name in names:
        path = output_dir / f"{name}.html"
        path.write_text(
            render_docs_page(
                name,
                documents[name],
                nav_items=nav_items,
                active_name=name,
                search_index=search_index,
            ),
            encoding="utf-8",
        )
        written.append(path)
    return written
