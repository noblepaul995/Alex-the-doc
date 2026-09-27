"""
GitHub repository handling for `alex github`: parse a repo reference in
whatever form the user pasted it, check via the real GitHub API whether
it's public or private and whether the caller can actually see it, and
clone it locally so the rest of the pipeline can run against it exactly
like any other local repository path.

Deliberately thin — this module's only job is "can I see this repo, and
can I get a local copy of it," not general GitHub API access. No new
dependency: uses `httpx` (already a dependency for every LLM provider)
for the API call and shells out to the user's own `git` for cloning,
the same way `agents/changelog_agent.py` shells out to `git log` rather
than reimplementing Git.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

_GITHUB_URL_PATTERNS = [
    # https://github.com/owner/repo, https://github.com/owner/repo.git,
    # with or without a trailing slash or further path (e.g. /tree/main).
    re.compile(r"^(?:https?://)?(?:www\.)?github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?(?:[/?#].*)?$"),
    # git@github.com:owner/repo.git (SSH form).
    re.compile(r"^git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?/?$"),
    # Bare "owner/repo" shorthand.
    re.compile(r"^([\w.\-]+)/([\w.\-]+?)(?:\.git)?$"),
]


def parse_github_url(text: str) -> tuple[str, str] | None:
    """Extract `(owner, repo)` from any of the common ways someone might paste a GitHub repo reference, or `None` if it doesn't match any of them."""
    stripped = text.strip()
    for pattern in _GITHUB_URL_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.group(1), match.group(2)
    return None


@dataclass
class RepoAccessResult:
    owner: str
    repo: str
    exists: bool
    """False could mean the repo genuinely doesn't exist, or it's private and the token (if any) can't see it — GitHub's API deliberately returns 404 for both, to avoid confirming a private repo's existence to someone without access."""
    private: bool | None
    """`None` when `exists` is False, since GitHub doesn't reveal this for a repo the caller can't see."""
    default_branch: str | None
    detail: str


async def check_repo_access(owner: str, repo: str, token: str | None = None) -> RepoAccessResult:
    """
    Query the real GitHub API for whether `owner/repo` exists and is
    visible with the given token (or anonymously, if `token` is None).
    Never raises on a 404 — that's an expected, meaningful result here
    (private-and-inaccessible, or nonexistent), not an error condition.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
    except httpx.HTTPError as exc:
        return RepoAccessResult(owner, repo, exists=False, private=None, default_branch=None, detail=f"network error: {exc}")

    if response.status_code == 200:
        data = response.json()
        return RepoAccessResult(
            owner=owner,
            repo=repo,
            exists=True,
            private=data.get("private", False),
            default_branch=data.get("default_branch"),
            detail="",
        )

    if response.status_code == 404:
        detail = (
            "not found, or private and inaccessible with the token provided"
            if token
            else "not found, or private (a token is needed to tell the difference)"
        )
        return RepoAccessResult(owner, repo, exists=False, private=None, default_branch=None, detail=detail)

    if response.status_code in (401, 403):
        return RepoAccessResult(owner, repo, exists=False, private=None, default_branch=None, detail=f"token rejected (HTTP {response.status_code})")

    return RepoAccessResult(owner, repo, exists=False, private=None, default_branch=None, detail=f"unexpected HTTP {response.status_code}")


def clone_repo(owner: str, repo: str, dest: Path, *, token: str | None = None, branch: str | None = None) -> Path:
    """
    Shallow-clone (`--depth 1`) `owner/repo` into `dest`, returning the
    path actually cloned to. Raises `RuntimeError` (with the token
    redacted from the message, even though it's also never passed
    through `shell=True` so it can't leak via a shell history either) on
    failure — the caller decides how to present that to the user.
    """
    if token:
        clone_url = f"https://{token}@github.com/{owner}/{repo}.git"
    else:
        clone_url = f"https://github.com/{owner}/{repo}.git"

    dest.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if branch:
        command += ["--branch", branch]
    command += [clone_url, str(dest)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        redacted_stderr = result.stderr.replace(token, "***") if token else result.stderr
        raise RuntimeError(f"git clone failed: {redacted_stderr.strip()}")
    return dest