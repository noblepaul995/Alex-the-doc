"""
Live HTTP endpoint smoke-tester driven by `agents.api_agent.detect_endpoints`.

Splits planning (`build_test_plan`) from execution (`run_test_plan`) so
guardrails — read-only by default, unresolved path-parameter skipping,
and `--dry-run` preview — are pure, deterministic, and unit-testable
without network traffic.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from agents.api_agent import EndpointCandidate, extract_method_and_path

_SAFE_METHODS = frozenset({"GET", "HEAD"})
_PATH_PARAM_RE = re.compile(r"\{([A-Za-z_]\w*)\}|:([A-Za-z_]\w*)")


@dataclass
class ApiTestCase:
    method: str
    route_template: str
    resolved_path: str | None
    status: str  # "PENDING" | "SKIPPED"
    skip_reason: str | None = None
    source: str = ""


@dataclass
class ApiTestResult:
    case: ApiTestCase
    status_code: int | None = None
    latency_ms: float | None = None
    outcome: str = "SKIPPED"  # "PASS" | "FAIL" | "SKIPPED" | "ERROR"
    detail: str = ""


def parse_kv_flags(items: list[str], *, sep: str = "=", label: str = "option") -> dict[str, str]:
    """Parse repeatable `key=value` or `Key: Value` CLI options into a dict."""
    parsed: dict[str, str] = {}
    for raw in items:
        if sep not in raw:
            raise ValueError(f"Invalid {label} {raw!r}: expected '{sep}' separator.")
        key, value = raw.split(sep, 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid {label} {raw!r}: empty key.")
        parsed[key] = value
    return parsed


def substitute_path_params(route_path: str, params: dict[str, str]) -> tuple[str | None, list[str]]:
    """
    Replace `{param}` and `:param` placeholders in `route_path` using `params`.
    Returns `(resolved_path, missing_param_names)`. When any parameter is
    missing, `resolved_path` is `None`.
    """
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in params:
            return params[name]
        if name not in missing:
            missing.append(name)
        return match.group(0)

    resolved = _PATH_PARAM_RE.sub(_replace, route_path)
    if missing:
        return None, missing
    return resolved, []


def build_test_plan(
    candidates: list[EndpointCandidate],
    *,
    include_writes: bool = False,
    params: dict[str, str] | None = None,
) -> list[ApiTestCase]:
    """
    Convert discovered `EndpointCandidate`s into an ordered list of
    `ApiTestCase`s with read-only and path-parameter guardrails applied.
    """
    param_map = params or {}
    plan: list[ApiTestCase] = []
    seen: set[tuple[str, str]] = set()

    for candidate in sorted(candidates, key=lambda c: (c.file_path, c.symbol.name)):
        method, route_path = extract_method_and_path(candidate)
        if not method or not route_path:
            continue
        key = (method, route_path)
        if key in seen:
            continue
        seen.add(key)

        source = f"{candidate.file_path}::{candidate.symbol.name}"

        if method not in _SAFE_METHODS and not include_writes:
            plan.append(
                ApiTestCase(
                    method=method,
                    route_template=route_path,
                    resolved_path=None,
                    status="SKIPPED",
                    skip_reason="write method skipped (pass --include-writes)",
                    source=source,
                )
            )
            continue

        resolved, missing = substitute_path_params(route_path, param_map)
        if missing:
            plan.append(
                ApiTestCase(
                    method=method,
                    route_template=route_path,
                    resolved_path=None,
                    status="SKIPPED",
                    skip_reason=f"missing path param(s): {', '.join(missing)}",
                    source=source,
                )
            )
            continue

        plan.append(
            ApiTestCase(
                method=method,
                route_template=route_path,
                resolved_path=resolved,
                status="PENDING",
                source=source,
            )
        )

    return plan


async def run_test_plan(
    plan: list[ApiTestCase],
    base_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[ApiTestResult]:
    """Execute all `PENDING` test cases against `base_url` via `httpx.AsyncClient`."""
    clean_base = base_url.rstrip("/")
    results: list[ApiTestResult] = []

    async with httpx.AsyncClient(
        base_url=clean_base,
        headers=headers or {},
        timeout=timeout,
        transport=transport,
    ) as client:
        for case in plan:
            if case.status == "SKIPPED" or case.resolved_path is None:
                results.append(
                    ApiTestResult(
                        case=case,
                        outcome="SKIPPED",
                        detail=case.skip_reason or "skipped",
                    )
                )
                continue

            target_path = case.resolved_path if case.resolved_path.startswith("/") else f"/{case.resolved_path}"
            start = time.perf_counter()
            try:
                response = await client.request(case.method, target_path)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                outcome = "PASS" if 200 <= response.status_code < 400 else "FAIL"
                results.append(
                    ApiTestResult(
                        case=case,
                        status_code=response.status_code,
                        latency_ms=elapsed_ms,
                        outcome=outcome,
                        detail=response.reason_phrase or "",
                    )
                )
            except httpx.HTTPError as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                results.append(
                    ApiTestResult(
                        case=case,
                        latency_ms=elapsed_ms,
                        outcome="ERROR",
                        detail=str(exc),
                    )
                )

    return results
