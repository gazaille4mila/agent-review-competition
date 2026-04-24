#!/usr/bin/env python3
"""
Pre-flight environment check for the Koala Science review agent.

Run this before starting the autonomous agent to verify that all required
tools, credentials, and external APIs are reachable.

Usage::

    uv run python check_env.py

Each check prints PASS or FAIL with a short explanation.  The script exits
with code 0 only when every check passes.

References
----------
- Koala Science MCP endpoint: https://koala.science/mcp
- GitHub Models API: https://docs.github.com/en/github-models/prototyping-with-ai-models
- GitHub Copilot CLI: https://github.com/github/copilot-cli
- uv project manager: https://docs.astral.sh/uv/
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid

# ---------------------------------------------------------------------------
# Optional: load .env so the checks work in a local dev setup
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not yet installed — that's fine for the version check

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

_passed: list[str] = []
_failed: list[str] = []


def _ok(label: str, detail: str = "") -> None:
    msg = f"  {GREEN}PASS{RESET}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _passed.append(label)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  {RED}FAIL{RESET}  {label}"
    if detail:
        msg += f"\n         {YELLOW}→ {detail}{RESET}"
    print(msg)
    _failed.append(label)


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------

def check_python_version() -> None:
    _section("Python version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 10):
        _ok("Python ≥ 3.10", version_str)
    else:
        _fail("Python ≥ 3.10", f"found {version_str} — upgrade to 3.10+")


# ---------------------------------------------------------------------------
# 2. Required environment variables
# ---------------------------------------------------------------------------

def check_env_vars() -> None:
    _section("Environment variables")
    required = {
        "KOALA_API_KEY": "API key from your koala.science account",
        "KOALA_AGENT_ID": "registered agent ID from koala.science",
    }
    for var, hint in required.items():
        val = os.getenv(var, "")
        if val and val not in ("your_koala_api_key_here", "your_agent_id_here"):
            _ok(var, f"{val[:4]}…{'*' * max(0, len(val) - 4)}")
        else:
            _fail(var, f"not set — {hint}")


# ---------------------------------------------------------------------------
# 3. gh CLI installation and authentication
# ---------------------------------------------------------------------------

def check_gh_cli() -> tuple[str | None, bool]:
    """Return (token, ok) where ok=True if the CLI is usable."""
    _section("GitHub CLI (gh)")

    # 3a. Is gh installed?
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True, text=True, timeout=10
        )
        version_line = result.stdout.splitlines()[0] if result.stdout else "?"
        _ok("gh installed", version_line.strip())
    except FileNotFoundError:
        _fail("gh installed", "not found — install from https://cli.github.com/")
        return None, False
    except subprocess.TimeoutExpired:
        _fail("gh installed", "timed out")
        return None, False

    # 3b. Is gh authenticated?
    try:
        token_result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=10
        )
        token = token_result.stdout.strip()
        if token:
            _ok("gh authenticated", f"token {token[:4]}…")
            return token, True
        else:
            _fail(
                "gh authenticated",
                "no token — run `gh auth login` (or `gh auth login --with-token`)"
            )
            return None, False
    except subprocess.TimeoutExpired:
        _fail("gh authenticated", "timed out")
        return None, False


# ---------------------------------------------------------------------------
# 4. GitHub Models API
# ---------------------------------------------------------------------------

async def check_github_models(token: str) -> None:
    _section("GitHub Models API")
    import httpx

    endpoint = "https://models.inference.ai.azure.com/chat/completions"
    model = os.getenv("GH_MODEL", "gpt-4o")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 5,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            _ok(f"GitHub Models API ({model})", f"response: {reply!r}")
        elif resp.status_code == 401:
            _fail(
                f"GitHub Models API ({model})",
                f"HTTP 401 — token not authorised for GitHub Models"
            )
        elif resp.status_code == 404:
            _fail(
                f"GitHub Models API ({model})",
                f"HTTP 404 — model {model!r} not found; check GH_MODEL env var"
            )
        else:
            _fail(
                f"GitHub Models API ({model})",
                f"HTTP {resp.status_code}: {resp.text[:120]}"
            )
    except httpx.ConnectError:
        _fail(
            f"GitHub Models API ({model})",
            "connection refused — check network / firewall"
        )
    except httpx.TimeoutException:
        _fail(f"GitHub Models API ({model})", "request timed out after 30 s")


# ---------------------------------------------------------------------------
# 5. Koala Science MCP API
# ---------------------------------------------------------------------------

async def check_koala_mcp() -> None:
    _section("Koala Science MCP API")
    import httpx

    api_key = os.getenv("KOALA_API_KEY", "")
    agent_id = os.getenv("KOALA_AGENT_ID", "")
    endpoint = os.getenv("MCP_ENDPOINT", "https://koala.science/mcp")

    if not api_key or api_key == "your_koala_api_key_here":
        _fail("Koala MCP reachable", "KOALA_API_KEY not set — skipping")
        return

    body = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "koala-review-agent-preflight",
                "version": "1.0.0",
                "agentId": agent_id,
            },
        },
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                endpoint,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
        if resp.status_code in (200, 201):
            _ok("Koala MCP reachable", f"HTTP {resp.status_code}")

            # Try to parse the session ID as an extra sanity-check
            content_type = resp.headers.get("content-type", "")
            session_id: str | None = None
            if "text/event-stream" in content_type or resp.text.lstrip().startswith("data:"):
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            payload = json.loads(raw)
                            result = payload.get("result", {})
                            session_id = result.get("sessionId") or result.get("session_id")
                        except (json.JSONDecodeError, AttributeError):
                            pass
            else:
                try:
                    payload = resp.json()
                    result = payload.get("result", {})
                    session_id = (result or {}).get("sessionId") or (result or {}).get("session_id")
                except Exception:
                    pass

            if session_id:
                _ok("Koala MCP session", f"session_id={session_id[:8]}…")
            else:
                _ok("Koala MCP session", "no session_id in response (may be normal)")
        elif resp.status_code == 401:
            _fail("Koala MCP reachable", "HTTP 401 — KOALA_API_KEY rejected")
        elif resp.status_code == 403:
            _fail("Koala MCP reachable", "HTTP 403 — agent not authorised")
        else:
            _fail("Koala MCP reachable", f"HTTP {resp.status_code}: {resp.text[:120]}")
    except httpx.ConnectError:
        _fail("Koala MCP reachable", "connection refused — check network / firewall")
    except httpx.TimeoutException:
        _fail("Koala MCP reachable", "request timed out after 20 s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _async_main() -> int:
    print(f"\n{BOLD}Koala Science review-agent — pre-flight checks{RESET}")
    print("=" * 52)

    check_python_version()
    check_env_vars()
    token, gh_ok = check_gh_cli()

    if gh_ok and token:
        await check_github_models(token)
    else:
        _section("GitHub Models API")
        _fail("GitHub Models API", "skipped — gh CLI not available")

    await check_koala_mcp()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 52}")
    total = len(_passed) + len(_failed)
    if _failed:
        print(
            f"{BOLD}{RED}Result: {len(_failed)}/{total} checks failed{RESET}\n"
            f"Fix the issues above before running `uv run koala-agent`."
        )
        return 1
    else:
        print(
            f"{BOLD}{GREEN}Result: all {total} checks passed ✓{RESET}\n"
            f"Ready to run: uv run koala-agent"
        )
        return 0


def main() -> None:
    sys.exit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
