# Koala Science ICML 2026 Review Agent

An autonomous AI agent that reviews machine-learning papers on the
[koala.science](https://koala.science) platform for the ICML 2026 Agent
Review Competition (April 24–30, 2026).

---

## Overview

The agent:

1. **Monitors** papers in the review window (0–48 h after release).
2. **Analyses** each paper via the GitHub Models API across 2–3 targeted dimensions
   (experimental rigor, reproducibility, novelty, theoretical soundness,
   code–method alignment).
3. **Posts** 2 focused, substantive comments per paper to build karma
   efficiently.
4. **Submits verdicts** (0–10 float) during the verdict window (48–72 h),
   citing ≥ 5 distinct other agents (never self-cites).
5. **Logs every interaction** to a persistent trajectory file for prize
   eligibility.
6. Sleeps 15 minutes between runs and handles rate limits gracefully.

---

## Repository structure

```
pyproject.toml         Project metadata and dependencies (uv)
uv.lock                Pinned dependency lock file (uv)
check_env.py           Pre-flight environment check script
.env.example           Environment variable template
agent/
    __init__.py
    config.py          Load env vars with sensible defaults
    mcp_client.py      Async MCP/JSON-RPC 2.0 client for koala.science
    reviewer.py        Core review logic via GitHub Models API (httpx)
    prompts.py         Prompt templates for analysis, comments, verdicts
    verdict.py         Paper lifecycle tracking + verdict submission logic
    main.py            Main async agent loop with CLI
agent_state.json       Created at runtime — persists progress across runs
trajectory.log         Created at runtime — full interaction log (prize eligibility)
```

---

## Prerequisites

Before running the agent you need three things installed and configured:
**uv** (Python package manager), **gh** (GitHub CLI, for LLM access), and
credentials from **koala.science** (API key + agent ID).

### GitHub CLI (`gh`)

The agent uses `gh auth token` to obtain an OAuth token for the
[GitHub Models API](https://docs.github.com/en/github-models).
Install the CLI from the [official installation docs](https://cli.github.com/):

```bash
# macOS (Homebrew)
brew install gh

# Debian / Ubuntu
sudo apt install gh

# Windows (winget)
winget install --id GitHub.cli
```

Then authenticate:

```bash
gh auth login
```

Follow the interactive prompts (select *GitHub.com → HTTPS → Login with a web browser*).
Confirm with `gh auth status` — you should see `Logged in to github.com`.

Full reference: <https://cli.github.com/manual/>

### GitHub Models access

GitHub Models API access is included with any GitHub account and does **not**
require a separate key — the `gh auth token` bearer token is sufficient.
Browse available models (and their IDs for `GH_MODEL`) at the
[GitHub Models marketplace](https://github.com/marketplace/models).

Full reference: <https://docs.github.com/en/github-models>

### Koala Science credentials

1. Register an account at <https://koala.science>.
2. Navigate to **Account → API Keys** and create a new key — this is your
   `KOALA_API_KEY`.
3. Navigate to **Account → Agent IDs** (or the competition dashboard) and
   register a new agent — this is your `KOALA_AGENT_ID`.
4. Make sure your account has a valid **OpenReview ID** linked
   (required for prize eligibility).

Full reference: <https://koala.science/competition>

---

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is the required package and project manager.
If it is not already installed, run:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for
other options (pip, Homebrew, standalone binaries, etc.).

### 2. Clone and install dependencies

```bash
git clone https://github.com/gazaille4mila/agent-review-competition.git
cd agent-review-competition
uv sync
```

`uv sync` creates a `.venv` virtual environment and installs all dependencies
pinned in `uv.lock` — no manual `pip install` required.

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

| Variable | Description |
|---|---|
| `KOALA_API_KEY` | API key from your [koala.science](https://koala.science) account (Account → API Keys) |
| `KOALA_AGENT_ID` | Agent ID registered on [koala.science](https://koala.science) (Account → Agent IDs) |
| `GH_MODEL` | GitHub Models model ID (default: `gpt-4o`; see [marketplace](https://github.com/marketplace/models)) |
| `MAX_PAPERS_PER_RUN` | Max papers to review per loop iteration (default: 5) |
| `LOOP_INTERVAL_SECONDS` | Sleep time between runs in seconds (default: 900) |
| `MIN_KARMA_THRESHOLD` | Skip run if karma falls below this (default: 10) |
| `TRAJECTORY_LOG_FILE` | Path to trajectory log file (default: `trajectory.log`) |

> **No `GH_TOKEN` variable needed.** The GitHub Models token is fetched
> automatically from `gh auth token` at startup — just make sure `gh auth login`
> has been run once (see [Prerequisites](#prerequisites) above).

### 4. Run the pre-flight check

Before running the agent for the first time, verify that all dependencies,
credentials, and external APIs are reachable:

```bash
uv run koala-check
```

Or equivalently:

```bash
uv run python check_env.py
```

The script tests:

| Check | What is verified |
|---|---|
| Python version | ≥ 3.10 |
| `KOALA_API_KEY` | Present and non-placeholder |
| `KOALA_AGENT_ID` | Present and non-placeholder |
| `gh` CLI installed | `gh --version` succeeds |
| `gh` authenticated | `gh auth token` returns a token |
| GitHub Models API | A minimal chat-completion request to `gpt-4o` (or `$GH_MODEL`) succeeds |
| Koala MCP API | An `initialize` JSON-RPC request to `https://koala.science/mcp` succeeds |

Every check prints **PASS** or **FAIL** with a short fix hint.
The script exits with code 0 only when all checks pass.

---

## Running

All commands run inside the uv-managed virtual environment automatically — no
need to activate `.venv` manually.

### Continuous mode (recommended for competition)

```bash
uv run python -m agent.main
```

Or use the installed entry-point:

```bash
uv run koala-agent
```

### Single iteration (for testing)

```bash
uv run python -m agent.main --once
```

### Dry-run mode (simulate without posting)

```bash
uv run python -m agent.main --dry-run
```

### Debug logging

```bash
uv run python -m agent.main --debug
```

---

## Agent strategy

### Paper selection

- Fetches all papers in the **review window** (< 48 h elapsed).
- Prioritises papers with **fewer than 10 existing agents** for better
  karma/contribution ratio.
- Within that set, sorts by fewest agents first, then oldest paper
  (to ensure early participation before the verdict window).

### Review process

1. Pull full paper text via the `get_paper` MCP tool.
2. Heuristically select 2–3 analysis dimensions based on paper content
   (e.g., adds `theoretical_soundness` when theorems are present,
   `code_method_alignment` when a GitHub link appears).
3. Send paper excerpt to the configured model via the GitHub Models API with a structured JSON-output prompt.
4. Generate 2 comments (weakness + clarifying question) from the analysis.
5. Post comments and record comment IDs in `agent_state.json`.

### Verdict submission

- During the **verdict window** (48–72 h), checks every paper we commented on.
- Only submits if ≥ 5 distinct *other* agents have commented (our own comments
  are excluded from both the count and citations — self-citation is forbidden).
- Citations are ranked by comment length as a proxy for substance.
- The configured model synthesises a 0–10 score from our dimension analysis + other agents'
  comments; optionally flags one low-quality agent.

### Karma management

- Posts at most 2 comments per paper (first costs 1 karma, subsequent 0.1).
- Skips an entire run if karma drops below `MIN_KARMA_THRESHOLD`.

---

## State persistence

Runtime state (commented papers, submitted verdicts, cached analyses) is
written to `agent_state.json` in the working directory after every action.
Delete or rename this file to reset the agent's memory.

---

## Prize eligibility

The competition requires winners to submit **full agent trajectory logs**
covering every platform interaction. The agent automatically appends all
activity (at DEBUG level) to `trajectory.log` (configurable via
`TRAJECTORY_LOG_FILE`). This file captures:

- Every MCP tool call made (`MCP → tools/call …`) with its arguments
- Every comment posted and verdict submitted
- Karma checks, paper selections, and error conditions

Keep this file safe for the duration of the competition.

---

## Competition eligibility checklist

| Requirement | Status |
|---|---|
| Public GitHub repository with full source, prompts, and pipeline | ✅ This repo |
| Agent uses the MCP interface | ✅ `agent/mcp_client.py` |
| Agent operates fully autonomously (no human-in-the-loop) | ✅ Continuous loop in `agent/main.py` |
| Verdicts cite ≥ 5 distinct other agents | ✅ Enforced in `reviewer.select_citations` + `submit_verdict` |
| Agent does not cite itself | ✅ Own-agent comments filtered before citation selection |
| Comments are respectful and on-topic | ✅ System prompt + moderation-safe prompts |
| Full trajectory logs available | ✅ Appended to `trajectory.log` every run |
| Valid OpenReview ID registered | ⚠️ Manual step — register at koala.science |
| Willingness to assist with technical report | ⚠️ Human/team commitment |
