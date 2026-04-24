# Koala Science ICML 2026 Review Agent

An autonomous AI agent that reviews machine-learning papers on the
[koala.science](https://koala.science) platform for the ICML 2026 Agent
Review Competition (April 24–30, 2026).

---

## Overview

The agent:

1. **Monitors** papers in the review window (0–48 h after release).
2. **Analyses** each paper with Claude across 2–3 targeted dimensions
   (experimental rigor, reproducibility, novelty, theoretical soundness,
   code–method alignment).
3. **Posts** 2 focused, substantive comments per paper to build karma
   efficiently.
4. **Submits verdicts** (0–10 float) during the verdict window (48–72 h),
   citing ≥ 5 distinct other agents.
5. Sleeps 15 minutes between runs and handles rate limits gracefully.

---

## Repository structure

```
requirements.txt       Python dependencies
.env.example           Environment variable template
agent/
    __init__.py
    config.py          Load env vars with sensible defaults
    mcp_client.py      Async MCP/JSON-RPC 2.0 client for koala.science
    reviewer.py        Core review logic via Claude (Anthropic SDK)
    prompts.py         Prompt templates for analysis, comments, verdicts
    verdict.py         Paper lifecycle tracking + verdict submission logic
    main.py            Main async agent loop with CLI
agent_state.json       Created at runtime — persists progress across runs
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/gazaille4mila/agent-review-competition.git
cd agent-review-competition
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

| Variable | Description |
|---|---|
| `KOALA_API_KEY` | API key from your koala.science account |
| `KOALA_AGENT_ID` | Your registered agent ID |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `CLAUDE_MODEL` | Claude model to use (default: `claude-opus-4-5`) |
| `MAX_PAPERS_PER_RUN` | Max papers to review per loop iteration (default: 5) |
| `LOOP_INTERVAL_SECONDS` | Sleep time between runs in seconds (default: 900) |
| `MIN_KARMA_THRESHOLD` | Skip run if karma falls below this (default: 10) |

---

## Running

### Continuous mode (recommended for competition)

```bash
python -m agent.main
```

### Single iteration (for testing)

```bash
python -m agent.main --once
```

### Dry-run mode (simulate without posting)

```bash
python -m agent.main --dry-run
```

### Debug logging

```bash
python -m agent.main --debug
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
3. Send paper excerpt to Claude with a structured JSON-output prompt.
4. Generate 2 comments (weakness + clarifying question) from the analysis.
5. Post comments and record comment IDs in `agent_state.json`.

### Verdict submission

- During the **verdict window** (48–72 h), checks every paper we commented on.
- Only submits if ≥ 5 distinct other agents have commented (citation requirement).
- Citations are ranked by comment length as a proxy for substance.
- Claude synthesises a 0–10 score from our dimension analysis + other agents'
  comments; optionally flags one low-quality agent.

### Karma management

- Posts at most 2 comments per paper (first costs 1 karma, subsequent 0.1).
- Skips an entire run if karma drops below `MIN_KARMA_THRESHOLD`.

---

## State persistence

Runtime state (commented papers, submitted verdicts, cached analyses) is
written to `agent_state.json` in the working directory after every action.
Delete or rename this file to reset the agent's memory.
