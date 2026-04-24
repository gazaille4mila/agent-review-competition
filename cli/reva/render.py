"""
Renderers for ATIF steps → terminal / Textual Rich Text / HTML.

These functions are intentionally pure: given one ATIF step (a dict) they
produce display output. The same step dict drives `reva log`, `reva view`,
and `reva view --web` so rendering decisions stay consistent.
"""

from __future__ import annotations

import html
import json
import textwrap
from typing import Any

import click
from rich.text import Text


# --------------------------------------------------------------------------- #
# tool-call summary (shared)
# --------------------------------------------------------------------------- #


def summarize_tool_call(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("function_name", "?")
    args = tool_call.get("arguments") or {}
    if not isinstance(args, dict):
        return str(args)[:200]

    if name == "Bash":
        return (args.get("command") or "").strip()
    if name == "WebFetch":
        return args.get("url") or ""
    if name in ("Write", "Edit", "Read"):
        return args.get("file_path") or ""
    if name == "Skill":
        return args.get("skill") or ""
    if name in ("Grep", "Glob"):
        return args.get("pattern") or args.get("query") or ""
    return json.dumps(args, ensure_ascii=False)[:200]


TOOL_COLORS_RICH: dict[str, str] = {
    "Bash": "dark_orange",
    "Read": "steel_blue",
    "Write": "medium_orchid",
    "Edit": "medium_orchid",
    "WebFetch": "turquoise2",
    "WebSearch": "turquoise2",
    "Grep": "khaki1",
    "Glob": "khaki1",
    "Skill": "spring_green2",
}

# click only supports the base 16 ANSI colors, so we use a coarser mapping.
TOOL_COLORS_CLICK: dict[str, str] = {
    "Bash": "bright_yellow",
    "Read": "blue",
    "Write": "magenta",
    "Edit": "magenta",
    "WebFetch": "cyan",
    "WebSearch": "cyan",
    "Grep": "yellow",
    "Glob": "yellow",
    "Skill": "bright_green",
}


# --------------------------------------------------------------------------- #
# terminal (click-styled) rendering
# --------------------------------------------------------------------------- #


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    lines = text.splitlines()
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width, subsequent_indent=indent) or [""])
    return "\n".join(wrapped)


def render_step_terminal(step: dict[str, Any], agent_name: str | None = None) -> list[str]:
    """Render one ATIF step as a list of click-styled terminal lines."""
    tag = f"[{agent_name[:28]}] " if agent_name else ""
    source = step.get("source")
    out: list[str] = []

    if source == "system":
        msg = step.get("message", "")
        if msg.startswith("session started"):
            out.append(click.style(f"\n{tag}▶ {msg}", fg="green", bold=True))
        elif msg.startswith("session ended"):
            out.append(click.style(f"\n{tag}■ {msg}\n", fg="red", bold=True))
        elif msg.startswith("rate limit"):
            out.append(click.style(f"{tag}⚠ {msg}", fg="magenta"))
        elif msg.startswith("[reva]"):
            out.append(click.style(f"{tag}{msg}", fg="bright_black", bold=True))
        else:
            out.append(click.style(f"{tag}{msg}", fg="bright_black"))
        return out

    if source == "user":
        msg = (step.get("message") or "").strip()
        if msg:
            out.append(click.style(f"\n{tag}▸ user", fg="bright_blue", bold=True))
            out.append(click.style(_wrap(msg, indent="  "), fg="bright_blue"))
        return out

    # agent
    reasoning = (step.get("reasoning_content") or "").strip()
    if reasoning:
        out.append(click.style(f"\n{tag}thinking:", fg="bright_black", bold=True))
        out.append(click.style(_wrap(reasoning, indent="  "), fg="bright_black"))

    message = (step.get("message") or "").strip()
    if message:
        out.append(
            click.style(f"\n{tag}» ", fg="cyan", bold=True) + _wrap(message, indent="  ")
        )

    for tc in step.get("tool_calls") or []:
        name = tc.get("function_name", "?")
        color = TOOL_COLORS_CLICK.get(name, "yellow")
        summary = summarize_tool_call(tc)
        out.append(click.style(f"\n{tag}⚙ {name}", fg=color, bold=True))
        if summary:
            out.append(click.style(_wrap(summary, indent="  "), fg=color))

    for result in (step.get("observation") or {}).get("results") or []:
        content = str(result.get("content") or "").strip()
        if content:
            out.append(
                click.style("  ← ", fg="bright_black")
                + click.style(_wrap(content, indent="    "), fg="bright_black")
            )

    return out


# --------------------------------------------------------------------------- #
# Textual (Rich Text) rendering
# --------------------------------------------------------------------------- #


def render_step_textual(step: dict[str, Any]) -> list[Text]:
    source = step.get("source")
    out: list[Text] = []

    if source == "system":
        msg = step.get("message", "")
        t = Text()
        if msg.startswith("session started"):
            t.append("\n▶ ", style="bold bright_green")
            t.append(msg, style="green")
        elif msg.startswith("session ended"):
            t.append("\n■ ", style="bold red")
            t.append(msg + "\n", style="red")
        elif msg.startswith("rate limit"):
            t.append("⚠ " + msg, style="bold magenta")
        elif msg.startswith("[reva]"):
            t.append(msg, style="bold dim")
        else:
            t.append(msg, style="color(244)")
        out.append(t)
        return out

    if source == "user":
        msg = (step.get("message") or "").strip()
        if msg:
            t = Text()
            t.append("\n▸ user\n", style="bold bright_blue")
            t.append(f"  {msg}", style="bright_blue")
            out.append(t)
        return out

    # agent
    reasoning = (step.get("reasoning_content") or "").strip()
    if reasoning:
        t = Text()
        t.append("\n💭 thinking\n", style="bold color(244)")
        t.append(f"   {reasoning}", style="italic color(240)")
        out.append(t)

    message = (step.get("message") or "").strip()
    if message:
        t = Text()
        t.append("\n» ", style="bold bright_cyan")
        t.append(message, style="bright_white")
        out.append(t)

    for tc in step.get("tool_calls") or []:
        name = tc.get("function_name", "?")
        color = TOOL_COLORS_RICH.get(name, "yellow")
        summary = summarize_tool_call(tc)
        t = Text()
        t.append(f"\n⚙ {name}", style=f"bold {color}")
        if summary:
            t.append(f"\n  {summary}", style=f"dim {color}")
        out.append(t)

    for result in (step.get("observation") or {}).get("results") or []:
        content = str(result.get("content") or "").strip()
        if content:
            t = Text()
            t.append("  ← ", style="dim")
            t.append(content[:400], style="color(245)")
            out.append(t)

    return out


# --------------------------------------------------------------------------- #
# HTML rendering (for reva view --web)
# --------------------------------------------------------------------------- #


def render_step_html(step: dict[str, Any]) -> str:
    """Render an ATIF step as an HTML fragment styled for the web viewer."""
    source = step.get("source", "system")
    step_id = step.get("step_id", "?")
    ts = step.get("timestamp", "")

    parts: list[str] = [
        f'<article class="step step-{html.escape(source)}" data-step-id="{step_id}">',
        '  <header class="step-header">',
        f'    <span class="step-id">#{step_id}</span>',
        f'    <span class="step-source">{html.escape(source)}</span>',
        f'    <time>{html.escape(ts)}</time>',
        '  </header>',
    ]

    reasoning = (step.get("reasoning_content") or "").strip()
    if reasoning:
        parts.append(
            f'  <details class="step-reasoning"><summary>thinking</summary>'
            f'<pre>{html.escape(reasoning)}</pre></details>'
        )

    message = (step.get("message") or "").strip()
    if message:
        parts.append(f'  <div class="step-message"><pre>{html.escape(message)}</pre></div>')

    tool_calls = step.get("tool_calls") or []
    if tool_calls:
        parts.append('  <ul class="tool-calls">')
        for tc in tool_calls:
            name = tc.get("function_name", "?")
            tc_id = tc.get("tool_call_id", "")
            args_json = json.dumps(tc.get("arguments") or {}, indent=2, ensure_ascii=False)
            color = TOOL_COLORS_RICH.get(name, "yellow").replace("_", "-")
            parts.append(
                f'    <li class="tool-call tool-{html.escape(name)}">'
                f'<details><summary>'
                f'<span class="tool-name tool-color-{html.escape(color)}">⚙ {html.escape(name)}</span> '
                f'<span class="tool-summary">{html.escape(summarize_tool_call(tc))}</span>'
                f'</summary><pre class="tool-args">{html.escape(args_json)}</pre>'
                f'<span class="tool-id">{html.escape(tc_id)}</span>'
                f'</details></li>'
            )
        parts.append('  </ul>')

    observations = (step.get("observation") or {}).get("results") or []
    if observations:
        parts.append('  <ul class="observations">')
        for obs in observations:
            content = str(obs.get("content") or "")
            call_id = obs.get("source_call_id", "")
            parts.append(
                f'    <li class="observation"><details><summary>'
                f'← result <span class="call-id">{html.escape(call_id)}</span>'
                f'</summary><pre>{html.escape(content[:4000])}</pre>'
                f'</details></li>'
            )
        parts.append('  </ul>')

    metrics = step.get("metrics") or {}
    if metrics:
        pieces = []
        if "prompt_tokens" in metrics:
            pieces.append(f'↑{metrics["prompt_tokens"]}')
        if "completion_tokens" in metrics:
            pieces.append(f'↓{metrics["completion_tokens"]}')
        if "cost_usd" in metrics:
            pieces.append(f'${metrics["cost_usd"]:.4f}')
        if pieces:
            parts.append(f'  <div class="step-metrics">{html.escape(" ".join(pieces))}</div>')

    parts.append('</article>')
    return "\n".join(parts)
