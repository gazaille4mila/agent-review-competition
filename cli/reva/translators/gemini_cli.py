"""
gemini-cli → ATIF translator.

Gemini CLI (invoked non-interactively with `-p`) emits plain text, not JSON.
We translate by:

  1. Emitting the initial prompt (from initial_prompt.txt) as a single user
     step the first time we're called on an empty trajectory.
  2. Buffering all consecutive agent output lines into a single step.
     Blank lines within agent output are preserved as paragraph separators
     inside the same step, rather than creating a new step per paragraph.
  3. Recognizing reva harness lines (``[reva] ...``) and flushing the current
     buffer as one agent step, then emitting the harness line as a system step.
  4. Recognizing noisy / session-restart prefixes and flushing likewise.

This batching reduces thousands of single-sentence steps into a manageable
number of logical turns.

Caveats:
  - Gemini CLI does not expose structured tool-call output, so every model turn
    lands in `message` with no `tool_calls`/`observation`.
  - Token usage / cost are not reported and stay zero in final_metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from reva.atif import TrajectoryBuilder

_STATE_KEY = "gemini_cli"
_NOISY_PREFIXES = (
    "YOLO mode is enabled",
    "Loaded cached credentials",
    "Data collection is disabled",
)


def translate(
    agent_dir: Path,
    lines: Iterable[str],
    builder: TrajectoryBuilder,
) -> Iterator[dict[str, Any]]:
    state = builder.state.setdefault(_STATE_KEY, {"buf": [], "seeded_user": False})
    buf: list[str] = state["buf"]

    # Seed the user step from initial_prompt.txt on first invocation
    if not state["seeded_user"]:
        state["seeded_user"] = True
        initial_path = agent_dir / "initial_prompt.txt"
        if initial_path.exists():
            initial = initial_path.read_text(encoding="utf-8").strip()
            if initial:
                yield builder.add_user_message(message=initial)
        if not builder.trajectory["agent"].get("model_name"):
            builder.set_agent_metadata(model_name="gemini")

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        # Blank lines: keep them in the buffer as paragraph separators
        # instead of flushing a step.  We append an empty string so the
        # final join preserves the visual blank line inside the message.
        if not stripped:
            if buf:
                buf.append("")
            continue

        if line.startswith("[reva]"):
            step = _flush_agent(builder, buf)
            if step is not None:
                yield step
            yield builder.add_system_message(message=stripped)
            continue

        if any(stripped.startswith(p) for p in _NOISY_PREFIXES):
            step = _flush_agent(builder, buf)
            if step is not None:
                yield step
            yield builder.add_system_message(message=stripped)
            continue

        buf.append(line)


def _flush_agent(
    builder: TrajectoryBuilder, buf: list[str]
) -> dict[str, Any] | None:
    """Flush the accumulated agent buffer as a single agent step."""
    text = "\n".join(buf).strip()
    buf.clear()
    if not text:
        return None
    return builder.add_agent_message(message=text)


def flush_pending(builder: TrajectoryBuilder) -> Iterator[dict[str, Any]]:
    """Force the pending paragraph buffer (if any) out as an agent step.

    Called by tailing consumers (SSE stream, reva log) so a log line without
    a trailing blank line still shows up in live views.
    """
    state = builder.state.get(_STATE_KEY)
    if not state:
        return
    buf = state.get("buf") or []
    if not buf:
        return
    step = _flush_agent(builder, buf)
    if step is not None:
        yield step
