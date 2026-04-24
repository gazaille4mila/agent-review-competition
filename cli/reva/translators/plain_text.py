"""
Generic plain-text translator — fallback for backends that don't have a
dedicated translator yet (aider, codex, opencode). Shares the same turn-
batching logic as gemini_cli: consecutive agent paragraphs are merged into
a single step so that the trajectory viewer shows logical turns rather than
one step per blank-line-separated paragraph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from reva.atif import TrajectoryBuilder

_STATE_KEY = "plain_text"


def translate(
    agent_dir: Path,
    lines: Iterable[str],
    builder: TrajectoryBuilder,
) -> Iterator[dict[str, Any]]:
    state = builder.state.setdefault(_STATE_KEY, {"buf": [], "seeded_user": False})
    buf: list[str] = state["buf"]

    if not state["seeded_user"]:
        state["seeded_user"] = True
        initial_path = agent_dir / "initial_prompt.txt"
        if initial_path.exists():
            initial = initial_path.read_text(encoding="utf-8").strip()
            if initial:
                yield builder.add_user_message(message=initial)

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        # Blank lines: preserve as paragraph separators inside the current
        # agent step instead of flushing a new step.
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
    """Force the pending paragraph buffer out as an agent step."""
    state = builder.state.get(_STATE_KEY)
    if not state:
        return
    buf = state.get("buf") or []
    if not buf:
        return
    step = _flush_agent(builder, buf)
    if step is not None:
        yield step
