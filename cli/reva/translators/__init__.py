"""
Per-backend translators from native agent logs to ATIF trajectories.

Each translator exposes a `translate(agent_dir, lines, builder)` function that
consumes an iterable of raw log lines and populates the given TrajectoryBuilder.
The dispatcher here picks the right translator based on the agent's backend.

Typical usage (live tailing):

    from reva.atif import TrajectoryBuilder
    from reva.translators import get_translator

    builder = TrajectoryBuilder(session_id="…", agent_name="my-agent")
    translate = get_translator(backend_name)

    for line in tail(agent_log):
        for step in translate(agent_dir, [line], builder):
            yield step                 # newly emitted ATIF steps
    builder.write(agent_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from reva.atif import TrajectoryBuilder

TranslateFn = Callable[[Path, Iterable[str], TrajectoryBuilder], Iterator[dict[str, Any]]]
FlushFn = Callable[[TrajectoryBuilder], Iterator[dict[str, Any]]]


def get_translator(backend_name: str) -> TranslateFn:
    """Return the translator function for a given backend."""
    if backend_name == "claude-code":
        from reva.translators.claude_code import translate as _fn
    elif backend_name == "gemini-cli":
        from reva.translators.gemini_cli import translate as _fn
    else:
        from reva.translators.plain_text import translate as _fn
    return _fn


def get_flusher(backend_name: str) -> FlushFn:
    """Return the flush-pending function for a given backend (forces any
    paragraph-buffered content to emit, for live views)."""
    if backend_name == "claude-code":
        from reva.translators.claude_code import flush_pending as _fn
    elif backend_name == "gemini-cli":
        from reva.translators.gemini_cli import flush_pending as _fn
    else:
        from reva.translators.plain_text import flush_pending as _fn
    return _fn


__all__ = ["get_translator", "get_flusher", "TranslateFn", "FlushFn"]
