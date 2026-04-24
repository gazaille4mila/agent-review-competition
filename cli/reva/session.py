"""
Session helpers — tie an agent directory to a TrajectoryBuilder and translator.

Public API:

    ctx = SessionContext.for_agent(agent_dir)          # loads config.json + backend
    for step in ctx.consume_lines(lines):              # translate + accumulate
        ...
    ctx.flush()                                        # persist .session/trajectory.json
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from reva.atif import TrajectoryBuilder, load_trajectory
from reva.translators import FlushFn, TranslateFn, get_flusher, get_translator


@dataclass
class SessionContext:
    agent_dir: Path
    agent_name: str
    backend_name: str
    builder: TrajectoryBuilder
    translate: TranslateFn
    flush_translator: FlushFn

    @classmethod
    def for_agent(cls, agent_dir: Path) -> "SessionContext":
        config_path = agent_dir / "config.json"
        cfg_data: dict[str, Any] = {}
        if config_path.exists():
            try:
                cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg_data = {}

        agent_name = cfg_data.get("name") or agent_dir.name
        backend_name = cfg_data.get("backend") or "claude-code"

        # Reuse an existing trajectory's session_id when present so we append
        # steps to a stable session when the user re-runs reva log.
        existing = load_trajectory(agent_dir)
        session_id = (
            (existing or {}).get("session_id")
            or cfg_data.get("session_id")
            or str(uuid.uuid4())
        )

        builder = TrajectoryBuilder(
            session_id=session_id,
            agent_name=agent_name,
        )
        return cls(
            agent_dir=agent_dir,
            agent_name=agent_name,
            backend_name=backend_name,
            builder=builder,
            translate=get_translator(backend_name),
            flush_translator=get_flusher(backend_name),
        )

    def consume_lines(self, lines: Iterable[str]) -> Iterator[dict[str, Any]]:
        """Feed raw log lines through the translator and yield ATIF steps."""
        yield from self.translate(self.agent_dir, lines, self.builder)

    def flush_pending(self) -> Iterator[dict[str, Any]]:
        """Emit any paragraph-buffered content as steps (live-view use)."""
        yield from self.flush_translator(self.builder)

    def trajectory(self) -> dict[str, Any]:
        return self.builder.trajectory

    def flush(self) -> Path:
        return self.builder.write(self.agent_dir)
