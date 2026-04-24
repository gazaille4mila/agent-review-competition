"""
ATIF v1.4 — Agent Trajectory Interchange Format.

Reference:
  - https://www.harborframework.com/docs/agents/trajectory-format
  - https://github.com/harbor-framework/harbor/blob/main/docs/rfcs/0001-trajectory-format.md

We store the trajectory as plain JSON-serializable dicts to avoid a pydantic
dependency. Schema constants and a small TrajectoryBuilder helper live here;
per-backend translators in `reva.translators` populate it.

Persistent layout (one trajectory per agent, rewritten incrementally):

    <agent_dir>/.session/trajectory.json

Each step carries:
  step_id (int, 1-indexed)
  timestamp (ISO 8601 UTC)
  source ("user" | "agent" | "system")
  message (str, optional)
  model_name (str, optional — agent steps only)
  reasoning_content (str, optional — agent "thinking" blocks)
  tool_calls (list[ToolCall], optional)
  observation ({"results": [{source_call_id, content}, ...]}, optional)
  metrics (dict, optional)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ATIF-v1.4"
TRAJECTORY_FILENAME = "trajectory.json"
SESSION_DIR_NAME = ".session"


def now_iso() -> str:
    """UTC timestamp, ISO 8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def trajectory_path(agent_dir: Path) -> Path:
    return agent_dir / SESSION_DIR_NAME / TRAJECTORY_FILENAME


def new_trajectory(
    *,
    session_id: str,
    agent_name: str,
    model_name: str = "",
    agent_version: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an empty ATIF trajectory document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
            **({"extra": extra} if extra else {}),
        },
        "steps": [],
        "final_metrics": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "total_steps": 0,
        },
    }


def make_step(
    *,
    step_id: int,
    source: str,
    message: str | None = None,
    model_name: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a single ATIF step dict, omitting unset optional fields."""
    assert source in ("user", "agent", "system"), f"invalid source: {source!r}"
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": timestamp or now_iso(),
        "source": source,
    }
    if message is not None:
        step["message"] = message
    if model_name:
        step["model_name"] = model_name
    if reasoning_content:
        step["reasoning_content"] = reasoning_content
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation and observation.get("results"):
        step["observation"] = observation
    if metrics:
        step["metrics"] = metrics
    return step


def make_tool_call(
    *,
    tool_call_id: str,
    function_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call_id,
        "function_name": function_name,
        "arguments": arguments or {},
    }


def make_observation(results: list[dict[str, str]]) -> dict[str, Any]:
    return {"results": results}


class TrajectoryBuilder:
    """
    Accumulate ATIF steps while translating a backend log.

    Translators instantiate this, call `add_*` methods per parsed event, and
    the caller eventually writes it to disk via `write(agent_dir)`.
    """

    def __init__(
        self,
        *,
        session_id: str,
        agent_name: str,
        model_name: str = "",
        agent_version: str = "",
    ):
        self.trajectory = new_trajectory(
            session_id=session_id,
            agent_name=agent_name,
            model_name=model_name,
            agent_version=agent_version,
        )
        self._next_id = 1
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}  # id -> tool_call
        # Per-translator scratch state (e.g. buffered paragraph for plain-text
        # backends). Translators own their own keys here.
        self.state: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # step construction
    # ------------------------------------------------------------------ #

    def _allocate_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def add_step(self, **kwargs) -> dict[str, Any]:
        step = make_step(step_id=self._allocate_id(), **kwargs)
        self.trajectory["steps"].append(step)
        self.trajectory["final_metrics"]["total_steps"] = len(self.trajectory["steps"])
        self._apply_metrics(step.get("metrics"))
        return step

    def add_user_message(self, message: str, **extra) -> dict[str, Any]:
        return self.add_step(source="user", message=message, **extra)

    def add_agent_message(
        self,
        *,
        message: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.add_step(
            source="agent",
            message=message,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            model_name=model_name or self.trajectory["agent"].get("model_name") or None,
            metrics=metrics,
        )

    def add_system_message(self, message: str, **extra) -> dict[str, Any]:
        return self.add_step(source="system", message=message, **extra)

    def attach_observation(self, tool_call_id: str, content: str) -> None:
        """
        Attach a tool result to the most-recent agent step whose tool_calls
        include this id. Creates an observation entry if needed.
        """
        for step in reversed(self.trajectory["steps"]):
            if step.get("source") != "agent":
                continue
            ids = {tc["tool_call_id"] for tc in step.get("tool_calls", [])}
            if tool_call_id in ids:
                obs = step.setdefault("observation", {"results": []})
                obs["results"].append({"source_call_id": tool_call_id, "content": content})
                return
        # orphan observation — record as a system step so it isn't lost
        self.add_system_message(
            message=f"orphan tool_result for {tool_call_id}",
            observation=make_observation([{"source_call_id": tool_call_id, "content": content}]),
        )

    def set_agent_metadata(
        self,
        *,
        model_name: str | None = None,
        version: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if model_name:
            self.trajectory["agent"]["model_name"] = model_name
        if version:
            self.trajectory["agent"]["version"] = version
        if session_id:
            self.trajectory["session_id"] = session_id

    def _apply_metrics(self, metrics: dict[str, Any] | None) -> None:
        if not metrics:
            return
        fm = self.trajectory["final_metrics"]
        if "prompt_tokens" in metrics:
            fm["total_prompt_tokens"] += int(metrics.get("prompt_tokens") or 0)
        if "completion_tokens" in metrics:
            fm["total_completion_tokens"] += int(metrics.get("completion_tokens") or 0)
        if "cached_tokens" in metrics:
            fm["total_cached_tokens"] += int(metrics.get("cached_tokens") or 0)
        if "cost_usd" in metrics:
            fm["total_cost_usd"] = round(
                fm["total_cost_usd"] + float(metrics.get("cost_usd") or 0.0), 6
            )

    def set_final_metrics(self, **kwargs) -> None:
        """Overwrite specific final_metrics fields (used by 'result' events)."""
        for k, v in kwargs.items():
            if v is not None:
                self.trajectory["final_metrics"][k] = v

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def write(self, agent_dir: Path) -> Path:
        """Serialize to <agent_dir>/.session/trajectory.json atomically."""
        path = trajectory_path(agent_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.trajectory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path


def load_trajectory(agent_dir: Path) -> dict[str, Any] | None:
    """Read an existing trajectory from disk, or None if missing/invalid."""
    path = trajectory_path(agent_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
