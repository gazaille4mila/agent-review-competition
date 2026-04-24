"""
claude-code stream-json → ATIF translator.

Claude Code's `--output-format stream-json --verbose` emits one JSON object
per line. We map:

    type=system, subtype=init           → set agent metadata (model, session_id)
    type=assistant, content[]           → one agent step per message:
        thinking block                    → reasoning_content
        text block                        → message
        tool_use block                    → tool_calls[]
    type=user, content[].tool_result    → attach_observation on most recent step
    type=result                         → set final_metrics (cost, turns)
    type=rate_limit_event (non-allowed) → system step
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from reva.atif import TrajectoryBuilder, make_observation, make_tool_call


def flush_pending(builder: TrajectoryBuilder) -> Iterator[dict[str, Any]]:
    """No buffered state for claude-code — one JSON event == one step."""
    return iter(())


def translate(
    agent_dir: Path,
    lines: Iterable[str],
    builder: TrajectoryBuilder,
) -> Iterator[dict[str, Any]]:
    """Yield newly-appended ATIF steps for each raw line consumed."""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            # stray plain text (e.g. [reva] harness log) — emit as system
            step = builder.add_system_message(message=line)
            yield step
            continue

        if not isinstance(d, dict):
            continue

        typ = d.get("type")

        if typ == "system" and d.get("subtype") == "init":
            builder.set_agent_metadata(
                model_name=d.get("model"),
                session_id=d.get("session_id"),
            )
            step = builder.add_system_message(
                message=f"session started model={d.get('model', '?')}"
            )
            yield step

        elif typ == "assistant":
            msg = d.get("message", {}) or {}
            content = msg.get("content", []) or []
            reasoning_parts: list[str] = []
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    thought = (block.get("thinking") or "").strip()
                    if thought:
                        reasoning_parts.append(thought)
                elif btype == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                elif btype == "tool_use":
                    tool_calls.append(
                        make_tool_call(
                            tool_call_id=block.get("id") or f"call_{len(tool_calls)}",
                            function_name=block.get("name") or "?",
                            arguments=block.get("input") or {},
                        )
                    )

            if not (reasoning_parts or text_parts or tool_calls):
                continue

            metrics = _extract_assistant_metrics(msg)
            model_name = msg.get("model") or None

            step = builder.add_agent_message(
                message="\n\n".join(text_parts) if text_parts else None,
                reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
                tool_calls=tool_calls or None,
                model_name=model_name,
                metrics=metrics,
            )
            yield step

        elif typ == "user":
            msg = d.get("message", {}) or {}
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                tool_call_id = block.get("tool_use_id") or block.get("tool_call_id") or ""
                raw_content = block.get("content", "")
                content_str = _stringify_tool_result(raw_content)
                builder.attach_observation(tool_call_id, content_str)
            # no new step — observation was attached in place
            continue

        elif typ == "result":
            builder.set_final_metrics(
                total_cost_usd=d.get("cost_usd"),
                total_steps=d.get("num_turns"),
            )
            cost = d.get("cost_usd")
            turns = d.get("num_turns")
            cost_str = f" cost=${cost:.4f}" if cost else ""
            step = builder.add_system_message(
                message=f"session ended turns={turns}{cost_str}"
            )
            yield step

        elif typ == "rate_limit_event":
            status = (d.get("rate_limit_info") or {}).get("status", "?")
            if status != "allowed":
                step = builder.add_system_message(message=f"rate limit: {status}")
                yield step


def _stringify_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _extract_assistant_metrics(msg: dict[str, Any]) -> dict[str, Any] | None:
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return None
    metrics: dict[str, Any] = {}
    if "input_tokens" in usage:
        metrics["prompt_tokens"] = usage.get("input_tokens") or 0
    if "output_tokens" in usage:
        metrics["completion_tokens"] = usage.get("output_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_create = usage.get("cache_creation_input_tokens") or 0
    if cache_read or cache_create:
        metrics["cached_tokens"] = int(cache_read) + int(cache_create)
    return metrics or None
