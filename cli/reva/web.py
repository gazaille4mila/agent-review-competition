"""
reva view --web — FastAPI server exposing the ATIF trajectory for each agent
and streaming updates via server-sent events.

Endpoints:

    GET  /                                    → HTML shell (single-page app)
    GET  /api/agents                          → list of agents
    GET  /api/agents/{name}/trajectory        → full ATIF trajectory (JSON)
    GET  /api/agents/{name}/stream            → SSE live step stream

The server translates `agent.log` on demand through the same
`SessionContext` used by `reva log` and `reva view`, so all three views
render the same data.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from reva.atif import load_trajectory
from reva.session import SessionContext
from reva.tmux import list_sessions

WEB_STATIC_DIR = Path(__file__).parent / "web_static"


def _list_agents(cfg) -> list[dict[str, Any]]:
    if not cfg.agents_dir.exists():
        return []
    running = {s.agent_name for s in list_sessions()}
    out: list[dict[str, Any]] = []
    for d in sorted(cfg.agents_dir.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        try:
            cfg_data = json.loads((d / "config.json").read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}
        traj_path = d / ".session" / "trajectory.json"
        step_count = 0
        if traj_path.exists():
            try:
                step_count = len(json.loads(traj_path.read_text(encoding="utf-8")).get("steps") or [])
            except Exception:
                pass
        out.append({
            "name": d.name,
            "backend": cfg_data.get("backend", "?"),
            "running": d.name in running,
            "has_log": (d / "agent.log").exists(),
            "step_count": step_count,
        })
    return out


def _ensure_trajectory(agent_dir: Path) -> dict[str, Any]:
    """
    Ensure <agent_dir>/.session/trajectory.json exists and is fresh. Returns
    the latest trajectory dict. If agent.log is newer than the cached file,
    replay it through the translator.
    """
    log_path = agent_dir / "agent.log"
    traj_path = agent_dir / ".session" / "trajectory.json"

    if traj_path.exists() and log_path.exists():
        if traj_path.stat().st_mtime >= log_path.stat().st_mtime:
            loaded = load_trajectory(agent_dir)
            if loaded is not None:
                return loaded

    if not log_path.exists():
        loaded = load_trajectory(agent_dir)
        return loaded or {"schema_version": "ATIF-v1.4", "steps": []}

    sess = SessionContext.for_agent(agent_dir)
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for _ in sess.consume_lines(lines):
        pass
    for _ in sess.flush_pending():
        pass
    sess.flush()
    return sess.trajectory()


def _build_app(cfg) -> FastAPI:
    app = FastAPI(title="reva view", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_STATIC_DIR / "index.html")

    @app.get("/static/{path:path}")
    def static(path: str) -> FileResponse:
        target = (WEB_STATIC_DIR / path).resolve()
        try:
            target.relative_to(WEB_STATIC_DIR.resolve())
        except ValueError:
            raise HTTPException(404)
        if not target.exists():
            raise HTTPException(404)
        return FileResponse(target)

    @app.get("/api/agents")
    def api_agents() -> JSONResponse:
        return JSONResponse(_list_agents(cfg))

    @app.get("/api/agents/{name}/trajectory")
    def api_trajectory(name: str) -> JSONResponse:
        agent_dir = cfg.agents_dir / name
        if not agent_dir.exists():
            raise HTTPException(404, f"No agent named {name!r}")
        return JSONResponse(_ensure_trajectory(agent_dir))

    @app.get("/api/agents/{name}/stream")
    async def api_stream(name: str) -> StreamingResponse:
        agent_dir = cfg.agents_dir / name
        if not agent_dir.exists():
            raise HTTPException(404, f"No agent named {name!r}")

        async def event_source() -> AsyncIterator[bytes]:
            sess = SessionContext.for_agent(agent_dir)
            log_path = agent_dir / "agent.log"

            # Replay any existing content first so late subscribers see full history.
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                    for step in sess.consume_lines(fh.readlines()):
                        yield _sse("step", step)
                for step in sess.flush_pending():
                    yield _sse("step", step)

            yield _sse("snapshot", sess.trajectory())

            # Then tail for new lines.
            last_size = log_path.stat().st_size if log_path.exists() else 0
            while True:
                await asyncio.sleep(0.5)
                if not log_path.exists():
                    continue
                try:
                    size = log_path.stat().st_size
                except OSError:
                    continue
                if size <= last_size:
                    yield b": keep-alive\n\n"
                    continue
                with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(last_size)
                    new_lines = fh.readlines()
                last_size = size
                emitted_any = False
                for step in sess.consume_lines(new_lines):
                    emitted_any = True
                    yield _sse("step", step)
                # Force out any paragraph-buffered content so live viewers
                # don't wait for a trailing blank line.
                for step in sess.flush_pending():
                    emitted_any = True
                    yield _sse("step", step)
                if emitted_any:
                    sess.flush()
                    yield _sse("final_metrics", sess.trajectory().get("final_metrics", {}))

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_source(), media_type="text/event-stream", headers=headers)

    return app


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def serve(cfg, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = _build_app(cfg)
    print(f"reva view --web serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
