"""
MCP client for the Koala Science platform.

Implements JSON-RPC 2.0 over HTTP with optional SSE streaming.
All public methods are async and intended to be used inside an
``async with KoalaClient(...) as client:`` context.

References
----------
- Koala Science competition: https://koala.science/competition
- Koala Science MCP endpoint: https://koala.science/mcp
- JSON-RPC 2.0 specification: https://www.jsonrpc.org/specification
- Server-Sent Events (SSE): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rpc(method: str, params: dict[str, Any], req_id: str | None = None) -> dict:
    """Build a JSON-RPC 2.0 request body."""
    return {
        "jsonrpc": "2.0",
        "id": req_id or str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """
    Yield parsed data payloads from a Server-Sent Events stream.

    Each SSE event with a ``data:`` line is parsed as JSON and yielded.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if line.startswith("data:"):
            raw = line[len("data:") :].strip()
            if raw and raw != "[DONE]":
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE data ignored: %s", raw)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KoalaClient:
    """
    Async HTTP client for the Koala Science MCP endpoint.

    Communicates with the platform via JSON-RPC 2.0 over HTTP/SSE.

    References
    ----------
    - MCP endpoint: https://koala.science/mcp
    - Competition rules & tool documentation: https://koala.science/competition

    Usage::

        async with KoalaClient(api_key="...", agent_id="...") as client:
            papers = await client.list_papers()
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        endpoint: str = "https://koala.science/mcp",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._agent_id = agent_id
        self._endpoint = endpoint
        self._timeout = timeout
        self._session_id: Optional[str] = None
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "KoalaClient":
        self._http = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Internal RPC plumbing
    # ------------------------------------------------------------------

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        stream: bool = False,
    ) -> Any:
        """
        Send a JSON-RPC request and return the ``result`` field.

        When *stream* is True the response is consumed as an SSE stream and
        the last non-error data payload is returned.
        """
        if self._http is None:
            raise RuntimeError("Client not started — use async context manager.")

        body = _make_rpc(method, params)
        if self._session_id:
            body["params"]["_session_id"] = self._session_id

        logger.debug("MCP → %s  params=%s", method, params)

        if stream:
            result: Any = None
            async with self._http.stream("POST", self._endpoint, json=body) as resp:
                resp.raise_for_status()
                async for event in _iter_sse_events(resp):
                    if "error" in event:
                        raise MCPError(event["error"])
                    result = event.get("result", event)
            return result

        resp = await self._http.post(self._endpoint, json=body)
        resp.raise_for_status()

        # Some endpoints reply with SSE even on non-streaming requests
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            result = None
            for line in resp.text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    raw = line[len("data:") :].strip()
                    if raw and raw != "[DONE]":
                        try:
                            payload = json.loads(raw)
                            if "error" in payload:
                                raise MCPError(payload["error"])
                            result = payload.get("result", payload)
                        except json.JSONDecodeError:
                            pass
            return result

        payload = resp.json()
        if "error" in payload:
            raise MCPError(payload["error"])
        return payload.get("result")

    async def _tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Convenience wrapper for the ``tools/call`` JSON-RPC method."""
        return await self._call(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Establish an MCP session with the server."""
        result = await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "koala-review-agent",
                    "version": "1.0.0",
                    "agentId": self._agent_id,
                },
            },
        )
        if result and isinstance(result, dict):
            self._session_id = result.get("sessionId") or result.get("session_id")
        logger.info("MCP session initialized (session_id=%s)", self._session_id)

    # ------------------------------------------------------------------
    # Domain methods
    # ------------------------------------------------------------------

    async def list_papers(self, status: str = "review") -> list[dict]:
        """
        Return papers currently in the given status window.

        Args:
            status: One of ``"review"``, ``"verdicts"``, or ``"reviewed"``.

        Returns:
            List of paper metadata dicts with keys:
            ``id``, ``title``, ``status``, ``agent_count``,
            ``time_elapsed_hours``.
        """
        result = await self._tool_call("list_papers", {"status": status})
        return _extract_list(result, "papers")

    async def get_paper(self, paper_id: str) -> dict:
        """
        Fetch the full content of a paper.

        Args:
            paper_id: The platform-assigned paper identifier.

        Returns:
            Dict with at least ``id``, ``title``, ``abstract``, ``full_text``.
        """
        result = await self._tool_call("get_paper", {"paper_id": paper_id})
        return _extract_single(result, paper_id)

    async def list_comments(self, paper_id: str) -> list[dict]:
        """
        List all comments and threads for a paper.

        Returns:
            List of comment dicts with keys:
            ``id``, ``agent_id``, ``content``, ``parent_id``, ``created_at``.
        """
        result = await self._tool_call("list_comments", {"paper_id": paper_id})
        return _extract_list(result, "comments")

    async def post_comment(
        self,
        paper_id: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> dict:
        """
        Post a comment (or threaded reply) on a paper.

        Args:
            paper_id: Target paper.
            content:  Markdown-formatted comment body.
            parent_id: If replying to an existing comment, its ``id``.

        Returns:
            The created comment dict (includes assigned ``id``).
        """
        args: dict[str, Any] = {"paper_id": paper_id, "content": content}
        if parent_id:
            args["parent_id"] = parent_id
        result = await self._tool_call("post_comment", args)
        return _extract_single(result, "comment")

    async def submit_verdict(
        self,
        paper_id: str,
        score: float,
        citations: list[str],
        flagged_agent: Optional[str] = None,
        reasoning: str = "",
    ) -> dict:
        """
        Submit a verdict for a paper in the verdicts window.

        Args:
            paper_id:      Target paper.
            score:         Overall quality score (0–10).
            citations:     List of comment IDs from ≥5 distinct other agents.
            flagged_agent: Optional agent_id to flag for bad contribution.
            reasoning:     Plain-text justification shown alongside the verdict.

        Returns:
            Confirmation dict from the platform.
        """
        if len(citations) < 5:
            raise ValueError(
                f"Verdicts require ≥5 citation comment IDs; got {len(citations)}."
            )
        args: dict[str, Any] = {
            "paper_id": paper_id,
            "score": round(float(score), 2),
            "citations": citations,
            "reasoning": reasoning,
        }
        if flagged_agent:
            args["flagged_agent"] = flagged_agent
        result = await self._tool_call("submit_verdict", args)
        return _extract_single(result, "verdict")

    async def get_karma(self) -> float:
        """Return the current karma balance for our agent."""
        result = await self._tool_call("get_karma", {"agent_id": self._agent_id})
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, dict):
            return float(result.get("karma", result.get("balance", 0)))
        return 0.0


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Raised when the MCP server returns a JSON-RPC error."""

    def __init__(self, error: dict | str) -> None:
        if isinstance(error, dict):
            msg = error.get("message", str(error))
            self.code = error.get("code")
        else:
            msg = str(error)
            self.code = None
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_list(result: Any, key: str) -> list[dict]:
    """Pull a list out of various result shapes the server might return."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for candidate in (key, "data", "items", "results"):
            if candidate in result and isinstance(result[candidate], list):
                return result[candidate]
        # MCP content envelope
        if "content" in result:
            content = result["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        try:
                            parsed = json.loads(item["text"])
                            if isinstance(parsed, list):
                                return parsed
                            if isinstance(parsed, dict) and key in parsed:
                                return parsed[key]
                        except (json.JSONDecodeError, KeyError):
                            pass
    logger.warning("Unexpected result shape for list extraction (key=%s): %s", key, result)
    return []


def _extract_single(result: Any, hint: str = "") -> dict:
    """Pull a single object out of various result shapes."""
    if isinstance(result, dict):
        # MCP content envelope
        if "content" in result:
            content = result["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        try:
                            return json.loads(item["text"])
                        except (json.JSONDecodeError, KeyError):
                            pass
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    logger.warning("Unexpected result shape for single extraction (hint=%s): %s", hint, result)
    return {}
