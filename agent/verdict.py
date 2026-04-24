"""
Verdict management and paper-lifecycle tracking.

This module owns the state that must persist across agent runs:
- Which papers we have commented on (and when).
- Which papers we have submitted verdicts for.
- Time-window helpers for the competition lifecycle.

References
----------
- Competition timeline & verdict rules: https://koala.science/competition
- ``submit_verdict`` MCP tool: https://koala.science/mcp
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Competition time windows (hours since paper was released)
REVIEW_WINDOW_START_H = 0.0
REVIEW_WINDOW_END_H = 48.0
VERDICT_WINDOW_START_H = 48.0
VERDICT_WINDOW_END_H = 72.0


class VerdictManager:
    """
    Tracks per-paper activity and manages verdict submission logic.

    State is persisted to a JSON file so that it survives agent restarts.

    Args:
        state_file: Path to the JSON state file.
    """

    def __init__(self, state_file: str = "agent_state.json") -> None:
        self._state_file = state_file
        self._state: dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        """Load state from disk, returning a blank state on first run."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    logger.info("Loaded state from %s", self._state_file)
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load state file: %s", exc)
        return {
            "commented_papers": {},   # paper_id → {comment_ids: [], posted_at: iso}
            "verdict_papers": {},     # paper_id → {score: float, submitted_at: iso}
            "analyses": {},           # paper_id → analysis dict
        }

    def save(self) -> None:
        """Persist current state to disk."""
        try:
            with open(self._state_file, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save state: %s", exc)

    # ------------------------------------------------------------------
    # Comment tracking
    # ------------------------------------------------------------------

    def record_comment(
        self,
        paper_id: str,
        comment_id: str,
        comment_text: str = "",
    ) -> None:
        """Record that we posted a comment on *paper_id*."""
        entry = self._state["commented_papers"].setdefault(paper_id, {
            "comment_ids": [],
            "comment_texts": [],
            "posted_at": _now_iso(),
        })
        if comment_id not in entry["comment_ids"]:
            entry["comment_ids"].append(comment_id)
            entry["comment_texts"].append(comment_text)
        self.save()

    def has_commented(self, paper_id: str) -> bool:
        """Return True if we have posted at least one comment on *paper_id*."""
        return paper_id in self._state["commented_papers"]

    def get_our_comment_texts(self, paper_id: str) -> list[str]:
        """Return the texts of comments we posted on *paper_id*."""
        entry = self._state["commented_papers"].get(paper_id, {})
        return entry.get("comment_texts", [])

    # ------------------------------------------------------------------
    # Analysis caching
    # ------------------------------------------------------------------

    def cache_analysis(self, paper_id: str, analysis: dict[str, Any]) -> None:
        """Cache Claude's dimension analysis for *paper_id*."""
        self._state["analyses"][paper_id] = analysis
        self.save()

    def get_cached_analysis(self, paper_id: str) -> Optional[dict[str, Any]]:
        """Return cached analysis or None."""
        return self._state["analyses"].get(paper_id)

    # ------------------------------------------------------------------
    # Verdict tracking
    # ------------------------------------------------------------------

    def record_verdict(self, paper_id: str, score: float) -> None:
        """Record that we have submitted a verdict for *paper_id*."""
        self._state["verdict_papers"][paper_id] = {
            "score": score,
            "submitted_at": _now_iso(),
        }
        self.save()

    def has_submitted_verdict(self, paper_id: str) -> bool:
        """Return True if we have already submitted a verdict for *paper_id*."""
        return paper_id in self._state["verdict_papers"]

    # ------------------------------------------------------------------
    # Time-window helpers
    # ------------------------------------------------------------------

    @staticmethod
    def in_review_window(time_elapsed_hours: float) -> bool:
        """Return True if a paper is still in the review (comment) window."""
        return REVIEW_WINDOW_START_H <= time_elapsed_hours < REVIEW_WINDOW_END_H

    @staticmethod
    def in_verdict_window(time_elapsed_hours: float) -> bool:
        """Return True if a paper is in the verdict-submission window."""
        return VERDICT_WINDOW_START_H <= time_elapsed_hours < VERDICT_WINDOW_END_H

    @staticmethod
    def time_remaining_in_review(time_elapsed_hours: float) -> float:
        """Hours remaining in the review window (negative if past)."""
        return REVIEW_WINDOW_END_H - time_elapsed_hours

    @staticmethod
    def time_remaining_in_verdict(time_elapsed_hours: float) -> float:
        """Hours remaining in the verdict window (negative if past)."""
        return VERDICT_WINDOW_END_H - time_elapsed_hours

    # ------------------------------------------------------------------
    # Verdict readiness check
    # ------------------------------------------------------------------

    def can_submit_verdict(
        self,
        paper_id: str,
        other_comments: list[dict],
        time_elapsed_hours: float,
        min_other_agents: int = 5,
    ) -> tuple[bool, str]:
        """
        Decide whether we are ready to submit a verdict for *paper_id*.

        Returns:
            ``(True, "")`` when all conditions are met, or
            ``(False, "<reason>")`` when not.
        """
        if not self.in_verdict_window(time_elapsed_hours):
            return False, (
                f"Not in verdict window "
                f"(elapsed={time_elapsed_hours:.1f}h, "
                f"window={VERDICT_WINDOW_START_H}-{VERDICT_WINDOW_END_H}h)"
            )

        if self.has_submitted_verdict(paper_id):
            return False, "Verdict already submitted."

        if not self.has_commented(paper_id):
            return False, "We have not commented on this paper yet."

        distinct_agents = {
            c.get("agent_id") for c in other_comments if c.get("agent_id")
        }
        if len(distinct_agents) < min_other_agents:
            return False, (
                f"Only {len(distinct_agents)} distinct other agents "
                f"(need {min_other_agents})."
            )

        return True, ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
