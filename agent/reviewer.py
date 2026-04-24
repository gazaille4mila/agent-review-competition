"""
Core review logic: paper analysis, comment generation, and verdict scoring.

All public methods are *async* and call the GitHub Models API via httpx,
authenticated through the GitHub CLI (``gh auth token``).

References
----------
- GitHub Copilot CLI (``gh``): https://github.com/github/copilot-cli
- GitHub Models API (OpenAI-compatible): https://docs.github.com/en/github-models/prototyping-with-ai-models
- ``gh auth token`` reference: https://cli.github.com/manual/gh_auth_token
- GitHub Models inference endpoint: https://models.inference.ai.azure.com
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from typing import Any, Optional

import httpx

from .config import Config
from .prompts import (
    SYSTEM_PROMPT,
    build_comment_prompt,
    build_review_prompt,
    build_verdict_prompt,
    build_discussion_reply_prompt,
)

logger = logging.getLogger(__name__)

# Focus areas to pick from — we randomly select 2–3 per paper so each review
# is distinctive and doesn't over-spend tokens.
_ALL_FOCUS_AREAS = [
    "experimental_rigor",
    "reproducibility",
    "novelty_literature",
    "theoretical_soundness",
    "code_method_alignment",
]

# Comment types we cycle through when posting multiple comments on a paper
_COMMENT_SEQUENCE = ["weakness", "question", "reproducibility"]


class PaperReviewer:
    """
    Orchestrates LLM-powered review of a single paper.

    Uses the GitHub Models API (OpenAI-compatible chat completions) for all
    language-model calls, authenticated via the GitHub CLI token.

    References
    ----------
    - GitHub Models API docs: https://docs.github.com/en/github-models/prototyping-with-ai-models
    - Available models: https://github.com/marketplace/models
    - ``gh auth login``: https://cli.github.com/manual/gh_auth_login

    Args:
        config: Loaded :class:`~agent.config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # Obtain a GitHub token from the gh CLI (requires `gh auth login`).
        try:
            self._gh_token = subprocess.check_output(
                ["gh", "auth", "token"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "GitHub CLI (`gh`) is not installed or not authenticated. "
                "Run `gh auth login` before starting the agent."
            ) from exc
        # Single shared HTTP client — reused across all _chat calls.
        self._http_client = httpx.AsyncClient(timeout=120.0)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def analyze_paper(self, paper_text: str) -> dict[str, Any]:
        """
        Run a structured analysis of *paper_text* across 2–3 focus areas.

        Returns:
            Dict keyed by dimension name, each containing
            ``score`` (1-5), ``observations`` (list), ``main_concern`` (str|None).
        """
        focus_areas = _pick_focus_areas(paper_text)
        logger.info("Analyzing paper on dimensions: %s", focus_areas)

        prompt = build_review_prompt(paper_text, focus_areas)
        raw = await self._chat(prompt, max_tokens=1500)

        analysis = _parse_json_response(raw, fallback={})
        if not analysis:
            logger.warning("Analysis JSON parse failed; returning empty dict.")
        return analysis

    # ------------------------------------------------------------------
    # Comment generation
    # ------------------------------------------------------------------

    async def generate_comment(
        self,
        paper_text: str,
        analysis: dict[str, Any],
        comment_type: str,
    ) -> str:
        """
        Generate a single ready-to-post review comment.

        Args:
            paper_text:   Full paper text.
            analysis:     Dict returned by :meth:`analyze_paper`.
            comment_type: One of ``weakness``, ``question``, ``reproducibility``,
                          ``strength``.

        Returns:
            Comment text suitable for direct posting.
        """
        prompt = build_comment_prompt(paper_text, analysis, comment_type)
        comment = await self._chat(prompt, max_tokens=600)
        return comment.strip()

    async def generate_comments_for_paper(
        self,
        paper_text: str,
        analysis: dict[str, Any],
        n: int = 2,
    ) -> list[str]:
        """
        Generate *n* distinct comments covering different aspects.

        Args:
            paper_text: Full paper text.
            analysis:   Dict from :meth:`analyze_paper`.
            n:          Number of comments to generate (default 2, max 3).

        Returns:
            List of comment strings.
        """
        n = min(n, len(_COMMENT_SEQUENCE))
        comments: list[str] = []
        for comment_type in _COMMENT_SEQUENCE[:n]:
            try:
                text = await self.generate_comment(paper_text, analysis, comment_type)
                comments.append(text)
            except Exception:
                logger.exception("Failed to generate %s comment", comment_type)
        return comments

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    async def generate_verdict_score(
        self,
        paper_text: str,
        analysis: dict[str, Any],
        other_comments: list[dict],
        our_posted_comments: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute an overall verdict score plus justification.

        Args:
            paper_text:          Full paper text.
            analysis:            Our dimension analysis.
            other_comments:      Comments from other agents (list of dicts).
            our_posted_comments: Texts of comments we already posted.

        Returns:
            Dict with keys ``score`` (float 0-10), ``reasoning`` (str),
            ``flagged_agent`` (str|None).
        """
        prompt = build_verdict_prompt(
            paper_text,
            our_posted_comments or [],
            other_comments,
            analysis,
        )
        raw = await self._chat(prompt, max_tokens=800)
        verdict = _parse_json_response(raw, fallback={})

        # Clamp score to [0, 10]
        if "score" in verdict:
            verdict["score"] = max(0.0, min(10.0, float(verdict["score"])))
        else:
            # Fallback: average dimension scores scaled to 0-10
            verdict["score"] = _fallback_score(analysis)
            verdict.setdefault("reasoning", "Score derived from dimension analysis.")
            verdict.setdefault("flagged_agent", None)

        return verdict

    # ------------------------------------------------------------------
    # Citation selection
    # ------------------------------------------------------------------

    def select_citations(
        self,
        other_comments: list[dict],
        min_count: int = 5,
    ) -> list[str]:
        """
        Choose which comment IDs to cite in our verdict.

        Strategy: prefer longer, more substantive comments. If there are
        fewer than *min_count* distinct agents we cannot submit a verdict.

        Args:
            other_comments: List of comment dicts with ``id``, ``agent_id``,
                            ``content`` keys.
            min_count:      Minimum distinct agents required.

        Returns:
            List of comment IDs (one per distinct agent, up to 10).

        Raises:
            ValueError: If fewer than *min_count* distinct agents are present.
        """
        by_agent: dict[str, dict] = {}
        for c in other_comments:
            agent = c.get("agent_id", "")
            if not agent:
                continue
            existing = by_agent.get(agent)
            # Keep the longest comment per agent as proxy for substance
            if existing is None or len(c.get("content", "")) > len(
                existing.get("content", "")
            ):
                by_agent[agent] = c

        if len(by_agent) < min_count:
            raise ValueError(
                f"Only {len(by_agent)} distinct agents found; need {min_count}."
            )

        # Sort by comment length descending, take up to 10
        ranked = sorted(
            by_agent.values(),
            key=lambda c: len(c.get("content", "")),
            reverse=True,
        )
        return [c["id"] for c in ranked[:10] if "id" in c]

    # ------------------------------------------------------------------
    # Discussion replies
    # ------------------------------------------------------------------

    async def generate_discussion_reply(
        self,
        paper_text: str,
        thread_context: list[dict],
        our_perspective: str,
    ) -> str:
        """Generate a reply to an existing discussion thread."""
        prompt = build_discussion_reply_prompt(
            paper_text, thread_context, our_perspective
        )
        reply = await self._chat(prompt, max_tokens=400)
        return reply.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _chat(self, user_prompt: str, max_tokens: int = 1000) -> str:
        """
        Send a single user-turn message via the GitHub Models API and return the reply.

        Calls the OpenAI-compatible ``/chat/completions`` endpoint at
        https://models.inference.ai.azure.com using the token obtained from
        ``gh auth token``.
        """
        response = await self._http_client.post(
            "https://models.inference.ai.azure.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self._gh_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._config.gh_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _pick_focus_areas(paper_text: str) -> list[str]:
    """
    Heuristically pick 2–3 focus dimensions based on paper content.

    Falls back to a random subset when heuristics are inconclusive.
    """
    areas = list(_ALL_FOCUS_AREAS)
    selected: list[str] = []

    text_lower = paper_text.lower()

    # Always check experimental rigor for empirical papers
    if any(kw in text_lower for kw in ("experiment", "baseline", "benchmark", "dataset")):
        selected.append("experimental_rigor")

    # Check for theory content
    if any(kw in text_lower for kw in ("theorem", "proof", "lemma", "proposition")):
        selected.append("theoretical_soundness")

    # Check for code links → alignment check
    if any(kw in text_lower for kw in ("github.com", "code is available", "open-source")):
        selected.append("code_method_alignment")

    # Fill remaining slots randomly
    remaining = [a for a in areas if a not in selected]
    random.shuffle(remaining)
    while len(selected) < 2:
        selected.append(remaining.pop(0))

    return selected[:3]


def _parse_json_response(raw: str, fallback: Any = None) -> Any:
    """Extract and parse the first JSON object/array found in *raw*."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the first '{' or '[' and try from there
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

    logger.debug("Could not parse JSON from response: %s", raw[:200])
    return fallback


def _fallback_score(analysis: dict[str, Any]) -> float:
    """Compute a 0-10 score from average dimension scores (1-5 scale)."""
    scores = [
        v.get("score", 3)
        for v in analysis.values()
        if isinstance(v, dict) and "score" in v
    ]
    if not scores:
        return 5.0
    avg = sum(scores) / len(scores)
    # Map [1, 5] → [0, 10]
    return round((avg - 1) * 2.5, 1)
