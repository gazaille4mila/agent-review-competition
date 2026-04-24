"""
Main agent loop for the Koala Science ICML 2026 Review Competition.

Usage
-----
    python -m agent.main               # run continuously
    python -m agent.main --dry-run     # simulate without posting anything
    python -m agent.main --once        # run one iteration then exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from .config import load_config
from .mcp_client import KoalaClient, MCPError
from .reviewer import PaperReviewer
from .verdict import VerdictManager

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agent.main")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _handle_signal(sig: int, _frame: object) -> None:
    logger.info("Received signal %s — shutting down after current run.", sig)
    _shutdown_event.set()


# ---------------------------------------------------------------------------
# Core iteration
# ---------------------------------------------------------------------------


async def run_once(
    client: KoalaClient,
    reviewer: PaperReviewer,
    verdict_mgr: VerdictManager,
    config,
    dry_run: bool = False,
) -> None:
    """
    Execute one full agent iteration:

    1. Check karma; abort if below threshold.
    2. Fetch papers in review + verdict windows.
    3. For each eligible paper:
       - Post comments (review window).
       - Submit verdict (verdict window).
    """
    # --- Karma guard ---
    try:
        karma = await client.get_karma()
        logger.info("Current karma: %.1f", karma)
        if karma < config.min_karma_threshold:
            logger.warning(
                "Karma %.1f is below threshold %.1f — skipping this run.",
                karma,
                config.min_karma_threshold,
            )
            return
    except MCPError as exc:
        logger.warning("Could not fetch karma (%s); proceeding anyway.", exc)
        karma = float("inf")

    # --- Gather papers ---
    review_papers: list[dict] = []
    verdict_papers: list[dict] = []

    try:
        review_papers = await client.list_papers(status="review")
        logger.info("Papers in review window: %d", len(review_papers))
    except MCPError as exc:
        logger.error("list_papers(review) failed: %s", exc)

    try:
        verdict_papers = await client.list_papers(status="verdicts")
        logger.info("Papers in verdict window: %d", len(verdict_papers))
    except MCPError as exc:
        logger.error("list_papers(verdicts) failed: %s", exc)

    # --- Prioritise review papers ---
    # Sort: fewest agents first → oldest (longest elapsed time)
    review_papers.sort(
        key=lambda p: (p.get("agent_count", 999), -(p.get("time_elapsed_hours", 0)))
    )
    # Prefer papers with < 10 agents (better karma opportunities)
    review_papers = [p for p in review_papers if p.get("agent_count", 0) < 10]

    papers_processed = 0
    for paper_meta in review_papers:
        if papers_processed >= config.max_papers_per_run:
            break
        paper_id = paper_meta.get("id") or paper_meta.get("paper_id", "")
        if not paper_id:
            continue

        elapsed = paper_meta.get("time_elapsed_hours", 0.0)
        if not VerdictManager.in_review_window(elapsed):
            continue
        if verdict_mgr.has_commented(paper_id):
            logger.debug("Already commented on %s; skipping.", paper_id)
            continue

        logger.info(
            "Processing review paper: %s ('%s'), elapsed=%.1fh",
            paper_id,
            paper_meta.get("title", "?"),
            elapsed,
        )

        await _handle_review_paper(
            paper_id=paper_id,
            client=client,
            reviewer=reviewer,
            verdict_mgr=verdict_mgr,
            dry_run=dry_run,
        )
        papers_processed += 1

    # --- Handle verdict papers ---
    for paper_meta in verdict_papers:
        paper_id = paper_meta.get("id") or paper_meta.get("paper_id", "")
        if not paper_id:
            continue

        elapsed = paper_meta.get("time_elapsed_hours", 0.0)
        try:
            other_comments = await client.list_comments(paper_id)
        except MCPError as exc:
            logger.error("list_comments(%s) failed: %s", paper_id, exc)
            continue

        ready, reason = verdict_mgr.can_submit_verdict(
            paper_id=paper_id,
            other_comments=other_comments,
            time_elapsed_hours=elapsed,
        )
        if not ready:
            logger.info("Verdict for %s not ready: %s", paper_id, reason)
            continue

        logger.info(
            "Submitting verdict for paper: %s ('%s')",
            paper_id,
            paper_meta.get("title", "?"),
        )
        await _handle_verdict_paper(
            paper_id=paper_id,
            other_comments=other_comments,
            client=client,
            reviewer=reviewer,
            verdict_mgr=verdict_mgr,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Paper handlers
# ---------------------------------------------------------------------------


async def _handle_review_paper(
    paper_id: str,
    client: KoalaClient,
    reviewer: PaperReviewer,
    verdict_mgr: VerdictManager,
    dry_run: bool,
) -> None:
    """Fetch, analyse, and comment on a paper in the review window."""
    # Fetch paper content
    try:
        paper = await client.get_paper(paper_id)
    except MCPError as exc:
        logger.error("get_paper(%s) failed: %s", paper_id, exc)
        return

    full_text = (
        paper.get("full_text")
        or paper.get("content")
        or paper.get("text")
        or ""
    )
    if not full_text:
        logger.warning("Paper %s has no text; skipping.", paper_id)
        return

    # Use cached analysis if available
    analysis = verdict_mgr.get_cached_analysis(paper_id)
    if not analysis:
        try:
            analysis = await reviewer.analyze_paper(full_text)
            verdict_mgr.cache_analysis(paper_id, analysis)
        except Exception:
            logger.exception("Analysis failed for paper %s", paper_id)
            return

    # Generate 2 comments
    try:
        comments = await reviewer.generate_comments_for_paper(full_text, analysis, n=2)
    except Exception:
        logger.exception("Comment generation failed for paper %s", paper_id)
        return

    for comment_text in comments:
        if not comment_text:
            continue
        if dry_run:
            logger.info("[DRY RUN] Would post comment on %s:\n%s", paper_id, comment_text[:300])
            # Still record locally so verdict logic works in dry-run mode
            verdict_mgr.record_comment(paper_id, f"dry_{paper_id}", comment_text)
        else:
            try:
                result = await client.post_comment(paper_id, comment_text)
                comment_id = result.get("id", "unknown")
                verdict_mgr.record_comment(paper_id, comment_id, comment_text)
                logger.info("Posted comment %s on paper %s", comment_id, paper_id)
            except MCPError as exc:
                logger.error("post_comment(%s) failed: %s", paper_id, exc)
                break  # Stop posting if we hit an error (e.g. rate limit)


async def _handle_verdict_paper(
    paper_id: str,
    other_comments: list[dict],
    client: KoalaClient,
    reviewer: PaperReviewer,
    verdict_mgr: VerdictManager,
    dry_run: bool,
) -> None:
    """Compute and submit a verdict for a paper in the verdict window."""
    try:
        paper = await client.get_paper(paper_id)
    except MCPError as exc:
        logger.error("get_paper(%s) failed during verdict: %s", paper_id, exc)
        return

    full_text = (
        paper.get("full_text")
        or paper.get("content")
        or paper.get("text")
        or ""
    )

    analysis = verdict_mgr.get_cached_analysis(paper_id) or {}
    our_comments = verdict_mgr.get_our_comment_texts(paper_id)

    # Score the paper
    try:
        verdict_data = await reviewer.generate_verdict_score(
            paper_text=full_text,
            analysis=analysis,
            other_comments=other_comments,
            our_posted_comments=our_comments,
        )
    except Exception:
        logger.exception("Verdict generation failed for paper %s", paper_id)
        return

    score: float = verdict_data.get("score", 5.0)
    reasoning: str = verdict_data.get("reasoning", "")
    flagged_agent: Optional[str] = verdict_data.get("flagged_agent")

    # Build citation list
    try:
        citations = reviewer.select_citations(other_comments, min_count=5)
    except ValueError as exc:
        logger.warning("Cannot cite for %s: %s", paper_id, exc)
        return

    if dry_run:
        logger.info(
            "[DRY RUN] Would submit verdict for %s: score=%.1f citations=%s",
            paper_id,
            score,
            citations,
        )
        verdict_mgr.record_verdict(paper_id, score)
        return

    try:
        await client.submit_verdict(
            paper_id=paper_id,
            score=score,
            citations=citations,
            flagged_agent=flagged_agent,
            reasoning=reasoning,
        )
        verdict_mgr.record_verdict(paper_id, score)
        logger.info(
            "Submitted verdict for %s: score=%.1f flagged=%s",
            paper_id,
            score,
            flagged_agent,
        )
    except MCPError as exc:
        logger.error("submit_verdict(%s) failed: %s", paper_id, exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def agent_loop(dry_run: bool = False, run_once_flag: bool = False) -> None:
    """
    Continuously run the agent until interrupted.

    Args:
        dry_run:       If True, simulate all actions without posting.
        run_once_flag: If True, run a single iteration then exit.
    """
    config = load_config()
    verdict_mgr = VerdictManager(state_file=config.state_file)
    reviewer = PaperReviewer(config=config)

    logger.info(
        "Agent starting (model=%s, dry_run=%s, interval=%ds)",
        config.claude_model,
        dry_run,
        config.loop_interval_seconds,
    )

    async with KoalaClient(
        api_key=config.koala_api_key,
        agent_id=config.koala_agent_id,
        endpoint=config.mcp_endpoint,
    ) as client:
        while not _shutdown_event.is_set():
            try:
                await run_once(client, reviewer, verdict_mgr, config, dry_run=dry_run)
            except Exception:
                logger.exception("Unexpected error in agent loop; will retry.")

            if run_once_flag or _shutdown_event.is_set():
                break

            logger.info(
                "Sleeping %d seconds until next run.", config.loop_interval_seconds
            )
            try:
                await asyncio.wait_for(
                    _shutdown_event.wait(),
                    timeout=config.loop_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass  # Normal — timeout means it's time to run again

    logger.info("Agent shut down cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Koala Science ICML 2026 Review Agent"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without posting comments or verdicts.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration then exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Register graceful-shutdown handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    asyncio.run(agent_loop(dry_run=args.dry_run, run_once_flag=args.once))


if __name__ == "__main__":
    main()
