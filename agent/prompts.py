"""
Prompt templates for the Koala Science review agent.

Each public function returns a ready-to-use string that can be passed
directly to the Anthropic SDK.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a rigorous, constructive AI reviewer for machine-learning papers
submitted to ICML 2026. Your role mirrors that of a senior program-committee
member: you read papers carefully, identify genuine strengths and weaknesses,
and provide actionable feedback that helps authors improve their work.

Guidelines
----------
- Be specific and evidence-based: quote or paraphrase the relevant passages
  and equation numbers when raising concerns.
- Be respectful and professional at all times.
- Prioritise substance over style: focus on scientific merit, not prose.
- Avoid vague statements like "the paper is unclear." Instead, say exactly
  which claim is unclear and why.
- Do not hallucinate references or fabricate experimental results.
- When uncertain, say so explicitly.
- Keep comments concise: aim for 150–350 words per comment.
"""

# ---------------------------------------------------------------------------
# Review analysis
# ---------------------------------------------------------------------------

_DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "experimental_rigor": (
        "Assess the experimental design: Are baselines appropriate and recent? "
        "Are ablations provided? Are results statistically reported with variance "
        "or confidence intervals? Is the number of seeds/runs sufficient?"
    ),
    "reproducibility": (
        "Assess reproducibility: Is the methodology described in enough detail "
        "to re-implement? Is code available? Are hyperparameters, datasets, and "
        "compute requirements specified? Are there sufficient implementation details?"
    ),
    "novelty_literature": (
        "Assess novelty and related-work coverage: Is the contribution clearly "
        "differentiated from prior work? Are the most relevant references cited? "
        "Is the comparison to related methods fair?"
    ),
    "theoretical_soundness": (
        "Assess theoretical content: Are proofs correct and complete? Are "
        "assumptions clearly stated and justified? Is the mathematical notation "
        "consistent and unambiguous?"
    ),
    "code_method_alignment": (
        "If the paper includes a GitHub URL, assess whether the released code "
        "matches the described method. Identify any discrepancies between the "
        "algorithm in the paper and the implementation."
    ),
}


def build_review_prompt(paper_text: str, focus_areas: list[str]) -> str:
    """
    Build a prompt asking Claude to analyse *paper_text* on *focus_areas*.

    Args:
        paper_text:  Full text of the paper.
        focus_areas: 2–3 dimension keys from ``_DIMENSION_DESCRIPTIONS``.

    Returns:
        A user-turn prompt string.
    """
    dimension_block = "\n\n".join(
        f"### {area.replace('_', ' ').title()}\n{_DIMENSION_DESCRIPTIONS.get(area, area)}"
        for area in focus_areas
    )

    return f"""\
Please analyse the following machine-learning paper across these specific
dimensions:

{dimension_block}

For each dimension provide:
1. A score from 1 (very poor) to 5 (excellent).
2. 2–4 bullet points of specific observations with paper evidence (e.g.,
   "Section 4.2 does not report standard deviations for Table 1").
3. A one-sentence summary of the main concern (if any).

Return your analysis as valid JSON matching this schema:
{{
  "<dimension_key>": {{
    "score": <int 1-5>,
    "observations": ["<observation>", ...],
    "main_concern": "<one sentence or null>"
  }},
  ...
}}

--- PAPER START ---
{paper_text[:12000]}
--- PAPER END ---
"""


# ---------------------------------------------------------------------------
# Comment generation
# ---------------------------------------------------------------------------

_COMMENT_TYPE_INSTRUCTIONS: dict[str, str] = {
    "weakness": (
        "Write a focused, evidence-based weakness comment. "
        "Describe one specific methodological or experimental problem, "
        "cite the relevant section/table/figure, and suggest how the authors "
        "could address it."
    ),
    "question": (
        "Write a clarifying question about a specific claim or design choice "
        "that is ambiguous or unexplained. Be concrete and point to the exact "
        "location in the paper."
    ),
    "strength": (
        "Write a comment highlighting a genuine strength of the paper. "
        "Be specific: quote or paraphrase the relevant contribution and explain "
        "why it is valuable to the community."
    ),
    "reproducibility": (
        "Write a comment focused on reproducibility. Identify missing "
        "implementation details, hyperparameters, compute budget, or code "
        "that would be needed to reproduce the main results."
    ),
}


def build_comment_prompt(
    paper_text: str,
    analysis: dict[str, Any],
    comment_type: str,
) -> str:
    """
    Build a prompt to generate a single, ready-to-post review comment.

    Args:
        paper_text:   Full paper text (will be truncated to save tokens).
        analysis:     Output of ``build_review_prompt`` (already parsed).
        comment_type: Key from ``_COMMENT_TYPE_INSTRUCTIONS``.

    Returns:
        A user-turn prompt string.
    """
    instruction = _COMMENT_TYPE_INSTRUCTIONS.get(
        comment_type,
        "Write a constructive and specific review comment about this paper.",
    )

    analysis_summary = _format_analysis_for_prompt(analysis)

    return f"""\
Based on the paper and the review analysis below, {instruction}

Your comment will be posted publicly. It must be:
- 150–350 words
- Written in first-person reviewer voice
- Specific (cite sections, tables, figures, or equations)
- Constructive and professional

Do NOT include phrases like "In conclusion" or score-like ratings.
Output only the comment text — no preamble, no JSON.

--- ANALYSIS SUMMARY ---
{analysis_summary}
--- PAPER EXCERPT ---
{paper_text[:6000]}
---
"""


# ---------------------------------------------------------------------------
# Verdict reasoning
# ---------------------------------------------------------------------------


def build_verdict_prompt(
    paper_text: str,
    our_comments: list[str],
    other_comments: list[dict],
    score_analysis: dict[str, Any],
) -> str:
    """
    Build a prompt to generate a verdict score and justification.

    Args:
        paper_text:      Full paper text.
        our_comments:    The comment texts we already posted.
        other_comments:  List of ``{agent_id, content}`` dicts from other agents.
        score_analysis:  Dimension analysis dict from ``build_review_prompt``.

    Returns:
        A user-turn prompt string.
    """
    our_block = "\n\n".join(f"- {c}" for c in our_comments) or "(none posted yet)"

    other_block = "\n\n".join(
        f"[Agent {c.get('agent_id', '?')}]: {c.get('content', '')[:400]}"
        for c in other_comments[:20]
    ) or "(no other comments)"

    analysis_summary = _format_analysis_for_prompt(score_analysis)

    return f"""\
You must now produce a final verdict for this paper.

Your dimension-level analysis:
{analysis_summary}

Comments you have already posted:
{our_block}

Comments from other agents (up to 20):
{other_block}

Task
----
1. Synthesise all evidence into an overall quality score from 0.0 to 10.0
   (0 = strong reject, 5 = borderline, 10 = strong accept).
2. Write a 200–400-word verdict justification that:
   - References your key concerns and strengths.
   - Acknowledges relevant points raised by other agents.
   - States clearly whether you recommend acceptance.
3. Optionally identify one other agent whose comments were low-quality or
   off-topic (set to null if none).

Return valid JSON:
{{
  "score": <float 0-10>,
  "reasoning": "<200-400 word justification>",
  "flagged_agent": "<agent_id or null>"
}}

--- PAPER EXCERPT ---
{paper_text[:4000]}
---
"""


# ---------------------------------------------------------------------------
# Discussion reply
# ---------------------------------------------------------------------------


def build_discussion_reply_prompt(
    paper_text: str,
    thread_context: list[dict],
    our_perspective: str,
) -> str:
    """
    Build a prompt to generate a reply within an existing discussion thread.

    Args:
        paper_text:      Full paper text.
        thread_context:  Ordered list of ``{agent_id, content}`` dicts in thread.
        our_perspective: Summary of our analysis relevant to this thread.

    Returns:
        A user-turn prompt string.
    """
    thread_block = "\n\n".join(
        f"[Agent {m.get('agent_id', '?')}]: {m.get('content', '')}"
        for m in thread_context
    )

    return f"""\
You are participating in a peer-review discussion thread about an ML paper.

Thread so far:
{thread_block}

Your analysis relevant to this thread:
{our_perspective}

Write a reply (100–250 words) that:
- Directly addresses the most recent point in the thread.
- Adds new information or evidence from the paper where possible.
- Remains professional and constructive.
- Does NOT simply agree or repeat what has been said.

Output only the reply text — no preamble.

--- PAPER EXCERPT ---
{paper_text[:3000]}
---
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_analysis_for_prompt(analysis: dict[str, Any]) -> str:
    """Format a dimension-analysis dict into a readable summary string."""
    if not analysis:
        return "(no analysis available)"
    lines: list[str] = []
    for dim, data in analysis.items():
        if not isinstance(data, dict):
            continue
        score = data.get("score", "?")
        concern = data.get("main_concern") or "none identified"
        obs = data.get("observations", [])
        lines.append(f"**{dim.replace('_', ' ').title()}** (score {score}/5)")
        lines.append(f"  Main concern: {concern}")
        for o in obs[:3]:
            lines.append(f"  • {o}")
    return "\n".join(lines)
