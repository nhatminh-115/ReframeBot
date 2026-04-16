"""Task routing logic.

Maps a conversation turn to one of three effective task labels:
  TASK_1 — CBT / academic stress
  TASK_2 — Crisis (handled upstream before this module is called)
  TASK_3 — Out-of-scope

Priority chain (highest to lowest):
  0. Follow-up turn inside an academic context  → TASK_1
  1. Academic keyword present anywhere in recent context → TASK_1
  2. Guardrail says TASK_2 with HIGH confidence but crisis detector is negative
     (false alarm) → TASK_1
  3. Guardrail says TASK_2 with LOW confidence (weak/ambiguous signal)
     → TASK_1  [FIX: was incorrectly TASK_3]
  4. All other cases → keep guardrail label
"""
from __future__ import annotations

import logging
from typing import Dict, List

from reframebot.constants import ACADEMIC_KEYWORD_PATTERNS, FOLLOWUP_PHRASES

logger = logging.getLogger(__name__)


def resolve_task(
    *,
    guardrail_label: str,
    guardrail_score: float,
    crisis_confidence_threshold: float,
    history: List[Dict[str, str]],
) -> str:
    """Return the effective task label for the current turn."""

    last_user_prompt = history[-1]["content"] if history else ""

    # --- Build recent context (last 8 messages) ---
    recent_context = " ".join(m["content"].lower() for m in history[-8:])
    has_academic_keyword = any(p.search(recent_context) for p in ACADEMIC_KEYWORD_PATTERNS)

    # --- Detect short follow-up / clarification turns ---
    last_lower = last_user_prompt.lower().strip()
    is_followup = len(last_lower) <= 80 and any(p in last_lower for p in FOLLOWUP_PHRASES)

    # Check whether the previous assistant turn contained a question
    last_assistant_content = ""
    for msg in reversed(history[:-1]):
        if msg.get("role") in ("assistant", "bot"):
            last_assistant_content = msg.get("content", "").lower()
            break
    assistant_asked_question = "?" in last_assistant_content

    # --- Priority 0: Follow-up in academic context ---
    if has_academic_keyword and (is_followup or assistant_asked_question):
        logger.debug("Route: follow-up in academic context → TASK_1")
        return "TASK_1"

    # --- Priority 1: Academic keyword present ---
    if has_academic_keyword:
        logger.debug("Route: academic keyword detected → TASK_1")
        return "TASK_1"

    # --- Priority 2 & 3: Guardrail flagged TASK_2 but crisis detector said no ---
    if guardrail_label == "TASK_2":
        if guardrail_score >= crisis_confidence_threshold:
            # High-confidence false alarm from guardrail
            logger.debug(
                "Route: guardrail TASK_2 high-score (%.4f) but no crisis signal → TASK_1",
                guardrail_score,
            )
        else:
            # Low-confidence TASK_2 — still ambiguous, stay empathetic
            logger.debug(
                "Route: guardrail TASK_2 low-score (%.4f) → TASK_1 (was TASK_3 before fix)",
                guardrail_score,
            )
        return "TASK_1"

    # --- Priority 4: Trust the guardrail ---
    logger.debug("Route: guardrail label %s (score %.4f)", guardrail_label, guardrail_score)
    return guardrail_label
