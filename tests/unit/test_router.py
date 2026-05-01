"""Tests for task routing logic.

resolve_task() is pure Python — no ML dependencies.
"""
import pytest

from reframebot.router import resolve_task

_THRESHOLD = 0.90  # default crisis_confidence_threshold


def _history(*user_messages: str) -> list[dict]:
    return [{"role": "user", "content": m} for m in user_messages]


def _conv(*pairs) -> list[dict]:
    """Build history from (role, content) pairs."""
    return [{"role": role, "content": content} for role, content in pairs]


# ---------------------------------------------------------------------------
# Academic keyword override → always TASK_1
# ---------------------------------------------------------------------------

def test_academic_keyword_in_current_turn():
    history = _history("I'm scared of failing my exam")
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.95,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


def test_academic_keyword_in_earlier_turn():
    history = _history("I have a thesis due", "I don't know what to do")
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.95,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


def test_academic_keyword_gpa():
    history = _history("My GPA fell below 3.0 and I feel terrible")
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.80,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


# ---------------------------------------------------------------------------
# No academic keyword — trust the guardrail
# ---------------------------------------------------------------------------

def test_no_keyword_guardrail_task1():
    history = _history("I feel overwhelmed and lost")
    assert resolve_task(
        guardrail_label="TASK_1",
        guardrail_score=0.88,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


def test_no_keyword_guardrail_task3():
    history = _history("What is the capital of France?")
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.97,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_3"


# ---------------------------------------------------------------------------
# Guardrail says TASK_2 → always TASK_1 (crisis handled upstream by detect_crisis)
# ---------------------------------------------------------------------------

def test_guardrail_task2_high_score_trusted():
    # High-confidence TASK_2 — guardrail may catch indirect crisis language
    # that regex/semantic detector misses; trust it.
    history = _history("I don't see any reason in living right now")
    assert resolve_task(
        guardrail_label="TASK_2",
        guardrail_score=0.9994,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_2"


def test_guardrail_task2_low_score_becomes_task1():
    # Low-confidence TASK_2 — ambiguous signal, respond with empathy not crisis escalation
    history = _history("I can't handle this pressure anymore")
    assert resolve_task(
        guardrail_label="TASK_2",
        guardrail_score=0.55,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


# ---------------------------------------------------------------------------
# Follow-up phrases in academic context → TASK_1
# ---------------------------------------------------------------------------

def test_followup_phrase_in_academic_context():
    history = _conv(
        ("user", "I'm stressed about my exams"),
        ("assistant", "What would make you feel more prepared?"),
        ("user", "I don't know"),  # follow-up phrase, short
    )
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.90,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


def test_followup_after_assistant_question():
    # No explicit academic keyword in recent user turn, but assistant asked a ?
    history = _conv(
        ("user", "I have a presentation tomorrow"),
        ("assistant", "How are you feeling about it?"),
        ("user", "not sure"),
    )
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.90,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"


def test_followup_without_academic_context_keeps_guardrail_label():
    # Short follow-up but NO academic context in any recent turn
    history = _conv(
        ("user", "I feel sad"),
        ("assistant", "Tell me more."),
        ("user", "idk"),
    )
    assert resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.92,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_3"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_history_does_not_crash():
    # resolve_task should handle empty history gracefully
    result = resolve_task(
        guardrail_label="TASK_3",
        guardrail_score=0.95,
        crisis_confidence_threshold=_THRESHOLD,
        history=[],
    )
    assert result == "TASK_3"


def test_academic_keyword_overrides_task2():
    history = _history("I want to drop my course because I'm failing")
    assert resolve_task(
        guardrail_label="TASK_2",
        guardrail_score=0.93,
        crisis_confidence_threshold=_THRESHOLD,
        history=history,
    ) == "TASK_1"
