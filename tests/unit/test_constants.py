"""Tests for regex patterns in constants.py.

These tests are purely logic-based — no ML models needed.
"""
import pytest

from reframebot.constants import (
    ACADEMIC_KEYWORD_PATTERNS,
    BENIGN_METAPHOR_RE,
    REAL_CRISIS_RE,
)


def _matches_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# REAL_CRISIS_RE — must match genuine crisis language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I want to kill myself",
    "I'm going to end my life",
    "I want to take my life",
    "I hurt myself when I'm stressed",
    "I've been thinking about self-harm",
    "I want to overdose on something",
    "I want to die and I mean it",
    "I don't want to live anymore",
    "I do not want to live like this",
    "There's no reason to live",
    "I wish I were dead",
    "I wish I was dead",
    "I'm suicidal",
    "Tôi muốn tự tử",
    "Tôi muốn chết",
])
def test_real_crisis_re_matches(text: str):
    assert _matches_any(REAL_CRISIS_RE, text), f"Expected crisis match for: {text!r}"


@pytest.mark.parametrize("text", [
    "I'm stressed about my exams",
    "I might fail my course",
    "This homework is killing me",  # metaphorical
    "I'm dying to finish this project",  # idiomatic
    "My GPA is really suffering",
    "I hate this assignment",
])
def test_real_crisis_re_no_false_positives(text: str):
    assert not _matches_any(REAL_CRISIS_RE, text), f"Unexpected crisis match for: {text!r}"


# ---------------------------------------------------------------------------
# BENIGN_METAPHOR_RE — must catch common non-literal uses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I'm dying of embarrassment",
    "I'm dying of laughter",
    "That killed me, it was so funny",
    "You really need to kill it on that presentation",
])
def test_benign_metaphor_re_matches(text: str):
    assert _matches_any(BENIGN_METAPHOR_RE, text), f"Expected benign match for: {text!r}"


def test_benign_overrides_crisis_pattern():
    # "I want to die of embarrassment" triggers a crisis regex but is benign
    text = "I want to die of embarrassment in front of everyone"
    has_real = _matches_any(REAL_CRISIS_RE, text)
    has_benign = _matches_any(BENIGN_METAPHOR_RE, text)
    # keyword_crisis = has_real AND NOT has_benign
    keyword_crisis = has_real and not has_benign
    assert not keyword_crisis, "Benign metaphor should suppress the crisis flag"


# ---------------------------------------------------------------------------
# ACADEMIC_KEYWORD_PATTERNS — must catch common academic stress terms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I have an exam tomorrow",
    "My GPA dropped this semester",
    "The deadline is at midnight",
    "I'm stressed about my thesis",
    "I bombed the midterm",
    "I missed the assignment deadline",
    "My presentation is next week",
    "I'm burning out from studying",
    "I feel like I'm experiencing burnout",
    "I think I have imposter syndrome",
])
def test_academic_keyword_patterns_match(text: str):
    assert _matches_any(ACADEMIC_KEYWORD_PATTERNS, text), f"Expected academic keyword match for: {text!r}"


@pytest.mark.parametrize("text", [
    "I feel very sad today",
    "Nobody understands me",
    "I had a bad dream",
])
def test_academic_keyword_no_false_positives(text: str):
    assert not _matches_any(ACADEMIC_KEYWORD_PATTERNS, text), f"Unexpected academic match for: {text!r}"
