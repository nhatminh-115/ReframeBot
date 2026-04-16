"""Unit tests for guardrail crisis detection.

Strategy:
- Regex / keyword logic: tested directly via constants (no models needed).
- detect_crisis(): semantic similarity is mocked so the full function
  is exercisable without loading the SentenceTransformer.
"""
from unittest.mock import patch

import pytest

from reframebot.constants import BENIGN_METAPHOR_RE, REAL_CRISIS_RE


# ---------------------------------------------------------------------------
# Helpers — test regex constants directly (zero model dependency)
# ---------------------------------------------------------------------------

def _matches_crisis(text: str) -> bool:
    return any(r.search(text) for r in REAL_CRISIS_RE)


def _matches_benign(text: str) -> bool:
    return any(r.search(text) for r in BENIGN_METAPHOR_RE)


# ---------------------------------------------------------------------------
# Real crisis patterns
# ---------------------------------------------------------------------------

class TestRealCrisisPatterns:
    @pytest.mark.parametrize("text", [
        "I want to kill myself",
        "I'm thinking about suicide",
        "I want to end my life",
        "I want to take my life",
        "I want to hurt myself",
        "I want to self-harm",
        "I want to overdose",
        "I want to die tonight",
        "I don't want to live anymore",
        "I wish I was dead",
        "There's no reason to live",
        "Tôi muốn tự tử",
        "Tôi muốn chết",
    ])
    def test_detects_genuine_crisis(self, text: str):
        assert _matches_crisis(text), f"Expected crisis match for: {text!r}"

    @pytest.mark.parametrize("text", [
        "I'm dying of embarrassment",
        "That killed me, so funny",
        "Kill it on stage tomorrow",
        "I'm stressed about my exam",
        "I feel hopeless about my grades",
        "I'm exhausted and burned out",
    ])
    def test_does_not_match_non_crisis(self, text: str):
        assert not _matches_crisis(text), f"Unexpected crisis match for: {text!r}"


# ---------------------------------------------------------------------------
# Benign metaphor patterns (override)
# ---------------------------------------------------------------------------

class TestBenignMetaphorPatterns:
    @pytest.mark.parametrize("text", [
        "I'm dying of embarrassment after that presentation",
        "dying of laughter",
        "that killed me",
        "we're going to kill it tomorrow",
    ])
    def test_detects_benign_metaphor(self, text: str):
        assert _matches_benign(text), f"Expected benign match for: {text!r}"

    @pytest.mark.parametrize("text", [
        "I want to kill myself",
        "I want to end my life",
    ])
    def test_real_crisis_not_flagged_as_benign(self, text: str):
        assert not _matches_benign(text), f"Unexpected benign match for: {text!r}"


# ---------------------------------------------------------------------------
# detect_crisis() — semantic similarity mocked, logic tested end-to-end
# ---------------------------------------------------------------------------

SIM_THRESHOLD = 0.62
SIM_MARGIN = 0.08


def _detect(text: str, crisis_sim: float = 0.0, academic_sim: float = 0.0) -> dict:
    """Call detect_crisis() with mocked cosine similarity values."""
    from reframebot.services import guardrail

    with patch.object(guardrail, "_max_cosine_similarity", side_effect=[crisis_sim, academic_sim]):
        return guardrail.detect_crisis(text, SIM_THRESHOLD, SIM_MARGIN)


class TestDetectCrisis:
    def test_keyword_crisis_triggers_regardless_of_semantic(self):
        result = _detect("I want to kill myself", crisis_sim=0.0, academic_sim=0.0)
        assert result["is_crisis"] is True
        assert result["keyword"] is True

    def test_benign_metaphor_suppresses_keyword_match(self):
        result = _detect("I'm dying of embarrassment", crisis_sim=0.0, academic_sim=0.0)
        assert result["is_crisis"] is False
        assert result["benign_metaphor"] is True

    def test_semantic_crisis_triggers_when_above_threshold(self):
        result = _detect("I feel like there's no point", crisis_sim=0.70, academic_sim=0.55)
        assert result["is_crisis"] is True
        assert result["semantic"] is True

    def test_semantic_crisis_suppressed_when_academic_too_close(self):
        # High crisis_sim but margin is too small
        result = _detect("I feel like giving up on my studies", crisis_sim=0.65, academic_sim=0.60)
        assert result["semantic"] is False

    def test_semantic_crisis_suppressed_when_below_threshold(self):
        result = _detect("I feel a little sad", crisis_sim=0.45, academic_sim=0.20)
        assert result["is_crisis"] is False

    def test_both_signals_false_for_plain_text(self):
        result = _detect("I need help with my essay deadline", crisis_sim=0.20, academic_sim=0.75)
        assert result["is_crisis"] is False
        assert result["keyword"] is False
        assert result["semantic"] is False

    def test_result_contains_all_expected_keys(self):
        result = _detect("test", crisis_sim=0.3, academic_sim=0.1)
        assert set(result.keys()) == {
            "is_crisis", "keyword", "semantic",
            "crisis_sim", "academic_sim", "benign_metaphor",
        }
