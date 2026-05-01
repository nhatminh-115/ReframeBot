"""Tests for guardrail service helpers.

ML models are mocked — no GPU or model files needed.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from reframebot.services.guardrail import build_guardrail_input, detect_crisis

_D = 384  # all-MiniLM-L6-v2 embedding dimension


def _zero_proto() -> np.ndarray:
    return np.zeros((1, _D), dtype=np.float32)


def _unit_proto(axis: int = 0) -> np.ndarray:
    v = np.zeros((1, _D), dtype=np.float32)
    v[0, axis] = 1.0
    return v


def _mock_embedder(return_vec: np.ndarray) -> MagicMock:
    """Embedder that always returns the given (1, D) array."""
    m = MagicMock()
    m.encode.return_value = return_vec
    return m


# ---------------------------------------------------------------------------
# build_guardrail_input
# ---------------------------------------------------------------------------

def test_build_guardrail_input_empty_history():
    assert build_guardrail_input([], context_turns=3, max_chars=700) == ""


def test_build_guardrail_input_single_user_turn():
    history = [{"role": "user", "content": "I failed my exam"}]
    result = build_guardrail_input(history, context_turns=3, max_chars=700)
    assert result == "I failed my exam"


def test_build_guardrail_input_ignores_assistant_turns():
    history = [
        {"role": "user", "content": "I'm stressed"},
        {"role": "assistant", "content": "I understand."},
        {"role": "user", "content": "I can't sleep"},
    ]
    result = build_guardrail_input(history, context_turns=3, max_chars=700)
    assert "I understand." not in result
    assert "I'm stressed" in result
    assert "I can't sleep" in result


def test_build_guardrail_input_respects_context_turns():
    history = [
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
        {"role": "user", "content": "turn three"},
        {"role": "user", "content": "turn four"},
    ]
    result = build_guardrail_input(history, context_turns=2, max_chars=700)
    assert "turn four" in result
    assert "turn three" in result
    assert "turn one" not in result
    assert "turn two" not in result


def test_build_guardrail_input_truncates_to_max_chars():
    long_message = "x" * 1000
    history = [{"role": "user", "content": long_message}]
    result = build_guardrail_input(history, context_turns=3, max_chars=200)
    assert len(result) <= 200


# ---------------------------------------------------------------------------
# detect_crisis — mocked embedder
# ---------------------------------------------------------------------------

def test_detect_crisis_keyword_match():
    mock_emb = _mock_embedder(_zero_proto())
    with (
        patch("reframebot.services.guardrail._embedder", mock_emb),
        patch("reframebot.services.guardrail._crisis_proto_emb", _zero_proto()),
        patch("reframebot.services.guardrail._academic_proto_emb", _zero_proto()),
    ):
        result = detect_crisis("I want to kill myself", sim_threshold=0.8, sim_margin=0.1)

    assert result["is_crisis"] is True
    assert result["keyword"] is True
    assert result["benign_metaphor"] is False


def test_detect_crisis_benign_metaphor_suppresses_keyword():
    mock_emb = _mock_embedder(_zero_proto())
    with (
        patch("reframebot.services.guardrail._embedder", mock_emb),
        patch("reframebot.services.guardrail._crisis_proto_emb", _zero_proto()),
        patch("reframebot.services.guardrail._academic_proto_emb", _zero_proto()),
    ):
        # "I want to die of embarrassment" — crisis regex fires but benign suppresses it
        result = detect_crisis(
            "I want to die of embarrassment in front of everyone",
            sim_threshold=0.8,
            sim_margin=0.1,
        )

    assert result["is_crisis"] is False
    assert result["benign_metaphor"] is True


def test_detect_crisis_semantic_signal():
    # query embedding aligns with crisis prototype → high cosine sim
    crisis_proto = _unit_proto(axis=0)     # [1, 0, 0, ...]
    academic_proto = _unit_proto(axis=1)   # [0, 1, 0, ...]
    query_vec = _unit_proto(axis=0)        # same direction as crisis → sim = 1.0

    mock_emb = _mock_embedder(query_vec)
    with (
        patch("reframebot.services.guardrail._embedder", mock_emb),
        patch("reframebot.services.guardrail._crisis_proto_emb", crisis_proto),
        patch("reframebot.services.guardrail._academic_proto_emb", academic_proto),
    ):
        result = detect_crisis(
            "some ambiguous text",
            sim_threshold=0.5,   # crisis_sim=1.0 >= 0.5 ✓
            sim_margin=0.1,      # 1.0 - 0.0 = 1.0 >= 0.1 ✓
        )

    assert result["is_crisis"] is True
    assert result["semantic"] is True
    assert result["crisis_sim"] == pytest.approx(1.0, abs=1e-5)
    assert result["academic_sim"] == pytest.approx(0.0, abs=1e-5)


def test_detect_crisis_no_signal():
    mock_emb = _mock_embedder(_zero_proto())
    with (
        patch("reframebot.services.guardrail._embedder", mock_emb),
        patch("reframebot.services.guardrail._crisis_proto_emb", _zero_proto()),
        patch("reframebot.services.guardrail._academic_proto_emb", _zero_proto()),
    ):
        result = detect_crisis(
            "I'm worried about my exam grade",
            sim_threshold=0.62,
            sim_margin=0.08,
        )

    assert result["is_crisis"] is False
    assert result["keyword"] is False
    assert result["semantic"] is False
