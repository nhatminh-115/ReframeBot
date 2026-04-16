"""Guardrail classifier service.

Responsibilities:
- Load the DistilBERT guardrail pipeline once at startup.
- Provide the shared SentenceTransformer embedder (also used by RAG).
- Build multi-turn input text for the classifier.
- Detect crisis via regex + cosine-similarity (semantic) signals.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from reframebot.config import Settings
from reframebot.constants import (
    ACADEMIC_STRESS_PROTOTYPES,
    BENIGN_METAPHOR_RE,
    CRISIS_PROTOTYPES,
    REAL_CRISIS_RE,
)

logger = logging.getLogger(__name__)

# Module-level singletons — populated by load()
_guardrail_pipeline = None
_embedder: SentenceTransformer | None = None
_crisis_proto_emb: np.ndarray | None = None
_academic_proto_emb: np.ndarray | None = None


def load(settings: Settings) -> None:
    global _guardrail_pipeline, _embedder, _crisis_proto_emb, _academic_proto_emb

    # --- Guardrail classifier ---
    guardrail_path = settings.guardrail_path
    if not Path(guardrail_path).exists():
        raise FileNotFoundError(
            f"Guardrail model not found at '{guardrail_path}'. "
            "Set GUARDRAIL_PATH in .env or train the model first."
        )
    logger.info("Loading guardrail model from: %s", guardrail_path)
    _guardrail_pipeline = pipeline(
        "text-classification",
        model=guardrail_path,
        tokenizer=guardrail_path,
        device=-1,
    )
    logger.info("Guardrail model ready.")

    # --- Shared embedder ---
    logger.info("Loading embedding model: %s", settings.router_embed_model)
    _embedder = SentenceTransformer(settings.router_embed_model)
    logger.info("Embedding model ready.")

    # --- Pre-compute prototype embeddings ---
    _crisis_proto_emb = _embed(CRISIS_PROTOTYPES)
    _academic_proto_emb = _embed(ACADEMIC_STRESS_PROTOTYPES)
    logger.info("Prototype embeddings ready.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed(texts: List[str]) -> np.ndarray:
    assert _embedder is not None, "Call guardrail.load() before embedding."
    return np.asarray(_embedder.encode(texts, normalize_embeddings=True), dtype=np.float32)


def _max_cosine_similarity(query: str, proto_emb: np.ndarray) -> float:
    if proto_emb is None or len(proto_emb) == 0:
        return 0.0
    q = np.asarray(
        _embedder.encode([query], normalize_embeddings=True)[0], dtype=np.float32
    )
    return float(np.max(proto_emb @ q))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedder() -> SentenceTransformer:
    assert _embedder is not None, "Call guardrail.load() first."
    return _embedder


def classify(text: str) -> Dict[str, object]:
    """Run the guardrail classifier and return label + score."""
    assert _guardrail_pipeline is not None, "Call guardrail.load() first."
    result = _guardrail_pipeline(text)[0]
    return {"label": result["label"], "score": result["score"]}


def build_guardrail_input(
    history: List[Dict[str, str]],
    context_turns: int,
    max_chars: int,
) -> str:
    """Concatenate the most recent user turns into a single classifier input."""
    if not history:
        return ""

    user_texts: List[str] = []
    for msg in reversed(history):
        if (msg.get("role") or "").lower() == "user":
            user_texts.append((msg.get("content") or "").strip())
            if len(user_texts) >= max(1, context_turns):
                break

    merged = "\n".join(t for t in reversed(user_texts) if t).strip()
    if len(merged) > max_chars:
        merged = merged[-max_chars:]
    return merged


def detect_crisis(
    user_text: str,
    sim_threshold: float,
    sim_margin: float,
) -> Dict[str, object]:
    """Return crisis detection results via regex + semantic similarity."""
    text = user_text or ""

    has_benign = any(r.search(text) for r in BENIGN_METAPHOR_RE)
    has_real_pattern = any(r.search(text) for r in REAL_CRISIS_RE)
    keyword_crisis = bool(has_real_pattern and not has_benign)

    crisis_sim = _max_cosine_similarity(text, _crisis_proto_emb)
    academic_sim = _max_cosine_similarity(text, _academic_proto_emb)
    semantic_crisis = bool(
        crisis_sim >= sim_threshold
        and (crisis_sim - academic_sim) >= sim_margin
    )

    return {
        "is_crisis": bool(keyword_crisis or semantic_crisis),
        "keyword": keyword_crisis,
        "semantic": semantic_crisis,
        "crisis_sim": float(crisis_sim),
        "academic_sim": float(academic_sim),
        "benign_metaphor": bool(has_benign),
    }
