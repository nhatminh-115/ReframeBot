"""RAG (Retrieval-Augmented Generation) service.

Responsibilities:
- Load the ChromaDB persistent collection at startup.
- Retrieve the top-k most relevant knowledge chunks for a query.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from reframebot.config import Settings

logger = logging.getLogger(__name__)

_collection: Optional[chromadb.Collection] = None
_embedder: Optional[SentenceTransformer] = None


def load(settings: Settings, embedder: SentenceTransformer) -> None:
    global _collection, _embedder

    _embedder = embedder
    rag_path = Path(settings.rag_db_path)

    if not rag_path.exists():
        logger.warning(
            "RAG database not found at '%s'. "
            "Run scripts/build_rag_db.py to create it. RAG will be disabled.",
            rag_path,
        )
        return

    client = chromadb.PersistentClient(path=str(rag_path))
    _collection = client.get_collection(name="cbt_knowledge")
    logger.info("RAG database loaded from: %s", rag_path)


def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """Return concatenated knowledge chunks relevant to *query*, or '' if unavailable."""
    if _collection is None or _embedder is None:
        return ""

    try:
        query_embedding = _embedder.encode([query]).tolist()
        results = _collection.query(query_embeddings=query_embedding, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if docs:
            return "\n\n".join(docs)
    except Exception:
        logger.exception("RAG retrieval failed for query: %s", query[:80])

    return ""
