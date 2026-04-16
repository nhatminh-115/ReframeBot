"""Build the ChromaDB RAG database from data/knowledge.txt.

Usage:
    uv run python scripts/build_rag_db.py
    uv run python scripts/build_rag_db.py --text data/knowledge.txt --db rag_db --chunk 500 --stride 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB RAG database from a text file")
    parser.add_argument("--text", default=str(_REPO_ROOT / "data" / "knowledge.txt"))
    parser.add_argument("--db", default=str(_REPO_ROOT / "rag_db"))
    parser.add_argument("--chunk", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--stride", type=int, default=400, help="Stride between chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    text_path = Path(args.text)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    print(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model)

    print(f"Reading: {text_path}")
    text = text_path.read_text(encoding="utf-8")
    chunks = [text[i : i + args.chunk] for i in range(0, len(text), args.stride)]
    print(f"Created {len(chunks)} chunks (size={args.chunk}, stride={args.stride})")

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_or_create_collection(name="cbt_knowledge")

    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )

    print(f"RAG database saved to: {args.db}")


if __name__ == "__main__":
    main()
