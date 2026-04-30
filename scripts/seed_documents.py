"""Seed the RAG document store from data/sample_docs/."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from app.rag import get_retriever
from app.rag.chunker import TextChunker

SAMPLE_DIR = Path("./data/sample_docs")


async def main() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    retriever = get_retriever()

    docs = []
    for path in SAMPLE_DIR.glob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, chunk in enumerate(chunker.chunk(text)):
            doc_id = hashlib.sha1(f"{path.name}:{i}".encode()).hexdigest()
            docs.append(
                {
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {"source": path.name, "chunk": i},
                }
            )
    if not docs:
        print("No documents found.")
        return
    added, skipped = await retriever.ingest(docs)
    print(f"Seeded {added} chunks (skipped {skipped}) from {SAMPLE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
