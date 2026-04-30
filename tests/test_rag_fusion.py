"""RRF fusion + BM25 retriever unit tests (no network)."""
from __future__ import annotations


def test_rrf_basic():  # noqa: ANN201
    from app.rag.fusion import reciprocal_rank_fusion

    bm25 = [("A", 0.9), ("B", 0.5), ("C", 0.3)]
    dense = [("B", 0.95), ("C", 0.6), ("A", 0.5)]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    ids = [d for d, _ in fused]
    # B appears at rank 2 in bm25 and rank 1 in dense — should usually beat A & C.
    assert "B" in ids[:2]
    assert ids[0] in {"A", "B"}


def test_rrf_disjoint_lists():  # noqa: ANN201
    from app.rag.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[("X", 1.0)], [("Y", 1.0)]], k=60)
    ids = sorted(d for d, _ in fused)
    assert ids == ["X", "Y"]


def test_bm25_ranking():  # noqa: ANN201
    from app.rag.bm25_retriever import BM25Retriever

    docs = [
        {"id": "1", "text": "cats are cute and friendly"},
        {"id": "2", "text": "dogs love to play fetch"},
        {"id": "3", "text": "the cat sat on the mat"},
    ]
    r = BM25Retriever()
    r.index(docs)
    hits = r.search("cat", top_k=3)
    top_ids = [h[0] for h in hits]
    assert "1" in top_ids[:2] or "3" in top_ids[:2]
