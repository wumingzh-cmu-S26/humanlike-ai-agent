"""Reciprocal Rank Fusion (RRF) for combining ranked lists from different retrievers."""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using RRF.

    score(d) = Σ 1 / (k + rank_i(d))   for each list i where d appears.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _orig_score) in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
