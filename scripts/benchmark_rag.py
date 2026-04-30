"""Benchmark hybrid RAG latency on a synthetic corpus.

Usage:
    python scripts/benchmark_rag.py --num-docs 1000 --num-queries 50
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import string
import time

from app.rag import get_retriever


def _rand_sentence(words: int = 12) -> str:
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
        for _ in range(words)
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=200)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--rerank", action="store_true", default=True)
    args = parser.parse_args()

    retriever = get_retriever()
    print(f"Generating {args.num_docs} synthetic docs...")
    docs = [
        {
            "id": f"bench-{i}",
            "text": " ".join(_rand_sentence() for _ in range(random.randint(5, 20))),
            "metadata": {"source": "benchmark"},
        }
        for i in range(args.num_docs)
    ]
    t0 = time.perf_counter()
    added, skipped = await retriever.ingest(docs)
    print(f"Ingest: {added} added / {skipped} skipped in {(time.perf_counter() - t0):.2f}s")

    print(f"Running {args.num_queries} queries...")
    latencies: list[float] = []
    breakdown: list[dict[str, float]] = []
    for _ in range(args.num_queries):
        q = _rand_sentence(words=random.randint(3, 8))
        t0 = time.perf_counter()
        hits, timings = await retriever.retrieve(q, use_reranker=args.rerank)
        latencies.append((time.perf_counter() - t0) * 1000)
        breakdown.append(timings)
        if not hits:
            continue

    if not latencies:
        return

    sorted_lat = sorted(latencies)

    def pct(p: float) -> float:
        idx = min(int(len(sorted_lat) * p / 100), len(sorted_lat) - 1)
        return sorted_lat[idx]

    print("\nLatency (ms):")
    print(f"  mean   = {statistics.mean(latencies):.1f}")
    print(f"  median = {statistics.median(latencies):.1f}")
    print(f"  p95    = {pct(95):.1f}")
    print(f"  p99    = {pct(99):.1f}")

    avg = {k: statistics.mean(b.get(k, 0.0) for b in breakdown) for k in breakdown[0]}
    print("\nMean per stage (ms):")
    for k, v in avg.items():
        print(f"  {k:14s} = {v:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
