"""Reciprocal Rank Fusion of ranked result lists."""

from __future__ import annotations

from collections.abc import Sequence

from policy_rag.models import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60, limit: int | None = None
) -> list[ScoredChunk]:
    """Fuse several rankings: score(d) = sum over lists of 1 / (k + rank(d)).

    The fused `score` is the RRF score; `vector_score`/`lexical_score` carry the best raw
    score seen for the chunk in each modality so callers can apply relevance thresholds.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            contribution = 1.0 / (k + rank)
            existing = fused.get(item.chunk.chunk_id)
            if existing is None:
                fused[item.chunk.chunk_id] = ScoredChunk(
                    chunk=item.chunk,
                    score=contribution,
                    vector_score=item.vector_score,
                    lexical_score=item.lexical_score,
                )
            else:
                existing.score += contribution
                existing.vector_score = max(existing.vector_score, item.vector_score)
                existing.lexical_score = max(existing.lexical_score, item.lexical_score)
    ordered = sorted(fused.values(), key=lambda s: (-s.score, s.chunk.chunk_id))
    return ordered[:limit] if limit else ordered
