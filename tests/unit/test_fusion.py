import pytest

from policy_rag.models import Chunk, ScoredChunk
from policy_rag.store.fusion import reciprocal_rank_fusion


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", doc_title="D", section="s", version="1", position=0,
        text=cid, content_hash="h",
    )  # fmt: skip


def test_rrf_scores_and_orders_by_sum_of_reciprocal_ranks():
    dense = [ScoredChunk(chunk=_chunk("a"), score=0.9, vector_score=0.9),
             ScoredChunk(chunk=_chunk("b"), score=0.5, vector_score=0.5)]  # fmt: skip
    lexical = [ScoredChunk(chunk=_chunk("b"), score=7.0, lexical_score=7.0),
               ScoredChunk(chunk=_chunk("c"), score=3.0, lexical_score=3.0)]  # fmt: skip
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ids = [s.chunk.chunk_id for s in fused]
    assert ids == ["b", "a", "c"]
    b = fused[0]
    assert b.score == pytest.approx(1 / 62 + 1 / 61)
    assert b.vector_score == 0.5 and b.lexical_score == 7.0


def test_rrf_limit_and_validation():
    ranking = [ScoredChunk(chunk=_chunk(str(i)), score=1.0) for i in range(5)]
    assert len(reciprocal_rank_fusion([ranking], limit=2)) == 2
    assert reciprocal_rank_fusion([]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking], k=0)
