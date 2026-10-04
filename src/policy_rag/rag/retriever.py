"""Hybrid retrieval: dense + lexical top-k fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from policy_rag.embeddings.base import EmbeddingProvider
from policy_rag.models import ScoredChunk
from policy_rag.store.base import VectorStore
from policy_rag.store.fusion import reciprocal_rank_fusion

LEXICAL_SQUASH = 6.0


@dataclass
class RetrievalResult:
    results: list[ScoredChunk]
    best_vector: float
    best_lexical: float

    @property
    def relevance(self) -> float:
        """Best raw similarity seen across modalities, used for the no-answer threshold.

        BM25 scores are unbounded, so they are squashed into [0, 1) with x / (x + LEXICAL_SQUASH);
        the dense cosine score is used as-is. A single shared term (BM25 ~ 3) maps to ~0.33.
        """
        lexical = self.best_lexical / (self.best_lexical + LEXICAL_SQUASH)
        return max(self.best_vector, lexical)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        top_k_vector: int = 8,
        top_k_lexical: int = 8,
        top_k_final: int = 4,
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self.top_k_vector = top_k_vector
        self.top_k_lexical = top_k_lexical
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

    def retrieve(self, question: str, doc_ids: set[str] | None = None) -> RetrievalResult:
        query_vec = self._embedder.embed_query(question)
        dense = self._store.vector_search(query_vec, self.top_k_vector, doc_ids)
        lexical = self._store.lexical_search(question, self.top_k_lexical, doc_ids)
        fused = reciprocal_rank_fusion([dense, lexical], k=self.rrf_k, limit=self.top_k_final)
        return RetrievalResult(
            results=fused,
            best_vector=max((s.vector_score for s in dense), default=0.0),
            best_lexical=max((s.lexical_score for s in lexical), default=0.0),
        )
