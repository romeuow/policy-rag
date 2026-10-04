"""FAQ matching: answer directly when a stored FAQ question is close enough."""

from __future__ import annotations

from dataclasses import dataclass

from policy_rag.embeddings.base import EmbeddingProvider
from policy_rag.models import Faq
from policy_rag.store.memory import InMemoryVectorStore
from policy_rag.text import tokenize


@dataclass
class FaqMatch:
    faq: Faq
    score: float


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class FaqMatcher:
    """Scores a question against FAQs using max(cosine of embeddings, token Jaccard)."""

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.8) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self._embedder = embedder
        self.threshold = threshold

    def best_match(self, question: str, faqs: list[Faq]) -> FaqMatch | None:
        if not faqs:
            return None
        query_vec = self._embedder.embed_query(question)
        faq_vecs = self._embedder.embed_documents([f.question for f in faqs])
        best: FaqMatch | None = None
        for faq, vec in zip(faqs, faq_vecs, strict=True):
            score = max(InMemoryVectorStore.cosine(query_vec, vec), jaccard(question, faq.question))
            if best is None or score > best.score:
                best = FaqMatch(faq=faq, score=score)
        return best

    def match(self, question: str, faqs: list[Faq]) -> FaqMatch | None:
        best = self.best_match(question, faqs)
        return best if best and best.score >= self.threshold else None
