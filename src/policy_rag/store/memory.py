"""In-memory vector store with cosine similarity and a simple BM25 index."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from policy_rag.models import Chunk, ScoredChunk
from policy_rag.text import tokenize


class InMemoryVectorStore:
    """Good enough for demos and tests; everything lives in Python dicts."""

    def __init__(self, bm25_k1: float = 1.5, bm25_b: float = 0.75) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()
        self._k1 = bm25_k1
        self._b = bm25_b

    # -- persistence -------------------------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self._remove(chunk.chunk_id)
            tokens = tokenize(chunk.text)
            self._chunks[chunk.chunk_id] = chunk
            self._tokens[chunk.chunk_id] = tokens
            self._df.update(set(tokens))
        return len(chunks)

    def _remove(self, chunk_id: str) -> None:
        tokens = self._tokens.pop(chunk_id, [])
        for term in set(tokens):
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._chunks.pop(chunk_id, None)

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        removed = 0
        for chunk_id in list(chunk_ids):
            if chunk_id in self._chunks:
                self._remove(chunk_id)
                removed += 1
        return removed

    def delete_document(self, doc_id: str) -> int:
        return self.delete_chunks(self.chunk_ids(doc_id))

    def chunk_ids(self, doc_id: str) -> set[str]:
        return {cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id}

    def count(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    # -- retrieval -----------------------------------------------------------------------------
    def _candidates(self, doc_ids: set[str] | None) -> list[Chunk]:
        if doc_ids is None:
            return list(self._chunks.values())
        return [c for c in self._chunks.values() if c.doc_id in doc_ids]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def vector_search(
        self, query: list[float], k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=c, score=s, vector_score=s)
            for c in self._candidates(doc_ids)
            if (s := self.cosine(query, c.embedding)) > 0
        ]
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:k]

    def lexical_search(
        self, query: str, k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        candidates = self._candidates(doc_ids)
        if not query_terms or not candidates:
            return []
        n_docs = len(self._chunks)
        avg_len = sum(len(t) for t in self._tokens.values()) / max(n_docs, 1)
        results: list[ScoredChunk] = []
        for chunk in candidates:
            tokens = self._tokens[chunk.chunk_id]
            tf = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * len(tokens) / max(avg_len, 1))
                score += idf * freq * (self._k1 + 1) / denom
            if score > 0:
                results.append(ScoredChunk(chunk=chunk, score=score, lexical_score=score))
        results.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return results[:k]
