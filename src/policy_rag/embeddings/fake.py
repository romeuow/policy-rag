"""Deterministic, offline embedding provider based on feature hashing of word n-grams.

The Claude API does not expose an embeddings endpoint, so production uses Voyage AI. This fake
exists so that the whole pipeline (ingest, hybrid search, FAQ matching) runs offline with
reasonable recall on lexical overlap.
"""

from __future__ import annotations

import hashlib
import math

from policy_rag.text import tokenize


class FakeEmbeddingProvider:
    dim: int

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        return tokens + bigrams

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
