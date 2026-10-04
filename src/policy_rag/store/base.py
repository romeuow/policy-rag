"""Vector store protocol."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from policy_rag.models import Chunk, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Persistence + retrieval for chunks. Must support both dense and lexical search."""

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks keyed by `chunk_id`. Returns number of chunks written."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document. Returns number of chunks removed."""
        ...

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int: ...

    def chunk_ids(self, doc_id: str) -> set[str]: ...

    def count(self) -> int: ...

    def vector_search(
        self, query: list[float], k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        """Cosine similarity top-k, optionally restricted to the given documents."""
        ...

    def lexical_search(
        self, query: str, k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        """BM25 / full-text top-k, optionally restricted to the given documents."""
        ...
