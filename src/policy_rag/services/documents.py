"""Document registry + ingestion orchestration on top of a VectorStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from policy_rag.embeddings.base import EmbeddingProvider
from policy_rag.ingest.chunker import chunk_document
from policy_rag.ingest.parsers import SUPPORTED_SUFFIXES, ParsedDocument, parse_file, parse_text
from policy_rag.models import Document
from policy_rag.store.base import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        chunk_max_chars: int = 900,
        chunk_overlap: int = 120,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._docs: dict[str, Document] = {}
        self._sources: dict[str, str] = {}
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap = chunk_overlap

    # -- queries -------------------------------------------------------------------------------
    def list(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.title)

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def active_doc_ids(self) -> set[str]:
        return {d.doc_id for d in self._docs.values() if d.active}

    # -- mutations -----------------------------------------------------------------------------
    def ingest_parsed(self, parsed: ParsedDocument, raw_text: str | None = None) -> Document:
        """Idempotent ingest: unchanged chunks are re-upserted by hash, stale ones removed."""
        chunks = chunk_document(parsed, self.chunk_max_chars, self.chunk_overlap)
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors, strict=True):
            chunk.embedding = vec
        new_ids = {c.chunk_id for c in chunks}
        stale = self._store.chunk_ids(parsed.doc_id) - new_ids
        removed = self._store.delete_chunks(stale)
        written = self._store.upsert(chunks)

        existing = self._docs.get(parsed.doc_id)
        now = datetime.now(UTC)
        doc = Document(
            doc_id=parsed.doc_id,
            title=parsed.title,
            version=parsed.version,
            source_name=parsed.source_name,
            content_hash=parsed.content_hash,
            active=existing.active if existing else True,
            chunk_count=len(chunks),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._docs[doc.doc_id] = doc
        if raw_text is not None:
            self._sources[doc.doc_id] = raw_text
        logger.info(
            "document_ingested",
            extra={"doc_id": doc.doc_id, "chunks": written, "removed": removed},
        )
        return doc

    def ingest_text(self, text: str, source_name: str, doc_id: str | None = None) -> Document:
        parsed = parse_text(text, source_name=source_name, doc_id=doc_id)
        return self.ingest_parsed(parsed, raw_text=text)

    def ingest_path(self, path: Path) -> Document:
        parsed = parse_file(path)
        raw = path.read_text(encoding="utf-8") if path.suffix.lower() != ".pdf" else None
        return self.ingest_parsed(parsed, raw_text=raw)

    def ingest_directory(self, directory: Path) -> list[Document]:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        return [self.ingest_path(p) for p in files]

    def reindex(self, doc_id: str) -> Document | None:
        raw = self._sources.get(doc_id)
        doc = self._docs.get(doc_id)
        if doc is None or raw is None:
            return None
        return self.ingest_text(raw, source_name=doc.source_name, doc_id=doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        self._store.delete_document(doc_id)
        del self._docs[doc_id]
        self._sources.pop(doc_id, None)
        return True

    def set_active(self, doc_id: str, active: bool) -> Document | None:
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        doc.active = active
        doc.updated_at = datetime.now(UTC)
        return doc
