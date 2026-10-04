"""Postgres + pgvector store (production). Not exercised by the offline test-suite.

Schema is managed by Alembic (see `alembic/versions`). All SQL is parameterised.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from policy_rag.models import Chunk, ScoredChunk

_SELECT_COLUMNS = "chunk_id, doc_id, doc_title, section, version, position, text, content_hash"


class PgVectorStore:
    def __init__(self, dsn: str, embedding_dim: int = 1024) -> None:
        self._dsn = dsn
        self._dim = embedding_dim

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        register_vector(conn)
        return conn

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            doc_title=row["doc_title"],
            section=row["section"],
            version=row["version"],
            position=row["position"],
            text=row["text"],
            content_hash=row["content_hash"],
        )

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        sql = """
            INSERT INTO chunks (chunk_id, doc_id, doc_title, section, version, position,
                                text, content_hash, embedding)
            VALUES (%(chunk_id)s, %(doc_id)s, %(doc_title)s, %(section)s, %(version)s,
                    %(position)s, %(text)s, %(content_hash)s, %(embedding)s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_title = EXCLUDED.doc_title,
                section = EXCLUDED.section,
                version = EXCLUDED.version,
                position = EXCLUDED.position,
                text = EXCLUDED.text,
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(sql, [c.model_dump() for c in chunks])
            conn.commit()
        return len(chunks)

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
            conn.commit()
            return cur.rowcount

    def delete_document(self, doc_id: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            conn.commit()
            return cur.rowcount

    def chunk_ids(self, doc_id: str) -> set[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s", (doc_id,))
            return {row["chunk_id"] for row in cur.fetchall()}

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks")
            return int(cur.fetchone()["n"])

    def vector_search(
        self, query: list[float], k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        where = "WHERE doc_id = ANY(%(doc_ids)s)" if doc_ids is not None else ""
        sql = f"""
            SELECT {_SELECT_COLUMNS}, 1 - (embedding <=> %(query)s::vector) AS score
            FROM chunks {where}
            ORDER BY embedding <=> %(query)s::vector
            LIMIT %(k)s
        """
        params = {"query": query, "k": k, "doc_ids": sorted(doc_ids) if doc_ids else None}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            ScoredChunk(chunk=self._row_to_chunk(r), score=r["score"], vector_score=r["score"])
            for r in rows
        ]

    def lexical_search(
        self, query: str, k: int, doc_ids: set[str] | None = None
    ) -> list[ScoredChunk]:
        where = "AND doc_id = ANY(%(doc_ids)s)" if doc_ids is not None else ""
        sql = f"""
            SELECT {_SELECT_COLUMNS},
                   ts_rank_cd(tsv, plainto_tsquery('portuguese', %(query)s)) AS score
            FROM chunks
            WHERE tsv @@ plainto_tsquery('portuguese', %(query)s) {where}
            ORDER BY score DESC
            LIMIT %(k)s
        """
        params = {"query": query, "k": k, "doc_ids": sorted(doc_ids) if doc_ids else None}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            ScoredChunk(chunk=self._row_to_chunk(r), score=r["score"], lexical_score=r["score"])
            for r in rows
        ]
