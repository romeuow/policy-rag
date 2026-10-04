"""create chunks table with pgvector + full-text index

Revision ID: 0001
Revises:
Create Date: 2026-10-04
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Keep in sync with EMBEDDING_DIM (voyage-3 => 1024).
EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE chunks (
            chunk_id      TEXT PRIMARY KEY,
            doc_id        TEXT NOT NULL,
            doc_title     TEXT NOT NULL,
            section       TEXT NOT NULL DEFAULT '',
            version       TEXT NOT NULL DEFAULT '1.0',
            position      INTEGER NOT NULL DEFAULT 0,
            text          TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            embedding     vector({EMBEDDING_DIM}) NOT NULL,
            tsv           tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', text)) STORED,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX chunks_doc_id_idx ON chunks (doc_id)")
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw_idx ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX chunks_tsv_gin_idx ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
