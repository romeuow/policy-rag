from pathlib import Path

from policy_rag.services.documents import DocumentService
from tests.conftest import SAMPLE_MARKDOWN


def test_ingest_text_is_idempotent_and_reindex_removes_stale_chunks(store, embedder):
    service = DocumentService(store, embedder, chunk_max_chars=300, chunk_overlap=40)
    doc = service.ingest_text(SAMPLE_MARKDOWN, "politica-de-teste.md")
    count = store.count()
    assert doc.chunk_count == count > 1
    assert service.ingest_text(SAMPLE_MARKDOWN, "politica-de-teste.md").chunk_count == count
    assert store.count() == count

    shorter = SAMPLE_MARKDOWN.split("## 2. Regras")[0]
    updated = service.ingest_text(shorter, "politica-de-teste.md")
    assert updated.chunk_count < count
    assert store.count() == updated.chunk_count
    assert updated.created_at == doc.created_at
    assert updated.updated_at >= doc.updated_at


def test_reindex_delete_and_active_flag(store, embedder):
    service = DocumentService(store, embedder)
    doc = service.ingest_text(SAMPLE_MARKDOWN, "politica-de-teste.md", doc_id="custom-id")
    assert doc.doc_id == "custom-id"
    assert service.reindex("custom-id").chunk_count == doc.chunk_count
    assert service.reindex("missing") is None
    assert service.set_active("custom-id", False).active is False
    assert service.active_doc_ids() == set()
    assert service.set_active("missing", True) is None
    assert service.delete("custom-id") is True
    assert service.delete("custom-id") is False
    assert store.count() == 0


def test_ingest_directory_reads_supported_files(store, embedder, tmp_path: Path):
    (tmp_path / "a.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    (tmp_path / "b.txt").write_text("Texto simples de política.", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    service = DocumentService(store, embedder)
    docs = service.ingest_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert service.get("b").title == "b"
    assert len(service.list()) == 2
