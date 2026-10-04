import pytest

from policy_rag.api.container import build_container, build_embedder, build_llm, build_store
from policy_rag.config import Settings
from policy_rag.embeddings.fake import FakeEmbeddingProvider
from policy_rag.llm.fake import FakeLLMClient
from policy_rag.store.memory import InMemoryVectorStore


def test_demo_mode_wires_fakes():
    settings = Settings(app_mode="demo", corpus_dir=None)
    container = build_container(settings)
    assert isinstance(container.embedder, FakeEmbeddingProvider)
    assert isinstance(container.store, InMemoryVectorStore)
    assert isinstance(container.llm, FakeLLMClient)
    assert container.ingest_corpus() == 0


def test_missing_corpus_dir_is_tolerated(tmp_path):
    settings = Settings(app_mode="demo", corpus_dir=str(tmp_path / "missing"))
    assert build_container(settings).ingest_corpus() == 0


def test_prod_mode_requires_credentials(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(app_mode="prod", voyage_api_key=None, database_url=None)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        build_embedder(settings)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        build_store(settings)


def test_prod_mode_builds_real_clients(monkeypatch: pytest.MonkeyPatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())
    settings = Settings(
        app_mode="prod",
        voyage_api_key="voyage-test",
        database_url="postgresql://user:pass@localhost:5432/db",
        llm_model="claude-opus-5-5",
    )
    assert build_llm(settings).model == "claude-opus-5-5"
    assert build_embedder(settings).model == settings.voyage_model
    assert type(build_store(settings)).__name__ == "PgVectorStore"
