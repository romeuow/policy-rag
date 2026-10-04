import json
import math

import httpx
import pytest

from policy_rag.embeddings.fake import FakeEmbeddingProvider
from policy_rag.embeddings.voyage import VoyageEmbeddingProvider
from policy_rag.store.memory import InMemoryVectorStore


def test_fake_embeddings_are_deterministic_and_normalised():
    provider = FakeEmbeddingProvider(dim=128)
    a = provider.embed_query("Quantos dias de férias eu tenho?")
    b = provider.embed_query("Quantos dias de férias eu tenho?")
    assert a == b
    assert len(a) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, abs_tol=1e-9)


def test_fake_embeddings_similarity_tracks_lexical_overlap():
    provider = FakeEmbeddingProvider()
    q = provider.embed_query("prazo para solicitar reembolso de despesas")
    close, far = provider.embed_documents(
        [
            "O reembolso de despesas deve ser solicitado em até 30 dias.",
            "A senha deve ter 12 caracteres.",
        ]
    )
    assert InMemoryVectorStore.cosine(q, close) > InMemoryVectorStore.cosine(q, far)


def test_fake_embeddings_handle_empty_text_and_validate_dim():
    provider = FakeEmbeddingProvider(dim=16)
    assert provider.embed_query("") == [0.0] * 16
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dim=0)


def test_voyage_provider_calls_api_and_normalises(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://voyage.test/v1")
    provider = VoyageEmbeddingProvider(api_key="voyage-test-key", dim=2, client=client)
    vectors = provider.embed_documents(["a", "b"])
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]  # ordered by index, L2-normalised
    assert captured["url"].endswith("/v1/embeddings")
    assert captured["auth"] == "Bearer voyage-test-key"
    assert json.loads(captured["json"])["input_type"] == "document"
    assert provider.embed_query("q") == [0.6, 0.8]
    assert provider.embed_documents([]) == []


def test_voyage_provider_requires_key():
    with pytest.raises(ValueError):
        VoyageEmbeddingProvider(api_key="")
