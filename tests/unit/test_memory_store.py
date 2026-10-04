from policy_rag.ingest.chunker import chunk_document
from policy_rag.store.memory import InMemoryVectorStore


def _ingest(store, embedder, doc):
    chunks = chunk_document(doc, max_chars=300, overlap=40)
    for chunk, vec in zip(chunks, embedder.embed_documents([c.text for c in chunks]), strict=True):
        chunk.embedding = vec
    store.upsert(chunks)
    return chunks


def test_upsert_is_idempotent_and_delete_works(store, embedder, sample_doc):
    chunks = _ingest(store, embedder, sample_doc)
    assert store.count() == len(chunks)
    _ingest(store, embedder, sample_doc)
    assert store.count() == len(chunks)
    assert store.chunk_ids("politica-de-teste") == {c.chunk_id for c in chunks}
    assert store.get(chunks[0].chunk_id) is not None
    assert store.delete_chunks([chunks[0].chunk_id, "missing"]) == 1
    assert store.delete_document("politica-de-teste") == len(chunks) - 1
    assert store.count() == 0
    assert store.lexical_search("prazo", 3) == []


def test_lexical_search_ranks_matching_section_first(store, embedder, sample_doc):
    _ingest(store, embedder, sample_doc)
    results = store.lexical_search("qual o limite de reembolso por dia?", k=3)
    assert results
    assert "Regra B" in results[0].chunk.section
    assert results[0].lexical_score > 0


def test_vector_search_and_doc_filter(store, embedder, sample_doc):
    _ingest(store, embedder, sample_doc)
    query = embedder.embed_query("prazo de solicitação em dias úteis")
    results = store.vector_search(query, k=2)
    assert len(results) == 2
    assert results[0].vector_score >= results[1].vector_score
    assert store.vector_search(query, k=2, doc_ids={"outro"}) == []
    assert store.lexical_search("prazo", k=2, doc_ids={"outro"}) == []
    assert store.lexical_search("prazo", k=2, doc_ids={"politica-de-teste"})


def test_cosine_edge_cases():
    assert InMemoryVectorStore.cosine([], [1.0]) == 0.0
    assert InMemoryVectorStore.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert InMemoryVectorStore.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
