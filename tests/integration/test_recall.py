"""Retrieval quality on the fictional corpus using only the offline fakes."""

from pathlib import Path

import yaml

from tests.conftest import FIXTURES_DIR


def _questions() -> list[dict]:
    return yaml.safe_load(Path(FIXTURES_DIR / "questions.yaml").read_text(encoding="utf-8"))


def test_corpus_ingests_all_documents(ingested):
    docs = ingested.documents.list()
    assert len(docs) == 7
    assert all(d.chunk_count > 0 for d in docs)
    assert ingested.store.count() == sum(d.chunk_count for d in docs)


def test_recall_at_3_is_at_least_80_percent(ingested):
    questions = _questions()
    hits = 0
    misses = []
    for item in questions:
        result = ingested.pipeline._retriever.retrieve(
            item["question"], ingested.documents.active_doc_ids()
        )
        top_docs = [s.chunk.doc_id for s in result.results[:3]]
        if item["expected_doc"] in top_docs:
            hits += 1
        else:
            misses.append((item["question"], top_docs))
    recall = hits / len(questions)
    assert recall >= 0.8, f"recall@3={recall:.2f}, misses={misses}"


def test_out_of_domain_question_scores_below_threshold(ingested):
    result = ingested.pipeline._retriever.retrieve(
        "Qual a receita de bolo de cenoura?", ingested.documents.active_doc_ids()
    )
    assert result.relevance < ingested.pipeline.min_relevance
