"""Wires concrete implementations according to `APP_MODE`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from policy_rag.config import Settings
from policy_rag.embeddings.base import EmbeddingProvider
from policy_rag.embeddings.fake import FakeEmbeddingProvider
from policy_rag.llm.base import LLMClient
from policy_rag.llm.fake import FakeLLMClient
from policy_rag.rag.faq import FaqMatcher
from policy_rag.rag.pipeline import RagPipeline
from policy_rag.rag.retriever import HybridRetriever
from policy_rag.services.documents import DocumentService
from policy_rag.services.feedback import AnswerStore, FaqStore, FeedbackStore
from policy_rag.services.metrics import Metrics
from policy_rag.services.sessions import SessionStore
from policy_rag.store.base import VectorStore
from policy_rag.store.memory import InMemoryVectorStore

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: EmbeddingProvider
    store: VectorStore
    llm: LLMClient
    documents: DocumentService
    pipeline: RagPipeline
    sessions: SessionStore
    metrics: Metrics
    answers: AnswerStore
    feedback: FeedbackStore
    faqs: FaqStore

    def ingest_corpus(self) -> int:
        if not self.settings.corpus_dir:
            return 0
        path = Path(self.settings.corpus_dir)
        if not path.is_dir():
            logger.warning("corpus_dir_missing", extra={"path": str(path)})
            return 0
        docs = self.documents.ingest_directory(path)
        logger.info("corpus_ingested", extra={"documents": len(docs), "chunks": self.store.count()})
        return len(docs)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.app_mode == "demo":
        return FakeEmbeddingProvider(dim=settings.embedding_dim)
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is required in prod mode")
    from policy_rag.embeddings.voyage import VoyageEmbeddingProvider

    return VoyageEmbeddingProvider(
        api_key=settings.voyage_api_key,
        model=settings.voyage_model,
        base_url=settings.voyage_base_url,
        dim=settings.embedding_dim,
    )


def build_store(settings: Settings) -> VectorStore:
    if settings.app_mode == "demo":
        return InMemoryVectorStore()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required in prod mode")
    from policy_rag.store.pgvector import PgVectorStore

    return PgVectorStore(settings.database_url, embedding_dim=settings.embedding_dim)


def build_llm(settings: Settings) -> LLMClient:
    if settings.app_mode == "demo":
        return FakeLLMClient()
    from policy_rag.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens)


def build_container(
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
) -> Container:
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    llm = llm or build_llm(settings)
    documents = DocumentService(
        store, embedder, settings.chunk_max_chars, settings.chunk_overlap_chars
    )
    retriever = HybridRetriever(
        store,
        embedder,
        top_k_vector=settings.top_k_vector,
        top_k_lexical=settings.top_k_lexical,
        top_k_final=settings.top_k_final,
        rrf_k=settings.rrf_k,
    )
    pipeline = RagPipeline(
        retriever,
        llm,
        FaqMatcher(embedder, threshold=settings.faq_match_threshold),
        min_relevance=settings.min_relevance_score,
    )
    return Container(
        settings=settings,
        embedder=embedder,
        store=store,
        llm=llm,
        documents=documents,
        pipeline=pipeline,
        sessions=SessionStore(max_turns=settings.session_history_turns),
        metrics=Metrics(),
        answers=AnswerStore(),
        feedback=FeedbackStore(),
        faqs=FaqStore(),
    )
