"""End-to-end question answering: FAQ short-circuit -> hybrid retrieval -> guardrails -> LLM."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from policy_rag.llm.base import LLMClient, LLMRefusalError
from policy_rag.models import Answer, ChatMessage, Citation, Faq, new_id
from policy_rag.rag.faq import FaqMatcher
from policy_rag.rag.guardrails import NO_ANSWER_TEXT, screen_chunks
from policy_rag.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class TokenEvent:
    text: str


@dataclass
class CitationsEvent:
    citations: list[Citation]


@dataclass
class DoneEvent:
    answer: Answer


PipelineEvent = TokenEvent | CitationsEvent | DoneEvent


def _excerpt(text: str, limit: int = 240) -> str:
    _, _, body = text.partition("\n")
    body = " ".join((body or text).split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        faq_matcher: FaqMatcher,
        min_relevance: float = 0.30,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._faq = faq_matcher
        self.min_relevance = min_relevance

    def answer(
        self,
        session_id: str,
        question: str,
        history: list[ChatMessage],
        faqs: list[Faq],
        active_doc_ids: set[str] | None,
    ) -> Iterator[PipelineEvent]:
        message_id = new_id()

        faq_match = self._faq.match(question, faqs)
        if faq_match:
            logger.info("faq_hit", extra={"faq_id": faq_match.faq.faq_id, "score": faq_match.score})
            yield TokenEvent(faq_match.faq.answer)
            yield CitationsEvent([])
            yield DoneEvent(
                Answer(
                    message_id=message_id,
                    session_id=session_id,
                    question=question,
                    text=faq_match.faq.answer,
                    source="faq",
                )
            )
            return

        retrieval = self._retriever.retrieve(question, active_doc_ids)
        screening = screen_chunks([s.chunk for s in retrieval.results])
        if screening.flagged:
            logger.warning(
                "prompt_injection_suspected",
                extra={"chunk_ids": list(screening.flagged), "patterns": screening.flagged},
            )
        safe_ids = {c.chunk_id for c in screening.safe}
        results = [s for s in retrieval.results if s.chunk.chunk_id in safe_ids]

        if not results or retrieval.relevance < self.min_relevance:
            logger.info(
                "no_answer",
                extra={"relevance": round(retrieval.relevance, 4), "results": len(results)},
            )
            yield TokenEvent(NO_ANSWER_TEXT)
            yield CitationsEvent([])
            yield DoneEvent(
                Answer(
                    message_id=message_id,
                    session_id=session_id,
                    question=question,
                    text=NO_ANSWER_TEXT,
                    source="none",
                    flagged_chunks=list(screening.flagged),
                )
            )
            return

        citations = [
            Citation(
                index=idx,
                doc_id=s.chunk.doc_id,
                doc_title=s.chunk.doc_title,
                section=s.chunk.section,
                chunk_id=s.chunk.chunk_id,
                score=round(s.score, 6),
                excerpt=_excerpt(s.chunk.text),
            )
            for idx, s in enumerate(results, start=1)
        ]

        parts: list[str] = []
        try:
            for fragment in self._llm.stream_answer(question, [s.chunk for s in results], history):
                parts.append(fragment)
                yield TokenEvent(fragment)
        except LLMRefusalError:
            parts = [NO_ANSWER_TEXT]
            yield TokenEvent(NO_ANSWER_TEXT)
            citations = []

        yield CitationsEvent(citations)
        yield DoneEvent(
            Answer(
                message_id=message_id,
                session_id=session_id,
                question=question,
                text="".join(parts),
                source="rag" if citations else "none",
                citations=citations,
                flagged_chunks=list(screening.flagged),
            )
        )
