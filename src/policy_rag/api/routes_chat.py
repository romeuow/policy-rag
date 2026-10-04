"""POST /chat (SSE) and POST /feedback."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from policy_rag.api.deps import ContainerDep
from policy_rag.api.schemas import ChatRequest, FeedbackRequest
from policy_rag.api.sse import encode_event
from policy_rag.models import Feedback
from policy_rag.rag.guardrails import (
    EmptyQuestionError,
    QuestionTooLongError,
    validate_question,
)
from policy_rag.rag.pipeline import CitationsEvent, DoneEvent, TokenEvent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(body: ChatRequest, container: ContainerDep) -> StreamingResponse:
    try:
        question = validate_question(body.question, container.settings.max_question_chars)
    except QuestionTooLongError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except EmptyQuestionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    container.metrics.inc("questions_total")
    started = time.perf_counter()

    def event_stream() -> Iterator[str]:
        try:
            events = container.pipeline.answer(
                session_id=body.session_id,
                question=question,
                history=container.sessions.history(body.session_id),
                faqs=container.faqs.list(),
                active_doc_ids=container.documents.active_doc_ids(),
            )
            for event in events:
                if isinstance(event, TokenEvent):
                    yield encode_event("token", {"text": event.text})
                elif isinstance(event, CitationsEvent):
                    yield encode_event(
                        "citations", [c.model_dump(mode="json") for c in event.citations]
                    )
                elif isinstance(event, DoneEvent):
                    answer = event.answer
                    container.answers.add(answer)
                    container.sessions.append(body.session_id, question, answer.text)
                    if answer.source == "faq":
                        container.metrics.inc("faq_hits_total")
                    elif answer.source == "none":
                        container.metrics.inc("no_answer_total")
                    else:
                        container.metrics.inc("rag_answers_total")
                    if answer.flagged_chunks:
                        container.metrics.inc("injection_flags_total", len(answer.flagged_chunks))
                    logger.info(
                        "chat_answered",
                        extra={
                            "session_id": body.session_id,
                            "message_id": answer.message_id,
                            "source": answer.source,
                            "citations": len(answer.citations),
                            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        },
                    )
                    yield encode_event(
                        "done",
                        {
                            "message_id": answer.message_id,
                            "source": answer.source,
                            "session_id": body.session_id,
                        },
                    )
        except Exception:
            container.metrics.inc("errors_total")
            logger.exception("chat_failed", extra={"session_id": body.session_id})
            yield encode_event("error", {"message": "internal error while answering"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def feedback(body: FeedbackRequest, container: ContainerDep) -> dict:
    answer = container.answers.get(body.message_id)
    if answer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown message_id")
    item = container.feedback.add(
        Feedback(
            message_id=body.message_id,
            rating=body.rating,
            comment=body.comment,
            question=answer.question,
            answer=answer.text,
        )
    )
    container.metrics.inc(f"feedback_{body.rating}_total")
    logger.info("feedback_received", extra={"message_id": body.message_id, "rating": body.rating})
    return item.model_dump(mode="json")
