"""Domain models shared across the package."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    version: str
    position: int
    text: str
    content_hash: str
    embedding: list[float] = Field(default_factory=list)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0


class Document(BaseModel):
    doc_id: str
    title: str
    version: str = "1.0"
    source_name: str
    content_hash: str
    active: bool = True
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Citation(BaseModel):
    index: int
    doc_id: str
    doc_title: str
    section: str
    chunk_id: str
    score: float
    excerpt: str


AnswerSource = Literal["rag", "faq", "none"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Answer(BaseModel):
    message_id: str = Field(default_factory=new_id)
    session_id: str
    question: str
    text: str
    source: AnswerSource
    citations: list[Citation] = Field(default_factory=list)
    flagged_chunks: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class Faq(BaseModel):
    faq_id: str = Field(default_factory=new_id)
    question: str
    answer: str
    created_at: datetime = Field(default_factory=_now)


class Feedback(BaseModel):
    feedback_id: str = Field(default_factory=new_id)
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = None
    question: str | None = None
    answer: str | None = None
    created_at: datetime = Field(default_factory=_now)
