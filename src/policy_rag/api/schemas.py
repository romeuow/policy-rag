"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=10_000)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=128)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class DocumentCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=200, pattern=r"^[\w\-. ]+\.(md|markdown|txt)$")
    content: str = Field(min_length=1, max_length=2_000_000)
    doc_id: str | None = Field(default=None, max_length=128, pattern=r"^[a-z0-9\-]+$")


class DocumentPatch(BaseModel):
    active: bool


class FaqCreate(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    answer: str = Field(min_length=1, max_length=5000)
