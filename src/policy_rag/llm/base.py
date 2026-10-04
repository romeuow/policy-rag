"""LLM client protocol."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from policy_rag.models import ChatMessage, Chunk


class LLMRefusalError(RuntimeError):
    """Raised when the model refuses to answer (stop_reason == 'refusal')."""


@runtime_checkable
class LLMClient(Protocol):
    def stream_answer(
        self, question: str, chunks: list[Chunk], history: list[ChatMessage]
    ) -> Iterator[str]:
        """Yield answer text fragments grounded on `chunks` (cited as [n], 1-based)."""
        ...
