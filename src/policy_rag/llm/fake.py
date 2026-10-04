"""Extractive fake LLM used in tests and demo mode. No network, deterministic."""

from __future__ import annotations

import re
from collections.abc import Iterator

from policy_rag.models import ChatMessage, Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class FakeLLMClient:
    def __init__(self, max_sentences_per_chunk: int = 2, max_chunks: int = 3) -> None:
        self.max_sentences = max_sentences_per_chunk
        self.max_chunks = max_chunks
        self.calls: list[dict] = []

    @staticmethod
    def _body(chunk: Chunk) -> str:
        # Chunk text starts with the section heading on its own line; skip it.
        _, _, body = chunk.text.partition("\n") if chunk.section else ("", "", chunk.text)
        return " ".join(body.split())

    def stream_answer(
        self, question: str, chunks: list[Chunk], history: list[ChatMessage]
    ) -> Iterator[str]:
        self.calls.append({"question": question, "chunks": len(chunks), "history": len(history)})
        if not chunks:
            yield "Não encontrei essa informação na base de políticas."
            return
        parts: list[str] = ["Com base nas políticas internas:"]
        for idx, chunk in enumerate(chunks[: self.max_chunks], start=1):
            sentences = _SENTENCE.split(self._body(chunk))[: self.max_sentences]
            parts.append(f"{' '.join(sentences)} [{idx}]")
        text = "\n\n".join(parts)
        # Emit word + trailing whitespace together so the stream has one event per word.
        yield from (tok for tok in re.split(r"(?<=\s)(?=\S)", text) if tok)
