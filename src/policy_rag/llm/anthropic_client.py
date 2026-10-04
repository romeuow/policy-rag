"""Production LLM client using the official `anthropic` SDK with streaming."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from policy_rag.llm.base import LLMRefusalError
from policy_rag.models import ChatMessage, Chunk
from policy_rag.rag.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    """Streams grounded answers via `client.messages.stream(...)` and its `text_stream`."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        max_tokens: int = 2048,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def stream_answer(
        self, question: str, chunks: list[Chunk], history: list[ChatMessage]
    ) -> Iterator[str]:
        messages: list[dict[str, str]] = [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": build_user_prompt(question, chunks)})
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            logger.warning("llm_refusal", extra={"model": self.model})
            raise LLMRefusalError("The model refused to answer this question.")
        if final.stop_reason == "max_tokens":
            logger.warning("llm_truncated", extra={"model": self.model})
