"""Short per-session chat history kept in memory."""

from __future__ import annotations

from collections import deque

from policy_rag.models import ChatMessage


class SessionStore:
    def __init__(self, max_turns: int = 6) -> None:
        self._max_messages = max_turns * 2
        self._sessions: dict[str, deque[ChatMessage]] = {}

    def history(self, session_id: str) -> list[ChatMessage]:
        return list(self._sessions.get(session_id, ()))

    def append(self, session_id: str, question: str, answer: str) -> None:
        bucket = self._sessions.setdefault(session_id, deque(maxlen=self._max_messages))
        bucket.append(ChatMessage(role="user", content=question))
        bucket.append(ChatMessage(role="assistant", content=answer))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
