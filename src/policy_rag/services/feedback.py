"""In-memory stores for answers, feedback and FAQs."""

from __future__ import annotations

from collections import OrderedDict

from policy_rag.models import Answer, Faq, Feedback


class AnswerStore:
    """Keeps recent answers so feedback can be linked to the question/answer text."""

    def __init__(self, max_items: int = 2000) -> None:
        self._items: OrderedDict[str, Answer] = OrderedDict()
        self._max = max_items

    def add(self, answer: Answer) -> None:
        self._items[answer.message_id] = answer
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, message_id: str) -> Answer | None:
        return self._items.get(message_id)


class FeedbackStore:
    def __init__(self) -> None:
        self._items: list[Feedback] = []

    def add(self, feedback: Feedback) -> Feedback:
        self._items.append(feedback)
        return feedback

    def list(self, rating: str | None = None) -> list[Feedback]:
        items = self._items if rating is None else [f for f in self._items if f.rating == rating]
        return sorted(items, key=lambda f: f.created_at, reverse=True)


class FaqStore:
    def __init__(self) -> None:
        self._items: dict[str, Faq] = {}

    def add(self, faq: Faq) -> Faq:
        self._items[faq.faq_id] = faq
        return faq

    def list(self) -> list[Faq]:
        return sorted(self._items.values(), key=lambda f: f.created_at)

    def delete(self, faq_id: str) -> bool:
        return self._items.pop(faq_id, None) is not None
