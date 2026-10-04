"""Simple in-memory counters exposed at GET /metrics."""

from __future__ import annotations

from threading import Lock


class Metrics:
    _KEYS = (
        "questions_total",
        "faq_hits_total",
        "no_answer_total",
        "rag_answers_total",
        "feedback_up_total",
        "feedback_down_total",
        "injection_flags_total",
        "errors_total",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters = dict.fromkeys(self._KEYS, 0)

    def inc(self, key: str, amount: int = 1) -> None:
        if key not in self._counters:
            raise KeyError(f"unknown metric: {key}")
        with self._lock:
            self._counters[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)
