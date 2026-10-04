import json
import logging

import pytest

from policy_rag.logging import JsonFormatter, configure_logging
from policy_rag.models import Answer, Faq, Feedback
from policy_rag.services.feedback import AnswerStore, FaqStore, FeedbackStore
from policy_rag.services.metrics import Metrics
from policy_rag.services.sessions import SessionStore


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    record.session_id = "s1"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello x"
    assert payload["session_id"] == "s1"
    assert payload["level"] == "INFO"
    assert "ts" in payload


def test_json_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "err", (), True)
        import sys

        record.exc_info = sys.exc_info()
    assert "boom" in json.loads(JsonFormatter().format(record))["exc_info"]


def test_configure_logging_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_session_store_keeps_bounded_history():
    sessions = SessionStore(max_turns=2)
    for i in range(4):
        sessions.append("s", f"q{i}", f"a{i}")
    history = sessions.history("s")
    assert [m.content for m in history] == ["q2", "a2", "q3", "a3"]
    assert sessions.history("other") == []
    sessions.clear("s")
    assert sessions.history("s") == []


def test_metrics_counters():
    metrics = Metrics()
    metrics.inc("questions_total")
    metrics.inc("feedback_up_total", 2)
    snap = metrics.snapshot()
    assert snap["questions_total"] == 1 and snap["feedback_up_total"] == 2
    with pytest.raises(KeyError):
        metrics.inc("nope")


def test_answer_feedback_and_faq_stores():
    answers = AnswerStore(max_items=2)
    for i in range(3):
        answers.add(
            Answer(message_id=f"m{i}", session_id="s", question="q", text="a", source="rag")
        )
    assert answers.get("m0") is None and answers.get("m2") is not None

    feedback = FeedbackStore()
    feedback.add(Feedback(message_id="m2", rating="down", comment="ruim"))
    feedback.add(Feedback(message_id="m1", rating="up"))
    assert [f.rating for f in feedback.list("down")] == ["down"]
    assert len(feedback.list()) == 2

    faqs = FaqStore()
    faq = faqs.add(Faq(question="Q?", answer="A"))
    assert faqs.list() == [faq]
    assert faqs.delete(faq.faq_id) is True
    assert faqs.delete(faq.faq_id) is False
