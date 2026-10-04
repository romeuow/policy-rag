import pytest

from policy_rag.models import Faq
from policy_rag.rag.faq import FaqMatcher, jaccard


def test_jaccard_handles_empty_and_identical():
    assert jaccard("", "férias") == 0.0
    assert jaccard("quantos dias de férias", "quantos dias de férias") == 1.0


def test_faq_matcher_exact_and_paraphrase(embedder):
    faqs = [
        Faq(question="Quantos dias de férias tenho por ano?", answer="30 dias corridos."),
        Faq(question="Qual o limite de alimentação em viagem?", answer="R$ 120 por dia."),
    ]
    matcher = FaqMatcher(embedder, threshold=0.8)
    exact = matcher.match("Quantos dias de férias tenho por ano?", faqs)
    assert exact is not None and exact.faq.answer == "30 dias corridos."
    assert exact.score == pytest.approx(1.0)
    close = matcher.match("quantos dias de férias eu tenho por ano", faqs)
    assert close is not None and close.faq.answer == "30 dias corridos."
    assert matcher.match("Como configuro a VPN corporativa?", faqs) is None
    assert matcher.match("qualquer coisa", []) is None


def test_faq_matcher_validates_threshold(embedder):
    with pytest.raises(ValueError):
        FaqMatcher(embedder, threshold=0)
