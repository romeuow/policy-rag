from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from policy_rag.llm.anthropic_client import AnthropicLLMClient
from policy_rag.llm.base import LLMClient, LLMRefusalError
from policy_rag.llm.fake import FakeLLMClient
from policy_rag.models import ChatMessage, Chunk
from policy_rag.rag.prompts import SYSTEM_PROMPT, build_user_prompt


def _chunk(cid: str, text: str, section: str = "Seção") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", doc_title="Doc", section=section, version="1", position=0,
        text=f"{section}\n{text}" if section else text, content_hash="h",
    )  # fmt: skip


def test_fake_llm_is_extractive_and_cites_chunks():
    llm = FakeLLMClient()
    assert isinstance(llm, LLMClient)
    chunks = [_chunk("a", "Primeira frase. Segunda frase. Terceira frase."), _chunk("b", "Outra.")]
    text = "".join(llm.stream_answer("pergunta", chunks, []))
    assert "Primeira frase. Segunda frase. [1]" in text
    assert "Terceira" not in text
    assert "Outra. [2]" in text
    assert llm.calls[0]["chunks"] == 2


def test_fake_llm_without_chunks_says_not_found():
    text = "".join(FakeLLMClient().stream_answer("q", [], []))
    assert "Não encontrei" in text


class _StubStream:
    def __init__(self, texts, stop_reason):
        self._texts = texts
        self._stop_reason = stop_reason

    @property
    def text_stream(self):
        yield from self._texts

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop_reason, content=[])


class _StubMessages:
    def __init__(self, texts, stop_reason="end_turn"):
        self.kwargs = None
        self._stream = _StubStream(texts, stop_reason)

    @contextmanager
    def stream(self, **kwargs):
        self.kwargs = kwargs
        yield self._stream


def test_anthropic_client_streams_text_and_builds_prompt():
    messages = _StubMessages(["Resposta ", "[1]"])
    client = AnthropicLLMClient(model="claude-opus-5-5", client=SimpleNamespace(messages=messages))
    chunks = [_chunk("a", "Férias de 30 dias.")]
    history = [ChatMessage(role="user", content="oi"), ChatMessage(role="assistant", content="olá")]
    out = list(client.stream_answer("Quantos dias?", chunks, history))
    assert out == ["Resposta ", "[1]"]
    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["max_tokens"] == 2048
    assert "temperature" not in kwargs and "thinking" not in kwargs
    assert kwargs["messages"][-1] == {
        "role": "user",
        "content": build_user_prompt("Quantos dias?", chunks),
    }
    assert kwargs["messages"][0] == {"role": "user", "content": "oi"}
    assert "<trechos>" in kwargs["messages"][-1]["content"]
    assert 'documento="Doc"' in kwargs["messages"][-1]["content"]


def test_anthropic_client_raises_on_refusal():
    messages = _StubMessages([], stop_reason="refusal")
    client = AnthropicLLMClient(client=SimpleNamespace(messages=messages))
    with pytest.raises(LLMRefusalError):
        list(client.stream_answer("q", [_chunk("a", "x")], []))


def test_anthropic_client_logs_truncation(caplog):
    messages = _StubMessages(["x"], stop_reason="max_tokens")
    client = AnthropicLLMClient(client=SimpleNamespace(messages=messages))
    with caplog.at_level("WARNING"):
        assert list(client.stream_answer("q", [], [])) == ["x"]
    assert "llm_truncated" in caplog.text


def test_anthropic_client_default_constructor_uses_sdk(monkeypatch: pytest.MonkeyPatch):
    import anthropic

    created = {}

    class _FakeAnthropic:
        def __init__(self):
            created["ok"] = True

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    client = AnthropicLLMClient()
    assert created["ok"] and client.model == "claude-opus-5-5"
