from pathlib import Path

from fastapi.testclient import TestClient

from policy_rag.api.app import create_app
from policy_rag.api.sse import parse_sse
from tests.conftest import SAMPLE_MARKDOWN


def _chat(client: TestClient, question: str, session_id: str = "s1"):
    response = client.post("/chat", json={"session_id": session_id, "question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_health_and_metrics(client: TestClient):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] == 7 and health["chunks"] > 0
    assert client.get("/metrics").json()["questions_total"] == 0


def test_chat_streams_tokens_citations_and_done(client: TestClient):
    events = _chat(client, "Qual o limite diário de alimentação em viagem nacional?")
    kinds = [e.event for e in events]
    assert kinds[0] == "token" and kinds[-2:] == ["citations", "done"]
    assert kinds.count("token") > 1
    text = "".join(e.data["text"] for e in events if e.event == "token")
    assert "R$ 120,00" in text
    citations = events[-2].data
    assert citations[0]["doc_id"] == "02-politica-de-reembolso"
    assert {"doc_title", "section", "chunk_id", "score"} <= set(citations[0])
    done = events[-1].data
    assert done["source"] == "rag" and done["message_id"]
    assert client.get("/metrics").json()["rag_answers_total"] == 1


def test_chat_keeps_session_history(client: TestClient):
    _chat(client, "Quantos dias de férias eu tenho por ano?", session_id="hist")
    _chat(client, "E posso fracionar?", session_id="hist")
    history = client.app.state.container.sessions.history("hist")
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]


def test_chat_rejects_long_or_empty_questions(client: TestClient):
    too_long = "a" * (client.app.state.container.settings.max_question_chars + 1)
    assert client.post("/chat", json={"session_id": "s", "question": too_long}).status_code == 413
    assert client.post("/chat", json={"session_id": "s", "question": "   "}).status_code == 422
    assert client.post("/chat", json={"question": "x"}).status_code == 422


def test_chat_no_answer_increments_metric(client: TestClient):
    events = _chat(client, "Qual a receita de bolo de cenoura?")
    assert events[-1].data["source"] == "none"
    assert client.get("/metrics").json()["no_answer_total"] == 1


def test_chat_error_is_reported_as_sse_event(client: TestClient):
    container = client.app.state.container

    class _Broken:
        def retrieve(self, *a, **kw):
            raise RuntimeError("boom")

    container.pipeline._retriever = _Broken()
    events = _chat(client, "Quantos dias de férias?")
    assert events[-1].event == "error"
    assert client.get("/metrics").json()["errors_total"] == 1


def test_feedback_persists_and_is_listed_for_admin(client: TestClient, admin_headers):
    done = _chat(client, "Quantos dias de férias eu tenho por ano?")[-1].data
    response = client.post(
        "/feedback",
        json={"message_id": done["message_id"], "rating": "down", "comment": "incompleta"},
    )
    assert response.status_code == 201
    assert response.json()["question"].startswith("Quantos dias")
    assert client.post("/feedback", json={"message_id": "nope", "rating": "up"}).status_code == 404
    listed = client.get("/admin/feedback", params={"rating": "down"}, headers=admin_headers).json()
    assert len(listed) == 1 and listed[0]["comment"] == "incompleta"
    assert (
        client.get("/admin/feedback", params={"rating": "x"}, headers=admin_headers).status_code
        == 422
    )
    assert client.get("/metrics").json()["feedback_down_total"] == 1


def test_admin_requires_token(client: TestClient):
    assert client.get("/admin/documents").status_code == 401
    assert client.get("/admin/documents", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/admin/faqs").status_code == 401
    assert client.get("/admin/feedback").status_code == 401


def test_admin_document_crud_and_reindex(client: TestClient, admin_headers):
    docs = client.get("/admin/documents", headers=admin_headers).json()
    assert len(docs) == 7

    created = client.post(
        "/admin/documents",
        json={"filename": "politica-de-teste.md", "content": SAMPLE_MARKDOWN},
        headers=admin_headers,
    )
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]
    assert created.json()["title"] == "Política de Teste"
    assert len(client.get("/admin/documents", headers=admin_headers).json()) == 8

    bad = client.post(
        "/admin/documents", json={"filename": "evil.exe", "content": "x"}, headers=admin_headers
    )
    assert bad.status_code == 422

    reindexed = client.post(f"/admin/documents/{doc_id}/reindex", headers=admin_headers)
    assert reindexed.status_code == 200 and reindexed.json()["chunk_count"] > 0
    assert client.post("/admin/documents/nope/reindex", headers=admin_headers).status_code == 404

    patched = client.patch(
        f"/admin/documents/{doc_id}", json={"active": False}, headers=admin_headers
    )
    assert patched.status_code == 200 and patched.json()["active"] is False
    assert (
        client.patch(
            "/admin/documents/nope", json={"active": True}, headers=admin_headers
        ).status_code
        == 404
    )

    assert client.delete(f"/admin/documents/{doc_id}", headers=admin_headers).status_code == 204
    assert client.delete(f"/admin/documents/{doc_id}", headers=admin_headers).status_code == 404


def test_admin_faq_crud_and_faq_hit(client: TestClient, admin_headers):
    created = client.post(
        "/admin/faqs",
        json={"question": "Qual o telefone do RH?", "answer": "11 90000-0000"},
        headers=admin_headers,
    )
    assert created.status_code == 201
    faq_id = created.json()["faq_id"]
    assert len(client.get("/admin/faqs", headers=admin_headers).json()) == 1

    events = _chat(client, "Qual o telefone do RH?")
    assert events[0].data["text"] == "11 90000-0000"
    assert events[-1].data["source"] == "faq"
    assert client.get("/metrics").json()["faq_hits_total"] == 1

    assert client.delete(f"/admin/faqs/{faq_id}", headers=admin_headers).status_code == 204
    assert client.delete(f"/admin/faqs/{faq_id}", headers=admin_headers).status_code == 404


def test_frontend_is_served_when_dist_exists(settings, container, tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "robots.txt").write_text("User-agent: *", encoding="utf-8")
    settings.web_dist_dir = str(dist)
    app = create_app(settings=settings, container=container, ingest_on_startup=False)
    with TestClient(app) as client:
        assert client.get("/").text == "<html>spa</html>"
        assert client.get("/admin").text == "<html>spa</html>"
        assert client.get("/robots.txt").text == "User-agent: *"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/admin/documents").status_code == 401


def test_unhandled_exception_returns_500(settings, container):
    app = create_app(settings=settings, container=container, ingest_on_startup=False)

    @app.get("/boom")
    def boom():
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")
    assert response.status_code == 500 and response.json()["detail"] == "internal server error"
