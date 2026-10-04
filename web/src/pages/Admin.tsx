import { useCallback, useEffect, useMemo, useState } from "react";
import { AdminApi, type DocumentInfo, type FaqInfo, type FeedbackInfo } from "../lib/api";

export default function AdminPage() {
  const [token, setToken] = useState(() => sessionStorage.getItem("admin_token") ?? "");
  const api = useMemo(() => (token ? new AdminApi(token) : null), [token]);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [faqs, setFaqs] = useState<FaqInfo[]>([]);
  const [feedback, setFeedback] = useState<FeedbackInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [faqQuestion, setFaqQuestion] = useState("");
  const [faqAnswer, setFaqAnswer] = useState("");

  const refresh = useCallback(async () => {
    if (!api) return;
    try {
      const [d, f, fb] = await Promise.all([api.listDocuments(), api.listFaqs(), api.listNegativeFeedback()]);
      setDocs(d);
      setFaqs(f);
      setFeedback(fb);
      setError(null);
      sessionStorage.setItem("admin_token", token);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [api, token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const run = (action: () => Promise<unknown>) => async () => {
    try {
      await action();
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  async function upload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !api) return;
    const content = await file.text();
    await run(() => api.uploadDocument(file.name, content))();
    event.target.value = "";
  }

  return (
    <section className="admin">
      <label className="token">
        Token de administrador
        <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="X-Admin-Token" />
      </label>
      {error && <p className="error">{error}</p>}
      {!api && <p className="hint">Informe o token para carregar os dados.</p>}

      {api && (
        <>
          <h2>Documentos ({docs.length})</h2>
          <label className="upload">
            Enviar .md <input type="file" accept=".md,.markdown,.txt" onChange={upload} />
          </label>
          <table>
            <thead>
              <tr>
                <th>Título</th>
                <th>Versão</th>
                <th>Chunks</th>
                <th>Ativo</th>
                <th>Ações</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.doc_id}>
                  <td>
                    {d.title}
                    <div className="hint">{d.doc_id}</div>
                  </td>
                  <td>{d.version}</td>
                  <td>{d.chunk_count}</td>
                  <td>
                    <input type="checkbox" checked={d.active} onChange={run(() => api.setActive(d.doc_id, !d.active))} />
                  </td>
                  <td>
                    <button type="button" onClick={run(() => api.reindexDocument(d.doc_id))}>Reindexar</button>
                    <button type="button" className="danger" onClick={run(() => api.deleteDocument(d.doc_id))}>
                      Remover
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2>FAQs ({faqs.length})</h2>
          <form
            className="faq-form"
            onSubmit={(e) => {
              e.preventDefault();
              void run(async () => {
                await api.createFaq(faqQuestion, faqAnswer);
                setFaqQuestion("");
                setFaqAnswer("");
              })();
            }}
          >
            <input value={faqQuestion} onChange={(e) => setFaqQuestion(e.target.value)} placeholder="Pergunta" required minLength={3} />
            <input value={faqAnswer} onChange={(e) => setFaqAnswer(e.target.value)} placeholder="Resposta" required />
            <button type="submit">Adicionar FAQ</button>
          </form>
          <ul className="list">
            {faqs.map((f) => (
              <li key={f.faq_id}>
                <strong>{f.question}</strong>
                <div>{f.answer}</div>
                <button type="button" className="danger" onClick={run(() => api.deleteFaq(f.faq_id))}>
                  Remover
                </button>
              </li>
            ))}
          </ul>

          <h2>Feedbacks negativos ({feedback.length})</h2>
          <ul className="list">
            {feedback.map((fb) => (
              <li key={fb.feedback_id}>
                <strong>{fb.question}</strong>
                <div className="hint">{fb.answer}</div>
                {fb.comment && <div>Comentário: {fb.comment}</div>}
                <div className="hint">{new Date(fb.created_at).toLocaleString("pt-BR")}</div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
