import { useRef, useState } from "react";
import { sendFeedback, streamChat } from "../lib/api";
import type { Citation } from "../lib/sse";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations: Citation[];
  source?: "rag" | "faq" | "none";
  messageId?: string;
  feedback?: "up" | "down";
  streaming?: boolean;
}

function newSessionId(): string {
  return `web-${Math.random().toString(36).slice(2, 10)}`;
}

function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      <div className="citations-title">Fontes</div>
      {citations.map((c) => (
        <div key={c.chunk_id} className="citation">
          <button type="button" onClick={() => setOpen(open === c.index ? null : c.index)}>
            [{c.index}] {c.doc_title} — {c.section}
          </button>
          {open === c.index && <blockquote>{c.excerpt}</blockquote>}
        </div>
      ))}
    </div>
  );
}

export default function ChatPage() {
  const sessionId = useRef(newSessionId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);

  const update = (id: string, patch: Partial<Message> | ((m: Message) => Partial<Message>)) =>
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)),
    );

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    const userId = crypto.randomUUID();
    const answerId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", text: q, citations: [] },
      { id: answerId, role: "assistant", text: "", citations: [], streaming: true },
    ]);
    try {
      await streamChat(sessionId.current, q, {
        onToken: (text) => update(answerId, (m) => ({ text: m.text + text })),
        onCitations: (citations) => update(answerId, { citations }),
        onDone: (done) => update(answerId, { messageId: done.message_id, source: done.source, streaming: false }),
        onError: (message) => update(answerId, { text: `Erro: ${message}`, streaming: false }),
      });
    } finally {
      setBusy(false);
    }
  }

  async function rate(message: Message, rating: "up" | "down") {
    if (!message.messageId || message.feedback) return;
    const comment = rating === "down" ? window.prompt("O que faltou na resposta? (opcional)") ?? undefined : undefined;
    await sendFeedback(message.messageId, rating, comment);
    update(message.id, { feedback: rating });
  }

  return (
    <section>
      <div className="messages">
        {messages.length === 0 && (
          <p className="hint">
            Pergunte sobre férias, reembolso, segurança da informação, home office, conduta, benefícios ou viagens.
          </p>
        )}
        {messages.map((m) => (
          <article key={m.id} className={`bubble ${m.role}`}>
            <div className="bubble-text">{m.text || (m.streaming ? "…" : "")}</div>
            {m.role === "assistant" && !m.streaming && (
              <>
                <CitationList citations={m.citations} />
                <div className="actions">
                  {m.source && <span className="badge">{m.source}</span>}
                  <button type="button" disabled={!!m.feedback} onClick={() => rate(m, "up")} aria-label="útil">
                    👍
                  </button>
                  <button type="button" disabled={!!m.feedback} onClick={() => rate(m, "down")} aria-label="não útil">
                    👎
                  </button>
                  {m.feedback && <span className="hint">obrigado pelo feedback</span>}
                </div>
              </>
            )}
          </article>
        ))}
      </div>
      <form className="composer" onSubmit={ask}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ex.: Quantos dias de férias eu tenho por ano?"
          maxLength={1000}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Respondendo…" : "Perguntar"}
        </button>
      </form>
    </section>
  );
}
