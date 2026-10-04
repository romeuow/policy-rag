import { type Citation, type DoneEvent, isCitations, SSEStreamParser } from "./sse";

export interface ChatHandlers {
  onToken: (text: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDone: (done: DoneEvent) => void;
  onError: (message: string) => void;
}

export async function streamChat(sessionId: string, question: string, handlers: ChatHandlers) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    handlers.onError(`Erro ${response.status}: ${detail}`);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEStreamParser();
  const dispatch = (events: ReturnType<SSEStreamParser["feed"]>) => {
    for (const ev of events) {
      if (ev.event === "token") handlers.onToken((ev.data as { text: string }).text);
      else if (ev.event === "citations" && isCitations(ev.data)) handlers.onCitations(ev.data);
      else if (ev.event === "done") handlers.onDone(ev.data as DoneEvent);
      else if (ev.event === "error") handlers.onError((ev.data as { message: string }).message);
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    dispatch(parser.feed(decoder.decode(value, { stream: true })));
  }
  dispatch(parser.flush());
}

export async function sendFeedback(messageId: string, rating: "up" | "down", comment?: string) {
  const response = await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, rating, comment }),
  });
  if (!response.ok) throw new Error(`feedback failed: ${response.status}`);
}

export interface DocumentInfo {
  doc_id: string;
  title: string;
  version: string;
  source_name: string;
  active: boolean;
  chunk_count: number;
  updated_at: string;
}

export interface FaqInfo {
  faq_id: string;
  question: string;
  answer: string;
}

export interface FeedbackInfo {
  feedback_id: string;
  message_id: string;
  rating: "up" | "down";
  comment: string | null;
  question: string | null;
  answer: string | null;
  created_at: string;
}

export class AdminApi {
  constructor(private token: string) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Token": this.token,
        ...(init.headers ?? {}),
      },
    });
    if (response.status === 401) throw new Error("Token de administrador inválido.");
    if (!response.ok) throw new Error(`Erro ${response.status}: ${await response.text()}`);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  listDocuments = () => this.request<DocumentInfo[]>("/admin/documents");
  uploadDocument = (filename: string, content: string) =>
    this.request<DocumentInfo>("/admin/documents", {
      method: "POST",
      body: JSON.stringify({ filename, content }),
    });
  deleteDocument = (docId: string) =>
    this.request<void>(`/admin/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  reindexDocument = (docId: string) =>
    this.request<DocumentInfo>(`/admin/documents/${encodeURIComponent(docId)}/reindex`, {
      method: "POST",
    });
  setActive = (docId: string, active: boolean) =>
    this.request<DocumentInfo>(`/admin/documents/${encodeURIComponent(docId)}`, {
      method: "PATCH",
      body: JSON.stringify({ active }),
    });
  listFaqs = () => this.request<FaqInfo[]>("/admin/faqs");
  createFaq = (question: string, answer: string) =>
    this.request<FaqInfo>("/admin/faqs", { method: "POST", body: JSON.stringify({ question, answer }) });
  deleteFaq = (faqId: string) =>
    this.request<void>(`/admin/faqs/${encodeURIComponent(faqId)}`, { method: "DELETE" });
  listNegativeFeedback = () => this.request<FeedbackInfo[]>("/admin/feedback?rating=down");
}
