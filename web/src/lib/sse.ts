/** Minimal Server-Sent Events parser for `fetch` streams. */

export interface SSEEvent {
  event: string;
  data: unknown;
}

export interface Citation {
  index: number;
  doc_id: string;
  doc_title: string;
  section: string;
  chunk_id: string;
  score: number;
  excerpt: string;
}

export interface DoneEvent {
  message_id: string;
  source: "rag" | "faq" | "none";
  session_id: string;
}

function parseData(lines: string[]): unknown {
  const joined = lines.join("\n");
  try {
    return JSON.parse(joined);
  } catch {
    return joined;
  }
}

/** Parse one complete SSE block (no trailing blank line). Returns null for comments/empty. */
export function parseBlock(block: string): SSEEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    const idx = line.indexOf(":");
    const field = idx === -1 ? line : line.slice(0, idx);
    let value = idx === -1 ? "" : line.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: parseData(data) };
}

/** Parse a full SSE payload (useful in tests and for non-streaming fallbacks). */
export function parseSSE(text: string): SSEEvent[] {
  const events: SSEEvent[] = [];
  for (const block of text.replace(/\r\n/g, "\n").split("\n\n")) {
    const parsed = parseBlock(block);
    if (parsed) events.push(parsed);
  }
  return events;
}

/**
 * Incremental parser: feed chunks as they arrive, receive complete events.
 * Keeps the unterminated tail in its buffer between calls.
 */
export class SSEStreamParser {
  private buffer = "";

  feed(chunk: string): SSEEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const events: SSEEvent[] = [];
    let idx: number;
    while ((idx = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
    }
    return events;
  }

  flush(): SSEEvent[] {
    const rest = this.buffer;
    this.buffer = "";
    const parsed = rest.trim() ? parseBlock(rest) : null;
    return parsed ? [parsed] : [];
  }
}

export function isCitations(data: unknown): data is Citation[] {
  return Array.isArray(data) && data.every((c) => typeof c === "object" && c !== null && "chunk_id" in c);
}
