import { describe, expect, it } from "vitest";
import { isCitations, parseSSE, SSEStreamParser } from "./sse";

const sample =
  'event: token\ndata: {"text": "Olá "}\n\n' +
  'event: token\ndata: {"text": "mundo [1]"}\n\n' +
  'event: citations\ndata: [{"index": 1, "doc_id": "d", "doc_title": "Doc", "section": "S", "chunk_id": "c1", "score": 0.5, "excerpt": "x"}]\n\n' +
  'event: done\ndata: {"message_id": "m1", "source": "rag", "session_id": "s"}\n\n';

describe("parseSSE", () => {
  it("parses token, citations and done events", () => {
    const events = parseSSE(sample);
    expect(events.map((e) => e.event)).toEqual(["token", "token", "citations", "done"]);
    expect(events[0].data).toEqual({ text: "Olá " });
    expect(isCitations(events[2].data)).toBe(true);
    expect((events[3].data as { message_id: string }).message_id).toBe("m1");
  });

  it("ignores comments and keeps plain-text data", () => {
    const events = parseSSE(": ping\n\ndata: plain\n\nevent: x\n\n");
    expect(events).toEqual([{ event: "message", data: "plain" }]);
  });
});

describe("SSEStreamParser", () => {
  it("reassembles events split across arbitrary chunk boundaries", () => {
    const parser = new SSEStreamParser();
    const events = [];
    for (let i = 0; i < sample.length; i += 7) {
      events.push(...parser.feed(sample.slice(i, i + 7)));
    }
    events.push(...parser.flush());
    expect(events.map((e) => e.event)).toEqual(["token", "token", "citations", "done"]);
    const text = events
      .filter((e) => e.event === "token")
      .map((e) => (e.data as { text: string }).text)
      .join("");
    expect(text).toBe("Olá mundo [1]");
  });
});
