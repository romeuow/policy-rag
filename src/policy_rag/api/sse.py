"""Server-Sent Events encoding and parsing (the parser is used by tests and tooling)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SSEEvent:
    event: str
    data: Any


def encode_event(event: str, data: Any) -> str:
    """Encode one SSE frame. Multi-line payloads are JSON so they never contain raw newlines."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def parse_sse(text: str) -> list[SSEEvent]:
    """Parse a full SSE stream into events. Tolerates comments and multi-line `data:` fields."""
    events: list[SSEEvent] = []
    for raw_block in text.replace("\r\n", "\n").split("\n\n"):
        block = raw_block.strip("\n")
        if not block:
            continue
        name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data_lines.append(value)
        if not data_lines:
            continue
        joined = "\n".join(data_lines)
        try:
            data: Any = json.loads(joined)
        except json.JSONDecodeError:
            data = joined
        events.append(SSEEvent(event=name, data=data))
    return events
