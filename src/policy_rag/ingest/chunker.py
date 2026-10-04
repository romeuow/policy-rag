"""Heading-aware chunking with character overlap and content-hash chunk IDs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from policy_rag.ingest.parsers import ParsedDocument
from policy_rag.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Section:
    heading: str
    text: str


def split_sections(body: str) -> list[Section]:
    """Split a Markdown body into sections keyed by their heading path."""
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [Section(heading="", text=body.strip())] if body.strip() else []

    sections: list[Section] = []
    path: dict[int, str] = {}
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading="", text=preamble))

    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        path = {lvl: name for lvl, name in path.items() if lvl < level}
        path[level] = title
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip()
        if not text:
            continue
        heading = " > ".join(path[lvl] for lvl in sorted(path) if lvl > 1) or title
        sections.append(Section(heading=heading, text=text))
    return sections


def split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `max_chars`, preferring paragraph boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = para
        while len(current) > max_chars:
            cut = current.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(current[:cut].strip())
            start = max(cut - overlap, 0)
            current = current[start:].strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_hash(doc_id: str, section: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}\x1f{section}\x1f{text}".encode()).hexdigest()
    return digest[:24]


def chunk_document(doc: ParsedDocument, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Turn a parsed document into chunks. Deterministic: same input -> same chunk IDs."""
    chunks: list[Chunk] = []
    position = 0
    for section in split_sections(doc.body):
        for piece in split_with_overlap(section.text, max_chars, overlap):
            text = f"{section.heading}\n{piece}" if section.heading else piece
            chunks.append(
                Chunk(
                    chunk_id=chunk_hash(doc.doc_id, section.heading, piece),
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section=section.heading,
                    version=doc.version,
                    position=position,
                    text=text,
                    content_hash=doc.content_hash,
                )
            )
            position += 1
    return chunks
