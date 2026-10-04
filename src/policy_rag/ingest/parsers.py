"""Parsers for Markdown, plain text and (optionally) PDF files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    version: str
    body: str
    source_name: str
    content_hash: str
    metadata: dict[str, str] = field(default_factory=dict)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[àáâãä]", "a", value)
    value = re.sub(r"[èéêë]", "e", value)
    value = re.sub(r"[ìíîï]", "i", value)
    value = re.sub(r"[òóôõö]", "o", value)
    value = re.sub(r"[ùúûü]", "u", value)
    value = value.replace("ç", "c")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "document"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, text[match.end() :]


def parse_text(text: str, source_name: str, doc_id: str | None = None) -> ParsedDocument:
    """Parse Markdown/plain text into a `ParsedDocument`, honouring YAML-like front matter."""
    meta, body = _parse_front_matter(text)
    h1 = _H1.search(body)
    title = meta.get("title") or (h1.group(1) if h1 else Path(source_name).stem)
    version = meta.get("version", "1.0")
    resolved_id = doc_id or slugify(Path(source_name).stem)
    return ParsedDocument(
        doc_id=resolved_id,
        title=title,
        version=version,
        body=body.strip(),
        source_name=source_name,
        content_hash=content_hash(text),
        metadata=meta,
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PDF support requires the optional 'pdf' extra (pypdf).") from exc
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def parse_file(path: Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse a supported file from disk."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    return parse_text(text, source_name=path.name, doc_id=doc_id)
