from pathlib import Path

import pytest

from policy_rag.ingest import parsers
from policy_rag.ingest.parsers import parse_file, parse_text, slugify


def test_parse_text_reads_front_matter_and_strips_it():
    doc = parse_text('---\ntitle: Minha Política\nversion: "2.0"\n---\n# X\n\nCorpo', "a.md")
    assert doc.title == "Minha Política"
    assert doc.version == "2.0"
    assert doc.body.startswith("# X")
    assert doc.doc_id == "a"
    assert len(doc.content_hash) == 64


def test_parse_text_falls_back_to_h1_then_filename():
    assert parse_text("# Título H1\n\ntexto", "arquivo.md").title == "Título H1"
    assert parse_text("só texto", "nome-do-arquivo.txt").title == "nome-do-arquivo"


def test_slugify_normalises_accents_and_symbols():
    assert slugify("Política de Férias (v2)!") == "politica-de-ferias-v2"
    assert slugify("###") == "document"


def test_parse_file_rejects_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "x.docx"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(path)


def test_parse_file_reads_txt_and_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    txt = tmp_path / "regras.txt"
    txt.write_text("Regras gerais da empresa", encoding="utf-8")
    assert parse_file(txt).body == "Regras gerais da empresa"

    monkeypatch.setattr(parsers, "_read_pdf", lambda p: "# PDF\n\nconteúdo extraído")
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    parsed = parse_file(pdf)
    assert parsed.title == "PDF"
    assert parsed.doc_id == "manual"
