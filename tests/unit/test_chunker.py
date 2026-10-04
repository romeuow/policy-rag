import pytest

from policy_rag.ingest.chunker import chunk_document, split_sections, split_with_overlap
from policy_rag.ingest.parsers import parse_text


def test_split_sections_builds_heading_paths(sample_doc):
    sections = split_sections(sample_doc.body)
    headings = [s.heading for s in sections]
    assert headings == ["1. Objetivo", "2. Regras > 2.1 Regra A", "2. Regras > 2.2 Regra B"]


def test_split_sections_without_headings_returns_single_section():
    assert [s.heading for s in split_sections("apenas texto")] == [""]
    assert split_sections("   ") == []


def test_split_with_overlap_respects_limits_and_overlap():
    text = " ".join(f"palavra{i}" for i in range(300))
    pieces = split_with_overlap(text, max_chars=200, overlap=40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    # overlap: the tail of piece N shows up at the beginning of piece N+1
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert nxt.startswith(prev[-40:].split(" ", 1)[-1][:10]) or prev[-15:] in nxt


def test_split_with_overlap_prefers_paragraph_boundaries():
    paragraphs = [f"Parágrafo {i} " + "x" * 80 for i in range(6)]
    pieces = split_with_overlap("\n\n".join(paragraphs), max_chars=200, overlap=0)
    assert all(len(p) <= 200 for p in pieces)
    assert pieces[0].startswith("Parágrafo 0")


def test_split_with_overlap_validates_arguments():
    with pytest.raises(ValueError):
        split_with_overlap("x", max_chars=0, overlap=0)
    with pytest.raises(ValueError):
        split_with_overlap("x", max_chars=10, overlap=10)
    assert split_with_overlap("", 10, 0) == []


def test_chunk_document_is_deterministic_and_carries_metadata(sample_doc):
    first = chunk_document(sample_doc, max_chars=300, overlap=50)
    second = chunk_document(sample_doc, max_chars=300, overlap=50)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert {c.doc_id for c in first} == {"politica-de-teste"}
    assert first[0].doc_title == "Política de Teste"
    assert first[0].version == "9.9"
    assert first[1].section == "2. Regras > 2.1 Regra A"
    assert first[1].text.startswith("2. Regras > 2.1 Regra A\n")
    assert [c.position for c in first] == list(range(len(first)))


def test_chunk_ids_change_only_for_modified_sections(sample_doc):
    original = {c.section: c.chunk_id for c in chunk_document(sample_doc)}
    modified_text = sample_doc.body.replace("10 dias úteis", "15 dias úteis")
    modified = parse_text(modified_text, source_name="politica-de-teste.md")
    changed = {c.section: c.chunk_id for c in chunk_document(modified)}
    assert original["1. Objetivo"] == changed["1. Objetivo"]
    assert original["2. Regras > 2.1 Regra A"] != changed["2. Regras > 2.1 Regra A"]
