"""Shared fixtures: demo-mode container wired with fakes and a FastAPI test client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from policy_rag.api.app import create_app
from policy_rag.api.container import Container, build_container
from policy_rag.config import Settings
from policy_rag.embeddings.fake import FakeEmbeddingProvider
from policy_rag.ingest.parsers import parse_text
from policy_rag.store.memory import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ADMIN_TOKEN = "test-admin-token-123"

SAMPLE_MARKDOWN = """---
title: Política de Teste
version: "9.9"
---

# Política de Teste

## 1. Objetivo

Primeiro parágrafo da seção de objetivo com texto suficiente para formar um chunk.

## 2. Regras

### 2.1 Regra A

A regra A diz que o prazo de solicitação é de 10 dias úteis.

### 2.2 Regra B

A regra B diz que o limite de reembolso é de R$ 100,00 por dia.
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_mode="demo",
        admin_token=ADMIN_TOKEN,
        corpus_dir=str(CORPUS_DIR),
        web_dist_dir=str(tmp_path / "no-dist"),
        log_level="WARNING",
    )


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dim=256)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def container(settings: Settings) -> Container:
    return build_container(settings)


@pytest.fixture
def ingested(container: Container) -> Container:
    container.ingest_corpus()
    return container


@pytest.fixture
def client(settings: Settings, container: Container) -> Iterator[TestClient]:
    app = create_app(settings=settings, container=container, ingest_on_startup=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_doc():
    return parse_text(SAMPLE_MARKDOWN, source_name="politica-de-teste.md")


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-Token": ADMIN_TOKEN}
