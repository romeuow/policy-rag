"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden by an env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_mode: Literal["demo", "prod"] = "demo"
    log_level: str = "INFO"
    admin_token: str = Field(default="change-me-admin-token", min_length=8)
    corpus_dir: str | None = "corpus"
    web_dist_dir: str = "web/dist"
    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 2048

    # Embeddings
    embedding_dim: int = 256
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3"
    voyage_base_url: str = "https://api.voyageai.com/v1"

    # Vector store
    database_url: str | None = None

    # Retrieval
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 120
    top_k_vector: int = 8
    top_k_lexical: int = 8
    top_k_final: int = 4
    rrf_k: int = 60
    min_relevance_score: float = 0.30
    faq_match_threshold: float = 0.80

    # Guardrails
    max_question_chars: int = 1000
    session_history_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
