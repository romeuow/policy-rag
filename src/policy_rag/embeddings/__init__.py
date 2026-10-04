"""Embedding providers behind the `EmbeddingProvider` protocol."""

from policy_rag.embeddings.base import EmbeddingProvider
from policy_rag.embeddings.fake import FakeEmbeddingProvider
from policy_rag.embeddings.voyage import VoyageEmbeddingProvider

__all__ = ["EmbeddingProvider", "FakeEmbeddingProvider", "VoyageEmbeddingProvider"]
