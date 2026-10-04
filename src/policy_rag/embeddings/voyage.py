"""Voyage AI embedding provider (production only, requires VOYAGE_API_KEY)."""

from __future__ import annotations

import math

import httpx


class VoyageEmbeddingProvider:
    """Calls the Voyage AI `/embeddings` endpoint over HTTPS using httpx."""

    dim: int

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        base_url: str = "https://api.voyageai.com/v1",
        dim: int = 1024,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("Voyage API key is required")
        self.model = model
        self.dim = dim
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    def _normalise(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.post(
            "/embeddings",
            headers=self._headers,
            json={"input": texts, "model": self.model, "input_type": input_type},
        )
        response.raise_for_status()
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        return [self._normalise(item["embedding"]) for item in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "query")[0]
