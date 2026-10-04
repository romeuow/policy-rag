"""Health and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from policy_rag import __version__
from policy_rag.api.deps import ContainerDep

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(container: ContainerDep) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "mode": container.settings.app_mode,
        "documents": len(container.documents.list()),
        "chunks": container.store.count(),
    }


@router.get("/metrics")
def metrics(container: ContainerDep) -> dict:
    return container.metrics.snapshot()
