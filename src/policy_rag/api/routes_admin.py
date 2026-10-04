"""Admin endpoints protected by X-Admin-Token."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from policy_rag.api.deps import ContainerDep, require_admin
from policy_rag.api.schemas import DocumentCreate, DocumentPatch, FaqCreate
from policy_rag.models import Faq

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/documents")
def list_documents(container: ContainerDep) -> list[dict]:
    return [d.model_dump(mode="json") for d in container.documents.list()]


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(body: DocumentCreate, container: ContainerDep) -> dict:
    doc = container.documents.ingest_text(body.content, body.filename, doc_id=body.doc_id)
    return doc.model_dump(mode="json")


@router.patch("/documents/{doc_id}")
def patch_document(doc_id: str, body: DocumentPatch, container: ContainerDep) -> dict:
    doc = container.documents.set_active(doc_id, body.active)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return doc.model_dump(mode="json")


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, container: ContainerDep) -> None:
    if not container.documents.delete(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


@router.post("/documents/{doc_id}/reindex")
def reindex_document(doc_id: str, container: ContainerDep) -> dict:
    doc = container.documents.reindex(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found or source unavailable")
    return doc.model_dump(mode="json")


@router.get("/faqs")
def list_faqs(container: ContainerDep) -> list[dict]:
    return [f.model_dump(mode="json") for f in container.faqs.list()]


@router.post("/faqs", status_code=status.HTTP_201_CREATED)
def create_faq(body: FaqCreate, container: ContainerDep) -> dict:
    faq = container.faqs.add(Faq(question=body.question, answer=body.answer))
    return faq.model_dump(mode="json")


@router.delete("/faqs/{faq_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_faq(faq_id: str, container: ContainerDep) -> None:
    if not container.faqs.delete(faq_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "faq not found")


@router.get("/feedback")
def list_feedback(container: ContainerDep, rating: str | None = None) -> list[dict]:
    if rating is not None and rating not in {"up", "down"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "rating must be up or down")
    return [f.model_dump(mode="json") for f in container.feedback.list(rating)]
