from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import ApplicationContainer, get_container
from app.api.schemas import DocumentResponse, IngestionResponse
from app.core.security import Principal, get_principal
from app.infrastructure.document_processing import UnsupportedDocumentError

router = APIRouter(prefix="/documents", tags=["documents"])
MAX_FILE_SIZE = 25 * 1024 * 1024


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
    allowed_groups: str = Form(""),
) -> IngestionResponse:
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds 25 MB.")
    try:
        document, created = await run_in_threadpool(
            container.ingestion.ingest,
            file.filename or "document",
            content,
            file.content_type,
            principal,
            [group.strip() for group in allowed_groups.split(",") if group.strip()],
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return IngestionResponse(
        document=DocumentResponse.model_validate(document),
        created=created,
        message="Document indexed." if created else "Identical content was already indexed.",
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> list[DocumentResponse]:
    return [
        DocumentResponse.model_validate(item)
        for item in container.repository.list_documents(principal)
    ]


@router.post("/jobs", status_code=202)
async def queue_document(
    file: UploadFile = File(...),
    allowed_groups: str = Form(""),
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    filename = Path((file.filename or "document").replace("\\", "/")).name[:200]
    if Path(filename).suffix.lower() not in container.ingestion.parser.supported_extensions:
        raise HTTPException(415, "Use PDF, TXT, or Markdown.")
    content = await file.read(container.settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(422, "Cannot ingest an empty document.")
    if len(content) > container.settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds 25 MB.")
    groups = [group.strip() for group in allowed_groups.split(",") if group.strip()]
    try:
        return await run_in_threadpool(container.jobs.submit, filename, content, principal, groups)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/jobs/{job_id}")
def job_status(
    job_id: str,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    job = container.jobs.get(job_id, principal)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/chunks/{chunk_id}")
def chunk_source(
    chunk_id: str,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    chunks = container.repository.get_chunks([chunk_id], principal)
    if not chunks:
        raise HTTPException(404, "Evidence not found.")
    chunk = chunks[0]
    return {
        "id": chunk.id,
        "text": chunk.text,
        "filename": chunk.filename,
        "page": chunk.page,
        "section": chunk.section,
        "version": chunk.version,
    }


@router.get("/{document_id}/download")
def download(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    document = container.repository.get_document(document_id)
    if not document or not principal.can_read(document):
        raise HTTPException(404, "Document not found.")
    return FileResponse(
        document.content_path, filename=document.filename, media_type="application/octet-stream"
    )


@router.get("/{document_id}/versions", response_model=list[DocumentResponse])
def versions(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
):
    document = container.repository.get_document(document_id)
    if not document or not principal.can_read(document):
        raise HTTPException(404, "Document not found.")
    return [
        DocumentResponse.model_validate(item)
        for item in container.repository.list_documents()
        if item.owner_id == document.owner_id
        and item.filename.casefold() == document.filename.casefold()
        and principal.can_read(item)
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> None:
    document = container.repository.get_document(document_id)
    if document is None or not principal.can_read(document):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    if not principal.admin and document.owner_id != principal.user_id:
        raise HTTPException(403, "Only the owner or administrator can delete this document.")
    await run_in_threadpool(container.ingestion.delete, document_id)
