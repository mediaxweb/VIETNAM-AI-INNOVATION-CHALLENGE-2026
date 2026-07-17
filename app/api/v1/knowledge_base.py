from functools import partial
import re
import unicodedata
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.schemas.auth import UserResponse
from app.api.schemas.knowledge_base import (
    KnowledgeBaseChunkDetailResponse,
    KnowledgeBaseDeleteFileRequest,
    KnowledgeBaseDeleteFileResponse,
    KnowledgeBaseDocumentTextRequest,
    KnowledgeBaseDocumentTextResponse,
    KnowledgeBaseIndexedFile,
    KnowledgeBaseIndexedFilesResponse,
    KnowledgeBaseProcessDocumentRequest,
    KnowledgeBaseProcessResponse,
)
from app.core.config import configs
from app.core.dependencies import get_current_user, get_openclaw_or_current_user_id
from app.services.knowledge_base_service import (
    KnowledgeBaseService,
)
from logs.logging_config import logger


router = APIRouter()

PDF_SIGNATURE = b"%PDF"
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


def get_knowledge_base_service() -> KnowledgeBaseService:
    """FastAPI dependency that instantiates the knowledge base service."""

    return KnowledgeBaseService()


def _uploaded_file_name(file: UploadFile) -> str:
    """Return a client-visible basename for an uploaded file."""

    raw_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not raw_name:
        raise ValueError("Uploaded file name is required.")
    if not raw_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return raw_name


def _safe_temp_file_name(file_name: str) -> str:
    normalized_name = unicodedata.normalize("NFKD", file_name)
    ascii_name = "".join(char for char in normalized_name if not unicodedata.combining(char))
    safe_name = re.sub(r"[^a-z0-9._-]+", "_", ascii_name.lower()).strip("._-")
    return safe_name or "document.pdf"


def _temp_upload_path(file_name: str) -> Path:
    temp_dir = Path(configs.temp_kb_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_temp_file_name(file_name)
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"

    return temp_dir / f"{uuid.uuid4().hex}_{safe_name}"


async def _save_uploaded_pdf(file: UploadFile, *, file_name: str) -> Path:
    temp_path = _temp_upload_path(file_name)
    try:
        first_chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
        if PDF_SIGNATURE not in first_chunk[:1024]:
            raise ValueError("Uploaded file must be a valid PDF.")

        with temp_path.open("wb") as output:
            output.write(first_chunk)
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                output.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return temp_path


@router.post("/process-document", response_model=KnowledgeBaseProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="Text-based PDF file to ingest."),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
) -> KnowledgeBaseProcessResponse:
    """Ingest one uploaded PDF into the current user's knowledge base."""

    document_path = ""
    temp_path: Path | None = None
    try:
        document_path = _uploaded_file_name(file)
        await service.record_user_file_indexing(
            user_id=current_user.id,
            document_path=document_path,
        )

        temp_path = await _save_uploaded_pdf(file, file_name=document_path)
        payload = KnowledgeBaseProcessDocumentRequest(document_path=str(temp_path))
        collection_name = KnowledgeBaseService._collection_name_for_user(current_user.id)
        result = await run_in_threadpool(
            partial(
                service.process_document_for_collection,
                payload,
                collection_name=collection_name,
                metadata_overrides={
                    "file_name": document_path,
                    "file_path": document_path,
                },
            ),
        )

        await service.record_user_indexed_file(
            user_id=current_user.id,
            document_path=document_path,
            result=result,
        )
        return KnowledgeBaseProcessResponse(
            document_count=result.document_count,
            node_count=result.node_count,
            collection_name=result.collection_name,
            message=result.message,
        )
    except ValueError as exc:
        if document_path:
            await service.record_user_indexed_file_failed(
                user_id=current_user.id,
                document_path=document_path,
                error=str(exc),
            )
        logger.warning("Knowledge base ingestion failed with a validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure path
        if document_path:
            await service.record_user_indexed_file_failed(
                user_id=current_user.id,
                document_path=document_path,
                error=str(exc),
            )
        logger.exception("Unexpected error while processing knowledge base ingestion: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process knowledge base") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        await file.close()


@router.post("/document-text", response_model=KnowledgeBaseDocumentTextResponse)
async def get_document_text(
    payload: KnowledgeBaseDocumentTextRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    user_id: str = Depends(get_openclaw_or_current_user_id),
) -> KnowledgeBaseDocumentTextResponse:
    """Return indexed text for one PDF page from the user's ChromaDB collection."""

    try:
        result = await run_in_threadpool(
            partial(
                service.get_document_text,
                payload.document_path,
                page_label=payload.page_label,
                user_id=user_id,
            ),
        )
        return KnowledgeBaseDocumentTextResponse(
            text=result.text,
            chunk_count=result.chunk_count,
        )
    except ValueError as exc:
        logger.warning("Knowledge base document text lookup failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Unexpected error while reading knowledge base document text: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read document text") from exc


@router.get("/chunks/{chunk_id}", response_model=KnowledgeBaseChunkDetailResponse)
async def get_chunk_detail(
    chunk_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    user_id: str = Depends(get_openclaw_or_current_user_id),
) -> KnowledgeBaseChunkDetailResponse:
    """Return indexed text and metadata for one ChromaDB chunk in the user's collection."""

    try:
        result = await run_in_threadpool(
            partial(
                service.get_chunk_detail,
                chunk_id,
                user_id=user_id,
            ),
        )
        metadata = result.metadata
        return KnowledgeBaseChunkDetailResponse(
            chunk_id=result.chunk_id,
            text=result.text,
            page_label=metadata.get("page_label"),
            file_name=metadata.get("file_name"),
            window=metadata.get("window"),
            metadata=metadata,
        )
    except ValueError as exc:
        logger.warning("Knowledge base chunk detail lookup failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Unexpected error while reading knowledge base chunk detail: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read chunk detail") from exc


async def _list_knowledge_base_files(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
) -> KnowledgeBaseIndexedFilesResponse:
    """Return files tracked in the current user's knowledge base registry."""

    try:
        indexed_files = await service.list_indexed_files(user_id=current_user.id)
        files = [
            KnowledgeBaseIndexedFile(
                file_name=item.file_name,
                file_path=item.file_path,
                document_id=item.document_id,
                index_status=item.index_status,
                chunk_count=item.chunk_count,
                page_count=item.page_count,
                last_error=item.last_error,
            )
            for item in indexed_files
        ]
        return KnowledgeBaseIndexedFilesResponse(
            total_count=len(files),
            files=files,
        )
    except ValueError as exc:
        logger.warning("Knowledge base indexed files lookup failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Unexpected error while listing indexed files: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list indexed files") from exc


@router.get("/files", response_model=KnowledgeBaseIndexedFilesResponse)
async def list_knowledge_base_files(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
) -> KnowledgeBaseIndexedFilesResponse:
    """Return files tracked in the current user's knowledge base registry."""

    return await _list_knowledge_base_files(service=service, current_user=current_user)


@router.delete("/files", response_model=KnowledgeBaseDeleteFileResponse)
async def delete_knowledge_base_file(
    payload: KnowledgeBaseDeleteFileRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
) -> KnowledgeBaseDeleteFileResponse:
    """Delete one file from the current user's knowledge base index."""

    try:
        result = await service.delete_user_indexed_file(
            user_id=current_user.id,
            document_path=payload.document_path,
        )
        return KnowledgeBaseDeleteFileResponse(
            status=result.status,
            message=result.message,
            document_path=result.document_path,
            deleted_node_count=result.deleted_node_count,
            collection_name=result.collection_name,
        )
    except ValueError as exc:
        logger.warning("Knowledge base file delete failed with a validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Unexpected error while deleting indexed file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete indexed file") from exc


@router.get(
    "/indexed-files",
    response_model=KnowledgeBaseIndexedFilesResponse,
    include_in_schema=False,
)
async def list_indexed_files(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
) -> KnowledgeBaseIndexedFilesResponse:
    """Backward-compatible alias for the original indexed files endpoint."""

    return await _list_knowledge_base_files(service=service, current_user=current_user)
