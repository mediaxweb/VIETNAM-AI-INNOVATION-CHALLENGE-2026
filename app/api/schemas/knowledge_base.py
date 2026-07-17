from typing import Any, Literal

from pydantic import BaseModel, Field


class KnowledgeBaseProcessDocumentRequest(BaseModel):
    """Internal request payload for processing a server-local PDF path."""

    document_path: str = Field(
        ...,
        description="Server-local PDF path to ingest into the knowledge base.",
    )


class KnowledgeBaseProcessResponse(BaseModel):
    """Response model describing the ingestion result."""

    document_count: int = Field(..., ge=0)
    node_count: int = Field(..., ge=0)
    collection_name: str
    message: str


class KnowledgeBaseDocumentTextRequest(BaseModel):
    """Request payload for reading one indexed PDF page from ChromaDB."""

    document_path: str = Field(
        ...,
        min_length=1,
        description="Uploaded PDF file name or document id used to find indexed ChromaDB chunks.",
    )
    page_label: str = Field(
        ...,
        min_length=1,
        description="PDF page label to return from indexed ChromaDB chunks.",
    )


class KnowledgeBaseDocumentTextResponse(BaseModel):
    """Indexed text for a single PDF page."""

    text: str
    chunk_count: int = Field(..., ge=0)


class KnowledgeBaseChunkDetailResponse(BaseModel):
    """Indexed ChromaDB chunk detail."""

    chunk_id: str
    text: str
    page_label: str | None = None
    file_name: str | None = None
    window: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseIndexedFile(BaseModel):
    """Single indexed file discovered in a user's knowledge base collection."""

    file_name: str | None = None
    file_path: str | None = None
    document_id: str | None = None
    index_status: str
    chunk_count: int = Field(..., ge=0)
    page_count: int = Field(..., ge=0)
    last_error: str | None = None


class KnowledgeBaseIndexedFilesResponse(BaseModel):
    """Indexed files available for the current user."""

    total_count: int = Field(..., ge=0)
    files: list[KnowledgeBaseIndexedFile] = Field(default_factory=list)


class KnowledgeBaseDeleteFileRequest(BaseModel):
    """Request payload for deleting one file from the user's knowledge base index."""

    document_path: str = Field(
        ...,
        min_length=1,
        description="Uploaded PDF file name or document id to remove from the user's index.",
    )


class KnowledgeBaseDeleteFileResponse(BaseModel):
    """Delete result for one user-scoped knowledge base file."""

    status: str = Field(..., min_length=1)
    message: str
    document_path: str = Field(..., min_length=1)
    deleted_node_count: int = Field(..., ge=0)
    collection_name: str


class ConversationHistoryMessage(BaseModel):
    """Single prior turn that can help interpret the current question."""

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Message role from the earlier conversation.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Message text for the earlier conversation turn.",
    )


class QuestionRequest(BaseModel):
    """Request payload for retrieval-augmented question answering."""

    question: str = Field(
        ...,
        min_length=1,
        description="Current user question to answer from retrieved document context.",
    )
    conversation_history: list[ConversationHistoryMessage] = Field(
        default_factory=list,
        description="Optional prior conversation turns to help interpret the current question.",
    )


class RetrievedChunk(BaseModel):
    """Single retrieved chunk returned without answer generation."""

    chunk_id: str
    text: str
    score: float | None = None
    page_label: str | None = None
    file_name: str | None = None
    window: str | None = None


class RetrieveChunksResponse(BaseModel):
    """Retrieved chunks for a question."""

    question: str
    chunks: list[RetrievedChunk]
