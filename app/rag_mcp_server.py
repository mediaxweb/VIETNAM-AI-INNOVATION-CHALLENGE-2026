from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from app.services.knowledge_base_service import KnowledgeBaseService


class KnowledgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    page: str | None = None
    excerpt: str = Field(min_length=1)


class KnowledgeEvidenceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[KnowledgeEvidence]


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Retrieved chunk requires {field_name}")
    return value.strip()


def _chunk_to_evidence(chunk: Any) -> KnowledgeEvidence:
    if not isinstance(chunk, Mapping):
        raise ValueError("Retrieved chunk must be an object")
    raw_page = chunk.get("page_label")
    page = None if raw_page is None else str(raw_page).strip() or None
    excerpt = chunk.get("window") or chunk.get("text")
    return KnowledgeEvidence(
        source_id=_required_text(chunk.get("chunk_id"), "chunk_id"),
        file_name=_required_text(chunk.get("file_name"), "file_name"),
        page=page,
        excerpt=_required_text(excerpt, "window or text"),
    )


async def retrieve_credit_evidence(
    domain: str,
    query: str,
    top_k: int,
    *,
    user_id: str,
    service: KnowledgeBaseService,
) -> KnowledgeEvidenceEnvelope:
    if domain != "credit":
        raise ValueError("search_knowledge domain must be credit")
    normalized_query = _required_text(query, "query")
    if type(top_k) is not int or top_k != 5:
        raise ValueError("search_knowledge top_k must be 5")
    normalized_user_id = _required_text(user_id, "RAG_MCP_USER_ID")

    try:
        result = await asyncio.to_thread(
            service.retrieve_chunks,
            normalized_query,
            user_id=normalized_user_id,
        )
    except Exception:
        raise RuntimeError("Credit knowledge retrieval failed") from None

    chunks = result.get("chunks") if isinstance(result, Mapping) else None
    if not isinstance(chunks, list):
        raise ValueError("RAG retrieval returned an invalid chunk list")
    evidence = [_chunk_to_evidence(chunk) for chunk in chunks[:top_k]]
    if not evidence:
        raise ValueError("RAG retrieval returned no evidence")
    source_ids = [item.source_id for item in evidence]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate evidence source_id")
    return KnowledgeEvidenceEnvelope(evidence=evidence)


async def retrieve_document_page(
    domain: str,
    source_id: str,
    *,
    user_id: str,
    service: KnowledgeBaseService,
) -> KnowledgeEvidenceEnvelope:
    if domain != "credit":
        raise ValueError("get_document_page domain must be credit")
    normalized_source_id = _required_text(source_id, "source_id")
    normalized_user_id = _required_text(user_id, "RAG_MCP_USER_ID")

    try:
        chunk = await asyncio.to_thread(
            service.get_chunk_detail,
            normalized_source_id,
            user_id=normalized_user_id,
        )
        metadata = chunk.metadata
        if not isinstance(metadata, Mapping):
            raise ValueError("Chunk metadata must be an object")
        normalized_file_name = _required_text(metadata.get("file_name"), "file_name")
        normalized_page = _required_text(metadata.get("page_label"), "page_label")
        result = await asyncio.to_thread(
            service.get_document_text,
            normalized_file_name,
            page_label=normalized_page,
            user_id=normalized_user_id,
        )
    except Exception:
        raise RuntimeError("Document page retrieval failed") from None

    result_file_name = _required_text(result.document_path, "document_path")
    result_page = _required_text(result.page_label, "page_label")
    if (result_file_name, result_page) != (normalized_file_name, normalized_page):
        raise ValueError("RAG returned a different document page")
    return KnowledgeEvidenceEnvelope(
        evidence=[
            KnowledgeEvidence(
                source_id=f"page:{normalized_source_id}",
                file_name=result_file_name,
                page=result_page,
                excerpt=_required_text(result.text, "page text"),
            )
        ]
    )


knowledge_base_service = KnowledgeBaseService()
mcp = FastMCP(
    "credit-rag",
    host=os.getenv("RAG_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("RAG_MCP_PORT", "8766")),
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


@mcp.tool(structured_output=True)
async def search_knowledge(
    domain: Literal["credit"],
    query: str,
    top_k: int = 5,
) -> KnowledgeEvidenceEnvelope:
    """Retrieve grounded evidence from the credit-policy knowledge base."""
    return await retrieve_credit_evidence(
        domain,
        query,
        top_k,
        user_id=os.getenv("RAG_MCP_USER_ID", ""),
        service=knowledge_base_service,
    )


@mcp.tool(structured_output=True)
async def get_document_page(
    domain: Literal["credit"],
    source_id: str,
) -> KnowledgeEvidenceEnvelope:
    """Read the full indexed page for evidence returned by search_knowledge."""
    return await retrieve_document_page(
        domain,
        source_id,
        user_id=os.getenv("RAG_MCP_USER_ID", ""),
        service=knowledge_base_service,
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
