from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

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
